"""PKG-F1-01: coleta verificavel -> leitura pelos dicionarios -> E2 de 2008 e elegibilidade de 2017 (sem alvo).

  python -m occasion.pipeline_pof coletar
  python -m occasion.pipeline_pof executar [--saida DIR] [--selado DIR] [--sem-referencias]
  python -m occasion.pipeline_pof reprodutibilidade

Caminhos vem de CASE_AMBEV_RAW_ROOT (obrigatoria), CASE_AMBEV_SAIDA_ROOT e CASE_AMBEV_SELADO_ROOT (opcionais).
"""

import argparse
import datetime as dt
import json
import platform
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import duckdb

from occasion.crosswalk import (COLUNAS_FEATURES_COMUNS, carregar_crosswalk, conferir_contexto_uf, decodificar_estratos,
                                harmonizar_features, ler_contexto_uf, ler_planilha_estratos, nomes_uf_do_dicionario,
                                validar_contra_dicionarios)
from occasion.episodios_2008 import construir_e2_2008
from occasion.episodios_2017 import construir_e2_pizza_2017, gravar_conjuntos_2017, pizza
from occasion.fontes import (EDICOES, Manifesto, carregar_manifesto, coletar, extrair_membros, raiz_bruta, raiz_saidas,
                             raiz_selado)
from occasion.microdados import (Layout, carregar_registro, ler_dicionario_xls, layout_de, perfil_consumo_2008,
                                 perfil_consumo_2017, somas_de_peso_morador)
from occasion.verificacao import (ARQUIVO_HASHES, REFERENCIAS, ConferenciaFalhou, chave_domicilio,
                                  chaves_completas, chaves_edicao_uf, conferir, escrever_csv, escrever_json, hashes_das_saidas,
                                  interseccao_chaves_domicilio)

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
MANIFESTO = CONFIGS / "manifesto_fontes.json"
CROSSWALK = CONFIGS / "crosswalk.json"
MEMORIA_DUCKDB = "1500MB"
COLUNAS = {
    "consumo_2008": ["COD_UF", "NUM_SEQ", "NUM_DV", "NUM_DOM", "NUM_UC", "COD_INFORMANTE", "ESTRATO_POF", "PESO_FINAL",
                     "NUM_QUADRO", "V9018", "V9015", "V9001", "RENDA_TOTAL", "DIA_SEMANA"],
    "morador_2008": ["COD_UF", "NUM_SEQ", "NUM_DV", "NUM_DOM", "NUM_UC", "COD_INFORMANTE", "PESO_FINAL", "IDADE_ANOS",
                     "V0405", "TIPO_SITUACAO_REG", "NIVEL_INSTRUCAO_MORADOR"],
    "domicilio_2008": ["COD_UF", "NUM_SEQ", "NUM_DV", "NUM_DOM", "PERIODO_REAL"],
    "consumo_2017": ["UF", "ESTRATO_POF", "TIPO_SITUACAO_REG", "COD_UPA", "NUM_DOM", "NUM_UC", "COD_INFORMANTE", "QUADRO",
                     "SEQ", "V9005", "V9007", "V9001", "V9015", "V9017", "V9018", "PESO_FINAL", "RENDA_TOTAL",
                     "DIA_SEMANA", "DIA_ATIPICO"],
    "morador_2017": ["UF", "COD_UPA", "NUM_DOM", "NUM_UC", "COD_INFORMANTE", "V0306", "V0403", "V0404", "PESO_FINAL", "NIVEL_INSTRUCAO"],
}
# (membro do zip de dados, aba do dicionario) por tabela
ORIGEM_OFICIAL = {
    "consumo_2008": ("T_CONSUMO_S.txt", "Consumo Alimentar"), "morador_2008": ("T_MORADOR_S.txt", "Morador"),
    "domicilio_2008": ("T_DOMICILIO_S.txt", "Domicílio"),
    "consumo_2017": ("CONSUMO_ALIMENTAR.txt", "Consumo Alimentar"), "morador_2017": ("MORADOR.txt", "Morador"),
}


@dataclass(frozen=True)
class Entradas:
    arquivos: Mapping[str, Path]
    layouts: Mapping[str, Layout]
    estratos: Mapping[str, Mapping[tuple[str, int], str]]  # edicao -> {(uf, codigo do estrato): tipo}
    contexto_uf: list[dict]
    dicionarios: Mapping[str, Mapping] | None = None  # se dado, o crosswalk e conferido contra eles


def entradas_oficiais(manifesto: Manifesto, raiz: Path) -> Entradas:
    """Extrai os membros do manifesto (com hash) e monta os layouts a partir dos dicionarios oficiais."""
    caminhos: dict[str, Path] = {}
    for fonte in manifesto.fontes:
        for caminho in extrair_membros(fonte, raiz):
            caminhos[f"{fonte.edicao}|{caminho.name}"] = caminho
    dicionarios = {ed: ler_dicionario_xls(_achar(caminhos, ed, "Dicionarios_de_variaveis.xls")) for ed in EDICOES}
    arquivos, layouts = {}, {}
    for tabela, (membro, aba) in ORIGEM_OFICIAL.items():
        edicao = "2008-2009" if tabela.endswith("2008") else "2017-2018"
        arquivos[tabela] = _achar(caminhos, edicao, membro)
        layouts[tabela] = layout_de(dicionarios[edicao][aba])
    nomes_uf = nomes_uf_do_dicionario(next(v for v in dicionarios["2017-2018"]["Consumo Alimentar"] if v.codigo == "UF"))
    planilhas = {"2008-2009": "Estratos_Geograficos_POF_2008-2009.xls", "2017-2018": "Estratos_POF_2017-2018.xls"}
    estratos = {ed: decodificar_estratos(ler_planilha_estratos(_achar(caminhos, ed, nome)), nomes_uf, ed)
                for ed, nome in planilhas.items()}
    contexto = carregar_crosswalk(CROSSWALK)["contexto_uf"]
    return Entradas(arquivos=arquivos, layouts=layouts, estratos=estratos, dicionarios=dicionarios,
                    contexto_uf=ler_contexto_uf(CONFIGS / contexto["arquivo"], contexto["sha256"]))


def _achar(caminhos: dict[str, Path], edicao: str, nome: str) -> Path:
    for chave, caminho in caminhos.items():
        if chave.startswith(edicao) and caminho.name == nome:
            return caminho
    raise FileNotFoundError(f"{nome} ({edicao}) nao esta entre os membros extraidos do manifesto")


def _carregar(con, entradas: Entradas, tabela: str, nome: str | None = None, where: str | None = None) -> int:
    return carregar_registro(con, nome or tabela, entradas.arquivos[tabela], entradas.layouts[tabela],
                             COLUNAS[tabela], where=where)


def _chaves_domicilio(con, tabela: str, edicao: str) -> set[str]:
    return {chave_domicilio(edicao, upa, dom)
            for upa, dom in con.execute(f"SELECT DISTINCT cod_upa, num_dom FROM {tabela}").fetchall()}


def _pessoas_e_domicilios(con, tabela: str) -> dict:
    p, d = con.execute(f"SELECT count(DISTINCT (cod_upa, num_dom, num_uc, cod_informante)), "
                       f"count(DISTINCT (cod_upa, num_dom)) FROM {tabela}").fetchone()
    return {"pessoas_com_pizza": p, "domicilios_com_pizza": d}


def _construir(con, entradas: Entradas) -> tuple[dict, dict, dict, dict]:
    for tabela in ("consumo_2008", "morador_2008", "domicilio_2008", "morador_2017"):
        _carregar(con, entradas, tabela)
    lista_17 = ",".join(f"'{c}'" for c in pizza("2017-2018"))
    _carregar(con, entradas, "consumo_2017", "consumo_2017_pizza", where=f"V9001 IN ({lista_17})")
    c08 = {**perfil_consumo_2008(con), **somas_de_peso_morador(con, "morador_2008", "IDADE_ANOS")}
    c17 = {**perfil_consumo_2017(con, entradas.arquivos["consumo_2017"], entradas.layouts["consumo_2017"]),
           **somas_de_peso_morador(con, "morador_2017", "V0403")}
    ufs_08, ufs_17 = c08.pop("ufs"), c17.pop("ufs")
    e08 = construir_e2_2008(con)
    j08 = e08.pop("juncoes")
    c08.update({**e08, **_pessoas_e_domicilios(con, "e2_2008")})
    e17 = construir_e2_pizza_2017(con)
    j17 = e17.pop("juncoes")
    c17.update({**e17, **_pessoas_e_domicilios(con, "x_2017")})
    tabela = carregar_crosswalk(CROSSWALK)
    conferidas = (validar_contra_dicionarios(tabela, entradas.dicionarios) if entradas.dicionarios is not None else None)
    conferir_contexto_uf(entradas.contexto_uf, sorted(set(ufs_08) | set(ufs_17)))
    j_feat = harmonizar_features(con, entradas.estratos, entradas.contexto_uf)
    chaves_unicas = {**j08.pop("chaves_unicas"), **j17.pop("chaves_unicas"), **j_feat.pop("chaves_unicas")}
    juncoes = {**j08, **j17, **j_feat, "chaves_unicas": chaves_unicas}
    chaves = chaves_edicao_uf(ufs_08, ufs_17)
    completas = chaves_completas(chaves)
    comuns = interseccao_chaves_domicilio(
        _chaves_domicilio(con, "e2_2008", "2008-2009"), _chaves_domicilio(con, "x_2017", "2017-2018"))
    contagens = {"2008-2009": c08, "2017-2018": c17}
    integracao = {"chaves_edicao_uf": len(chaves) if completas else f"{len(chaves)} (conjunto incompleto)",
                  "chaves_domicilio_prefixadas_em_comum": len(comuns)}
    crosswalk = {"conceitos": len(tabela["conceitos"]), "variaveis_conferidas_nos_dicionarios": conferidas,
                 "colunas_features_comuns": list(COLUNAS_FEATURES_COMUNS)}
    return contagens, {"chaves": chaves, **integracao}, juncoes, crosswalk


def executar(entradas: Entradas, saida: Path, selado: Path, referencias: Mapping[str, int] | None = None,
             memoria: str = MEMORIA_DUCKDB) -> dict:
    """Roda o pacote. Se `referencias` for dado e algo divergir, levanta ConferenciaFalhou antes de gravar."""
    saida, selado = Path(saida), Path(selado)
    with tempfile.TemporaryDirectory() as tmp:
        con = duckdb.connect()
        con.execute(f"SET memory_limit='{memoria}'")
        con.execute("SET enable_progress_bar=false")
        con.execute("SET temp_directory=?", [tmp])
        contagens, integracao, juncoes, crosswalk = _construir(con, entradas)
        obtidos = {f"{ed}.{k}": v for ed, c in contagens.items() for k, v in c.items()}
        obtidos.update({k: v for k, v in integracao.items() if k != "chaves"})
        conferencias = conferir(obtidos, referencias or {})
        reprovadas = [c for c in conferencias if not c["ok"]]
        if reprovadas:
            raise ConferenciaFalhou("; ".join(f"{c['nome']}: esperado {c['esperado']}, obtido {c['obtido']}"
                                              for c in reprovadas))
        relatorio = {"contagens": contagens, "chaves_edicao_uf": integracao["chaves_edicao_uf"],
                     "chaves_domicilio_prefixadas_em_comum": integracao["chaves_domicilio_prefixadas_em_comum"],
                     "juncoes": juncoes, "crosswalk": crosswalk, "conferencias": conferencias}
        con.execute("CREATE OR REPLACE TABLE chaves_edicao_uf(edicao VARCHAR, uf VARCHAR)")
        con.executemany("INSERT INTO chaves_edicao_uf VALUES (?, ?)", integracao["chaves"])
        escrever_csv(con, "e2_2008", saida / "e2_2008.csv", "episodio_id")
        escrever_csv(con, "chaves_edicao_uf", saida / "chaves_edicao_uf.csv", "edicao, uf")
        escrever_csv(con, "features_comuns_2008", saida / "features_comuns_2008.csv", "episodio_id")
        escrever_csv(con, "features_comuns_2017", saida / "features_comuns_2017.csv", "episodio_id")
        gravar_conjuntos_2017(con, saida, selado)
        con.close()
    escrever_json(relatorio, saida / "contagens.json")
    escrever_json(hashes_das_saidas(saida, selado), saida / ARQUIVO_HASHES)
    return relatorio


def _escrever(objeto) -> None:
    sys.stdout.write(json.dumps(objeto, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _cmd_coletar(args) -> int:
    manifesto, raiz = carregar_manifesto(MANIFESTO), raiz_bruta()
    resultados = coletar(manifesto, raiz)
    for fonte in manifesto.fontes:
        extrair_membros(fonte, raiz)
    _escrever([{"fonte": r.fonte_id, "status": r.status} for r in resultados])
    return 0


def _cmd_executar(args) -> int:
    manifesto, raiz = carregar_manifesto(MANIFESTO), raiz_bruta()
    saida, selado = Path(args.saida or raiz_saidas()), Path(args.selado or raiz_selado())
    inicio = dt.datetime.now(dt.timezone.utc)
    coleta = coletar(manifesto, raiz)
    relatorio = executar(entradas_oficiais(manifesto, raiz), saida, selado,
                         referencias=None if args.sem_referencias else REFERENCIAS)
    log = {"inicio_utc": inicio.isoformat(timespec="seconds"), "fim_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "python": platform.python_version(), "duckdb": duckdb.__version__,
           "coleta": [{"fonte": r.fonte_id, "status": r.status} for r in coleta],
           "hashes": hashes_das_saidas(saida, selado), "relatorio": relatorio}
    pasta_logs = raiz / "pkg_f1_01" / "logs"
    pasta_logs.mkdir(parents=True, exist_ok=True)
    (pasta_logs / f"execucao_{inicio:%Y%m%dT%H%M%SZ}.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _escrever(log)
    return 0


def _cmd_reprodutibilidade(_args) -> int:
    manifesto, raiz = carregar_manifesto(MANIFESTO), raiz_bruta()
    coletar(manifesto, raiz)
    entradas = entradas_oficiais(manifesto, raiz)
    selado = raiz_selado()  # a raiz selada ja e somente leitura: o teste de reprodutibilidade usa pasta irma
    base, base_selado = raiz_saidas() / "reprodutibilidade", selado.parent / f"{selado.name}_reprodutibilidade"
    hashes = []
    for i in (1, 2):
        executar(entradas, base / f"execucao_{i}", base_selado / f"execucao_{i}", referencias=REFERENCIAS)
        hashes.append(hashes_das_saidas(base / f"execucao_{i}", base_selado / f"execucao_{i}"))
    iguais = hashes[0] == hashes[1]
    _escrever({"hashes_iguais": iguais, "execucao_1": hashes[0], "execucao_2": hashes[1]})
    return 0 if iguais else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="occasion.pipeline_pof", description=__doc__)
    sub = parser.add_subparsers(dest="comando", required=True)
    sub.add_parser("coletar").set_defaults(func=_cmd_coletar)
    p = sub.add_parser("executar")
    p.add_argument("--saida")
    p.add_argument("--selado")
    p.add_argument("--sem-referencias", action="store_true")
    p.set_defaults(func=_cmd_executar)
    p = sub.add_parser("reprodutibilidade")
    p.set_defaults(func=_cmd_reprodutibilidade)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
