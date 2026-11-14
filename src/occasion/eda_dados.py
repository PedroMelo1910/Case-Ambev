"""EDA de 2008-2009 (PE-02): carga dos cinco arquivos do PE-01, alinhamento por `episodio_id` e integridade.

So le saidas de 2008. As matrizes (`variaveis_nativas`, `matriz_logistica`) nao trazem `episodio_id`: o alinhamento
delas e por posicao, que o PE-01 fixa ordenando por `episodio_id`. Aqui a posicao e conferida contra os campos
comuns antes de qualquer tabulacao.
"""

import csv
import hashlib
import json
from pathlib import Path

ARQUIVOS = ("e2", "fc", "ids", "nat", "mat")
NOMES_ARQUIVOS = {"e2": "e2_2008", "fc": "features_comuns_2008", "ids": "identificadores_2008",
          "nat": "variaveis_nativas_2008", "mat": "matriz_logistica_2008"}
_NUMERICOS = {"nat": {"dia_semana", "sexta_a_domingo", "idade", "log_renda_pc", "moradores_uc", "presenca_crianca",
                      "presenca_adolescente", "presenca_idoso"}}


CONFIG_ENTRADAS = Path(__file__).resolve().parents[2] / "configs" / "entradas_2008.json"


class AlinhamentoInvalido(ValueError):
    """Os arquivos de 2008 nao tem os mesmos episodios, ordem ou chaves unicas."""


class EntradasInvalidas(ValueError):
    """A pasta nao contem as saidas definitivas de 2008 do PE-01 (arquivo ausente ou hash diferente do registrado)."""


def conferir_entradas(saida: Path, config: Path = CONFIG_ENTRADAS) -> dict[str, str]:
    """Confere os cinco arquivos de 2008 contra os SHA-256 de `configs/entradas_2008.json` e devolve os hashes.

    A pasta vem da configuracao existente (CASE_AMBEV_SAIDA_ROOT ou --saida), nunca de caminho fixo no codigo. Uma pasta
    antiga, sem as matrizes do PE-01, e recusada aqui, antes de qualquer tabulacao ou ajuste."""
    registrados = json.loads(Path(config).read_text(encoding="utf-8"))["arquivos"]
    obtidos = {}
    for nome, esperado in registrados.items():
        caminho = Path(saida) / nome
        if not caminho.is_file():
            raise EntradasInvalidas(f"{nome} ausente em {saida}: aponte CASE_AMBEV_SAIDA_ROOT para as saidas do PE-01")
        obtidos[nome] = hashlib.sha256(caminho.read_bytes()).hexdigest()
        if obtidos[nome] != esperado:
            raise EntradasInvalidas(f"{nome}: hash {obtidos[nome][:12]} diferente do registrado {esperado[:12]}")
    return obtidos


def _numero(texto: str):
    if texto == "":
        return None
    try:
        return int(texto)
    except ValueError:
        return float(texto)


def _ler(caminho: Path, numericos: set[str] | None = None, todos_numericos: bool = False) -> list[dict]:
    with open(caminho, newline="", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    for linha in linhas:
        for chave, valor in linha.items():
            if todos_numericos or chave in (numericos or ()):
                linha[chave] = _numero(valor)
    return linhas


def carregar_episodios(saida: Path) -> list[dict]:
    """Lista de episodios {id, y, e2, fc, ids, nat, mat}. Recusa duplicados, conjuntos diferentes e ordem trocada."""
    saida = Path(saida)
    tabelas = {k: _ler(saida / f"{NOMES_ARQUIVOS[k]}.csv", _NUMERICOS.get(k), todos_numericos=(k == "mat")) for k in ARQUIVOS}
    base = [l["episodio_id"] for l in tabelas["e2"]]
    if len(set(base)) != len(base):
        raise AlinhamentoInvalido("episodio_id duplicado em e2_2008")
    ordem = sorted(base)
    for chave in ("fc", "ids"):
        ids = [l["episodio_id"] for l in tabelas[chave]]
        if set(ids) != set(base) or len(ids) != len(base):
            raise AlinhamentoInvalido(f"{NOMES_ARQUIVOS[chave]} nao tem os mesmos episodios de e2_2008")
        if ids != ordem:
            raise AlinhamentoInvalido(f"{NOMES_ARQUIVOS[chave]} fora da ordem por episodio_id")
    for chave in ("nat", "mat"):
        if len(tabelas[chave]) != len(base):
            raise AlinhamentoInvalido(f"{NOMES_ARQUIVOS[chave]} tem {len(tabelas[chave])} linhas; esperado {len(base)}")
    por_id = {l["episodio_id"]: l for l in tabelas["e2"]}
    return [{"id": ident, "y": int(por_id[ident]["y_guarana"]), "e2": por_id[ident], "fc": tabelas["fc"][i],
             "ids": tabelas["ids"][i], "nat": tabelas["nat"][i], "mat": tabelas["mat"][i]}
            for i, ident in enumerate(ordem)]


def chave_pessoa(e: dict) -> tuple:
    d = e["e2"]
    return (d["cod_upa"], d["num_dom"], d["num_uc"], d["cod_informante"])


def chave_uc(e: dict) -> tuple:
    return chave_pessoa(e)[:3]


def contagens(eps: list[dict]) -> dict:
    """Episodios, positivos e as unidades em que se agrupam (pessoa, UC, domicilio, UPA, estrato, UF)."""
    domicilio = {e["e2"]["cod_upa"] + "|" + e["e2"]["num_dom"] for e in eps}
    return {"episodios": len(eps), "positivos": sum(e["y"] for e in eps), "pessoas": len({chave_pessoa(e) for e in eps}),
            "unidades_consumo": len({chave_uc(e) for e in eps}), "domicilios": len(domicilio),
            "upas": len({e["e2"]["cod_upa"] for e in eps}), "estratos": len({(e["e2"]["uf"], e["e2"]["estrato_pof"]) for e in eps}),
            "ufs": len({e["e2"]["uf"] for e in eps}), "refrigerante": sum(int(e["e2"]["y_refrigerante"]) for e in eps)}


def _checagem(nome: str, ok: bool, detalhe: str = "") -> dict:
    return {"nome": nome, "ok": bool(ok), "detalhe": detalhe}


def _regras_hora() -> dict:
    from occasion.variaveis import carregar_regras
    return {h: nome for nome, horas in carregar_regras()["faixas_hora"].items() for h in horas}


def _conferir(eps: list[dict], nome: str, teste) -> dict:
    def divergente(e: dict) -> bool:
        try:
            return not teste(e)
        except (KeyError, ValueError, TypeError):  # campo ausente ou fora do dominio conta como divergencia
            return True

    erros = sum(1 for e in eps if divergente(e))
    return _checagem(nome, erros == 0, f"{erros} divergencias em {len(eps)} episodios")


def verificar_integridade(eps: list[dict]) -> dict:
    """Checagens de juncao e consistencia entre os cinco arquivos. `ok` so e verdadeiro se todas passam."""
    faixa = _regras_hora()
    pesos_por_pessoa: dict = {}
    for e in eps:
        pesos_por_pessoa.setdefault(chave_pessoa(e), set()).add(e["e2"]["peso_final"])
    checagens = [
        _conferir(eps, "upa_ids_igual_e2", lambda e: e["ids"]["cod_upa"] == e["e2"]["cod_upa"]),
        _conferir(eps, "peso_ids_igual_e2", lambda e: float(e["ids"]["peso_final"]) == float(e["e2"]["peso_final"])),
        _conferir(eps, "dia_semana_nativas_igual_features_comuns", lambda e: int(e["fc"]["dia_semana"]) == e["nat"]["dia_semana"]),
        _conferir(eps, "dia_semana_e2_igual_features_comuns", lambda e: int(e["fc"]["dia_semana"]) == int(e["e2"]["dia_semana"])),
        _conferir(eps, "regiao_nativas_igual_features_comuns", lambda e: e["fc"]["grande_regiao"] == e["nat"]["grande_regiao"]),
        _conferir(eps, "area_nativas_igual_features_comuns", lambda e: e["fc"]["area_comparavel"] == e["nat"]["area"]),
        _conferir(eps, "idade_nativas_igual_features_comuns", lambda e: int(e["fc"]["idade"]) == e["nat"]["idade"]),
        _conferir(eps, "sexo_nativas_igual_features_comuns", lambda e: e["fc"]["sexo"] == e["nat"]["sexo"]),
        _conferir(eps, "faixa_hora_nativas_a_partir_da_hora",
                  lambda e: faixa[int(e["fc"]["hora"])] == e["nat"]["faixa_hora"]),
        _conferir(eps, "sexta_a_domingo_nativas_igual_matriz", lambda e: e["nat"]["sexta_a_domingo"] == e["mat"]["sexta_a_domingo"]),
        _conferir(eps, "faixa_hora_matriz_igual_nativas",
                  lambda e: e["mat"][f"hora_{e['nat']['faixa_hora']}"] == 1 and sum(e["mat"][f"hora_{h}"] for h in set(faixa.values())) == 1),
        _conferir(eps, "alvo_dentro_de_refrigerante", lambda e: int(e["e2"]["y_guarana"]) <= int(e["e2"]["y_refrigerante"])),
        _checagem("peso_constante_dentro_da_pessoa", all(len(p) == 1 for p in pesos_por_pessoa.values()),
                  f"{sum(len(p) > 1 for p in pesos_por_pessoa.values())} pessoas com mais de um peso"),
        _checagem("sem_episodio_duplicado", len({e["id"] for e in eps}) == len(eps)),
    ]
    return {"ok": all(c["ok"] for c in checagens), "checagens": checagens}
