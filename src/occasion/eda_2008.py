"""PE-02: analise exploratoria de desenvolvimento, SO com POF 2008-2009 (nada de 2017 e aberto).

  python -m occasion.eda_2008 --saida <pe01/saida> --destino <docs/eda_2008> [--raw <pasta dos dados publicos>]

Leitura principal sem pesos. O peso W1 (peso da pessoa / dias de registro da pessoa no arquivo de consumo) e uma
construcao nossa, usada so como sensibilidade; sem --raw as colunas W1 nao sao geradas. Saidas: tabelas agregadas,
figuras SVG conferidas contra as tabelas e `resultados.json` com hashes.

Convencao de mascaramento (do projeto, sem garantia de anonimizacao): nas tabelas e figuras, niveis com 1 a 4 episodios
saem com n, k e taxas suprimidos. Niveis com n >= 5 saem completos, mesmo com k de 1 a 4. A convencao vale para as
tabelas; nao cobre a prosa dos documentos.

A pasta de entradas vem de --saida ou de CASE_AMBEV_SAIDA_ROOT e so e aceita se os cinco arquivos tiverem os SHA-256
de `configs/entradas_2008.json`.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from occasion.eda_dados import (carregar_episodios, chave_pessoa, chave_uc, conferir_entradas, contagens,
                                verificar_integridade)
from occasion.eda_graficos import conferir_figura, figura_taxas
from occasion.fontes import raiz_saidas
from occasion.eda_taxas import (CELULA_MINIMA, colunas_redundantes, cramer_v, diferenca_por_upa, grupos_completos,
                                rotulo_contagem, taxas_por_nivel)

SEMENTE, REPLICAS = 20261113, 5000
SEMENTES_ESTABILIDADE = (SEMENTE, 1, 2, 3, 4)
TABELAS_ESTABILIDADE = ("periodo", "dia_semana", "grande_regiao")
FAIXAS = ["manha", "almoco", "tarde", "noite"]
REGIOES = ["Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"]
IDADES = ["10_17", "18_29", "30_44", "45_59", "60_mais"]
ESCOLAR = ["ate_fundamental", "medio", "superior"]
AREAS = ["capital", "urbano_nao_capital", "rural"]
DIAS = {1: "1_seg", 2: "2_ter", 3: "3_qua", 4: "4_qui", 5: "5_sex", 6: "6_sab", 7: "7_dom"}
GRUPOS_INDICADORES = {"hora": [f"hora_{f}" for f in FAIXAS], "regiao": [f"regiao_{r.lower().replace('-', '_')}" for r in REGIOES],
                      "area": [f"area_{a}" for a in AREAS], "idade": [f"idade_{i}" for i in IDADES],
                      "escolaridade": [f"escolaridade_{e}" for e in ESCOLAR]}


def _sha256(caminho: Path) -> str:
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def dias_de_registro(raw: Path) -> dict:
    """Dias distintos de registro por pessoa no arquivo de consumo de 2008 (base do peso W1). Exige os microdados."""
    import duckdb
    from occasion.microdados import carregar_registro, layout_de, ler_dicionario_xls
    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false")
    dic = ler_dicionario_xls(raw / "pof_2008_2009/extraido/documentacao/Dicionarios_de_variaveis.xls")
    carregar_registro(con, "consumo", raw / "pof_2008_2009/extraido/dados/T_CONSUMO_S.txt", layout_de(dic["Consumo Alimentar"]),
                      ["COD_UF", "NUM_SEQ", "NUM_DV", "NUM_DOM", "NUM_UC", "COD_INFORMANTE", "NUM_QUADRO"])
    linhas = con.execute("""
        SELECT lpad(CAST(COD_UF AS INT)::VARCHAR, 2, '0') || lpad(CAST(NUM_SEQ AS INT)::VARCHAR, 3, '0') || CAST(NUM_DV AS INT)::VARCHAR,
               CAST(NUM_DOM AS INT), CAST(NUM_UC AS INT), CAST(COD_INFORMANTE AS INT), count(DISTINCT NUM_QUADRO)
        FROM consumo GROUP BY ALL""").fetchall()
    return {(u, d, c, i): n for u, d, c, i, n in linhas}


def _chave_dias(e: dict) -> tuple:
    u, d, c, i = chave_pessoa(e)
    return (u, int(d), int(c), int(i))


def _tercis(eps: list[dict]) -> tuple[float, float]:
    rendas = sorted(e["nat"]["log_renda_pc"] for e in eps if e["nat"]["log_renda_pc"] is not None)
    return rendas[len(rendas) // 3], rendas[2 * len(rendas) // 3]


def _faixa_renda(valor, cortes) -> str:
    if valor is None:
        return "sem_renda"
    return "1_menor" if valor < cortes[0] else "2_media" if valor < cortes[1] else "3_maior"


def _moradores(n: int) -> str:
    return str(n) if n < 5 else "5_mais"


def _presenca(v) -> str:
    return "desconhecida" if v is None else str(v)


def especificar(eps: list[dict]) -> dict:
    """nome da tabela -> (funcao episodio->nivel, niveis ordenados). Cortes de renda so descritivos (todos os de 2008)."""
    cortes = _tercis(eps)
    ufs = sorted({e["e2"]["uf"] for e in eps})
    ordem = [f"{d}|{f}" for d in ("seg_qui", "sex_dom") for f in FAIXAS]
    return {
        "faixa_hora": (lambda e: e["nat"]["faixa_hora"], FAIXAS),
        "global": (lambda e: "todos", ["todos"]),
        "dia_coleta": (lambda e: str(e["fc"]["dia_coleta"]), sorted({str(e["fc"]["dia_coleta"]) for e in eps})),
        "dia_semana": (lambda e: DIAS[e["nat"]["dia_semana"]], list(DIAS.values())),
        "sexta_a_domingo": (lambda e: "sex_dom" if e["nat"]["sexta_a_domingo"] else "seg_qui", ["seg_qui", "sex_dom"]),
        "periodo": (lambda e: f"{'sex_dom' if e['nat']['sexta_a_domingo'] else 'seg_qui'}|{e['nat']['faixa_hora']}", ordem),
        "grande_regiao": (lambda e: e["nat"]["grande_regiao"], REGIOES),
        "uf": (lambda e: e["e2"]["uf"], ufs),
        "area": (lambda e: e["nat"]["area"], AREAS),
        "faixa_etaria": (lambda e: e["nat"]["faixa_etaria"], IDADES),
        "sexo": (lambda e: e["nat"]["sexo"], ["feminino", "masculino"]),
        "escolaridade": (lambda e: e["nat"]["escolaridade"], ESCOLAR),
        "renda_per_capita_tercil": (lambda e: _faixa_renda(e["nat"]["log_renda_pc"], cortes), ["1_menor", "2_media", "3_maior", "sem_renda"]),
        "moradores_uc": (lambda e: _moradores(e["nat"]["moradores_uc"]), ["1", "2", "3", "4", "5_mais"]),
        "presenca_crianca": (lambda e: _presenca(e["nat"]["presenca_crianca"]), ["0", "1", "desconhecida"]),
        "presenca_adolescente": (lambda e: _presenca(e["nat"]["presenca_adolescente"]), ["0", "1", "desconhecida"]),
        "presenca_idoso": (lambda e: _presenca(e["nat"]["presenca_idoso"]), ["0", "1", "desconhecida"]),
    }


def tabelas_de_taxas(eps: list[dict], pesos: dict | None, replicas: int = REPLICAS) -> dict[str, list[dict]]:
    """Uma tabela por variavel: n, k, taxa sem peso com IC por UPA e, se houver `pesos`, a sensibilidade W1."""
    tabelas = {}
    for nome, (fn, niveis) in especificar(eps).items():
        obs = [{"nivel": fn(e), "cluster": e["e2"]["cod_upa"], "y": e["y"],
                "w": (float(e["e2"]["peso_final"]) / pesos[_chave_dias(e)]) if pesos else 1.0} for e in eps]
        tabelas[nome] = taxas_por_nivel(obs, niveis, replicas=replicas, semente=SEMENTE, ponderar=pesos is not None)
    return tabelas


def _publicavel(linha: dict) -> dict:
    """Linha para publicacao: n de 1 a 4 mascarado e sem taxas (poucas pessoas)."""
    if 0 < linha["n"] < CELULA_MINIMA:
        return {c: (None if c not in ("nivel", "suporte") else v) for c, v in linha.items()} | {"n": "<5", "k": "<5"}
    return dict(linha)


def gravar_tabela(linhas: list[dict], caminho: Path) -> None:
    publicas = [_publicavel(l) for l in linhas]
    colunas = list(publicas[0])
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(colunas)
        w.writerows([[("" if v is None else (round(v, 5) if isinstance(v, float) else v)) for v in l.values()] for l in publicas])


def estrutura(eps: list[dict]) -> dict:
    """Dependencia entre episodios: quantos por pessoa, domicilio e UPA, e como os positivos se agrupam."""
    def distribuicao(chave) -> dict:
        grupos = Counter(chave(e) for e in eps)
        return {"grupos": len(grupos), "episodios_por_grupo": dict(sorted(Counter(grupos.values()).items())),
                "episodios_em_grupos_com_2_ou_mais": sum(n for n in grupos.values() if n > 1)}

    def positivos(chave) -> dict:
        total, pos = Counter(chave(e) for e in eps), Counter(chave(e) for e in eps if e["y"])
        return {"grupos_com_positivo": len(pos), "positivos_por_grupo": dict(sorted(Counter(pos.values()).items())),
                "positivos_em_grupos_com_2_ou_mais_positivos": sum(n for n in pos.values() if n > 1),
                "grupos_com_positivo_e_2_ou_mais_episodios": sum(1 for g in pos if total[g] > 1)}

    dom = lambda e: e["e2"]["cod_upa"] + "|" + e["e2"]["num_dom"]  # noqa: E731
    upa = lambda e: e["e2"]["cod_upa"]  # noqa: E731
    return {"pessoa": distribuicao(chave_pessoa), "domicilio": distribuicao(dom), "upa": distribuicao(upa),
            "positivos_domicilio": positivos(dom), "positivos_upa": positivos(upa),
            "estratos_com_uma_upa": sum(1 for v in _upas_por_estrato(eps).values() if v == 1)}


def _upas_por_estrato(eps: list[dict]) -> dict:
    por: dict = defaultdict(set)
    for e in eps:
        por[(e["e2"]["uf"], e["e2"]["estrato_pof"])].add(e["e2"]["cod_upa"])
    return {k: len(v) for k, v in por.items()}


def _quantis(valores: list[float], pontos=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)) -> dict:
    ordenados = sorted(valores)
    saida = {f"p{int(p * 100):02d}": ordenados[min(len(ordenados) - 1, int(p * len(ordenados)))] for p in pontos}
    return {"min": ordenados[0], **saida, "max": ordenados[-1], "n": len(ordenados)}


def qualidade(eps: list[dict]) -> dict:
    """Ausencias, valores extremos e categorias raras (so descricao das variaveis; nada do alvo)."""
    rendas = [math.exp(e["nat"]["log_renda_pc"]) for e in eps if e["nat"]["log_renda_pc"] is not None]
    mediana = statistics.median(rendas)
    logs = [e["nat"]["log_renda_pc"] for e in eps if e["nat"]["log_renda_pc"] is not None]
    q1, q3 = _quantis(logs)["p25"], _quantis(logs)["p75"]
    desconhecidos = {c: sum(e["mat"][c] is None for e in eps) for c in eps[0]["mat"]}
    return {"colunas_com_desconhecido": {c: n for c, n in desconhecidos.items() if n},
            "renda_status": dict(Counter(e["nat"]["renda_status"] for e in eps)),
            "renda_per_capita_reais_jan2009": _quantis(rendas), "renda_acima_de_10_medianas": sum(r > 10 * mediana for r in rendas),
            "log_renda_fora_de_1_5_iqr": sum(1 for v in logs if v < q1 - 1.5 * (q3 - q1) or v > q3 + 1.5 * (q3 - q1)),
            "idade": _quantis([e["nat"]["idade"] for e in eps]), "moradores_uc": dict(sorted(Counter(e["nat"]["moradores_uc"] for e in eps).items())),
            "peso_da_pessoa": _quantis([float(e["e2"]["peso_final"]) for e in eps]),
            "episodios_de_madrugada_0h_a_4h": sum(int(e["fc"]["hora"]) <= 4 for e in eps)}


def categorias_raras(tabelas: dict) -> list[dict]:
    """Niveis com menos de 30 episodios ou menos de 5 positivos (leitura so descritiva); n de 1 a 4 sai mascarado, com k."""
    return [{"tabela": t, "nivel": l["nivel"], "n": rotulo_contagem(l["n"]), "k": rotulo_contagem(l["n"]) if 0 < l["n"] < CELULA_MINIMA else l["k"],
             "suporte": l["suporte"]}
            for t, linhas in tabelas.items() for l in linhas if l["suporte"] != "ok"]


def _por_uc(eps: list[dict]) -> dict:
    por: dict = defaultdict(list)
    for e in eps:
        por[chave_uc(e)].append(e)
    return por


def redundancias(eps: list[dict]) -> dict:
    """Colinearidade da matriz de 28 colunas e associacao entre as variaveis nativas (sem o alvo)."""
    colunas = list(eps[0]["mat"])
    matriz = {c: [e["mat"][c] for e in eps] for c in colunas}
    espec = especificar(eps)
    categoricas = ["faixa_hora", "sexta_a_domingo", "grande_regiao", "area", "faixa_etaria", "sexo", "escolaridade",
                   "renda_per_capita_tercil", "moradores_uc", "presenca_crianca", "presenca_adolescente", "presenca_idoso"]
    valores = {c: [espec[c][0](e) for e in eps] for c in categoricas}
    pares_v = [{"a": a, "b": b, "cramer_v": cramer_v(valores[a], valores[b])} for i, a in enumerate(categoricas) for b in categoricas[i + 1:]]
    uc_inconsistente = sum(1 for lote in _por_uc(eps).values() if len({(e["nat"]["moradores_uc"], e["nat"]["log_renda_pc"]) for e in lote}) > 1)
    somam = grupos_completos(matriz, GRUPOS_INDICADORES)
    return {"grupos_de_indicadores_somam_1": somam, "colunas": len(colunas), "grupos_completos_colineares_com_intercepto": sum(somam.values()),
            "f45_igual_ao_produto": all(e["mat"]["sexta_a_domingo_x_noite"] == e["mat"]["sexta_a_domingo"] * e["mat"]["hora_noite"] for e in eps),
            "pares_de_colunas_com_correlacao_abs_ge_0_5": [{**p, "correlacao": round(p["correlacao"], 3)} for p in colunas_redundantes(matriz, 0.5)],
            "pares_categoricos_com_cramer_v_ge_0_3": sorted([dict(p, cramer_v=round(p["cramer_v"], 3)) for p in pares_v if p["cramer_v"] and p["cramer_v"] >= 0.3],
                                                            key=lambda p: -p["cramer_v"]),
            "unidades_de_consumo_com_valores_diferentes_entre_episodios": uc_inconsistente}


def contrastes(eps: list[dict], replicas: int) -> dict:
    """Contraste planejado em 13/11, depois de ver as marginais de 2008: a variavel `sexta_a_domingo` foi fixada antes do alvo
    (plano 7.3), mas a comparacao de taxas foi escrita depois. Em 2008 e estimativa exploratoria; so 2017 permite leitura confirmatoria."""
    nivel = especificar(eps)["sexta_a_domingo"][0]
    obs = [{"nivel": nivel(e), "cluster": e["e2"]["cod_upa"], "y": e["y"]} for e in eps]
    return {"sexta_a_domingo_menos_segunda_a_quinta": diferenca_por_upa(obs, "sex_dom", "seg_qui", replicas, SEMENTE)}


def bootstrap_registro(tabelas: dict, contraste: dict) -> dict:
    """Replicas descartadas (nivel nao sorteado) e niveis com bootstrap degenerado (k = 0 ou k = n), por tabela. So niveis com n >= 5."""
    visiveis = {t: [l for l in linhas if l["n"] >= CELULA_MINIMA] for t, linhas in tabelas.items()}  # sem niveis vazios nem suprimidos
    descartadas = {t: {l["nivel"]: l["boot_descartadas"] for l in linhas if l["boot_descartadas"]} for t, linhas in visiveis.items()}
    degenerados = {t: [l["nivel"] for l in linhas if l["boot_degenerado"]] for t, linhas in visiveis.items()}
    return {"replicas_descartadas_por_nivel": {t: d for t, d in descartadas.items() if d},
            "niveis_com_bootstrap_degenerado": {t: d for t, d in degenerados.items() if d},
            "motivo_do_descarte": "o nivel nao foi sorteado em nenhuma UPA da reamostra: taxa indefinida",
            "contraste_replicas_descartadas": {c: v["replicas"] - v["replicas_validas"] for c, v in contraste.items()}}


def estabilidade(eps: list[dict], replicas: int, sementes=SEMENTES_ESTABILIDADE, tabelas=TABELAS_ESTABILIDADE) -> dict:
    """Variacao dos limites do bootstrap (pontos percentuais) entre sementes, para as tabelas citadas no texto e o contraste."""
    espec, saida = especificar(eps), {}
    for nome in tabelas:
        fn, niveis = espec[nome]
        obs = [{"nivel": fn(e), "cluster": e["e2"]["cod_upa"], "y": e["y"]} for e in eps]
        por_semente = [taxas_por_nivel(obs, niveis, replicas=replicas, semente=s) for s in sementes]
        amplitude = 0.0
        for linhas_do_nivel in zip(*por_semente):
            for limite in ("boot_inf", "boot_sup"):
                valores = [l[limite] for l in linhas_do_nivel if l[limite] is not None]
                if len(valores) == len(sementes):
                    amplitude = max(amplitude, max(valores) - min(valores))
        saida[nome] = round(100 * amplitude, 2)
    nivel = espec["sexta_a_domingo"][0]
    obs = [{"nivel": nivel(e), "cluster": e["e2"]["cod_upa"], "y": e["y"]} for e in eps]
    por_semente = [diferenca_por_upa(obs, "sex_dom", "seg_qui", replicas, s) for s in sementes]
    inferiores, superiores = [round(100 * d["boot_inf"], 2) for d in por_semente], [round(100 * d["boot_sup"], 2) for d in por_semente]
    return {"sementes": list(sementes), "replicas": replicas, "amplitude_maxima_dos_limites_em_pp": saida,
            "contraste_limite_inferior_pp": inferiores, "contraste_limite_superior_pp": superiores,
            "contraste_amplitude_maxima_pp": round(max(max(inferiores) - min(inferiores), max(superiores) - min(superiores)), 2)}


def figuras(tabelas: dict, destino: Path, referencia: float) -> dict:
    """Quatro figuras que sustentam leituras do texto; cada uma e conferida contra a tabela de origem."""
    perfil = [dict(l, nivel=f"{prefixo}: {l['nivel']}") for prefixo, tab in
              (("idade", "faixa_etaria"), ("escolaridade", "escolaridade"), ("renda", "renda_per_capita_tercil"), ("sexo", "sexo"))
              for l in tabelas[tab] if l["n"]]
    pedidos = {"taxa_periodo": ("Guarana entre episodios com pizza: dia x periodo (2008-2009)", tabelas["periodo"]),
               "taxa_dia_semana": ("Guarana entre episodios com pizza: dia da semana (2008-2009)", tabelas["dia_semana"]),
               "taxa_regiao": ("Guarana entre episodios com pizza: grande regiao (2008-2009)", tabelas["grande_regiao"]),
               "taxa_perfil": ("Guarana entre episodios com pizza: perfil (2008-2009)", perfil)}
    conferencia = {}
    for nome, (titulo, linhas) in pedidos.items():
        caminho = destino / f"{nome}.svg"
        linhas = [l for l in linhas if not 0 < l["n"] < CELULA_MINIMA]  # mesma regra de supressao das tabelas
        figura_taxas(titulo, linhas, caminho, referencia=referencia)
        conferencia[nome] = conferir_figura(caminho, linhas)
    return conferencia


def verificar_tabelas(tabelas: dict, total: int, positivos: int) -> dict:
    """Variaveis que particionam os episodios devem somar 1.168 episodios e 101 positivos."""
    soma = {t: (sum(l["n"] for l in linhas), sum(l["k"] for l in linhas)) for t, linhas in tabelas.items()}
    return {"ok": all(v == (total, positivos) for v in soma.values()),
            "divergentes": {t: v for t, v in soma.items() if v != (total, positivos)}}


def executar(saida: Path, destino: Path, raw: Path | None, replicas: int = REPLICAS) -> dict:
    saida, destino = Path(saida), Path(destino)
    entradas = conferir_entradas(saida)  # recusa pasta antiga ou arquivo com hash diferente do registrado
    (destino / "tabelas").mkdir(parents=True, exist_ok=True)
    (destino / "figuras").mkdir(parents=True, exist_ok=True)
    eps = carregar_episodios(saida)
    integridade = verificar_integridade(eps)
    if not integridade["ok"]:
        raise ValueError(f"integridade falhou: {[c for c in integridade['checagens'] if not c['ok']]}")
    pesos = dias_de_registro(raw) if raw else None
    if pesos is not None and any(_chave_dias(e) not in pesos for e in eps):
        raise ValueError("episodio sem pessoa no arquivo de consumo (dias de registro)")
    tabelas = tabelas_de_taxas(eps, pesos, replicas)
    for nome, linhas in tabelas.items():
        gravar_tabela(linhas, destino / "tabelas" / f"{nome}.csv")
    c = contagens(eps)
    ref = c["positivos"] / c["episodios"]
    contraste = contrastes(eps, replicas)
    verif = verificar_tabelas(tabelas, c["episodios"], c["positivos"])
    graficos = figuras(tabelas, destino / "figuras", ref)
    resultado = {
        "versao": "PE-02", "escopo": "somente POF 2008-2009; nenhum arquivo de 2017 lido",
        "parametros": {"semente": SEMENTE, "replicas_bootstrap_de_upas": replicas,
                       "peso_w1": "peso da pessoa / dias de registro (construcao nossa)" if pesos else "nao calculado"},
        "contagens": c, "alvo": {"y_guarana": c["positivos"], "taxa": ref, "episodios_com_refrigerante": c["refrigerante"],
                                  "guarana_entre_refrigerante": c["positivos"] / c["refrigerante"]},
        "integridade": integridade, "contrastes": contraste, "bootstrap": bootstrap_registro(tabelas, contraste),
        "estabilidade": estabilidade(eps, replicas), "estrutura": estrutura(eps), "qualidade": qualidade(eps), "redundancias": redundancias(eps),
        "categorias_raras": categorias_raras(tabelas), "verificacao_das_tabelas": verif, "conferencia_dos_graficos": graficos,
        "dias_de_registro": dict(sorted(Counter(pesos[_chave_dias(e)] for e in eps).items())) if pesos else None,
        "entradas_sha256": entradas,
        "saidas_sha256": {str(p.relative_to(destino)): _sha256(p) for p in sorted(destino.rglob("*"))
                          if p.is_file() and p.name != "resultados.json"},
    }
    (destino / "resultados.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return resultado


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="occasion.eda_2008", description=__doc__)
    p.add_argument("--saida", default=None, help="pasta com as cinco saidas de 2008 do PE-01 (padrao: CASE_AMBEV_SAIDA_ROOT)")
    p.add_argument("--destino", required=True)
    p.add_argument("--raw", default=os.environ.get("CASE_AMBEV_RAW_ROOT"), help="dados publicos (para o peso W1)")
    p.add_argument("--replicas", type=int, default=REPLICAS)
    args = p.parse_args(argv)
    r = executar(Path(args.saida or raiz_saidas()), Path(args.destino), Path(args.raw) if args.raw else None, args.replicas)
    sys.stdout.write(f"EDA 2008: {r['contagens']['episodios']} episodios, {r['contagens']['positivos']} positivos; "
                     f"integridade={r['integridade']['ok']}; tabelas={r['verificacao_das_tabelas']['ok']}; "
                     f"graficos={all(g['ok'] for g in r['conferencia_dos_graficos'].values())}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
