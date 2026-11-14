"""PE-01: 13 variaveis derivadas e 28 colunas da regressao logistica (plano 7.3), iguais em 2008-2009 e 2017-2018.

Uma unica funcao (`derivar_nativas`) serve ao desenvolvimento e a futura inferencia. Regras e faixas ficam em
`configs/variaveis.json`. Nada aqui e aprendido dos dados: imputacao, padronizacao e cortes por quantil ficam
para dentro das dobras de treino. Valor desconhecido continua desconhecido (None/vazio), nunca vira zero.
Este modulo nao le alvo nem pesos de 2017 e nao importa o caminho de 2008, onde estao as bebidas.
"""

import json
import math
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from types import MappingProxyType

import duckdb

from occasion.crosswalk import conferir_chave_unica, registrar_juncao

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "variaveis.json"
_STATUS_RENDA = ("valida", "ausente", "zero", "negativa")
_COLUNAS_NATIVAS = (
    ("dia_semana", "INT"), ("faixa_hora", "VARCHAR"), ("sexta_a_domingo", "INT"), ("grande_regiao", "VARCHAR"),
    ("area", "VARCHAR"), ("faixa_etaria", "VARCHAR"), ("idade", "INT"), ("sexo", "VARCHAR"),
    ("escolaridade", "VARCHAR"), ("log_renda_pc", "DOUBLE"), ("renda_status", "VARCHAR"), ("moradores_uc", "INT"),
    ("presenca_crianca", "INT"), ("presenca_adolescente", "INT"), ("presenca_idoso", "INT"))
_COLUNAS_DOUBLE = ("log_renda_pc",)
_IDADE_MAXIMA = 111  # mesmo limite de crosswalk.idade_anos (dicionario de 2017: 0 a 111)


class RegrasInvalidas(ValueError):
    """`configs/variaveis.json` nao e coerente com a especificacao de 13 variaveis e 28 colunas."""


def _congelar(valor):
    if isinstance(valor, dict):
        return MappingProxyType({k: _congelar(v) for k, v in valor.items()})
    if isinstance(valor, list):
        return tuple(_congelar(v) for v in valor)
    return valor


def _conferir_cobertura(regras: Mapping) -> None:
    horas = [h for hs in regras["faixas_hora"].values() for h in hs]
    niveis = [n for ns in regras["escolaridade_3"].values() for n in ns]
    if sorted(horas) != list(range(24)) or sorted(niveis) != list(range(1, 8)):
        raise RegrasInvalidas("faixas de hora (0..23) e de escolaridade (1..7) devem cobrir o dominio sem sobreposicao")
    faixas = regras["faixas_etarias"]
    contiguas = all(a["max"] is not None and b["min"] == a["max"] + 1 for a, b in zip(faixas, faixas[1:]))
    if not contiguas or faixas[-1]["max"] is not None or faixas[0]["min"] != 10:
        raise RegrasInvalidas("faixas etarias devem ser contiguas, de 10 anos ate o limite superior aberto")


@cache
def carregar_regras(caminho: Path = CONFIG) -> Mapping:
    """Regras do JSON, imutaveis (o resultado e compartilhado entre chamadas)."""
    regras = json.loads(Path(caminho).read_text(encoding="utf-8"))
    colunas = regras["colunas_logistica"]
    declaradas = [c for var in regras["variaveis"] for c in var["colunas"]]
    if len(regras["variaveis"]) != 13 or len(colunas) != 28 or sorted(declaradas) != sorted(colunas):
        raise RegrasInvalidas("esperado 13 variaveis e 28 colunas coerentes com as variaveis declaradas")
    _conferir_cobertura(regras)
    return _congelar(regras)


COLUNAS_LOGISTICA: tuple[str, ...] = tuple(carregar_regras()["colunas_logistica"])


# --- validacao -----------------------------------------------------------------------------------
def _inteiro(valor, rotulo: str) -> int:
    if isinstance(valor, bool) or not isinstance(valor, int):
        raise ValueError(f"{rotulo} invalido: {valor!r}")
    return valor


def _no_dominio(valor, dominio: Sequence[str], rotulo: str) -> str:
    if valor not in dominio:
        raise ValueError(f"{rotulo} desconhecido: {valor!r}")
    return valor


def _dentro(idade: int, minimo: int, maximo: int | None) -> bool:
    return idade >= minimo and (maximo is None or idade <= maximo)


# --- regras por variavel -------------------------------------------------------------------------
def faixa_hora(hora) -> str:
    hora = _inteiro(hora, "hora")
    for nome, horas in carregar_regras()["faixas_hora"].items():
        if hora in horas:
            return nome
    raise ValueError(f"hora fora de 0..23: {hora}")


def sexta_a_domingo(dia_semana) -> int:
    dia = _inteiro(dia_semana, "dia da semana")
    if not 1 <= dia <= 7:
        raise ValueError(f"dia da semana fora de 1..7: {dia}")
    return int(dia in carregar_regras()["dias_sexta_a_domingo"])


def faixa_etaria(idade) -> str | None:
    """None para idade desconhecida ou abaixo de 10 anos (fora da populacao do consumo individual)."""
    if idade is None:
        return None
    idade = _inteiro(idade, "idade")
    if idade < 0:
        raise ValueError(f"idade negativa: {idade}")
    for faixa in carregar_regras()["faixas_etarias"]:
        if _dentro(idade, faixa["min"], faixa["max"]):
            return faixa["nome"]
    return None


def escolaridade_3(nivel) -> str:
    nivel = _inteiro(nivel, "nivel de instrucao")
    for nome, niveis in carregar_regras()["escolaridade_3"].items():
        if nivel in niveis:
            return nome
    raise ValueError(f"nivel de instrucao fora de 1..7: {nivel}")


def _idades_da_uc(idades: Sequence) -> list[int | None]:
    """Normaliza as idades dos moradores: None/NaN = desconhecida; fora de 0..111 interrompe."""
    saida: list[int | None] = []
    for idade in idades:
        if idade is None or (isinstance(idade, float) and math.isnan(idade)):
            saida.append(None)
            continue
        idade = _inteiro(idade, "idade de morador")
        if not 0 <= idade <= _IDADE_MAXIMA:
            raise ValueError(f"idade de morador fora de 0..{_IDADE_MAXIMA}: {idade}")
        saida.append(idade)
    return saida


def _presenca(idades: Sequence[int | None], minimo: int, maximo: int | None) -> int | None:
    """1 se algum morador conhecido esta na faixa; 0 se todos sao conhecidos e nenhum esta; None se ha idade desconhecida."""
    if any(i is not None and _dentro(i, minimo, maximo) for i in idades):
        return 1
    return None if any(i is None for i in idades) else 0


def _renda(renda_total, moradores: int, fator: float) -> tuple[float | None, str]:
    if renda_total is None or (isinstance(renda_total, float) and not math.isfinite(renda_total)):
        return None, "ausente"
    if renda_total == 0:
        return None, "zero"
    if renda_total < 0:
        return None, "negativa"
    return math.log(renda_total * fator / moradores), "valida"


# --- derivacao ------------------------------------------------------------------------------------
def derivar_nativas(edicao: str, registro: Mapping) -> dict:
    """Variaveis derivadas na forma nativa (baselines e boosting). `registro` traz hora, dia_semana, grande_regiao,
    area_comparavel, idade, sexo, escolaridade (1..7), renda_total (da UC) e idades_uc (todos os moradores da UC)."""
    regras = carregar_regras()
    if edicao not in regras["fator_renda"]:
        raise ValueError(f"edicao desconhecida: {edicao!r}")
    idades_uc = _idades_da_uc(registro["idades_uc"] or [])
    if not idades_uc:
        raise ValueError("unidade de consumo sem moradores")
    log_renda, status = _renda(registro["renda_total"], len(idades_uc), regras["fator_renda"][edicao])
    presencas = {nome: _presenca(idades_uc, f["min"], f["max"]) for nome, f in regras["presencas"].items()}
    return {
        "dia_semana": registro["dia_semana"], "faixa_hora": faixa_hora(registro["hora"]),
        "sexta_a_domingo": sexta_a_domingo(registro["dia_semana"]),
        "grande_regiao": _no_dominio(registro["grande_regiao"], regras["regioes"], "grande regiao"),
        "area": _no_dominio(registro["area_comparavel"], regras["areas"], "area"),
        "faixa_etaria": faixa_etaria(registro["idade"]), "idade": registro["idade"],
        "sexo": _no_dominio(registro["sexo"], regras["sexos"], "sexo"),
        "escolaridade": escolaridade_3(registro["escolaridade"]),
        "log_renda_pc": log_renda, "renda_status": status, "moradores_uc": len(idades_uc), **presencas,
    }


def _slug(texto: str) -> str:
    return texto.lower().replace("-", "_")


def codificar_logistica(nativas: Mapping) -> list:
    """As 28 colunas da logistica, na ordem de COLUNAS_LOGISTICA (um indicador por nivel; faixa etaria desconhecida
    deixa os cinco indicadores vazios)."""
    regras = carregar_regras()
    valores: dict = {f"hora_{h}": int(nativas["faixa_hora"] == h) for h in regras["faixas_hora"]}
    valores["sexta_a_domingo"] = nativas["sexta_a_domingo"]
    valores.update({f"regiao_{_slug(r)}": int(nativas["grande_regiao"] == r) for r in regras["regioes"]})
    valores.update({f"area_{a}": int(nativas["area"] == a) for a in regras["areas"]})
    valores.update({f"idade_{f['nome']}": None if nativas["faixa_etaria"] is None else int(nativas["faixa_etaria"] == f["nome"])
                    for f in regras["faixas_etarias"]})
    valores["sexo_feminino"] = int(nativas["sexo"] == "feminino")
    valores.update({f"escolaridade_{e}": int(nativas["escolaridade"] == e) for e in regras["escolaridade_3"]})
    valores["log_renda_pc"] = nativas["log_renda_pc"]
    valores["moradores_uc"] = nativas["moradores_uc"]
    valores["presenca_crianca_0_11"] = nativas["presenca_crianca"]
    valores["presenca_adolescente_12_17"] = nativas["presenca_adolescente"]
    valores["presenca_idoso_60_mais"] = nativas["presenca_idoso"]
    valores["sexta_a_domingo_x_noite"] = int(nativas["sexta_a_domingo"] == 1 and nativas["faixa_hora"] == "noite")
    return [valores[c] for c in COLUNAS_LOGISTICA]


# --- construcao sobre as tabelas DuckDB -----------------------------------------------------------
# edicao -> (ano, features comuns, episodios, moradores normalizados, coluna de idade, SQL do cod_upa nos moradores, tem peso)
_ORIGENS = {
    "2008-2009": ("2008", "features_comuns_2008", "e2_2008", "morador_2008_norm", "idade_anos",
                  "lpad(cod_uf::VARCHAR, 2, '0') || lpad(seq::VARCHAR, 3, '0') || dv::VARCHAR", True),
    "2017-2018": ("2017", "features_comuns_2017", "x_2017", "morador_2017_norm", "idade", "cod_upa", False),
}


def _consulta(edicao: str) -> str:
    _, feat, episodios, moradores, idade, upa, com_peso = _ORIGENS[edicao]
    peso = ", e.peso_final" if com_peso else ""
    return f"""
        WITH uc AS (SELECT {upa} AS cod_upa, num_dom, num_uc, list({idade} ORDER BY {idade}) AS idades
                    FROM {moradores} GROUP BY ALL)
        SELECT f.episodio_id, f.hora, f.dia_semana, f.grande_regiao, f.area_comparavel, f.idade, f.sexo, f.escolaridade,
               e.renda_total, e.cod_upa{peso}, u.idades
        FROM {feat} f JOIN {episodios} e USING (episodio_id)
        LEFT JOIN uc u ON u.cod_upa = e.cod_upa AND u.num_dom = e.num_dom AND u.num_uc = e.num_uc
        ORDER BY f.episodio_id"""


def _criar_tabela(con: duckdb.DuckDBPyConnection, nome: str, colunas: Sequence[tuple[str, str]],
                  linhas: list[tuple]) -> None:
    """Cria a tabela com a coluna `ordem` (posicao do episodio, por episodio_id) que fixa o alinhamento entre arquivos."""
    con.execute(f"CREATE OR REPLACE TABLE {nome}(ordem INT, {', '.join(f'{c} {t}' for c, t in colunas)})")
    con.executemany(f"INSERT INTO {nome} VALUES (?, {', '.join('?' * len(colunas))})",
                    [(i, *linha) for i, linha in enumerate(linhas)])


def _registro(linha: tuple) -> dict:
    campos = ("hora", "dia_semana", "grande_regiao", "area_comparavel", "idade", "sexo", "escolaridade", "renda_total")
    return {**dict(zip(campos, linha[1:9])), "idades_uc": linha[-1]}


def _derivar_do_episodio(edicao: str, linha: tuple) -> dict:
    try:
        return derivar_nativas(edicao, _registro(linha))
    except ValueError as erro:
        raise ValueError(f"episodio {linha[0]}: {erro}") from erro


def _resumo(nativas: list[dict], matriz: list[list]) -> dict:
    status = {s: sum(n["renda_status"] == s for n in nativas) for s in _STATUS_RENDA}
    desconhecidos = {c: sum(linha[i] is None for linha in matriz) for i, c in enumerate(COLUNAS_LOGISTICA)}
    return {"episodios": len(nativas), "desconhecidos_por_coluna": desconhecidos, "renda_status": status}


def _construir_edicao(con: duckdb.DuckDBPyConnection, edicao: str, juncoes: dict) -> dict:
    ano, feat, _, _, _, _, com_peso = _ORIGENS[edicao]
    antes = con.execute(f"SELECT count(*) FROM {feat}").fetchone()[0]
    linhas = con.execute(_consulta(edicao)).fetchall()
    registrar_juncao(juncoes, f"variaveis_{ano}", antes, len(linhas))
    sem_unidade = [l[0] for l in linhas if l[-1] is None]
    if sem_unidade:
        raise ValueError(f"{len(sem_unidade)} episodios de {edicao} sem unidade de consumo no cadastro de moradores")
    nativas = [_derivar_do_episodio(edicao, l) for l in linhas]
    matriz = [codificar_logistica(n) for n in nativas]
    _criar_tabela(con, f"variaveis_nativas_{ano}", _COLUNAS_NATIVAS, [tuple(n[c] for c, _ in _COLUNAS_NATIVAS) for n in nativas])
    _criar_tabela(con, f"matriz_logistica_{ano}",
                  [(c, "DOUBLE" if c in _COLUNAS_DOUBLE else "INT") for c in COLUNAS_LOGISTICA], [tuple(m) for m in matriz])
    colunas_ids = [("episodio_id", "VARCHAR"), ("edicao", "VARCHAR"), ("cod_upa", "VARCHAR")] + ([("peso_final", "DOUBLE")] if com_peso else [])
    _criar_tabela(con, f"identificadores_{ano}", colunas_ids,
                  [(l[0], edicao, l[9], *(l[10:11] if com_peso else ())) for l in linhas])
    return _resumo(nativas, matriz)


def construir_variaveis(con: duckdb.DuckDBPyConnection) -> dict:
    """Cria, por edicao, variaveis_nativas_*, matriz_logistica_* e identificadores_* (mesma ordem de linhas, por episodio_id).
    Exige features_comuns_*, e2_2008, x_2017 e os moradores normalizados. Devolve juncoes e resumo (sem microdados)."""
    juncoes: dict = {}
    unicas = {f"{origem[2]}.episodio_id": conferir_chave_unica(con, origem[2], ["episodio_id"]) for origem in _ORIGENS.values()}
    resumo = {edicao: _construir_edicao(con, edicao, juncoes) for edicao in _ORIGENS}
    juncoes["chaves_unicas"] = unicas
    return {"juncoes": juncoes, "variaveis": {"variaveis_derivadas": len(carregar_regras()["variaveis"]),
                                              "colunas_logistica": len(COLUNAS_LOGISTICA), **resumo}}
