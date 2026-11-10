"""Verificacao do pacote: chaves edicao x UF, contagens de referencia e escrita deterministica das saidas.

As contagens de referencia vem de RELATORIO_AUDITORIA_POF.md e do parecer da fase 2 (secao 7). As de 2017
NAO incluem guarana nem refrigerante: 181 e 801 so podem ser conferidos na abertura unica (parecer, 7.1).
"""

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import duckdb

from occasion.fontes import EDICOES, sha256_arquivo

ARQUIVO_HASHES = "hashes.json"
UFS_BRASIL = (
    "11", "12", "13", "14", "15", "16", "17", "21", "22", "23", "24", "25", "26", "27", "28", "29",
    "31", "32", "33", "35", "41", "42", "43", "50", "51", "52", "53",
)


# --- chaves ---------------------------------------------------------------------------------------
def chaves_edicao_uf(ufs_2008: Iterable[str], ufs_2017: Iterable[str]) -> list[tuple[str, str]]:
    return sorted({(EDICOES[0], uf) for uf in ufs_2008} | {(EDICOES[1], uf) for uf in ufs_2017})


def chaves_completas(chaves: Iterable[tuple[str, str]]) -> bool:
    """True quando as chaves sao exatamente 2 edicoes x 27 UFs (54)."""
    return set(chaves) == {(e, uf) for e in EDICOES for uf in UFS_BRASIL}


def chave_domicilio(edicao: str, cod_upa: str | int, num_dom: str | int) -> str:
    """Chave de domicilio com prefixo da edicao (a numeracao de UPA/domicilio se repete entre as POF)."""
    return f"{edicao}|{cod_upa}|{int(num_dom):02d}"


def interseccao_chaves_domicilio(a: set[str], b: set[str]) -> set[str]:
    """Interseccao de CHAVES PREFIXADAS. Vazia por construcao entre edicoes: confere que o prefixo evita colisao
    de numeracao, mas NAO mede sobreposicao de domicilios reais (as POF nao trazem identificador comum)."""
    return a & b


# --- conferencia contra as referencias ------------------------------------------------------------
class ConferenciaFalhou(RuntimeError):
    """Alguma contagem difere da referencia; nada foi gravado."""


def conferir(obtidos: Mapping[str, object], referencias: Mapping[str, int]) -> list[dict]:
    return [{"nome": nome, "esperado": esperado, "obtido": obtidos.get(nome), "ok": obtidos.get(nome) == esperado}
            for nome, esperado in referencias.items()]


# --- saidas deterministicas -----------------------------------------------------------------------
def escrever_csv(con: duckdb.DuckDBPyConnection, tabela: str, destino: Path, ordenar_por: str) -> Path:
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY (SELECT * FROM {tabela} ORDER BY {ordenar_por}) TO ? (FORMAT csv, HEADER)", [str(destino)])
    return destino


def escrever_json(objeto, destino: Path) -> Path:
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(objeto, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destino


def hashes_das_saidas(saida: Path, selado: Path) -> dict[str, str]:
    """SHA-256 de cada arquivo de saida e do conjunto selado (exceto o proprio hashes.json)."""
    return {p.name: sha256_arquivo(p) for pasta in (Path(saida), Path(selado)) if pasta.exists()
            for p in sorted(pasta.iterdir()) if p.is_file() and p.name != ARQUIVO_HASHES}


# --- contagens de referencia ----------------------------------------------------------------------
REFERENCIAS: dict[str, int] = {
    # 2008-2009: registro de consumo (T_CONSUMO_S) e morador (relatorio, secoes 10)
    "2008-2009.linhas_consumo": 735814,
    "2008-2009.pessoas_consumo": 34003,
    "2008-2009.domicilios_consumo": 13569,
    "2008-2009.upas_consumo": 4328,
    "2008-2009.pessoas_dia_consumo": 66903,
    "2008-2009.soma_peso_consumo": 160511094,
    "2008-2009.soma_peso_morador_10_mais": 160511093,
    "2008-2009.episodios_e2_total": 287680,
    "2008-2009.episodios_com_pizza": 1168,
    "2008-2009.pessoas_com_pizza": 1106,
    "2008-2009.domicilios_com_pizza": 715,
    "2008-2009.com_guarana": 101,
    "2008-2009.com_algum_refrigerante": 655,
    # 2017-2018: registro de consumo (CONSUMO_ALIMENTAR) e morador (relatorio, secoes 2, 5 e 8)
    "2017-2018.linhas_consumo": 1175390,
    "2017-2018.linhas_adicao": 168264,
    "2017-2018.itens_alimento": 1007126,
    "2017-2018.pessoas_consumo": 46164,
    "2017-2018.domicilios_consumo": 20112,
    "2017-2018.ucs_consumo": 20146,
    "2017-2018.upas_consumo": 5404,
    "2017-2018.pessoas_dia_consumo": 85018,
    "2017-2018.soma_peso_consumo": 179899000,
    "2017-2018.soma_peso_morador_total": 207103790,
    "2017-2018.soma_peso_morador_10_mais": 179899000,
    "2017-2018.episodios_e2_total": 459847,
    "2017-2018.episodios_com_pizza": 1703,
    "2017-2018.pessoas_com_pizza": 1620,
    "2017-2018.domicilios_com_pizza": 1004,
    # integracao
    "chaves_edicao_uf": 54,
    "chaves_domicilio_prefixadas_em_comum": 0,
}
