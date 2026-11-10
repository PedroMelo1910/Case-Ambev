"""Leitura dos microdados: dicionario oficial (xls) -> layout -> tabelas DuckDB, e perfis de contagem.

Dicionario e leitor ficam juntos porque so fazem sentido em par: o layout vem do dicionario e o
leitor recusa qualquer linha que nao tenha o comprimento declarado nele.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import duckdb
import xlrd

MARCA_CABECALHO = "Posição Inicial"
AMOSTRA_BYTES = 65536


@dataclass(frozen=True)
class Variavel:
    posicao: int
    tamanho: int
    codigo: str
    descricao: str
    categorias: tuple[str, ...]


@dataclass(frozen=True)
class Layout:
    campos: dict[str, tuple[int, int]]  # codigo -> (posicao inicial base 1, tamanho)
    comprimento: int


class LayoutInvalido(ValueError):
    """Linha com comprimento diferente do dicionario ou coluna inexistente."""


# --- dicionario oficial ---------------------------------------------------------------------------
def _texto(valor) -> str:
    return str(int(valor)) if isinstance(valor, float) and valor.is_integer() else str(valor).strip()


def extrair_variaveis(linhas: list[list]) -> tuple[str, list[Variavel]]:
    """Converte as linhas de uma aba em variaveis; linhas sem posicao continuam a anterior."""
    registro = next((v.strip() for lin in linhas[:5] for v in lin if isinstance(v, str) and "REGISTRO" in v.upper()), "")
    inicio = next((i + 1 for i, lin in enumerate(linhas)
                   if any(isinstance(v, str) and v.strip().startswith(MARCA_CABECALHO) for v in lin)), None)
    if inicio is None:
        return registro, []
    brutas: list[dict] = []
    for linha in linhas[inicio:]:
        pos, tam, _dec, cod, desc, cat = ([_texto(v) for v in linha[:6]] + [""] * 6)[:6]
        if pos and cod:
            brutas.append({"posicao": int(float(pos)), "tamanho": int(float(tam)) if tam else 0, "codigo": cod,
                           "descricao": desc, "categorias": [cat] if cat else []})
        elif brutas:
            if cat:
                brutas[-1]["categorias"].append(cat)
            if desc:
                brutas[-1]["descricao"] += " " + desc
    return registro, [Variavel(b["posicao"], b["tamanho"], b["codigo"], b["descricao"], tuple(b["categorias"]))
                      for b in brutas]


def ler_dicionario_xls(caminho: Path) -> dict[str, list[Variavel]]:
    """Retorna {nome_da_aba_sem_espacos_nas_pontas: variaveis} do xls oficial."""
    abas = {}
    for folha in xlrd.open_workbook(str(caminho)).sheets():
        linhas = [[folha.cell_value(r, c) for c in range(folha.ncols)] for r in range(folha.nrows)]
        abas[folha.name.strip()] = extrair_variaveis(linhas)[1]
    return abas


def layout_de(variaveis: Iterable[Variavel]) -> Layout:
    campos = {v.codigo: (v.posicao, v.tamanho) for v in variaveis}
    return Layout(campos=campos, comprimento=max(p + t - 1 for p, t in campos.values()))


# --- leitor de largura fixa -----------------------------------------------------------------------
def _origem(arquivo: Path) -> str:
    with open(arquivo, "rb") as f:
        fim = "\\r\\n" if b"\r\n" in f.read(AMOSTRA_BYTES) else "\\n"
    # latin-1 mapeia 1 byte = 1 caractere, o que mantem as posicoes do dicionario exatas.
    return ("SELECT linha FROM read_csv(?, columns={'linha': 'VARCHAR'}, header=false, delim='\\x01', quote='', "
            f"escape='', auto_detect=false, encoding='latin-1', new_line='{fim}')")


def sql_colunas(arquivo: Path, layout: Layout, colunas: list[str]) -> str:
    """Subconsulta (com um `?` para o caminho) que fatia `colunas` do arquivo, sem materializar."""
    ausentes = [c for c in colunas if c not in layout.campos]
    if ausentes:
        raise LayoutInvalido(f"Colunas ausentes no dicionario: {ausentes}")
    expressoes = ", ".join(
        f"nullif(trim(substr(linha, {layout.campos[c][0]}, {layout.campos[c][1]})), '') AS \"{c}\"" for c in colunas)
    return f"SELECT {expressoes} FROM ({_origem(Path(arquivo))})"


def carregar_registro(con: duckdb.DuckDBPyConnection, tabela: str, arquivo: Path, layout: Layout,
                      colunas: list[str], where: str | None = None) -> int:
    """Cria `tabela` com `colunas` (texto, sem espacos nas pontas; vazio vira NULL).

    O comprimento de TODAS as linhas e conferido antes do filtro `where`. Retorna as linhas gravadas.
    """
    fatiada = sql_colunas(Path(arquivo), layout, colunas)
    fora = con.execute(f"SELECT count(*) FROM ({_origem(Path(arquivo))}) WHERE length(linha) <> {layout.comprimento}",
                       [str(arquivo)]).fetchone()[0]
    if fora:
        raise LayoutInvalido(f"{fora} linhas de {Path(arquivo).name} nao tem {layout.comprimento} posicoes")
    filtro = f" WHERE {where}" if where else ""
    con.execute(f"CREATE OR REPLACE TABLE {tabela} AS SELECT * FROM ({fatiada}){filtro}", [str(arquivo)])
    return con.execute(f"SELECT count(*) FROM {tabela}").fetchone()[0]


# --- perfis agregados (somente escalares; nenhum codigo de produto e consultado) -------------------
_PESSOA_17 = "COD_UPA, NUM_DOM, NUM_UC, COD_INFORMANTE"
_PESSOA_08 = "COD_UF, NUM_SEQ, NUM_DV, NUM_DOM, NUM_UC, COD_INFORMANTE"
_COLUNAS_17 = ["UF", "COD_UPA", "NUM_DOM", "NUM_UC", "COD_INFORMANTE", "QUADRO", "V9005", "V9007", "V9015", "PESO_FINAL"]


def perfil_consumo_2017(con: duckdb.DuckDBPyConnection, arquivo: Path, layout: Layout) -> dict:
    fatiada, p = sql_colunas(arquivo, layout, _COLUNAS_17), [str(arquivo)]
    adicao = "(V9005 IS NULL AND V9007 IS NULL)"
    linhas, adicoes, pessoas, domicilios, ucs, upas, pessoas_dia, e2 = con.execute(f"""
        WITH t AS ({fatiada})
        SELECT count(*), count(*) FILTER (WHERE {adicao}), count(DISTINCT ({_PESSOA_17})),
               count(DISTINCT (COD_UPA, NUM_DOM)), count(DISTINCT (COD_UPA, NUM_DOM, NUM_UC)), count(DISTINCT COD_UPA),
               count(DISTINCT ({_PESSOA_17}, QUADRO)),
               count(DISTINCT ({_PESSOA_17}, QUADRO, V9015)) FILTER (WHERE NOT {adicao})
        FROM t""", p).fetchone()
    soma_peso = con.execute(f"""
        WITH t AS ({fatiada}), x AS (SELECT min(CAST(PESO_FINAL AS DOUBLE)) w FROM t GROUP BY {_PESSOA_17})
        SELECT CAST(round(sum(w)) AS BIGINT) FROM x""", p).fetchone()[0]
    ufs = [r[0] for r in con.execute(f"WITH t AS ({fatiada}) SELECT DISTINCT UF FROM t ORDER BY 1", p).fetchall()]
    return {"linhas_consumo": linhas, "linhas_adicao": adicoes, "itens_alimento": linhas - adicoes,
            "pessoas_consumo": pessoas, "domicilios_consumo": domicilios, "ucs_consumo": ucs, "upas_consumo": upas,
            "pessoas_dia_consumo": pessoas_dia, "episodios_e2_total": e2, "soma_peso_consumo": soma_peso, "ufs": ufs}


def perfil_consumo_2008(con: duckdb.DuckDBPyConnection, tabela: str = "consumo_2008") -> dict:
    linhas, pessoas, domicilios, upas, pessoas_dia = con.execute(f"""
        SELECT count(*), count(DISTINCT ({_PESSOA_08})), count(DISTINCT (COD_UF, NUM_SEQ, NUM_DV, NUM_DOM)),
               count(DISTINCT (COD_UF, NUM_SEQ, NUM_DV)), count(DISTINCT ({_PESSOA_08}, NUM_QUADRO)) FROM {tabela}""").fetchone()
    soma_peso = con.execute(f"""
        WITH x AS (SELECT min(CAST(PESO_FINAL AS DOUBLE)) w FROM {tabela} GROUP BY {_PESSOA_08})
        SELECT CAST(round(sum(w)) AS BIGINT) FROM x""").fetchone()[0]
    ufs = [r[0] for r in con.execute(f"SELECT DISTINCT lpad(COD_UF, 2, '0') FROM {tabela} ORDER BY 1").fetchall()]
    return {"linhas_consumo": linhas, "pessoas_consumo": pessoas, "domicilios_consumo": domicilios,
            "upas_consumo": upas, "pessoas_dia_consumo": pessoas_dia, "soma_peso_consumo": soma_peso, "ufs": ufs}


def somas_de_peso_morador(con: duckdb.DuckDBPyConnection, tabela: str, coluna_idade: str) -> dict:
    total, dez_mais = con.execute(f"""
        SELECT CAST(round(sum(CAST(PESO_FINAL AS DOUBLE))) AS BIGINT),
               CAST(round(sum(CAST(PESO_FINAL AS DOUBLE)) FILTER (WHERE CAST({coluna_idade} AS INT) >= 10)) AS BIGINT)
        FROM {tabela}""").fetchone()
    return {"soma_peso_morador_total": total, "soma_peso_morador_10_mais": dez_mais}
