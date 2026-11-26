"""Monta tabelas DuckDB a partir dos dados sinteticos usando o leitor de largura fixa."""

import duckdb

from sinteticos_pof import COLUNAS, escrever_largura_fixa, layout_sintetico
from coleta import Layout, carregar_registro


def carregar(con, tmp_path, tabela, linhas, nome=None, where=None):
    campos, comp = layout_sintetico(tabela)
    arq = escrever_largura_fixa(tmp_path / f"{nome or tabela}.txt", tabela, linhas)
    cols = [c for c, _ in COLUNAS[tabela]]
    carregar_registro(con, nome or tabela, arq, Layout(campos=campos, comprimento=comp), cols, where=where)


def con_2008(tmp_path, consumo, morador, domicilio):
    con = duckdb.connect()
    carregar(con, tmp_path, "consumo_2008", consumo)
    carregar(con, tmp_path, "morador_2008", morador)
    carregar(con, tmp_path, "domicilio_2008", domicilio)
    return con


def con_2017(tmp_path, consumo, morador, codigos_pizza):
    con = duckdb.connect()
    lista = ",".join(f"'{c}'" for c in codigos_pizza)
    carregar(con, tmp_path, "consumo_2017", consumo, nome="consumo_2017_pizza", where=f"V9001 IN ({lista})")
    carregar(con, tmp_path, "morador_2017", morador)
    return con
