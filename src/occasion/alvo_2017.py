"""Alvo de 2017-2018, construido por codigo e so pela avaliacao temporal, depois do marco de exposicao.

Guarana (e refrigerante) no MESMO episodio elegivel com pizza. A unidade (pessoa x dia x hora), a elegibilidade, a
exclusao das linhas de adicao e o `episodio_id` vem do construtor existente (`episodios_2017.construir_e2_pizza_2017`),
chamado aqui sem alteracao; este modulo so acrescenta as marcas de bebida pela mesma chave do episodio.

As listas de bebidas repetem as de `episodios_2008` (igualdade conferida por teste): importa-las faria o caminho de 2017
alcancar o modulo das bebidas. Nenhum modulo de treinamento ou de preparacao importa este arquivo.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import duckdb

from occasion.crosswalk import sql_hora
from occasion.episodios_2017 import EDICAO, construir_e2_pizza_2017, pizza
from occasion.fontes import Fonte, Manifesto, extrair_membros
from occasion.microdados import Layout, carregar_registro, layout_de, ler_dicionario_xls

GUARANA = ("8200301", "8200302", "8201601", "8201602", "8207401", "8207402")
REFRIGERANTE = (
    "8200101", "8200102", "8200202", "8200208", "8200211", "8200301", "8200302", "8200407", "8200411", "8200505",
    "8200507", "8201017", "8201802", "8203501", "8200701", "8200702", "8200801", "8201511", "8201601", "8201602",
    "8207401", "8207402", "8207501", "8207601", "8207701", "8201502", "8201902", "8205402", "8209102", "8207705",
)
# Colunas, arquivos e abas iguais aos de `pipeline_pof` (conferido por teste), que nao pode ser importado daqui.
COLUNAS_CONSUMO = ["UF", "ESTRATO_POF", "TIPO_SITUACAO_REG", "COD_UPA", "NUM_DOM", "NUM_UC", "COD_INFORMANTE", "QUADRO",
                   "SEQ", "V9005", "V9007", "V9001", "V9015", "V9017", "V9018", "PESO_FINAL", "RENDA_TOTAL",
                   "DIA_SEMANA", "DIA_ATIPICO"]
COLUNAS_MORADOR = ["UF", "COD_UPA", "NUM_DOM", "NUM_UC", "COD_INFORMANTE", "V0306", "V0403", "V0404", "PESO_FINAL", "NIVEL_INSTRUCAO"]
ARQ_DICIONARIO = "Dicionarios_de_variaveis.xls"
ARQ_CONSUMO, ABA_CONSUMO = "CONSUMO_ALIMENTAR.txt", "Consumo Alimentar"
ARQ_MORADOR, ABA_MORADOR = "MORADOR.txt", "Morador"
MEMORIA_DUCKDB = "1500MB"
_EPISODIO = "cod_upa, num_dom, num_uc, cod_informante, quadro, hora"  # chave do episodio em x_2017


class AlvoInvalido(ValueError):
    """Fontes, dicionario ou resultado fora do esperado para construir o alvo."""


@dataclass(frozen=True)
class FontesDoAlvo:
    """O que a preparacao deixa pronto, sem ler consumo: onde estao os dados e como fatiar os dois registros."""
    raiz: Path
    dados: Fonte
    layout_consumo: Layout
    layout_morador: Layout


def _lista(codigos) -> str:
    return "(" + ",".join(f"'{c}'" for c in codigos) + ")"


def construir_alvo(con: duckdb.DuckDBPyConnection) -> int:
    """De `consumo_2017_itens` (linhas de pizza e de refrigerante) e `morador_2017` cria `alvo_2017` com um episodio
    elegivel por linha: episodio_id, y_guarana, y_refrigerante. Devolve o numero de episodios."""
    con.execute(f"CREATE OR REPLACE TABLE consumo_2017_pizza AS "
                f"SELECT * FROM consumo_2017_itens WHERE V9001 IN {_lista(pizza(EDICAO))}")
    episodios = construir_e2_pizza_2017(con)["episodios_com_pizza"]  # unidade, filtros e episodio_id ja estabelecidos
    con.execute(f"""
        CREATE OR REPLACE TABLE bebida_2017 AS
        SELECT COD_UPA cod_upa, CAST(NUM_DOM AS INT) num_dom, CAST(NUM_UC AS INT) num_uc,
               CAST(COD_INFORMANTE AS INT) cod_informante, CAST(QUADRO AS INT) quadro, {sql_hora(EDICAO, 'V9015')} hora,
               max((V9001 IN {_lista(GUARANA)})::INT) y_guarana, max(1) y_refrigerante
        FROM consumo_2017_itens
        WHERE V9001 IN {_lista(REFRIGERANTE)} AND NOT (V9005 IS NULL AND V9007 IS NULL) GROUP BY ALL""")
    con.execute(f"""
        CREATE OR REPLACE TABLE alvo_2017 AS
        SELECT x.episodio_id, coalesce(b.y_guarana, 0) y_guarana, coalesce(b.y_refrigerante, 0) y_refrigerante
        FROM x_2017 x LEFT JOIN bebida_2017 b USING ({_EPISODIO})""")
    linhas, distintos = con.execute("SELECT count(*), count(DISTINCT episodio_id) FROM alvo_2017").fetchone()
    if not linhas == distintos == episodios:
        raise AlvoInvalido(f"alvo com {linhas} linhas e {distintos} episodios distintos; esperado {episodios}")
    return episodios


def _fonte_com(manifesto: Manifesto, arquivo: str) -> Fonte:
    achadas = [f for f in manifesto.fontes
               if f.edicao == EDICAO and any(Path(membro.destino_relativo).name == arquivo for membro in f.membros)]
    if len(achadas) != 1:
        raise AlvoInvalido(f"esperada uma fonte de {EDICAO} com {arquivo} no manifesto de fontes; ha {len(achadas)}")
    return achadas[0]


def _extraido(caminhos: list[Path], nome: str) -> Path:
    return next(caminho for caminho in caminhos if caminho.name == nome)


def _layout(abas: dict, aba: str, colunas: list[str]) -> Layout:
    variaveis = abas.get(aba) or []
    campos = {v.codigo for v in variaveis}
    ausentes = [c for c in colunas if c not in campos]
    if ausentes:
        raise AlvoInvalido(f"dicionario oficial, aba {aba!r}: coluna ausente {ausentes}")
    return layout_de(variaveis)


def preparar_fontes(raiz: Path, manifesto: Manifesto) -> FontesDoAlvo:
    """ANTES da exposicao: layouts pelo dicionario oficial (extraido e conferido por hash) e presenca do arquivo de
    dados. Nada de consumo ou de morador e extraido nem lido aqui."""
    raiz = Path(raiz)
    documentos = extrair_membros(_fonte_com(manifesto, ARQ_DICIONARIO), raiz)
    abas = ler_dicionario_xls(_extraido(documentos, ARQ_DICIONARIO))
    dados = _fonte_com(manifesto, ARQ_CONSUMO)
    if not (raiz / dados.caminho_relativo).is_file():
        raise FileNotFoundError(f"arquivo de dados de {EDICAO} ausente: {dados.caminho_relativo}")
    return FontesDoAlvo(raiz=raiz, dados=dados, layout_consumo=_layout(abas, ABA_CONSUMO, COLUNAS_CONSUMO),
                        layout_morador=_layout(abas, ABA_MORADOR, COLUNAS_MORADOR))


def alvo_das_fontes(fontes: FontesDoAlvo) -> dict[str, tuple[int, int]]:
    """DEPOIS do marco de exposicao: extrai consumo e morador (conferidos por hash), constroi o alvo e devolve
    {episodio_id: (y_guarana, y_refrigerante)}. Nada e gravado em disco."""
    arquivos = extrair_membros(fontes.dados, fontes.raiz)
    itens = _lista(sorted(set(pizza(EDICAO)) | set(REFRIGERANTE)))
    with tempfile.TemporaryDirectory() as tmp:
        con = duckdb.connect()
        try:
            con.execute(f"SET memory_limit='{MEMORIA_DUCKDB}'")
            con.execute("SET temp_directory=?", [tmp])
            carregar_registro(con, "consumo_2017_itens", _extraido(arquivos, ARQ_CONSUMO), fontes.layout_consumo,
                              COLUNAS_CONSUMO, where=f"V9001 IN {itens}")
            carregar_registro(con, "morador_2017", _extraido(arquivos, ARQ_MORADOR), fontes.layout_morador, COLUNAS_MORADOR)
            construir_alvo(con)
            linhas = con.execute("SELECT episodio_id, y_guarana, y_refrigerante FROM alvo_2017").fetchall()
        finally:
            con.close()
    return {episodio: (int(guarana), int(refrigerante)) for episodio, guarana, refrigerante in linhas}
