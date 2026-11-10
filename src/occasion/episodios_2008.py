"""Episodio E2 (pessoa x dia de coleta x hora) e alvo estrito da POF 2008-2009.

Entrada (tabelas DuckDB de texto, criadas pelo leitor): consumo_2008, morador_2008, domicilio_2008.
Saida: tabela e2_2008 com os episodios que contem pizza e os rotulos y_guarana e y_refrigerante.
Logica reaproveitada de auditoria_pof/scripts/pof2008_comparavel.py (definicao E2).
"""

import duckdb

from occasion.crosswalk import conferir_chave_unica, registrar_juncao, sql_hora
from occasion.episodios_2017 import pizza

EDICAO = "2008-2009"
# ATENCAO (portao do alvo de 2017, parecer da fase 2, criterio 9): as listas de bebidas ficam NESTE modulo,
# que o caminho de desenvolvimento de 2017 nunca importa. Fonte: auditoria_pof/scripts/listas_codigos.py.
GUARANA = ("8200301", "8200302", "8201601", "8201602", "8207401", "8207402")
REFRIGERANTE = (
    "8200101", "8200102", "8200202", "8200208", "8200211", "8200301", "8200302", "8200407", "8200411", "8200505",
    "8200507", "8201017", "8201802", "8203501", "8200701", "8200702", "8200801", "8201511", "8201601", "8201602",
    "8207401", "8207402", "8207501", "8207601", "8207701", "8201502", "8201902", "8205402", "8209102", "8207705",
)
_PESSOA = "cod_uf, seq, dv, num_dom, num_uc, cod_inf"


def _lista(codigos) -> str:
    return "(" + ",".join(f"'{c}'" for c in codigos) + ")"


def construir_e2_2008(con: duckdb.DuckDBPyConnection) -> dict:
    """Cria e2_2008 e devolve as contagens (E2 total, com pizza, com guarana, com algum refrigerante)."""
    con.execute(f"""
        CREATE OR REPLACE TABLE item_2008 AS
        SELECT CAST(COD_UF AS INT) cod_uf, CAST(NUM_SEQ AS INT) seq, CAST(NUM_DV AS INT) dv, CAST(NUM_DOM AS INT) num_dom,
               CAST(NUM_UC AS INT) num_uc, CAST(COD_INFORMANTE AS INT) cod_inf, CAST(NUM_QUADRO AS INT) quadro,
               {sql_hora(EDICAO, 'V9015')} hora, V9018, ESTRATO_POF, DIA_SEMANA, CAST(PESO_FINAL AS DOUBLE) peso,
               CAST(RENDA_TOTAL AS DOUBLE) renda,
               (V9001 IN {_lista(pizza(EDICAO))})::INT p, (V9001 IN {_lista(GUARANA)})::INT g,
               (V9001 IN {_lista(REFRIGERANTE)})::INT r
        FROM consumo_2008""")
    con.execute(f"""
        CREATE OR REPLACE TABLE ep_2008 AS
        SELECT {_PESSOA}, quadro, hora, max(p) p, max(g) g, max(r) r, min(ESTRATO_POF) estrato_pof, min(DIA_SEMANA) dia_semana,
               min(peso) peso, min(renda) renda, max((V9018 = '1')::INT) fonte_dentro, max((V9018 = '2')::INT) fonte_fora
        FROM item_2008 GROUP BY ALL""")
    total = con.execute("SELECT count(*) FROM ep_2008").fetchone()[0]
    con.execute("""
        CREATE OR REPLACE TABLE morador_2008_norm AS
        SELECT CAST(COD_UF AS INT) cod_uf, CAST(NUM_SEQ AS INT) seq, CAST(NUM_DV AS INT) dv, CAST(NUM_DOM AS INT) num_dom,
               CAST(NUM_UC AS INT) num_uc, CAST(COD_INFORMANTE AS INT) cod_inf, CAST(IDADE_ANOS AS INT) idade_anos,
               V0405 sexo, TIPO_SITUACAO_REG tipo_situacao, NIVEL_INSTRUCAO_MORADOR nivel_instrucao
        FROM morador_2008""")
    con.execute("""
        CREATE OR REPLACE TABLE domicilio_2008_norm AS
        SELECT CAST(COD_UF AS INT) cod_uf, CAST(NUM_SEQ AS INT) seq, CAST(NUM_DV AS INT) dv, CAST(NUM_DOM AS INT) num_dom,
               CAST(PERIODO_REAL AS INT) periodo_real
        FROM domicilio_2008""")
    juncoes = {"chaves_unicas": {
        "morador_2008": conferir_chave_unica(con, "morador_2008_norm", _PESSOA.split(", ")),
        "domicilio_2008": conferir_chave_unica(con, "domicilio_2008_norm", ["cod_uf", "seq", "dv", "num_dom"])}}
    antes = con.execute("SELECT count(*) FROM ep_2008 WHERE p = 1").fetchone()[0]
    con.execute("""
        CREATE OR REPLACE TABLE ep_2008_morador AS
        SELECT e.*, m.idade_anos, m.sexo, m.tipo_situacao, m.nivel_instrucao
        FROM ep_2008 e LEFT JOIN morador_2008_norm m USING (cod_uf, seq, dv, num_dom, num_uc, cod_inf) WHERE e.p = 1""")
    registrar_juncao(juncoes, "episodios_com_morador_2008", antes, con.execute("SELECT count(*) FROM ep_2008_morador").fetchone()[0])
    con.execute(f"""
        CREATE OR REPLACE TABLE e2_2008 AS
        SELECT format('{EDICAO}|{{}}{{}}{{}}|{{}}|{{}}|{{}}|{{}}|{{}}', lpad(e.cod_uf::VARCHAR, 2, '0'), lpad(e.seq::VARCHAR, 3, '0'),
                      e.dv, lpad(e.num_dom::VARCHAR, 2, '0'), e.num_uc, lpad(e.cod_inf::VARCHAR, 2, '0'), e.quadro,
                      lpad(e.hora::VARCHAR, 2, '0')) episodio_id,
               '{EDICAO}' edicao, lpad(e.cod_uf::VARCHAR, 2, '0') uf,
               lpad(e.cod_uf::VARCHAR, 2, '0') || lpad(e.seq::VARCHAR, 3, '0') || e.dv::VARCHAR cod_upa,
               e.num_dom, e.num_uc, e.cod_inf cod_informante, e.quadro, e.hora, e.estrato_pof, e.tipo_situacao,
               d.periodo_real, e.dia_semana, e.idade_anos, e.sexo, e.renda renda_total, e.peso peso_final,
               e.fonte_dentro, e.fonte_fora, e.nivel_instrucao, e.g y_guarana, e.r y_refrigerante
        FROM ep_2008_morador e LEFT JOIN domicilio_2008_norm d USING (cod_uf, seq, dv, num_dom)""")
    registrar_juncao(juncoes, "episodios_com_domicilio_2008", antes, con.execute("SELECT count(*) FROM e2_2008").fetchone()[0])
    sem = con.execute("SELECT count(*) FROM e2_2008 WHERE idade_anos IS NULL OR periodo_real IS NULL").fetchone()[0]
    if sem:
        raise ValueError(f"{sem} episodios de 2008-2009 sem correspondencia em MORADOR ou DOMICILIO")
    return {
        "episodios_e2_total": total,
        "episodios_com_pizza": con.execute("SELECT count(*) FROM e2_2008").fetchone()[0],
        "com_guarana": con.execute("SELECT sum(y_guarana) FROM e2_2008").fetchone()[0],
        "com_algum_refrigerante": con.execute("SELECT sum(y_refrigerante) FROM e2_2008").fetchone()[0],
        "juncoes": juncoes,
    }
