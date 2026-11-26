"""Preparo das escolhas e auditoria das refeicoes (src/preparo.py) com dados sinteticos: chave unica, grafo de
sabores, classificacao de cada registro bruto, episodios, alvo e sensibilidade. Nenhum teste faz rede nem abre microdados."""

import json

import duckdb
import pytest

import preparo

# ---------- escolhas: chave e grafo ----------

def test_chave_repetida_interrompe_a_gravacao(tmp_path):
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT * FROM (VALUES ('a'), ('a')) v(chave)")
    with pytest.raises(preparo.EntradaInvalida, match="repetida"):
        preparo._gravar_parquet(con, "t", tmp_path / "t.parquet", {"chave": ["chave"], "descricoes": {}})
    assert not (tmp_path / "t.parquet").exists()


def test_escolhas_usam_a_prioridade_de_sabor_e_a_particao():
    # a prioridade (com o sabor desconhecido separado) e testada em test_escolhas_pof_v2.py
    sql = preparo._SQL_ESCOLHAS
    assert "{sabor} sabor" in sql and "pt.particao" in sql and "mapa_particao" in sql


# ---------- auditoria das refeicoes: dos registros brutos ate a tabela do alvo ----------

def _bruto_2017(con, linhas):
    colunas = preparo.COLUNAS_AUDITORIA["2017-2018"]
    con.execute(f"CREATE TABLE bruto ({', '.join(f'{c} VARCHAR' for c in colunas)})")
    con.executemany(f"INSERT INTO bruto VALUES ({', '.join('?' * len(colunas))})",
                    [tuple(linha.get(c) for c in colunas) for linha in linhas])


def _registro(seq, codigo, hora="20H", v9005="1", v9007="1", upa="350000001"):
    return {"UF": "35", "ESTRATO_POF": "3501", "COD_UPA": upa, "NUM_DOM": "1", "NUM_UC": "1", "COD_INFORMANTE": "1",
            "QUADRO": "72", "SEQ": str(seq), "V9005": v9005, "V9007": v9007, "V9001": codigo, "V9015": hora,
            "V9017": "4", "V9018": "1", "PESO_FINAL": "100.0", "DIA_SEMANA": "SEXTA-FEIRA", "DIA_ATIPICO": "2"}


def test_auditoria_classifica_cada_registro_bruto_uma_unica_vez():
    con = duckdb.connect()
    _bruto_2017(con, [
        _registro(1, preparo.PIZZA_ESTRITA[0]),             # comida (pizza)
        _registro(2, preparo.GUARANA[0]),                   # bebida: guarana
        _registro(3, preparo.SABOR_DESCONHECIDO_POF[0]),    # bebida: refrigerante de sabor nao especificado
        _registro(4, preparo.AGUA_2017[0]),                 # agua pura
        _registro(5, "1234567", v9005=None, v9007=None),    # linha de adicao (2017)
        _registro(6, "7654321"),                            # codigo fora do cadastro
        _registro(7, None),                                 # sem codigo
    ])
    cadastro = {preparo.PIZZA_ESTRITA[0], preparo.GUARANA[0], preparo.SABOR_DESCONHECIDO_POF[0], preparo.AGUA_2017[0],
                "1234567"}
    nao = [preparo.GUARANA[0], preparo.SABOR_DESCONHECIDO_POF[0], preparo.AGUA_2017[0]]
    linhas = con.execute(f"SELECT codigo, classe, pizza, guarana, refri_ne, hora FROM "
                         f"({preparo.sql_linhas_auditoria('2017-2018', cadastro, nao)}) ORDER BY chave_bruta").fetchall()
    classes = {codigo: classe for codigo, classe, *_ in linhas}
    assert classes == {preparo.PIZZA_ESTRITA[0]: "comida", preparo.GUARANA[0]: "outra_bebida_ou_nao_comida",
                       preparo.SABOR_DESCONHECIDO_POF[0]: "outra_bebida_ou_nao_comida", preparo.AGUA_2017[0]: "agua",
                       "1234567": "adicao", "7654321": "fora_do_cadastro", None: "sem_codigo"}
    marcas = {codigo: (p, g, ne) for codigo, _, p, g, ne, _ in linhas}
    assert marcas[preparo.PIZZA_ESTRITA[0]] == (1, 0, 0) and marcas[preparo.GUARANA[0]] == (0, 1, 0)
    assert marcas[preparo.SABOR_DESCONHECIDO_POF[0]] == (0, 0, 1)
    assert {hora for *_, hora in linhas} == {20}


def _linhas_auditadas(con, refeicoes):
    """Tabela `linhas` minima: cada refeicao e uma lista de (classe, pizza, guarana, refri_ne) na mesma pessoa e hora."""
    con.execute("""CREATE TABLE linhas (upa VARCHAR, dom INT, uc INT, inf INT, quadro INT, hora INT, uf VARCHAR,
        estrato INT, peso DOUBLE, dia_bruto VARCHAR, classe VARCHAR, pizza INT, guarana INT, cola INT, outro_refri INT,
        refri_ne INT)""")
    for dom, itens in enumerate(refeicoes, 1):
        for classe, pizza, guarana, refri_ne in itens:
            con.execute("INSERT INTO linhas VALUES ('350000001', ?, 1, 1, 72, 20, '35', 3501, 2.0, '5', ?, ?, ?, 0, 0, ?)",
                        [dom, classe, pizza, guarana, refri_ne])


def test_auditoria_monta_episodios_e_separa_alvo_inconclusivo_e_sensibilidade():
    con = duckdb.connect()
    _linhas_auditadas(con, [
        [("comida", 1, 0, 0), ("outra_bebida_ou_nao_comida", 0, 1, 0)],  # pizza com guarana
        [("comida", 1, 0, 0), ("outra_bebida_ou_nao_comida", 0, 0, 1)],  # pizza + refrigerante nao especificado
        [("comida", 1, 0, 0)],                                           # pizza sem guarana
        [("comida", 0, 0, 0)],                                           # refeicao sem pizza
        [("agua", 0, 0, 0)],                                             # so agua: fora das refeicoes com comida
    ])
    preparo.episodios_auditoria(con)
    tipos = dict(con.execute("SELECT tipo, count(*) FROM ep GROUP BY 1").fetchall())
    assert tipos == {"com_comida": 4, "so_agua": 1}
    alvo = preparo.tabela_alvo(con, "TRUE")
    assert alvo["denominador_refeicoes_com_comida"] == 4
    assert {c: v["n"] for c, v in alvo["linhas"].items()} == {
        "pizza_com_guarana": 1, "pizza_inconclusivo_refri_nao_especificado": 1, "pizza_sem_guarana": 1, "sem_pizza": 1}
    s = preparo.sensibilidade_do_alvo(con, "TRUE")
    assert (s["positivos_estritos"], s["inconclusivos"]) == (1, 1)
    assert s["taxa_estrita_pct"] == pytest.approx(25.0) and s["taxa_limite_superior_pct"] == pytest.approx(50.0)
    assert s["taxa_excluindo_inconclusivos_pct"] == pytest.approx(100 / 3)


def test_comando_auditoria_grava_um_json_por_edicao_na_pasta_processada(tmp_path, monkeypatch):
    monkeypatch.setenv("CASE_AMBEV_RAW_ROOT", str(tmp_path / "bruto"))
    monkeypatch.setenv("CASE_AMBEV_PROCESSADO_ROOT", str(tmp_path / "processado"))
    chamadas = []
    monkeypatch.setattr(preparo, "auditar_refeicoes",
                        lambda edicao, raiz, temp=None: chamadas.append((edicao, temp)) or {"edicao": edicao})
    assert preparo.main(["auditoria", "--temp", str(tmp_path / "temp")]) == 0
    assert [e for e, _ in chamadas] == list(preparo.EDICOES_AUDITORIA)
    for edicao in preparo.EDICOES_AUDITORIA:
        gravado = json.loads((tmp_path / "processado" / f"auditoria_{edicao[:4]}.json").read_text(encoding="utf-8"))
        assert gravado == {"edicao": edicao}
    assert not (tmp_path / "processado" / "manifesto_tabelas.json").exists()  # a auditoria nao e tabela do pipeline


# ---------- protocolo: ligado ao codigo ----------

def _protocolo_v3(repo):
    return json.loads((repo / "configs" / "protocolo_v3_alvo_observado.json").read_text(encoding="utf-8"))


def test_protocolo_v3_usa_os_codigos_de_guarana_do_preparo(repo):
    positivo = _protocolo_v3(repo)["alvo"]["positivo"]
    assert all(codigo in positivo for codigo in preparo.GUARANA)
    assert preparo.SABOR_DESCONHECIDO_POF[0] in _protocolo_v3(repo)["alvo"]["inconclusivo"]


def test_protocolo_v3_lista_as_familias_que_a_modelagem_compara(repo):
    import modelagem
    import ocasiao_modelos
    familias = [c["id"] for c in _protocolo_v3(repo)["candidatos"]]
    assert familias == list(modelagem.GRADES) == list(ocasiao_modelos.ORDEM_DE_SIMPLICIDADE)


def test_blocos_da_modelagem_nao_usam_itens_nem_informacao_posterior():
    import ocasiao_modelos
    usados = {campo for bloco in ocasiao_modelos.BLOCOS.values() for campo in bloco}
    assert not usados & {"pizza", "guarana", "refrigerante", "sabor", "cola", "local_v9018", "ocasiao_v9017", "dia_atipico"}
