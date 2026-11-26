"""Decision Engine, expansao do contexto e rota /api/avaliar do Occasion Score observado.

Os testes da rota usam o artefato publicado (artifacts/ocasiao_observada_v1, gerado por `python -m modelagem testar`)
e sao pulados se ele nao existir.
"""

import http.client
import json
import threading
import types

import numpy as np
import pytest

import modelagem
import ocasiao_modelos as om
import preparo
from aplicacao import regras, servidor

COMPONENTES = {"pizza": 0.02, "guarana_dado_pizza": 0.05, "pizza_referencia": 0.005,
               "guarana_dado_pizza_referencia": 0.11, "guarana_sustentado": True}
CONTEXTO = {"dia_semana": 6, "periodo": "noite", "grande_regiao": "Sudeste", "area": "capital",
            "faixa_etaria": "todos", "sexo": "todos", "escolaridade": "todos"}


def _comparacao(faixa: str, nota: str | None = None) -> dict:
    return {"faixa": faixa, "estimativa": 0.0023, "referencia": 0.00063, "intervalo_pp": [0.05, 0.29], "nota": nota}


TERMOS_SEM_EVIDENCIA = ("em casa", "awareness", "marca", "doméstic")


def _sem_termos(texto: str) -> bool:
    return not any(t in texto.lower() for t in TERMOS_SEM_EVIDENCIA)


def test_ativacao_acima_vira_teste_de_combo_no_contexto_com_o_motivo_do_proprio_contexto():
    d = regras.decidir_ativacao(_comparacao("acima_do_habitual"), {**COMPONENTES, "guarana_sustentado": False})
    assert (d["regra"], d["categoria"], d["versao"]) == ("A2", "testar_combo_no_contexto", "ativacao_v3")
    assert "0,23%" in d["justificativa"] and "3,7× a média" in d["justificativa"]
    assert "+0,050 p.p." in d["justificativa"] and "+0,290 p.p." in d["justificativa"]
    assert _sem_termos(d["justificativa"])


def test_ativacao_nunca_sugere_awareness_mesmo_com_guarana_abaixo_da_media_entre_as_pizzas():
    d = regras.decidir_ativacao(_comparacao("acima_do_habitual"), COMPONENTES)
    assert (d["regra"], d["categoria"]) == ("A2", "testar_combo_no_contexto")
    assert "construir_associacao" not in regras.CATEGORIAS and "combo_consumo_em_casa" not in regras.CATEGORIAS


def test_rotulos_e_proximos_passos_nao_afirmam_canal_domestico_nem_marca():
    assert regras.CATEGORIAS["testar_combo_no_contexto"] == "Testar um combo de pizza com guaraná neste contexto"
    assert set(regras.PROXIMOS_PASSOS) == set(regras.CATEGORIAS)
    assert all(_sem_termos(t) for t in [*regras.CATEGORIAS.values(), *regras.PROXIMOS_PASSOS.values()])
    assert "grupo de controle" in regras.PROXIMOS_PASSOS["testar_combo_no_contexto"]


def test_tela_nao_descreve_regra_de_consumo_em_casa_nem_awareness(repo):
    pagina = (repo / "src" / "aplicacao" / "web" / "index.html").read_text(encoding="utf-8").lower()
    assert "consumo em casa" not in pagina and "awareness" not in pagina


@pytest.mark.parametrize("faixa, categoria", [("sem_diferenca_demonstrada", "sem_destaque"),
                                              ("abaixo_do_habitual", "baixa_prioridade"),
                                              ("evidencia_insuficiente", "nao_priorizar")])
def test_ativacao_sem_prioridade(faixa, categoria):
    d = regras.decidir_ativacao(_comparacao(faixa), COMPONENTES)
    assert d["categoria"] == categoria and _sem_termos(d["justificativa"])


def test_pouca_evidencia_orienta_a_usar_a_media_e_cita_o_motivo():
    d = regras.decidir_ativacao(_comparacao("evidencia_insuficiente", "4 refeições no treino"), None)
    assert d["regra"] == "A1" and "4 refeições no treino" in d["justificativa"]
    assert "média" in d["justificativa"]


def test_motivo_de_pouca_amostra_usa_milhar_e_diz_a_que_se_refere_cada_minimo():
    """Aceite visual de 25/11/2026: a tela mostrava '10425 refeições' e 'o mínimo para comparar é 30 e 3'."""
    motivos = dict(regras._motivos({"episodios": 10425, "positivos": 0}, ["segunda-feira"], True, True, 30, 3))
    texto = motivos["suporte_insuficiente"]
    assert "10.425 refeições" in texto
    assert "30 refeições e 3 com pizza e guaraná" in texto


def test_ativacao_recusa_faixa_desconhecida():
    with pytest.raises(preparo.EntradaInvalida):
        regras.decidir_ativacao(_comparacao("outra"), COMPONENTES)


def _resumo_minimo():
    return {"horas_por_periodo": {"noite": {"19": 3, "20": 1}},
            "composicao_publico": {"Sudeste|capital": {"18_29|feminino|medio": 3, "30_44|masculino|superior": 1}}}


def test_expandir_periodo_e_publico_geral_somam_um():
    dados, pesos = modelagem.expandir(preparo.preparar(CONTEXTO), _resumo_minimo())
    assert len(pesos) == 4 and pesos.sum() == pytest.approx(1.0)
    assert set(dados["hora"]) == {19, 20}
    assert pesos[(dados["hora"] == 19) & (dados["faixa_etaria"] == "18_29")][0] == pytest.approx(0.75 * 0.75)


def test_expandir_hora_informada_usa_so_essa_hora():
    contexto = {k: v for k, v in CONTEXTO.items() if k != "periodo"} | {"hora": 20}
    dados, pesos = modelagem.expandir(preparo.preparar(contexto), _resumo_minimo())
    assert set(dados["hora"]) == {20} and pesos.sum() == pytest.approx(1.0)


def test_expandir_publico_parcial_sem_observacao_e_recusado():
    contexto = {**CONTEXTO, "faixa_etaria": "60_mais"}
    with pytest.raises(preparo.EntradaInvalida):
        modelagem.expandir(preparo.preparar(contexto), _resumo_minimo())


def test_expandir_publico_completo_raro_usa_o_informado_e_conta_zero_no_recorte():
    contexto = {**CONTEXTO, "faixa_etaria": "60_mais", "sexo": "masculino", "escolaridade": "superior"}
    canonico = preparo.preparar(contexto)
    dados, pesos = modelagem.expandir(canonico, _resumo_minimo())
    assert set(dados["faixa_etaria"]) == {"60_mais"} and pesos.sum() == pytest.approx(1.0)
    assert modelagem.refeicoes_do_recorte(canonico, _resumo_minimo()) == 0


def test_recorte_raro_vira_evidencia_insuficiente_sem_alterar_a_original():
    original = {"faixa": "acima_do_habitual", "motivos": [], "nota": None}
    marcada = regras.marcar_recorte_raro(original, 4)
    assert marcada["faixa"] == "evidencia_insuficiente" and "recorte_raro" in marcada["motivos"]
    assert original["faixa"] == "acima_do_habitual"


@pytest.mark.skipif(not (modelagem.ARTEFATOS / "meta.json").exists(), reason="artefato publicado ausente")
def test_combinacao_rara_responde_e_nao_prioriza():
    r = modelagem.avaliar({"dia_semana": 3, "hora": 3, "grande_regiao": "Norte", "area": "rural",
                           "faixa_etaria": "60_mais", "sexo": "masculino", "escolaridade": "superior"})
    assert r["classification"] == "evidencia_insuficiente" and r["recommendation_category"] == "nao_priorizar"
    assert "recorte_raro" in r["comparacao"]["motivos"]


artefato = pytest.mark.skipif(not (modelagem.ARTEFATOS / "meta.json").exists(),
                              reason="artefato publicado ausente: rode python -m modelagem testar")


@artefato
def test_avaliar_devolve_os_seis_campos_do_tapi():
    r = modelagem.avaliar(CONTEXTO)
    for campo in ("occasion_score", "classification", "top_factors", "recommendation_category",
                  "recommendation_reason", "model_metadata"):
        assert campo in r
    assert 0 < r["occasion_score"] < 1
    assert r["recommendation_category"] in regras.CATEGORIAS


@artefato
def test_avaliar_muda_com_o_periodo_e_mantem_o_resto():
    noite = modelagem.avaliar(CONTEXTO)["occasion_score"]
    manha = modelagem.avaliar({**CONTEXTO, "periodo": "manha"})["occasion_score"]
    assert noite > manha


@artefato
def test_avaliar_publico_geral_e_media_dos_publicos():
    pacote = modelagem.carregar()
    geral = modelagem.avaliar(CONTEXTO)["occasion_score"]
    dados, pesos = modelagem.expandir(preparo.preparar(CONTEXTO), pacote.resumo)
    assert geral == pytest.approx(float(pesos @ pacote.modelo.prever(dados)))
    assert np.isclose(pesos.sum(), 1.0)


@artefato
@pytest.mark.parametrize("faixa, escolaridade", [("18_29", "superior"), ("10_17", "medio")])
def test_publico_parcial_observado_responde_mesmo_quando_trocar_um_campo_cai_em_combinacao_inexistente(
        faixa, escolaridade):
    """Aceite visual de 25/11/2026: 18-29 + superior com sexo 'todos' existe no treino, mas trocar a faixa para 10-17
    (sem superior) derrubava a avaliacao inteira. A comparacao do campo usa so as alternativas observadas."""
    contexto = {**CONTEXTO, "grande_regiao": "Nordeste", "faixa_etaria": faixa, "escolaridade": escolaridade}
    r = modelagem.avaliar(contexto)
    assert 0 < r["occasion_score"] < 1
    campos = {f["campo"] for f in r["fatores_avaliados"]}
    assert {"faixa_etaria", "escolaridade"} <= campos
    assert all(np.isfinite(f["efeito_pp"]) for f in r["fatores_avaliados"])


@artefato
def test_teste_aberto_impede_nova_selecao_e_novo_teste_mesmo_em_outra_pasta(tmp_path):
    with pytest.raises(SystemExit):
        modelagem.desenvolver(tmp_path)
    with pytest.raises(SystemExit):
        modelagem.testar(tmp_path)


@pytest.fixture
def no_ar():
    if not (modelagem.ARTEFATOS / "meta.json").exists():
        pytest.skip("artefato publicado ausente")
    instancia = servidor.criar_servidor("127.0.0.1", 0)
    linha = threading.Thread(target=instancia.serve_forever, daemon=True)
    linha.start()
    yield instancia.server_address[1]
    instancia.shutdown()
    instancia.server_close()


def _pedir(porta, metodo, caminho, corpo=None):
    conexao = http.client.HTTPConnection("127.0.0.1", porta, timeout=30)
    dados = json.dumps(corpo).encode() if corpo is not None else None
    cabecalhos = {"Host": f"127.0.0.1:{porta}", **({"Content-Type": "application/json"} if dados else {})}
    conexao.request(metodo, caminho, body=dados, headers=cabecalhos)
    resposta = conexao.getresponse()
    return resposta.status, json.loads(resposta.read())


def test_rota_avaliar_responde_no_envelope(no_ar):
    estado, corpo = _pedir(no_ar, "POST", "/api/avaliar", {"contexto": CONTEXTO})
    assert estado == 200 and corpo["success"] is True
    assert corpo["data"]["model_metadata"]["versao"] == modelagem.VERSAO


def test_rota_opcoes_traz_publico_geral(no_ar):
    estado, corpo = _pedir(no_ar, "GET", "/api/opcoes")
    assert estado == 200 and corpo["data"]["faixas_etarias"][0]["valor"] == "todos"


def test_rota_avaliar_recusa_corpo_fora_do_formato(no_ar):
    estado, corpo = _pedir(no_ar, "POST", "/api/avaliar", {"base": "historica", "contexto": CONTEXTO})
    assert estado == 400 and corpo["success"] is False


def test_rota_avaliar_recusa_campo_invalido(no_ar):
    estado, corpo = _pedir(no_ar, "POST", "/api/avaliar", {"contexto": {**CONTEXTO, "dia_semana": 9}})
    assert estado == 422 and corpo["error"]["type"] == "entrada_invalida"


def test_rotas_retiradas_nao_existem(no_ar):
    for caminho in ("/api/conversa", "/api/empresa/envios"):
        estado, _ = _pedir(no_ar, "POST", caminho, {"x": 1})
        assert estado == 404


# ---------- explicacao: importancia geral (desenvolvimento, fora da dobra) x contribuicao no cenario ----------

class _PorCampo:
    """Modelo de brinquedo: taxa observada por valor de um campo (aprende so com o que recebe em `ajustar`)."""

    def __init__(self, campo="periodo"):
        self.campo = campo

    def ajustar(self, dados, y):
        self.taxas = {v: float(y[dados[self.campo] == v].mean()) for v in np.unique(dados[self.campo])}
        self.media = float(y.mean())
        return self

    def prever(self, dados):
        return np.clip([self.taxas.get(v, self.media) for v in dados[self.campo]], 1e-4, 1 - 1e-4)


def _dados_sinteticos(n=4000, semente=3):
    rng = np.random.default_rng(semente)
    dados = {"upa": rng.integers(0, 400, n), "periodo": rng.choice(["noite", "manha"], n),
             "escolaridade": rng.choice(["medio", "superior"], n)}
    y = (rng.random(n) < np.where(dados["periodo"] == "noite", 0.30, 0.02)).astype(int)
    return dados, y


def test_importancia_fora_da_dobra_separa_campo_util_de_ruido_e_registra_a_procedencia():
    dados, y = _dados_sinteticos()
    divisoes = om.dobras(dados["upa"])
    r = modelagem.importancia_fora_da_dobra(lambda: _PorCampo("periodo"), dados, y, divisoes,
                                            grupos={"periodo": ("periodo",), "escolaridade": ("escolaridade",)})
    assert r["periodo"]["sustentada"] is True and r["escolaridade"]["sustentada"] is False
    assert r["escolaridade"]["aumento_medio"] == pytest.approx(0.0)
    assert len(r["periodo"]["aumentos"]) == modelagem.REPETICOES_PERMUTACAO
    assert len(r["periodo"]["aumento_por_dobra"]) == len(divisoes)


def test_importancia_fora_da_dobra_reproduz_a_perda_fora_da_dobra_da_validacao():
    dados, y = _dados_sinteticos()
    divisoes = om.dobras(dados["upa"])
    oof = om.validar(lambda: _PorCampo("periodo"), dados, y, divisoes)["oof"]
    r = modelagem.importancia_fora_da_dobra(lambda: _PorCampo("periodo"), dados, y, divisoes,
                                            grupos={"periodo": ("periodo",)})
    assert r["periodo"]["perda_base"] == pytest.approx(om.perda_log(y, oof))


def _pacote_de_brinquedo(ajuda: dict):
    resumo = {"horas_por_periodo": {"noite": {"20": 1}, "manha": {"8": 1}},
              "composicao_publico": {"Sudeste|capital": {"18_29|feminino|medio": 3, "18_29|feminino|superior": 1}},
              "distribuicoes": {"dia_semana": {"5": 1, "6": 1}, "periodo": {"noite": 1, "manha": 1},
                                "grande_regiao": {"Sudeste": 1}, "area": {"capital": 1},
                                "faixa_etaria": {"18_29": 1}, "sexo": {"feminino": 1},
                                "escolaridade": {"medio": 3, "superior": 1}}}

    class Modelo:
        def prever(self, dados):
            return (np.where(dados["periodo"] == "noite", 0.004, 0.001)
                    * np.where(dados["escolaridade"] == "superior", 2.0, 1.0))

    importancia = {c: {"sustentada": ajuda.get(c, False), "aumento_medio": 0.0} for c in modelagem.GRUPOS_PERMUTACAO}
    return types.SimpleNamespace(modelo=Modelo(), resumo=resumo, importancia_geral=importancia)


def test_fator_util_no_geral_mas_neutro_no_cenario_nao_aparece_e_o_inverso_tambem_nao():
    pacote = _pacote_de_brinquedo({"periodo": True, "dia_semana": True})
    canonico = preparo.preparar({**CONTEXTO, "dia_semana": 5, "escolaridade": "superior"})
    lista = modelagem.fatores(pacote, canonico, modelagem._media(pacote.modelo, canonico, pacote.resumo))
    por_campo = {f["campo"]: f for f in lista}
    assert por_campo["dia_semana"]["ajuda_o_modelo"] and por_campo["dia_semana"]["direcao"] == "neutro"
    assert not por_campo["escolaridade"]["ajuda_o_modelo"] and por_campo["escolaridade"]["direcao"] == "eleva"
    principais = modelagem.principais_fatores(lista)
    assert [f["campo"] for f in principais] == ["periodo"]
    assert principais[0]["importancia_geral"]["origem"] == modelagem.ORIGEM_IMPORTANCIA


def test_publico_geral_nao_recebe_fator_de_escolaridade():
    pacote = _pacote_de_brinquedo({"escolaridade": True, "periodo": True})
    canonico = preparo.preparar(CONTEXTO)
    lista = modelagem.fatores(pacote, canonico, modelagem._media(pacote.modelo, canonico, pacote.resumo))
    assert all(f["campo"] not in preparo.CAMPOS_PUBLICO for f in lista)
    assert all("escolaridade" not in f["texto"].lower() for f in lista)


@artefato
def test_artefato_usa_importancia_do_desenvolvimento_ligada_ao_modelo_publicado():
    pacote = modelagem.carregar()
    explicacao = json.loads((modelagem.ARTEFATOS / "explicacao.json").read_text(encoding="utf-8"))
    assert explicacao["origem"] == modelagem.ORIGEM_IMPORTANCIA and explicacao["particao"] == "desenvolvimento"
    assert explicacao["modelo_sha256"] == pacote.sha256["modelo.joblib"]
    assert {c: v["sustentada"] for c, v in pacote.importancia_geral.items()} == \
        {c: v["sustentada"] for c, v in explicacao["importancia"].items()}


@artefato
def test_avaliar_mostra_so_fatores_uteis_no_geral_e_ativos_no_cenario():
    r = modelagem.avaliar(CONTEXTO)
    assert r["top_factors"] == modelagem.principais_fatores(r["fatores_avaliados"])
    assert all(f["ajuda_o_modelo"] and f["direcao"] != "neutro" for f in r["top_factors"])
    assert r["model_metadata"]["explicacao"]["origem"] == modelagem.ORIGEM_IMPORTANCIA


# ---------- hora exata (fator "horario") x so o periodo (fator "periodo") ----------

def test_com_hora_exata_o_fator_e_o_horario_calculado_com_hora_e_periodo_juntos():
    pacote = _pacote_de_brinquedo({"periodo": True})
    contexto = {k: v for k, v in CONTEXTO.items() if k != "periodo"} | {"hora": 20}
    canonico = preparo.preparar(contexto)
    lista = modelagem.fatores(pacote, canonico, modelagem._media(pacote.modelo, canonico, pacote.resumo))
    campos = [f["campo"] for f in lista]
    assert "horario" in campos and "periodo" not in campos
    horario = next(f for f in lista if f["campo"] == "horario")
    assert horario["valor"] == 20 and horario["ajuda_o_modelo"] is True
    assert horario["texto"].startswith("O horário das 20h (noite) eleva a estimativa")
    assert "hora e período juntos" in horario["texto"]


def test_so_com_o_periodo_o_fator_continua_sendo_o_periodo():
    pacote = _pacote_de_brinquedo({"periodo": True})
    canonico = preparo.preparar(CONTEXTO)
    lista = modelagem.fatores(pacote, canonico, modelagem._media(pacote.modelo, canonico, pacote.resumo))
    periodo = next(f for f in lista if f["campo"] == "periodo")
    assert "horario" not in [f["campo"] for f in lista]
    assert periodo["texto"] == "O período da noite eleva a estimativa em relação à média dos períodos, no mesmo modelo."


def test_rotulo_do_campo_horario_existe_para_a_tela():
    assert regras.CAMPOS["horario"] == "Horário"


@artefato
def test_scores_dos_dois_caminhos_iguais_aos_registrados_antes_da_mudanca():
    geral = {"faixa_etaria": "todos", "sexo": "todos", "escolaridade": "todos"}
    so_periodo = modelagem.avaliar({"dia_semana": 5, "periodo": "noite", "grande_regiao": "Sudeste",
                                    "area": "capital", **geral})
    hora = modelagem.avaliar({"dia_semana": 3, "hora": 3, "grande_regiao": "Norte", "area": "rural",
                              "faixa_etaria": "60_mais", "sexo": "masculino", "escolaridade": "superior"})
    assert so_periodo["occasion_score"] == 0.0023194004592835605
    assert hora["occasion_score"] == 0.00016150496463078766
    assert "horario" in [f["campo"] for f in hora["fatores_avaliados"]]
    assert "periodo" in [f["campo"] for f in so_periodo["fatores_avaliados"]]


# ---------- analise exploratoria do alvo (python -m modelagem eda) ----------

def _escolhas_sinteticas(caminho, linhas):
    import duckdb
    colunas = ("particao", "periodo", "dia_semana", "hora", "grande_regiao", "area", "faixa_etaria", "sexo",
               "escolaridade", "local_v9018", "pizza", "guarana", "peso")
    valores = ", ".join("(" + ", ".join(repr(v) for v in linha) + ")" for linha in linhas)
    consulta = f"SELECT * FROM (VALUES {valores}) t({', '.join(colunas)})"
    duckdb.sql(f"COPY ({consulta}) TO '{caminho.as_posix()}' (FORMAT PARQUET)")


def test_eda_tabula_o_alvo_por_eixo_sem_abrir_o_teste_reservado(tmp_path):
    comum = ("Sudeste", "capital", "18_29", "feminino", "medio", "1")
    _escolhas_sinteticas(tmp_path / "escolhas_pof_2017.parquet", [
        ("desenvolvimento", "noite", 6, 20, *comum, 1, 1, 1.0),
        ("desenvolvimento", "noite", 6, 21, *comum, 1, 0, 3.0),
        ("desenvolvimento", "manha", 2, 8, *comum, 0, 0, 1.0),
        ("teste", "noite", 6, 20, *comum, 1, 1, 1.0),  # nao pode entrar em nenhuma contagem
    ])
    _escolhas_sinteticas(tmp_path / "escolhas_pof_2008.parquet",
                         [("desenvolvimento", "noite", 5, 20, *comum, 1, 1, 1.0)])
    eda = modelagem.analise_exploratoria(saida=tmp_path, pasta=tmp_path)
    assert set(eda) == {"2017_desenvolvimento", "2008_todas"}
    assert set(eda["2017_desenvolvimento"]) == set(modelagem.EIXOS_EDA)
    noite = next(linha for linha in eda["2017_desenvolvimento"]["periodo"] if linha["valor"] == "noite")
    assert (noite["refeicoes"], noite["positivos"], noite["refeicoes_com_pizza"]) == (2, 1, 2)
    assert noite["taxa"] == pytest.approx(0.5) and noite["taxa_ponderada"] == pytest.approx(0.25)
    assert noite["guarana_dado_pizza"] == pytest.approx(0.5)
    fim = {linha["valor"]: linha["refeicoes"] for linha in eda["2017_desenvolvimento"]["fim_de_semana_noite"]}
    assert fim == {"demais": 1, "sexta_a_domingo_noite": 2}
    assert json.loads((tmp_path / "eda.json").read_text(encoding="utf-8")) == eda
