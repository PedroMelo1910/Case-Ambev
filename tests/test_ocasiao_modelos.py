"""Metricas e regras de escolha do Occasion Score observado (dados sinteticos pequenos)."""

import numpy as np
import pytest

import ocasiao_modelos as om


def _dados(n=400, semente=0):
    rng = np.random.default_rng(semente)
    dias = rng.integers(1, 8, n)
    periodos = rng.choice(["manha", "almoco", "tarde", "noite"], n)
    noite_fds = (periodos == "noite") & (dias >= 5)
    pizza = (rng.random(n) < np.where(noite_fds, 0.5, 0.05)).astype(float)
    guarana = pizza * (rng.random(n) < 0.4)
    y = pizza * guarana
    return {"dia_semana": dias, "periodo": periodos, "hora": rng.integers(0, 24, n),
            "grande_regiao": rng.choice(["Sul", "Norte"], n), "area": rng.choice(["capital", "rural"], n),
            "faixa_etaria": rng.choice(["18_29", "30_44"], n), "sexo": rng.choice(["feminino", "masculino"], n),
            "escolaridade": rng.choice(["medio", "superior"], n), "pizza": pizza, "guarana": guarana,
            "upa": rng.integers(0, 40, n).astype(str)}, y


def test_priorizacao_sem_empates_captura_os_maiores():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    r = om.priorizacao(y, p, 0.2)  # 2 refeicoes: 0.9 (positiva) e 0.8 (negativa)
    assert r["positivos_capturados"] == pytest.approx(1.0)
    assert r["captura"] == pytest.approx(0.5)
    assert r["precisao"] == pytest.approx(0.5)
    assert r["lift"] == pytest.approx(0.5 / 0.2)


def test_priorizacao_divide_o_empate_da_fronteira_por_fracao():
    # 10 refeicoes empatadas com 4 positivos: os 20% (2 refeicoes) levam 2/10 dos positivos do grupo
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    r = om.priorizacao(y, np.full(10, 0.3), 0.2)
    assert r["positivos_capturados"] == pytest.approx(0.8)
    assert r["captura"] == pytest.approx(0.2)  # igual ao acaso: score constante nao prioriza nada


def test_priorizacao_matriz_de_confusao_soma_o_total():
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float)
    r = om.priorizacao(y, np.linspace(1, 0, 10), 0.2)
    m = r["matriz_confusao"]
    assert m["vp"] + m["fp"] + m["fn"] + m["vn"] == pytest.approx(10)


def test_dia_periodo_responde_taxa_global_para_celula_inedita_e_suaviza():
    dados, y = _dados()
    modelo = om.DiaPeriodo(m=1e9).ajustar(dados, y)
    assert np.allclose(modelo.prever(dados), y.mean(), atol=1e-6)  # m enorme = taxa global
    inedito = {**{k: v[:1] for k, v in dados.items()}, "dia_semana": np.array([9])}
    assert om.DiaPeriodo(m=0).ajustar(dados, y).prever(inedito)[0] == pytest.approx(y.mean())


def test_escolher_volta_ao_mais_simples_sem_vantagem_sustentada():
    resultados = {"global": {"perda_media": 0.010, "perda_por_dobra": [0.010, 0.011, 0.009, 0.010, 0.010]},
                  "logistica": {"perda_media": 0.00999, "perda_por_dobra": [0.0101, 0.0109, 0.0091, 0.0099, 0.0100]}}
    assert om.escolher(resultados)["escolhido"] == "global"


def test_escolher_aceita_vantagem_consistente():
    resultados = {"global": {"perda_media": 0.010, "perda_por_dobra": [0.010] * 5},
                  "logistica": {"perda_media": 0.008, "perda_por_dobra": [0.0080, 0.0081, 0.0079, 0.0080, 0.0080]}}
    assert om.escolher(resultados)["escolhido"] == "logistica"


def test_dobras_mantem_cada_upa_numa_so_dobra():
    dados, y = _dados()
    for treino, valida in om.dobras(dados["upa"]):
        assert not set(dados["upa"][treino]) & set(dados["upa"][valida])


def test_dois_passos_e_produto_dos_componentes():
    dados, y = _dados()
    modelo = om.DoisPassos("B1", 1.0, 1.0).ajustar(dados, y)
    p, g = modelo.componentes(dados)
    assert np.allclose(modelo.prever(dados), p * g)


def test_metricas_de_score_constante_tem_auc_neutra():
    y = np.array([0, 1, 0, 0], dtype=float)
    m = om.metricas(y, np.full(4, 0.25))
    assert m["roc_auc"] == 0.5
    assert m["calibracao"]["observado_sobre_esperado"] == pytest.approx(1.0)
