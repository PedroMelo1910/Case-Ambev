"""PE-01: 13 variaveis derivadas e 28 colunas (plano 7.3). Fronteiras, renda, composicao da UC, esquema e vazamento."""

import json

import pytest

import preparo as v

R08, R17 = "2008-2009", "2017-2018"
FATOR_2017 = 0.5895163383846578


def registro(**extra):
    base = {"hora": 20, "dia_semana": 6, "grande_regiao": "Sudeste", "area_comparavel": "capital", "idade": 34,
            "sexo": "feminino", "escolaridade": 5, "renda_total": 2000.0, "idades_uc": [34, 12]}
    base.update(extra)
    return base


# --- regras e esquema -----------------------------------------------------------------------------


def test_colunas_derivadas_pertencem_a_variaveis_declaradas():
    regras = v.carregar_regras()
    declaradas = [c for var in regras["variaveis"] for c in var["colunas"]]
    assert sorted(declaradas) == sorted(regras["colunas_logistica"])
    assert [var["id"] for var in regras["variaveis"]] == [
        "F03", "F04", "F07", "F09", "F12", "F13", "F14", "F15", "F18", "F19", "F20", "F21", "F45"]


def test_fator_monetario_so_reduz_2017_para_reais_de_janeiro_de_2009():
    regras = v.carregar_regras()
    assert regras["fator_renda"][R08] == 1.0
    assert regras["fator_renda"][R17] == pytest.approx(FATOR_2017, rel=1e-12)
    assert regras["fator_renda"][R17] < 1


# --- faixa de hora --------------------------------------------------------------------------------
@pytest.mark.parametrize("hora,faixa", [
    (0, "noite"), (4, "noite"), (5, "manha"), (10, "manha"), (11, "almoco"), (14, "almoco"),
    (15, "tarde"), (18, "tarde"), (19, "noite"), (23, "noite")])
def test_faixa_de_hora_nas_fronteiras(hora, faixa):
    assert v.faixa_hora(hora) == faixa


@pytest.mark.parametrize("hora", [-1, 24, None, "12", 12.5, True])
def test_hora_invalida_interrompe(hora):
    with pytest.raises(ValueError):
        v.faixa_hora(hora)


# --- sexta a domingo ------------------------------------------------------------------------------


# --- faixa etaria ---------------------------------------------------------------------------------
@pytest.mark.parametrize("idade,faixa", [
    (10, "10_17"), (17, "10_17"), (18, "18_29"), (29, "18_29"), (30, "30_44"), (44, "30_44"),
    (45, "45_59"), (59, "45_59"), (60, "60_mais"), (111, "60_mais")])
def test_faixa_etaria_nas_fronteiras(idade, faixa):
    assert v.faixa_etaria(idade) == faixa


@pytest.mark.parametrize("idade", [0, 9, None])
def test_idade_fora_da_populacao_do_consumo_e_desconhecida_nao_zero(idade):
    assert v.faixa_etaria(idade) is None


def test_idade_negativa_interrompe():
    with pytest.raises(ValueError):
        v.faixa_etaria(-1)


# --- escolaridade ---------------------------------------------------------------------------------
@pytest.mark.parametrize("nivel,grupo", [(1, "ate_fundamental"), (3, "ate_fundamental"), (4, "medio"), (5, "medio"),
                                         (6, "superior"), (7, "superior")])
def test_escolaridade_em_tres_niveis(nivel, grupo):
    assert v.escolaridade_3(nivel) == grupo


@pytest.mark.parametrize("nivel", [0, 8, None])
def test_escolaridade_desconhecida_interrompe(nivel):
    with pytest.raises(ValueError):
        v.escolaridade_3(nivel)


# --- renda ----------------------------------------------------------------------------------------


# --- composicao da unidade de consumo -------------------------------------------------------------


# --- categorias desconhecidas ---------------------------------------------------------------------


# --- matriz de 28 colunas -------------------------------------------------------------------------


# --- vazamento ------------------------------------------------------------------------------------


# --- robustez (achados da revisao) ----------------------------------------------------------------


def test_regras_sao_imutaveis():
    regras = v.carregar_regras()
    with pytest.raises(TypeError):
        regras["fator_renda"][R17] = 1.0
    assert isinstance(regras["colunas_logistica"], tuple)


def _config_com(repo, tmp_path, alterar):
    dados = json.loads((repo / "configs" / "variaveis.json").read_text(encoding="utf-8"))
    alterar(dados)
    caminho = tmp_path / "variaveis.json"
    caminho.write_text(json.dumps(dados), encoding="utf-8")
    return caminho


@pytest.mark.parametrize("alterar", [
    lambda d: d["faixas_hora"]["manha"].remove(10),
    lambda d: d["faixas_hora"]["tarde"].append(10),
    lambda d: d["escolaridade_3"]["medio"].append(3),
    lambda d: d["faixas_etarias"][1].update(min=19),
    lambda d: d["colunas_logistica"].pop(),
    lambda d: d["variaveis"].pop(),
], ids=["hora-lacuna", "hora-sobreposta", "escolaridade-sobreposta", "idade-lacuna", "27-colunas", "12-variaveis"])
def test_config_incoerente_e_recusada(repo, tmp_path, alterar):
    with pytest.raises(v.RegrasInvalidas):
        v.carregar_regras(_config_com(repo, tmp_path, alterar))


