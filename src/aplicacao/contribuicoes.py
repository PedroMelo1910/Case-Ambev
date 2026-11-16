"""Contribuicoes aditivas da logistica regressão logística em log-odds, agrupadas pela variavel original.

O pipeline padroniza cada coluna (z = (x - media) / escala) e aplica a logistica: log-odds = intercepto + soma(coef * z).
A contribuicao de uma coluna e coef * z; a de uma variavel e a soma das suas colunas (um indicador por nivel). Cada coluna
pertence a um unico grupo de `configs/variaveis.json`, entao nada e contado duas vezes; a interacao sexta a domingo x noite
e uma variavel propria (nao e repartida). A referencia (z = 0, contribuicao zero) e o perfil medio de 2008 usado na
padronizacao, que pode nao representar uma pessoa real. Sao contribuicoes em log-odds relativas a essa referencia: nao
sao pontos percentuais de probabilidade nem efeitos causais. modelo de taxa histórica nao passa por aqui: nao tem fatores de contexto.
"""

import math

import numpy as np

from occasion.variaveis import carregar_regras

TOLERANCIA = 1e-9
UNIDADE = "log-odds"
TOPO = 3
_SIM_NAO = {1: "sim", 0: "não", None: "desconhecido"}
AVISO = ("Contribuições aditivas em log-odds, relativas ao perfil médio de 2008; não são pontos percentuais de "
         "probabilidade nem efeito causal. Os três fatores exibidos não reconstroem sozinhos o resultado: o log-odds "
         "previsto é o intercepto mais a soma de todas as contribuições por variável, listadas em "
         "contribuicoes_por_variavel.")
DESCRICAO_REFERENCIA = ("Perfil médio de 2008: o ponto em que cada coluna vale a sua média no treino, usado na padronização "
                        "(por exemplo, a fração dos episódios em cada faixa de hora ou região). Esse perfil não representa "
                        "uma pessoa real; as contribuições medem a distância a ele.")
TRATAMENTO_INTERACAO = ("A interação sexta a domingo × noite é exibida como variável própria e sua contribuição não é "
                        "repartida entre sexta_a_domingo e faixa_hora, que mostram só os efeitos principais; cada coluna "
                        "entra em um único grupo, sem dupla contagem.")
_VALORES = {
    "faixa_hora": lambda n: n["faixa_hora"], "grande_regiao": lambda n: n["grande_regiao"], "area": lambda n: n["area"],
    "faixa_etaria": lambda n: n["faixa_etaria"], "sexo": lambda n: n["sexo"], "escolaridade": lambda n: n["escolaridade"],
    "sexta_a_domingo": lambda n: _SIM_NAO[n["sexta_a_domingo"]],
    "renda_per_capita_log": lambda n: math.exp(n["log_renda_pc"]), "moradores_uc": lambda n: n["moradores_uc"],
    "presenca_crianca_0_11": lambda n: _SIM_NAO[n["presenca_crianca"]],
    "presenca_adolescente_12_17": lambda n: _SIM_NAO[n["presenca_adolescente"]],
    "presenca_idoso_60_mais": lambda n: _SIM_NAO[n["presenca_idoso"]],
    "sexta_a_domingo_x_noite": lambda n: _SIM_NAO[int(n["sexta_a_domingo"] == 1 and n["faixa_hora"] == "noite")]}
_UNIDADES = {"renda_per_capita_log": "R$ por pessoa por mês, em reais de jan/2009", "moradores_uc": "moradores da UC"}


class ExplicacaoInconsistente(RuntimeError):
    """As contribuicoes nao reconstroem a previsao, ou as colunas nao conferem com os grupos. Nunca vira explicacao."""


def _etapas(pipeline):
    try:
        return pipeline.named_steps["standardscaler"], pipeline.named_steps["logisticregression"]
    except (AttributeError, KeyError) as erro:
        raise ExplicacaoInconsistente("pipeline sem StandardScaler e LogisticRegression") from erro


def explicar(pipeline, colunas: list[str], valores: list) -> dict:
    """Contribuicoes por variavel original e conferencia de que intercepto + soma reconstroem o log-odds previsto."""
    escala, logistica = _etapas(pipeline)
    grupos = carregar_regras()["variaveis"]
    if sorted(c for g in grupos for c in g["colunas"]) != sorted(colunas) or len(set(colunas)) != len(colunas):
        raise ExplicacaoInconsistente("colunas da logistica nao conferem com os grupos de variaveis (ou se repetem)")
    if len(valores) != len(colunas) or len(colunas) != escala.n_features_in_:
        raise ExplicacaoInconsistente(f"{len(valores)} valores para {len(colunas)} colunas ({escala.n_features_in_} no modelo)")
    x = np.array([np.nan if v is None else v for v in valores], dtype=float)
    if not np.isfinite(x).all():
        raise ExplicacaoInconsistente("valor desconhecido ou nao finito: a logistica recusa imputacao")
    por_coluna = logistica.coef_[0] * (x - escala.mean_) / escala.scale_
    indice = {coluna: i for i, coluna in enumerate(colunas)}
    intercepto = float(logistica.intercept_[0])
    previsto = float(pipeline.decision_function(x.reshape(1, -1))[0])
    reconstruido = intercepto + math.fsum(por_coluna)
    diferenca = abs(reconstruido - previsto)
    if diferenca > TOLERANCIA:
        raise ExplicacaoInconsistente(f"intercepto mais contribuicoes difere do log-odds previsto em {diferenca:.3e}")
    contribuicoes = [{"variavel": g["nome"], "id": g["id"],
                      "contribuicao_log_odds": math.fsum(por_coluna[indice[c]] for c in g["colunas"])} for g in grupos]
    return {"intercepto_log_odds": intercepto, "contribuicoes_por_variavel": contribuicoes, "log_odds_previsto": previsto,
            "log_odds_reconstruido": reconstruido, "diferenca_de_reconstrucao": diferenca,
            "probabilidade_na_referencia": 1.0 / (1.0 + math.exp(-intercepto))}


def _direcao(contribuicao: float) -> str:
    return "aumenta" if contribuicao > 0 else ("reduz" if contribuicao < 0 else "neutra")


def _item(item: dict, nativas: dict) -> dict:
    nome = item["variavel"]
    if nome not in _VALORES:
        raise ExplicacaoInconsistente(f"variavel sem valor exibivel: {nome}")
    saida = {"variavel": nome, "valor": _VALORES[nome](nativas), "contribuicao_log_odds": item["contribuicao_log_odds"],
             "direcao": _direcao(item["contribuicao_log_odds"])}
    if nome in _UNIDADES:
        saida["unidade_do_valor"] = _UNIDADES[nome]
    return saida


def apresentar(calculo: dict, nativas: dict, topo: int = TOPO) -> tuple[list[dict], dict]:
    """Principais fatores (maiores em modulo, com direcao) e o bloco `explicacao` do model_metadata."""
    itens = [_item(item, nativas) for item in calculo["contribuicoes_por_variavel"]]
    ordenados = sorted((i for i in itens if i["contribuicao_log_odds"] != 0), key=lambda i: -abs(i["contribuicao_log_odds"]))
    fatores = ordenados[:topo]
    exibidas = math.fsum(f["contribuicao_log_odds"] for f in fatores)
    soma_total = calculo["log_odds_previsto"] - calculo["intercepto_log_odds"]
    bloco = {"disponivel": True, "unidade": UNIDADE, "intercepto_log_odds": calculo["intercepto_log_odds"],
             "contribuicoes_por_variavel": itens, "log_odds_previsto": calculo["log_odds_previsto"],
             "log_odds_reconstruido": calculo["log_odds_reconstruido"],
             "diferenca_de_reconstrucao": calculo["diferenca_de_reconstrucao"],
             "soma_dos_fatores_exibidos": exibidas, "restante_nao_exibido": soma_total - exibidas,
             "referencia": {"descricao": DESCRICAO_REFERENCIA,
                            "probabilidade_na_referencia": calculo["probabilidade_na_referencia"]},
             "tratamento_da_interacao": TRATAMENTO_INTERACAO, "aviso": AVISO}
    return fatores, bloco
