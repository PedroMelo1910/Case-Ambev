"""Funcao de inferencia: contexto -> os seis campos do contrato, sem treinar, ajustar limiares nem reabrir avaliacao.

`inferir(contexto, modelo)` verifica a integridade dos artefatos empacotados, valida o contexto, aplica o artefato
congelado com as transformacoes e classes ja existentes (`occasion`), decide a recomendacao no motor de regras e monta
os campos: occasion_score, classification, top_factors, recommendation_category, recommendation_reason, model_metadata.
O padrao e modelo de taxa histórica (modelo selecionado, taxa constante, sem fatores de contexto); regressão logística e a logistica exploratoria, com
contribuicoes em log-odds. O papel de cada modelo vem do manifesto e de metadados da aplicacao, nunca do contexto.
O score estima a frequencia de guarana entre episodios com pizza, conforme o contexto, na POF 2008-2009: nao comprova
pedidos de delivery, marca nem efeito de marketing.
"""

import copy
import math
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from aplicacao import contribuicoes, entrada, modelos, regras
from aplicacao.regras import EntradaInvalida, recebido
from occasion import congelamento as cg
from occasion.variaveis import COLUNAS_LOGISTICA

ALVO = ("Episódio de consumo (pessoa × dia de coleta × hora) com pizza que também contém guaraná; o score estima a "
        "frequência de guaraná entre episódios com pizza, conforme o contexto.")
TREINO = "POF 2008-2009, 1.168 episódios com pizza (101 com guaraná); desenvolvimento e seleção só com 2008-2009."
DESCRICAO = {
    "B0": "Taxa constante: modelo selecionado com 2008 por parcimônia; devolve a taxa média geral e não distingue contextos.",
    "L": "Regressão logística com regularização ridge: análise exploratória que ordena contextos; não foi promovida a "
         "modelo selecionado."}
AVISO_ALVO = ("O score estima a frequência de guaraná entre episódios com pizza na POF 2008-2009, conforme o contexto; "
              "não comprova pedidos de delivery, a marca Antarctica nem efeito de marketing.")
AVISO_TAXA_HISTORICA = "O score do modelo de taxa histórica não é uma previsão para este contexto: é a taxa média geral de 2008, igual para qualquer contexto."
AVISO_REGRESSAO_LOGISTICA = ("Uso exploratório: a regressão logística não é o modelo selecionado; o modelo selecionado continua sendo o modelo de taxa histórica. "
           "A classe e o score da regressão logística não demonstram efeito causal.")
AVISO_DOMINIO = "O score foi calculado, mas o domínio do contexto não está verificado nas marginais: não sustenta recomendação."
AVISO_RENDA = ("A referência monetária da renda é declarada por quem informa e não pode ser verificada: confira "
               "renda_per_capita_jan_2009 em entrada, que é o valor usado pelo modelo.")
NOTA_REGRAS = ("regras redigidas após a avaliação temporal de 2017; não participaram da seleção com 2008 e não alteram "
               "score nem classe")
MOTIVO_CLASSE_TAXA_HISTORICA = ("modelo de taxa histórica é constante: o score é a própria taxa-base e a classe é sempre 'media' por construção, sem "
                    "relação com o contexto.")
MOTIVO_CLASSE_FORA = ("O domínio do contexto não está verificado nas marginais (fora ou desconhecido): a classe não deve "
                      "ser lida como propensão do contexto.")
MOTIVO_CLASSE_REGRESSAO_LOGISTICA = ("Compara o score da regressão logística com o limiar de alta (cobertura de desenvolvimento de 30%) e a taxa-base de 2008; classe exploratória, sem relação causal "
                   "demonstrada.")
TOLERANCIA_SCORE = 1e-9
EXPLICACAO_TAXA_HISTORICA = {"disponivel": False, "motivo": "modelo de taxa histórica é constante: não usa o contexto e não tem fatores contextuais; "
                                                "nenhum fator foi inventado."}
_AMBIENTE = ("python", "scikit-learn", "numpy", "scipy", "joblib")


NOMES_PUBLICOS = {"taxa_historica": "B0", "regressao_logistica": "L"}


def _nome_do_modelo(modelo: object) -> str:
    if type(modelo) is not str:
        raise EntradaInvalida(f"modelo deve ser um de {list(NOMES_PUBLICOS)}, recebido {recebido(modelo)}")
    nome = NOMES_PUBLICOS.get(modelo, modelo)
    if nome not in modelos.MODELOS:
        raise EntradaInvalida(f"modelo deve ser um de {list(NOMES_PUBLICOS)}, recebido {recebido(modelo)}")
    return nome


def _prever(pacote: modelos.Pacote, manifesto: dict, nome: str, contexto: entrada.Contexto) -> cg.Previsao:
    estimador = pacote.modelos[nome]
    if nome == "B0":
        return cg.prever_congelado(manifesto, nome, pacote.base, 1, modelo=estimador)
    desconhecidas = [coluna for coluna, valor in contexto.linha.items() if valor is None]
    if desconhecidas:
        raise EntradaInvalida(f"a regressão logística recusa valor desconhecido (sem imputacao): informe as idades de todos os "
                              f"moradores da UC; colunas afetadas: {desconhecidas}")
    return cg.prever_congelado(manifesto, nome, pacote.base, [contexto.linha], modelo=estimador)


def _aviso_log_loss(manifesto: dict) -> str:
    b0 = manifesto["modelos"]["B0"]["metricas_validacao"]["log_loss_media"]
    logistica = manifesto["modelos"]["L"]["metricas_validacao"]["log_loss_media"]
    verbo = "não reduziu" if logistica >= b0 else "reduziu"
    return (f"Na validação de 2008, a regressão logística {verbo} o log-loss do modelo de taxa histórica "
            f"(regressão logística {logistica:.4f}; taxa histórica {b0:.4f}); ganho de validação "
            "não demonstrado não comprova equivalência nem superioridade.")


def _classificacao(papel: modelos.Papel, contexto: entrada.Contexto) -> dict:
    if not papel.ordena_contextos:
        return {"interpretavel": False, "motivo": MOTIVO_CLASSE_TAXA_HISTORICA}
    if contexto.dominio["ok"] is not True:
        return {"interpretavel": False, "motivo": MOTIVO_CLASSE_FORA}
    return {"interpretavel": True, "motivo": MOTIVO_CLASSE_REGRESSAO_LOGISTICA}


def _metadados(pacote: modelos.Pacote, manifesto: dict, nome: str, contexto: entrada.Contexto, decisao: regras.Decisao,
               previsao: cg.Previsao, explicacao: dict) -> dict:
    bloco = manifesto["modelos"][nome]
    avisos = [AVISO_ALVO, AVISO_TAXA_HISTORICA if nome == "B0" else AVISO_REGRESSAO_LOGISTICA, AVISO_RENDA]
    if nome == "L":
        avisos.append(_aviso_log_loss(manifesto))
    if previsao.aviso:
        avisos.append(previsao.aviso)
    if contexto.dominio["ok"] is not True:
        avisos.append(AVISO_DOMINIO)
    return {
        "modelo": next(publico for publico, interno in NOMES_PUBLICOS.items() if interno == nome), "descricao": DESCRICAO[nome], "papel": asdict(pacote.papeis[nome]), "alvo": ALVO, "treino": TREINO,
        "regra": {"id": decisao.rule_id, "versao": decisao.rules_version, "nota": NOTA_REGRAS},
        "classificacao": _classificacao(pacote.papeis[nome], contexto), "dominio": contexto.dominio,
        "entrada": contexto.renda,
        "limiares": {"t30": bloco["t30_aplicavel"], "corte_media_baixa": manifesto["constantes"]["b"],
                     "cobertura_operacional": manifesto["constantes"]["cobertura_operacional"],
                     "limiares_fora_da_dobra": copy.deepcopy(bloco["limiares_fora_da_dobra"])},
        "metricas_validacao": copy.deepcopy(bloco["metricas_validacao"]), "limitacoes": list(bloco["limitacoes"]),
        "explicacao": explicacao, "avisos": avisos,
        "versoes": {"manifesto_sha256": pacote.manifesto_sha256, "manifesto": manifesto["versao_manifesto"],
                    "protocolo_de_treinamento": manifesto["protocolo_de_treinamento"]["versao"],
                    "protocolo_operacional": manifesto["protocolo_vigente"], "regras": decisao.rules_version,
                    "ambiente": {chave: manifesto["ambiente"][chave] for chave in _AMBIENTE}}}


def _explicar_regressao_logistica(pacote: modelos.Pacote, manifesto: dict, validado: entrada.Contexto, score: float) -> tuple[list, dict]:
    """Fatores e bloco de explicacao da regressão logística, conferidos contra o score emitido."""
    calculo = contribuicoes.explicar(pacote.modelos["L"], list(COLUNAS_LOGISTICA),
                                     [validado.linha[coluna] for coluna in COLUNAS_LOGISTICA])
    if not 0.0 < score < 1.0:
        raise contribuicoes.ExplicacaoInconsistente(f"score emitido fora de (0, 1): {score!r}")
    diferenca = abs(math.log(score / (1.0 - score)) - calculo["log_odds_previsto"])
    if diferenca > TOLERANCIA_SCORE:
        raise contribuicoes.ExplicacaoInconsistente(f"log-odds da explicacao difere do score emitido em {diferenca:.3e}")
    fatores, explicacao = contribuicoes.apresentar(calculo, validado.nativas)
    explicacao["score_taxa_historica"] = manifesto["constantes"]["p0"]
    return fatores, explicacao


def _decidir(papel: modelos.Papel, dominio_ok: bool | None, classe: str) -> regras.Decisao:
    """Papel e classe vem do manifesto e do modelo: se o motor os recusar, a falha e interna e nao do usuario."""
    try:
        return regras.decidir({"ordena_contextos": papel.ordena_contextos, "uso_exploratorio": papel.uso_exploratorio,
                               "dominio_ok": dominio_ok, "classification": classe})
    except EntradaInvalida as erro:
        raise modelos.IntegridadeInvalida(f"fatos internos incoerentes para o motor de regras (papel ou classe): {erro}") from erro


def inferir(contexto: Mapping[str, object], modelo: str = "taxa_historica", *, base: Path | str | None = None) -> dict:
    """Os seis campos do contrato para um contexto. Entrada invalida levanta EntradaInvalida; artefato ou ambiente
    divergente levanta IntegridadeInvalida; nenhuma das duas vira resultado."""
    nome = _nome_do_modelo(modelo)
    pacote = modelos.carregar(base)
    manifesto = pacote.manifesto
    validado = entrada.preparar(contexto)
    previsao = _prever(pacote, manifesto, nome, validado)
    score, classe = float(previsao.score[0]), str(previsao.classe[0])
    decisao = _decidir(pacote.papeis[nome], validado.dominio["ok"], classe)
    fatores, explicacao = [], EXPLICACAO_TAXA_HISTORICA
    if nome == "L":
        fatores, explicacao = _explicar_regressao_logistica(pacote, manifesto, validado, score)
    return {"occasion_score": score, "classification": classe, "top_factors": fatores,
            "recommendation_category": decisao.recommendation_category,
            "recommendation_reason": decisao.recommendation_reason,
            "model_metadata": _metadados(pacote, manifesto, nome, validado, decisao, previsao, copy.deepcopy(explicacao))}
