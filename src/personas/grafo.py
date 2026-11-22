"""Grafo de decisão da jornada (comida → bebida dado pizza → oferta), com máscara de ações válidas e softmax.

O grafo guia o caminho; a rede (`rede`) calcula as chances nos nós com dado; as hipóteses (`hipoteses`) deslocam os
logits nos nós sem dado e nos fatores de contexto; o simulador (`simulador`) sorteia as escolhas. Aqui ficam só a
estrutura (nós, ações, transições) e a álgebra de probabilidade: utilidades → máscara → softmax normalizado, somando
1 sobre as ações válidas. Nada neste módulo lê arquivo, treina nem sorteia.

Três nós observáveis, na ordem da jornada:

- `comida`: pizza ou outra refeição. Denominador = refeições com comida (recorte S1 da base). Rótulo observado na POF.
- `bebida`: guaraná, outro refrigerante, outra bebida ou nenhuma, dado que a refeição é pizza. Cabeça hierárquica
  (refri? → guaraná|refri), semântica (G) de máscara: zera a folha inválida e renormaliza sobre as válidas, a mesma
  lei do softmax mascarado (ARQUITETURA_REDE_NEURAL §3.3). Rótulo observado só aqui.
- `oferta`: aceita o combo, mantém a escolha ou desiste. Só entra quando o cenário informa uma oferta. Sem rótulo na
  POF: modo prior (hipótese de cenário), nunca aprendido.

Um nó só é avaliado quando alcançável: `bebida` exige pizza no nó `comida`; `oferta` exige que o cenário tenha oferta.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType

import numpy as np

# ---------- nós, ações e folhas ----------

COMIDA = "comida"
BEBIDA = "bebida"
OFERTA = "oferta"
NOS = (COMIDA, BEBIDA, OFERTA)

ACOES = MappingProxyType({
    COMIDA: ("pizza", "outra_refeicao"),
    # 'outra bebida' e 'nenhuma' estão agrupadas em 'sem_refrigerante': a POF 2008 não registrou água e a base não
    # distingue uma bebida não-refrigerante de nenhuma bebida. Agrupadas até haver rótulos observados (ver treino).
    BEBIDA: ("guarana", "outro_refrigerante", "sem_refrigerante"),
    OFERTA: ("aceita_combo", "mantem_escolha", "desiste"),
})

# Rótulos em linguagem comum das ações (texto exibível; nunca vira número).
ROTULOS = MappingProxyType({
    "pizza": "pizza", "outra_refeicao": "outra refeição",
    "guarana": "guaraná", "outro_refrigerante": "outro refrigerante",
    "sem_refrigerante": "outra bebida ou nenhuma (sem refrigerante)",
    "aceita_combo": "aceita o combo", "mantem_escolha": "mantém a escolha", "desiste": "desiste do pedido",
})

# Árvore da cabeça de bebida: refri = {guaraná, outro refrigerante}; não-refri = {sem refrigerante}.
# Cada ramo guarda (ações do ramo); a folha escolhida é P(ramo) · P(folha | ramo).
RAMOS_BEBIDA = MappingProxyType({
    "refrigerante": ("guarana", "outro_refrigerante"),
    "nao_refrigerante": ("sem_refrigerante",),
})

# Nós cujas probabilidades vêm da rede (têm rótulo observado na POF). `oferta` não está aqui: é modo prior.
NOS_APRENDIDOS = (COMIDA, BEBIDA)


def acoes_do_no(no: str) -> tuple[str, ...]:
    if no not in ACOES:
        raise ValueError(f"nó desconhecido: {no!r}")
    return ACOES[no]


# ---------- máscara e softmax ----------

def _softmax_mascarado(logits: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """Softmax só sobre as ações válidas (máscara 1). Ações inválidas recebem probabilidade exatamente 0 e o restante
    é renormalizado entre as válidas. Erro se nenhuma ação é válida: o grafo nunca escolhe uma ação por omissão."""
    logits = np.asarray(logits, dtype=float)
    mascara = np.asarray(mascara, dtype=bool)
    if logits.shape != mascara.shape or logits.ndim != 1:
        raise ValueError("logits e máscara devem ser vetores de mesmo tamanho")
    if not mascara.any():
        raise ValueError("nenhuma ação válida no nó: o grafo não escolhe ação por omissão")
    seguro = np.where(mascara, logits, -np.inf)
    seguro = seguro - np.max(seguro[mascara])  # estabilidade numérica
    expo = np.where(mascara, np.exp(seguro), 0.0)
    return expo / expo.sum()


def probabilidades(no: str, utilidades: Mapping[str, float], validas: Sequence[str] | None = None) -> dict:
    """Distribuição do nó: utilidade por ação → máscara de ações válidas → softmax normalizado (soma 1).

    `utilidades` traz uma utilidade (logit) por ação do nó. `validas`, quando dado, restringe às ações possíveis no
    momento (as demais ficam com probabilidade 0). O nó `bebida` usa a árvore hierárquica; os demais, softmax plano.
    """
    acoes = acoes_do_no(no)
    faltando = [a for a in acoes if a not in utilidades]
    if faltando:
        raise ValueError(f"utilidade ausente para {faltando} no nó {no}")
    disponiveis = tuple(acoes if validas is None else validas)
    desconhecidas = [a for a in disponiveis if a not in acoes]
    if desconhecidas:
        raise ValueError(f"ação válida fora do nó {no}: {desconhecidas}")
    mascara = np.array([a in disponiveis for a in acoes], dtype=bool)
    if no == BEBIDA:
        probs = _bebida_hierarquica(utilidades, mascara)
    else:
        probs = _softmax_mascarado(np.array([utilidades[a] for a in acoes]), mascara)
    return {acao: float(p) for acao, p in zip(acoes, probs)}


def _bebida_hierarquica(utilidades: Mapping[str, float], mascara: np.ndarray) -> np.ndarray:
    """Cabeça hierárquica de bebida, semântica (G): calcula P(folha) pela árvore completa e, se houver folha inválida,
    renormaliza sobre as válidas (equivale ao softmax mascarado sobre as folhas). P(ramo) por softmax entre os dois
    ramos, usando a utilidade máxima mascarada de cada ramo como representante; P(folha|ramo) por softmax dentro do
    ramo. Com máscara toda 1 (caso da POF), reduz a um softmax plano sobre as quatro folhas."""
    acoes = ACOES[BEBIDA]
    indice = {a: i for i, a in enumerate(acoes)}
    logits = np.array([utilidades[a] for a in acoes], dtype=float)
    cheia = _softmax_plano_arvore(logits, indice)
    valida = cheia * mascara
    total = valida.sum()
    if total <= 0:
        raise ValueError("nenhuma folha de bebida válida")
    return valida / total


def _softmax_plano_arvore(logits: np.ndarray, indice: Mapping[str, int]) -> np.ndarray:
    """P(folha) = P(ramo) · P(folha | ramo), sem máscara. P(ramo) via softmax entre log-soma-exp dos ramos (o que
    torna a fatoração idêntica ao softmax plano sobre as quatro folhas); mantida explícita para a leitura hierárquica
    e para a análise de sensibilidade (L) poder divergir dela se um dia for ativada."""
    probs = np.zeros(len(logits))
    valor_ramo = {}
    for ramo, folhas in RAMOS_BEBIDA.items():
        ls = np.array([logits[indice[f]] for f in folhas])
        valor_ramo[ramo] = _logsumexp(ls)
    ramos = list(RAMOS_BEBIDA)
    lv = np.array([valor_ramo[r] for r in ramos])
    p_ramo = np.exp(lv - _logsumexp(lv))
    for r, pr in zip(ramos, p_ramo):
        folhas = RAMOS_BEBIDA[r]
        ls = np.array([logits[indice[f]] for f in folhas])
        p_folha = np.exp(ls - _logsumexp(ls))
        for f, pf in zip(folhas, p_folha):
            probs[indice[f]] = pr * pf
    return probs


def _logsumexp(valores: np.ndarray) -> float:
    m = np.max(valores)
    return float(m + np.log(np.sum(np.exp(valores - m))))


# ---------- desfechos finais da jornada ----------
# Uma jornada termina num destes desfechos; o simulador agrega as frequências deles.

DESFECHOS = (
    "pizza_com_guarana", "pizza_outro_refrigerante", "pizza_sem_refrigerante",
    "outra_refeicao", "desistiu",
)

ROTULOS_DESFECHO = MappingProxyType({
    "pizza_com_guarana": "pizza com guaraná", "pizza_outro_refrigerante": "pizza com outro refrigerante",
    "pizza_sem_refrigerante": "pizza com outra bebida ou sem refrigerante",
    "outra_refeicao": "outra refeição", "desistiu": "desistiu do pedido",
})

_BEBIDA_PARA_DESFECHO = MappingProxyType({
    "guarana": "pizza_com_guarana", "outro_refrigerante": "pizza_outro_refrigerante",
    "sem_refrigerante": "pizza_sem_refrigerante",
})


def desfecho(comida: str, bebida: str | None, oferta: str | None) -> str:
    """Desfecho final a partir das ações escolhidas nos nós percorridos. Transições impossíveis são erro."""
    if comida == "outra_refeicao":
        return "outra_refeicao"
    if comida != "pizza":
        raise ValueError(f"ação de comida inválida: {comida!r}")
    if oferta == "desiste":
        return "desistiu"
    if bebida not in _BEBIDA_PARA_DESFECHO:
        raise ValueError(f"ação de bebida inválida para pizza: {bebida!r}")
    return _BEBIDA_PARA_DESFECHO[bebida]
