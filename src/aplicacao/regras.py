"""Motor de regras do Decision Engine: categoria de ativacao a partir de quatro fatos ja calculados.

Funcao pura, so biblioteca padrao. Nao calcula score, nao le arquivo e nao decide dominio: recebe `dominio_ok` pronto
(True verificado nas marginais, False fora do dominio, None desconhecido). O papel do modelo chega em dois fatos que a
aplicacao controla, nunca o usuario: `ordena_contextos` (capacidade de distinguir contextos) e `uso_exploratorio` (o
modelo nao e o selecionado e so e mostrado como analise exploratoria). Primeira regra cuja condicao vale define a saida;
as condicoes podem coincidir, e a ordem da tabela e parte da especificacao. A recomendacao e uma sugestao por regra: nao
estima retorno, nao afirma causalidade, nao identifica marca nem canal. Regras definidas depois da avaliacao temporal de
2017, e por isso a versao carrega esse sinal.
"""

import reprlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass

VERSAO_REGRAS = "2-pos-2017"
CLASSES = ("alta", "media", "baixa")
CAMPOS = ("ordena_contextos", "uso_exploratorio", "dominio_ok", "classification")

MAXIMO_EXTRAS_NA_MENSAGEM = 10
MAXIMO_BITS_NA_MENSAGEM = 64

_LIMITES_MARGINAIS ="Limites marginais válidos não garantem suporte suficiente para esta combinação de atributos."


class EntradaInvalida(ValueError):
    """Campo obrigatorio ausente, campo desconhecido ou valor de tipo ou dominio invalido. Nunca vira decisao."""


@dataclass(frozen=True)
class Regra:
    id: str
    categoria: str
    justificativa: str
    aplica: Callable[[Mapping], bool]


@dataclass(frozen=True)
class Decisao:
    rule_id: str
    recommendation_category: str
    recommendation_reason: str
    rules_version: str


# `dominio_ok` None (desconhecido) nao e False (invalido): cada um tem a sua regra, com a mesma categoria.
REGRAS = (
    Regra("R1a", "evidencia_insuficiente",
          "O contexto informado está fora do domínio verificado em 2008; "
          "o score, se exibido, não sustenta recomendação.",
          lambda e: e["dominio_ok"] is False),
    Regra("R1b", "evidencia_insuficiente",
          "O domínio do contexto informado não pôde ser verificado (há informação desconhecida); "
          "o score, se exibido, não sustenta recomendação.",
          lambda e: e["dominio_ok"] is None),
    Regra("R2", "sem_diferenciacao_contextual",
          "O modelo em uso não distingue contextos: seu score é o mesmo para qualquer contexto "
          "(no modelo de taxa histórica, a taxa média geral de 2008) e não é uma previsão para este contexto. "
          "Nenhuma ativação é derivada do contexto informado.",
          lambda e: not e["ordena_contextos"]),
    Regra("R3", "destaque_combo_pizza_guarana",
          "Classe alta em modelo contextual de uso exploratório: a regressão logística não é o modelo selecionado. "
          "Sugestão exploratória por regra: destacar a combinação pizza + guaraná ou um combo, sem estimativa de "
          "retorno e sem relação causal demonstrada. " + _LIMITES_MARGINAIS,
          lambda e: e["classification"] == "alta"),
    Regra("R4", "sem_prioridade_demonstrada",
          "Classe média ou baixa em modelo contextual de uso exploratório: a regressão logística não é o modelo selecionado, "
          "a regra não prioriza essas faixas e nenhum destaque foi demonstrado para este contexto. "
          "Isso não indica ausência de oportunidade. " + _LIMITES_MARGINAIS,
          lambda e: e["classification"] in ("media", "baixa")),
)


def recebido(valor: object) -> str:
    """Descricao curta e segura do valor recebido: repr truncado para tipos simples, so o nome do tipo para o resto.
    Inteiros enormes (o repr de mais de 4300 digitos levanta ValueError) viram um rotulo fixo."""
    if type(valor) is int and valor.bit_length() > MAXIMO_BITS_NA_MENSAGEM:
        return "<int grande>"
    if type(valor) in (bool, int, float, str, type(None)):
        return reprlib.repr(valor)
    return f"<{type(valor).__name__}>"


def conferir_campos(mapa: Mapping, campos: tuple[str, ...], rotulo: str, dica: str = "") -> None:
    """Exige exatamente `campos`: ausentes e extras viram EntradaInvalida (a lista de extras e cortada na mensagem)."""
    ausentes = [campo for campo in campos if campo not in mapa]
    if ausentes:
        raise EntradaInvalida(f"campos obrigatorios ausentes: {ausentes}")
    extras = sorted(recebido(campo) for campo in mapa if campo not in campos)
    if extras:
        resto = len(extras) - MAXIMO_EXTRAS_NA_MENSAGEM
        sufixo = f" (e mais {resto})" if resto > 0 else ""
        raise EntradaInvalida(f"campos que o {rotulo} nao aceita{dica}: {extras[:MAXIMO_EXTRAS_NA_MENSAGEM]}{sufixo}")


def validar(entrada: Mapping) -> dict:
    """Confere campos obrigatorios e tipos; devolve so os quatro fatos, ou levanta EntradaInvalida."""
    if not isinstance(entrada, Mapping):
        raise EntradaInvalida(f"entrada deve ser um dicionario com os campos {list(CAMPOS)}")
    conferir_campos(entrada, CAMPOS, "motor")
    for campo in ("ordena_contextos", "uso_exploratorio"):
        if not isinstance(entrada[campo], bool):
            raise EntradaInvalida(f"{campo} deve ser booleano, recebido {recebido(entrada[campo])}")
    if entrada["ordena_contextos"] and not entrada["uso_exploratorio"]:
        raise EntradaInvalida("modelo contextual sem uso_exploratorio nao tem regra definida: "
                              "so a logistica exploratoria ordena contextos nesta versao")
    dominio = entrada["dominio_ok"]
    if dominio is not None and not isinstance(dominio, bool):
        raise EntradaInvalida(f"dominio_ok deve ser booleano ou None, recebido {recebido(dominio)}")
    classe = entrada["classification"]
    if not isinstance(classe, str) or classe not in CLASSES:
        raise EntradaInvalida(f"classification deve ser uma de {list(CLASSES)}, recebido {recebido(classe)}")
    return {campo: entrada[campo] for campo in CAMPOS}


def decidir(entrada: Mapping) -> Decisao:
    """Categoria e justificativa da primeira regra que se aplica."""
    fatos = validar(entrada)
    for regra in REGRAS:
        if regra.aplica(fatos):
            return Decisao(regra.id, regra.categoria, regra.justificativa, VERSAO_REGRAS)
    raise RuntimeError(f"tabela de regras incompleta: nenhuma regra se aplica a {fatos}")
