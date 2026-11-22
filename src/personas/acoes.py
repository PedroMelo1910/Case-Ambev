"""Decidir: a categoria do cenario (motor de regras de `aplicacao.regras`) e a sugestao adicional de teste.

Os dois papeis ficam separados no contrato: `categoria` vem de `previsao_consumo.avaliar` (a mesma de Ocasioes) e
`acao` e uma sugestao de como medir o assunto da conversa (desconto, clima, feriado, territorio). A sugestao pode
aparecer mesmo num cenario sem prioridade demonstrada. Todo tamanho de amostra sai da formula abaixo, com as premissas
declaradas junto do numero.
"""

import math
from statistics import NormalDist

from aplicacao import regras

DESVIO_EXEMPLO = 8.0  # R$ por cliente elegivel: premissa de exemplo, a confirmar com dados do parceiro
DIFERENCA_EXEMPLO = 1.0  # R$: menor diferenca de margem que interessaria
CONFIANCA = 0.95
PODER = 0.80

ACOES = {
    "analise_clima_pedidos": {
        "titulo": "Medir a associação entre chuva e pedidos com dados datados",
        "passos": ["Juntar pedidos com data, hora e loja à chuva observada e à previsão emitida antes.",
                   "Comparar dias com e sem chuva no mesmo dia da semana e período.",
                   "Ler como associação: o clima não pode ser sorteado."],
        "dados_necessarios": ["pedidos com data, hora e loja", "chuva observada e prevista por hora"]},
    "comparar_feriado": {
        "titulo": "Comparar feriados com dias comuns em pedidos datados",
        "passos": ["Separar pedidos de feriados e de dias comuns do mesmo dia da semana.",
                   "Comparar a participação de pizza com guaraná e o ticket.", "Declarar os feriados antes de olhar."],
        "dados_necessarios": ["pedidos com data e loja", "calendário oficial do município"]},
    "teste_territorial": {
        "titulo": "Comparar marcas por renda da área com pedidos de entrega",
        "passos": ["Agregar pedidos com refrigerante por distrito ou setor de entrega.",
                   "Comparar a participação de cada marca entre quintis de renda da área.",
                   "Controlar cardápio, preço, plataforma e horário."],
        "dados_necessarios": ["pedidos com CEP ou setor de entrega", "produto com marca", "preço e disponibilidade"]},
}


def amostra_por_grupo(desvio: float, diferenca: float, confianca: float = CONFIANCA, poder: float = PODER) -> int:
    """Clientes por grupo para comparar duas medias: 2 (z_a + z_b)^2 desvio^2 / diferenca^2, arredondado para cima."""
    if desvio <= 0 or diferenca <= 0 or not 0 < confianca < 1 or not 0 < poder < 1:
        raise ValueError("desvio, diferença, confiança e poder precisam ser positivos (confiança e poder abaixo de 1)")
    z = NormalDist().inv_cdf(1 - (1 - confianca) / 2) + NormalDist().inv_cdf(poder)
    return math.ceil(2 * z ** 2 * desvio ** 2 / diferenca ** 2)


def _numero(valor: float) -> str:
    return f"{valor:,.0f}".replace(",", ".")


def _percentual(oferta: dict) -> str | None:
    p = oferta.get("percentual")
    return f"{p:g}%".replace(".", ",") if p else None


def teste_do_desconto(oferta: dict) -> dict:
    """Comparacao entre oferta e preco cheio, com a metrica alinhada a unidade sorteada."""
    pct = _percentual(oferta)
    desconto = f"com {pct} de desconto" if pct else "com desconto"
    n = amostra_por_grupo(DESVIO_EXEMPLO, DIFERENCA_EXEMPLO)
    amostra = {"por_grupo": n, "desvio": DESVIO_EXEMPLO, "diferenca": DIFERENCA_EXEMPLO, "confianca": CONFIANCA,
               "poder": PODER, "unidade": "clientes elegíveis",
               "texto": (f"Tamanho de exemplo, sob premissas a confirmar com dados do parceiro: com desvio-padrão de "
                         f"R$ {DESVIO_EXEMPLO:.0f} na margem por cliente elegível, menor diferença que interessa de "
                         f"R$ {DIFERENCA_EXEMPLO:.0f}, {CONFIANCA:.0%} de confiança e {PODER:.0%} de poder, cerca de "
                         f"{_numero(n)} clientes elegíveis por grupo.")}
    detalhe = {
        "titulo": "Vale testar o combo com um parceiro",
        "comparacao": (f"Combo pizza e guaraná {desconto} contra o mesmo combo a preço cheio, no mesmo parceiro e no "
                       "mesmo período."),
        "sorteio": "Sortear clientes elegíveis: metade vê o combo com desconto; a outra metade, o preço cheio.",
        "metrica_principal": ("Margem por cliente elegível: a margem do combo (preço menos custo, desconto, taxas e "
                              "frete) somada no grupo e dividida por todos os clientes sorteados, inclusive quem viu a "
                              "oferta e não comprou."),
        "metrica_secundaria": "Adesão ao combo: parcela dos clientes elegíveis que compraram o combo.",
        "protecao": "Acompanhar também ticket médio, cancelamentos e reclamações.",
        "sorteio_por_loja": ("Se o sorteio for por loja, a análise compara lojas (a margem por cliente elegível de cada "
                             "loja), e o tamanho da amostra depende do número de lojas e da variação entre elas, não "
                             "do número de pedidos."),
        "duracao": "Pelo menos 4 semanas, sem campanhas simultâneas e com os feriados declarados antes.",
        "amostra": amostra,
        "dados_necessarios": ["clientes elegíveis e grupo sorteado", "exposição à oferta", "pedidos com itens",
                              "preço, desconto, custo, taxas e frete", "status do pedido"]}
    detalhe["passos"] = [detalhe["comparacao"], detalhe["sorteio"], detalhe["metrica_principal"],
                         detalhe["metrica_secundaria"], detalhe["sorteio_por_loja"], amostra["texto"], detalhe["duracao"]]
    return detalhe


def _acao(ctx: dict, temas: frozenset) -> str | None:
    if "oferta" in ctx:
        return "teste_ab_desconto"
    if "clima" in ctx:
        return "analise_clima_pedidos"
    if ctx.get("feriado") and ctx["feriado"]["estado"] != "confirmado":
        return "comparar_feriado"
    if "distrito" in ctx or {"renda", "marca"} & temas:
        return "teste_territorial"
    return None


def decidir(aba: str, ctx: dict, leitura: dict | None, temas: frozenset) -> dict:
    base = {}
    if leitura:
        base = {"categoria": leitura["categoria"], "texto_categoria": leitura["texto_categoria"],
                "leitura_da_categoria": regras.LEITURA_DA_CATEGORIA[leitura["categoria"]],
                "papel_da_categoria": "motor_de_regras"}
    if aba != "recomendacoes":
        return {**base, "acao": None}
    acao = _acao(ctx, temas)
    detalhe = {} if acao is None else (teste_do_desconto(ctx["oferta"]["valor"]) if acao == "teste_ab_desconto"
                                       else dict(ACOES[acao]))
    return {**base, "regra": "acao_testavel_por_assunto", "versao": "2", "acao": acao,
            "papel_da_acao": "sugestao_adicional" if acao else None, **detalhe}
