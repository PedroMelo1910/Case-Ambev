"""Tabela de hipóteses ponderadas: os fatores de cenário que a POF não observa, com direção, peso e faixa.

A POF 2008–2009 não registra oferta, preço, desconto, canal de pedido, clima nem feriado (as próprias telas e o
`fluxo` já dizem isso). Esses fatores entram como deslocamentos de logit (Δ) aplicados DEPOIS da utilidade aprendida
pela rede, em nós específicos, nunca dentro do encoder. Cada hipótese é registrada aqui antes de rodar, com:

- `no` e `acao`: onde o Δ incide (nó do grafo e ação cuja utilidade ele desloca);
- `direcao`: sinal esperado do efeito ("favorece" / "reduz"), declarado antes de ver resultado;
- `peso`: Δ de logit no valor de referência do fator (0 = cenário de referência, sem efeito);
- `faixa`: intervalo para varredura de sensibilidade;
- `justificativa` e `fonte`: por que o sinal e a ordem de grandeza, com evidência externa quando há.

Separação declarada: um efeito aqui é HIPÓTESE, distinta de uma diferença aprendida nos dados. O desconto zero (e todo
fator ausente) coincide com o cenário de referência: nenhum Δ é aplicado. Variar um peso altera só as transições do nó
e ação que ele declara.

As faixas de desconto usam a elasticidade-preço de refrigerantes como ordem de grandeza (meta-análise de Andreyeva,
Marple et al., JAMA Netw Open 2022: elasticidade média ≈ -1,59 para refrigerantes açucarados). É referência de ordem de
grandeza, não um efeito medido no guaraná nem na aceitação de combo: continua rotulada como hipótese.
"""

from types import MappingProxyType

from personas import grafo

# Perfis de persona conhecidos pelo simulador (os mesmos de `aplicacao.personas`); o nome nunca entra na rede.
PERFIS = ("pai_de_familia", "jovem_sozinho")

# ---------- tabela de hipóteses (registrada antes de rodar) ----------
# Cada entrada: id -> dados da hipótese. `peso` é o Δ de logit no valor de referência do fator (ex.: um combo com
# desconto "cheio"); `por_perfil` permite um peso distinto por perfil quando a hipótese é sobre porção/composição.

HIPOTESES = MappingProxyType({
    "oferta_combo_base": {
        "no": grafo.OFERTA, "acao": "aceita_combo", "direcao": "favorece",
        "peso": 0.0, "faixa": (-0.5, 0.5),
        "descricao": "Ver uma oferta de combo, sem desconto, desloca a aceitação em relação a manter a escolha",
        "justificativa": "Valor de referência da aceitação do combo; sem oferta, o nó de oferta nem é alcançado.",
        "fonte": "hipótese de cenário (sem dado na POF)",
    },
    "desconto_combo": {
        "no": grafo.OFERTA, "acao": "aceita_combo", "direcao": "favorece",
        "peso": 0.0, "faixa": (0.0, 1.2), "por_ponto_percentual": 0.06,
        "descricao": "Desconto no combo aumenta a aceitação do combo",
        "justificativa": "Elasticidade-preço de refrigerantes como ordem de grandeza (meta-análise JAMA 2022, ≈ -1,59); "
                         "um desconto de 5% vira um Δ de logit pequeno e positivo na aceitação. Efeito no guaraná e na "
                         "aceitação de combo é hipótese, não medição.",
        "fonte": "Andreyeva, Marple et al., JAMA Netw Open 2022 (ordem de grandeza)",
    },
    "porcao_para_dividir": {
        "no": grafo.OFERTA, "acao": "aceita_combo", "direcao": "favorece",
        "peso": 0.0, "faixa": (-0.6, 0.6),
        "por_perfil": {"pai_de_familia": 0.35, "jovem_sozinho": -0.25},
        "descricao": "Porção para dividir favorece o combo no perfil família; porção individual favorece no perfil que "
                     "mora sozinho",
        "justificativa": "Hipótese de composição da casa: casa com mais moradores tende ao combo que serve a todos; "
                         "morador único tende à porção individual. É suposição de cenário, não efeito medido.",
        "fonte": "hipótese de cenário (composição da casa observada na POF orienta o sinal)",
    },
})

# ---------- fatores contextuais qualitativos (só a fala; nunca viram número) ----------
# Clima e feriado, quando coletados ao vivo, entram na fala de forma direcional (sobe/desce), sem percentual.
# A direção aqui descreve só o texto; o número do servidor não muda por causa deles.

CONTEXTO_QUALITATIVO = MappingProxyType({
    "chuva": {"direcao": "favorece", "sobre": "pedido por delivery de pizza",
              "frase": "a chuva tende a empurrar o pedido para o delivery, o que costuma favorecer pizza em casa"},
    "frio": {"direcao": "favorece", "sobre": "refeição quente em casa",
             "frase": "o frio tende a favorecer uma refeição quente em casa, como pizza"},
    "calor": {"direcao": "reduz", "sobre": "forno em casa", "contrapeso": "bebida gelada",
              "frase": "o calor tende a desfavorecer ligar o forno, mas puxa a bebida gelada junto"},
    "feriado": {"direcao": "favorece", "sobre": "pedido de pizza em casa",
                "frase": "feriado e véspera tendem a favorecer pedir pizza para comer em casa"},
    "fim_de_semana": {"direcao": "favorece", "sobre": "ocasião pizza + refrigerante",
                      "frase": "de sexta a domingo a ocasião de pizza com refrigerante tende a ser mais frequente"},
})


def delta_do_cenario(cenario: dict, perfil: str | None) -> dict:
    """Soma dos Δ de logit por (nó, ação), a partir do cenário e do perfil. Cenário de referência (nada informado)
    devolve dicionário vazio: nenhum deslocamento. Só os fatores presentes no cenário contribuem."""
    if perfil is not None and perfil not in PERFIS:
        raise ValueError(f"perfil desconhecido: {perfil!r}")
    deltas: dict[tuple[str, str], float] = {}

    def somar(no: str, acao: str, valor: float) -> None:
        if valor:
            deltas[(no, acao)] = deltas.get((no, acao), 0.0) + float(valor)

    if cenario.get("oferta"):
        h = HIPOTESES["oferta_combo_base"]
        somar(h["no"], h["acao"], _peso(h, perfil))
        porcao = HIPOTESES["porcao_para_dividir"]
        somar(porcao["no"], porcao["acao"], _peso(porcao, perfil))
        pct = _percentual_desconto(cenario["oferta"])
        if pct:
            d = HIPOTESES["desconto_combo"]
            somar(d["no"], d["acao"], min(d["faixa"][1], pct * d["por_ponto_percentual"]))
    return {f"{no}|{acao}": v for (no, acao), v in deltas.items()}


def _peso(hipotese: dict, perfil: str | None) -> float:
    porperfil = hipotese.get("por_perfil")
    if porperfil and perfil in porperfil:
        return float(porperfil[perfil])
    return float(hipotese["peso"])


def _percentual_desconto(oferta: object) -> float:
    if isinstance(oferta, dict):
        pct = oferta.get("percentual")
        if isinstance(pct, (int, float)) and pct:
            return float(pct)
    return 0.0


def registro() -> list[dict]:
    """Tabela legível das hipóteses, para o registro de verificação (o que é hipótese e qual o peso/faixa)."""
    linhas = []
    for ident, h in HIPOTESES.items():
        linhas.append({
            "id": ident, "no": h["no"], "acao": h["acao"], "direcao": h["direcao"], "peso": h["peso"],
            "faixa": list(h["faixa"]), "por_perfil": h.get("por_perfil"),
            "por_ponto_percentual": h.get("por_ponto_percentual"),
            "descricao": h["descricao"], "justificativa": h["justificativa"], "fonte": h["fonte"],
            "papel": "hipotese",
        })
    return linhas
