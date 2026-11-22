"""Afirmacoes quantitativas montadas pelo servidor: medida, denominador, valor, faixa, leitura e avisos.

Toda frase com numero da conversa sai daqui (ou do restante de `redacao`, tambem deterministico). O LLM nunca escreve
essas frases: recebe-as so para saber o que ja foi dito. A comparacao principal tem a mesma base nos dois lados (guarana
entre refeicoes com pizza); pizza entre refeicoes com comida e a estimativa conjunta tem outra base e ficam
identificadas como complementares.
"""

from personas import referencias as ref

MEDIDA = "Guaraná em refeições com pizza"
AVISO_EXPLORATORIO = "Estimativa exploratória: nenhum modelo por cenário superou a referência com segurança."
AVISO_FONTE = "Consumo registrado em 2008–2009: não mede pedidos, vendas nem marca."


def _faixa(n: dict, aproximada: bool = False) -> str:
    if not n["texto_intervalo"]:
        return ""
    return f" (faixa {'aproximada ' if aproximada else ''}de {n['texto_intervalo'][0]} a {n['texto_intervalo'][1]})"


def local_informado(ctx: dict) -> str | None:
    if "municipio" not in ctx:
        return None
    m = ref.municipio(ctx["municipio"]["valor"])
    texto = f"{m['nome']} ({'capital' if m['capital'] else m['uf']})"
    return f"{texto}, distrito {ctx['distrito']['nome']}" if "distrito" in ctx else texto


def leitura_do_cenario(ctx: dict, leitura: dict) -> dict:
    """Leitura de Ocasioes (mesma funcao do servidor) com a diferenca entre o local informado e o recorte usado."""
    cenario = dict(leitura["cenario"])
    informado = local_informado(ctx)
    estimativa = cenario.pop("recorte")
    texto = None
    if informado:
        nem_bairro = " nem bairro" if "distrito" in ctx else ""
        texto = (f"Você falou de {informado}; a estimativa usa o recorte {estimativa}, porque a pesquisa não identifica "
                 f"cidade{nem_bairro}.")
    return {**cenario, "recorte": {"informado": informado, "estimativa": estimativa, "texto": texto}}


def montar(numeros: list[dict], leitura_cenario: dict) -> list[dict]:
    n = {x["id"]: x for x in numeros}
    if not n:
        return []
    g, hab = n["n_guarana_dado_pizza"], n["n_guarana_habitual"]
    exploratoria = g["status"] == "exploratorio"
    avisos = ([AVISO_EXPLORATORIO] if exploratoria else []) + [AVISO_FONTE]
    saida = [{"id": "a_guarana", "papel": "principal", "medida": MEDIDA, "denominador": g["denominador"],
              "numeros": ["n_guarana_dado_pizza", "n_guarana_habitual"], "valor": g["texto_valor"],
              "referencia": hab["texto_valor"], "faixa": g["texto_intervalo"], "leitura": leitura_cenario["titulo"],
              "avisos": avisos,
              "texto": (f"{MEDIDA}: {g['texto_valor']} neste cenário{_faixa(g)} e {hab['texto_valor']} na referência "
                        f"histórica. {leitura_cenario['explicacao']}"
                        + (" É uma estimativa exploratória." if exploratoria else ""))}]
    if "n_pizza" in n:
        p = n["n_pizza"]
        saida.append({"id": "a_pizza", "papel": "complementar", "medida": "Pizza em refeições com comida",
                      "denominador": "refeições com comida", "numeros": ["n_pizza"], "valor": p["texto_valor"],
                      "referencia": None, "faixa": p["texto_intervalo"], "leitura": "modelo validado",
                      "avisos": ["Outra base: não se compara com a linha do guaraná."],
                      "texto": (f"Complementar, com outra base: pizza aparece em {p['texto_valor']} das refeições com "
                                f"comida neste cenário{_faixa(p)}, pelo modelo validado.")})
    if "n_pizza_e_guarana" in n:
        c = n["n_pizza_e_guarana"]
        saida.append({"id": "a_pizza_e_guarana", "papel": "complementar", "medida": "Pizza e guaraná na mesma refeição",
                      "denominador": "refeições com comida", "numeros": ["n_pizza_e_guarana"],
                      "valor": c["texto_valor"], "referencia": None, "faixa": c["texto_intervalo"],
                      "leitura": "estimativa composta", "avisos": ["Faixa aproximada.", AVISO_EXPLORATORIO],
                      "texto": (f"Pizza e guaraná juntos ficam em {c['texto_valor']} das refeições com comida"
                                f"{_faixa(c, aproximada=True)}. É uma estimativa composta e exploratória, porque "
                                "depende da parte do guaraná.")})
    return saida


def frase_dos_fatores(fatores: list[dict], exploratoria: bool) -> str | None:
    """Fatores do modelo exibido numa frase; nenhum quando a estimativa e uma taxa constante."""
    if not fatores:
        return None
    partes = [f"{f['sujeito']} ({'eleva' if f['direcao'] == 'eleva' else 'reduz'})" for f in fatores]
    lista = partes[0] if len(partes) == 1 else ", ".join(partes[:-1]) + " e " + partes[-1]
    modelo = "no modelo exploratório" if exploratoria else "no modelo"
    return (f"O que pesou na estimativa, {modelo}: {lista}, cada um comparado com a média do seu campo. É associação "
            "no modelo, não efeito.")
