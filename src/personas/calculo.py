"""Calcular: as probabilidades que o servidor publica, com fonte, denominador, suporte e situacao de validacao.

- Guarana entre as refeicoes com pizza (T2): `previsao_consumo.avaliar`, publico "todos"; estimativa exploratoria por
  contexto e a taxa habitual avaliada (referencia).
- Pizza entre as refeicoes com comida (T1): tabela validada de `dados/estimativas.json` (modelo publicado pelo protocolo).
- Pizza e guarana na mesma refeicao: estimativa composta T1 x T2, so se a checagem registrada passou; faixa pelo metodo
  delta na escala log (aproximacao declarada).
Texto e grafico leem a mesma lista `numeros`.
"""

import math

from aplicacao import previsao_consumo, regras
from personas import referencias as ref

FONTE_POF = "POF 2008–2009 (IBGE)"
Z = 1.959963984540054


def pct(valor: float) -> str:
    casas = 2 if 0 < valor < 0.01 else 1  # chances abaixo de 1% (pizza e guarana juntos) nao viram "0,2%" arredondado
    return f"{valor * 100:.{casas}f}%".replace(".", ",")


def suporte_texto(positivos: int) -> str:
    return "menos de 5" if 0 < positivos < 5 else str(positivos)  # convencao de mascaramento: 1 a 4 nao saem


def _numero(ident, rotulo, valor, intervalo, denominador, fonte, origem, status, suporte=None) -> dict:
    return {"id": ident, "rotulo": rotulo, "valor": valor, "intervalo": intervalo, "texto_valor": pct(valor),
            "texto_intervalo": [pct(v) for v in intervalo] if intervalo else None, "unidade": "proporção",
            "denominador": denominador, "fonte": fonte, "origem": origem, "status": status, "suporte": suporte,
            "papel": "calculo"}


def _t2(dia: int, periodo: str, regiao: str, area: str) -> tuple[list[dict], dict]:
    pedido = {"dia_semana": dia, "periodo": periodo, "grande_regiao": regiao, "area": area,
              **{campo: previsao_consumo.TODOS for campo in previsao_consumo.CAMPOS_PUBLICO}}
    r = previsao_consumo.avaliar(pedido)
    c, m = r["comparacao"], r["model_metadata"]
    resumo = previsao_consumo.carregar().resumo
    s = c["suporte"]
    contextual = _numero(
        "n_guarana_dado_pizza", "Guaraná nas refeições com pizza, neste contexto", r["occasion_score"],
        list(c["intervalo_estimativa"]) if c["incerteza_disponivel"] else None, "refeições com pizza",
        f"{FONTE_POF}, estimativa por cenário", {"funcao": "previsao_consumo.avaliar", "modelo": m["modelo_contextual"]},
        "exploratorio" if m["papel_do_modelo_contextual"] == "exploratorio" else "validado",
        {"refeicoes": s["episodios"], "com_guarana": suporte_texto(s["positivos"]),
         "celula": " · ".join(s["campos_considerados"])})
    habitual = _numero(
        "n_guarana_habitual", "Guaraná nas refeições com pizza, média geral", c["referencia"], None,
        "refeições com pizza", f"{FONTE_POF}, média de todas as refeições com pizza",
        {"funcao": "previsao_consumo.avaliar", "campo": "comparacao.referencia", "modelo": m["modelo_selecionado"]},
        "referencia", {"refeicoes": resumo["episodios"], "com_guarana": str(resumo["positivos"])})
    leitura = {"classificacao": c["faixa"], "texto_classificacao": regras.FAIXAS_COMPARACAO[c["faixa"]],
               "categoria": r["recommendation_category"], "texto_categoria": regras.CATEGORIAS[r["recommendation_category"]],
               "diferenca_pp": c["diferenca_pp"], "intervalo_pp": c["intervalo_pp"],
               "cenario": r["leitura"], "papel": m["papel_do_modelo_contextual"],
               # fatores do mesmo modelo que gera o numero exibido (vazio se for taxa constante)
               "fatores": [{"campo": f["campo"], "direcao": f["direcao"], "texto": f["texto"],
                            "sujeito": regras.sujeito_do_fator(f)} for f in regras.fatores_exibidos(r["top_factors"])]}
    return [contextual, habitual], leitura


def _t1(dia: int, periodo: str, regiao: str, area: str) -> dict | None:
    t1 = ref.carregar("estimativas.json")["t1"]
    celula = t1["contextos"].get(f"{dia}|{periodo}|{regiao}|{area}")
    if celula is None:
        return None
    publicado = t1["validacao"]["publicado"]
    status = "validado" if publicado in ("L1", "N1") else "referencia"
    numero = _numero("n_pizza", "Pizza numa refeição com comida, neste contexto", celula["estimativa"],
                     celula["intervalo"], "refeições com comida (sem as só de bebida ou adição)",
                     f"{FONTE_POF}, modelo de pizza por cenário",
                     {"tabela": t1["versao"], "modelo": publicado, "validacao": t1["validacao"]["rotulo"]}, status,
                     {"refeicoes": celula["refeicoes"], "com_pizza": suporte_texto(celula["com_pizza"])})
    return {**numero, "_dp_log": celula["dp_log"]}


def _composta(pizza: dict, guarana: dict) -> dict | None:
    composta = ref.carregar("estimativas.json")["composta"]
    if not composta["passa"] or guarana["intervalo"] is None:
        return None
    valor = pizza["valor"] * guarana["valor"]
    baixo, alto = guarana["intervalo"]
    dp = math.sqrt(pizza["_dp_log"] ** 2 + ((math.log(alto) - math.log(baixo)) / (2 * Z)) ** 2)
    intervalo = [valor * math.exp(-Z * dp), valor * math.exp(Z * dp)]
    return _numero("n_pizza_e_guarana", "Pizza e guaraná na mesma refeição (estimativa composta)", valor, intervalo,
                   "refeições com comida", f"{FONTE_POF}, pizza multiplicada por guaraná com pizza",
                   {"composicao": ["n_pizza", "n_guarana_dado_pizza"], "checagem": "observado/esperado "
                    f"{composta['o_e']:.3f} em {composta['positivos']} refeições com os dois"}, "estimativa_composta")


def calcular(ctx: dict) -> dict:
    dia, periodo = ctx["dia_semana"]["valor"], ctx["periodo"]["valor"]
    regiao, area = ctx["grande_regiao"]["valor"], ctx["area"]["valor"]
    numeros, leitura = _t2(dia, periodo, regiao, area)
    pizza = _t1(dia, periodo, regiao, area)
    if pizza is not None:
        composta = _composta(pizza, numeros[0])
        numeros.append({k: v for k, v in pizza.items() if k != "_dp_log"})
        if composta is not None:
            numeros.append(composta)
    ids = [n["id"] for n in numeros]
    complementares = [i for i in ("n_pizza", "n_pizza_e_guarana") if i in ids]
    grupos = [{"papel": "principal", "titulo": "Mesma base: guaraná entre refeições com pizza",
               "series": ["n_guarana_dado_pizza", "n_guarana_habitual"]}]
    if complementares:
        grupos.append({"papel": "complementar", "titulo": "Complementar, outra base: entre refeições com comida",
                       "series": complementares})
    grafico = {"tipo": "barras_com_intervalo", "titulo": "Probabilidades calculadas pelo servidor", "series": ids,
               "grupos": grupos, "referencia_de": {"n_guarana_dado_pizza": "n_guarana_habitual"},
               "rotulos_curtos": {"n_guarana_dado_pizza": "Neste cenário", "n_guarana_habitual": "Referência histórica",
                                  "n_pizza": "Pizza", "n_pizza_e_guarana": "Pizza e guaraná juntos"},
               "faixa_aproximada": ["n_pizza_e_guarana"],
               "rodape": "Consumo registrado na POF 2008–2009, sem pesos. Não é compra, marca nem 2026."}
    return {"numeros": numeros, "grafico": grafico, "leitura": leitura}
