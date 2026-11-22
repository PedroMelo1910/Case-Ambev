"""Simulador de jornadas: percorre o grafo sorteando ações e agrega os percentuais que o chat exibe.

Para um cenário (contexto + perfil + oferta), o simulador: (1) monta o registro de features; (2) pega as utilidades
dos nós com dado na rede treinada (`rede`); (3) soma os Δ das hipóteses de cenário (`hipoteses`) nas utilidades, por
nó e ação; (4) em cada nó, aplica a máscara e o softmax do grafo (`grafo.probabilidades`); (5) sorteia a ação com
`numpy.default_rng(semente)`; (6) repete por N jornadas e agrega as frequências dos desfechos.

O grafo guia o caminho, a rede dá as chances nos nós com dado, as hipóteses deslocam os nós sem dado, o simulador
sorteia. Contrato de saída (quando aplicável):

- `pizza`: chance de pizza entre todas as jornadas;
- `guarana_dado_pizza`: chance de guaraná entre as jornadas que escolheram pizza;
- `pizza_e_guarana`: chance de terminar com pizza e guaraná entre todas as jornadas;
- `aceita_combo_dado_oferta`: chance de aceitar o combo entre as jornadas que viram a oferta.

Comparar cenários (ex.: sem desconto × 5%) usa a MESMA população simulada: muda só a oferta, mantendo dia, lugar,
período, perfil e a semente. Os percentuais de probabilidade do grafo são calculáveis em forma fechada por jornada
(sem sorteio); o sorteio serve para testar o sistema e medir a estabilidade de Monte Carlo.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from personas import grafo, hipoteses, rede

SEMENTE_PADRAO = 20261122
JORNADAS_PADRAO = 20000


@dataclass(frozen=True)
class Distribuicoes:
    """Probabilidades dos nós para um cenário, em forma fechada (sem sorteio). `oferta` é None se o cenário não a tem.
    `bebida` é a distribuição USADA nos números (histórica por padrão); `bebida_rede` é a da rede, como experimento."""
    comida: dict
    bebida: dict
    oferta: dict | None
    bebida_rede: dict | None = None


def _bebida_historica(preditor: rede.Preditor) -> dict:
    """Distribuição de bebida pela taxa histórica observada (POF 2008), a referência do chat. A rede de bebida não
    supera essa referência (ver avaliacao.json), então o número confiável de guaraná vem daqui."""
    hist = preditor.metadados.get("bebida_historica")
    if not hist:  # artefato antigo sem as contagens: usa só a taxa de guaraná
        taxa_g = preditor.taxa_historica_guarana()
        return {"guarana": taxa_g, "outro_refrigerante": 0.0, "sem_refrigerante": 1.0 - taxa_g}
    total = sum(hist[a] for a in grafo.ACOES[grafo.BEBIDA])
    return {a: hist[a] / total for a in grafo.ACOES[grafo.BEBIDA]}


def distribuicoes(preditor: rede.Preditor, registro: Mapping, cenario: Mapping, perfil: str | None,
                  fonte_bebida: str = "historica") -> Distribuicoes:
    """Distribuições dos nós. A comida vem da rede (lá ela supera a referência). A bebida vem, por padrão, da TAXA
    HISTÓRICA (`fonte_bebida="historica"`): a rede de bebida é experimento e não supera a referência. A oferta vem das
    hipóteses de cenário. Os Δ das hipóteses são somados às utilidades antes da máscara/softmax do grafo."""
    utilidades = preditor.utilidades(registro)
    deltas = hipoteses.delta_do_cenario(dict(cenario), perfil)

    def com_delta(no: str, base: dict) -> dict:
        return {acao: base[acao] + deltas.get(f"{no}|{acao}", 0.0) for acao in base}

    comida = grafo.probabilidades(grafo.COMIDA, com_delta(grafo.COMIDA, utilidades[grafo.COMIDA]))
    bebida_rede = grafo.probabilidades(grafo.BEBIDA, utilidades[grafo.BEBIDA])  # experimento, sempre calculado
    bebida = _bebida_historica(preditor) if fonte_bebida == "historica" else bebida_rede
    oferta = None
    if cenario.get("oferta"):
        base = {acao: 0.0 for acao in grafo.ACOES[grafo.OFERTA]}  # modo prior: parte de utilidade zero
        oferta = grafo.probabilidades(grafo.OFERTA, com_delta(grafo.OFERTA, base))
    return Distribuicoes(comida=comida, bebida=bebida, oferta=oferta, bebida_rede=bebida_rede)


def _fechado(d: Distribuicoes) -> dict:
    """Percentuais em forma fechada, definidos sobre os DESFECHOS FINAIS da jornada e alinhados ao sorteio do grafo.

    Com oferta, uma jornada que escolhe pizza pode desistir e então não termina com pizza. Os eventos são:
    - `pizza`: terminar com pizza (qualquer bebida) entre todas as jornadas = P(pizza)·P(não desiste|oferta);
    - `guarana_dado_pizza`: guaraná entre quem TERMINOU com pizza = P(guaraná|pizza) (a desistência, independente da
      bebida no grafo, cancela nessa razão);
    - `pizza_e_guarana`: terminar com pizza e guaraná entre todas = `pizza` · `guarana_dado_pizza` (exato por
      construção, pois ambos são medidos sobre os desfechos finais);
    - `aceita_combo_dado_oferta`: aceitar o combo entre quem viu a oferta."""
    p_nao_desiste = 1.0 if d.oferta is None else (1.0 - d.oferta["desiste"])
    pizza = d.comida["pizza"] * p_nao_desiste
    guarana_dado_pizza = d.bebida["guarana"]
    saida = {
        "pizza": pizza,
        "guarana_dado_pizza": guarana_dado_pizza,
        "pizza_e_guarana": pizza * guarana_dado_pizza,
    }
    if d.oferta is not None:
        saida["aceita_combo_dado_oferta"] = d.oferta["aceita_combo"]
    return saida


def _sortear(d: Distribuicoes, rng: np.random.Generator) -> str:
    """Uma jornada: sorteia comida; se pizza, sorteia bebida; se há oferta, sorteia a reação. Devolve o desfecho."""
    acoes_comida = grafo.ACOES[grafo.COMIDA]
    comida = rng.choice(acoes_comida, p=[d.comida[a] for a in acoes_comida])
    if comida != "pizza":
        return grafo.desfecho(comida, None, None)
    acoes_bebida = grafo.ACOES[grafo.BEBIDA]
    bebida = rng.choice(acoes_bebida, p=[d.bebida[a] for a in acoes_bebida])
    oferta = None
    if d.oferta is not None:
        acoes_oferta = grafo.ACOES[grafo.OFERTA]
        oferta = rng.choice(acoes_oferta, p=[d.oferta[a] for a in acoes_oferta])
    return grafo.desfecho(comida, bebida, oferta)


def numeros_fechados(preditor: rede.Preditor, registro: Mapping, cenario: Mapping, perfil: str | None) -> dict:
    """Percentuais em forma fechada (exatos, instantâneos) para um cenário. É o que o chat usa: guaraná pela taxa
    histórica (referência), pizza pela rede. Não roda Monte Carlo no caminho da mensagem. Inclui, à parte, a estimativa
    experimental da rede para guaraná|pizza (`experimento_rede_guarana`), rotulada como experimento."""
    d = distribuicoes(preditor, registro, cenario, perfil)
    numeros = _fechado(d)
    numeros["experimento_rede_guarana"] = d.bebida_rede["guarana"] if d.bebida_rede else None
    return numeros


def comparar_fechado(preditor: rede.Preditor, registro: Mapping, base_cenario: Mapping, perfil: str | None,
                     variacoes: Mapping[str, Mapping]) -> dict:
    """Compara cenários em forma fechada (sem sorteio). Usado pelo chat para 'e com 5%?'."""
    base = numeros_fechados(preditor, registro, base_cenario, perfil)
    resultado = {"base": base, "variacoes": {}}
    for nome, cen in variacoes.items():
        nums = numeros_fechados(preditor, registro, cen, perfil)
        difs = {k: nums[k] - base.get(k, 0.0) for k in nums
                if isinstance(nums[k], (int, float)) and isinstance(base.get(k), (int, float))}
        resultado["variacoes"][nome] = {"numeros": nums, "diferenca": difs}
    return resultado


def simular(preditor: rede.Preditor, registro: Mapping, cenario: Mapping, perfil: str | None,
            jornadas: int = JORNADAS_PADRAO, semente: int = SEMENTE_PADRAO) -> dict:
    """Roda N jornadas sorteadas e agrega os percentuais. Também devolve os valores em forma fechada e o erro de Monte
    Carlo (|simulado − fechado|), para o registro de verificação. Mesma semente ⇒ mesmo resultado.

    A aceitação do combo é uma transição interna (a jornada não guarda a reação à oferta no desfecho), então ela é
    reportada em forma fechada; o sorteio confere pizza, guaraná|pizza e pizza+guaraná."""
    d = distribuicoes(preditor, registro, cenario, perfil)
    rng = np.random.default_rng(semente)
    contagem = {desf: 0 for desf in grafo.DESFECHOS}
    for _ in range(jornadas):
        contagem[_sortear(d, rng)] += 1
    n = jornadas
    # eventos sobre os desfechos finais: terminar com pizza exclui quem desistiu e quem foi para outra refeição
    terminou_pizza = sum(contagem[k] for k in grafo.DESFECHOS if k.startswith("pizza_"))
    simulado = {
        "pizza": terminou_pizza / n,
        "guarana_dado_pizza": contagem["pizza_com_guarana"] / terminou_pizza if terminou_pizza else None,
        "pizza_e_guarana": contagem["pizza_com_guarana"] / n,
    }
    fechado = _fechado(d)
    erro = {k: abs(simulado[k] - fechado[k]) for k in fechado
            if k in simulado and simulado[k] is not None}
    return {
        "numeros": fechado, "simulado": simulado, "erro_monte_carlo": erro,
        "contagem": contagem, "jornadas": n, "semente": semente,
        "distribuicoes": {"comida": d.comida, "bebida": d.bebida, "oferta": d.oferta},
    }


def comparar(preditor: rede.Preditor, registro: Mapping, base_cenario: Mapping, perfil: str | None,
             variacoes: Mapping[str, Mapping], jornadas: int = JORNADAS_PADRAO, semente: int = SEMENTE_PADRAO) -> dict:
    """Compara cenários na mesma população (mesma semente). `variacoes` mapeia um nome para o cenário alterado (ex.:
    {'sem_desconto': {...}, 'com_desconto': {...}}). Devolve os percentuais de cada e as diferenças para o cenário base."""
    base = simular(preditor, registro, base_cenario, perfil, jornadas, semente)
    resultado = {"base": base["numeros"], "variacoes": {}}
    for nome, cen in variacoes.items():
        v = simular(preditor, registro, cen, perfil, jornadas, semente)
        difs = {k: v["numeros"][k] - base["numeros"].get(k, 0.0) for k in v["numeros"]}
        resultado["variacoes"][nome] = {"numeros": v["numeros"], "diferenca": difs}
    return resultado
