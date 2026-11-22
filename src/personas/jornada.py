"""Ligação entre a conversa e a simulação: monta o registro de features do perfil + contexto, roda o simulador e
produz os números e a fala da aba Personas, com indicação de "Simulação".

A conversa conhece dia/período/região/área (campos do cálculo) e o perfil escolhido. A rede precisa também dos
atributos do perfil (faixa etária, sexo, escolaridade, composição da casa, renda per capita), que não vêm da mensagem:
eles vêm de um recorte representativo de cada perfil, declarado aqui e rotulado como característica do perfil, não como
algo que o usuário disse. O nome do perfil nunca entra na rede; só os atributos.

Se o artefato da rede não existir, `preditor()` devolve None e a conversa segue sem a simulação (degradação graciosa):
o número continua disponível pela via histórica de `calculo`.
"""

from functools import lru_cache

from personas import rede, simulador

# Atributos representativos de cada perfil (recorte da POF que define o perfil em `aplicacao.personas`).
# São características do perfil simulado, não declarações do usuário. Mantêm o contraste entre os dois sem usar o nome.
ATRIBUTOS_PERFIL = {
    "pai_de_familia": {"faixa_etaria": "30_44", "sexo": "masculino", "escolaridade": "medio",
                       "n_moradores": 4, "n_menores": 2, "renda_pc": 1200.0},
    "jovem_sozinho": {"faixa_etaria": "18_29", "sexo": "masculino", "escolaridade": "superior",
                      "n_moradores": 1, "n_menores": 0, "renda_pc": 2200.0},
}


@lru_cache(maxsize=1)
def preditor() -> rede.Preditor | None:
    """Rede treinada, ou None se o artefato/ PyTorch não estiver disponível. Carregada uma vez. Sem ela, a conversa
    segue pela via histórica (`calculo`): PyTorch é dependência opcional, isolada do núcleo."""
    try:
        return rede.carregar()
    except (FileNotFoundError, OSError, ValueError, ImportError):
        return None


def _registro(ctx: dict, perfil: str) -> dict:
    """Features da rede: contexto da conversa (dia/período/região/área) + atributos do perfil."""
    attr = ATRIBUTOS_PERFIL[perfil]
    return {
        "grande_regiao": ctx["grande_regiao"]["valor"], "area": ctx["area"]["valor"],
        "periodo": ctx["periodo"]["valor"], "dia_semana": ctx["dia_semana"]["valor"],
        **attr,
    }


def _cenario(ctx: dict) -> dict:
    """Cenário para as hipóteses: só a oferta é relevante ao simulador (clima/feriado vão para a fala, não ao número)."""
    cen = {}
    if "oferta" in ctx:
        cen["oferta"] = ctx["oferta"]["valor"]
    return cen


def simular_jornada(ctx: dict, perfil: str) -> dict | None:
    """Números da jornada para o contexto completo e o perfil, em forma fechada (exatos, instantâneos). None se a rede
    não está disponível ou faltam campos. O chat não roda Monte Carlo a cada mensagem: a forma fechada é exata e o
    sorteio (simulador.simular) fica para o registro de verificação."""
    pred = preditor()
    if pred is None or perfil not in ATRIBUTOS_PERFIL:
        return None
    if any(c not in ctx for c in ("dia_semana", "periodo", "grande_regiao", "area")):
        return None
    registro = _registro(ctx, perfil)
    cenario = _cenario(ctx)
    numeros = simulador.numeros_fechados(pred, registro, cenario, perfil)
    saida = {
        "numeros": numeros, "jornadas": "forma fechada (probabilidades exatas do grafo)",
        "modelo": "rede neural (RUR-Net) treinada na POF 2008–2009",
        "atributos_do_perfil": ATRIBUTOS_PERFIL[perfil],
    }
    if cenario.get("oferta"):
        saida["comparacao_desconto"] = _comparar_desconto(pred, registro, perfil, cenario)
    return saida


def _comparar_desconto(pred, registro, perfil, cenario) -> dict | None:
    """Compara preço cheio (0%) com o desconto que a pessoa informou, na mesma população (forma fechada). O percentual
    vem do cenário; sem percentual informado, não há comparação de desconto (só a aceitação base)."""
    oferta = cenario["oferta"]
    tipo = oferta.get("tipo", "combo")
    pct = oferta.get("percentual")
    if not pct:  # 0% ou não informado: o cenário de referência é o próprio; nada a comparar
        return None
    sem = {"oferta": {"percentual": None, "tipo": tipo}}
    com = {"oferta": {"percentual": pct, "tipo": tipo}}
    comp = simulador.comparar_fechado(pred, registro, sem, perfil, {"sem_desconto": sem, "com_desconto": com})
    dif = comp["variacoes"]["com_desconto"]["diferenca"].get("aceita_combo_dado_oferta")
    return {"percentual": pct,
            "sem_desconto": comp["base"].get("aceita_combo_dado_oferta"),
            "com_desconto": comp["variacoes"]["com_desconto"]["numeros"].get("aceita_combo_dado_oferta"),
            "diferenca_pp": dif * 100 if dif is not None else None}
