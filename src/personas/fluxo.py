"""Orquestracao das abas Personas e Recomendacoes: um turno por chamada, servidor sem estado.

Etapas (os mesmos nos da prova em LangGraph, fora da entrega): entender (`contexto.interpretar`) -> validar ->
recuperar evidencia (`evidencias`) -> calcular (`calculo`, so com o contexto completo) -> decidir (regras) ->
redigir (`redacao`, afirmacoes de `afirmacoes`) -> verificar. O LLM nao le a mensagem: so redige abertura e
fechamento a partir de campos normalizados. A pagina devolve `estado` a cada mensagem; ele e validado antes de usar.
"""

from collections.abc import Mapping
from datetime import date, datetime
from zoneinfo import ZoneInfo

from aplicacao import EntradaInvalida, recebido
from aplicacao.personas import PERSONAS
from personas import (
    VERSAO,
    acoes,
    afirmacoes,
    ao_vivo,
    calculo,
    contexto,
    evidencias,
    jornada,
    redacao,
)
from personas import referencias as ref

ABAS = ("personas", "recomendacoes")
MAXIMO_MENSAGEM = 500
MAXIMO_TURNOS = 200
FUSO = ZoneInfo("America/Sao_Paulo")
_RECOMECAR = (" recomecar ", " nova conversa ", " comecar de novo ")

HIPOTESES = {
    "h_feriado": ("Feriado muda a chance de pizza e guaraná",
                  "A POF de 2008–2009 não tem data do calendário; o modelo não considera feriado",
                  "Comparar pedidos datados de um parceiro em feriados e em dias comuns do mesmo dia da semana"),
    "h_chuva": ("Chuva muda a chance de pedir pizza e guaraná",
                "A POF não registra o clima de cada refeição",
                "Cruzar pedidos com hora e loja com a chuva observada (INMET) e a previsão meteorológica emitida antes, comparando dias "
                "parecidos; mede associação, porque o clima não pode ser sorteado"),
    "h_desconto": ("O desconto no combo aumenta a margem por pedido",
                   "A POF não registra preço, oferta nem exposição à oferta",
                   "Sortear clientes entre combo com desconto e preço cheio num restaurante parceiro e comparar a margem "
                   "incremental por pedido"),
    "h_renda_bairro_marca": ("No bairro de menor renda, Guaraná Antarctica ganha participação mesmo com a Coca-Cola à frente",
                             "A POF não tem bairro nem marca; a renda do distrito é de 2022 e descreve a área",
                             "Pedidos com refrigerante por CEP ou setor de entrega e produto com marca, por quintil de "
                             "renda da área, comparando a participação de cada marca"),
}

def _validar(pedido: object) -> tuple[str, str | None, str, dict]:
    if not isinstance(pedido, Mapping) or set(pedido) != {"aba", "persona", "mensagem", "estado"}:
        raise EntradaInvalida('o pedido deve ter exatamente os campos "aba", "persona", "mensagem" e "estado"')
    aba, persona, mensagem, estado = (pedido[k] for k in ("aba", "persona", "mensagem", "estado"))
    if aba not in ABAS:
        raise EntradaInvalida(f"aba deve ser uma de {list(ABAS)}, recebido {recebido(aba)}")
    if aba == "personas" and (type(persona) is not str or persona not in PERSONAS):
        raise EntradaInvalida(f"persona deve ser uma de {list(PERSONAS)}, recebido {recebido(persona)}")
    if aba == "recomendacoes" and persona is not None:
        raise EntradaInvalida("a aba Recomendações não usa persona")
    if type(mensagem) is not str or len(mensagem) > MAXIMO_MENSAGEM:
        raise EntradaInvalida(f"mensagem deve ser um texto de até {MAXIMO_MENSAGEM} caracteres")
    return aba, persona, mensagem, _validar_estado(estado)


def _validar_estado(estado: object) -> dict:
    if estado == {}:
        return {"turno": 0, "contexto": {}, "historico": []}
    if not isinstance(estado, Mapping) or set(estado) != {"turno", "contexto", "historico"}:
        raise EntradaInvalida("estado da conversa fora do formato")
    turno, historico = estado["turno"], estado["historico"]
    if type(turno) is not int or not 0 <= turno <= MAXIMO_TURNOS or not isinstance(historico, list):
        raise EntradaInvalida("estado da conversa inválido")
    for item in historico:
        if not isinstance(item, Mapping) or set(item) != {"campo", "de", "para", "turno"}:
            raise EntradaInvalida("histórico da conversa inválido")
    return {"turno": turno, "contexto": contexto.validar(estado["contexto"]), "historico": [dict(h) for h in historico]}


def _hipoteses(ctx: dict, temas: frozenset) -> list[dict]:
    ativos = []
    if ctx.get("feriado") and ctx["feriado"]["estado"] != "confirmado":
        ativos.append("h_feriado")
    if "clima" in ctx:
        ativos.append("h_chuva")
    if "oferta" in ctx:
        ativos.append("h_desconto")
    if "distrito" in ctx or {"renda", "marca", "territorio"} & temas:
        ativos.append("h_renda_bairro_marca")
    return [{"id": i, "texto": HIPOTESES[i][0], "por_que_nao_entra": HIPOTESES[i][1], "teste": HIPOTESES[i][2],
             "papel": "hipotese"} for i in ativos]


def _temas_do_contexto(ctx: dict) -> set[str]:
    return {tema for campo, tema in (("oferta", "desconto"), ("clima", "clima"), ("feriado", "feriado"),
                                     ("distrito", "territorio")) if campo in ctx}


def _evidencias(mensagem: str, temas: set) -> list[dict]:
    acervo = evidencias.carregar()
    achadas = acervo.buscar(mensagem, temas) if mensagem.strip() else []
    if {"territorio", "renda"} & temas:
        prioritarios = [acervo.cartao(i) for i in ("REN-SAB-001", "TER-CENSO-001", "REN-LIM-001")]
        achadas = prioritarios + [a for a in achadas if a["id"] not in {p["id"] for p in prioritarios}]
    return achadas[:6]


def _perguntas(da_mensagem: list[str], faltam: list[str]) -> list[str]:
    """Campos do calculo primeiro (a ambiguidade de cidade ou estado cobre regiao e area); o bairro por ultimo."""
    cobertos = {"grande_regiao", "area"} if "cidade_ou_estado" in da_mensagem else set()
    antes = [p for p in da_mensagem if p != "bairro"]
    depois = ["bairro"] if "bairro" in da_mensagem else []
    return list(dict.fromkeys([*antes, *(f for f in faltam if f not in cobertos), *depois]))


def _fatores_ao_vivo(ctx: dict, clima_fn=None, feriado_fn=None) -> dict:
    """Fatores contextuais qualitativos para a FALA (direcional, sem número). Coleta clima/feriado ao vivo só quando as
    funções de busca são injetadas (o servidor as passa); sem elas, não faz rede — é o caso dos testes. Clima/feriado
    informados pelo usuário entram direto. Degrada em silêncio se a busca falhar. Fim de semana vem do próprio dia."""
    fatores: list[str] = []
    fontes: list[str] = []
    if "dia_semana" in ctx and ctx["dia_semana"]["valor"] in (5, 6, 7):
        fatores.append("fim_de_semana")
    # clima informado pelo usuário entra direto; senão, tenta ao vivo (só se a função foi injetada)
    if "clima" in ctx:
        fatores.append(ctx["clima"]["valor"])
    elif clima_fn is not None and "data" in ctx and "grande_regiao" in ctx:
        clima = clima_fn(date.fromisoformat(ctx["data"]["valor"]), ctx["grande_regiao"]["valor"])
        if clima and clima["classe"] != "ameno":
            fatores.append(clima["classe"])
            fontes.append(clima["fonte"])
    # feriado: confirmado pela fonte local já vira fator; senão, tenta o calendário nacional ao vivo (se injetado)
    if ctx.get("feriado", {}).get("estado") == "confirmado":
        fatores.append("feriado")
    elif feriado_fn is not None and "feriado" not in ctx and "data" in ctx:
        fer = feriado_fn(date.fromisoformat(ctx["data"]["valor"]))
        if fer and fer["encontrado"]:
            fatores.append("feriado")
            fontes.append(fer["fonte"])
    ordem = ("feriado", "fim_de_semana", "chuva", "frio", "calor")
    fatores = [f for f in ordem if f in fatores]
    return {"fatores": fatores, "frase": ao_vivo.frase_contexto(fatores), "fontes": sorted(set(fontes))}


def conversar(pedido: object, hoje: date | None = None, redator=None, clima_ao_vivo=None, feriado_ao_vivo=None) -> dict:
    aba, persona, mensagem, estado = _validar(pedido)
    hoje = hoje or datetime.now(FUSO).date()
    if mensagem.strip() == "" or any(t in ref.normalizar(mensagem) for t in _RECOMECAR):
        estado = {"turno": 0, "contexto": {}, "historico": []}
    turno = estado["turno"] + 1
    lido = contexto.interpretar(mensagem, estado["contexto"], turno, hoje)
    ctx, temas = lido.contexto, lido.temas  # a leitura da mensagem fica no servidor; o LLM nunca a recebe
    novidade = bool(lido.historico or lido.temas or ctx != estado["contexto"])
    faltam = contexto.faltantes(ctx)
    calc = calculo.calcular(ctx) if not faltam else {"numeros": [], "grafico": None, "leitura": None}
    simulacao = jornada.simular_jornada(ctx, persona) if (aba == "personas" and not faltam) else None
    contexto_ao_vivo = (_fatores_ao_vivo(ctx, clima_ao_vivo, feriado_ao_vivo) if aba == "personas"
                        else {"fatores": [], "frase": None, "fontes": []})
    achadas = _evidencias(mensagem, set(temas) | (_temas_do_contexto(ctx) if novidade else set()))
    anterior = estado["contexto"]
    mudou = sorted({h["campo"] for h in lido.historico} | {k for k in ctx if k not in anterior}
                   | {k for k in ctx if k in anterior and ctx[k]["estado"] != anterior[k]["estado"]})
    repete_calculo = not faltam and not contexto.faltantes(anterior) and not set(mudou) & set(contexto.CAMPOS_DO_CALCULO)
    contrato = {
        "mudou": mudou, "repete_calculo": repete_calculo, "primeiro_calculo": bool(contexto.faltantes(anterior)),
        "versao": VERSAO, "aba": aba, "persona": persona, "contexto": ctx, "historico": estado["historico"] + lido.historico,
        "faltantes": faltam, "perguntas": _perguntas(lido.perguntas, faltam), "numeros": calc["numeros"],
        "grafico": calc["grafico"], "leitura": calc["leitura"], "evidencias": achadas,
        "hipoteses": _hipoteses(ctx, temas), "simulacao": simulacao, "contexto_ao_vivo": contexto_ao_vivo,
        "territorio": ref.territorio(ctx["distrito"]["valor"]) if "distrito" in ctx else None,
        "temas": sorted(temas), "sem_evidencia": bool(mensagem.strip()) and not achadas and not novidade,
        "decisao": acoes.decidir(aba, ctx, calc["leitura"], temas),
        "voz": redacao.voz(persona if aba == "personas" else None),
    }
    if aba == "personas":
        contrato["intencao"] = redacao.intencao_persona(mensagem, temas, ctx)
    leitura = calc["leitura"]
    contrato["leitura_cenario"] = afirmacoes.leitura_do_cenario(ctx, leitura) if leitura else None
    contrato["afirmacoes"] = afirmacoes.montar(calc["numeros"], contrato["leitura_cenario"]) if leitura else []
    contrato["fatores"] = leitura["fatores"] if leitura else []
    falas, modo, avisos = redacao.redigir(contrato, redator)
    fora = redacao.numeros_fora(" ".join(falas), redacao.permitidos(contrato, redacao.deterministico(contrato)))
    contrato.update(falas=falas, modo_redacao=modo,
                    verificacao={"todos_no_contrato": not fora, "numeros_fora": fora, "avisos": avisos},
                    estado={"turno": turno, "contexto": ctx, "historico": contrato["historico"]})
    return contrato
