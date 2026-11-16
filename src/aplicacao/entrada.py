"""Validacao do contexto, renda e dominio verificado, reutilizando as transformacoes de `occasion.variaveis`.

O contexto descreve UM episodio de consumo com pizza: hora (0 a 23) e dia (1=segunda ... 7=domingo), localizacao, a pessoa
(escolaridade em niveis do IBGE: 1-3 ate fundamental, 4-5 medio, 6-7 superior) e a unidade de consumo (UC) dela.
Valor ausente, de outro tipo, fora das categorias ou numa unidade que a aplicacao nao converte e recusado com o nome do
campo; nada e imputado. Renda: `renda_total_uc` e a renda bruta TOTAL MENSAL da UC (inclui renda nao monetaria), e a
renda per capita e essa renda dividida por TODOS os moradores da UC (`idades_uc`), na referencia monetaria declarada.
So duas referencias tem conversao implementada e registrada em `configs/variaveis.json` (IPCA): reais de jan/2009 (fator
1) e de jan/2018. Valores de outras datas, como 2026, sao recusados: o fator historico nao e conversor de renda atual.

Dominio verificado = limites MARGINAIS observados em 2008 (idade, renda per capita, moradores da UC); categorias validas
ja sao conferidas na validacao. Dentro dos limites marginais nao garante suporte para a combinacao de atributos.
"""

import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from aplicacao.regras import EntradaInvalida, conferir_campos, recebido
from occasion.variaveis import COLUNAS_LOGISTICA, carregar_regras, codificar_logistica, derivar_nativas

CAMPOS = ("hora", "dia_semana", "grande_regiao", "area", "idade", "sexo", "escolaridade", "renda_total_uc",
          "referencia_monetaria_renda", "idades_uc")
REFERENCIAS_RENDA = MappingProxyType({"reais_jan_2009": "2008-2009", "reais_jan_2018": "2017-2018"})  # -> chave de fator_renda
IDADE_MINIMA_PESSOA = carregar_regras()["faixas_etarias"][0]["min"]  # 10: populacao do consumo individual
IDADE_MAXIMA = 111  # mesmo limite de occasion.variaveis (dicionario de 2017: 0 a 111), conferido em teste
MAXIMO_MORADORES = 100
TOLERANCIA_LOG_RENDA = 1e-9

# Agregados de 2008-2009 (1.168 episodios com pizza), conferidos contra `variaveis_nativas_2008.csv`.
DOMINIO_2008 = MappingProxyType({"idade": (10, 94), "log_renda_pc": (3.9576016690393097, 10.192415848129171),
                                 "moradores_uc": (1, 11)})
FONTE_DOMINIO = MappingProxyType({
    "arquivo": "variaveis_nativas_2008.csv", "episodios": 1168,
    "sha256": "1c84465387ac7ddf06a3280fe12148b344c20044615b88967a39f2fc6eb21534",
    "nota": "minimo e maximo observados em 2008-2009; renda per capita em reais de jan/2009"})
NOTA_DOMINIO = ("Limites marginais válidos não garantem suporte suficiente para a combinação de atributos informada: "
                "o suporte da combinação não foi verificado.")
DEFINICAO_RENDA = ("Renda bruta total mensal da unidade de consumo (inclui renda não monetária e variação patrimonial, não "
                   "só o salário), dividida por todos os moradores da unidade de consumo e expressa em reais de jan/2009; a "
                   "referência informada é convertida pelo fator registrado no config (1 para jan/2009).")
DICA_DIA = "1=segunda, 2=terça, 3=quarta, 4=quinta, 5=sexta, 6=sábado, 7=domingo"
DICA_ESCOLARIDADE = "níveis do IBGE: 1-3 até fundamental, 4-5 médio, 6-7 superior"
MINIMO_RENDA, MAXIMO_RENDA = 1e-6, 1e12  # R$ por mes da UC: fora disso o log per capita perde sentido (ou da overflow)


@dataclass(frozen=True)
class Contexto:
    """Contexto validado: variaveis derivadas, linha da logistica (None = desconhecido), dominio e dados da renda."""
    nativas: dict
    linha: dict
    dominio: dict
    renda: dict


def _inteiro(valor: object, campo: str, minimo: int, maximo: int, dica: str = "") -> int:
    if isinstance(valor, bool) or not isinstance(valor, numbers.Integral) or not minimo <= valor <= maximo:
        complemento = f" ({dica})" if dica else ""
        raise EntradaInvalida(f"{campo} deve ser inteiro de {minimo} a {maximo}{complemento}, recebido {recebido(valor)}")
    return int(valor)


def _categoria(valor: object, campo: str, validos: tuple[str, ...]) -> str:
    if type(valor) is not str or valor not in validos:  # str exato: subclasses (ex.: np.str_) nao passam
        raise EntradaInvalida(f"{campo} deve ser um de {list(validos)}, recebido {recebido(valor)}")
    return valor


def _erro_renda(valor: object) -> EntradaInvalida:
    return EntradaInvalida(f"renda_total_uc deve ser numero finito de {MINIMO_RENDA:g} a {MAXIMO_RENDA:g} "
                           f"(R$ por mes da UC), recebido {recebido(valor)}")


def _renda_total(valor: object) -> float:
    if isinstance(valor, bool) or not isinstance(valor, numbers.Real):
        raise _erro_renda(valor)
    try:
        numero = float(valor)
    except (OverflowError, TypeError, ValueError):
        numero = math.nan
    if not math.isfinite(numero) or not MINIMO_RENDA <= numero <= MAXIMO_RENDA:
        raise _erro_renda(valor)
    return numero


def _referencia(valor: object) -> str:
    if type(valor) is not str or valor not in REFERENCIAS_RENDA:
        raise EntradaInvalida(
            f"referencia_monetaria_renda deve ser uma de {list(REFERENCIAS_RENDA)}, recebido {recebido(valor)}; "
            "nao ha conversao implementada para renda de outras datas (por exemplo, valores atuais)")
    return valor


def _idades_uc(valor: object) -> list:
    if isinstance(valor, (str, bytes)) or not isinstance(valor, (list, tuple)) or not valor:
        raise EntradaInvalida(f"idades_uc deve ser lista nao vazia com a idade de cada morador da UC, recebido {recebido(valor)}")
    if len(valor) > MAXIMO_MORADORES:
        raise EntradaInvalida(f"idades_uc com mais de {MAXIMO_MORADORES} moradores")
    return [None if idade is None else _inteiro(idade, "idades_uc (cada idade; None = desconhecida)", 0, IDADE_MAXIMA)
            for idade in valor]


def _validar(contexto: Mapping[str, object]) -> dict:
    if not isinstance(contexto, Mapping):
        raise EntradaInvalida(f"contexto deve ser um dicionario com os campos {list(CAMPOS)}")
    conferir_campos(contexto, CAMPOS, "contexto", " (o papel do modelo e controlado pela aplicacao)")
    regras = carregar_regras()
    valido = {
        "hora": _inteiro(contexto["hora"], "hora", 0, 23),
        "dia_semana": _inteiro(contexto["dia_semana"], "dia_semana", 1, 7, DICA_DIA),
        "grande_regiao": _categoria(contexto["grande_regiao"], "grande_regiao", regras["regioes"]),
        "area": _categoria(contexto["area"], "area", regras["areas"]),
        "idade": _inteiro(contexto["idade"], "idade", IDADE_MINIMA_PESSOA, IDADE_MAXIMA),
        "sexo": _categoria(contexto["sexo"], "sexo", regras["sexos"]),
        "escolaridade": _inteiro(contexto["escolaridade"], "escolaridade", 1, 7, DICA_ESCOLARIDADE),
        "renda_total_uc": _renda_total(contexto["renda_total_uc"]),
        "referencia_monetaria_renda": _referencia(contexto["referencia_monetaria_renda"]),
        "idades_uc": _idades_uc(contexto["idades_uc"])}
    if None not in valido["idades_uc"] and valido["idade"] not in valido["idades_uc"]:
        raise EntradaInvalida(f"idade ({valido['idade']}) nao consta em idades_uc: a pessoa e um dos moradores da UC")
    return valido


def _verificacao(valor, limites, dentro) -> dict:
    return {"valor": valor, "limites": list(limites), "dentro": dentro}


def avaliar_dominio(nativas: dict, idades_desconhecidas: int) -> dict:
    """Estado do dominio nas marginais: True (verificado), False (fora; prevalece) ou None (ha informacao desconhecida)."""
    idade_min, idade_max = DOMINIO_2008["idade"]
    log_min, log_max = DOMINIO_2008["log_renda_pc"]
    mor_min, mor_max = DOMINIO_2008["moradores_uc"]
    log_renda = nativas["log_renda_pc"]
    verificacoes = {
        "idade": _verificacao(nativas["idade"], (idade_min, idade_max), idade_min <= nativas["idade"] <= idade_max),
        "renda_per_capita_jan_2009": _verificacao(
            math.exp(log_renda), (math.exp(log_min), math.exp(log_max)),
            log_min - TOLERANCIA_LOG_RENDA <= log_renda <= log_max + TOLERANCIA_LOG_RENDA),
        "moradores_uc": _verificacao(nativas["moradores_uc"], (mor_min, mor_max), mor_min <= nativas["moradores_uc"] <= mor_max),
        "composicao_da_uc": {"valor": f"{idades_desconhecidas} idade(s) de morador desconhecida(s)",
                             "dentro": True if idades_desconhecidas == 0 else None}}
    resultados = [v["dentro"] for v in verificacoes.values()]
    ok = False if False in resultados else (None if None in resultados else True)
    estado = {True: "verificado_nas_marginais", False: "fora_do_dominio", None: "desconhecido"}[ok]
    return {"ok": ok, "estado": estado, "verificacoes": verificacoes, "suporte_da_combinacao": "nao_verificado",
            "nota": NOTA_DOMINIO, "fonte_dos_limites": dict(FONTE_DOMINIO)}


def preparar(contexto: Mapping[str, object]) -> Contexto:
    """Valida o contexto e devolve nativas, linha da logistica, dominio e dados da renda. Nao imputa nada."""
    valido = _validar(contexto)
    edicao = REFERENCIAS_RENDA[valido["referencia_monetaria_renda"]]
    nativas = derivar_nativas(edicao, {
        "hora": valido["hora"], "dia_semana": valido["dia_semana"], "grande_regiao": valido["grande_regiao"],
        "area_comparavel": valido["area"], "idade": valido["idade"], "sexo": valido["sexo"],
        "escolaridade": valido["escolaridade"], "renda_total": valido["renda_total_uc"],
        "idades_uc": valido["idades_uc"]})
    desconhecidas = sum(idade is None for idade in valido["idades_uc"])
    fator = carregar_regras()["fator_renda"][edicao]
    renda = {"referencia_monetaria": valido["referencia_monetaria_renda"], "renda_total_uc": valido["renda_total_uc"],
             "fator_de_conversao": fator, "moradores_uc": nativas["moradores_uc"],
             "renda_per_capita_jan_2009": math.exp(nativas["log_renda_pc"]), "definicao_da_renda": DEFINICAO_RENDA,
             "idades_desconhecidas": desconhecidas}
    return Contexto(nativas=nativas, linha=dict(zip(COLUNAS_LOGISTICA, codificar_logistica(nativas))),
                    dominio=avaliar_dominio(nativas, desconhecidas), renda=renda)
