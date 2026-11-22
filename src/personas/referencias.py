"""Fontes oficiais versionadas em `dados/`: calendario de feriados, municipios do IBGE, renda do responsavel por
distrito (Censo 2022) e a tabela validada de T1. Somente leitura; nenhuma chamada de rede em tempo de execucao.

Cobertura: feriados nacionais do Brasil e, para estaduais e municipais, so a cidade de Sao Paulo (3550308). O
territorio cobre os 96 distritos da capital paulista. Fora disso a resposta diz que a fonte nao cobre.
"""

import json
import re
import unicodedata
from datetime import date
from functools import lru_cache
from pathlib import Path

DADOS = Path(__file__).resolve().parent / "dados"
SP = "3550308"
REGIAO_DA_UF = {**dict.fromkeys(("AC", "AP", "AM", "PA", "RO", "RR", "TO"), "Norte"),
                **dict.fromkeys(("AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"), "Nordeste"),
                **dict.fromkeys(("ES", "MG", "RJ", "SP"), "Sudeste"), **dict.fromkeys(("PR", "RS", "SC"), "Sul"),
                **dict.fromkeys(("DF", "GO", "MS", "MT"), "Centro-Oeste")}


def normalizar(texto: str) -> str:
    """Minusculas, sem acento e com espacos nas pontas: casa palavras inteiras com `f" {termo} " in texto`."""
    sem_acento = unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode("ascii")
    return " " + re.sub(r"[^a-z0-9]+", " ", sem_acento).strip() + " "


@lru_cache(maxsize=None)
def carregar(nome: str) -> dict:
    return json.loads((DADOS / nome).read_text(encoding="utf-8"))


# ---------- calendario ----------

def feriado(dia: date, municipio: str | None) -> dict:
    """Feriado oficial na data. `cobertura_municipal` diz se a fonte cobre estaduais e municipais do municipio."""
    cal = carregar("calendario_2026.json")
    iso = dia.isoformat()
    cobertura = municipio in cal["municipios"]
    fontes = ["FER-NAC-2026"] + (["FER-SP-CAP-2026"] if cobertura else [])
    lista = cal["municipios"][municipio] if cobertura else [{**f, "esfera": "nacional"} for f in cal["nacionais"]]
    achado = next((f for f in lista if f["data"] == iso), None)
    fora_do_ano = dia.year != 2026
    return {"feriado": bool(achado), "nome": achado["nome"] if achado else None,
            "esfera": achado["esfera"] if achado else None, "fontes_consultadas": [] if fora_do_ano else fontes,
            "cobertura_municipal": cobertura and not fora_do_ano, "fora_do_ano": fora_do_ano}


def fonte(fonte_id: str) -> dict | None:
    calendario = carregar("calendario_2026.json")["fontes"]
    if fonte_id in calendario:
        return {"id": fonte_id, **calendario[fonte_id]}
    for nome in ("municipios.json", "territorio_sp.json"):
        dados = carregar(nome)["fonte"]
        if dados["id"] == fonte_id:
            return dados
    return None


# ---------- municipios ----------

@lru_cache(maxsize=None)
def _municipios() -> dict:
    dados = carregar("municipios.json")
    capitais = set(dados["capitais"])
    por_codigo = {c: {"codigo": c, "nome": n, "uf": uf, "capital": c in capitais, "grande_regiao": REGIAO_DA_UF[uf]}
                  for c, n, uf in dados["municipios"]}
    return {"codigo": por_codigo}


def municipio(codigo: str) -> dict:
    return _municipios()["codigo"][codigo]


def existe_municipio(codigo: object) -> bool:
    return isinstance(codigo, str) and codigo in _municipios()["codigo"]


def codigo_da_capital(uf: str) -> str:
    return next(c for c, m in _municipios()["codigo"].items() if m["uf"] == uf and m["capital"])


def capitais_no_texto(texto: str) -> list[str]:
    """Capitais citadas por nome (palavra inteira), exceto Sao Paulo, que exige marcador de cidade ou estado."""
    return [c for c, m in _municipios()["codigo"].items()
            if m["capital"] and c != SP and f" {normalizar(m['nome']).strip()} " in texto]


# ---------- territorio ----------

@lru_cache(maxsize=None)
def _distritos_por_nome() -> dict:
    return {normalizar(d["nome"]).strip(): c for c, d in carregar("territorio_sp.json")["distritos"].items()}


_NOMES_CURTOS = {"se": r" (?:na|da|distrito da|bairro da|regiao da) se "}  # "se" e conjuncao: exige marcador


def distritos_no_texto(texto: str) -> list[str]:
    achados = []
    for nome in sorted(_distritos_por_nome(), key=len, reverse=True):
        if nome in _NOMES_CURTOS and not re.search(_NOMES_CURTOS[nome], texto):
            continue
        if f" {nome} " in texto:
            texto = texto.replace(f" {nome} ", " ")
            achados.append(_distritos_por_nome()[nome])
    return achados


def existe_distrito(codigo: object) -> bool:
    return isinstance(codigo, str) and codigo in carregar("territorio_sp.json")["distritos"]


def territorio(codigo_distrito: str) -> dict:
    dados = carregar("territorio_sp.json")
    return {"fonte_id": dados["fonte"]["id"], "fonte": dados["fonte"], "ano": dados["ano"], "conceito": dados["conceito"],
            "nao_e": dados["nao_e"], "municipio": dados["municipio"],
            "distrito": {"codigo": codigo_distrito, **dados["distritos"][codigo_distrito]}}
