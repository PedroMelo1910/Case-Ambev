"""Coleta ao vivo de clima (Open-Meteo) e feriado nacional (BrasilAPI), só para a FALA direcional da persona.

Quando o usuário não informa clima nem feriado, a persona pode consultar essas fontes abertas e gratuitas e descrever
o efeito de forma direcional ("tende a subir/cair"), NUNCA como número: o percentual continua vindo do servidor
(`calculo`/`simulador`). As fontes:

- Clima: Open-Meteo (https://open-meteo.com), sem chave, por coordenada. Previsão diária de chuva e temperatura.
- Feriado nacional: BrasilAPI (https://brasilapi.com.br/api/feriados/v1/{ano}). Só nacionais; municipais ficam de fora.

Degradação graciosa: se a rede falhar, o tempo esgotar ou a resposta vier fora do formato, devolve `None` e a persona
responde sem o fator contextual — nunca trava. Timeout curto. As chamadas são injetáveis (`abrir`) para os testes
rodarem sem internet. Nenhuma chamada é feita na importação; só quando a persona pede.
"""

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import date

from personas import hipoteses

TEMPO_LIMITE = 4.0
AGENTE = "case-ambev-personas/1.0"
URL_FERIADOS = "https://brasilapi.com.br/api/feriados/v1/{ano}"
URL_CLIMA = ("https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
             "&daily=precipitation_sum,temperature_2m_max,temperature_2m_min&timezone=America%2FSao_Paulo"
             "&start_date={dia}&end_date={dia}")

# Coordenadas aproximadas das capitais usadas pelo contexto (quando a cidade é conhecida). Fallback por região.
COORD_REGIAO = {
    "Norte": (-3.1, -60.0), "Nordeste": (-8.05, -34.9), "Centro-Oeste": (-15.8, -47.9),
    "Sudeste": (-23.55, -46.63), "Sul": (-25.43, -49.27),
}


def _abrir_padrao(url: str, timeout: float):
    req = urllib.request.Request(url, headers={"User-Agent": AGENTE, "Accept": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def _buscar_json(url: str, abrir: Callable, timeout: float):
    """Devolve o JSON da URL, ou None em qualquer falha de rede, tempo ou formato (degradação graciosa)."""
    try:
        with abrir(url, timeout) as resposta:
            return json.loads(resposta.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, UnicodeDecodeError):
        return None


# ---------- feriado nacional ao vivo ----------

def feriado_nacional(dia: date, abrir: Callable = _abrir_padrao, timeout: float = TEMPO_LIMITE) -> dict | None:
    """Feriado nacional na data, pela BrasilAPI. None se a consulta falhar. `encontrado=False` se o dia não é feriado
    (consulta funcionou, mas a data não consta)."""
    dados = _buscar_json(URL_FERIADOS.format(ano=dia.year), abrir, timeout)
    if not isinstance(dados, list):
        return None
    iso = dia.isoformat()
    for item in dados:
        if isinstance(item, dict) and item.get("date") == iso:
            return {"encontrado": True, "nome": str(item.get("name", "feriado nacional")), "fonte": "BrasilAPI"}
    return {"encontrado": False, "fonte": "BrasilAPI"}


# ---------- clima ao vivo ----------

def clima(dia: date, regiao: str, coordenada: tuple[float, float] | None = None,
          abrir: Callable = _abrir_padrao, timeout: float = TEMPO_LIMITE) -> dict | None:
    """Clima previsto para a data e a região (ou coordenada), pela Open-Meteo. None se a consulta falhar.

    Classifica em 'chuva' (precipitação relevante), 'calor' (máxima alta), 'frio' (máxima baixa) ou 'ameno'. A
    classificação é só para a fala direcional; nenhum número vai para o cálculo."""
    lat, lon = coordenada if coordenada else COORD_REGIAO.get(regiao, COORD_REGIAO["Sudeste"])
    dados = _buscar_json(URL_CLIMA.format(lat=lat, lon=lon, dia=dia.isoformat()), abrir, timeout)
    try:
        diario = dados["daily"]
        chuva_mm = float(diario["precipitation_sum"][0])
        tmax = float(diario["temperature_2m_max"][0])
        tmin = float(diario["temperature_2m_min"][0])
    except (TypeError, KeyError, IndexError, ValueError):
        return None
    return {"classe": _classe_clima(chuva_mm, tmax), "chuva_mm": chuva_mm, "temperatura_max": tmax,
            "temperatura_min": tmin, "fonte": "Open-Meteo"}


def _classe_clima(chuva_mm: float, tmax: float) -> str:
    if chuva_mm >= 5.0:
        return "chuva"
    if tmax >= 30.0:
        return "calor"
    if tmax <= 18.0:
        return "frio"
    return "ameno"


# ---------- frase direcional (sem número) ----------

def frase_contexto(fatores: list[str]) -> str | None:
    """Monta uma frase direcional a partir dos fatores contextuais detectados ao vivo (clima/feriado/fim de semana).
    Descreve só a direção do efeito, com a ressalva de que é tendência, não número calculado."""
    partes = [hipoteses.CONTEXTO_QUALITATIVO[f]["frase"] for f in fatores
              if f in hipoteses.CONTEXTO_QUALITATIVO]
    if not partes:
        return None
    corpo = partes[0] if len(partes) == 1 else "; ".join(partes[:-1]) + "; e " + partes[-1]
    return (f"Pelo contexto do dia, {corpo}. É uma leitura de tendência pela situação, não um número calculado: a "
            "estimativa em percentual não muda por causa disso.")
