"""Adaptador opcional para GPT-OSS 20B no Groq (API compativel com OpenAI), so com a biblioteca padrao.

Desligado por padrao. Liga com CASE_AMBEV_LLM=groq e GROQ_API_KEY no ambiente do servidor (a chave nunca vai para a
pagina, para o log nem para o corpo da chamada). O LLM recebe so o pedido montado em `redacao.pedido_llm`: campos
normalizados, afirmacoes ja redigidas pelo servidor e trechos autorizados do acervo; nunca microdados, dados da
empresa nem a mensagem livre. Devolve JSON estrito com a abertura e o fechamento da conversa. Um 429 com
`retry-after` de ate 2 s tem uma nova tentativa; depois disso, `CotaEsgotada` e a resposta segue no modo deterministico.
"""

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping

URL = "https://api.groq.com/openai/v1/chat/completions"
MODELO = "openai/gpt-oss-20b"
TEMPO_LIMITE = 20
ESPERA_MAXIMA = 2.0
MAXIMO_TOKENS = 700
AGENTE = "case-ambev-personas/1.0"
# O LLM so devolve dois trechos de conversa; toda frase com numero e montada pelo servidor (`personas.redacao`).
ESQUEMA_TRECHOS = {"type": "object", "additionalProperties": False, "required": ["abertura", "fechamento"],
                   "properties": {"abertura": {"type": "string"}, "fechamento": {"type": "string"}}}  # o Groq recusa o User-Agent padrao do Python (HTTP 403, erro 1010)


class ErroLLM(RuntimeError):
    """O modelo de linguagem nao respondeu ou respondeu fora do formato."""


class CotaEsgotada(ErroLLM):
    """O provedor recusou por limite de uso (HTTP 429)."""


class Groq:
    def __init__(self, chave: str, modelo: str = MODELO, abrir: Callable = urllib.request.urlopen,
                 esperar: Callable[[float], None] = time.sleep):
        if not chave:
            raise ValueError("chave do Groq ausente")
        self._chave = chave
        self.modelo = modelo
        self._abrir = abrir
        self._esperar = esperar

    def __repr__(self) -> str:  # nunca expor a chave
        return f"Groq(modelo={self.modelo!r})"

    def _pedido(self, sistema: str, usuario: str) -> urllib.request.Request:
        corpo = {"model": self.modelo, "temperature": 0.2, "max_completion_tokens": MAXIMO_TOKENS,
                 "messages": [{"role": "system", "content": sistema}, {"role": "user", "content": usuario}],
                 "response_format": {"type": "json_schema",
                                     "json_schema": {"name": "trechos", "strict": True, "schema": ESQUEMA_TRECHOS}}}
        return urllib.request.Request(URL, data=json.dumps(corpo).encode("utf-8"), method="POST",
                                      headers={"Authorization": f"Bearer {self._chave}",
                                               "Content-Type": "application/json", "User-Agent": AGENTE})

    def _chamar(self, pedido: urllib.request.Request) -> dict:
        with self._abrir(pedido, timeout=TEMPO_LIMITE) as resposta:
            return json.loads(resposta.read().decode("utf-8"))

    def redigir(self, sistema: str, usuario: str) -> str:
        for tentativa in (1, 2):
            try:
                dados = self._chamar(self._pedido(sistema, usuario))
                return str(dados["choices"][0]["message"]["content"]).strip()
            except urllib.error.HTTPError as erro:
                if erro.code != 429:
                    raise ErroLLM(f"o provedor respondeu HTTP {erro.code}") from None
                espera = _segundos(erro.headers.get("retry-after") if erro.headers else None)
                if tentativa == 2 or espera is None or espera > ESPERA_MAXIMA:
                    raise CotaEsgotada("limite de uso do provedor atingido") from None
                self._esperar(espera)
            except (urllib.error.URLError, TimeoutError, OSError) as erro:
                raise ErroLLM(f"provedor indisponível: {type(erro).__name__}") from None
            except (KeyError, IndexError, ValueError, TypeError):
                raise ErroLLM("resposta do provedor fora do formato") from None
        raise CotaEsgotada("limite de uso do provedor atingido")


def _segundos(valor: object) -> float | None:
    try:
        return max(0.0, float(valor))
    except (TypeError, ValueError):
        return None


def configurado(ambiente: Mapping[str, str] | None = None) -> Groq | None:
    """Redator configurado no ambiente, ou None (modo deterministico)."""
    ambiente = os.environ if ambiente is None else ambiente
    if ambiente.get("CASE_AMBEV_LLM", "").strip().lower() != "groq" or not ambiente.get("GROQ_API_KEY"):
        return None
    return Groq(ambiente["GROQ_API_KEY"], modelo=ambiente.get("CASE_AMBEV_LLM_MODELO", MODELO))
