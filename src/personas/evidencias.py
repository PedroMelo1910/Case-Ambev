"""Recuperar evidencia: acervo curado (`dados/acervo.json`) indexado com SQLite FTS5 em memoria (BM25).

Nada e gravado em disco. Os cartoes sao contexto com fonte: nenhum entra no calculo. A busca expande sinonimos, ignora
palavras vazias e junta os cartoes cujas etiquetas casam com os temas do contexto. Sem termo que case, a lista volta
vazia e a resposta diz que faltou evidencia.
"""

import json
import sqlite3
from functools import lru_cache

from personas import referencias as ref

CAMPOS = ("id", "titulo", "texto", "etiquetas", "tipo_evidencia", "fonte", "url", "secao", "periodo", "territorio",
          "data_publicacao", "pode_enviar_externo")
TIPOS = ("medido_projeto", "estatistica_oficial", "literatura", "protocolo")
VAZIAS = frozenset("a o as os e de do da dos das em no na nos nas com por para um uma se que ou ao aos mais menos "
                   "como qual quais tem ter ser foi era vai voce eu me meu minha isso esse essa este esta e ja nao "
                   "sim muito pouco".split())
TEMA_ETIQUETAS = {"desconto": ("desconto",), "clima": ("clima",), "feriado": ("feriado",), "renda": ("renda",),
                  "territorio": ("territorio",), "marca": ("marca",), "compra": ("compra",)}


class Acervo:
    def __init__(self, dados: dict):
        self.versao = dados["versao"]
        self.sinonimos = {ref.normalizar(k).strip(): v for k, v in dados["sinonimos"].items()}
        self.cartoes = [self._conferir(c) for c in dados["cartoes"]]
        self._por_id = {c["id"]: c for c in self.cartoes}
        self._db = sqlite3.connect(":memory:", check_same_thread=False)
        self._db.execute("CREATE VIRTUAL TABLE cartao USING fts5(id UNINDEXED, titulo, texto, etiquetas, "
                         "tokenize='unicode61 remove_diacritics 2')")
        self._db.executemany("INSERT INTO cartao VALUES (?, ?, ?, ?)",
                             [(c["id"], c["titulo"], c["texto"], " ".join(c["etiquetas"])) for c in self.cartoes])

    @staticmethod
    def _conferir(cartao: dict) -> dict:
        if set(cartao) != set(CAMPOS) or cartao["tipo_evidencia"] not in TIPOS:
            raise ValueError(f"cartão fora do formato: {cartao.get('id')}")
        if cartao["url"] is not None and not cartao["url"].startswith("https://"):
            raise ValueError(f"cartão com link inseguro: {cartao['id']}")
        if "2017" in json.dumps(cartao, ensure_ascii=False):
            raise ValueError(f"cartão cita 2017 (ano da avaliação fechada): {cartao['id']}")
        return cartao

    def _termos(self, consulta: str) -> list[str]:
        texto = ref.normalizar(consulta)
        termos = [t for t in texto.split() if t not in VAZIAS and len(t) > 2]
        for chave, extras in self.sinonimos.items():
            if f" {chave} " in texto:
                termos += extras
        return list(dict.fromkeys(termos))

    def _bm25(self, termos: list[str], k: int) -> list[str]:
        if not termos:
            return []
        consulta = " OR ".join(f'"{t}"' for t in termos)
        linhas = self._db.execute("SELECT id FROM cartao WHERE cartao MATCH ? ORDER BY bm25(cartao) LIMIT ?",
                                  (consulta, k)).fetchall()
        return [i for (i,) in linhas]

    def buscar(self, consulta: str, temas: set, k: int = 4, maximo: int = 6) -> list[dict]:
        """BM25 da mensagem primeiro; depois os cartoes dos temas do contexto, ate `maximo`."""
        por_tema = list(dict.fromkeys(c["id"] for tema in sorted(temas) for c in self.cartoes
                                      if set(TEMA_ETIQUETAS.get(tema, ())) & set(c["etiquetas"])))
        ids = list(dict.fromkeys(self._bm25(self._termos(consulta), k) + por_tema))
        return [dict(self._por_id[i]) for i in ids[:maximo]]

    def cartao(self, ident: str) -> dict:
        return dict(self._por_id[ident])


@lru_cache(maxsize=1)
def carregar() -> Acervo:
    return Acervo(ref.carregar("acervo.json"))
