"""Rede neural da jornada (RUR-Net mínima, PyTorch): encoder + termo linear + resíduo neural, cabeças por nó.

Arquitetura fiel ao desenho `_planejamento/.../ARQUITETURA_REDE_NEURAL.md` (cenário A): uma rede de utilidade por
alternativa, com um termo linear que carrega o sinal principal e um resíduo neural pequeno e regularizado por cima,
compartilhando um encoder de perfil+contexto. Duas cabeças com rótulo observado na POF 2008:

- `comida`: utilidade de pizza vs. outra refeição (denominador = refeições com comida);
- `bebida`: cabeça hierárquica (refri? → guaraná|refri) sobre as refeições com pizza.

O estado da jornada e as ofertas NÃO entram no encoder (seriam valores nunca vistos na POF): os deslocamentos de
cenário (`hipoteses`) são somados às utilidades DEPOIS, fora da rede. Profundidade: embeddings → bloco linear+ELU com
conexão residual → cabeças. "Profunda" descreve a arquitetura; o ganho sobre a referência é medido, não presumido.

Este módulo define a codificação das features (igual no treino e na inferência) e a rede. O treino fica em
`src/treinar_personas.py` (fora do pacote servido, porque lê a base de 2008). A inferência carrega o artefato e expõe
`Preditor`, usado pelo simulador.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from personas import grafo

# ---------- vocabulário das features (fixo; a mesma ordem no treino e na inferência) ----------

CATEGORICAS = {
    "grande_regiao": ("Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"),
    "area": ("capital", "urbano_nao_capital", "rural"),
    "faixa_etaria": ("10_17", "18_29", "30_44", "45_59", "60_mais"),
    "sexo": ("masculino", "feminino"),
    "escolaridade": ("ate_fundamental", "medio", "superior"),
    "periodo": ("manha", "almoco", "tarde", "noite"),
}
# Dia como fim de semana (sexta a domingo) e indicador de noite, como no modelo histórico (representação D2).
BINARIAS = ("fim_de_semana", "noite")
# Numéricas padronizadas (média/DP do treino); composição da casa e renda per capita em log.
NUMERICAS = ("log_moradores", "n_menores", "log_renda_pc")

_SEXTA_A_DOMINGO = frozenset((5, 6, 7))


def _uma_quente(valor: str, niveis: Sequence[str]) -> list[float]:
    if valor not in niveis:
        raise ValueError(f"nível desconhecido {valor!r} em {list(niveis)}")
    return [1.0 if valor == n else 0.0 for n in niveis]


def vetor_bruto(registro: Mapping[str, object]) -> tuple[list[float], list[float]]:
    """Features de uma observação em duas partes: categóricas/binárias (não padronizadas) e numéricas (a padronizar).
    A ordem é fixa. O estado da jornada e a oferta não entram aqui (ver docstring do módulo)."""
    categorico: list[float] = []
    for campo, niveis in CATEGORICAS.items():
        categorico += _uma_quente(str(registro[campo]), niveis)
    dia = int(registro["dia_semana"])
    periodo = str(registro["periodo"])
    categorico.append(1.0 if dia in _SEXTA_A_DOMINGO else 0.0)
    categorico.append(1.0 if periodo == "noite" else 0.0)
    numerico = [
        float(np.log1p(max(0, int(registro["n_moradores"])))),
        float(int(registro["n_menores"])),
        float(np.log1p(max(0.0, float(registro["renda_pc"])))),
    ]
    return categorico, numerico


def dimensao_categorica() -> int:
    return sum(len(n) for n in CATEGORICAS.values()) + len(BINARIAS)


@dataclass(frozen=True)
class Padronizador:
    """Média e desvio das numéricas, ajustados só no treino da dobra. Guardados no artefato."""
    media: tuple[float, ...]
    desvio: tuple[float, ...]

    @staticmethod
    def ajustar(numericas: np.ndarray) -> Padronizador:
        media = numericas.mean(axis=0)
        desvio = numericas.std(axis=0)
        desvio = np.where(desvio < 1e-8, 1.0, desvio)
        return Padronizador(tuple(float(v) for v in media), tuple(float(v) for v in desvio))

    def aplicar(self, numericas: np.ndarray) -> np.ndarray:
        return (numericas - np.array(self.media)) / np.array(self.desvio)


def matriz_features(registros: Sequence[Mapping], padronizador: Padronizador) -> np.ndarray:
    categorico, numerico = [], []
    for r in registros:
        c, n = vetor_bruto(r)
        categorico.append(c)
        numerico.append(n)
    cat = np.array(categorico, dtype=np.float32)
    num = padronizador.aplicar(np.array(numerico, dtype=np.float32)).astype(np.float32)
    return np.hstack([cat, num])


def dimensao_entrada() -> int:
    return dimensao_categorica() + len(NUMERICAS)


# ---------- a rede (PyTorch, importado só quando usado) ----------

def construir_rede(largura: int = 16, dropout: float = 0.2, semente: int = 0):
    """Cria a RUR-Net. Importa torch aqui para o resto do pacote (servidor, grafo, hipóteses) não exigir torch.

    Cada cabeça (comida, bebida) tem encoder e porta próprios: o treino de uma não altera os pesos da outra. Isso
    garante que a cabeça de pizza, treinada e fixada primeiro, seja reproduzível independentemente do treino da bebida."""
    import torch
    from torch import nn

    torch.manual_seed(semente)
    d_entrada = dimensao_entrada()

    class _Cabeca(nn.Module):
        """Encoder residual próprio + termo linear + resíduo neural com porta. Uma por nó aprendido."""

        def __init__(self, n_acoes: int):
            super().__init__()
            self.entrada = nn.Linear(d_entrada, largura)
            self.norma = nn.LayerNorm(largura)
            self.oculta = nn.Linear(largura, largura)
            self.dropout = nn.Dropout(dropout)
            self.linear = nn.Linear(d_entrada, n_acoes)        # termo linear (sinal principal)
            self.residuo = nn.Linear(largura, n_acoes)         # resíduo neural regularizado
            self.porta = nn.Parameter(torch.tensor(0.1))       # porta inicial pequena: começa perto do linear

        def forward(self, x):
            h0 = self.dropout(torch.nn.functional.elu(self.norma(self.entrada(x))))
            h = h0 + self.dropout(torch.nn.functional.elu(self.oculta(h0)))
            return self.linear(x) + self.porta * self.residuo(h)

    class RURNet(nn.Module):
        """Duas cabeças com encoder e porta PRÓPRIOS (não compartilhados), para que o treino de uma não altere a outra.

        Isso atende à exigência de reprodutibilidade: a cabeça de comida (pizza) é treinada e fixada primeiro; a cabeça
        de bebida (experimento) é treinada depois, sobre parâmetros exclusivos dela, sem mudar um único peso de pizza.

        Saídas (logits por ação):
        - comida: 2 logits (pizza, outra_refeicao);
        - bebida: logits das folhas de bebida (guaraná, outro refrigerante, sem refrigerante). A álgebra hierárquica e
          a máscara são aplicadas em `grafo`, fora da rede; a rede só entrega os logits.
        """

        def __init__(self):
            super().__init__()
            self.comida = _Cabeca(len(grafo.ACOES[grafo.COMIDA]))
            self.bebida = _Cabeca(len(grafo.ACOES[grafo.BEBIDA]))

        def forward(self, x):
            return {"comida": self.comida(x), "bebida": self.bebida(x)}

    return RURNet()

    return RURNet()


# ---------- inferência: carregar o artefato e prever ----------

VERSAO = "personas_rede_v1"
ARTEFATOS = Path(__file__).resolve().parents[2] / "artifacts" / VERSAO


@dataclass(frozen=True)
class Preditor:
    """Preditor da rede treinada: devolve as utilidades (logits) por nó para um registro de perfil+contexto.
    Funciona sem a base de treino; só precisa do artefato (pesos + padronizador + metadados)."""
    rede: object
    padronizador: Padronizador
    metadados: dict

    def utilidades(self, registro: Mapping[str, object]) -> dict[str, dict[str, float]]:
        import torch

        x = matriz_features([registro], self.padronizador)
        with torch.no_grad():
            saida = self.rede(torch.from_numpy(x))
        return {
            grafo.COMIDA: _por_acao(grafo.COMIDA, saida["comida"][0].numpy()),
            grafo.BEBIDA: _por_acao(grafo.BEBIDA, saida["bebida"][0].numpy()),
        }

    def taxa_historica_guarana(self) -> float:
        """Taxa histórica de guaraná entre as ocasiões com pizza (POF 2008), das contagens do treino. É a REFERÊNCIA de
        guaraná usada no chat: no nó de bebida a rede não supera essa taxa (ver avaliacao.json), então ela é a fonte
        confiável; a rede fica como experimento."""
        c = self.metadados["contagens"]
        return c["pizza_com_guarana"] / c["episodios_com_pizza"]


def _por_acao(no: str, logits: np.ndarray) -> dict[str, float]:
    return {acao: float(v) for acao, v in zip(grafo.ACOES[no], logits)}


def carregar(base: Path | str | None = None) -> Preditor:
    """Carrega a rede treinada do artefato. Confere a versão e reconstrói a arquitetura antes de ler os pesos."""
    import torch

    base = Path(base) if base else ARTEFATOS
    meta = json.loads((base / "meta.json").read_text(encoding="utf-8"))
    if meta["versao"] != VERSAO:
        raise ValueError(f"artefato de outra versão: {meta['versao']}")
    pad = Padronizador(tuple(meta["padronizador"]["media"]), tuple(meta["padronizador"]["desvio"]))
    rede = construir_rede(largura=meta["largura"], dropout=0.0, semente=0)
    estado = torch.load(base / "rede.pt", map_location="cpu", weights_only=True)
    rede.load_state_dict(estado)
    rede.eval()
    return Preditor(rede=rede, padronizador=pad, metadados=meta)
