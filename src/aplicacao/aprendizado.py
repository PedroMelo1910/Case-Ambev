"""Treino e avaliacao compartilhados pelo treino historico (POF 2008-2009) e pelo treino com dados da empresa.

- Referencias sem aprendizado de maquina no formato do scikit-learn: TaxaHabitual (a taxa do treino, igual para qualquer
  contexto) e FrequenciaPorCelula (taxa por celula, uma coluna com o codigo da celula, encolhida para a taxa do treino:
  (positivos + m * p0) / (n + m); celula sem observacao no treino recebe p0).
- Metricas de uma estimativa probabilistica: log-loss, Brier, calibracao, discriminacao e a faixa priorizada (os
  `cobertura` x n contextos de maior estimativa, com empates sorteados): precisao, captura e lift. Preditor constante nao
  ordena: AUC, AP, inclinacao de calibracao e faixa ficam None, sem valor inventado.
- Experimento controlado: cada candidato tem espaco de busca e orcamento explicitos (GridSearchCV ou RandomizedSearchCV,
  log-loss); busca e ajuste usam so o treino de cada divisao externa. Escolha por parcimonia: M* e o candidato de menor
  log-loss medio; escolhe-se o mais simples cuja diferenca de perda para M* tem intervalo de 95% (reamostragem de grupos)
  que nao fica inteiro acima de zero. O teste final, quando existe, nunca entra aqui.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------- referencias sem aprendizado de maquina ----------

def _proba(p: np.ndarray) -> np.ndarray:
    return np.column_stack([1.0 - p, p])


class TaxaHabitual(ClassifierMixin, BaseEstimator):
    def fit(self, x, y):
        y = np.asarray(y)
        self.classes_ = np.array([0, 1])
        self.taxa_ = float(np.mean(y))
        return self

    def predict_proba(self, x):
        return _proba(np.full(len(x), self.taxa_))

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


class FrequenciaPorCelula(ClassifierMixin, BaseEstimator):
    def __init__(self, m: float = 10.0):
        self.m = m

    @staticmethod
    def _celulas(x) -> np.ndarray:
        return np.asarray(x)[:, 0].astype(str)

    def fit(self, x, y):
        y = np.asarray(y)
        celulas = self._celulas(x)
        self.classes_ = np.array([0, 1])
        self.taxa_ = float(np.mean(y))
        self.taxas_ = {c: float((y[celulas == c].sum() + self.m * self.taxa_) / ((celulas == c).sum() + self.m))
                       for c in np.unique(celulas)}
        return self

    def predict_proba(self, x):
        return _proba(np.array([self.taxas_.get(c, self.taxa_) for c in self._celulas(x)], dtype=float))

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


# ---------- metricas ----------

EPS = 1e-12
SEMENTE_EMPATES = 20261117


def perdas_log(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _constante(p: np.ndarray) -> bool:
    return bool(np.ptp(p) < 1e-12)


def calibracao(y: np.ndarray, p: np.ndarray) -> dict | None:
    """Intercepto e inclinacao da regressao de y no logito de p (sem penalizacao)."""
    if _constante(p) or len(np.unique(y)) < 2:
        return None
    p = np.clip(p, EPS, 1 - EPS)
    logito = np.log(p / (1 - p)).reshape(-1, 1)
    ajuste = LogisticRegression(C=1e10, max_iter=10_000).fit(logito, y)
    return {"intercepto": float(ajuste.intercept_[0]), "inclinacao": float(ajuste.coef_[0, 0])}


def faixa_priorizada(y: np.ndarray, p: np.ndarray, referencia: float, cobertura: float) -> dict | None:
    if _constante(p):
        return None
    k = max(1, math.ceil(cobertura * len(y)))
    sorteio = np.random.default_rng(SEMENTE_EMPATES).permutation(len(p))  # empates nao seguem a ordem das linhas
    topo = sorteio[np.argsort(-p[sorteio], kind="mergesort")][:k]
    precisao = float(np.mean(y[topo]))
    total = float(np.sum(y))
    return {"cobertura": k / len(y), "contextos": int(k), "precisao": precisao,
            "captura": float(np.sum(y[topo]) / total) if total else None,
            "lift": precisao / referencia if referencia > 0 else None}


def _media(valores: list) -> object:
    if any(v is None for v in valores):
        return None
    if isinstance(valores[0], dict):
        return {chave: _media([v[chave] for v in valores]) for chave in valores[0]}
    return float(np.mean(valores))


def media_por_divisao(resumos: list[dict]) -> dict:
    """Media das metricas calculadas em cada divisao (nao das previsoes promediadas). Metrica ausente em alguma divisao
    (por exemplo, AUC de preditor constante) fica None, sem valor inventado."""
    saida = {chave: _media([r[chave] for r in resumos]) for chave in resumos[0] if chave not in ("n", "positivos")}
    return {**saida, "n_medio_por_divisao": float(np.mean([r["n"] for r in resumos])), "divisoes": len(resumos)}


def resumo(y, p, referencia: float, cobertura: float = 0.2) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    duas_classes = len(np.unique(y)) == 2
    ordena = duas_classes and not _constante(p)
    return {"n": int(len(y)), "positivos": int(y.sum()), "log_loss": float(np.mean(perdas_log(y, p))),
            "brier": float(np.mean((p - y) ** 2)), "o_e": float(y.sum() / p.sum()) if p.sum() > 0 else None,
            "auc": float(roc_auc_score(y, p)) if ordena else None,
            "ap": float(average_precision_score(y, p)) if ordena else None,
            "calibracao": calibracao(y, p), "faixa_priorizada": faixa_priorizada(y, p, referencia, cobertura)}


# ---------- experimento ----------

PONTUACAO = "neg_log_loss"
GRADE_C = tuple(float(10 ** e) for e in np.arange(-3.0, 2.01, 0.5))  # 11 valores, do mais regularizado ao menos
GRADE_M = (1.0, 5.0, 20.0, 50.0, 200.0)
ESPACO_BOOSTING = {"max_depth": [1, 2, 3], "max_iter": [50, 100, 200], "learning_rate": [0.03, 0.1],
                   "min_samples_leaf": [20, 50, 100], "l2_regularization": [0.0, 1.0]}
ITERACOES_BOOSTING = 10


@dataclass(frozen=True)
class Candidato:
    nome: str
    ordem: int  # complexidade crescente: desempata pela parcimonia
    ordena_contextos: bool
    descricao: str
    construir: Callable[[int], object]
    espaco: Mapping = field(default_factory=dict)
    busca: str | None = None  # "grade" ou "aleatoria"
    iteracoes: int | None = None

    def orcamento(self, dobras_internas: int | None) -> dict:
        combinacoes = (int(np.prod([len(v) for v in self.espaco.values()])) if self.busca == "grade"
                       else (self.iteracoes or 0))
        return {"busca": self.busca, "combinacoes": combinacoes, "dobras_internas": dobras_internas}


def _logistica(semente: int) -> Pipeline:
    return Pipeline([("escala", StandardScaler()),
                     ("logistica", LogisticRegression(l1_ratio=0, solver="lbfgs", max_iter=5000))])


def _boosting(semente: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(early_stopping=False, random_state=semente)


def candidatos_padrao(incluir_boosting: bool = True) -> tuple[Candidato, ...]:
    lista = [
        Candidato("habitual", 0, False, "taxa média do treino, igual para qualquer contexto", lambda s: TaxaHabitual()),
        Candidato("frequencia", 1, True, "frequência histórica por dia da semana e período, encolhida para a média",
                  lambda s: FrequenciaPorCelula(), {"m": list(GRADE_M)}, "grade"),
        Candidato("logistica", 2, True, "regressão logística com regularização ridge", _logistica,
                  {"logistica__C": list(GRADE_C)}, "grade"),
    ]
    if incluir_boosting:
        lista.append(Candidato("boosting", 3, True, "gradient boosting raso (árvores de baixa profundidade)", _boosting,
                               ESPACO_BOOSTING, "aleatoria", ITERACOES_BOOSTING))
    return tuple(lista)


def _parametros_publicos(params: Mapping) -> dict:
    return {chave.split("__")[-1]: valor for chave, valor in params.items()}


def ajustar(candidato: Candidato, x: np.ndarray, y: np.ndarray, grupos: np.ndarray | None,
            interno: Callable[[], object], semente: int) -> tuple[object, dict]:
    """Ajusta o candidato com a busca interna (so nestes dados) e devolve o estimador reajustado e os hiperparametros."""
    estimador = candidato.construir(semente)
    if candidato.busca is None:
        return estimador.fit(x, y), {}
    if candidato.busca == "grade":
        busca = GridSearchCV(estimador, dict(candidato.espaco), scoring=PONTUACAO, cv=interno(), error_score="raise")
    else:
        busca = RandomizedSearchCV(estimador, dict(candidato.espaco), n_iter=candidato.iteracoes, scoring=PONTUACAO,
                                   cv=interno(), random_state=semente, error_score="raise")
    busca.fit(x, y, groups=grupos)
    return busca.best_estimator_, _parametros_publicos(busca.best_params_)


def _cruzamentos(grupos: np.ndarray | None, externas: Sequence) -> int | None:
    if grupos is None:
        return None
    return int(sum(len(set(grupos[tr]) & set(grupos[te])) for tr, te in externas))


def avaliar(matrizes: Mapping[str, np.ndarray], y: np.ndarray, grupos: np.ndarray | None, externas: Sequence,
            interno: Callable[[], object], candidatos: Sequence[Candidato], semente: int, cobertura: float = 0.2) -> dict:
    """Previsoes fora da divisao (media quando um registro e testado em varias repeticoes), perdas por registro (media
    das perdas de cada divisao, a base da selecao) e metricas calculadas em cada divisao e depois promediadas."""
    y = np.asarray(y, dtype=int)
    n = len(y)
    cruzados = _cruzamentos(grupos, externas)
    if cruzados:
        raise ValueError(f"{cruzados} grupo(s) aparecem no treino e no teste da mesma divisao")
    saida: dict = {}
    vezes = np.zeros(n)
    for candidato in candidatos:
        x = np.asarray(matrizes[candidato.nome])
        soma, perdas, vezes, escolhas, por_divisao = np.zeros(n), np.zeros(n), np.zeros(n), [], []
        for k, (treino, teste) in enumerate(externas):
            g = None if grupos is None else grupos[treino]
            modelo, params = ajustar(candidato, x[treino], y[treino], g, interno, semente + k)
            p = modelo.predict_proba(x[teste])[:, 1]
            soma[teste] += p
            perdas[teste] += perdas_log(y[teste], p)
            vezes[teste] += 1
            escolhas.append(params)
            por_divisao.append(resumo(y[teste], p, float(np.mean(y[teste])), cobertura))
        avaliados = vezes > 0
        previsoes = np.full(n, np.nan)
        previsoes[avaliados] = soma[avaliados] / vezes[avaliados]
        perda_media = np.full(n, np.nan)
        perda_media[avaliados] = perdas[avaliados] / vezes[avaliados]
        saida[candidato.nome] = {
            "descricao": candidato.descricao, "ordem": candidato.ordem, "ordena_contextos": candidato.ordena_contextos,
            "previsoes": previsoes, "perdas": perda_media, "hiperparametros_por_dobra": escolhas,
            "orcamento": candidato.orcamento(getattr(interno(), "n_splits", None)),
            "metricas": media_por_divisao(por_divisao)}
    return {"candidatos": saida, "avaliados": vezes > 0,
            "auditoria": {"divisoes": len(externas), "grupos_nos_dois_lados": cruzados}}


def intervalo_por_grupos(diferencas: np.ndarray, grupos: np.ndarray | None, semente: int, replicas: int,
                         nivel: float = 0.95) -> list[float]:
    """Intervalo percentil da media de `diferencas`, reamostrando grupos inteiros (cada registro e um grupo se None)."""
    codigo = np.arange(len(diferencas)) if grupos is None else np.unique(grupos, return_inverse=True)[1]
    somas = np.bincount(codigo, weights=diferencas)
    tamanhos = np.bincount(codigo).astype(float)
    rng = np.random.default_rng(semente)
    total = len(somas)
    medias = np.empty(replicas)
    for r in range(replicas):
        pesos = np.bincount(rng.integers(0, total, size=total), minlength=total)
        medias[r] = pesos @ somas / (pesos @ tamanhos)
    cauda = (1 - nivel) / 2
    return [float(v) for v in np.quantile(medias, [cauda, 1 - cauda])]


def selecionar(resultado: dict, grupos: np.ndarray | None, semente: int, replicas: int = 2000) -> dict:
    """Regra de parcimonia sobre as perdas pareadas fora da divisao."""
    mascara = resultado["avaliados"]
    g = None if grupos is None else np.asarray(grupos)[mascara]
    candidatos = resultado["candidatos"]
    medias = {nome: float(np.mean(c["perdas"][mascara])) for nome, c in candidatos.items()}
    melhor = min(medias, key=medias.get)
    comparacoes = {}
    for nome, c in candidatos.items():
        d = c["perdas"][mascara] - candidatos[melhor]["perdas"][mascara]
        ic = [0.0, 0.0] if nome == melhor else intervalo_por_grupos(d, g, semente, replicas)
        comparacoes[nome] = {"diferenca_media": float(np.mean(d)), "ic": ic}
    ordem = sorted(candidatos, key=lambda nome: candidatos[nome]["ordem"])
    selecionado = next(nome for nome in ordem if comparacoes[nome]["ic"][0] <= 0)
    return {"menor_perda": melhor, "selecionado": selecionado, "log_loss_medio": medias, "comparacoes": comparacoes,
            "regra": ("parcimonia: o candidato mais simples cuja diferenca de log-loss para o de menor perda tem "
                      "intervalo de 95% (reamostragem de grupos) que nao fica inteiro acima de zero"),
            "replicas": replicas, "semente": semente}


def _parametros_internos(params: Mapping, nomes: Mapping) -> dict:
    """Recoloca o prefixo do passo (por exemplo, logistica__C) a partir do nome publico do hiperparametro."""
    return {next(n for n in nomes if n == publico or n.endswith(f"__{publico}")): valor for publico, valor in params.items()}


def reamostrar(candidato: Candidato, params: Mapping, x: np.ndarray, y: np.ndarray, grupos: np.ndarray | None,
               replicas: int, semente: int) -> list[tuple[object, float]]:
    """Modelos reajustados (hiperparametro fixo) em reamostras de grupos inteiros, com a taxa do treino de cada uma."""
    codigo = np.arange(len(y)) if grupos is None else np.unique(grupos, return_inverse=True)[1]
    membros = [np.flatnonzero(codigo == g) for g in range(codigo.max() + 1)]
    rng = np.random.default_rng(semente)
    modelos = []
    for r in range(replicas):
        indices = np.concatenate([membros[g] for g in rng.integers(0, len(membros), size=len(membros))])
        estimador = candidato.construir(semente + r)
        estimador.set_params(**_parametros_internos(params, estimador.get_params()))
        modelos.append((estimador.fit(x[indices], y[indices]), float(np.mean(y[indices]))))
    return modelos
