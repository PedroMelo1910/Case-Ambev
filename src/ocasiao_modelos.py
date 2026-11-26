"""Candidatos, metricas e validacao do Occasion Score observado (pizza e guarana na mesma refeicao, POF/IBGE).

Cada candidato recebe as colunas da refeicao (dicionario de arrays) e devolve P(pizza e guarana | contexto). Os mesmos
blocos de variaveis e as mesmas dobras valem para todos, para que a comparacao meca o que cada bloco acrescenta.
Protocolo: configs/protocolo_v3_alvo_observado.json.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

EPS = 1e-6
SEMENTE = 20261124
DOBRAS = 5
COBERTURA = 0.20
BINS_CALIBRACAO = 10

# Blocos cumulativos (protocolo v3): calendario, + territorio, + publico.
BLOCOS = {
    "B1": ("dia_semana", "periodo", "hora", "dia_periodo"),
    "B12": ("dia_semana", "periodo", "hora", "dia_periodo", "grande_regiao", "area"),
    "B123": ("dia_semana", "periodo", "hora", "dia_periodo", "grande_regiao", "area", "faixa_etaria", "sexo",
             "escolaridade"),
}
ORDEM_DE_SIMPLICIDADE = ("global", "dia_periodo", "logistica", "dois_passos", "boosting")


def colunas(dados: Mapping[str, np.ndarray], nomes: Sequence[str]) -> np.ndarray:
    """Matriz de texto com as colunas pedidas; `dia_periodo` e a interacao dia x periodo."""
    def coluna(nome: str) -> np.ndarray:
        if nome == "dia_periodo":
            return np.char.add(np.char.add(dados["dia_semana"].astype(str), "|"), dados["periodo"].astype(str))
        return dados[nome].astype(str)
    return np.column_stack([coluna(n) for n in nomes]).astype(object)


# ---------- candidatos ----------

@dataclass
class Global:
    nome: str = "global"
    taxa: float = 0.0

    def ajustar(self, dados, y):
        self.taxa = float(np.mean(y))
        return self

    def prever(self, dados):
        return np.full(len(dados["dia_semana"]), self.taxa)


@dataclass
class DiaPeriodo:
    """Frequencia por dia x periodo, suavizada para a taxa global com `m` refeicoes; celula inedita = taxa global."""
    m: float = 0.0
    nome: str = "dia_periodo"
    taxas: dict = field(default_factory=dict)
    taxa_global: float = 0.0
    contagens: dict = field(default_factory=dict)

    def ajustar(self, dados, y):
        celulas = colunas(dados, ["dia_periodo"])[:, 0]
        self.taxa_global = float(np.mean(y))
        chaves, inverso = np.unique(celulas, return_inverse=True)
        n = np.bincount(inverso)
        pos = np.bincount(inverso, weights=y)
        self.taxas = {c: float((p + self.m * self.taxa_global) / (k + self.m)) if k + self.m > 0 else self.taxa_global
                      for c, k, p in zip(chaves, n, pos, strict=True)}
        self.contagens = {c: (int(k), int(p)) for c, k, p in zip(chaves, n, pos, strict=True)}
        return self

    def prever(self, dados):
        celulas = colunas(dados, ["dia_periodo"])[:, 0]
        return np.array([self.taxas.get(c, self.taxa_global) for c in celulas])


@dataclass
class Logistica:
    bloco: str = "B1"
    C: float = 0.1
    nome: str = "logistica"
    modelo: object = None

    def ajustar(self, dados, y):
        self.modelo = make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                                    LogisticRegression(C=self.C, max_iter=3000))
        self.modelo.fit(colunas(dados, BLOCOS[self.bloco]), y)
        return self

    def prever(self, dados):
        return self.modelo.predict_proba(colunas(dados, BLOCOS[self.bloco]))[:, 1]


@dataclass
class DoisPassos:
    """P(pizza | x) x P(guarana | pizza, x), com duas logisticas regularizadas no mesmo bloco."""
    bloco: str = "B1"
    C_pizza: float = 0.1
    C_guarana: float = 0.1
    nome: str = "dois_passos"
    pizza: object = None
    guarana: object = None

    def ajustar(self, dados, y):
        self.pizza = Logistica(self.bloco, self.C_pizza).ajustar(dados, dados["pizza"])
        com_pizza = dados["pizza"] == 1
        sub = {k: v[com_pizza] for k, v in dados.items()}
        self.guarana = Logistica(self.bloco, self.C_guarana).ajustar(sub, dados["guarana"][com_pizza])
        return self

    def componentes(self, dados):
        return self.pizza.prever(dados), self.guarana.prever(dados)

    def prever(self, dados):
        p, g = self.componentes(dados)
        return p * g


@dataclass
class Boosting:
    bloco: str = "B123"
    taxa_aprendizado: float = 0.05
    folhas: int = 15
    l2: float = 1.0
    nome: str = "boosting"
    modelo: object = None

    def ajustar(self, dados, y):
        n = len(BLOCOS[self.bloco])
        arvores = HistGradientBoostingClassifier(
            learning_rate=self.taxa_aprendizado, max_leaf_nodes=self.folhas, l2_regularization=self.l2,
            max_iter=300, min_samples_leaf=200, early_stopping=False, categorical_features=[True] * n,
            random_state=SEMENTE)
        self.modelo = make_pipeline(OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                                   encoded_missing_value=-1), arvores)
        self.modelo.fit(colunas(dados, BLOCOS[self.bloco]), y)
        return self

    def prever(self, dados):
        return self.modelo.predict_proba(colunas(dados, BLOCOS[self.bloco]))[:, 1]


# ---------- metricas ----------

def perda_log(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def priorizacao(y: np.ndarray, p: np.ndarray, cobertura: float = COBERTURA) -> dict:
    """Exatamente `cobertura` das refeicoes com maior score; empate na fronteira entra por fracao (valor esperado)."""
    n, positivos = len(y), float(np.sum(y))
    k = cobertura * n
    _, inverso = np.unique(-np.asarray(p, dtype=float), return_inverse=True)  # grupos em ordem decrescente de score
    tamanho = np.bincount(inverso)
    pos = np.bincount(inverso, weights=y)
    acumulado = np.cumsum(tamanho)
    cheio = int(np.searchsorted(acumulado, k, side="right"))  # grupos inteiramente dentro
    capturados = float(pos[:cheio].sum())
    dentro = float(acumulado[cheio - 1]) if cheio else 0.0
    if cheio < len(tamanho) and k > dentro:
        capturados += float(pos[cheio]) * (k - dentro) / float(tamanho[cheio])
    taxa = positivos / n
    precisao = capturados / k
    tp, fp, fn = capturados, k - capturados, positivos - capturados
    return {"captura": capturados / positivos if positivos else None, "positivos_capturados": capturados,
            "precisao": precisao, "lift": precisao / taxa if taxa else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if positivos else None,
            "matriz_confusao": {"vp": tp, "fp": fp, "fn": fn, "vn": n - k - fn}}


def calibracao(y: np.ndarray, p: np.ndarray, bins: int = BINS_CALIBRACAO) -> dict:
    ordem = np.argsort(p, kind="mergesort")
    grupos = [g for g in np.array_split(ordem, bins) if len(g)]
    tabela = [{"previsto": float(np.mean(p[g])), "observado": float(np.mean(y[g])), "n": int(len(g)),
               "positivos": int(np.sum(y[g]))} for g in grupos]
    ece = sum(t["n"] * abs(t["previsto"] - t["observado"]) for t in tabela) / len(y)
    return {"ece": float(ece), "por_decil": tabela, "observado_sobre_esperado": float(np.sum(y) / np.sum(p))}


def metricas(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    variavel = np.ptp(p) > 1e-15
    return {"n": int(len(y)), "positivos": int(np.sum(y)), "perda_log": perda_log(y, p),
            "brier": float(np.mean((p - y) ** 2)),
            "pr_auc": float(average_precision_score(y, p)) if variavel else float(np.mean(y)),
            "roc_auc": float(roc_auc_score(y, p)) if variavel else 0.5,
            **{f"{k}_20": v for k, v in priorizacao(y, p).items()}, "calibracao": calibracao(y, p)}


# ---------- validacao cruzada agrupada ----------

def dobras(grupos: np.ndarray, n_dobras: int = DOBRAS) -> list[tuple[np.ndarray, np.ndarray]]:
    divisor = GroupKFold(n_splits=n_dobras, shuffle=True, random_state=SEMENTE)
    return list(divisor.split(np.zeros(len(grupos)), groups=grupos))


def subconjunto(dados: Mapping[str, np.ndarray], indices: np.ndarray) -> dict:
    return {k: v[indices] for k, v in dados.items()}


def validar(fabrica: Callable[[], object], dados: Mapping[str, np.ndarray], y: np.ndarray,
            divisoes: Sequence[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Previsoes fora da dobra e perda por dobra para um candidato com hiperparametros fixos."""
    oof = np.zeros(len(y))
    por_dobra = []
    for treino, valida in divisoes:
        modelo = fabrica().ajustar(subconjunto(dados, treino), y[treino])
        oof[valida] = modelo.prever(subconjunto(dados, valida))
        por_dobra.append(perda_log(y[valida], oof[valida]))
    return {"oof": oof, "perda_por_dobra": por_dobra, "perda_media": float(np.mean(por_dobra))}


def escolher(resultados: Mapping[str, dict]) -> dict:
    """Regra do protocolo v3: menor perda media; vantagem menor que 1 erro-padrao da diferenca pareada sobre um
    candidato mais simples devolve a escolha ao mais simples."""
    familias = sorted(resultados, key=ORDEM_DE_SIMPLICIDADE.index)
    melhor = min(familias, key=lambda f: resultados[f]["perda_media"])
    trilha = []
    for simples in familias[:familias.index(melhor)]:
        dif = np.array(resultados[melhor]["perda_por_dobra"]) - np.array(resultados[simples]["perda_por_dobra"])
        ep = float(np.std(dif, ddof=1) / np.sqrt(len(dif)))
        sustentada = bool(dif.mean() < -ep)
        trilha.append({"contra": simples, "diferenca_media": float(dif.mean()), "erro_padrao": ep,
                       "vantagem_sustentada": sustentada})
        if not sustentada:
            return {"escolhido": simples, "melhor_pontual": melhor, "trilha": trilha}
    return {"escolhido": melhor, "melhor_pontual": melhor, "trilha": trilha}


def bootstrap_por_upa(y: np.ndarray, previsoes: Mapping[str, np.ndarray], upa: np.ndarray, reps: int = 1000,
                      semente: int = SEMENTE) -> dict:
    """Intervalos de 95% (reamostragem de UPAs) para perda log, PR-AUC e captura nos 20%; as diferencas sao contra o
    ultimo nome de `previsoes` (a referencia)."""
    rng = np.random.default_rng(semente)
    _, codigo = np.unique(upa, return_inverse=True)
    ordem = np.argsort(codigo, kind="mergesort")
    limites = np.cumsum(np.bincount(codigo))[:-1]
    membros = np.split(ordem, limites)
    nomes = list(previsoes)
    amostras = {n: {"perda_log": [], "pr_auc": [], "captura_20": []} for n in nomes}
    for _ in range(reps):
        idx = np.concatenate([membros[g] for g in rng.integers(0, len(membros), len(membros))])
        yb = y[idx]
        if yb.sum() == 0:
            continue
        for n in nomes:
            pb = previsoes[n][idx]
            amostras[n]["perda_log"].append(perda_log(yb, pb))
            amostras[n]["pr_auc"].append(float(average_precision_score(yb, pb)) if np.ptp(pb) > 0 else float(yb.mean()))
            amostras[n]["captura_20"].append(priorizacao(yb, pb)["captura"])

    def ic(valores) -> list[float]:
        return [float(np.quantile(valores, 0.025)), float(np.quantile(valores, 0.975))]

    saida = {n: {m: ic(v) for m, v in d.items()} for n, d in amostras.items()}
    if len(nomes) > 1:
        ref = nomes[-1]
        saida["diferencas_contra_" + ref] = {
            n: {m: ic(np.array(amostras[n][m]) - np.array(amostras[ref][m])) for m in amostras[n]} for n in nomes[:-1]}
    return saida
