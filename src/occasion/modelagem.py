"""Funcoes do protocolo de modelagem (`configs/protocolo_modelagem.json`): divisoes, baselines, logistica, selecao
dentro das dobras, comparacao pareada, limiar de cobertura, classes, calibracao, P5, permutacao e artefatos.

Tudo aqui e funcao pura sobre arrays: nada le arquivo de dados. A leitura e a execucao ficam em `modelagem_2008.py`.
"""

import hashlib
import io
import json
import warnings
from dataclasses import dataclass
from itertools import combinations, permutations
from pathlib import Path

import joblib
import numpy as np
import sklearn
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning, InconsistentVersionWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EPS = np.finfo(float).eps
CASAS_EMPATE = 12
ARTEFATOS_JSON = {"B0", "B1", "B2"}
ARTEFATOS_JOBLIB = {"L", "G"}


class ArtefatoInvalido(ValueError):
    """Artefato com hash ou versao diferente do registrado nos metadados."""


@dataclass(frozen=True)
class Dados:
    """Matriz da logistica, alvo, grupos (UPA), celulas de B1/B2 e pesos W1 (None quando nao calculados)."""
    x: np.ndarray
    y: np.ndarray
    grupos: np.ndarray
    cel_b1: np.ndarray
    cel_b2: np.ndarray
    w1: np.ndarray | None = None


def carregar_protocolo(caminho: Path) -> dict:
    return json.loads(Path(caminho).read_text(encoding="utf-8"))


def sha256_arquivo(caminho: Path) -> str:
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


# ---------- divisoes ----------

def particoes(y: np.ndarray, grupos: np.ndarray, n_dobras: int, semente: int) -> list[tuple[np.ndarray, np.ndarray]]:
    divisor = StratifiedGroupKFold(n_splits=n_dobras, shuffle=True, random_state=semente)
    return list(divisor.split(np.zeros(len(y)), y, grupos))


def dobra_por_episodio(parts: list[tuple[np.ndarray, np.ndarray]], n: int) -> np.ndarray:
    dobra = np.full(n, -1)
    for k, (_, teste) in enumerate(parts):
        dobra[teste] = k
    return dobra


def _coatribuicao(dobra_upa: np.ndarray) -> np.ndarray:
    iguais = dobra_upa[:, None] == dobra_upa[None, :]
    return iguais[np.triu_indices(len(dobra_upa), 1)]


def diversidade(dobras: list[np.ndarray], grupos: np.ndarray, y: np.ndarray) -> dict:
    """Fracao de pares de UPAs cuja condicao 'mesma dobra / dobras diferentes' muda entre duas repeticoes."""
    _, primeiro, inverso = np.unique(grupos, return_index=True, return_inverse=True)
    com_positivo = np.bincount(inverso, weights=y) > 0
    resumo = {"pares_identicos": 0}
    fracoes = {"com_positivo": [], "sem_positivo": []}
    for a, b in combinations(range(len(dobras)), 2):
        da, db = dobras[a][primeiro], dobras[b][primeiro]
        total = np.mean(_coatribuicao(da) != _coatribuicao(db))
        resumo["pares_identicos"] += int(total == 0)
        for nome, mascara in (("com_positivo", com_positivo), ("sem_positivo", ~com_positivo)):
            fracoes[nome].append(float(np.mean(_coatribuicao(da[mascara]) != _coatribuicao(db[mascara]))))
    for nome, valores in fracoes.items():
        v = np.array(valores) if valores else np.zeros(1)
        resumo[f"fracao_mudanca_{nome}"] = {"min": float(v.min()), "mediana": float(np.median(v)), "max": float(v.max())}
    return resumo


# ---------- baselines e logistica ----------

def ajustar_celulas(celulas: np.ndarray, y: np.ndarray, m_: float) -> dict:
    p0 = float(np.mean(y))
    taxas = {}
    for celula in np.unique(celulas):
        sel = celulas == celula
        taxas[str(celula)] = float((y[sel].sum() + m_ * p0) / (sel.sum() + m_))
    return {"p0": p0, "m": float(m_), "taxas": taxas}


def prever_celulas(modelo: dict, celulas: np.ndarray) -> np.ndarray:
    return np.array([modelo["taxas"].get(str(c), modelo["p0"]) for c in celulas], dtype=float)


def normalizar_pesos(w: np.ndarray) -> np.ndarray:
    return np.asarray(w, dtype=float) / np.mean(w)


def limites_aparo(w_treino: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(w_treino, [1, 99])
    return float(lo), float(hi)


def aparar_pesos(w: np.ndarray, limites: tuple[float, float] | None = None) -> np.ndarray:
    lo, hi = limites if limites is not None else limites_aparo(w)
    return np.clip(np.asarray(w, dtype=float), lo, hi)


def ajustar_logistica(x: np.ndarray, y: np.ndarray, C: float, pesos: np.ndarray | None = None,
                      max_iter: int = 10000, tol: float = 1e-6):
    """Ridge (l1_ratio=0) com lbfgs, intercepto nao penalizado; escala ajustada so em `x`, sem pesos. Nao convergir e erro."""
    modelo = make_pipeline(StandardScaler(), LogisticRegression(l1_ratio=0.0, solver="lbfgs", C=C, fit_intercept=True,
                                                                max_iter=max_iter, tol=tol))
    params = {} if pesos is None else {"logisticregression__sample_weight": normalizar_pesos(pesos)}
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        modelo.fit(x, y, **params)
    return modelo


def _pesos_divisao(dados: Dados, treino: np.ndarray, teste: np.ndarray, modo: str | None):
    """Pesos de ajuste (treino) e de perda (teste) para A1 ('w1') e A1t ('w1t'); aparo pelos percentis do treino."""
    if modo is None:
        return None, None
    if dados.w1 is None:
        raise ValueError(f"modo de peso {modo!r} exige W1")
    w_tr, w_te = dados.w1[treino], dados.w1[teste]
    if modo == "w1t":
        limites = limites_aparo(w_tr)
        w_tr, w_te = aparar_pesos(w_tr, limites), aparar_pesos(w_te, limites)
    return w_tr, w_te


def ajustar(tipo: str, param, dados: Dados, treino: np.ndarray, colunas=None, modo_peso: str | None = None):
    if tipo == "B0":
        return {"p0": float(np.mean(dados.y[treino]))}
    if tipo in ("B1", "B2"):
        cel = dados.cel_b1 if tipo == "B1" else dados.cel_b2
        return ajustar_celulas(cel[treino], dados.y[treino], param)
    x = dados.x if colunas is None else dados.x[:, colunas]
    w_tr, _ = _pesos_divisao(dados, treino, treino, modo_peso)
    return ajustar_logistica(x[treino], dados.y[treino], C=10.0 ** param, pesos=w_tr)


def prever(tipo: str, modelo, dados: Dados, indices: np.ndarray, colunas=None) -> np.ndarray:
    if tipo == "B0":
        return np.full(len(indices), modelo["p0"])
    if tipo in ("B1", "B2"):
        return prever_celulas(modelo, (dados.cel_b1 if tipo == "B1" else dados.cel_b2)[indices])
    x = dados.x if colunas is None else dados.x[:, colunas]
    return modelo.predict_proba(x[indices])[:, 1]


def perdas_log(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def perda_log(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
    return float(np.average(perdas_log(y, p), weights=w))


def escolher_na_grade(perdas: list[float], tolerancia: float) -> int:
    """Grade ordenada do mais para o menos regularizado: o primeiro dentro da tolerancia do minimo."""
    if not perdas or not np.all(np.isfinite(perdas)):
        raise ValueError(f"perdas internas vazias ou nao finitas: {perdas}")
    melhor = min(perdas)
    return next(i for i, v in enumerate(perdas) if v - melhor < tolerancia)


def perdas_internas(tipo: str, grade: list, dados: Dados, idx: np.ndarray, semente: int, n_dobras: int,
                    colunas=None, modo_peso: str | None = None) -> tuple[list[float], int]:
    """Log-loss das previsoes internas fora da dobra (juntando as dobras) para cada valor da grade, e o menor numero de
    positivos numa dobra interna."""
    parts = particoes(dados.y[idx], dados.grupos[idx], n_dobras, semente)
    menor_k = min(int(dados.y[idx][te].sum()) for _, te in parts)
    pesos_perda = np.empty(len(idx)) if modo_peso else None
    previsoes = {i: np.empty(len(idx)) for i in range(len(grade))}
    for tr, te in parts:
        treino, teste = idx[tr], idx[te]
        if modo_peso:
            pesos_perda[te] = _pesos_divisao(dados, treino, teste, modo_peso)[1]
        for i, valor in enumerate(grade):
            modelo = ajustar(tipo, valor, dados, treino, colunas, modo_peso)
            previsoes[i][te] = prever(tipo, modelo, dados, teste, colunas)
    perdas = [perda_log(dados.y[idx], previsoes[i], pesos_perda) for i in range(len(grade))]
    return perdas, menor_k


def validacao_cruzada(tipo: str, grade: list, dados: Dados, sementes_externas: list[int], semente_interna, n_dobras: int,
                      tolerancia: float, colunas=None, modo_peso: str | None = None) -> dict:
    """Previsoes fora da dobra (R x n) com selecao do hiperparametro dentro de cada dobra externa."""
    n = len(dados.y)
    previsoes = np.full((len(sementes_externas), n), np.nan)
    escolhas = []
    menor_k_interno = n
    for r, semente in enumerate(sementes_externas):
        for k, (treino, teste) in enumerate(particoes(dados.y, dados.grupos, n_dobras, semente)):
            if len(grade) > 1:
                perdas, menor_k = perdas_internas(tipo, grade, dados, treino, semente_interna(r, k), n_dobras, colunas,
                                                  modo_peso)
                menor_k_interno = min(menor_k_interno, menor_k)
                valor = grade[escolher_na_grade(perdas, tolerancia)]
            else:
                valor = grade[0]
            escolhas.append(valor)
            modelo = ajustar(tipo, valor, dados, treino, colunas, modo_peso)
            previsoes[r, teste] = prever(tipo, modelo, dados, teste, colunas)
    return {"previsoes": previsoes, "escolhas": escolhas, "menor_k_interno": menor_k_interno}


def selecao_repetida(tipo: str, grade: list, dados: Dados, idx: np.ndarray, sementes: list[int], n_dobras: int,
                     tolerancia: float, colunas=None, modo_peso: str | None = None) -> tuple:
    """Hiperparametro do reajuste unico: menor log-loss interna media nas repeticoes."""
    if len(grade) == 1:
        return grade[0], [0.0]
    medias = np.mean([perdas_internas(tipo, grade, dados, idx, s, n_dobras, colunas, modo_peso)[0] for s in sementes],
                     axis=0)
    return grade[escolher_na_grade(list(medias), tolerancia)], [float(v) for v in medias]


# ---------- comparacao e incerteza ----------

def contagens_bootstrap(grupos: np.ndarray, replicas: int, semente: int) -> np.ndarray:
    """Matriz replicas x episodios: quantas vezes a UPA de cada episodio foi sorteada (UPAs inteiras)."""
    _, inverso = np.unique(grupos, return_inverse=True)
    n_upas = inverso.max() + 1
    sorteio = np.random.default_rng(semente).integers(0, n_upas, size=(replicas, n_upas))
    contagem = np.zeros((replicas, n_upas))
    np.add.at(contagem, (np.repeat(np.arange(replicas), n_upas), sorteio.ravel()), 1)
    return contagem[:, inverso]


def intervalo(valores: np.ndarray, nivel: float = 0.95) -> tuple[float, float]:
    a = (1 - nivel) / 2
    lo, hi = np.quantile(valores, [a, 1 - a])  # interpolacao linear = quantil tipo 7
    return float(lo), float(hi)


def delta_pareado(perda_a: np.ndarray, perda_b: np.ndarray, grupos: np.ndarray, replicas: int = 5000,
                  semente: int = 0, contagens: np.ndarray | None = None) -> dict:
    d = np.asarray(perda_a) - np.asarray(perda_b)
    w = contagens if contagens is not None else contagens_bootstrap(grupos, replicas, semente)
    return {"delta": float(d.mean()), "ic": intervalo(w @ d / w.sum(axis=1))}


def escolher_modelo(medias: dict, ics: dict, ordem: list[str]) -> dict:
    """M* = menor log-loss; escolhido = o mais simples que M* nao supera (IC de delta(M* - candidato) nao inteiro < 0)."""
    faltando = [c for c in medias if c not in ics]
    if faltando:
        raise ValueError(f"sem intervalo contra M* para {faltando}")
    m_estrela = min(medias, key=medias.get)
    for candidato in [c for c in ordem if c in medias]:
        if candidato == m_estrela or ics[candidato][1] >= 0:
            return {"m_estrela": m_estrela, "escolhido": candidato}
    return {"m_estrela": m_estrela, "escolhido": m_estrela}


# ---------- limiar, classes e metricas ----------

def limiar_cobertura(s: np.ndarray, q: float, casas: int = CASAS_EMPATE) -> dict:
    r = np.round(np.asarray(s, dtype=float), casas)
    valores, contagens = np.unique(r, return_counts=True)
    valores, contagens = valores[::-1], contagens[::-1]
    acumulada = np.cumsum(contagens) / len(r)
    if acumulada[0] > q:
        return {"q": q, "t": None, "cobertura": 0.0, "grupo_topo": int(contagens[0]),
                "cobertura_minima_necessaria": float(acumulada[0])}
    i = int(np.nonzero(acumulada <= q)[0][-1])
    return {"q": q, "t": float(valores[i]), "cobertura": float(acumulada[i]), "grupo_topo": int(contagens[0]),
            "grupo_em_t": int(contagens[i]), "grupo_abaixo": int(contagens[i + 1]) if i + 1 < len(contagens) else 0}


def sinalizados(s: np.ndarray, t: float | None) -> np.ndarray:
    if t is None:
        return np.zeros(len(s), dtype=bool)
    return np.round(np.asarray(s, dtype=float), CASAS_EMPATE) >= t


def classes(s: np.ndarray, t30: float | None, b: float) -> np.ndarray:
    alta = sinalizados(s, t30)
    return np.where(alta, "alta", np.where(np.asarray(s) >= b, "media", "baixa"))


def metricas_no_limiar(y: np.ndarray, alta: np.ndarray) -> dict:
    vp, fp = int(np.sum(alta & (y == 1))), int(np.sum(alta & (y == 0)))
    fn, vn = int(np.sum(~alta & (y == 1))), int(np.sum(~alta & (y == 0)))
    precisao = vp / (vp + fp) if vp + fp else None
    recall = vp / (vp + fn) if vp + fn else None
    if precisao is None or recall is None:
        f1 = None
    else:
        f1 = 2 * precisao * recall / (precisao + recall) if precisao + recall > 0 else 0.0
    base = float(np.mean(y))
    lift = precisao / base if precisao is not None and base > 0 else None
    return {"vp": vp, "fp": fp, "fn": fn, "vn": vn, "cobertura": float(np.mean(alta)), "precisao": precisao,
            "recall": recall, "f1": f1, "lift": lift}


def _newton(matriz: np.ndarray, y: np.ndarray, w: np.ndarray, offset: np.ndarray, iteracoes: int = 50) -> np.ndarray:
    """Logistica sem penalizacao, vetorizada em replicas: w (R x n), matriz (n x p). Devolve coeficientes (R x p)."""
    beta = np.zeros((w.shape[0], matriz.shape[1]))
    for _ in range(iteracoes):
        mu = 1 / (1 + np.exp(-(beta @ matriz.T + offset)))
        grad = (w * (y - mu)) @ matriz
        peso = w * mu * (1 - mu)
        hess = np.einsum("rn,ni,nj->rij", peso, matriz, matriz)
        passo = np.linalg.solve(hess, grad[..., None])[..., 0]
        beta = beta + passo
        if np.max(np.abs(passo)) < 1e-10:
            break
    return beta


def calibracao(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> dict:
    """O/E, intercepto com inclinacao fixa em 1 (calibracao no todo) e inclinacao de calibracao; w = (R x n) ou None."""
    unico = w is None
    w = np.ones((1, len(y))) if unico else np.atleast_2d(w)
    lp = np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1))
    o_e = (w @ y) / (w @ p)
    intercepto = _newton(np.ones((len(y), 1)), y, w, lp)[:, 0]
    inclinacao = _newton(np.column_stack([np.ones(len(y)), lp]), y, w, np.zeros(len(y)))[:, 1]
    if unico:
        return {"o_e": float(o_e[0]), "intercepto_no_todo": float(intercepto[0]), "inclinacao": float(inclinacao[0])}
    return {"o_e": o_e, "intercepto_no_todo": intercepto, "inclinacao": inclinacao}


# ---------- P5 ----------

def _alta_repeticao(p: np.ndarray, q: float) -> np.ndarray:
    return sinalizados(p, limiar_cobertura(p, q)["t"])


def _quartis(v: list[float]) -> list[float]:
    return [float(x) for x in np.quantile(v, [0, 0.25, 0.5, 0.75, 1])]


def p5(a0: np.ndarray, a1: np.ndarray, q: float) -> dict:
    """Diagnostico descritivo: A0 contra A1 sempre entre particoes diferentes, com a regua A0 contra A0."""
    alta0 = [_alta_repeticao(p, q) for p in a0]
    alta1 = [_alta_repeticao(p, q) for p in a1]
    d_x = [float(np.mean(alta0[r] != alta1[s])) for r, s in permutations(range(len(a0)), 2)]
    rho_x = [float(spearmanr(a0[r], a1[s]).statistic) for r, s in permutations(range(len(a0)), 2)]
    d_0 = [float(np.mean(alta0[r] != alta0[s])) for r, s in combinations(range(len(a0)), 2)]
    rho_0 = [float(spearmanr(a0[r], a0[s]).statistic) for r, s in combinations(range(len(a0)), 2)]
    relevante = bool(np.median(d_x) > max(d_0) and np.median(rho_x) < min(rho_0))
    return {"d_cruzada": d_x, "rho_cruzada": rho_x, "d_regua": d_0, "rho_regua": rho_0,
            "quartis": {"d_cruzada": _quartis(d_x), "d_regua": _quartis(d_0), "rho_cruzada": _quartis(rho_x),
                        "rho_regua": _quartis(rho_0)},
            "convencao_relevante": relevante}


# ---------- permutacao ----------

def permutar_por_upa(y: np.ndarray, grupos: np.ndarray, semente: int) -> np.ndarray:
    """Troca os vetores de rotulos entre UPAs com o mesmo numero de episodios (agrupamento preservado)."""
    rng = np.random.default_rng(semente)
    upas = np.unique(grupos)
    indices = {u: np.nonzero(grupos == u)[0] for u in upas}
    novo = np.array(y, copy=True)
    for tamanho in sorted({len(v) for v in indices.values()}):
        mesmas = [u for u in upas if len(indices[u]) == tamanho]
        for destino, origem in zip(mesmas, rng.permutation(len(mesmas))):
            novo[indices[destino]] = y[indices[mesmas[origem]]]
    return novo


# ---------- artefatos ----------

def salvar_artefato(modelo, nome: str, pasta: Path, metadados: dict) -> Path:
    """JSON para B0/B1/B2; joblib para estimadores do scikit-learn. Metadados com SHA-256 e versao ao lado."""
    if nome not in ARTEFATOS_JSON | ARTEFATOS_JOBLIB:
        raise ArtefatoInvalido(f"nome de artefato invalido: {nome!r}")
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    if nome in ARTEFATOS_JSON:
        caminho = pasta / f"{nome}.json"
        caminho.write_text(json.dumps(modelo, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8")
    else:
        caminho = pasta / f"{nome}.joblib"
        joblib.dump(modelo, caminho)
    meta = {**metadados, "arquivo": caminho.name, "sha256": sha256_arquivo(caminho), "sklearn": sklearn.__version__}
    (pasta / f"{nome}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return caminho


def carregar_artefato(caminho: Path, nome: str, sha256_esperado: str | None = None):
    """Carrega so artefatos deste pipeline. Le os bytes uma vez, confere o SHA-256 (o externo, se informado, prevalece
    sobre o dos metadados) e a versao do scikit-learn, e so entao desserializa os mesmos bytes."""
    if nome not in ARTEFATOS_JSON | ARTEFATOS_JOBLIB:
        raise ArtefatoInvalido(f"nome de artefato invalido: {nome!r}")
    caminho = Path(caminho)
    try:
        meta = json.loads((caminho.parent / f"{nome}.meta.json").read_text(encoding="utf-8"))
        registrado, versao = meta["sha256"], meta["sklearn"]
    except (OSError, ValueError, KeyError) as erro:
        raise ArtefatoInvalido(f"{caminho.name}: metadados ausentes ou incompletos ({erro})") from erro
    conteudo = caminho.read_bytes()
    obtido = hashlib.sha256(conteudo).hexdigest()
    if obtido != registrado or (sha256_esperado is not None and obtido != sha256_esperado):
        raise ArtefatoInvalido(f"{caminho.name}: SHA-256 diferente do registrado")
    if versao != sklearn.__version__:
        raise ArtefatoInvalido(f"{caminho.name}: gravado com scikit-learn {versao}, ambiente {sklearn.__version__}")
    if nome in ARTEFATOS_JSON:
        return json.loads(conteudo.decode("utf-8"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        return joblib.load(io.BytesIO(conteudo))
