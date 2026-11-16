"""Executa o protocolo de modelagem (`configs/protocolo_modelagem_v1_2.json`; 1.0 e 1.1 preservados) nos dados de
desenvolvimento de 2008-2009. O registro guarda o hash do codigo antes e depois: mudanca durante a execucao interrompe.

Uso: python -m occasion.modelagem_2008 --destino <pasta> --protocolo-sha256 <hash registrado>
     python -m occasion.modelagem_2008 --comparar <pasta1> <pasta2>

Entradas: CASE_AMBEV_SAIDA_ROOT (ou --saida) com as cinco saidas de 2008 conferidas por `configs/entradas_2008.json`, e
CASE_AMBEV_RAW_ROOT para o peso W1. Exige OMP_NUM_THREADS=1 e OPENBLAS_NUM_THREADS=1. Previsoes por episodio e artefatos
vao para --destino, que deve ficar fora do repositorio.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from occasion import congelamento as cg
from occasion import modelagem as m
from occasion.eda_2008 import dias_de_registro
from occasion.eda_dados import carregar_episodios, chave_pessoa, conferir_entradas
from occasion.fontes import raiz_bruta

REPO = Path(__file__).resolve().parents[2]
PROTOCOLO = REPO / "configs" / "protocolo_modelagem_v1_2.json"  # 1.0 e 1.1 preservados ao lado, sem alteracao
PROTOCOLOS = {"1.0": "configs/protocolo_modelagem.json", "1.1": "configs/protocolo_modelagem_v1_1.json",
              "1.2": "configs/protocolo_modelagem_v1_2.json"}
VARIAVEIS = REPO / "configs" / "variaveis.json"
SEMENTE_BASE = 20261114
SEMENTE_PERMUTACAO = 20262000
CANDIDATOS_FINAIS = (("B0", "B0", None), ("B1", "B1", None), ("B2", "B2", None), ("L", "L", None), ("G", "G", None),
                     ("A1", "L", "w1"))
LEITURAS = ("B0", "B1", "B2", "L", "G")


class ExecucaoInterrompida(RuntimeError):
    """Um portao do protocolo falhou; nada depois dele foi calculado."""


# ---------- leitura ----------

def _chave_dias(e: dict) -> tuple:
    u, d, c, i = chave_pessoa(e)
    return (u, int(d), int(c), int(i))


def carregar_dados(saida: Path, raw: Path, colunas: list[str], espec_g: dict) -> tuple[m.Dados, dict]:
    hashes = conferir_entradas(saida)
    eps = carregar_episodios(saida)
    if any(e["ids"]["cod_upa"] != e["e2"]["cod_upa"] or e["ids"]["episodio_id"] != e["id"] for e in eps):
        raise ExecucaoInterrompida("identificadores desalinhados de e2_2008")
    if any(e["mat"][c] is None for e in eps for c in colunas) or any(e["nat"]["renda_status"] != "valida" for e in eps):
        raise ExecucaoInterrompida("valor ausente na matriz: o protocolo recusa imputacao")
    x = np.array([[e["mat"][c] for c in colunas] for e in eps], dtype=float)
    try:
        x_g, cat_g = m.codificar_nativas([e["nat"] for e in eps], espec_g)
    except ValueError as erro:
        raise ExecucaoInterrompida(f"variaveis nativas de G: {erro}") from erro
    dias = dias_de_registro(raw)
    sem_dias = [e["id"] for e in eps if dias.get(_chave_dias(e), 0) < 1]
    if sem_dias:
        raise ExecucaoInterrompida(f"{len(sem_dias)} episodios sem dias de registro no consumo (W1 indefinido)")
    w1 = np.array([float(e["e2"]["peso_final"]) / dias[_chave_dias(e)] for e in eps])
    dados = m.Dados(x=x, y=np.array([e["y"] for e in eps]), grupos=np.array([e["e2"]["cod_upa"] for e in eps]),
                    cel_b1=np.array([f"{e['nat']['dia_semana']}|{e['nat']['faixa_hora']}" for e in eps]),
                    cel_b2=np.array([f"{e['nat']['grande_regiao']}|{e['nat']['faixa_hora']}" for e in eps]), w1=w1,
                    x_g=x_g, cat_g=cat_g)
    return dados, {"entradas_sha256": hashes, "regiao": np.array([e["nat"]["grande_regiao"] for e in eps])}


# ---------- grades e sementes ----------

def grades(prot: dict) -> dict:
    """Cada grade ordenada do mais para o menos regularizado (m decrescente; C crescente; G por `grade_boosting`)."""
    cand = prot["candidatos"]
    return {"B0": [None], "B1": sorted(cand["B1"]["grade_m"], reverse=True),
            "B2": sorted(cand["B2"]["grade_m"], reverse=True), "L": sorted(cand["L"]["grade_C_log10"]),
            "G": m.grade_boosting(cand["G"]["grade"])}


def sementes_externas(r_total: int) -> list[int]:
    return [SEMENTE_BASE + r for r in range(r_total)]


def semente_interna(r: int, k: int) -> int:
    return SEMENTE_BASE + 1000 * (r + 1) + k


# ---------- portoes ----------

def portoes(dados: m.Dados, prot: dict) -> dict:
    part = prot["particoes"]
    dobras, menor_ext, menor_int, sobreposicao = [], len(dados.y), len(dados.y), 0
    for r, semente in enumerate(sementes_externas(part["repeticoes"])):
        parts = m.particoes(dados.y, dados.grupos, part["dobras_externas"], semente)
        dobras.append(m.dobra_por_episodio(parts, len(dados.y)))
        for k, (tr, te) in enumerate(parts):
            sobreposicao += len(set(dados.grupos[tr]) & set(dados.grupos[te]))
            menor_ext = min(menor_ext, int(dados.y[te].sum()))
            for tr2, te2 in m.particoes(dados.y[tr], dados.grupos[tr], part["dobras_internas"], semente_interna(r, k)):
                sobreposicao += len(set(dados.grupos[tr][tr2]) & set(dados.grupos[tr][te2]))
                menor_int = min(menor_int, int(dados.y[tr][te2].sum()))
    div = m.diversidade(dobras, dados.grupos, dados.y)
    checagens = {"upa": sobreposicao == 0,
                 "k_externo": menor_ext >= part["minimo_positivos_dobra_externa"],
                 "k_interno": menor_int >= part["minimo_positivos_dobra_interna"],
                 "diversidade": div["pares_identicos"] == 0 and div["fracao_mudanca_com_positivo"]["min"] > 0}
    return {"upas_nos_dois_lados": sobreposicao, "menor_k_dobra_externa": menor_ext, "menor_k_dobra_interna": menor_int,
            "diversidade": div, "falhas": [nome for nome, ok in checagens.items() if not ok]}


# ---------- avaliacao ----------

def _faixa(valores: list[float]) -> dict:
    return {"min": float(min(valores)), "mediana": float(np.median(valores)), "max": float(max(valores))}


def resumo_modelo(y: np.ndarray, previsoes: np.ndarray, contagens: np.ndarray, ic_ordenacao: bool) -> dict:
    s = previsoes.mean(axis=0)
    perdas_rep = [m.perda_log(y, p) for p in previsoes]
    cal_rep = [m.calibracao(y, p) for p in previsoes]
    res = {"log_loss_media": float(np.mean(perdas_rep)), "log_loss_repeticoes": _faixa(perdas_rep),
           "log_loss_s_medio": m.perda_log(y, s), "brier_s": float(brier_score_loss(y, s)),
           "brier_ic": m.intervalo((contagens @ (s - y) ** 2) / contagens.sum(axis=1)),
           "auc_s": float(roc_auc_score(y, s)), "ap_s": float(average_precision_score(y, s)),
           "auc_repeticoes": _faixa([float(roc_auc_score(y, p)) for p in previsoes]),
           "calibracao_s": m.calibracao(y, s),
           "calibracao_ic": {k: m.intervalo(v) for k, v in m.calibracao(y, s, contagens).items()},
           "calibracao_repeticoes": {k: _faixa([c[k] for c in cal_rep]) for k in cal_rep[0]}}
    if ic_ordenacao:
        res["auc_ic"] = m.intervalo(np.array([roc_auc_score(y, s, sample_weight=w) for w in contagens]))
        res["ap_ic"] = m.intervalo(np.array([average_precision_score(y, s, sample_weight=w) for w in contagens]))
    return res


def ponderadas(y: np.ndarray, s: np.ndarray, w1: np.ndarray) -> dict:
    return {"log_loss": m.perda_log(y, s, w1), "brier": float(brier_score_loss(y, s, sample_weight=w1)),
            "o_e": float(np.sum(w1 * y) / np.sum(w1 * s)), "ap": float(average_precision_score(y, s, sample_weight=w1))}


def comparar(perdas: dict, a: str, b: str, grupos: np.ndarray, contagens: np.ndarray) -> dict:
    res = m.delta_pareado(perdas[a].mean(axis=0), perdas[b].mean(axis=0), grupos, contagens=contagens)
    return {**res, "delta_repeticoes": _faixa([float(v) for v in (perdas[a] - perdas[b]).mean(axis=1)])}


def borda(escolhas: list, grade: list) -> dict:
    if len(grade) < 2:
        return {}
    n = len(escolhas)
    out = {"fracao_primeiro": sum(e == grade[0] for e in escolhas) / n,
           "fracao_ultimo": sum(e == grade[-1] for e in escolhas) / n,
           "distribuicao": {str(v): sum(e == v for e in escolhas) for v in grade}}
    if isinstance(grade[0], tuple):  # G: extremos de cada dimensao (max_depth, max_iter, min_samples_leaf)
        out["por_dimensao"] = {}
        for i, nome in enumerate(("max_depth", "max_iter", "min_samples_leaf")):
            valores = sorted({c[i] for c in grade})
            out["por_dimensao"][nome] = {"menor": sum(e[i] == valores[0] for e in escolhas) / n,
                                         "maior": sum(e[i] == valores[-1] for e in escolhas) / n,
                                         "distribuicao": {str(v): sum(e[i] == v for e in escolhas) for v in valores}}
    return out


def limiares(y: np.ndarray, s: np.ndarray, coberturas: list[float]) -> dict:
    out = {}
    for q in coberturas:
        lim = m.limiar_cobertura(s, q)
        out[f"{q:.2f}"] = {**lim, "metricas": m.metricas_no_limiar(y, m.sinalizados(s, lim["t"]))}
    return out


def _sinais(a0, a1, limiar: float) -> dict:
    c0 = a0.named_steps["logisticregression"].coef_[0]
    c1 = a1.named_steps["logisticregression"].coef_[0]
    validos = (np.abs(c0) >= limiar) & (np.abs(c1) >= limiar)
    trocam = validos & (np.sign(c0) != np.sign(c1))
    return {"considerados": int(validos.sum()), "trocam_sinal": int(trocam.sum()),
            "fracao": float(trocam.sum() / validos.sum()) if validos.sum() else None,
            "A0": [float(c) for c in c0], "A1": [float(c) for c in c1]}


# ---------- etapas ----------

def modelos_finais(dados: m.Dados, grade: dict, prot: dict, pasta: Path) -> tuple[dict, dict]:
    """Reajuste unico de cada candidato nas 1.168 observacoes; artefato gravado, recarregado e conferido."""
    part, tol = prot["particoes"], prot["selecao_interna"]["tolerancia_empate"]
    sementes = [SEMENTE_BASE + 5000 + r for r in range(part["repeticoes"])]
    todos = np.arange(len(dados.y))
    meta = {"protocolo_sha256": m.sha256_arquivo(PROTOCOLO), "protocolo_versao": prot["versao"],
            "treino": "2008-2009, 1.168 episodios"}
    espec_g = prot["candidatos"]["G"]
    meta_g = {"colunas": espec_g["colunas"], "categoricas": espec_g["categoricas"],
              "mascara_categorica": [bool(v) for v in dados.cat_g], "fixos": dict(m.G_FIXOS)}
    out, previstos, ajustados = {}, {}, {}
    for nome, tipo, modo in CANDIDATOS_FINAIS:
        valor, curva = m.selecao_repetida(tipo, grade[tipo], dados, todos, sementes, part["dobras_internas"], tol,
                                          modo_peso=modo)
        modelo = m.ajustar(tipo, valor, dados, todos, modo_peso=modo)
        ajustados[nome], previstos[nome] = modelo, m.prever(tipo, modelo, dados, todos)
        out[nome] = {"hiperparametro": valor, "curva_interna": curva, "media_prevista_2008": float(previstos[nome].mean())}
        if nome == "A1":
            continue
        extra = meta_g if nome == "G" else {}
        caminho = m.salvar_artefato(modelo, nome, pasta, {**meta, **extra, "hiperparametro": valor})
        recarregado = m.carregar_artefato(caminho, nome, sha256_esperado=m.sha256_arquivo(caminho))
        diff = float(np.max(np.abs(m.prever(tipo, recarregado, dados, todos) - previstos[nome])))
        out[nome].update({"artefato": caminho.name, "sha256": m.sha256_arquivo(caminho), "recarga_max_diff": diff,
                          "recarga_ok": diff <= prot["reproducao"]["tolerancia_previsoes"]})
    out["G"]["n_iter"] = int(ajustados["G"].n_iter_)
    out["G"]["valores_distintos_2008"] = int(len(np.unique(np.round(previstos["G"], m.CASAS_EMPATE))))
    lr = ajustados["L"].named_steps["logisticregression"]
    out["L"]["coeficientes_padronizados"] = [float(c) for c in lr.coef_[0]]
    out["L"]["intercepto"] = float(lr.intercept_[0])
    out["coeficientes_A0_A1"] = _sinais(ajustados["L"], ajustados["A1"], prot["p5"]["limiar_coeficiente"])
    return out, previstos


def permutacoes(dados: m.Dados, grade: dict, prot: dict, contagens: np.ndarray) -> dict:
    cfg, part, tol = prot["permutacao"], prot["particoes"], prot["selecao_interna"]["tolerancia_empate"]
    deltas, supera = [], 0
    for j in range(cfg["permutacoes"]):
        semente = SEMENTE_PERMUTACAO + j
        perm = m.Dados(x=dados.x, y=m.permutar_por_upa(dados.y, dados.grupos, semente), grupos=dados.grupos,
                       cel_b1=dados.cel_b1, cel_b2=dados.cel_b2, w1=dados.w1)

        def interna(r: int, k: int, s: int = semente) -> int:
            return s + 1000 * (k + 1)

        pl = m.validacao_cruzada("L", grade["L"], perm, [semente], interna, part["dobras_externas"], tol)["previsoes"][0]
        pb = m.validacao_cruzada("B0", grade["B0"], perm, [semente], interna, part["dobras_externas"], tol)["previsoes"][0]
        res = m.delta_pareado(m.perdas_log(perm.y, pl), m.perdas_log(perm.y, pb), dados.grupos, contagens=contagens)
        deltas.append(res["delta"])
        supera += int(res["ic"][1] < 0)
    mediana = float(np.median(deltas))
    return {"deltas_L_menos_B0": deltas, "mediana": mediana, "L_supera_B0": supera,
            "sinal_para_investigar": bool(supera > 2 or mediana < 0)}


def ablacoes(dados: m.Dados, prot: dict, colunas: list[str], perdas: dict, contagens: np.ndarray, vc) -> dict:
    out = {}
    for grupo, retiradas in prot["ablacoes"].items():
        idx = [i for i, c in enumerate(colunas) if c not in retiradas]
        v = vc("L", colunas_idx=idx)
        rep = np.array([m.perdas_log(dados.y, p) for p in v["previsoes"]])
        res = m.delta_pareado(rep.mean(axis=0), perdas["L"].mean(axis=0), dados.grupos, contagens=contagens)
        out[grupo] = {**res, "colunas_retiradas": len(retiradas), "log_loss_media": float(rep.mean()),
                      "borda_grade": borda(v["escolhas"], sorted(prot["candidatos"]["L"]["grade_C_log10"]))}
    return out


def regiao_fora(dados: m.Dados, grade: dict, prot: dict, regiao: np.ndarray) -> dict:
    cfg, part, tol = prot["regiao_fora"], prot["particoes"], prot["selecao_interna"]["tolerancia_empate"]
    minimo = cfg["suprimir_taxa_auc_se_k_menor_que"]
    out = {}
    for i, nome in enumerate(cfg["regioes"]):
        treino, teste = np.nonzero(regiao != nome)[0], np.nonzero(regiao == nome)[0]
        sementes = [SEMENTE_BASE + 7000 + 100 * i + r for r in range(part["repeticoes"])]
        yt = dados.y[teste]
        k = int(yt.sum())
        linha = {"n": len(teste), "k": k, "taxa": float(yt.mean()) if k >= minimo else None, "modelos": {}}
        for tipo in cfg["modelos"]:
            valor, _ = m.selecao_repetida(tipo, grade[tipo], dados, treino, sementes, part["dobras_internas"], tol)
            p = m.prever(tipo, m.ajustar(tipo, valor, dados, treino), dados, teste)
            auc = float(roc_auc_score(yt, p)) if k >= minimo and len(set(np.round(p, 12))) > 1 else None
            linha["modelos"][tipo] = {"hiperparametro": valor, "log_loss": m.perda_log(yt, p),
                                      "o_e": float(yt.sum() / p.sum()), "media_prevista": float(p.mean()), "auc": auc}
        out[nome] = linha
    return out


def aplicacao_do_limiar(prot: dict, lims: dict, s_oof: np.ndarray, s_final: np.ndarray, constante: bool = False) -> dict:
    """Limiares fora da dobra aplicados sem mudanca ao modelo final de 2008, e as classes nas duas leituras.

    Com preditor constante por construcao (B0) os limiares ficam como diagnostico: nao se aplica t30 ao modelo final,
    `alta` e vazia e todo contexto recebe `media`, com a comparacao numerica uniforme de `modelagem.classes`."""
    ref = prot["entradas"]["taxa_referencia"]
    b = ref["positivos"] / ref["episodios"]
    t30 = lims["0.30"]["t"]
    t_final = None if constante else t30
    out = {"b": b, "t": {q: v["t"] for q, v in lims.items()},
           "cobertura_fora_da_dobra": {q: v["cobertura"] for q, v in lims.items()},
           "cobertura_modelo_final_2008_otimista": {q: float(np.mean(m.sinalizados(s_final, v["t"])))
                                                    for q, v in lims.items()},
           "classes_fora_da_dobra": dict(Counter(m.classes(s_oof, t30, b).tolist())),
           "classes_modelo_final_2008": dict(Counter(m.classes(s_final, t_final, b).tolist()))}
    if constante:
        out["preditor_constante"] = {"limiares_sao_diagnostico": True, "t_aplicado_ao_modelo_final": None,
                                     "alta": "vazia", "classes": "media para todo contexto", "ordena_contextos": False,
                                     "classes_fora_da_dobra": "artefato de diferencas de taxa entre dobras"}
    return out


def diagnostico_p5(a0_rep: np.ndarray, a1_rep: np.ndarray, q: float) -> dict:
    p5 = m.p5(a0_rep, a1_rep, q)
    a0, a1 = a0_rep.mean(axis=0), a1_rep.mean(axis=0)
    alta0 = m.sinalizados(a0, m.limiar_cobertura(a0, q)["t"])
    alta1 = m.sinalizados(a1, m.limiar_cobertura(a1, q)["t"])
    return {"quartis": p5["quartis"], "convencao_relevante": p5["convencao_relevante"],
            "max_d_regua": max(p5["d_regua"]), "min_rho_regua": min(p5["rho_regua"]),
            "s_medio": {"d": float(np.mean(alta0 != alta1)), "rho": float(spearmanr(a0, a1).statistic)}}


# ---------- execucao ----------

def executar(saida: Path, raw: Path, destino: Path, sha_esperado: str, etapas: set[str]) -> dict:
    sha = m.sha256_arquivo(PROTOCOLO)
    if sha != sha_esperado:
        raise ExecucaoInterrompida(f"protocolo com hash {sha[:12]} diferente do registrado {sha_esperado[:12]}")
    prot = m.carregar_protocolo(PROTOCOLO)
    colunas = json.loads(VARIAVEIS.read_text(encoding="utf-8"))["colunas_logistica"]
    dados, extra = carregar_dados(saida, raw, colunas, prot["candidatos"]["G"])
    y, g = dados.y, dados.grupos
    prev_w1 = float(np.sum(dados.w1 * y) / np.sum(dados.w1))
    if abs(prev_w1 - prot["pesos"]["conferencia_prevalencia_W1"]) > 5e-6:
        raise ExecucaoInterrompida(f"prevalencia ponderada W1 {prev_w1:.6f} difere do registro da EDA")
    destino.mkdir(parents=True, exist_ok=True)
    codigo_antes = cg.hashes_do_codigo(REPO)
    reg = {"protocolo_sha256": sha, "protocolo_versao": prot["versao"], "sklearn": sklearn.__version__,
           "numpy": np.__version__,
           "protocolos_sha256": {v: m.sha256_arquivo(REPO / rel) for v, rel in PROTOCOLOS.items()},
           "codigo_sha256_antes": codigo_antes,
           "uv_lock_sha256": m.sha256_arquivo(REPO / "uv.lock"), "entradas_sha256": extra["entradas_sha256"],
           "threads": {k: os.environ.get(k) for k in prot["reproducao"]["threads"]}, "n": len(y),
           "positivos": int(y.sum()), "upas": len(set(g)), "prevalencia_w1": prev_w1, "etapas": sorted(etapas),
           "tempos_s": {}}
    relogio = time.perf_counter()

    reg["portoes"] = portoes(dados, prot)
    reg["tempos_s"]["portoes"] = round(time.perf_counter() - relogio, 1)
    if reg["portoes"]["falhas"]:
        _gravar(destino, reg, {})
        raise ExecucaoInterrompida(f"portoes falharam: {reg['portoes']['falhas']}")

    part, tol = prot["particoes"], prot["selecao_interna"]["tolerancia_empate"]
    grade = grades(prot)
    boot = prot["avaliacao"]["bootstrap"]
    contagens = m.contagens_bootstrap(g, boot["replicas"], boot["semente"])
    externas = sementes_externas(part["repeticoes"])

    def vc(tipo: str, colunas_idx=None, modo=None) -> dict:
        return m.validacao_cruzada(tipo, grade[tipo], dados, externas, semente_interna, part["dobras_externas"], tol,
                                   colunas_idx, modo)

    especs = {"B0": ("B0", None), "B1": ("B1", None), "B2": ("B2", None), "L": ("L", None), "A1": ("L", "w1"),
              "A1t": ("L", "w1t"), "G": ("G", None)}
    oof = {}
    for nome, (tipo, modo) in especs.items():
        t0 = time.perf_counter()
        oof[nome] = vc(tipo, modo=modo)
        reg["tempos_s"][nome] = round(time.perf_counter() - t0, 1)
    previsoes = {k: v["previsoes"] for k, v in oof.items()}
    perdas = {k: np.array([m.perdas_log(y, p) for p in v]) for k, v in previsoes.items()}

    modelos = {}
    for nome, v in oof.items():
        modelos[nome] = {**resumo_modelo(y, v["previsoes"], contagens, nome in LEITURAS),
                         "dentro_da_dobra": m.metricas_dentro_da_dobra(y, v["previsoes"], v["dobras"]),
                         "borda_grade": borda(v["escolhas"], grade[especs[nome][0]]),
                         "menor_k_interno": v["menor_k_interno"]}
        if nome in ("L", "A1", "A1t"):
            modelos[nome]["ponderadas_w1"] = ponderadas(y, previsoes[nome].mean(axis=0), dados.w1)

    ordem = prot["candidatos"]["ordem_simplicidade"]
    medias = {c: modelos[c]["log_loss_media"] for c in ordem}
    m_estrela = min(medias, key=medias.get)
    contra_m = {c: comparar(perdas, m_estrela, c, g, contagens) for c in ordem}
    escolha = m.escolher_modelo(medias, {c: contra_m[c]["ic"] for c in ordem}, ordem)
    descritivas = {f"{a}-{b}": comparar(perdas, a, b, g, contagens)
                   for a, b in (("B1", "B0"), ("L", "B1"), ("B2", "B1"), ("L", "B0"), ("A1", "L"), ("A1t", "L"),
                                ("G", "B0"), ("G", "B1"), ("G", "B2"), ("G", "L"))}
    coberturas = prot["limiar"]["coberturas"]
    escolhido = escolha["escolhido"]
    lims = {c: limiares(y, previsoes[c].mean(axis=0), coberturas)
            for c in dict.fromkeys([escolhido, "B1", "B2", "L", "G"])}

    t0 = time.perf_counter()
    finais, previstos = modelos_finais(dados, grade, prot, destino / "artefatos")
    reg["tempos_s"]["modelos_finais"] = round(time.perf_counter() - t0, 1)
    for nome in LEITURAS:
        modelos[nome]["leitura_preditor"] = m.leitura_preditor(previsoes[nome], oof[nome]["dobras"], previstos[nome])
        if modelos[nome]["leitura_preditor"]["constante_por_construcao"]:
            modelos[nome]["nao_interpretavel"] = {"metricas": list(m.METRICAS_NAO_INTERPRETAVEIS_CONSTANTE),
                                                  "motivo": prot["leituras_descritivas"]["tratamento_b0"]["motivo_metricas"]}
    aplicacao = aplicacao_do_limiar(prot, lims[escolhido], previsoes[escolhido].mean(axis=0), previstos[escolhido],
                                    modelos[escolhido]["leitura_preditor"]["constante_por_construcao"])
    p5_saida = {**diagnostico_p5(previsoes["L"], previsoes["A1"], prot["limiar"]["operacional"]),
                "coeficientes": finais.pop("coeficientes_A0_A1")}
    resultados = {"modelos": modelos, "m_estrela": m_estrela, "contra_m_estrela": contra_m,
                  "escolha_provisoria": escolha, "comparacoes_descritivas": descritivas, "limiares": lims,
                  "finais": finais, "aplicacao_do_limiar": aplicacao, "p5": p5_saida}

    etapas_extra = {"permutacao": lambda: permutacoes(dados, grade, prot, contagens),
                    "ablacoes": lambda: ablacoes(dados, prot, colunas, perdas, contagens, vc),
                    "regioes": lambda: regiao_fora(dados, grade, prot, extra["regiao"])}
    for nome, funcao in etapas_extra.items():
        if nome in etapas:
            t0 = time.perf_counter()
            resultados[nome] = funcao()
            reg["tempos_s"][nome] = round(time.perf_counter() - t0, 1)

    reg["tempos_s"]["total"] = round(time.perf_counter() - relogio, 1)
    codigo_depois = cg.hashes_do_codigo(REPO)
    reg["codigo"] = {"antes": codigo_antes, "depois": codigo_depois, "inalterado": codigo_antes == codigo_depois}
    if not reg["codigo"]["inalterado"]:
        _gravar(destino, reg, {})
        mudou = sorted(k for k in codigo_antes.keys() | codigo_depois.keys() if codigo_antes.get(k) != codigo_depois.get(k))
        raise ExecucaoInterrompida(f"o codigo mudou durante a execucao: {mudou}")
    np.savez_compressed(destino / "previsoes_fora_da_dobra.npz", **previsoes)
    np.savez_compressed(destino / "previsoes_modelos_finais_2008.npz", **previstos)
    if not all(np.array_equal(v["dobras"], oof["B0"]["dobras"]) for v in oof.values()):
        raise ExecucaoInterrompida("candidatos com particoes externas diferentes: pareamento quebrado")
    np.savez_compressed(destino / "dobras_externas.npz", dobras=oof["B0"]["dobras"])
    _gravar(destino, reg, resultados)
    return {"registro": reg, "resultados": resultados}


def _json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    raise TypeError(type(obj))


def _gravar(destino: Path, registro: dict, resultados: dict) -> None:
    for nome, conteudo in (("registro.json", registro), ("resultados.json", resultados)):
        (destino / nome).write_text(json.dumps(conteudo, default=_json, indent=1, ensure_ascii=False), encoding="utf-8")


def comparar_execucoes(a: Path, b: Path, tolerancia: float) -> dict:
    """Mesma operacao repetida: previsoes, hiperparametros, limiares e escolha devem coincidir."""
    diffs = {}
    for arquivo in ("previsoes_fora_da_dobra.npz", "previsoes_modelos_finais_2008.npz"):
        pa, pb = np.load(a / arquivo), np.load(b / arquivo)
        diffs.update({f"{arquivo}:{k}": float(np.max(np.abs(pa[k] - pb[k]))) for k in pa.files})
    ra, rb = (json.loads((p / "resultados.json").read_text(encoding="utf-8")) for p in (a, b))
    if set(ra["finais"]) != set(rb["finais"]):
        raise ExecucaoInterrompida(f"execucoes com candidatos diferentes: {sorted(set(ra['finais']) ^ set(rb['finais']))}")
    hip = {k: (ra["finais"][k]["hiperparametro"], rb["finais"][k]["hiperparametro"])
           for k in ra["finais"] if "hiperparametro" in ra["finais"][k]}
    lim = {c: {q: (v["t"], rb["limiares"][c][q]["t"]) for q, v in d.items()} for c, d in ra["limiares"].items()}
    dobras = _comparar_dobras(a, b)
    artefatos = {k: ra["finais"][k]["sha256"] == rb["finais"][k]["sha256"]
                 for k in ra["finais"] if "sha256" in ra["finais"][k] and "sha256" in rb["finais"].get(k, {})}
    ok = (all(v <= tolerancia for v in diffs.values()) and all(x == y for x, y in hip.values())
          and all(abs((x or 0) - (y or 0)) <= tolerancia and (x is None) == (y is None)
                  for d in lim.values() for x, y in d.values())
          and ra["escolha_provisoria"] == rb["escolha_provisoria"] and dobras is not False and all(artefatos.values()))
    return {"max_diff": diffs, "hiperparametros_finais": hip, "limiares": lim, "dobras_identicas": dobras,
            "artefatos_sha_iguais": artefatos, "identicas": ok}


def _comparar_dobras(a: Path, b: Path) -> bool | None:
    """True/False quando as duas pastas gravaram `dobras_externas.npz`; None quando alguma nao tem (execucoes antigas)."""
    if not ((a / "dobras_externas.npz").exists() and (b / "dobras_externas.npz").exists()):
        return None
    return bool(np.array_equal(np.load(a / "dobras_externas.npz")["dobras"], np.load(b / "dobras_externas.npz")["dobras"]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--saida", default=os.environ.get("CASE_AMBEV_SAIDA_ROOT"))
    p.add_argument("--destino")
    p.add_argument("--protocolo-sha256")
    p.add_argument("--etapas", default="permutacao,ablacoes,regioes", help="etapas opcionais, separadas por virgula")
    p.add_argument("--comparar", nargs=2, metavar=("PASTA1", "PASTA2"))
    args = p.parse_args(argv)
    prot = m.carregar_protocolo(PROTOCOLO)
    if args.comparar:
        a, b = Path(args.comparar[0]), Path(args.comparar[1])
        res = comparar_execucoes(a, b, prot["reproducao"]["tolerancia_previsoes"])
        (b / "comparacao.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        sys.stdout.write(json.dumps({"identicas": res["identicas"]}) + "\n")
        return 0 if res["identicas"] else 1
    for var, valor in prot["reproducao"]["threads"].items():
        if os.environ.get(var) != valor:
            p.error(f"defina {var}={valor} antes de executar")
    if not (args.saida and args.destino and args.protocolo_sha256):
        p.error("--saida (ou CASE_AMBEV_SAIDA_ROOT), --destino e --protocolo-sha256 sao obrigatorios")
    etapas = {e for e in args.etapas.split(",") if e}
    out = executar(Path(args.saida), raiz_bruta(dict(os.environ)), Path(args.destino), args.protocolo_sha256, etapas)
    sys.stdout.write(json.dumps({"escolha_provisoria": out["resultados"]["escolha_provisoria"],
                                 "tempos_s": out["registro"]["tempos_s"]}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
