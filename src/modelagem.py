"""Occasion Score observado: chance de uma refeicao com comida incluir pizza e guarana (POF/IBGE), do treino a tela.

Etapas (protocolo configs/protocolo_v3_alvo_observado.json), com `python -m modelagem <etapa>`:
  desenvolver  compara os candidatos nas 5 dobras do desenvolvimento de 2017 e registra a escolha pela regra fixada
  testar       treina a escolha no desenvolvimento, abre o teste reservado UMA vez e grava o artefato da aplicacao
  temporal     treina a mesma especificacao em 2008-2009 e avalia no teste de 2017 (variaveis comuns)
  explicar     mede, so no desenvolvimento e fora da dobra, quais campos ajudam o modelo em geral (explicacao.json)
  eda          tabula o alvo por eixo no desenvolvimento de 2017 e em 2008 (eda.json na pasta processada; sem o teste)
A aplicacao chama `avaliar(contexto)`, que devolve os seis campos do TAPI e a leitura em linguagem de marketing.
"""

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from functools import cache
from itertools import product
from pathlib import Path

import duckdb
import joblib
import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

import ocasiao_modelos as om
import preparo
from aplicacao import regras

VERSAO = "ocasiao_observada_v1"
RAIZ = Path(__file__).resolve().parents[1]
ARTEFATOS = RAIZ / "artifacts" / VERSAO
REAMOSTRAS = 30
REPETICOES_PERMUTACAO = 5
CAMPOS_BASE = ("episodio_id", "upa", "dia_semana", "periodo", "hora", "grande_regiao", "area", "faixa_etaria", "sexo",
               "escolaridade", "pizza", "guarana", "sabor", "local_v9018", "peso")
CAMPOS_CONTEXTO = ("dia_semana", "periodo", "grande_regiao", "area")
# Grupos de permutacao: o periodo leva a hora junto (os dois descrevem o mesmo momento da refeicao).
GRUPOS_PERMUTACAO = {"dia_semana": ("dia_semana",), "periodo": ("periodo", "hora"), "grande_regiao": ("grande_regiao",),
                     "area": ("area",), "faixa_etaria": ("faixa_etaria",), "sexo": ("sexo",),
                     "escolaridade": ("escolaridade",)}
ORIGEM_IMPORTANCIA = "desenvolvimento_fora_da_dobra"
PERGUNTA_GERAL = "o campo ajuda o modelo em geral? (permutação fora da dobra no desenvolvimento)"
PERGUNTA_DO_CENARIO = "o campo contribuiu para esta previsão? (troca só desse campo no contexto consultado)"
MEDIDA = "Pizza e guaraná na mesma refeição"
BASE_DA_ESTIMATIVA = "Consumo observado · POF/IBGE 2017–2018"

GRADES = {
    "global": [("global", {"familia": "global"})],
    "dia_periodo": [(f"dia_periodo_m{m}", {"familia": "dia_periodo", "m": m}) for m in (0, 50, 200, 1000, 5000)],
    "logistica": [(f"logistica_{b}_C{c}", {"familia": "logistica", "bloco": b, "C": c})
                  for b in ("B1", "B12", "B123") for c in (0.01, 0.1, 1.0)],
    "dois_passos": [(f"dois_passos_{b}_C{c}", {"familia": "dois_passos", "bloco": b, "C": c})
                    for b in ("B1", "B12", "B123") for c in (0.1, 1.0)],
    "boosting": [(f"boosting_{b}_lr{lr}_f{f}", {"familia": "boosting", "bloco": b, "lr": lr, "folhas": f})
                 for b in ("B1", "B123") for lr in (0.03, 0.1) for f in (7, 15)],
}


def fabricar(spec: dict):
    familia = spec["familia"]
    if familia == "global":
        return om.Global()
    if familia == "dia_periodo":
        return om.DiaPeriodo(m=spec["m"])
    if familia == "logistica":
        return om.Logistica(spec["bloco"], spec["C"])
    if familia == "dois_passos":
        return om.DoisPassos(spec["bloco"], spec["C"], spec["C"])
    if familia == "boosting":
        return om.Boosting(spec["bloco"], spec["lr"], spec["folhas"], 1.0)
    raise ValueError(f"familia desconhecida: {familia}")


# ---------- dados ----------

def carregar_base(edicao: str, particao: str | None = None, pasta: Path | None = None) -> dict:
    arquivo = (pasta or preparo.raiz_processado()) / f"escolhas_pof_{edicao}.parquet"
    filtro = "TRUE" if particao is None else "particao = ?"
    parametros = [str(arquivo)] + ([] if particao is None else [particao])
    d = duckdb.connect().execute(f"SELECT {', '.join(CAMPOS_BASE)} FROM read_parquet(?) WHERE {filtro} "
                                 "ORDER BY episodio_id", parametros).fetchnumpy()
    return {k: np.asarray(v) for k, v in d.items()}


def alvo(dados: dict) -> np.ndarray:
    return (dados["pizza"] * dados["guarana"]).astype(float)


def _escrever(caminho: Path, conteudo: dict) -> str:
    texto = json.dumps(conteudo, ensure_ascii=False, indent=1, default=float)
    caminho.write_text(texto, encoding="utf-8")
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


# ---------- analise exploratoria do alvo ----------

EIXOS_EDA = {"periodo": "periodo", "dia_semana": "dia_semana", "fim_de_semana_noite":
             "CASE WHEN dia_semana IN (5,6,7) AND periodo = 'noite' THEN 'sexta_a_domingo_noite' ELSE 'demais' END",
             "grande_regiao": "grande_regiao", "area": "area", "faixa_etaria": "faixa_etaria", "sexo": "sexo",
             "escolaridade": "escolaridade", "local_v9018": "local_v9018", "hora": "hora"}


def _taxas_por_eixo(con, arquivo: Path, filtro: str, eixo: str) -> list[dict]:
    linhas = con.execute(f"""SELECT {eixo} v, count(*) n, sum(pizza*guarana) pos, sum(peso*pizza*guarana)/sum(peso) pond,
        avg(pizza) pizza, sum(guarana*pizza)/nullif(sum(pizza),0) g_dado_p, sum(pizza) n_pizza
        FROM read_parquet(?) WHERE {filtro} GROUP BY 1 ORDER BY 1""", [str(arquivo)]).fetchall()
    return [{"valor": str(v), "refeicoes": n, "positivos": int(p), "taxa": p / n, "taxa_ponderada": w,
             "pizza": pz, "guarana_dado_pizza": g, "refeicoes_com_pizza": int(npz)}
            for v, n, p, w, pz, g, npz in linhas]


def analise_exploratoria(saida: Path | None = None, pasta: Path | None = None) -> dict:
    """Taxas do alvo por eixo, com contagens: refeicoes, positivos (pizza e guarana), taxa bruta e ponderada, taxa de
    pizza e de guarana entre as refeicoes com pizza. Usa o desenvolvimento de 2017 e 2008 inteira; nao le o teste."""
    pasta = Path(pasta or preparo.raiz_processado())
    bases = {"2017_desenvolvimento": (pasta / "escolhas_pof_2017.parquet", "particao = 'desenvolvimento'"),
             "2008_todas": (pasta / "escolhas_pof_2008.parquet", "TRUE")}
    con = duckdb.connect()
    resultado = {nome: {e: _taxas_por_eixo(con, a, f, sql) for e, sql in EIXOS_EDA.items()}
                 for nome, (a, f) in bases.items()}
    destino = Path(saida or pasta)
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "eda.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=1), encoding="utf-8")
    return resultado


# ---------- desenvolvimento ----------

def _melhor(configuracoes: dict, prefixo: str) -> str:
    return min((n for n in configuracoes if n.startswith(prefixo)), key=lambda n: configuracoes[n]["perda_media"])


def _recusar_se_teste_aberto(saida: Path, etapa: str) -> None:
    """O teste desta versao e aberto uma vez: com teste.json na pasta pedida OU na pasta oficial da versao, nem a
    selecao nem o teste rodam de novo (nova escolha exige outra VERSAO)."""
    for pasta in {Path(saida), ARTEFATOS}:
        if (pasta / "teste.json").exists():
            raise SystemExit(f"{etapa}: o teste reservado de {VERSAO} ja foi aberto ({pasta / 'teste.json'}); "
                             "nova escolha exige outra versao")


def desenvolver(saida: Path = ARTEFATOS) -> dict:
    _recusar_se_teste_aberto(saida, "desenvolver")
    dados = carregar_base("2017", "desenvolvimento")
    y = alvo(dados)
    divisoes = om.dobras(dados["upa"])
    configuracoes = {}
    for grade in GRADES.values():
        for nome, spec in grade:
            r = om.validar(lambda spec=spec: fabricar(spec), dados, y, divisoes)
            configuracoes[nome] = {"spec": spec, "perda_media": r["perda_media"], "perda_por_dobra": r["perda_por_dobra"],
                                   "fora_da_dobra": om.metricas(y, r["oof"])}
    melhores = {f: _melhor(configuracoes, f) for f in GRADES}
    escolha = om.escolher({f: configuracoes[n] for f, n in melhores.items()})
    referencia = configuracoes[melhores["dia_periodo"]]
    contribuicao = {}
    for familia in ("logistica", "dois_passos"):
        for b in om.BLOCOS:
            trilha = om.escolher({"dia_periodo": referencia,
                                  familia: configuracoes[_melhor(configuracoes, f"{familia}_{b}_")]})["trilha"]
            contribuicao[f"{familia}_{b}"] = trilha[0] if trilha else {"contra": "dia_periodo",
                                                                       "vantagem_sustentada": False,
                                                                       "nota": "nao supera a regra dia x periodo"}
    resultado = {"versao": VERSAO, "n": int(len(y)), "positivos": int(y.sum()), "configuracoes": configuracoes,
                 "melhor_por_familia": melhores, "escolha": escolha,
                 "spec_escolhida": configuracoes[melhores[escolha["escolhido"]]]["spec"],
                 "spec_referencia": referencia["spec"], "contribuicao_dos_blocos_contra_dia_periodo": contribuicao,
                 "nota": "hiperparametros escolhidos nas mesmas 5 dobras (grade pequena): leve otimismo nas perdas"}
    saida.mkdir(parents=True, exist_ok=True)
    _escrever(saida / "desenvolvimento.json", resultado)
    return resultado


# ---------- teste reservado ----------

def _segmentos(teste: dict, y: np.ndarray, previsoes: dict) -> dict:
    fim = np.isin(teste["dia_semana"], (5, 6, 7))
    eixos = {"periodo": teste["periodo"], "grande_regiao": teste["grande_regiao"], "area": teste["area"],
             "faixa_etaria": teste["faixa_etaria"], "dia": np.where(fim, "sexta_a_domingo", "segunda_a_quinta")}
    saida = {}
    for eixo, valores in eixos.items():
        saida[eixo] = {}
        for v in np.unique(valores):
            sel = valores == v
            linha = {"n": int(sel.sum()), "positivos": int(y[sel].sum()), "taxa_observada": float(y[sel].mean()),
                     "instavel": bool(y[sel].sum() < 30)}
            for nome, p in previsoes.items():
                linha[f"previsto_{nome}"] = float(p[sel].mean())
                if 0 < y[sel].sum() < sel.sum():
                    linha[f"pr_auc_{nome}"] = float(average_precision_score(y[sel], p[sel]))
            saida[eixo][str(v)] = linha
    return saida


def importancia_por_permutacao(modelo, dados: dict, y: np.ndarray, repeticoes: int = REPETICOES_PERMUTACAO) -> dict:
    """Aumento da perda logaritmica ao embaralhar cada grupo de campos (linhas inteiras do grupo, mesma permutacao)."""
    rng = np.random.default_rng(om.SEMENTE)
    base = om.perda_log(y, modelo.prever(dados))
    saida = {}
    for grupo, campos in GRUPOS_PERMUTACAO.items():
        aumentos = []
        for _ in range(repeticoes):
            ordem = rng.permutation(len(y))
            trocado = {**dados, **{c: dados[c][ordem] for c in campos}}
            aumentos.append(om.perda_log(y, modelo.prever(trocado)) - base)
        media, desvio = float(np.mean(aumentos)), float(np.std(aumentos, ddof=1))
        saida[grupo] = {"aumento_medio": media, "desvio": desvio, "sustentada": bool(media - desvio > 0)}
    return saida


def _resumir_aumentos(aumentos: list, por_dobra: np.ndarray, base: float, base_por_dobra: list) -> dict:
    media, desvio = float(np.mean(aumentos)), float(np.std(aumentos, ddof=1))
    return {"aumento_medio": media, "desvio": desvio, "sustentada": bool(media - desvio > 0),
            "aumentos": [float(a) for a in aumentos], "aumento_por_dobra": [float(a) for a in por_dobra.mean(axis=0)],
            "perda_base": base, "perda_base_por_dobra": base_por_dobra}


def importancia_fora_da_dobra(fabrica, dados: dict, y: np.ndarray, divisoes: list,
                              repeticoes: int = REPETICOES_PERMUTACAO, grupos: dict | None = None) -> dict:
    """Aumento da perda logaritmica fora da dobra ao embaralhar cada grupo de campos dentro da dobra de validacao.

    Cada dobra e prevista por um modelo treinado nas demais (as mesmas `divisoes` da selecao); em cada repeticao todas
    as dobras sao embaralhadas e a perda e medida sobre todas as linhas, como em `importancia_por_permutacao`.
    Criterio: media menos 1 desvio das repeticoes > 0. So recebe o desenvolvimento: o teste reservado nao entra."""
    grupos = GRUPOS_PERMUTACAO if grupos is None else grupos
    rng = np.random.default_rng(om.SEMENTE)
    modelos = [(fabrica().ajustar(om.subconjunto(dados, treino), y[treino]), valida) for treino, valida in divisoes]
    oof = np.zeros(len(y))
    for modelo, valida in modelos:
        oof[valida] = modelo.prever(om.subconjunto(dados, valida))
    base = om.perda_log(y, oof)
    base_por_dobra = [om.perda_log(y[v], oof[v]) for _, v in modelos]
    saida = {}
    for grupo, campos in grupos.items():
        aumentos, por_dobra = [], np.zeros((repeticoes, len(modelos)))
        for r in range(repeticoes):
            trocado_oof = np.zeros(len(y))
            for k, (modelo, valida) in enumerate(modelos):
                parte = om.subconjunto(dados, valida)
                ordem = rng.permutation(len(valida))
                trocado_oof[valida] = modelo.prever({**parte, **{c: parte[c][ordem] for c in campos}})
                por_dobra[r, k] = om.perda_log(y[valida], trocado_oof[valida]) - base_por_dobra[k]
            aumentos.append(om.perda_log(y, trocado_oof) - base)
        saida[grupo] = _resumir_aumentos(aumentos, por_dobra, base, base_por_dobra)
    return saida


def _componentes(modelo, teste: dict, dev: dict) -> dict | None:
    if not isinstance(modelo, om.DoisPassos):
        return None
    p, g = modelo.componentes(teste)
    pizza = teste["pizza"].astype(float)
    com = pizza == 1
    gy = teste["guarana"][com].astype(float)
    taxa_g = float(dev["guarana"][dev["pizza"] == 1].mean())
    metricas_pizza = {k: v for k, v in om.metricas(pizza, p).items() if k != "calibracao"}
    return {"pizza": {**metricas_pizza,
                      "perda_log_taxa_constante": om.perda_log(pizza, np.full(len(pizza), dev["pizza"].mean()))},
            "guarana_dado_pizza": {"n": int(com.sum()), "positivos": int(gy.sum()),
                                   "perda_log": om.perda_log(gy, g[com]),
                                   "perda_log_taxa_constante": om.perda_log(gy, np.full(len(gy), taxa_g)),
                                   "roc_auc": float(roc_auc_score(gy, g[com])) if 0 < gy.sum() < len(gy) else None}}


def _sensibilidade(teste: dict, previsoes: dict) -> dict:
    inconclusivo = (teste["pizza"] == 1) & (teste["guarana"] == 0) & (teste["sabor"] == "desconhecido")
    y = alvo(teste)
    manter = ~inconclusivo
    return {"inconclusivos_no_teste": int(inconclusivo.sum()),
            "pr_auc_sem_inconclusivos": {n: om.metricas(y[manter], p[manter])["pr_auc"] for n, p in previsoes.items()},
            "pr_auc_inconclusivo_como_positivo": {n: om.metricas(np.maximum(y, inconclusivo), p)["pr_auc"]
                                                  for n, p in previsoes.items()}}


def _reamostras(spec: dict, dev: dict, y: np.ndarray, n: int = REAMOSTRAS) -> list:
    """Modelos reajustados em reamostras de UPAs do desenvolvimento (mesma especificacao): incerteza da estimativa."""
    rng = np.random.default_rng(om.SEMENTE + 1)
    upas, codigo = np.unique(dev["upa"], return_inverse=True)
    ordem = np.argsort(codigo, kind="mergesort")
    membros = np.split(ordem, np.cumsum(np.bincount(codigo))[:-1])
    saida = []
    for _ in range(n):
        idx = np.concatenate([membros[g] for g in rng.integers(0, len(upas), len(upas))])
        saida.append((fabricar(spec).ajustar(om.subconjunto(dev, idx), y[idx]), float(y[idx].mean())))
    return saida


def _composicao(dev: dict) -> dict:
    composicao: dict = {}
    for r, a, f, s, e in zip(dev["grande_regiao"], dev["area"], dev["faixa_etaria"], dev["sexo"], dev["escolaridade"],
                             strict=True):
        celula = composicao.setdefault(f"{r}|{a}", {})
        celula[f"{f}|{s}|{e}"] = celula.get(f"{f}|{s}|{e}", 0) + 1
    return composicao


def _resumo_de_treino(dev: dict, y: np.ndarray, referencia) -> dict:
    celula = om.colunas(dev, ["dia_periodo"])[:, 0]
    suporte = {c: {"episodios": int((celula == c).sum()), "positivos": int(y[celula == c].sum())} for c in np.unique(celula)}
    horas = {}
    for p in preparo.PERIODOS:
        valores, n = np.unique(dev["hora"][dev["periodo"] == p], return_counts=True)
        horas[p] = {str(int(h)): int(k) for h, k in zip(valores, n, strict=True)}
    distribuicoes = {c: {str(v): int(n) for v, n in zip(*np.unique(dev[c], return_counts=True), strict=True)}
                     for c in GRUPOS_PERMUTACAO}
    return {"taxa_global_treino": float(y.mean()), "refeicoes_treino": int(len(y)), "positivos_treino": int(y.sum()),
            "guarana_dado_pizza_treino": float(dev["guarana"][dev["pizza"] == 1].mean()),
            "pizza_treino": float(dev["pizza"].mean()),
            "positivos_em_casa": float(np.mean(dev["local_v9018"][y == 1] == "1")),
            "suporte_dia_periodo": suporte, "composicao_publico": _composicao(dev), "horas_por_periodo": horas,
            "distribuicoes": distribuicoes, "referencia_regra": {"m": referencia.m}}


def testar(saida: Path = ARTEFATOS) -> dict:
    _recusar_se_teste_aberto(saida, "testar")
    desenvolvimento = json.loads((saida / "desenvolvimento.json").read_text(encoding="utf-8"))
    dev, teste = carregar_base("2017", "desenvolvimento"), carregar_base("2017", "teste")
    ydev, yteste = alvo(dev), alvo(teste)
    spec, spec_ref = desenvolvimento["spec_escolhida"], desenvolvimento["spec_referencia"]
    escolhido = fabricar(spec).ajustar(dev, ydev)
    referencia = fabricar(spec_ref).ajustar(dev, ydev)
    taxa = om.Global().ajustar(dev, ydev)
    previsoes = {"escolhido": escolhido.prever(teste), "global": taxa.prever(teste), "dia_periodo": referencia.prever(teste)}
    resultado = {
        "versao": VERSAO, "aberto_uma_vez": True, "spec_escolhida": spec, "spec_referencia": spec_ref,
        "metricas": {n: om.metricas(yteste, p) for n, p in previsoes.items()},
        "incerteza_bootstrap_upa": om.bootstrap_por_upa(yteste, previsoes, teste["upa"]),
        "segmentos": _segmentos(teste, yteste, {"escolhido": previsoes["escolhido"],
                                                "dia_periodo": previsoes["dia_periodo"]}),
        "componentes": _componentes(escolhido, teste, dev), "sensibilidade_do_alvo": _sensibilidade(teste, previsoes),
        "taxa_ponderada_teste": float(np.sum(teste["peso"] * yteste) / np.sum(teste["peso"])),
        "importancia_por_permutacao": importancia_por_permutacao(escolhido, teste, yteste)}
    hash_teste = _escrever(saida / "teste.json", resultado)
    publicar(saida, escolhido, referencia, _reamostras(spec, dev, ydev), _resumo_de_treino(dev, ydev, referencia),
             resultado, hash_teste)
    return resultado


def publicar(saida: Path, escolhido, referencia, reamostras: list, resumo: dict, teste: dict, hash_teste: str) -> None:
    m = teste["metricas"]
    chaves = ("n", "positivos", "perda_log", "pr_auc", "roc_auc", "captura_20", "lift_20", "precisao_20")
    avaliacao = {n: {**{k: m[n][k] for k in chaves},
                     "observado_sobre_esperado": m[n]["calibracao"]["observado_sobre_esperado"]} for n in m}
    resumo = {**resumo, "versao": VERSAO, "spec": teste["spec_escolhida"], "teste_sha256": hash_teste,
              "importancia": teste["importancia_por_permutacao"], "avaliacao": avaliacao}
    joblib.dump({"versao": VERSAO, "modelo": escolhido, "referencia": referencia, "reamostras": reamostras},
                saida / "modelo.joblib")
    _escrever(saida / "resumo.json", resumo)
    sha = {n: hashlib.sha256((saida / n).read_bytes()).hexdigest() for n in ("modelo.joblib", "resumo.json")}
    _escrever(saida / "meta.json", {"versao": VERSAO, "sha256": sha, "scikit-learn": sklearn.__version__})


# ---------- avaliacao temporal 2008 -> 2017 ----------

def temporal(saida: Path = ARTEFATOS) -> dict:
    desenvolvimento = json.loads((saida / "desenvolvimento.json").read_text(encoding="utf-8"))
    d08, teste = carregar_base("2008"), carregar_base("2017", "teste")
    y08, yteste = alvo(d08), alvo(teste)
    previsoes = {nome: fabricar(desenvolvimento[chave]).ajustar(d08, y08).prever(teste)
                 for nome, chave in (("escolhido_2008", "spec_escolhida"), ("dia_periodo_2008", "spec_referencia"))}
    previsoes["global_2008"] = np.full(len(yteste), y08.mean())
    resultado = {"treino": "POF 2008-2009 inteira (regra de comida v1, sem agua registrada)",
                 "avaliacao": "particao de teste de 2017 (mesmas refeicoes do teste do modelo de 2017)",
                 "taxa_2008": float(y08.mean()), "taxa_teste_2017": float(yteste.mean()),
                 "metricas": {n: om.metricas(yteste, p) for n, p in previsoes.items()},
                 "leitura": "Vale para a ORDEM dos contextos (AUC, PR-AUC e captura): o que 2008 aprendeu se mantem em "
                            "2017? O nivel (O/E) mistura consumo e registro: em 2008, 11,8% das pizzas com refrigerante "
                            "tem sabor nao especificado (2,6% em 2017), o que subnotifica o guarana de 2008."}
    _escrever(saida / "temporal.json", resultado)
    return resultado


# ---------- explicacao (desenvolvimento, fora da dobra) ----------

def _sha(caminho: Path) -> str:
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def explicar(saida: Path = ARTEFATOS) -> dict:
    """Grava explicacao.json: importancia por permutacao fora da dobra no desenvolvimento, com a mesma especificacao, as
    mesmas 5 dobras por UPA e a mesma metrica da selecao. Nao treina o modelo publicado, nao le o teste reservado e
    nao altera modelo.joblib, resumo.json, meta.json nem teste.json."""
    meta = json.loads((saida / "meta.json").read_text(encoding="utf-8"))
    desenvolvimento = json.loads((saida / "desenvolvimento.json").read_text(encoding="utf-8"))
    spec = desenvolvimento["spec_escolhida"]
    registrada = desenvolvimento["configuracoes"][desenvolvimento["melhor_por_familia"][spec["familia"]]]
    arquivo = preparo.raiz_processado() / "escolhas_pof_2017.parquet"
    dev = carregar_base("2017", "desenvolvimento")
    y = alvo(dev)
    importancia = importancia_fora_da_dobra(lambda: fabricar(spec), dev, y, om.dobras(dev["upa"]))
    recalculada = next(iter(importancia.values()))["perda_base_por_dobra"]
    confere = bool(np.allclose(recalculada, registrada["perda_por_dobra"], rtol=0, atol=1e-12))
    if not confere:
        raise SystemExit("explicar: as previsoes fora da dobra nao reproduzem desenvolvimento.json; nada foi gravado")
    resultado = {
        "versao": VERSAO, "origem": ORIGEM_IMPORTANCIA, "particao": "desenvolvimento", "teste_reservado_usado": False,
        "pergunta": PERGUNTA_GERAL, "spec": spec, "n": int(len(y)), "positivos": int(y.sum()),
        "metodo": "cada dobra prevista por um modelo treinado nas outras quatro; cada grupo de campos embaralhado dentro "
                  "da dobra de validacao; aumento da perda logaritmica sobre todas as linhas do desenvolvimento",
        "dobras": om.DOBRAS, "agrupamento": "UPA (GroupKFold embaralhado, a mesma divisao de `desenvolver`)",
        "semente": om.SEMENTE, "repeticoes": REPETICOES_PERMUTACAO, "grupos": GRUPOS_PERMUTACAO,
        "criterio": "habilita o campo para exibicao quando a media menos 1 desvio das repeticoes > 0",
        "reproducao": {"perda_por_dobra_registrada": registrada["perda_por_dobra"],
                       "perda_por_dobra_recalculada": recalculada, "confere": confere},
        "modelo_sha256": meta["sha256"]["modelo.joblib"], "resumo_sha256": meta["sha256"]["resumo.json"],
        "desenvolvimento_sha256": _sha(saida / "desenvolvimento.json"), "base_sha256": _sha(arquivo),
        "scikit-learn": sklearn.__version__, "importancia": importancia,
        "nota": "a importancia medida no teste reservado (resumo.json e teste.json) fica como registro historico e "
                "nao habilita mais fatores na tela"}
    _escrever(saida / "explicacao.json", resultado)
    return resultado


def _importancia_geral(pasta: Path, meta: dict) -> dict:
    """Le explicacao.json e confere que ela foi medida para o modelo e o resumo publicados."""
    try:
        explicacao = json.loads((pasta / "explicacao.json").read_text(encoding="utf-8"))
        ligada = (explicacao["versao"] == VERSAO and explicacao["origem"] == ORIGEM_IMPORTANCIA
                  and explicacao["modelo_sha256"] == meta["sha256"]["modelo.joblib"]
                  and explicacao["resumo_sha256"] == meta["sha256"]["resumo.json"])
        importancia = explicacao["importancia"]
    except (OSError, KeyError, TypeError, ValueError) as erro:
        raise preparo.ArtefatoInvalido(f"explicacao.json ausente ou ilegivel ({type(erro).__name__}); rode "
                                       "python -m modelagem explicar") from erro
    if not ligada or set(importancia) != set(GRUPOS_PERMUTACAO):
        raise preparo.ArtefatoInvalido("explicacao.json nao corresponde ao modelo publicado")
    return {c: {"sustentada": bool(v["sustentada"]), "aumento_medio": float(v["aumento_medio"]),
                "desvio": float(v["desvio"]), "origem": ORIGEM_IMPORTANCIA} for c, v in importancia.items()}


# ---------- inferencia ----------

@dataclass(frozen=True)
class Pacote:
    modelo: object
    referencia: object
    reamostras: list
    resumo: dict
    sha256: dict
    guarana_sustentado: bool  # P(guarana | pizza, contexto) melhor que a taxa constante no teste reservado
    importancia_geral: dict  # explicacao.json: campos que ajudam o modelo (desenvolvimento, fora da dobra)


def _guarana_sustentado(pasta: Path, resumo: dict) -> bool:
    """Le o teste.json do artefato (hash conferido contra o resumo) e diz se o componente do guarana generalizou."""
    texto = (pasta / "teste.json").read_text(encoding="utf-8")
    if hashlib.sha256(texto.encode("utf-8")).hexdigest() != resumo["teste_sha256"]:
        raise preparo.ArtefatoInvalido("teste.json diferente do registrado no resumo do Occasion Score observado")
    componentes = json.loads(texto)["componentes"]
    if not componentes:
        return False
    g = componentes["guarana_dado_pizza"]
    return bool(g["perda_log"] < g["perda_log_taxa_constante"])


@cache
def _carregar(pasta: Path) -> Pacote:
    meta = preparo.conferir_artefatos(pasta, "artefatos do Occasion Score observado")
    resumo = json.loads((pasta / "resumo.json").read_text(encoding="utf-8"))
    dados = joblib.load(pasta / "modelo.joblib")  # so artefato deste pipeline, com hash conferido acima
    if dados["versao"] != VERSAO or resumo["versao"] != VERSAO:
        raise preparo.ArtefatoInvalido("modelo e resumo do Occasion Score observado não correspondem")
    return Pacote(dados["modelo"], dados["referencia"], dados["reamostras"], resumo, meta["sha256"],
                  _guarana_sustentado(pasta, resumo), _importancia_geral(pasta, meta))


def carregar(pasta: Path | str | None = None) -> Pacote:
    return _carregar(Path(pasta) if pasta else ARTEFATOS)


def _combina(canonico: dict, chave: str) -> dict | None:
    valores = dict(zip(preparo.CAMPOS_PUBLICO, chave.split("|"), strict=True))
    return valores if all(canonico[c] in (preparo.TODOS, valores[c]) for c in preparo.CAMPOS_PUBLICO) else None


def _publicos(canonico: dict, resumo: dict) -> list[tuple[dict, float]]:
    """Publico das linhas: o informado (peso 1) quando os tres campos vem preenchidos; com 'todos', a composicao
    observada na regiao e area e, se ela nao tiver a combinacao pedida, a composicao nacional."""
    if all(canonico[c] != preparo.TODOS for c in preparo.CAMPOS_PUBLICO):
        return [({c: canonico[c] for c in preparo.CAMPOS_PUBLICO}, 1.0)]
    composicao = resumo["composicao_publico"]
    local = composicao.get(f"{canonico['grande_regiao']}|{canonico['area']}", {})
    publicos = [(v, n) for chave, n in local.items() if (v := _combina(canonico, chave))]
    if not publicos:
        nacional: dict = {}
        for celula in composicao.values():
            for chave, n in celula.items():
                nacional[chave] = nacional.get(chave, 0) + n
        publicos = [(v, n) for chave, n in nacional.items() if (v := _combina(canonico, chave))]
    if not publicos:
        opcoes = {"faixa_etaria": "Qualquer faixa", "sexo": "Qualquer sexo", "escolaridade": "Qualquer escolaridade"}
        alternativas = [rotulo for campo, rotulo in opcoes.items() if canonico[campo] != preparo.TODOS
                        and any(_combina({**canonico, campo: preparo.TODOS}, chave) for chave in nacional)]
        sugestao = " ou ".join(alternativas) if alternativas else "Público geral"
        raise preparo.EntradaInvalida(f"a pesquisa não registrou refeições com essa combinação de público. "
                                      f"Para ampliar a consulta, selecione {sugestao}.")
    return publicos


def refeicoes_do_recorte(canonico: dict, resumo: dict) -> int:
    """Refeicoes do treino com a regiao, a area e o publico informados (o publico geral conta todas da celula)."""
    local = resumo["composicao_publico"].get(f"{canonico['grande_regiao']}|{canonico['area']}", {})
    return int(sum(n for chave, n in local.items() if _combina(canonico, chave)))


def expandir(canonico: dict, resumo: dict) -> tuple[dict, np.ndarray]:
    """Linhas do modelo para um contexto: horas do periodo (se a hora nao veio) e publico 'todos' pela composicao
    observada no treino para a regiao e o tipo de area. Pesos somam 1."""
    if "hora" in canonico:
        horas = [(canonico["hora"], 1.0)]
    else:
        contagem = resumo["horas_por_periodo"][canonico["periodo"]]
        total = sum(contagem.values())
        horas = [(int(h), n / total) for h, n in contagem.items()]
    publicos = _publicos(canonico, resumo)
    total = sum(n for _, n in publicos)
    linhas = [({**canonico, **pub, "hora": h}, ph * n / total) for (h, ph), (pub, n) in product(horas, publicos)]
    dados = {c: np.array([linha[c] for linha, _ in linhas]) for c in
             ("dia_semana", "periodo", "hora", "grande_regiao", "area", *preparo.CAMPOS_PUBLICO)}
    return dados, np.array([p for _, p in linhas])


def _media(modelo, canonico: dict, resumo: dict) -> float:
    dados, pesos = expandir(canonico, resumo)
    return float(pesos @ modelo.prever(dados))


def fatores(pacote: Pacote, canonico: dict, estimativa: float) -> list[dict]:
    """Duas perguntas separadas por campo. `ajuda_o_modelo` (geral): importancia por permutacao fora da dobra no
    desenvolvimento (explicacao.json). `efeito_pp` e `direcao` (deste cenario): estimativa do contexto menos a media
    obtida trocando so esse campo por cada valor observado no treino (ponderada pela frequencia). Campos de publico
    marcados como 'todos' (publico geral) nao entram: o cenario nao tem um valor deles para explicar."""
    campos = [*CAMPOS_CONTEXTO, *(c for c in preparo.CAMPOS_PUBLICO if canonico[c] != preparo.TODOS)]
    saida = []
    for campo in campos:
        efeito = (estimativa - _media_das_alternativas(pacote, canonico, campo)) * 100
        direcao = "neutro" if abs(efeito) < regras.LIMIAR_NEUTRO_PP_OBSERVADO else ("eleva" if efeito > 0 else "reduz")
        geral = pacote.importancia_geral[campo]
        fator = {"campo": campo, "valor": canonico[campo], "efeito_pp": efeito, "direcao": direcao,
                 **(_como_horario(canonico) if campo == "periodo" and "hora" in canonico else {}),
                 "ajuda_o_modelo": bool(geral["sustentada"]),
                 "importancia_geral": {"sustentada": bool(geral["sustentada"]),
                                       "aumento_medio_perda_log": float(geral["aumento_medio"]),
                                       "origem": ORIGEM_IMPORTANCIA}}
        saida.append({**fator, "texto": regras.frase_do_fator(fator)})
    return sorted(saida, key=lambda f: (not f["ajuda_o_modelo"], -abs(f["efeito_pp"])))


def _media_das_alternativas(pacote: Pacote, canonico: dict, campo: str) -> float:
    """Media da estimativa trocando so `campo` por cada valor observado no treino, ponderada pela frequencia do valor.
    Com publico parcial, um valor trocado pode nao ter nenhuma refeicao com o resto do publico (por exemplo 10-17 anos
    com superior): esse valor sai da media e os pesos dos demais sao renormalizados. O valor do proprio cenario ja
    passou pela mesma expansao, entao sempre sobra pelo menos um."""
    soma = peso = 0.0
    for valor, n in pacote.resumo["distribuicoes"][campo].items():
        trocado = {**canonico, campo: int(valor) if campo == "dia_semana" else valor}
        if campo == "periodo":
            trocado.pop("hora", None)  # outro periodo: horas desse periodo, nao a hora informada
        try:
            media = _media(pacote.modelo, trocado, pacote.resumo)
        except preparo.EntradaInvalida:
            continue
        soma += n * media
        peso += n
    return soma / peso


def _como_horario(canonico: dict) -> dict:
    """Com hora exata, a troca do periodo substitui tambem a hora (a alternativa usa as horas de cada periodo): o
    fator passa a ser o horario, medido com hora e periodo juntos; a importancia geral e a do grupo periodo + hora."""
    return {"campo": "horario", "valor": int(canonico["hora"]), "periodo": canonico["periodo"],
            "grupo_de_importancia": "periodo"}


def principais_fatores(lista: list[dict]) -> list[dict]:
    """top_factors: campo que ajuda o modelo em geral E que mudou a estimativa deste cenario (direcao nao neutra)."""
    return [f for f in lista if f["ajuda_o_modelo"] and f["direcao"] != "neutro"][:regras.MAXIMO_FATORES]


def _componentes_do_contexto(pacote: Pacote, dados: dict, pesos: np.ndarray) -> dict | None:
    if not isinstance(pacote.modelo, om.DoisPassos):
        return None
    p, g = pacote.modelo.componentes(dados)
    pizza = float(pesos @ p)
    return {"pizza": pizza, "guarana_dado_pizza": float(pesos @ (p * g)) / pizza,
            "pizza_referencia": pacote.resumo["pizza_treino"],
            "guarana_dado_pizza_referencia": pacote.resumo["guarana_dado_pizza_treino"],
            "guarana_sustentado": pacote.guarana_sustentado}


def avaliar(contexto: dict, pasta: Path | str | None = None) -> dict:
    pacote = carregar(pasta)
    resumo = pacote.resumo
    canonico = preparo.preparar(contexto)
    dados, pesos = expandir(canonico, resumo)
    estimativa = float(pesos @ pacote.modelo.prever(dados))
    reamostradas = [(float(pesos @ m.prever(dados)), taxa) for m, taxa in pacote.reamostras]
    celula = f"{canonico['dia_semana']}|{canonico['periodo']}"
    suporte = resumo["suporte_dia_periodo"].get(celula, {"episodios": 0, "positivos": 0})
    alcance = {"considerados": [regras.DIAS[canonico["dia_semana"]], regras.PERIODOS[canonico["periodo"]]],
               "nao_cobertos": [regras.filtro("grande_regiao", canonico["grande_regiao"]),
                                regras.filtro("area", canonico["area"]),
                                *(regras.filtro(c, canonico[c]) for c in preparo.CAMPOS_PUBLICO
                                  if canonico[c] != preparo.TODOS)]}
    comparacao = regras.comparar(estimativa, resumo["taxa_global_treino"],
                                 regras.intervalo([v - t for v, t in reamostradas]), suporte, alcance)
    comparacao["intervalo_estimativa"] = list(regras.intervalo([v for v, _ in reamostradas]))
    recorte = refeicoes_do_recorte(canonico, resumo)
    if recorte < regras.MINIMO_EPISODIOS:
        comparacao = regras.marcar_recorte_raro(comparacao, recorte)
    componentes = _componentes_do_contexto(pacote, dados, pesos)
    lista = fatores(pacote, canonico, estimativa)
    decisao = regras.decidir_ativacao(comparacao, componentes)
    leitura = {**regras.leitura_do_cenario(comparacao, MEDIDA, "na média das refeições", False),
               "base": BASE_DA_ESTIMATIVA, "recorte": regras.recorte(canonico["grande_regiao"], canonico["area"])}
    return {
        "occasion_score": estimativa, "classification": comparacao["faixa"],
        "top_factors": principais_fatores(lista),
        "recommendation_category": decisao["categoria"], "recommendation_reason": decisao["justificativa"],
        "model_metadata": {"versao": VERSAO, "modelo": regras.MODELOS.get(resumo["spec"]["familia"]),
                           "spec": resumo["spec"], "alvo": MEDIDA, "unidade": "refeição com comida (pessoa × dia × hora)",
                           "fonte": BASE_DA_ESTIMATIVA, "avaliacao_teste": resumo["avaliacao"],
                           "regra": {"id": decisao["regra"], "versao": decisao["versao"],
                                     "inativas": dict(regras.REGRAS_INATIVAS)},
                           "explicacao": {"origem": ORIGEM_IMPORTANCIA, "arquivo": "explicacao.json",
                                          "pergunta_geral": PERGUNTA_GERAL, "pergunta_do_cenario": PERGUNTA_DO_CENARIO},
                           "avisos": list(regras.AVISOS_OBSERVADO), "sha256": dict(pacote.sha256)},
        "leitura": leitura, "proximo_passo": regras.PROXIMOS_PASSOS[decisao["categoria"]],
        "comparacao": {**comparacao, "referencia_nome": "média das refeições com comida da amostra de treino "
                                                        "(POF 2017–2018, sem peso amostral)",
                       "regra_dia_periodo": float(pesos @ pacote.referencia.prever(dados)), "celula_de_suporte": celula},
        "componentes": componentes, "fatores_avaliados": lista,
        "contexto": {**canonico, "hora_informada": "hora" in canonico, "linhas_expandidas": int(len(pesos)),
                     "refeicoes_do_recorte_no_treino": recorte}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Occasion Score observado (protocolo v3)")
    parser.add_argument("etapa", choices=("desenvolver", "testar", "temporal", "explicar", "eda"))
    parser.add_argument("--saida", type=Path, help="pasta de saida (padrao: o artefato publicado; na eda, a pasta "
                                                   f"processada de {preparo.VAR_PROCESSADO})")
    argumentos = parser.parse_args(argv)
    if argumentos.etapa == "eda":
        analise_exploratoria(argumentos.saida)
        destino = argumentos.saida or preparo.raiz_processado()
        sys.stdout.write(json.dumps({"eda": (Path(destino) / "eda.json").as_posix()}, ensure_ascii=False) + "\n")
        return 0
    etapas = {"desenvolver": desenvolver, "testar": testar, "temporal": temporal, "explicar": explicar}
    resultado = etapas[argumentos.etapa](argumentos.saida or ARTEFATOS)
    resumo = {k: resultado[k] for k in ("versao", "escolha", "spec_escolhida", "origem") if k in resultado}
    sys.stdout.write(json.dumps(resumo, ensure_ascii=False, default=float) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
