"""Treino e avaliacao da versao reduzida (`contexto_reduzido_v1`) somente com a POF 2008-2009.

Uso: PYTHONPATH=src python -m treinar_consumo --artefatos <pasta> --registros <pasta fora do repositorio>
Entradas: as cinco saidas de 2008 conferidas por `configs/entradas_2008.json` (CASE_AMBEV_SAIDA_ROOT). Nada de 2017 e
lido. Esta avaliacao e posterior a abertura de 2017: e validacao cruzada agrupada por UPA em 2008, nao teste independente.

Etapas: (1) compara as representacoes do dia so com a logistica; (2) compara habitual, frequencia por dia x periodo,
logistica e boosting na representacao escolhida; (3) reajusta o modelo contextual em todo 2008 e 1000 reamostras de UPAs
para o intervalo da diferenca contra o habitual; (4) grava artefatos (sem linha a linha) e, a parte, as previsoes.
"""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

from aplicacao import aprendizado as ex, previsao_consumo
from occasion.eda_dados import carregar_episodios, conferir_entradas

SEMENTE = 20261116
REPETICOES = 5
DOBRAS = 5
REPLICAS_SELECAO = 2000
REPLICAS_INTERVALO = 1000
ORDEM_REPRESENTACOES = ("D2", "D4", "D7")
ROTULO = ("validação cruzada agrupada por UPA na POF 2008–2009, feita depois da avaliação de 2017–2018: "
          "não é teste independente nem validação temporal")


def carregar(saida: Path) -> dict:
    hashes = conferir_entradas(saida)
    episodios = carregar_episodios(saida)
    canonicos = [previsao_consumo.canonico_de_nativas(e["nat"]) for e in episodios]
    return {"hashes": hashes, "canonicos": canonicos, "y": np.array([e["y"] for e in episodios], dtype=int),
            "grupos": np.array([e["ids"]["cod_upa"] for e in episodios])}


def matriz(canonicos: list[dict], representacao: str) -> np.ndarray:
    return np.vstack([previsao_consumo.codificar(c, representacao) for c in canonicos])


def celulas_baseline(canonicos: list[dict]) -> np.ndarray:
    """Baseline do TAPI: frequencia historica por dia da semana (os sete) e periodo."""
    return np.array([[f"{c['dia_semana']}|{c['periodo']}"] for c in canonicos], dtype=object)


def divisoes(y: np.ndarray, grupos: np.ndarray) -> list:
    saida = []
    for r in range(REPETICOES):
        saida += list(StratifiedGroupKFold(DOBRAS, shuffle=True, random_state=SEMENTE + r).split(y, y, grupos))
    return saida


def interno() -> StratifiedGroupKFold:
    return StratifiedGroupKFold(DOBRAS, shuffle=True, random_state=SEMENTE + 999)


def _sem_previsoes(resultado: dict) -> dict:
    return {nome: {k: v for k, v in c.items() if k not in ("previsoes", "perdas")} for nome, c in resultado["candidatos"].items()}


def comparar_representacoes(dados: dict, externas: list) -> tuple[str, dict]:
    logistica = next(c for c in ex.candidatos_padrao(False) if c.nome == "logistica")
    variantes = [ex.Candidato(f"logistica_{r}", i, True, f"logística, dia: {previsao_consumo.REPRESENTACOES_DIA[r]}", logistica.construir,
                              logistica.espaco, logistica.busca) for i, r in enumerate(ORDEM_REPRESENTACOES)]
    matrizes = {f"logistica_{r}": matriz(dados["canonicos"], r) for r in ORDEM_REPRESENTACOES}
    resultado = ex.avaliar(matrizes, dados["y"], dados["grupos"], externas, interno, variantes, SEMENTE)
    escolha = ex.selecionar(resultado, dados["grupos"], SEMENTE, REPLICAS_SELECAO)
    return escolha["selecionado"].removeprefix("logistica_"), {"selecao": escolha, "candidatos": _sem_previsoes(resultado)}


def contagens(canonicos: list[dict], y: np.ndarray, representacao: str) -> dict:
    episodios, positivos = Counter(), Counter()
    for c, alvo in zip(canonicos, y):
        celula = previsao_consumo.celula_de_suporte(c, representacao)
        episodios[celula] += 1
        positivos[celula] += int(alvo)
    campos = ("dia_semana", "periodo", "grande_regiao", "area", "faixa_etaria", "sexo", "escolaridade")
    return {"suporte": {k: {"episodios": episodios[k], "positivos": positivos[k]} for k in sorted(episodios)},
            "distribuicoes": {campo: dict(sorted(Counter(str(c[campo]) for c in canonicos).items())) for campo in campos},
            "composicao_publico": dict(sorted(Counter("|".join(c[f] for f in previsao_consumo.CAMPOS_PUBLICO) for c in canonicos).items()))}


def executar(saida: Path, artefatos: Path, registros: Path) -> dict:
    dados = carregar(saida)
    externas = divisoes(dados["y"], dados["grupos"])
    representacao, etapa_dia = comparar_representacoes(dados, externas)
    x = matriz(dados["canonicos"], representacao)
    matrizes = {"habitual": x, "frequencia": celulas_baseline(dados["canonicos"]), "logistica": x, "boosting": x}
    candidatos = ex.candidatos_padrao(True)
    resultado = ex.avaliar(matrizes, dados["y"], dados["grupos"], externas, interno, candidatos, SEMENTE)
    escolha = ex.selecionar(resultado, dados["grupos"], SEMENTE, REPLICAS_SELECAO)
    por_nome = {c.nome: c for c in candidatos}
    contextual = escolha["selecionado"] if por_nome[escolha["selecionado"]].ordena_contextos else min(
        (n for n in ("logistica", "boosting")), key=lambda n: escolha["log_loss_medio"][n])
    modelo, params = ex.ajustar(por_nome[contextual], matrizes[contextual], dados["y"], dados["grupos"], interno, SEMENTE)
    reamostras = ex.reamostrar(por_nome[contextual], params, matrizes[contextual], dados["y"], dados["grupos"],
                               REPLICAS_INTERVALO, SEMENTE + 5000)
    pacote = {"versao": previsao_consumo.VERSAO, "representacao_dia": representacao, "colunas": list(previsao_consumo.colunas(representacao)),
              "contextual": contextual, "modelo": modelo, "hiperparametros": params, "reamostras": reamostras,
              "referencia": float(np.mean(dados["y"]))}
    resumo = {
        "versao": previsao_consumo.VERSAO, "fonte": "POF 2008-2009 (IBGE), episódios pessoa × dia × hora com pizza",
        "alvo": "guaraná na mesma ocasião, entre ocasiões com pizza", "alcance_da_avaliacao": ROTULO,
        "entradas_sha256": dados["hashes"], "episodios": int(len(dados["y"])), "positivos": int(dados["y"].sum()),
        "grupos": int(len(np.unique(dados["grupos"]))), "campos": list(previsao_consumo.CAMPOS),
        "campos_removidos": ["renda per capita", "moradores da unidade de consumo", "presença de criança, adolescente e idoso"],
        "particoes": {"tipo": "StratifiedGroupKFold por UPA", "dobras": DOBRAS, "repeticoes": REPETICOES,
                      "sementes": [SEMENTE + r for r in range(REPETICOES)], "interna": {"dobras": DOBRAS, "semente": SEMENTE + 999}},
        "representacao_do_dia": {"escolhida": representacao, "descricao": previsao_consumo.REPRESENTACOES_DIA[representacao],
                                 "nota": "comparação feita só com 2008, depois da avaliação de 2017", **etapa_dia},
        "candidatos": _sem_previsoes(resultado),
        "nota_das_metricas": ("médias por divisão; a mesma validação cruzada escolheu a representação do dia, o "
                              "hiperparâmetro e o modelo, então as métricas são otimistas"),
        "selecao": escolha, "auditoria": resultado["auditoria"],
        "modelo_contextual": {"nome": contextual, "hiperparametros": params,
                              "papel": "selecionado" if contextual == escolha["selecionado"] else "exploratorio",
                              "reamostras": REPLICAS_INTERVALO},
        "referencia_habitual": pacote["referencia"], **contagens(dados["canonicos"], dados["y"], representacao),
        "ambiente": {"scikit-learn": sklearn.__version__, "numpy": np.__version__}}
    _gravar(artefatos, registros, pacote, resumo, resultado)
    return resumo


def _json(caminho: Path, conteudo: dict) -> None:
    caminho.write_text(json.dumps(conteudo, ensure_ascii=False, indent=1, default=_serializavel) + "\n", encoding="utf-8")


def _serializavel(valor: object) -> object:
    if isinstance(valor, np.generic):
        return valor.item()
    raise TypeError(f"valor nao serializavel: {type(valor).__name__}")


def _gravar(artefatos: Path, registros: Path, pacote: dict, resumo: dict, resultado: dict) -> None:
    artefatos.mkdir(parents=True, exist_ok=True)
    registros.mkdir(parents=True, exist_ok=True)
    joblib.dump(pacote, artefatos / "modelo.joblib", compress=3)
    _json(artefatos / "resumo.json", resumo)
    sha = {nome: hashlib.sha256((artefatos / nome).read_bytes()).hexdigest() for nome in ("modelo.joblib", "resumo.json")}
    _json(artefatos / "meta.json", {"versao": previsao_consumo.VERSAO, "sha256": sha, "scikit-learn": sklearn.__version__})
    _json(registros / "previsoes_fora_da_dobra.json",
          {nome: [None if np.isnan(v) else float(v) for v in c["previsoes"]] for nome, c in resultado["candidatos"].items()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino da versão reduzida, só com a POF 2008-2009.")
    parser.add_argument("--artefatos", required=True, type=Path)
    parser.add_argument("--registros", required=True, type=Path, help="pasta fora do repositório (previsões por episódio)")
    argumentos = parser.parse_args()
    saida = os.environ.get("CASE_AMBEV_SAIDA_ROOT")
    if not saida:
        raise SystemExit("defina CASE_AMBEV_SAIDA_ROOT com as saídas de 2008 (configs/entradas_2008.json)")
    executar(Path(saida), argumentos.artefatos, argumentos.registros)


if __name__ == "__main__":
    main()
