"""Modelo da empresa: plano e treino com selecao dentro do treino, e inferencia com o modelo disponibilizado.

A escolha do modelo usa so a validacao dentro do treino; o teste final, quando existe, e apenas relatado. A inferencia
faz a mesma comparacao com o habitual e aplica as mesmas regras do modo historico (`regras`).
"""

import hashlib
import json
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold, TimeSeriesSplit

from aplicacao import ArtefatoInvalido, EntradaInvalida, aprendizado as ex, conferir_artefatos, empresa_dados, regras


# ---------- treino e selecao ----------
# Plano e execucao do experimento com dados da empresa.
#
# Plano (registrado antes do treino): validacao possivel, referencia, candidatos, orcamento e sementes. Validacao temporal
# so quando ha historico: pelo menos 90 dias, 3 meses distintos e 10 ocasioes com guarana depois do corte (ultimos 20% do
# tempo). Sem isso, avaliacao por lojas fora do treino (se houver 5 ou mais) ou por divisao estratificada, sempre com a
# limitacao escrita: nenhuma delas atende a validacao temporal pedida pelo TAPI.
#
# Execucao: escolha so com a validacao dentro do treino; o teste final e relatado e nunca usado para escolher. Grava em
# `pasta` o modelo (codificador, modelo contextual, reamostras, referencia), o resultado agregado (sem linha a linha) e os
# SHA-256. Mesma entrada e mesmo plano produzem o mesmo resultado.

# ---------- plano ----------

VERSAO = "empresa_v1"
SEMENTE = 20261116
FRACAO_TESTE = 0.2
MINIMO_DIAS, MINIMO_MESES, MINIMO_POSITIVOS_TESTE = 90, 3, 10
MINIMO_GRUPOS = 5
BLOCOS_DE_TEMPO, MINIMO_POSITIVOS_POR_BLOCO = 20, 2
MINIMO_BLOCOS_INCERTEZA = 20
MINIMO_POSITIVOS_BOOSTING = 200
REAMOSTRAS = 1000


def _corte(registros: list[dict]) -> tuple[object, list, list]:
    ordenados = sorted(registros, key=lambda r: r["momento"])
    corte = ordenados[int(len(ordenados) * (1 - FRACAO_TESTE))]["momento"]
    return corte, [r for r in ordenados if r["momento"] < corte], [r for r in ordenados if r["momento"] >= corte]


def _validacao(registros: list[dict]) -> dict:
    momentos = [r["momento"] for r in registros]
    dias = (max(momentos) - min(momentos)).days
    meses = len({(m.year, m.month) for m in momentos})
    corte, treino, teste = _corte(registros)
    positivos_teste = sum(r["y"] for r in teste)
    tamanho = max(1, len(treino) // BLOCOS_DE_TEMPO)
    menor_bloco = min(sum(r["y"] for r in treino[i:i + tamanho]) for i in range(0, tamanho * BLOCOS_DE_TEMPO, tamanho))
    if (dias >= MINIMO_DIAS and meses >= MINIMO_MESES and positivos_teste >= MINIMO_POSITIVOS_TESTE
            and menor_bloco >= MINIMO_POSITIVOS_POR_BLOCO):
        return {"tipo": "temporal", "corte": corte.isoformat(), "treino": len(treino), "teste": len(teste),
                "selecao": "divisões em ordem de tempo dentro do período de treino",
                "limitacao": "Um único corte no tempo; o desempenho em períodos mais distantes pode ser outro."}
    motivo = (f"o histórico cobre {dias} dias e {meses} mês(es), com {positivos_teste} ocasiões com guaraná depois do "
              f"corte e no mínimo {menor_bloco} em cada vigésimo do treino; o mínimo é {MINIMO_DIAS} dias, "
              f"{MINIMO_MESES} meses, {MINIMO_POSITIVOS_TESTE} ocasiões depois do corte e {MINIMO_POSITIVOS_POR_BLOCO} "
              "por vigésimo do treino")
    grupos = {r["grupo"] for r in registros if r["grupo"] is not None}
    if len(grupos) >= MINIMO_GRUPOS and all(r["grupo"] is not None for r in registros):
        return {"tipo": "grupos", "corte": None, "treino": len(registros), "teste": 0,
                "selecao": "lojas inteiras fora do treino em cada divisão",
                "limitacao": f"Sem validação temporal ({motivo}): mede generalização para outras lojas e não atende "
                             "à validação temporal pedida no termo de abertura do projeto (TAPI)."}
    return {"tipo": "estratificada", "corte": None, "treino": len(registros), "teste": 0,
            "selecao": "divisões estratificadas pelo alvo",
            "limitacao": f"Sem validação temporal nem grupos ({motivo}): a avaliação é otimista e não atende à "
                         "validação temporal pedida no termo de abertura do projeto (TAPI)."}


def montar_plano(diagnostico: dict) -> dict:
    if not diagnostico["pode_treinar"]:
        bloqueios = [p["mensagem"] for p in diagnostico["problemas"] if p["bloqueia"]]
        raise EntradaInvalida("o arquivo não sustenta o treino: " + " ".join(bloqueios))
    registros = diagnostico["registros"]
    validacao = _validacao(registros)
    treino = _corte(registros)[1] if validacao["tipo"] == "temporal" else registros
    positivos_treino = sum(r["y"] for r in treino)
    candidatos = ex.candidatos_padrao(positivos_treino >= MINIMO_POSITIVOS_BOOSTING)
    plano = {
        "versao": VERSAO, "unidade": diagnostico["unidade"], "alvo": diagnostico["alvo"]["definicao"],
        "variaveis": [v["campo"] for v in diagnostico["variaveis"]["usadas"]],
        "variaveis_excluidas": diagnostico["variaveis"]["excluidas"], "validacao": validacao,
        "referencia": {"definicao": "taxa média do período de treino (só ocasiões usadas no treino, anteriores ao corte "
                                    "quando há validação temporal)"},
        "candidatos": [c.nome for c in candidatos],
        "orcamento": {c.nome: c.orcamento(3) for c in candidatos},
        "boosting": "incluído" if positivos_treino >= MINIMO_POSITIVOS_BOOSTING else
                    f"fora: {positivos_treino} ocasiões com guaraná no treino (mínimo {MINIMO_POSITIVOS_BOOSTING})",
        "metrica_de_escolha": "log-loss nas divisões de validação do treino; o teste final só é relatado",
        "reamostras_para_intervalo": REAMOSTRAS, "sementes": {"geral": SEMENTE}, "dados_sha256": diagnostico["dados_sha256"],
        "mapeamento": diagnostico["mapeamento"], "declaracao": diagnostico["declaracao"]}
    plano["configuracao_sha256"] = hashlib.sha256(json.dumps(plano, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return plano


# ---------- execucao ----------

DOBRAS_SELECAO, DOBRAS_INTERNAS = 4, 3


def _divisoes(tipo: str, x: np.ndarray, y: np.ndarray, grupos: np.ndarray | None) -> tuple[list, Callable, np.ndarray | None]:
    if tipo == "temporal":
        return list(TimeSeriesSplit(DOBRAS_SELECAO).split(x)), lambda: TimeSeriesSplit(DOBRAS_INTERNAS), None
    if tipo == "grupos":
        externas = list(StratifiedGroupKFold(DOBRAS_SELECAO, shuffle=True, random_state=SEMENTE).split(x, y, grupos))
        return externas, lambda: GroupKFold(DOBRAS_INTERNAS), grupos
    externas = list(StratifiedKFold(DOBRAS_SELECAO, shuffle=True, random_state=SEMENTE).split(x, y))
    return externas, lambda: StratifiedKFold(DOBRAS_INTERNAS, shuffle=True, random_state=SEMENTE + 1), None


def _blocos(tipo: str, registros: list[dict]) -> np.ndarray | None:
    """Grupos para reamostrar: semanas no temporal, lojas na validacao por grupos, nenhum no estratificado."""
    if tipo == "temporal":
        return np.array(["%d-%02d" % r["momento"].isocalendar()[:2] for r in registros])
    if tipo == "grupos":
        return np.array([str(r["grupo"]) for r in registros])
    return None


def _contagens(registros: list[dict], campos: tuple) -> dict:
    episodios, positivos = Counter(), Counter()
    for r in registros:
        celula = empresa_dados.Codificador.celula_de_suporte(r)
        episodios[celula] += 1
        positivos[celula] += r["y"]
    return {"suporte": {k: {"episodios": episodios[k], "positivos": positivos[k]} for k in sorted(episodios)},
            "distribuicoes": {c: dict(sorted(Counter(r[c] for r in registros).items(), key=lambda kv: str(kv[0])))
                              for c in campos if c not in ("preco", "temperatura")}}


def _teste_final(candidatos, matrizes_treino, matrizes_teste, y_tr, y_te, interno, referencia) -> tuple[dict, dict]:
    ajustados, metricas = {}, {}
    for c in candidatos:
        modelo, params = ex.ajustar(c, matrizes_treino[c.nome], y_tr, None, interno, SEMENTE)
        ajustados[c.nome] = (modelo, params)
        metricas[c.nome] = ex.resumo(y_te, modelo.predict_proba(matrizes_teste[c.nome])[:, 1], referencia)
    return ajustados, metricas


def _ajuste_final(plano: dict, codificador, candidatos: list, contextual: str, matrizes: dict, y_tr: np.ndarray,
                  grupos: np.ndarray | None, interno: Callable, referencia: float, teste: list,
                  posteriores: int) -> tuple[object, dict, dict]:
    """Modelo contextual final e o relato do teste final (quando ha periodo posterior), que nunca escolhe nada."""
    por_nome = {c.nome: c for c in candidatos}
    if not teste:
        modelo, params = ex.ajustar(por_nome[contextual], matrizes[contextual], y_tr, grupos, interno, SEMENTE)
        return modelo, params, {"disponivel": False, "usado_para_escolher": False,
                                "motivo": "sem período posterior separado: ver a limitação da validação"}
    y_te = np.array([r["y"] for r in teste])
    matrizes_teste = {nome: codificador.entrada(nome, teste) for nome in plano["candidatos"]}
    ajustados, metricas_teste = _teste_final(candidatos, matrizes, matrizes_teste, y_tr, y_te, interno, referencia)
    modelo, params = ajustados[contextual]
    return modelo, params, {"disponivel": True, "usado_para_escolher": False, "ocasioes": len(teste),
                            "excluidas_por_canal_novo": posteriores - len(teste), "metricas": metricas_teste}


def _avisar(etapa: Callable[[str], None] | None, nome: str) -> None:
    if etapa:
        etapa(nome)


def executar(plano: dict, diagnostico: dict, pasta: Path, etapa: Callable[[str], None] | None = None) -> dict:
    _avisar(etapa, "preparacao")
    tipo = plano["validacao"]["tipo"]
    registros = sorted(diagnostico["registros"], key=lambda r: r["momento"])
    corte = datetime.fromisoformat(plano["validacao"]["corte"]) if tipo == "temporal" else None
    treino = [r for r in registros if corte is None or r["momento"] < corte]
    codificador = empresa_dados.ajustar(treino, plano["variaveis"])
    posteriores = [r for r in registros if corte is not None and r["momento"] >= corte]
    teste = [r for r in posteriores if "canal" not in codificador.campos or r["canal"] in codificador.canais]
    y_tr = np.array([r["y"] for r in treino])
    x_tr = codificador.matriz(treino)
    matrizes = {nome: codificador.entrada(nome, treino) for nome in plano["candidatos"]}
    grupos = np.array([str(r["grupo"]) for r in treino]) if tipo == "grupos" else None
    externas, interno, grupos_avaliacao = _divisoes(tipo, x_tr, y_tr, grupos)
    candidatos = [c for c in ex.candidatos_padrao(True) if c.nome in plano["candidatos"]]
    _avisar(etapa, "ajuste_e_selecao")
    avaliacao = ex.avaliar(matrizes, y_tr, grupos_avaliacao, externas, interno, candidatos, SEMENTE)
    blocos = _blocos(tipo, treino)
    selecao = {**ex.selecionar(avaliacao, blocos, SEMENTE), "base": "validacao_no_treino"}
    referencia = float(np.mean(y_tr))
    por_nome = {c.nome: c for c in candidatos}
    contextual = selecao["selecionado"] if por_nome[selecao["selecionado"]].ordena_contextos else min(
        (n for n in plano["candidatos"] if por_nome[n].ordena_contextos), key=lambda n: selecao["log_loss_medio"][n])
    _avisar(etapa, "teste_final")
    modelo, params, teste_final = _ajuste_final(plano, codificador, candidatos, contextual, matrizes, y_tr,
                                                grupos_avaliacao, interno, referencia, teste, len(posteriores))
    _avisar(etapa, "reamostragem")
    reamostras = ex.reamostrar(por_nome[contextual], params, matrizes[contextual], y_tr, blocos,
                               plano["reamostras_para_intervalo"], SEMENTE + 5000)
    papel = "selecionado" if contextual == selecao["selecionado"] else "exploratorio"
    blocos_distintos = 0 if blocos is None else len(set(blocos.tolist()))
    incerteza = blocos_distintos >= MINIMO_BLOCOS_INCERTEZA
    resultado = {
        "versao": VERSAO, "plano_sha256": plano["configuracao_sha256"], "dados_sha256": plano["dados_sha256"],
        "validacao": plano["validacao"], "selecao": selecao, "teste_final": teste_final,
        "candidatos": {n: {k: v for k, v in c.items() if k not in ("previsoes", "perdas")}
                       for n, c in avaliacao["candidatos"].items()},
        "modelo_contextual": {"nome": contextual, "papel": papel, "hiperparametros": params},
        "incerteza": {"disponivel": incerteza, "blocos": blocos_distintos, "minimo": MINIMO_BLOCOS_INCERTEZA,
                      "unidade": {"temporal": "semana", "grupos": "loja"}.get(tipo, "nenhuma")},
        "referencia_habitual": referencia, "ocasioes_de_treino": len(treino), "positivos_de_treino": int(y_tr.sum()),
        "campos": list(codificador.campos), "canais": list(codificador.canais),
        **_contagens(treino, codificador.campos)}
    _avisar(etapa, "gravacao")
    _gravar(Path(pasta), {"versao": VERSAO, "codificador": codificador, "contextual": contextual, "modelo": modelo,
                          "reamostras": reamostras, "referencia": referencia, "papel": papel,
                          "incerteza_disponivel": incerteza}, resultado)
    return json.loads(_texto(resultado))


def _serializavel(valor: object) -> object:
    if isinstance(valor, np.generic):
        return valor.item()
    raise TypeError(f"valor não serializável: {type(valor).__name__}")


def _texto(conteudo: dict) -> str:
    return json.dumps(conteudo, ensure_ascii=False, indent=1, default=_serializavel, sort_keys=True)


def _gravar(pasta: Path, pacote: dict, resultado: dict) -> None:
    pasta.mkdir(parents=True, exist_ok=True)
    joblib.dump(pacote, pasta / "modelo.joblib", compress=3)
    (pasta / "resultado.json").write_text(_texto(resultado) + "\n", encoding="utf-8")
    sha = {n: hashlib.sha256((pasta / n).read_bytes()).hexdigest() for n in ("modelo.joblib", "resultado.json")}
    (pasta / "meta.json").write_text(_texto({"versao": VERSAO, "sha256": sha, "scikit-learn": sklearn.__version__}) + "\n",
                                     encoding="utf-8")


# ---------- inferencia com o modelo da empresa ----------
# Avaliar um contexto com o modelo da empresa disponibilizado: mesma comparacao com o habitual e mesmas regras.

MODO = "dados_da_empresa"
REFERENCIA = "taxa média do período de treino dos dados enviados"


@lru_cache(maxsize=4)
def carregar(pasta: Path) -> tuple[dict, dict, dict]:
    """Modelo, resultado e metadados de um treino; integridade conferida antes de abrir o joblib."""
    meta = conferir_artefatos(pasta, "modelo da empresa")
    try:
        resultado = json.loads((pasta / "resultado.json").read_text(encoding="utf-8"))
        return joblib.load(pasta / "modelo.joblib"), resultado, meta
    except (ImportError, AttributeError) as erro:  # classe gravada em modulo que mudou ou deixou de existir
        do_projeto = isinstance(erro, AttributeError) or (getattr(erro, "name", None) or "").startswith("aplicacao")
        motivo = "gerado por outra versão do código: treine novamente" if do_projeto else "exige um pacote não instalado"
        raise ArtefatoInvalido(f"modelo da empresa {motivo}") from erro
    except (OSError, ValueError, EOFError) as erro:
        raise ArtefatoInvalido(f"modelo da empresa ilegível ({type(erro).__name__})") from erro


def _alcance(resultado: dict) -> str:
    validacao = resultado["validacao"]
    if validacao["tipo"] == "temporal":
        return f"validação temporal: treino antes de {validacao['corte'][:10]}, avaliação depois; {validacao['limitacao']}"
    return validacao["limitacao"]


def suporte_do_contexto(registro: dict, campos: tuple, contextual: str, contagens: dict) -> tuple[str, dict, dict]:
    """Contagem por dia, periodo e regiao; sem regiao quando o modelo contextual e a frequencia, que nao a usa. As
    variaveis opcionais do modelo (canal, promocao, preco, temperatura) nao entram na contagem e ficam declaradas."""
    contados = [regras.DIAS[registro["dia_semana"]], regras.PERIODOS[registro["periodo"]]]
    nao_cobertos = [regras.filtro(c, registro[c]) for c in empresa_dados.OPCIONAIS if c in campos]
    celula = empresa_dados.Codificador.celula_de_suporte(registro)
    if contextual == "frequencia":
        prefixo = f"{empresa_dados.Codificador.celula(registro)}|"
        alcance = {"considerados": contados,
                   "nao_cobertos": [regras.filtro("grande_regiao", registro["grande_regiao"]), *nao_cobertos]}
        return prefixo + "todas_as_regioes", regras.somar_suporte(contagens, prefixo), alcance
    alcance = {"considerados": [*contados, registro["grande_regiao"]], "nao_cobertos": nao_cobertos}
    return celula, contagens.get(celula, {"episodios": 0, "positivos": 0}), alcance


def avaliar(contexto: dict, pasta: Path, experimento: str) -> dict:
    pacote, resultado, meta = carregar(pasta)
    codificador: empresa_dados.Codificador = pacote["codificador"]
    registro = empresa_dados.preparar(contexto, codificador)
    contextual = pacote["contextual"]
    linha = codificador.entrada(contextual, [registro])
    estimativa = float(pacote["modelo"].predict_proba(linha)[0, 1])
    diferencas = [float(m.predict_proba(linha)[0, 1]) - taxa for m, taxa in pacote["reamostras"]]
    celula, suporte, alcance = suporte_do_contexto(registro, codificador.campos, contextual, resultado["suporte"])
    r = regras.comparar(estimativa, pacote["referencia"], regras.intervalo(diferencas), suporte, alcance,
                            incerteza_disponivel=pacote["incerteza_disponivel"])
    decisao = regras.decidir(r["faixa"], pacote["papel"], [r["nota"]] if r["nota"] else [], alcance["nao_cobertos"])
    distribuicoes = {c: {(int(k) if c in ("dia_semana", "promocao") else k): n for k, n in d.items()}
                     for c, d in resultado["distribuicoes"].items()}

    def prever(registros: list[dict]) -> np.ndarray:
        return pacote["modelo"].predict_proba(codificador.entrada(contextual, registros))[:, 1]

    lista = [] if contextual == "habitual" else (  # taxa constante: sem fatores
        regras.fatores_por_campo(prever, [registro], [1.0], tuple(distribuicoes), distribuicoes)[:regras.MAXIMO_FATORES])
    leitura = {**regras.leitura_do_cenario(r, "Guaraná em pedidos com pizza", "na referência do período de treino",
                                           pacote["papel"] == "exploratorio"),
               "base": "Modelo treinado com os dados da empresa", "recorte": None}
    avisos = ["Resultado calculado com os dados enviados pela empresa; vale para o período e as lojas desse arquivo."]
    if contextual == "frequencia":
        avisos.append("O modelo de frequência usa só o dia da semana e o período: região e demais campos não alteram a "
                      "estimativa.")
    if pacote["papel"] == "exploratorio":
        avisos.insert(0, "Nenhum modelo por contexto superou a taxa habitual com segurança; a estimativa é exploratória.")
    return {
        "occasion_score": estimativa, "classification": r["faixa"],
        "top_factors": [{**f, "texto": regras.frase_do_fator(f)} for f in lista],
        "recommendation_category": decisao["categoria"], "recommendation_reason": decisao["justificativa"],
        "leitura": leitura, "proximo_passo": regras.PROXIMOS_PASSOS[decisao["categoria"]],
        "model_metadata": {"modo": MODO, "versao": resultado["versao"], "experimento": experimento,
                           "fonte": "arquivo enviado pela empresa", "dados_sha256": resultado["dados_sha256"],
                           "alcance_da_avaliacao": _alcance(resultado),
                           "modelo_selecionado": regras.MODELOS[resultado["selecao"]["selecionado"]],
                           "modelo_contextual": regras.MODELOS[pacote["contextual"]],
                           "papel_do_modelo_contextual": pacote["papel"], "regra": {"id": decisao["regra"], "versao": decisao["versao"]},
                           "avisos": avisos, "sha256": dict(meta["sha256"])},  # copia: nao aponta para o cache
        "comparacao": {**r, "referencia_nome": REFERENCIA, "celula_de_suporte": celula},
        "contexto": {"dia": {"dia_semana": registro["dia_semana"], "representacao": "D7",
                             "como_o_modelo_usa": "cada dia da semana separado"},
                     "periodo": regras.PERIODOS[registro["periodo"]], "publico": "não se aplica a este modelo"}}
