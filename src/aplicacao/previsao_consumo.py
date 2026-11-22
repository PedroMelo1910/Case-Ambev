"""Modelo historico de consumo (POF 2008-2009): contrato das entradas e inferencia com o artefato empacotado.

Primeiro bloco: os campos do contexto, sua validacao e a codificacao, igual no treino e na inferencia. Segundo bloco:
carga do artefato `artifacts/contexto_reduzido_v1` (hash e versao do scikit-learn conferidos antes de abrir o modelo) e
avaliacao de um contexto. O treino fica em `src/treinar_consumo.py`, fora do pacote servido, porque le os dados
de 2008.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType

import joblib
import numpy as np

from aplicacao import ArtefatoInvalido, EntradaInvalida, conferir_artefatos, conferir_campos, conferir_inteiro, recebido, regras
from occasion.variaveis import carregar_regras, faixa_hora


# ---------- contrato e codificacao das entradas ----------
# Contexto da versao reduzida: sete campos, sem renda nem composicao da casa, mesma codificacao no treino e na inferencia.
#
# Campos: dia da semana (1=segunda ... 7=domingo), hora (0 a 23, convertida no periodo registrado pela regra de
# `occasion.variaveis`) ou, no lugar dela, o proprio periodo (`manha`, `almoco`, `tarde`, `noite`; o modelo so distingue
# o periodo, entao nenhuma hora representativa e inventada), grande regiao, tipo de area, e tres campos de publico (faixa etaria, sexo, escolaridade em tres
# grupos). Cada campo de publico aceita "todos": a inferencia faz a media sobre a composicao observada no treino; aqui
# "todos" so e validado e nunca e codificado. Nada e imputado: campo ausente, extra ou invalido e recusado.
#
# Tres representacoes do dia sao oferecidas para comparacao com os dados de desenvolvimento: D7 (um indicador por dia),
# D4 (segunda a quinta, sexta, sabado, domingo) e D2 (sexta a domingo e a interacao com a noite, como no protocolo v1.3).

VERSAO = "contexto_reduzido_v1"
CAMPOS = ("dia_semana", "hora", "grande_regiao", "area", "faixa_etaria", "sexo", "escolaridade")
CAMPOS_PUBLICO = ("faixa_etaria", "sexo", "escolaridade")
TODOS = "todos"
_REGRAS = carregar_regras()
PERIODOS = tuple(_REGRAS["faixas_hora"])
REGIOES = tuple(_REGRAS["regioes"])
AREAS = tuple(_REGRAS["areas"])
FAIXAS_ETARIAS = tuple(f["nome"] for f in _REGRAS["faixas_etarias"])
SEXOS = tuple(_REGRAS["sexos"])
ESCOLARIDADES = tuple(_REGRAS["escolaridade_3"])
VALORES_PUBLICO = MappingProxyType({"faixa_etaria": FAIXAS_ETARIAS, "sexo": SEXOS, "escolaridade": ESCOLARIDADES})
_SEXTA_A_DOMINGO = frozenset(_REGRAS["dias_sexta_a_domingo"])

REPRESENTACOES_DIA = MappingProxyType({
    "D2": "sexta a domingo juntos, com destaque para a noite desses dias",
    "D4": "segunda a quinta juntos; sexta, sábado e domingo separados",
    "D7": "cada dia da semana separado"})
_GRUPOS_D4 = MappingProxyType({1: "segunda_a_quinta", 2: "segunda_a_quinta", 3: "segunda_a_quinta",
                               4: "segunda_a_quinta", 5: "sexta", 6: "sabado", 7: "domingo"})


def grupo_do_dia(dia: int, representacao: str) -> str:
    """Grupo em que a representacao coloca o dia (1=segunda ... 7=domingo)."""
    if representacao == "D7":
        return f"dia_{dia}"
    if representacao == "D4":
        return _GRUPOS_D4[dia]
    if representacao == "D2":
        return "sexta_a_domingo" if dia in _SEXTA_A_DOMINGO else "segunda_a_quinta"
    raise ValueError(f"representacao do dia desconhecida: {representacao!r}")


def _colunas_dia(representacao: str) -> tuple[str, ...]:
    if representacao == "D7":
        return tuple(f"dia_{d}" for d in range(1, 8))
    if representacao == "D4":
        return ("dia_segunda_a_quinta", "dia_sexta", "dia_sabado", "dia_domingo")
    if representacao == "D2":
        return ("sexta_a_domingo", "sexta_a_domingo_x_noite")
    raise ValueError(f"representacao do dia desconhecida: {representacao!r}")


def colunas(representacao: str) -> tuple[str, ...]:
    """Colunas codificadas na ordem fixa: periodo, dia, regiao, area, faixa etaria, sexo, escolaridade."""
    return (tuple(f"hora_{p}" for p in PERIODOS) + _colunas_dia(representacao)
            + tuple(f"regiao_{r.lower().replace('-', '_')}" for r in REGIOES) + tuple(f"area_{a}" for a in AREAS)
            + tuple(f"idade_{f}" for f in FAIXAS_ETARIAS) + ("sexo_feminino",)
            + tuple(f"escolaridade_{e}" for e in ESCOLARIDADES))


def _categoria(valor: object, campo: str, validos: tuple[str, ...]) -> str:
    if type(valor) is not str or valor not in validos:
        raise EntradaInvalida(f"{campo} deve ser um de {list(validos)}, recebido {recebido(valor)}")
    return valor


def campos_informados(contexto: Mapping[str, object], campos: tuple[str, ...]) -> tuple[str, ...]:
    """Campos esperados com `periodo` no lugar de `hora` quando o contexto traz o periodo; recusa os dois ou nenhum."""
    if ("hora" in contexto) == ("periodo" in contexto):
        raise EntradaInvalida("informe exatamente um entre hora e periodo (o modelo usa só o período do dia)")
    if "periodo" not in contexto:
        return campos
    return tuple("periodo" if campo == "hora" else campo for campo in campos)


def periodo_informado(contexto: Mapping[str, object]) -> tuple[str, int | None]:
    """Periodo do contexto e a hora, se veio: da `hora` pela regra congelada ou do `periodo` validado."""
    if "periodo" in contexto:
        return _categoria(contexto["periodo"], "periodo", PERIODOS), None
    hora = conferir_inteiro(contexto["hora"], "hora", 0, 23)
    return faixa_hora(hora), hora


def preparar(contexto: Mapping[str, object]) -> dict:
    """Contexto validado na forma canonica (com `periodo`); nao altera a entrada e nao imputa nada."""
    if not isinstance(contexto, Mapping):
        raise EntradaInvalida(f"contexto deve ser um dicionario com os campos {list(CAMPOS)}")
    conferir_campos(contexto, campos_informados(contexto, CAMPOS), "contexto",
                    " (renda, composicao da casa, clima e alvo nao fazem parte desta versao)")
    periodo, hora = periodo_informado(contexto)
    canonico = {"dia_semana": conferir_inteiro(contexto["dia_semana"], "dia_semana", 1, 7), "periodo": periodo,
                "grande_regiao": _categoria(contexto["grande_regiao"], "grande_regiao", REGIOES),
                "area": _categoria(contexto["area"], "area", AREAS)}
    for campo in CAMPOS_PUBLICO:
        canonico[campo] = _categoria(contexto[campo], campo, (*VALORES_PUBLICO[campo], TODOS))
    return canonico if hora is None else {**canonico, "hora": hora}


def canonico_de_nativas(nativas: Mapping[str, object]) -> dict:
    """Registro de treino (linha de `variaveis_nativas_2008`) na mesma forma canonica, so com os sete campos."""
    canonico = {"dia_semana": int(nativas["dia_semana"]), "periodo": nativas["faixa_hora"],
                "grande_regiao": nativas["grande_regiao"], "area": nativas["area"],
                "faixa_etaria": nativas["faixa_etaria"], "sexo": nativas["sexo"], "escolaridade": nativas["escolaridade"]}
    vocabulario = {"periodo": PERIODOS, "grande_regiao": REGIOES, "area": AREAS, **VALORES_PUBLICO}
    fora = {campo: canonico[campo] for campo, validos in vocabulario.items() if canonico[campo] not in validos}
    if fora or not 1 <= canonico["dia_semana"] <= 7:
        raise ValueError(f"registro de treino fora do vocabulário: {fora or canonico['dia_semana']}")
    return canonico


def codificar(canonico: Mapping[str, object], representacao: str) -> np.ndarray:
    """Linha 0/1 nas `colunas(representacao)`. Publico "todos" precisa ser expandido antes (ver `_expandir_publico`)."""
    abertos = [campo for campo in CAMPOS_PUBLICO if canonico[campo] == TODOS]
    if abertos:
        raise EntradaInvalida(f"campos com 'todos' precisam ser expandidos antes da codificacao: {abertos}")
    dia, periodo = canonico["dia_semana"], canonico["periodo"]
    if representacao == "D2":
        fim = int(dia in _SEXTA_A_DOMINGO)
        valores_dia = [fim, int(fim and periodo == "noite")]
    else:
        grupo = grupo_do_dia(dia, representacao)
        valores_dia = [int(c in (grupo, f"dia_{grupo}")) for c in _colunas_dia(representacao)]
    linha = ([int(periodo == p) for p in PERIODOS] + valores_dia + [int(canonico["grande_regiao"] == r) for r in REGIOES]
             + [int(canonico["area"] == a) for a in AREAS] + [int(canonico["faixa_etaria"] == f) for f in FAIXAS_ETARIAS]
             + [int(canonico["sexo"] == "feminino")] + [int(canonico["escolaridade"] == e) for e in ESCOLARIDADES])
    return np.asarray(linha, dtype=float)


def celula_de_suporte(canonico: Mapping[str, object], representacao: str) -> str:
    """Celula usada para medir o suporte: grupo do dia na representacao, periodo e grande regiao."""
    return f"{grupo_do_dia(canonico['dia_semana'], representacao)}|{canonico['periodo']}|{canonico['grande_regiao']}"


# ---------- inferencia com o modelo historico ----------
# Demonstracao historica: avaliar um contexto com a versao reduzida treinada so com a POF 2008-2009.
#
# Fluxo: contexto -> estimativa -> comparacao com o habitual -> evidencia (suporte e intervalo) -> sugestao de priorizacao.
# O publico "todos" e a media das estimativas sobre a composicao observada em 2008 (contagens agregadas no resumo).
# Os artefatos so sao carregados se os SHA-256 de `meta.json` e a versao do scikit-learn conferem.

ARTEFATOS = Path(__file__).resolve().parents[2] / "artifacts" / VERSAO
MODO = "demonstracao_historica"
CAMPOS_DE_CONTEXTO = ("dia_semana", "periodo", "grande_regiao", "area")
MEDIDA = "Guaraná em refeições com pizza"
BASE_DA_ESTIMATIVA = "Estimativa histórica de consumo · POF/IBGE, 2008–2009"
AVISO_FREQUENCIA = ("O modelo de frequência usa só o dia da semana e o período: região, tipo de área e público não "
                    "alteram a estimativa.")


@dataclass(frozen=True)
class Pacote:
    resumo: dict
    modelo: object
    reamostras: list
    referencia: float
    representacao: str
    colunas: list
    contextual: str
    papel: str
    sha256: dict


@cache
def _carregar(base: Path) -> Pacote:
    meta = conferir_artefatos(base, "artefatos da versão reduzida")
    resumo = json.loads((base / "resumo.json").read_text(encoding="utf-8"))
    dados = joblib.load(base / "modelo.joblib")  # so artefato deste pipeline, com hash conferido acima
    if dados["versao"] != VERSAO or abs(dados["referencia"] - resumo["referencia_habitual"]) > 1e-12:
        raise ArtefatoInvalido("modelo e resumo da versão reduzida não correspondem")
    return Pacote(resumo, dados["modelo"], dados["reamostras"], dados["referencia"], dados["representacao_dia"],
                  dados["colunas"], dados["contextual"], resumo["modelo_contextual"]["papel"], meta["sha256"])


def carregar(base: Path | str | None = None) -> Pacote:
    return _carregar(Path(base) if base else ARTEFATOS)


def _expandir_publico(canonico: dict, composicao: dict) -> tuple[list[dict], np.ndarray, str]:
    abertos = [c for c in CAMPOS_PUBLICO if canonico[c] == TODOS]
    if not abertos:
        return [canonico], np.array([1.0]), "informado"
    registros, pesos = [], []
    for chave, n in composicao.items():
        valores = dict(zip(CAMPOS_PUBLICO, chave.split("|")))
        if all(canonico[c] in (TODOS, valores[c]) for c in CAMPOS_PUBLICO):
            registros.append({**canonico, **valores})
            pesos.append(n)
    if not registros:
        raise EntradaInvalida("não há ocasiões de 2008–2009 com essa combinação de público; escolha 'todos' em um "
                              "dos campos")
    nomes = ", ".join(regras.CAMPOS[c].lower() for c in abertos)
    descricao = f"média sobre a composição observada em 2008–2009 para {nomes} (público 'todos')"
    return registros, np.asarray(pesos, dtype=float) / sum(pesos), descricao


def entrada(pacote: Pacote, registros: list[dict]) -> np.ndarray:
    """Entrada do modelo contextual, igual a do treino: celula dia x periodo para a frequencia, colunas codificadas
    para os demais (ver treinar_consumo)."""
    if pacote.contextual == "frequencia":
        return np.array([[f"{r['dia_semana']}|{r['periodo']}"] for r in registros], dtype=object)
    return np.vstack([codificar(r, pacote.representacao) for r in registros])


def _previsor(pacote: Pacote):
    def prever(registros: list[dict]) -> np.ndarray:
        return pacote.modelo.predict_proba(entrada(pacote, registros))[:, 1]
    return prever


def _distribuicoes(resumo: dict) -> dict:
    return {campo: {(int(v) if campo == "dia_semana" else v): n for v, n in contagens.items()}
            for campo, contagens in resumo["distribuicoes"].items()}


def _fatores(pacote: Pacote, registros: list[dict], pesos: np.ndarray, canonico: dict) -> list[dict]:
    if pacote.contextual == "habitual":  # taxa constante: nenhum campo muda a estimativa, a lista fica vazia
        return []
    campos = CAMPOS_DE_CONTEXTO + tuple(c for c in CAMPOS_PUBLICO if canonico[c] != TODOS)
    lista = regras.fatores_por_campo(_previsor(pacote), registros, pesos, campos,
                              _distribuicoes(pacote.resumo))
    return [{**f, "texto": regras.frase_do_fator(f)} for f in lista[:regras.MAXIMO_FATORES]]


def _avisos(pacote: Pacote) -> list[str]:
    avisos = ["Estimativa calculada com ocasiões de consumo com pizza registradas na POF 2008–2009: não mede pedidos, "
              "vendas, preços nem a marca Guaraná Antarctica."]
    if pacote.contextual == "frequencia":
        avisos.append(AVISO_FREQUENCIA)
    if pacote.papel == "exploratorio":
        avisos.insert(0, "Nos dados de 2008–2009, nenhum modelo por contexto superou a taxa habitual com segurança. "
                         "A estimativa por contexto é exploratória.")
    return avisos


def _metadados(pacote: Pacote, decisao: dict) -> dict:
    resumo = pacote.resumo
    selecionado = resumo["selecao"]["selecionado"]
    metricas = {nome: {k: c["metricas"][k] for k in ("log_loss", "brier", "auc", "ap")}
                for nome, c in resumo["candidatos"].items()}
    return {"modo": MODO, "versao": VERSAO, "fonte": resumo["fonte"], "alvo": resumo["alvo"],
            "alcance_da_avaliacao": resumo["alcance_da_avaliacao"],
            "modelo_selecionado": regras.MODELOS[selecionado], "modelo_contextual": regras.MODELOS[pacote.contextual],
            "papel_do_modelo_contextual": pacote.papel, "metricas_de_validacao": metricas,
            "campos_removidos": list(resumo["campos_removidos"]),  # copias: a resposta nao pode apontar para o cache
            "regra": {"id": decisao["regra"], "versao": decisao["versao"]},
            "avisos": _avisos(pacote), "sha256": dict(pacote.sha256)}


def suporte_do_contexto(pacote: Pacote, canonico: dict) -> tuple[str, dict, dict]:
    """Celula contada, contagem e alcance: grupo do dia, periodo e regiao; sem regiao quando o modelo contextual e a
    frequencia, que nao a usa. Filtros escolhidos fora da contagem (area, publico informado) ficam declarados."""
    grupo = grupo_do_dia(canonico["dia_semana"], pacote.representacao)
    contados = [regras.GRUPOS_DIA[grupo], regras.PERIODOS[canonico["periodo"]]]
    nao_cobertos = [regras.filtro("area", canonico["area"])]
    nao_cobertos += [regras.filtro(c, canonico[c]) for c in CAMPOS_PUBLICO if canonico[c] != TODOS]
    celula = celula_de_suporte(canonico, pacote.representacao)
    if pacote.contextual == "frequencia":
        prefixo = f"{grupo}|{canonico['periodo']}|"
        alcance = {"considerados": contados,
                   "nao_cobertos": [regras.filtro("grande_regiao", canonico["grande_regiao"]), *nao_cobertos]}
        return prefixo + "todas_as_regioes", regras.somar_suporte(pacote.resumo["suporte"], prefixo), alcance
    alcance = {"considerados": [*contados, canonico["grande_regiao"]], "nao_cobertos": nao_cobertos}
    return celula, pacote.resumo["suporte"].get(celula, {"episodios": 0, "positivos": 0}), alcance


def avaliar(contexto: dict, base: Path | str | None = None) -> dict:
    pacote = carregar(base)
    canonico = preparar(contexto)
    registros, pesos, publico = _expandir_publico(canonico, pacote.resumo["composicao_publico"])
    linhas = entrada(pacote, registros)
    estimativa = float(pesos @ pacote.modelo.predict_proba(linhas)[:, 1])
    reamostradas = [(float(pesos @ m.predict_proba(linhas)[:, 1]), taxa) for m, taxa in pacote.reamostras]
    diferencas = [valor - taxa for valor, taxa in reamostradas]
    celula, suporte, alcance = suporte_do_contexto(pacote, canonico)
    resultado = regras.comparar(estimativa, pacote.referencia, regras.intervalo(diferencas), suporte, alcance)
    # Faixa da propria estimativa (percentis das reamostras), distinta da faixa da diferenca para o habitual.
    resultado["intervalo_estimativa"] = list(regras.intervalo([valor for valor, _ in reamostradas]))
    decisao = regras.decidir(resultado["faixa"], pacote.papel, [resultado["nota"]] if resultado["nota"] else [],
                                  alcance["nao_cobertos"])
    grupo = grupo_do_dia(canonico["dia_semana"], pacote.representacao)
    leitura = {**regras.leitura_do_cenario(resultado, MEDIDA, "na referência histórica", pacote.papel == "exploratorio"),
               "base": BASE_DA_ESTIMATIVA, "recorte": regras.recorte(canonico["grande_regiao"], canonico["area"])}
    return {
        "occasion_score": estimativa, "classification": resultado["faixa"],
        "top_factors": _fatores(pacote, registros, pesos, canonico),
        "recommendation_category": decisao["categoria"], "recommendation_reason": decisao["justificativa"],
        "leitura": leitura, "proximo_passo": regras.PROXIMOS_PASSOS[decisao["categoria"]],
        "model_metadata": _metadados(pacote, decisao),
        "comparacao": {**resultado, "referencia_nome": "média geral das ocasiões com pizza em 2008–2009",
                       "celula_de_suporte": celula},
        "contexto": {"dia": {"dia_semana": canonico["dia_semana"], "representacao": pacote.representacao, "grupo": grupo,
                             "como_o_modelo_usa": REPRESENTACOES_DIA[pacote.representacao]},
                     "periodo": regras.PERIODOS[canonico["periodo"]], "publico": publico}}
