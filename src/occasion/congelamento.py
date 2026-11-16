"""Congelamento: manifesto operacional, verificacao de integridade e predicao minima dos artefatos congelados.

O manifesto liga protocolos, entradas, codigo, ambiente, artefatos e registros por SHA-256. Nada aqui ajusta, seleciona ou
recalibra modelos: so grava, confere e aplica o que ja foi ajustado. Os estimadores mantem os formatos ja adotados (JSON para
B0/B1/B2, joblib para L/G), sem serializacao propria, e o hash de cada joblib vem do manifesto, nao do `.meta` vizinho.
"""

import hashlib
import json
import numbers
import platform
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import joblib
import numpy as np
import scipy
import sklearn

from occasion import modelagem as m

VERSAO_MANIFESTO = "1"
_PADROES_CODIGO = ("src/occasion/*.py", "configs/*")
_ARQUIVOS_RAIZ = ("pyproject.toml", "uv.lock")
_ARQUIVOS_EXECUCAO = ("registro.json", "resultados.json", "previsoes_fora_da_dobra.npz",
                      "previsoes_modelos_finais_2008.npz", "dobras_externas.npz")
_COBERTURAS = ("0.30", "0.20", "0.40")
_CONTROLES = ("episodios", "positivos_guarana", "episodios_com_refrigerante")
_SHA256 = re.compile(r"[0-9a-f]{64}")
LIMITACOES_GERAIS = (
    "validacao repetida e agrupada por UPA em 2008; o desempenho do candidato escolhido e otimista porque a mesma "
    "validacao seleciona e avalia",
    "intervalos por bootstrap de UPAs sao condicionais aos ajustes e ignoram o estrato; 101 positivos em 70 UPAs dao pouco poder",
    "ganho nao demonstrado nao comprova equivalencia entre modelos")
LIMITACOES_CONSTANTE = (
    "preditor constante por construcao: nao ordena contextos e nao fornece fatores de contexto",
    "AUC, AP e inclinacao agregadas refletem so diferencas de taxa entre dobras e nao sao desempenho do preditor",
    "limiares e classes fora da dobra sao diagnostico de diferencas entre dobras, nao prioridade")
LEITURAS_PRECEDENCIA = ("modelos.*.leitura_preditor", "modelos.*.nao_interpretavel", "aplicacao_do_limiar.preditor_constante")


class CongelamentoInvalido(RuntimeError):
    """Manifesto, codigo, ambiente, artefato ou registro diferente do que foi congelado."""


# ---------- hashes e ambiente ----------

def hashes_do_codigo(repo: Path) -> dict[str, str]:
    """SHA-256 de src/occasion/*.py, configs/*, pyproject.toml e uv.lock, por caminho relativo ao repositorio."""
    repo = Path(repo)
    arquivos = [p for padrao in _PADROES_CODIGO for p in repo.glob(padrao) if p.is_file()]
    arquivos += [repo / nome for nome in _ARQUIVOS_RAIZ if (repo / nome).is_file()]
    return {p.relative_to(repo).as_posix(): m.sha256_arquivo(p) for p in sorted(arquivos)}


def versoes_do_ambiente() -> dict[str, str]:
    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "numpy": np.__version__,
            "scipy": scipy.__version__, "joblib": joblib.__version__}


def _ler_json(caminho: Path) -> dict:
    try:
        return json.loads(Path(caminho).read_text(encoding="utf-8"))
    except (OSError, ValueError) as erro:
        raise CongelamentoInvalido(f"{Path(caminho).name}: ausente ou ilegivel ({erro})") from erro


def _caminho_relativo(base: Path, relativo: str) -> Path:
    """Caminho do manifesto sempre relativo e sem `..`: nada aponta para fora da base."""
    posix = PurePosixPath(str(relativo))
    if not relativo or posix.is_absolute() or ".." in posix.parts or "\\" in relativo or re.match(r"^[A-Za-z]:", relativo):
        raise CongelamentoInvalido(f"caminho invalido no manifesto: {relativo!r}")
    return Path(base).joinpath(*posix.parts)


def _hash_relativo(base: Path, relativo: str) -> str:
    caminho = _caminho_relativo(base, relativo)
    if not caminho.is_file():
        raise CongelamentoInvalido(f"arquivo ausente: {relativo}")
    return m.sha256_arquivo(caminho)


# ---------- geracao ----------

def _contrato(nome: str, espec_g: dict, colunas_logistica: list[str]) -> dict:
    if nome == "B0":
        return {"colunas": [], "observacao": "nao usa entradas: prediz p0 para qualquer contexto"}
    if nome == "B1":
        niveis = {c: espec_g["categoricas"][c] for c in ("dia_semana", "faixa_hora")}
        return {"celula": ["dia_semana", "faixa_hora"], "formato_celula": "{dia_semana}|{faixa_hora}", "niveis": niveis,
                "celula_ausente": "p0 gravado no artefato"}
    if nome == "L":
        return {"colunas": list(colunas_logistica), "ausentes": "erro", "ordem": "a da lista"}
    if nome == "G":
        return {"colunas": list(espec_g["colunas"]), "categoricas": espec_g["categoricas"], "ausentes": "erro"}
    raise CongelamentoInvalido(f"comparador sem contrato de entrada: {nome}")


def _metricas_validacao(mod: dict, constante: bool) -> dict:
    cal, dentro = mod["calibracao_s"], mod["dentro_da_dobra"]
    out = {"log_loss_media": mod["log_loss_media"], "brier_s": mod["brier_s"], "o_e": cal["o_e"],
           "intercepto_no_todo": cal["intercepto_no_todo"],
           "dentro_da_dobra": {"log_loss_mediana": dentro["log_loss"]["mediana"], "brier_mediana": dentro["brier"]["mediana"]}}
    if not constante:
        out.update({"auc_s": mod["auc_s"], "ap_s": mod["ap_s"], "inclinacao": cal["inclinacao"]})
        out["dentro_da_dobra"]["auc_mediana"] = dentro["auc"]["mediana"]
    return out


def _bloco_modelo(nome: str, res: dict, espec_g: dict, colunas_logistica: list[str]) -> dict:
    mod, lims = res["modelos"][nome], res["limiares"][nome]
    leitura = mod["leitura_preditor"]
    constante = bool(leitura["constante_por_construcao"])
    limiares = {q: lims[q]["t"] for q in _COBERTURAS}
    return {"hiperparametro": res["finais"][nome]["hiperparametro"], "entradas": _contrato(nome, espec_g, colunas_logistica),
            "limiares_fora_da_dobra": limiares, "t30_aplicavel": None if constante else limiares["0.30"],
            "ordena_contextos": not constante, "fatores_de_contexto": [] if constante else None,
            "leitura_preditor": leitura, "nao_interpretavel": mod.get("nao_interpretavel"),
            "metricas_validacao": _metricas_validacao(mod, constante),
            "limitacoes": [*LIMITACOES_GERAIS, *(LIMITACOES_CONSTANTE if constante else ())]}


def _artefatos(base: Path, execucao: str, res: dict) -> dict:
    out = {}
    for nome, final in res["finais"].items():
        if "artefato" not in final:
            continue
        rel, meta = f"{execucao}/artefatos/{final['artefato']}", f"{execucao}/artefatos/{nome}.meta.json"
        sha = _hash_relativo(base, rel)
        if sha != final.get("sha256"):
            raise CongelamentoInvalido(f"artefato {nome}: hash do arquivo difere do registrado nos resultados da execucao")
        out[nome] = {"arquivo": rel, "sha256": sha, "meta": meta, "meta_sha256": _hash_relativo(base, meta),
                     "formato": "json" if nome in m.ARTEFATOS_JSON else "joblib"}
    return out


def _leituras(base: Path, execucao: str, res: dict, cfg: dict, anteriores: dict) -> dict:
    """Vincula `leituras_revisadas.json` as fontes e hashes de origem e declara qual fonte tem precedencia."""
    dados = _ler_json(_caminho_relativo(base, cfg["arquivo"]))
    bloco = dados.get(cfg["chave"])
    if bloco is None:
        raise CongelamentoInvalido(f"leituras: chave {cfg['chave']!r} ausente em {cfg['arquivo']}")
    fontes = {}
    for nome, sha in bloco["fontes_sha256"].items():
        arquivo = f"{cfg['pasta_fontes']}/{nome}"
        fontes[nome] = {"arquivo": arquivo, "sha256": sha, "confere": _hash_relativo(base, arquivo) == sha}
    if not all(f["confere"] for f in fontes.values()):
        raise CongelamentoInvalido("leituras: fonte com hash diferente do registrado em leituras_revisadas")
    equivale = {nome: res["modelos"][nome]["leitura_preditor"] == leitura for nome, leitura in bloco["leitura_preditor"].items()}
    if not all(equivale.values()):
        raise CongelamentoInvalido(f"leituras revisadas divergem do resultado oficial: {sorted(k for k, v in equivale.items() if not v)}")
    superados = [{"execucao": nome, "campos": v["campos_superados"]} for nome, v in anteriores.items() if v.get("campos_superados")]
    return {"arquivo": cfg["arquivo"], "sha256": _hash_relativo(base, cfg["arquivo"]), "chave": cfg["chave"], "fontes": fontes,
            "equivale_ao_resultado_oficial": equivale,
            "precedencia": {"fonte_oficial": f"{execucao}/resultados.json", "campos": list(LEITURAS_PRECEDENCIA),
                            "superados": superados,
                            "regra": ("para estes campos vale o resultado da execucao de origem; os mesmos campos das "
                                      "execucoes superadas foram gravados antes da correcao da semantica de "
                                      "leitura_preditor e nao sao o resultado oficial; previsoes, log-loss, "
                                      "hiperparametros e limiares nao mudaram")}}


def manifesto_da_execucao(*, repo: Path, base: Path, execucao: str, protocolos: Mapping[str, str], protocolo_vigente: str,
                          leituras_revisadas: dict | None = None, execucoes_anteriores: dict | None = None,
                          registros_extra: Mapping[str, str] | None = None, controles: dict | None = None) -> dict:
    """Monta o manifesto a partir de uma execucao ja gravada (`base/execucao/`). Recusa se o codigo atual difere do da
    execucao, se o codigo mudou durante ela, se o escolhido difere do protocolo ou se algum hash nao confere."""
    repo, base = Path(repo), Path(base)
    pasta = base / execucao
    reg, res = _ler_json(pasta / "registro.json"), _ler_json(pasta / "resultados.json")
    blocos = {v: {"arquivo": rel, "sha256": m.sha256_arquivo(repo / rel)} for v, rel in protocolos.items()}
    prot = m.carregar_protocolo(repo / protocolos[protocolo_vigente])
    if prot.get("versao") != protocolo_vigente or reg.get("protocolo_sha256") != blocos[protocolo_vigente]["sha256"]:
        raise CongelamentoInvalido(f"a execucao nao usou o protocolo {protocolo_vigente} registrado")
    codigo, usado = hashes_do_codigo(repo), reg.get("codigo") or {}
    if usado.get("inalterado") is not True:
        raise CongelamentoInvalido("o codigo mudou durante a execucao ou o hash do codigo nao foi registrado")
    if usado.get("antes") != codigo:
        mudou = sorted(k for k in codigo.keys() | usado.get("antes", {}).keys() if codigo.get(k) != usado.get("antes", {}).get(k))
        raise CongelamentoInvalido(f"codigo atual difere do codigo da execucao: {mudou}")
    if prot["classes"]["comparacao_numerica"]["casas_decimais"] != m.CASAS_EMPATE:
        raise CongelamentoInvalido("precisao numerica do protocolo difere da do codigo")
    temporal, escolhido = prot["avaliacao_temporal"], res["escolha_provisoria"]["escolhido"]
    if escolhido != temporal["selecionado_em_2008"]:
        raise CongelamentoInvalido(f"selecionado na execucao ({escolhido}) difere do protocolo ({temporal['selecionado_em_2008']})")
    comparadores = list(temporal["comparadores_previamente_definidos"])
    artefatos = _artefatos(base, execucao, res)
    if not set(comparadores) <= set(artefatos):
        raise CongelamentoInvalido(f"artefato ausente para comparadores: {sorted(set(comparadores) - set(artefatos))}")
    ref = prot["entradas"]["taxa_referencia"]
    colunas = _ler_json(repo / "configs" / "variaveis.json")["colunas_logistica"]
    anteriores = {}
    for nome, v in (execucoes_anteriores or {}).items():
        anteriores[nome] = {**{k: x for k, x in v.items() if k != "arquivos"},
                            "arquivos_sha256": {rel: _hash_relativo(base, rel) for rel in v.get("arquivos", [])}}
    registros = {f"{execucao}/{a}": _hash_relativo(base, f"{execucao}/{a}") for a in _ARQUIVOS_EXECUCAO}
    registros.update({rel: _hash_relativo(base, rel) for rel in (registros_extra or {}).values()})
    p0 = _ler_json(_caminho_relativo(base, artefatos["B0"]["arquivo"]))["p0"]
    manifesto = {
        "versao_manifesto": VERSAO_MANIFESTO, "gerado_em_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocolo_vigente": protocolo_vigente, "protocolos": blocos, "entradas_sha256": reg.get("entradas_sha256", {}),
        "codigo": {"raiz": "repositorio", "sha256": codigo},
        "execucao_de_origem": {"pasta": execucao, "codigo_inalterado": True, "codigo_igual_ao_do_manifesto": True,
                               "protocolo_sha256": reg["protocolo_sha256"]},
        "execucoes_anteriores": anteriores,
        "ambiente": {**versoes_do_ambiente(), "uv_lock_sha256": codigo.get("uv.lock"), "threads": reg.get("threads")},
        "artefatos": artefatos,
        "selecao": {"selecionado": escolhido, "comparadores": comparadores, "m_estrela_2008": res["m_estrela"],
                    "regra": prot["avaliacao"]["regra_escolha"], "papel_dos_comparadores": temporal["papel"],
                    "motivo": "parcimonia: ganho nao demonstrado, que nao comprova equivalencia"},
        "constantes": {"p0": p0, "b": ref["positivos"] / ref["episodios"], "b_fracao": f"{ref['positivos']}/{ref['episodios']}",
                       "casas_comparacao": m.CASAS_EMPATE, "ordem_classes": prot["classes"]["ordem"],
                       "coberturas": prot["limiar"]["coberturas"], "cobertura_operacional": prot["limiar"]["operacional"]},
        "modelos": {nome: _bloco_modelo(nome, res, prot["candidatos"]["G"], colunas) for nome in comparadores},
        "registros": registros,
        "avaliacao_temporal": {"controles": {k: (controles or temporal["controles_registrados"])[k] for k in _CONTROLES},
                               "autorizacao_formato": temporal["autorizacao"]["formato"], "comparadores": comparadores,
                               "selecionado": escolhido, "papel": temporal["papel"], "sem_reajuste": temporal["sem_reajuste"],
                               "rotulo_obrigatorio": temporal["rotulo_obrigatorio"], "relato": temporal["relato"],
                               "bootstrap": prot["avaliacao"]["bootstrap"]}}
    if leituras_revisadas is not None:
        manifesto["leituras_corrigidas"] = _leituras(base, execucao, res, leituras_revisadas, anteriores)
    return manifesto


def gravar_manifesto(manifesto: dict, destino: Path) -> str:
    """Grava o manifesto (JSON ordenado) e devolve o SHA-256 do arquivo, que e informado a parte na verificacao."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(manifesto, sort_keys=True, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return m.sha256_arquivo(destino)


# ---------- verificacao ----------

def _conferir_arquivos(base: Path, hashes: Mapping[str, str], rotulo: str, problemas: list[str]) -> None:
    for rel, esperado in hashes.items():
        try:
            if _hash_relativo(base, rel) != esperado:
                problemas.append(f"{rotulo}: {rel} alterado")
        except CongelamentoInvalido as erro:
            problemas.append(f"{rotulo}: {erro}")


def _conferir_ambiente(mf: dict, problemas: list[str]) -> None:
    atual, esperado = versoes_do_ambiente(), mf["ambiente"]
    for chave in ("scikit-learn", "numpy", "scipy", "joblib"):
        if atual[chave] != esperado.get(chave):
            problemas.append(f"ambiente: {chave} {esperado.get(chave)} no manifesto, {atual[chave]} instalado")
    if ".".join(atual["python"].split(".")[:2]) != ".".join(str(esperado.get("python")).split(".")[:2]):
        problemas.append(f"ambiente: python {esperado.get('python')} no manifesto, {atual['python']} instalado")


def verificar_manifesto(caminho: Path, sha256_esperado: str, repo: Path, base: Path | None = None,
                        raiz_entradas: Path | None = None) -> dict:
    """Confere o manifesto contra o hash informado a parte e tudo o que ele lista; devolve o manifesto ou levanta
    CongelamentoInvalido com todas as divergencias encontradas."""
    caminho = Path(caminho)
    if not isinstance(sha256_esperado, str) or not _SHA256.fullmatch(sha256_esperado):
        raise CongelamentoInvalido("hash do manifesto informado nao e um SHA-256 valido")
    try:
        bruto = caminho.read_bytes()
    except OSError as erro:
        raise CongelamentoInvalido(f"manifesto ilegivel: {erro}") from erro
    obtido = hashlib.sha256(bruto).hexdigest()
    if obtido != sha256_esperado:
        raise CongelamentoInvalido(f"manifesto: SHA-256 {obtido[:12]} diferente do informado {sha256_esperado[:12]}")
    try:
        mf = json.loads(bruto)
    except ValueError as erro:
        raise CongelamentoInvalido(f"manifesto ilegivel: {erro}") from erro
    if mf.get("versao_manifesto") != VERSAO_MANIFESTO:
        raise CongelamentoInvalido(f"manifesto: versao {mf.get('versao_manifesto')!r} nao suportada")
    repo, base = Path(repo), Path(base) if base else caminho.parent
    problemas: list[str] = []
    for versao, p in mf["protocolos"].items():
        try:
            if m.sha256_arquivo(repo / p["arquivo"]) != p["sha256"]:
                problemas.append(f"protocolo {versao}: {p['arquivo']} alterado")
        except OSError:
            problemas.append(f"protocolo {versao}: {p['arquivo']} ausente")
    atual, congelado = hashes_do_codigo(repo), mf["codigo"]["sha256"]
    for rel in sorted(congelado.keys() | atual.keys()):
        if rel not in atual:
            problemas.append(f"codigo: {rel} ausente")
        elif rel not in congelado:
            problemas.append(f"codigo: {rel} nao consta no manifesto")
        elif atual[rel] != congelado[rel]:
            problemas.append(f"codigo: {rel} alterado")
    _conferir_ambiente(mf, problemas)
    for nome, a in mf["artefatos"].items():
        _conferir_arquivos(base, {a["arquivo"]: a["sha256"], a["meta"]: a["meta_sha256"]}, f"artefato {nome}", problemas)
    _conferir_arquivos(base, mf["registros"], "registro", problemas)
    for nome, v in mf.get("execucoes_anteriores", {}).items():
        _conferir_arquivos(base, v.get("arquivos_sha256", {}), f"execucao anterior {nome}", problemas)
    if "leituras_corrigidas" in mf:
        lc = mf["leituras_corrigidas"]
        _conferir_arquivos(base, {lc["arquivo"]: lc["sha256"], **{f["arquivo"]: f["sha256"] for f in lc["fontes"].values()}},
                           "leituras", problemas)
    if raiz_entradas is not None:
        for nome, esperado in mf.get("entradas_sha256", {}).items():
            arquivo = Path(raiz_entradas) / nome
            if not arquivo.is_file() or m.sha256_arquivo(arquivo) != esperado:
                problemas.append(f"entrada {nome}: ausente ou diferente do registrado")
    if problemas:
        raise CongelamentoInvalido("manifesto nao confere: " + "; ".join(problemas))
    return mf


# ---------- carregamento e predicao minima ----------

def carregar_modelo_congelado(manifesto: dict, nome: str, base: Path):
    """Carrega o artefato exigindo o SHA-256 do manifesto (nunca so o do `.meta` vizinho)."""
    a = manifesto.get("artefatos", {}).get(nome)
    if a is None:
        raise CongelamentoInvalido(f"artefato {nome} ausente do manifesto")
    sha = a.get("sha256")
    if not isinstance(sha, str) or not _SHA256.fullmatch(sha):
        raise CongelamentoInvalido(f"artefato {nome}: sha256 ausente no manifesto")
    caminho = _caminho_relativo(base, a["arquivo"])
    try:
        return m.carregar_artefato(caminho, nome, sha256_esperado=sha)
    except (m.ArtefatoInvalido, OSError) as erro:
        raise CongelamentoInvalido(f"artefato {nome}: {erro}") from erro


@dataclass(frozen=True)
class Previsao:
    """Saida minima de um modelo congelado. `fatores_de_contexto` e [] quando o preditor nao ordena contextos e None
    quando o modelo ordena mas esta funcao nao explica (explicabilidade fica fora deste pacote)."""
    modelo: str
    score: np.ndarray
    classe: np.ndarray
    ordena_contextos: bool
    fatores_de_contexto: list | None
    aviso: str | None


def _linhas(entradas) -> list[Mapping]:
    if isinstance(entradas, (Mapping, str, bytes)) or not isinstance(entradas, Sequence):
        raise ValueError("entradas devem ser uma lista de contextos (dicionarios)")
    linhas = list(entradas)
    if not linhas:
        raise ValueError("entradas vazias")
    if not all(isinstance(linha, Mapping) for linha in linhas):
        raise ValueError("cada contexto deve ser um dicionario")
    return linhas


def _nivel(linha: Mapping, chave: str, niveis: list):
    valor = linha.get(chave)
    valor = valor.item() if isinstance(valor, np.generic) else valor
    if isinstance(valor, bool) or not isinstance(valor, (str, int)) or valor not in niveis:
        raise ValueError(f"{chave} = {valor!r} fora dos niveis validos {niveis}")
    return valor


def _matriz(linhas: list[Mapping], colunas: list[str]) -> np.ndarray:
    x = np.empty((len(linhas), len(colunas)))
    for i, linha in enumerate(linhas):
        for j, coluna in enumerate(colunas):
            if coluna not in linha:
                raise ValueError(f"linha {i}: coluna {coluna} ausente")
            valor = linha[coluna]
            if isinstance(valor, (bool, np.bool_)) or not isinstance(valor, numbers.Real) or not np.isfinite(valor):
                raise ValueError(f"linha {i}: {coluna} = {valor!r} nao e numero finito")
            x[i, j] = float(valor)
    return x


def _escore(nome: str, contrato: dict, modelo, entradas) -> np.ndarray:
    if nome == "B0":
        n = entradas if isinstance(entradas, int) and not isinstance(entradas, bool) else len(entradas)
        if n < 1:
            raise ValueError("B0: informe ao menos um contexto")
        return np.full(n, float(modelo["p0"]))
    linhas = _linhas(entradas)
    if nome == "B1":
        celulas = ["|".join(str(_nivel(linha, c, contrato["niveis"][c])) for c in contrato["celula"]) for linha in linhas]
        return m.prever_celulas(modelo, np.array(celulas))
    if nome == "L":
        return modelo.predict_proba(_matriz(linhas, contrato["colunas"]))[:, 1]
    if nome == "G":
        x, _ = m.codificar_nativas(linhas, {"colunas": contrato["colunas"], "categoricas": contrato["categoricas"]})
        return modelo.predict_proba(x)[:, 1]
    raise CongelamentoInvalido(f"modelo sem contrato de predicao: {nome}")


def prever_congelado(manifesto: dict, nome: str, base: Path, entradas, modelo=None) -> Previsao:
    """Aplica o artefato congelado, sem reajuste. Entrada fora do contrato (coluna ausente, nivel desconhecido, valor nao
    finito) e erro, nunca imputacao. `classe` usa t30 do manifesto (None para preditor constante) e o corte b, na
    mesma precisao do limiar."""
    bloco = manifesto.get("modelos", {}).get(nome)
    if bloco is None:
        raise CongelamentoInvalido(f"modelo {nome} ausente do manifesto")
    modelo = modelo if modelo is not None else carregar_modelo_congelado(manifesto, nome, base)
    score = _escore(nome, bloco["entradas"], modelo, entradas)
    classe = m.classes(score, bloco["t30_aplicavel"], manifesto["constantes"]["b"])
    aviso = None if bloco["ordena_contextos"] else (
        f"{nome} nao fornece ordenacao contextual: o score e a taxa-base e nao distingue contextos")
    fatores = None if bloco["fatores_de_contexto"] is None else list(bloco["fatores_de_contexto"])
    return Previsao(nome, score, classe, bool(bloco["ordena_contextos"]), fatores, aviso)
