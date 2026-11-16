"""Comando unico da avaliacao temporal (2017-2018): PREPARADO, nao executado com dados reais.

Uso: python -m occasion.abrir_avaliacao --manifesto <json> --manifesto-sha256 <hash> --autorizacao ABRIR-2017:<hash>
         --destino <pasta> [--trabalho <pasta>] [--selado <pasta>] [--bruta <pasta>] [--registro <pasta>]
         [--retomada RETOMAR-2017:<hash>:<tentativa> --motivo <texto>]

O alvo nao e um arquivo de entrada: e construido por codigo (`occasion.alvo_2017`) a partir do consumo registrado no
manifesto de fontes, depois do marco de exposicao. Antes do marco so se leem o dicionario oficial e os contextos.

O lancador `occasion.abrir_avaliacao` inicia este modulo em processo novo, com pasta de bytecode exclusiva; iniciado de
outra forma, o comando recusa. A origem e o hash dos modulos realmente carregados sao conferidos contra o manifesto
(`occasion.origem_codigo`) apos a verificacao do manifesto, antes da exposicao e antes de gravar o resultado.

Este modulo e o unico que le o conjunto reservado e nenhum modulo de desenvolvimento o importa. Ele nao ajusta, nao
seleciona e nao recalibra nada: confere o manifesto de congelamento (protocolos, codigo, ambiente, artefatos), aplica os
artefatos congelados aos contextos e so depois acessa alvos e pesos, para conferir os controles e calcular as metricas.

Controle de acesso: `chmod 0444` na pasta selada protege contra ESCRITA, nao contra leitura. O controle do fluxo e este
comando: autorizacao textual ligada ao hash de um manifesto especifico, pastas separadas e o registro de aberturas.

Registro de aberturas: um unico local, FIXADO NO MANIFESTO como caminho absoluto, identificado pelo hash do manifesto e
independente do destino e de onde o manifesto e os artefatos estejam copiados. CASE_AMBEV_REGISTRO_ROOT e --registro so
servem para conferir: se informam outra raiz, o comando recusa. As entradas de 2017 (identificadores, nativas, matriz e
pesos) sao conferidas pelos SHA-256 do manifesto, os pesos so depois do marco. Cada tentativa cria no registro, em
modo exclusivo, a sua reserva ANTES da primeira leitura de alvo ou peso; o historico so recebe acrescimos. Enquanto nao
ha exposicao, uma falha e de preparacao e pode ser repetida. Depois dela nada volta ao estado "nao aberto": so uma
retomada tecnica, com autorizacao numerada e motivo, sobre o mesmo manifesto integro. Divergencia de controle encerra a
tentativa e nao admite retomada.
"""

import argparse
import csv
import json
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from occasion import congelamento as cg
from occasion import modelagem as m
from occasion.abrir_avaliacao import ProcessoInvalido, conferir_processo
from occasion.alvo_2017 import FontesDoAlvo, alvo_das_fontes, preparar_fontes
from occasion.episodios_2017 import estado_do_selo
from occasion.fontes import AmbienteInvalido, carregar_manifesto, raiz_bruta, raiz_saidas, raiz_selado
from occasion.origem_codigo import conferir_origem_carregada

REPO = Path(__file__).resolve().parents[2]
ARQ_ESTADO, ARQ_RESULTADO = "estado_abertura.json", "resultado_avaliacao_temporal.json"
VAR_RAIZ_REGISTRO = "CASE_AMBEV_REGISTRO_ROOT"  # e este registro, nao o destino nem a pasta do manifesto, que decide a abertura
PREPARACAO_FALHOU, EXPOSICAO_INICIADA, APOS_EXPOSICAO, CONCLUIDA = (
    "preparacao_falhou", "exposicao_iniciada", "interrompida_apos_exposicao", "concluida")
ROTULO_RETOMADA = "retomada tecnica apos exposicao ao alvo: mesmo manifesto, mesmos modelos e mesmas regras"
ARQ_IDS, ARQ_NATIVAS, ARQ_MATRIZ, ARQ_PESOS = ("identificadores_2017.csv", "variaveis_nativas_2017.csv",
                                                "matriz_logistica_2017.csv", "ids_pesos_2017.csv")
NATIVAS_NUMERICAS = {"dia_semana", "sexta_a_domingo", "idade", "log_renda_pc", "moradores_uc", "presenca_crianca",
                     "presenca_adolescente", "presenca_idoso"}
COBERTURAS = ("0.30", "0.20", "0.40")
DIRECAO = "modelo menos B0; negativo favorece o modelo"
MOTIVO_SEM_ORDENACAO = ("preditor constante por construcao: AUC, AP e inclinacao de calibracao nao se aplicam "
                        "(o AUC de um escore constante com as duas classes e 0,5 por definicao)")


class AvaliacaoInterrompida(RuntimeError):
    """Um portao falhou; nada depois dele foi calculado."""


class AutorizacaoAusente(AvaliacaoInterrompida):
    """A autorizacao textual exata, ligada ao hash do manifesto, nao foi informada."""


class PastasInvalidas(AvaliacaoInterrompida):
    """Pastas de trabalho, selada, destino e repositorio se sobrepoem ou usam link simbolico."""


class AberturaJaRealizada(AvaliacaoInterrompida):
    """O registro de aberturas ja marca exposicao ao alvo para este manifesto."""


class RetomadaInvalida(AvaliacaoInterrompida):
    """Retomada sem exposicao anterior, sem a autorizacao numerada ou o motivo, ou vedada pelo historico."""


class RegistroInvalido(AvaliacaoInterrompida):
    """Historico de aberturas ilegivel: nada e liberado ate que ele seja examinado."""


class ContratoInvalido(AvaliacaoInterrompida):
    """Arquivo, coluna, alinhamento ou valor fora do contrato de entrada."""


class ControleFalhou(AvaliacaoInterrompida):
    """Um controle registrado (episodios, positivos, episodios com refrigerante) nao confere."""

    def __init__(self, divergentes: dict):
        super().__init__(f"controles divergentes, nenhuma metrica foi calculada: {divergentes}")
        self.divergentes = divergentes


@dataclass(frozen=True)
class Contextos:
    """O que os modelos recebem, lido ANTES da exposicao: identificador, UPA, nativas e as 28 colunas da logistica."""
    ids: tuple[str, ...]
    grupos: np.ndarray
    nativas: tuple[dict, ...]
    matriz: tuple[dict, ...]


@dataclass(frozen=True)
class Reservado:
    """Alvos e pesos por episodio, lidos so DEPOIS do marco de exposicao; a ordem e livre, o alinhamento e por `ids`."""
    ids: tuple[str, ...]
    pesos: np.ndarray
    y: np.ndarray
    y_refrigerante: np.ndarray


@dataclass(frozen=True)
class ConjuntoAvaliacao:
    """Contextos e reservado alinhados por posicao. `y` e `y_refrigerante` so servem aos controles e as metricas."""
    ids: tuple[str, ...]
    grupos: np.ndarray
    pesos: np.ndarray
    nativas: tuple[dict, ...]
    matriz: tuple[dict, ...]
    y: np.ndarray
    y_refrigerante: np.ndarray


# ---------- autorizacao, pastas e registro de aberturas ----------

def autorizacao_exigida(sha256_manifesto: str) -> str:
    return f"ABRIR-2017:{sha256_manifesto}"


def autorizacao_de_retomada(sha256_manifesto: str, tentativa: int) -> str:
    return f"RETOMAR-2017:{sha256_manifesto}:{tentativa}"


def _agora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sobrepoe(a: Path, b: Path) -> bool:
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def validar_pastas(trabalho: Path, selado: Path, destino: Path, repo: Path) -> None:
    """Pastas distintas, sem aninhamento entre si, fora do repositorio e sem link simbolico."""
    pastas = {"trabalho": Path(trabalho), "selado": Path(selado), "destino": Path(destino), "repositorio": Path(repo)}
    for nome, pasta in pastas.items():
        if pasta.is_symlink():
            raise PastasInvalidas(f"{nome}: link simbolico nao e aceito ({pasta.name})")
    reais = {nome: pasta.resolve() for nome, pasta in pastas.items()}
    for a, b in (("trabalho", "selado"), ("destino", "selado"), ("destino", "trabalho"), ("selado", "repositorio"),
                 ("trabalho", "repositorio"), ("destino", "repositorio")):
        if _sobrepoe(reais[a], reais[b]):
            raise PastasInvalidas(f"{a} e {b} se sobrepoem; use pastas separadas e fora do repositorio")


def _gravar_json(arquivo: Path, conteudo: dict) -> None:
    """Grava por arquivo parcial e troca de nome: uma queda nao deixa JSON pela metade."""
    parcial = arquivo.with_name(arquivo.name + ".parcial")
    parcial.write_text(json.dumps(conteudo, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(parcial, arquivo)


def raiz_registro(env: Mapping[str, str] | None = None) -> Path | None:
    """Raiz do registro informada por CASE_AMBEV_REGISTRO_ROOT, se houver. Quem decide e a raiz fixada no manifesto;
    esta so e conferida contra ela."""
    valor = (os.environ if env is None else env).get(VAR_RAIZ_REGISTRO)
    return Path(valor) if valor else None


@dataclass(frozen=True)
class Registro:
    """Registro de aberturas de UM manifesto: uma reserva exclusiva por tentativa e um historico so de acrescimos."""
    pasta: Path
    sha256_manifesto: str

    @property
    def marco(self) -> Path:
        return self.pasta / f"exposicao_{self.sha256_manifesto}.json"

    @property
    def historico(self) -> Path:
        return self.pasta / f"historico_{self.sha256_manifesto}.jsonl"

    def reserva(self, tentativa: int) -> Path:
        """Arquivo que a tentativa cria em modo exclusivo: o marco de exposicao na primeira, um por retomada depois."""
        return self.marco if tentativa == 1 else self.pasta / f"retomada_{self.sha256_manifesto}_{tentativa}.json"

    def tentativas(self, eventos: list[dict]) -> int:
        """Tentativas que ja reservaram a exposicao, pelo historico ou pelos arquivos de reserva (o que for maior)."""
        reservas = int(self.marco.exists()) + len(list(self.pasta.glob(f"retomada_{self.sha256_manifesto}_*.json")))
        return max(reservas, sum(e.get("status") == EXPOSICAO_INICIADA for e in eventos))

    def eventos(self) -> list[dict]:
        if not self.historico.is_file():
            return []
        try:
            return [json.loads(linha) for linha in self.historico.read_text(encoding="utf-8").splitlines() if linha.strip()]
        except ValueError as erro:
            raise RegistroInvalido(f"{self.historico.name} ilegivel ({erro}); examine o registro antes de qualquer tentativa") from erro

    def anotar(self, evento: dict) -> None:
        self.pasta.mkdir(parents=True, exist_ok=True)
        with open(self.historico, "a", encoding="utf-8") as f:
            f.write(json.dumps({**evento, "momento_utc": _agora()}, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def marcar_exposicao(self, evento: dict) -> None:
        """Cria a reserva da tentativa em modo exclusivo e so entao anota a exposicao: duas execucoes com a mesma
        autorizacao nunca passam as duas daqui, e portanto nunca leem o reservado ao mesmo tempo."""
        self.pasta.mkdir(parents=True, exist_ok=True)
        tentativa = evento["tentativa"]
        try:
            with open(self.reserva(tentativa), "x", encoding="utf-8") as f:
                f.write(json.dumps({**evento, "momento_utc": _agora()}, ensure_ascii=False) + "\n")
        except FileExistsError as erro:
            if tentativa == 1:
                raise AberturaJaRealizada("outra execucao ja marcou a exposicao ao alvo para este manifesto") from erro
            raise RetomadaInvalida(f"a tentativa {tentativa} ja foi reservada por outra execucao") from erro
        self.anotar(evento)


def conferir_abertura(registro: Registro, retomada: str | None, motivo: str | None) -> int:
    """Devolve o numero da tentativa (1 sem exposicao anterior). Com exposicao registrada, so uma retomada tecnica valida."""
    eventos = registro.eventos()
    feitas = registro.tentativas(eventos)
    if not feitas:
        if retomada is not None:
            raise RetomadaInvalida("nao ha exposicao registrada para retomar; use a abertura normal")
        return 1
    if retomada is None:
        raise AberturaJaRealizada(
            f"o registro {registro.pasta.name} ja marca exposicao ao alvo para este manifesto; trocar o destino ou copiar o "
            "manifesto nao libera nova tentativa. Retomada tecnica exige --retomada e --motivo; nova abertura exige nova "
            "decisao declarada")
    if any(e.get("divergencia_de_controle") for e in eventos):
        raise RetomadaInvalida("divergencia de controle encerrou a tentativa: nao ha retomada; uma nova abertura exige "
                               "nova autorizacao explicita e novo manifesto")
    if any(e.get("status") == CONCLUIDA for e in eventos):
        raise RetomadaInvalida("avaliacao ja concluida para este manifesto: nao ha o que retomar")
    tentativa = feitas + 1
    if retomada != autorizacao_de_retomada(registro.sha256_manifesto, tentativa):
        raise RetomadaInvalida(f"autorizacao de retomada ausente ou diferente: informe exatamente "
                               f"RETOMAR-2017:<sha256 do manifesto>:{tentativa}")
    if not (motivo or "").strip():
        raise RetomadaInvalida("a retomada exige um motivo, que fica gravado no historico")
    return tentativa


def _encerrar(registro: Registro, destino: Path, evento: dict, status: str, erro: BaseException) -> None:
    """Anota a falha no historico e no destino. O marco de exposicao, se existir, nunca e removido."""
    final = {**evento, "status": status, "motivo": f"{type(erro).__name__}: {erro}"}
    if isinstance(erro, ControleFalhou):
        final.update(divergencia_de_controle=True, controles=erro.divergentes)
    registro.anotar(final)
    _gravar_json(destino / ARQ_ESTADO, final)


# ---------- contrato e controles ----------

def conferir_contextos(c: Contextos) -> None:
    n = len(c.ids)
    if n == 0:
        raise ContratoInvalido("conjunto de avaliacao vazio")
    if len(set(c.ids)) != n:
        raise ContratoInvalido("episodio_id duplicado")
    for nome, valores in (("grupos", c.grupos), ("nativas", c.nativas), ("matriz", c.matriz)):
        if len(valores) != n:
            raise ContratoInvalido(f"{nome} tem {len(valores)} linhas; esperado {n}")
    if any(not str(g) for g in c.grupos):
        raise ContratoInvalido("UPA vazia")


def juntar(c: Contextos, r: Reservado) -> ConjuntoAvaliacao:
    """Alinha o reservado aos contextos pelo identificador; os dois conjuntos de episodios devem ser iguais."""
    for nome, valores in (("pesos", r.pesos), ("y", r.y), ("y_refrigerante", r.y_refrigerante)):
        if len(valores) != len(r.ids):
            raise ContratoInvalido(f"{nome} tem {len(valores)} linhas; esperado {len(r.ids)}")
    if len(set(r.ids)) != len(r.ids) or set(r.ids) != set(c.ids):
        raise ContratoInvalido("episodio_id do conjunto reservado difere dos identificadores")
    posicao = {episodio: i for i, episodio in enumerate(r.ids)}
    ordem = np.array([posicao[episodio] for episodio in c.ids])
    return ConjuntoAvaliacao(ids=c.ids, grupos=c.grupos, pesos=np.asarray(r.pesos)[ordem], nativas=c.nativas, matriz=c.matriz,
                             y=np.asarray(r.y)[ordem], y_refrigerante=np.asarray(r.y_refrigerante)[ordem])


def conferir_contrato(c: ConjuntoAvaliacao) -> None:
    if not (np.isin(c.y, (0, 1)).all() and np.isin(c.y_refrigerante, (0, 1)).all()):
        raise ContratoInvalido("os alvos devem ser 0 ou 1")
    if (np.asarray(c.y) > np.asarray(c.y_refrigerante)).any():
        raise ContratoInvalido("episodio com guarana e sem refrigerante: o alvo deve estar dentro de refrigerante")
    pesos = np.asarray(c.pesos, dtype=float)
    if not np.isfinite(pesos).all() or (pesos <= 0).any():
        raise ContratoInvalido("pesos devem ser finitos e positivos")


def conferir_controles(c: ConjuntoAvaliacao, registrados: dict) -> dict:
    obtidos = {"episodios": len(c.ids), "positivos_guarana": int(np.sum(c.y)),
               "episodios_com_refrigerante": int(np.sum(c.y_refrigerante))}
    divergentes = {k: {"registrado": registrados[k], "obtido": obtidos[k]} for k in registrados if obtidos[k] != registrados[k]}
    if divergentes:
        raise ControleFalhou(divergentes)
    return obtidos


def _prever(mf: dict, base: Path, c: Contextos, comparadores: list[str]) -> dict:
    entradas = {"B0": len(c.ids), "B1": list(c.nativas), "L": list(c.matriz), "G": list(c.nativas)}
    previsoes = {}
    for nome in comparadores:
        modelo = cg.carregar_modelo_congelado(mf, nome, base)
        previsoes[nome] = cg.prever_congelado(mf, nome, base, entradas[nome], modelo=modelo)
    return previsoes


# ---------- metricas ----------

def _ic(valores) -> list[float]:
    return [float(v) for v in m.intervalo(np.asarray(valores, dtype=float))]


def _com_ic(valor: float, replicas) -> dict:
    return {"valor": float(valor), "ic": _ic(replicas)}


def _replicas_validas(funcao: Callable, y: np.ndarray, s: np.ndarray, cont: np.ndarray) -> list[float]:
    """Metrica por replica de bootstrap, ignorando replicas sem as duas classes."""
    return [funcao(y, s, sample_weight=w) for w in cont if w[y == 1].sum() > 0 and w[y == 0].sum() > 0]


def _taxa_por_classe(classe: np.ndarray, y: np.ndarray, cont: np.ndarray) -> dict:
    out = {}
    for nome in ("alta", "media", "baixa"):
        marca = (classe == nome).astype(float)
        if not marca.any():
            continue
        den, num = cont @ marca, cont @ (marca * y)
        ok = den > 0
        out[nome] = {"n": int(marca.sum()), "positivos": int((marca * y).sum()), "taxa": float((marca * y).sum() / marca.sum()),
                     "ic": _ic(num[ok] / den[ok]) if ok.any() else None}
    return out


def _metricas_do_modelo(nome: str, c: ConjuntoAvaliacao, previsao: cg.Previsao, cont: np.ndarray, bloco: dict) -> dict:
    y, s, w = np.asarray(c.y), previsao.score, np.asarray(c.pesos, dtype=float)
    perda, soma = m.perdas_log(y, s), cont.sum(axis=1)
    aplicado = previsao.ordena_contextos and bloco["t30_aplicavel"] is not None
    t30 = bloco["t30_aplicavel"] if aplicado else None
    out = {"interpretacao": {"ordena_contextos": previsao.ordena_contextos, "fatores_de_contexto": previsao.fatores_de_contexto,
                             "aviso": previsao.aviso},
           "log_loss": _com_ic(perda.mean(), cont @ perda / soma),
           "brier": _com_ic(np.mean((s - y) ** 2), cont @ (s - y) ** 2 / soma),
           "o_e": _com_ic(y.sum() / s.sum(), (cont @ y) / (cont @ s)),
           "ponderadas": {"log_loss": m.perda_log(y, s, w), "brier": float(brier_score_loss(y, s, sample_weight=w)),
                          "o_e": float(np.sum(w * y) / np.sum(w * s))}}
    if previsao.ordena_contextos:
        cal, cal_ic = m.calibracao(y, s), m.calibracao(y, s, cont)
        out["auc"] = _com_ic(roc_auc_score(y, s), _replicas_validas(roc_auc_score, y, s, cont))
        out["ap"] = _com_ic(average_precision_score(y, s), _replicas_validas(average_precision_score, y, s, cont))
        out["calibracao_inclinacao"] = _com_ic(cal["inclinacao"], cal_ic["inclinacao"])
        out["calibracao_intercepto_no_todo"] = _com_ic(cal["intercepto_no_todo"], cal_ic["intercepto_no_todo"])
    else:
        out.update(auc=None, ap=None, calibracao_inclinacao=None, motivo_metricas_ausentes=MOTIVO_SEM_ORDENACAO)
    out["classes"] = {"contagem": {k: int((previsao.classe == k).sum()) for k in ("alta", "media", "baixa")
                                   if (previsao.classe == k).any()},
                      "taxa_por_classe": _taxa_por_classe(previsao.classe, y, cont)}
    out["limiar_t30"] = {"aplicado": aplicado, "t": t30, "diagnostico_fora_da_dobra": bloco["limiares_fora_da_dobra"]["0.30"]}
    out["matriz_de_confusao_t30"] = m.metricas_no_limiar(y, m.sinalizados(s, t30))
    out["cobertura_por_limiar"] = {
        q: {"t": bloco["limiares_fora_da_dobra"][q], "aplicado": previsao.ordena_contextos,
            "cobertura_obtida": float(np.mean(m.sinalizados(s, bloco["limiares_fora_da_dobra"][q])))
            if previsao.ordena_contextos else 0.0} for q in COBERTURAS}
    return out


def _limitacoes(config: dict) -> list[str]:
    return [
        f"{config['rotulo_obrigatorio']}: os agregados de 2017 ({config['controles']}) eram conhecidos antes do congelamento",
        (f"{config['selecionado']} e o modelo escolhido com 2008; os demais sao comparadores previamente definidos "
         "e nao substituem a selecao original: qualquer selecao posterior e exploratoria e deve ser identificada assim"),
        ("o modelo escolhido e um preditor constante: log-loss, Brier e O/E dependem so da taxa de 2017, ja conhecida pelos "
         "agregados; AUC, AP e inclinacao nao se aplicam"),
        "sem reajuste, selecao de hiperparametro, recalibracao nem mudanca de limiar ou de classes com dados de 2017",
        "intervalos por bootstrap de UPAs sao condicionais aos modelos congelados e ignoram o estrato",
        "a composicao de dias, regioes e perfis difere entre as edicoes; cobertura e classes sao resultado, sem correcao"]


def _calcular(mf: dict, c: ConjuntoAvaliacao, previsoes: dict, obtidos: dict, manifesto_sha256: str, replicas: int | None,
              selado: Path | None) -> dict:
    config, boot = mf["avaliacao_temporal"], mf["avaliacao_temporal"]["bootstrap"]
    n_replicas = boot["replicas"] if replicas is None else int(replicas)
    cont = m.contagens_bootstrap(np.asarray(c.grupos), n_replicas, boot["semente"])
    modelos = {nome: _metricas_do_modelo(nome, c, previsoes[nome], cont, mf["modelos"][nome]) for nome in config["comparadores"]}
    perdas = {nome: m.perdas_log(np.asarray(c.y), previsoes[nome].score) for nome in config["comparadores"]}
    base = config["selecionado"]
    comparacoes = {}
    for nome in config["comparadores"]:
        if nome == base:
            continue
        d = m.delta_pareado(perdas[nome], perdas[base], np.asarray(c.grupos), contagens=cont)
        comparacoes[nome] = {"delta_log_loss": d["delta"], "ic": [float(v) for v in d["ic"]], "direcao": DIRECAO,
                             "papel": "comparador descritivo; nao substitui a selecao feita com 2008"}
    pesos, y = np.asarray(c.pesos, dtype=float), np.asarray(c.y)
    selo = estado_do_selo(selado) if selado is not None else None
    return {"rotulo": config["rotulo_obrigatorio"], "manifesto_sha256": manifesto_sha256, "selecionado_em_2008": base,
            "comparadores": list(config["comparadores"]), "n": len(c.ids),
            "prevalencia": {"nao_ponderada": float(y.mean()), "ponderada": float(np.sum(pesos * y) / np.sum(pesos))},
            "controles": {"registrados": config["controles"], "obtidos": obtidos},
            "bootstrap": {"replicas": n_replicas, "semente": boot["semente"], "unidade": "cod_upa",
                          "diferente_do_manifesto": n_replicas != boot["replicas"]},
            "modelos": modelos, "comparacoes_com_b0": comparacoes,
            "exposicao_previa": {"agregados_conhecidos": config["controles"],
                                 "consequencia": ("as metricas globais do preditor constante sao funcao direta da taxa de "
                                                  "2017, ja conhecida; a abertura so traz informacao nova pelos "
                                                  "comparadores e pelas classes")},
            "limitacoes": _limitacoes(config), "selo": selo, "gerado_em_utc": _agora()}


# ---------- comando unico ----------

def _conferir_local_do_registro(registro: Path, outras: dict[str, Path]) -> None:
    """O registro e um local proprio: fora do destino, do repositorio e das pastas de dados."""
    real = Path(registro).resolve()
    for nome, pasta in outras.items():
        if _sobrepoe(real, Path(pasta).resolve()):
            raise PastasInvalidas(f"registro e {nome} se sobrepoem; o registro de aberturas precisa de uma pasta propria")


def _raiz_fixada_do_registro(mf: dict, informada: Path | None) -> Path:
    """A raiz do registro e a fixada no manifesto. Uma raiz informada so serve para conferir: se difere, recusa."""
    fixada = (mf.get("registro_de_aberturas") or {}).get("raiz")
    if not fixada:
        raise RegistroInvalido("manifesto sem a raiz do registro de aberturas fixada: gere o manifesto operacional")
    if informada is not None and Path(informada).resolve() != Path(fixada):
        raise RegistroInvalido(f"registro informado ({informada}) difere do fixado no manifesto ({fixada}); mudar a raiz "
                               "exige novo manifesto e nova autorizacao")
    return Path(fixada)


def _conferir_entradas(mf: dict, arquivos: Mapping[str, Path], nomes: tuple[str, ...]) -> None:
    """Cada arquivo de entrada da avaliacao tem o SHA-256 registrado no manifesto (vindo dos registros de aquisicao)."""
    registradas = mf.get("entradas_avaliacao_sha256") or {}
    for nome in nomes:
        if nome not in registradas or m.sha256_arquivo(arquivos[nome]) != registradas[nome]:
            raise ContratoInvalido(f"entrada {nome}: ausente do manifesto ou diferente da registrada")


def avaliar(*, manifesto: Path, manifesto_sha256: str, autorizacao: str, repo: Path, destino: Path,
            carregar_contextos: Callable[[], Contextos], carregar_reservado: Callable[[], Reservado],
            registro: Path | None = None, entradas: Mapping[str, Path] | None = None,
            pastas: tuple[Path, Path] | None = None, replicas: int | None = None, retomada: str | None = None,
            motivo: str | None = None, processo: dict | None = None) -> dict:
    """Abre o conjunto de avaliacao uma unica vez por manifesto.

    Preparacao (sem acesso a alvo ou peso; falha aqui pode ser repetida): autorizacao, pastas, manifesto, origem do
    codigo carregado, registro de aberturas, leitura e contrato dos contextos, previsoes dos modelos congelados.
    Exposicao: o marco e gravado e so entao `carregar_reservado` e chamado; seguem contrato, controles, metricas e
    resultado. Qualquer falha daqui em diante fica no historico e nao volta ao estado "nao aberto".
    A origem do codigo carregado e conferida tres vezes: apos o manifesto, antes do marco e antes de gravar o resultado."""
    if autorizacao != autorizacao_exigida(manifesto_sha256):
        raise AutorizacaoAusente("autorizacao ausente ou diferente: informe exatamente ABRIR-2017:<sha256 do manifesto>")
    manifesto, repo, destino = Path(manifesto), Path(repo), Path(destino)
    if pastas is not None:
        validar_pastas(pastas[0], pastas[1], destino, repo)
    mf = cg.verificar_manifesto(manifesto, manifesto_sha256, repo)
    codigo = mf["codigo"]["sha256"]
    carregado = conferir_origem_carregada(codigo, repo)  # o que o Python importou, nao so o que esta na pasta
    fixada = _raiz_fixada_do_registro(mf, registro)
    _conferir_local_do_registro(fixada, {"destino": destino, "repositorio": repo,
                                         **({"trabalho": pastas[0], "selado": pastas[1]} if pastas else {})})
    registro = Registro(fixada, manifesto_sha256)
    tentativa = conferir_abertura(registro, retomada, motivo)
    config = mf["avaliacao_temporal"]
    evento = {"manifesto_sha256": manifesto_sha256, "tentativa": tentativa, "destino": str(destino), "manifesto": str(manifesto),
              "retomada": {"autorizacao": retomada, "motivo": motivo.strip()} if retomada else None,
              "codigo_carregado": carregado, "processo": processo}
    destino.mkdir(parents=True, exist_ok=True)
    try:
        if entradas is not None:
            _conferir_entradas(mf, entradas, (ARQ_IDS, ARQ_NATIVAS, ARQ_MATRIZ))
        contextos = carregar_contextos()
        conferir_contextos(contextos)
        previsoes = _prever(mf, manifesto.parent, contextos, config["comparadores"])
        conferir_origem_carregada(codigo, repo)  # ultima conferencia antes de expor o alvo
    except Exception as erro:
        _encerrar(registro, destino, evento, PREPARACAO_FALHOU, erro)
        raise
    registro.marcar_exposicao({**evento, "status": EXPOSICAO_INICIADA})  # antes de qualquer leitura de alvo ou peso
    try:
        if entradas is not None:
            _conferir_entradas(mf, entradas, (ARQ_PESOS,))
        conjunto = juntar(contextos, carregar_reservado())
        conferir_contrato(conjunto)
        obtidos = conferir_controles(conjunto, config["controles"])
        resultado = _calcular(mf, conjunto, previsoes, obtidos, manifesto_sha256, replicas, pastas[1] if pastas else None)
        resultado["abertura"] = {"tentativa": tentativa, "registro": str(registro.pasta),
                                 "retomada": {**evento["retomada"], "rotulo": ROTULO_RETOMADA} if retomada else None}
        conferir_origem_carregada(codigo, repo)  # nada e gravado se o codigo mudou durante a execucao
        arquivo = destino / ARQ_RESULTADO
        _gravar_json(arquivo, resultado)
        final = {**evento, "status": CONCLUIDA, "resultado": ARQ_RESULTADO, "resultado_sha256": m.sha256_arquivo(arquivo)}
        registro.anotar(final)
    except BaseException as erro:  # inclui interrupcao pelo teclado: a exposicao ja ocorreu e fica registrada
        _encerrar(registro, destino, evento, APOS_EXPOSICAO, erro)
        raise
    _gravar_json(destino / ARQ_ESTADO, final)
    return resultado


# ---------- leitores de arquivos ----------

def _ler_csv(caminho: Path, obrigatorias: tuple[str, ...]) -> list[dict]:
    try:
        with open(caminho, newline="", encoding="utf-8") as f:
            linhas = list(csv.DictReader(f))
    except OSError as erro:
        raise ContratoInvalido(f"{Path(caminho).name}: ilegivel ({erro})") from erro
    if not linhas:
        raise ContratoInvalido(f"{Path(caminho).name}: sem linhas")
    faltando = [c for c in obrigatorias if c not in linhas[0]]
    if faltando:
        raise ContratoInvalido(f"{Path(caminho).name}: coluna ausente: {faltando}")
    return linhas


def _numero(texto: str, rotulo: str):
    if texto in ("", None):
        return None
    try:
        return int(texto)
    except ValueError:
        try:
            return float(texto)
        except ValueError as erro:
            raise ContratoInvalido(f"{rotulo}: {texto!r} nao e numero") from erro


def carregar_contextos_arquivos(trabalho: Path) -> Contextos:
    """Le identificadores, nativas e matriz da pasta de trabalho. Nao abre a pasta selada nem o arquivo de alvo."""
    trabalho = Path(trabalho)
    ids_ = _ler_csv(trabalho / ARQ_IDS, ("episodio_id", "cod_upa"))
    nativas = _ler_csv(trabalho / ARQ_NATIVAS, ("dia_semana", "faixa_hora"))
    matriz = _ler_csv(trabalho / ARQ_MATRIZ, ())
    ids = [linha["episodio_id"] for linha in ids_]
    if ids != sorted(ids):
        raise ContratoInvalido("identificadores fora da ordem por episodio_id")
    if len(nativas) != len(ids) or len(matriz) != len(ids):
        raise ContratoInvalido("nativas e matriz devem ter uma linha por episodio, na ordem dos identificadores")
    return Contextos(
        ids=tuple(ids), grupos=np.array([linha["cod_upa"] for linha in ids_]),
        nativas=tuple({c: (_numero(v, c) if c in NATIVAS_NUMERICAS else v) for c, v in linha.items()} for linha in nativas),
        matriz=tuple({c: _numero(v, c) for c, v in linha.items()} for linha in matriz))


def montar_reservado(selado: Path, alvo: Mapping[str, tuple[int, int]]) -> Reservado:
    """Junta o alvo construido por codigo ({episodio_id: (guarana, refrigerante)}) aos pesos da pasta selada. Os dois
    conjuntos de episodios precisam ser exatamente iguais."""
    pesos = _ler_csv(Path(selado) / ARQ_PESOS, ("episodio_id", "peso_final"))
    por_peso = {linha["episodio_id"]: linha for linha in pesos}
    if len(por_peso) != len(pesos) or set(por_peso) != set(alvo):
        raise ContratoInvalido("episodio_id do alvo construido difere do arquivo de pesos, ou ha episodio repetido")
    ids = sorted(por_peso)
    return Reservado(
        ids=tuple(ids), pesos=np.array([_numero(por_peso[i]["peso_final"], "peso_final") for i in ids], dtype=float),
        y=np.array([alvo[i][0] for i in ids]), y_refrigerante=np.array([alvo[i][1] for i in ids]))


def comando(argv: list[str] | None = None, processo: dict | None = None) -> int:
    """Le os argumentos, avalia e devolve o codigo de saida (0 concluida, 3 interrompida)."""
    p = argparse.ArgumentParser(prog="occasion.abrir_avaliacao", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifesto", required=True, help="manifesto de congelamento (json)")
    p.add_argument("--manifesto-sha256", required=True, help="SHA-256 do manifesto, informado a parte")
    p.add_argument("--autorizacao", required=True, help="texto exato ABRIR-2017:<sha256 do manifesto>")
    p.add_argument("--destino", required=True, help="pasta dos resultados, fora do repositorio")
    p.add_argument("--bruta", help="raiz dos dados publicos, de onde o alvo e construido (padrao: CASE_AMBEV_RAW_ROOT)")
    p.add_argument("--repo", default=str(REPO))
    p.add_argument("--trabalho", help="pasta de trabalho (padrao: CASE_AMBEV_SAIDA_ROOT)")
    p.add_argument("--selado", help="pasta selada (padrao: CASE_AMBEV_SELADO_ROOT)")
    p.add_argument("--replicas", type=int, help="substitui as replicas do manifesto; fica registrado no resultado")
    p.add_argument("--retomada", help="so apos exposicao registrada: texto exato RETOMAR-2017:<sha256 do manifesto>:<tentativa>")
    p.add_argument("--motivo", help="motivo da retomada tecnica; fica gravado no historico de aberturas")
    p.add_argument("--registro", help=f"so para conferir: deve ser a raiz fixada no manifesto (ou {VAR_RAIZ_REGISTRO})")
    args = p.parse_args(argv)
    try:
        trabalho, selado = Path(args.trabalho or raiz_saidas()), Path(args.selado or raiz_selado())
        bruta = Path(args.bruta or raiz_bruta())
        manifesto_fontes = Path(args.repo) / "configs" / "manifesto_fontes.json"  # hash conferido pelo manifesto de congelamento
        preparadas: list[FontesDoAlvo] = []

        def contextos() -> Contextos:
            preparadas.append(preparar_fontes(bruta, carregar_manifesto(manifesto_fontes)))  # sem ler consumo
            return carregar_contextos_arquivos(trabalho)

        def reservado() -> Reservado:  # so e chamado depois do marco de exposicao
            return montar_reservado(selado, alvo_das_fontes(preparadas[0]))

        resultado = avaliar(manifesto=Path(args.manifesto), manifesto_sha256=args.manifesto_sha256,
                            autorizacao=args.autorizacao, repo=Path(args.repo), destino=Path(args.destino),
                            registro=Path(args.registro) if args.registro else raiz_registro(),
                            entradas={**{nome: trabalho / nome for nome in (ARQ_IDS, ARQ_NATIVAS, ARQ_MATRIZ)},
                                      ARQ_PESOS: selado / ARQ_PESOS},
                            carregar_contextos=contextos, carregar_reservado=reservado,
                            pastas=(trabalho, selado), replicas=args.replicas, retomada=args.retomada, motivo=args.motivo,
                            processo=processo)
    except (AvaliacaoInterrompida, cg.CongelamentoInvalido, AmbienteInvalido) as erro:
        sys.stderr.write(f"avaliacao temporal interrompida: {erro}\n")
        return 3
    except Exception as erro:  # falha inesperada: ja registrada por `avaliar`; sai com o mesmo codigo, sem traceback
        sys.stderr.write(f"avaliacao temporal interrompida por falha inesperada: {type(erro).__name__}: {erro}\n")
        return 3
    sys.stdout.write(json.dumps({"rotulo": resultado["rotulo"], "resultado": str(Path(args.destino) / ARQ_RESULTADO),
                                 "status": CONCLUIDA}, ensure_ascii=False) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entrada do processo filho: so roda se foi iniciado por `occasion.abrir_avaliacao`, com bytecode novo e exclusivo."""
    try:
        processo = conferir_processo()
    except ProcessoInvalido as erro:
        sys.stderr.write(f"avaliacao temporal interrompida: {erro}\n")
        return 3
    return comando(argv, processo)


if __name__ == "__main__":
    sys.exit(main())
