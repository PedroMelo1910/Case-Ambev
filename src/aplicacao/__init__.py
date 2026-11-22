"""Ferramenta de oportunidade por contexto (pizza + guarana): aplicacao fora do codigo cientifico congelado.

Modulos: `previsao_consumo` (contrato das entradas e inferencia historica), `aprendizado` (estimadores, metricas e
avaliacao, usados pelo treino historico e pelo da empresa), `regras` (rotulos, comparacao com o habitual e priorizacao),
`empresa_dados`, `empresa_modelo` e `empresa_servico` (dados da empresa) e
`servidor` (interface local). Aqui ficam so as excecoes e as conferencias comuns.
"""

import hashlib
import json
import reprlib
from collections.abc import Mapping
from pathlib import Path

import sklearn

MAXIMO_EXTRAS_NA_MENSAGEM = 10
MAXIMO_BITS_NA_MENSAGEM = 64


class EntradaInvalida(ValueError):
    """Campo obrigatorio ausente, campo desconhecido ou valor de tipo ou dominio invalido. Nunca vira decisao."""


class ArtefatoInvalido(RuntimeError):
    """Artefato de modelo ausente, ilegivel, alterado depois do treino ou de outra versao do scikit-learn."""


def conferir_artefatos(pasta: Path, descricao: str) -> dict:
    """Confere os SHA-256 de `meta.json` e a versao do scikit-learn antes de qualquer `joblib.load`. Detecta corrupcao e
    troca acidental; nao protege contra quem pode reescrever a pasta inteira (o hash fica nela)."""
    try:
        meta = json.loads((Path(pasta) / "meta.json").read_text(encoding="utf-8"))
        for nome, esperado in meta["sha256"].items():
            if hashlib.sha256((Path(pasta) / nome).read_bytes()).hexdigest() != esperado:
                raise ArtefatoInvalido(f"{descricao}: {nome} diferente do registrado em meta.json")
        versao = meta["scikit-learn"]
    except (OSError, KeyError, TypeError, ValueError) as erro:
        raise ArtefatoInvalido(f"{descricao}: arquivos ausentes ou ilegíveis ({type(erro).__name__})") from erro
    if versao != sklearn.__version__:
        raise ArtefatoInvalido(f"{descricao}: gerado com scikit-learn {versao}, ambiente {sklearn.__version__}")
    return meta


def conferir_inteiro(valor: object, campo: str, minimo: int, maximo: int) -> int:
    """Inteiro de verdade (bool nao conta) dentro de [minimo, maximo]; senao EntradaInvalida."""
    if isinstance(valor, bool) or not isinstance(valor, int) or not minimo <= valor <= maximo:
        raise EntradaInvalida(f"{campo} deve ser inteiro de {minimo} a {maximo}, recebido {recebido(valor)}")
    return valor


def recebido(valor: object) -> str:
    """Descricao curta e segura do valor recebido: repr truncado para tipos simples, so o nome do tipo para o resto.
    Inteiros enormes (o repr de mais de 4300 digitos levanta ValueError) viram um rotulo fixo."""
    if type(valor) is int and valor.bit_length() > MAXIMO_BITS_NA_MENSAGEM:
        return "<int grande>"
    if type(valor) in (bool, int, float, str, type(None)):
        return reprlib.repr(valor)
    return f"<{type(valor).__name__}>"


def conferir_campos(mapa: Mapping, campos: tuple[str, ...], rotulo: str, dica: str = "") -> None:
    """Exige exatamente `campos`: ausentes e extras viram EntradaInvalida (a lista de extras e cortada na mensagem)."""
    ausentes = [campo for campo in campos if campo not in mapa]
    if ausentes:
        raise EntradaInvalida(f"campos obrigatorios ausentes: {ausentes}")
    extras = sorted(recebido(campo) for campo in mapa if campo not in campos)
    if extras:
        resto = len(extras) - MAXIMO_EXTRAS_NA_MENSAGEM
        sufixo = f" (e mais {resto})" if resto > 0 else ""
        raise EntradaInvalida(f"campos que o {rotulo} nao aceita{dica}: {extras[:MAXIMO_EXTRAS_NA_MENSAGEM]}{sufixo}")
