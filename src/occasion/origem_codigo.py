"""Origem e integridade do codigo realmente carregado.

Conferir os arquivos de uma pasta nao diz de onde o Python importou o pacote. Aqui se confere o que esta em `sys.modules`:
cada modulo de `occasion` precisa ter sido carregado do arquivo-fonte do repositorio conferido, e esse arquivo precisa ter o
SHA-256 registrado no manifesto. So usa a biblioteca padrao.
"""

import hashlib
import os
import sys
from collections.abc import Mapping
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType

PACOTE = "occasion"


class OrigemInvalida(RuntimeError):
    """Modulo carregado de outro lugar, sem arquivo-fonte ou com fonte diferente da registrada no manifesto."""


def modulos_carregados() -> dict[str, ModuleType]:
    """Modulos do pacote em `sys.modules`, pelo nome real (o comando executado com `-m` aparece como `__main__`)."""
    achados = {}
    for nome, modulo in list(sys.modules.items()):
        spec = getattr(modulo, "__spec__", None)
        real = spec.name if spec is not None else nome
        if real == PACOTE or real.startswith(PACOTE + "."):
            achados[real] = modulo
    return achados


def _problema(nome: str, modulo: ModuleType, raiz: Path, hashes: Mapping[str, str]) -> tuple[str | None, str, str | None]:
    """(problema, caminho relativo, sha256 lido) de um modulo carregado."""
    partes = nome.split(".")[1:]
    if len(partes) > 1:
        return f"{nome}: subpacote inesperado", nome, None
    arquivo = "__init__.py" if not partes else f"{partes[0]}.py"
    relativo = f"src/{PACOTE}/{arquivo}"
    spec, origem = getattr(modulo, "__spec__", None), getattr(modulo, "__file__", None)
    if spec is None or type(spec.loader) is not SourceFileLoader or not origem or spec.origin != origem:
        return f"{nome}: nao foi carregado de arquivo-fonte ({origem})", relativo, None
    try:
        mesmo = os.path.samefile(origem, raiz / arquivo)
    except OSError:
        mesmo = False
    if not mesmo:
        return f"{nome}: carregado de {origem}, fora do repositorio conferido", relativo, None
    sha = hashlib.sha256(Path(origem).read_bytes()).hexdigest()
    if sha != hashes.get(relativo):
        return f"{nome}: a fonte carregada difere da registrada no manifesto ({relativo})", relativo, sha
    return None, relativo, sha


def conferir_origem_carregada(hashes_do_manifesto: Mapping[str, str], repo: Path) -> dict[str, str]:
    """Confere todos os modulos de `occasion` carregados neste processo e devolve {caminho relativo: sha256}.

    Levanta OrigemInvalida com todas as divergencias. Deve ser repetida: so enxerga o que ja foi importado e o que esta
    em disco no momento da chamada."""
    raiz = Path(repo) / "src" / PACOTE
    modulos = modulos_carregados()
    problemas, conferidos = [], {}
    pacote = modulos.get(PACOTE)
    if pacote is not None and len(list(getattr(pacote, "__path__", []))) != 1:
        problemas.append(f"{PACOTE}: pacote com mais de uma pasta de origem")
    for nome in sorted(modulos):
        problema, relativo, sha = _problema(nome, modulos[nome], raiz, hashes_do_manifesto)
        if problema:
            problemas.append(problema)
        else:
            conferidos[relativo] = sha
    if problemas:
        raise OrigemInvalida("codigo carregado nao confere: " + "; ".join(problemas))
    return conferidos
