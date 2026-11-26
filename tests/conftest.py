"""Configuracao dos testes (pasta tests/ do repositorio).

Testes que exigem os microdados reais usam a marca `real` e CASE_AMBEV_RAW_ROOT.
"""

import stat
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(scope="session")
def repo() -> Path:
    return REPO


@pytest.fixture
def liberar_ao_final():
    """Devolve permissao de escrita a pastas seladas para o pytest conseguir limpar."""
    pastas = []
    yield pastas.append
    for pasta in pastas:
        for caminho in [pasta, *Path(pasta).rglob("*")]:
            if caminho.exists():
                caminho.chmod(caminho.stat().st_mode | stat.S_IWUSR)
