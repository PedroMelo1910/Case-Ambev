"""Lancador do comando unico da avaliacao temporal.

Uso: python -m occasion.abrir_avaliacao <mesmos argumentos de occasion.avaliacao_temporal>

A avaliacao roda em um PROCESSO NOVO, cuja pasta de bytecode e criada agora, vazia, so para ele, e definida ANTES de o
Python iniciar (`-X pycache_prefix`). Assim nenhum bytecode antigo, de `__pycache__` ou de um PYTHONPYCACHEPREFIX herdado,
e reaproveitado: tudo e compilado das fontes que a avaliacao confere. O processo filho recusa rodar se nao foi iniciado
assim. Este modulo so usa a biblioteca padrao e nao importa o restante do pacote.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MODULO_AVALIACAO = "occasion.avaliacao_temporal"
VAR_CACHE_DO_PROCESSO = "CASE_AMBEV_CACHE_DO_PROCESSO"
ARQ_LANCADOR = "lancador.json"


class ProcessoInvalido(RuntimeError):
    """A avaliacao nao foi iniciada pelo lancador, com pasta de bytecode nova e exclusiva."""


def lancar(argv: list[str]) -> int:
    """Cria a pasta de bytecode do processo, inicia a avaliacao nela e devolve o codigo de saida."""
    cache = Path(tempfile.mkdtemp(prefix="avaliacao_temporal_cache_"))
    try:
        (cache / ARQ_LANCADOR).write_text(json.dumps({"pid_lancador": os.getpid()}), encoding="utf-8")
        origem = str(Path(__file__).resolve().parents[1])  # o filho importa o pacote do mesmo lugar que este lancador
        ambiente = {k: v for k, v in os.environ.items() if k != "PYTHONPYCACHEPREFIX"}
        ambiente["PYTHONPATH"] = os.pathsep.join([origem, *filter(None, [os.environ.get("PYTHONPATH")])])
        ambiente[VAR_CACHE_DO_PROCESSO] = str(cache)
        comando = [sys.executable, "-X", f"pycache_prefix={cache}", "-m", MODULO_AVALIACAO, *argv]
        return subprocess.run(comando, env=ambiente, check=False).returncode
    finally:
        shutil.rmtree(cache, ignore_errors=True)


def conferir_processo() -> dict:
    """No processo filho: a pasta de bytecode em uso e a que o lancador (processo pai) criou para ele."""
    cache = os.environ.get(VAR_CACHE_DO_PROCESSO)
    try:
        pid_lancador = json.loads((Path(cache) / ARQ_LANCADOR).read_text(encoding="utf-8"))["pid_lancador"]
        em_uso = sys.pycache_prefix is not None and os.path.samefile(sys.pycache_prefix, cache)
    except (OSError, ValueError, KeyError, TypeError):
        pid_lancador, em_uso = None, False
    if not em_uso or pid_lancador != os.getppid():
        raise ProcessoInvalido("inicie a avaliacao por `python -m occasion.abrir_avaliacao`: ela exige um processo novo, com "
                               "pasta de bytecode criada so para ele antes de o Python iniciar")
    return {"pid": os.getpid(), "cache": cache}


def main(argv: list[str] | None = None) -> int:
    return lancar(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    sys.exit(main())
