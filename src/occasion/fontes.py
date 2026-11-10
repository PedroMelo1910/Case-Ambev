"""Fontes do IBGE: caminhos de ambiente, SHA-256, manifesto e coleta idempotente.

Tudo o que garante que os arquivos usados sao os registrados no manifesto fica aqui: um arquivo
com hash divergente interrompe a execucao em qualquer etapa (download, verificacao ou extracao).
"""

import hashlib
import json
import os
import re
import unicodedata
import urllib.request
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

BLOCO_BYTES = 1024 * 1024
TIMEOUT_S = 120
EDICOES = ("2008-2009", "2017-2018")
VAR_RAIZ_BRUTA, VAR_RAIZ_SAIDAS, VAR_RAIZ_SELADO = "CASE_AMBEV_RAW_ROOT", "CASE_AMBEV_SAIDA_ROOT", "CASE_AMBEV_SELADO_ROOT"
_SHA256, _DATA = re.compile(r"[0-9a-f]{64}"), re.compile(r"\d{4}-\d{2}-\d{2}")


# --- ambiente (caminhos sempre fora do repositorio, vindos de variaveis) -------------------------
class AmbienteInvalido(RuntimeError):
    """Variavel de ambiente obrigatoria ausente."""


def _ler(var: str, env: Mapping[str, str] | None) -> str | None:
    return (os.environ if env is None else env).get(var) or None


def raiz_bruta(env: Mapping[str, str] | None = None) -> Path:
    valor = _ler(VAR_RAIZ_BRUTA, env)
    if valor is None:
        raise AmbienteInvalido(f"Defina {VAR_RAIZ_BRUTA} com a pasta dos dados publicos (fora do repositorio).")
    return Path(valor)


def raiz_saidas(env: Mapping[str, str] | None = None) -> Path:
    valor = _ler(VAR_RAIZ_SAIDAS, env)
    return Path(valor) if valor else raiz_bruta(env) / "pkg_f1_01" / "saidas"


def raiz_selado(env: Mapping[str, str] | None = None) -> Path:
    """Pasta do conjunto selado de 2017 (somente leitura, fora da arvore de trabalho)."""
    valor = _ler(VAR_RAIZ_SELADO, env)
    return Path(valor) if valor else raiz_bruta(env) / "pkg_f1_01" / "selado_2017"


# --- hash -----------------------------------------------------------------------------------------
class HashDivergente(RuntimeError):
    """O arquivo nao corresponde ao SHA-256 (ou ao tamanho) registrado no manifesto."""


def sha256_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        while bloco := f.read(BLOCO_BYTES):
            h.update(bloco)
    return h.hexdigest()


def verificar_sha256(caminho: Path, esperado: str, tamanho_bytes: int | None = None) -> None:
    """Interrompe (HashDivergente) se tamanho ou SHA-256 divergirem do esperado."""
    caminho = Path(caminho)
    if tamanho_bytes is not None and caminho.stat().st_size != tamanho_bytes:
        raise HashDivergente(f"{caminho.name}: tamanho {caminho.stat().st_size} diverge do manifesto ({tamanho_bytes})")
    obtido = sha256_arquivo(caminho)
    if obtido != esperado:
        raise HashDivergente(f"{caminho.name}: SHA-256 {obtido} diverge do manifesto ({esperado})")


# --- manifesto ------------------------------------------------------------------------------------
class ManifestoInvalido(ValueError):
    """Manifesto incompleto ou inseguro."""


@dataclass(frozen=True)
class Membro:
    nome: str
    destino_relativo: str
    sha256: str
    tamanho_bytes: int


@dataclass(frozen=True)
class Fonte:
    id: str
    edicao: str
    url: str
    caminho_relativo: str
    tamanho_bytes: int
    sha256: str
    baixado_em: str
    last_modified_remoto: str | None
    membros: tuple[Membro, ...]
    citacao: str


@dataclass(frozen=True)
class Manifesto:
    licenca_nota: str
    fontes: tuple[Fonte, ...]


def _relativo(valor: str, contexto: str) -> str:
    caminho = PurePosixPath(valor)
    if caminho.is_absolute() or ".." in caminho.parts or "\\" in valor or re.match(r"^[A-Za-z]:", valor):
        raise ManifestoInvalido(f"{contexto}: caminho deve ser relativo e sem '..': {valor!r}")
    return valor


def _membro(d: dict, contexto: str) -> Membro:
    if not _SHA256.fullmatch(d["sha256"]):
        raise ManifestoInvalido(f"{contexto}: sha256 invalido")
    return Membro(d["nome"], _relativo(d["destino_relativo"], contexto), d["sha256"], int(d["tamanho_bytes"]))


def _fonte(d: dict) -> Fonte:
    ctx = d.get("id", "?")
    regras = [(d["edicao"] in EDICOES, "edicao desconhecida"), (bool(_SHA256.fullmatch(d["sha256"])), "sha256 invalido"),
              (d["url"].startswith("https://"), "URL deve ser https"),
              (bool(_DATA.fullmatch(d["baixado_em"])), "baixado_em deve ser AAAA-MM-DD"),
              ("IBGE" in d["citacao"] and d["url"] in d["citacao"], "citacao deve nomear o IBGE e conter a URL")]
    for ok, msg in regras:
        if not ok:
            raise ManifestoInvalido(f"{ctx}: {msg}")
    return Fonte(id=d["id"], edicao=d["edicao"], url=d["url"], caminho_relativo=_relativo(d["caminho_relativo"], ctx),
                 tamanho_bytes=int(d["tamanho_bytes"]), sha256=d["sha256"], baixado_em=d["baixado_em"],
                 last_modified_remoto=d.get("last_modified_remoto"),
                 membros=tuple(_membro(m, ctx) for m in d.get("membros", [])), citacao=d["citacao"])


def carregar_manifesto(caminho: Path) -> Manifesto:
    bruto = json.loads(Path(caminho).read_text(encoding="utf-8"))
    try:
        return Manifesto(licenca_nota=bruto["licenca_nota"], fontes=tuple(_fonte(f) for f in bruto["fontes"]))
    except KeyError as erro:
        raise ManifestoInvalido(f"campo ausente no manifesto: {erro}") from erro


# --- coleta ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ResultadoColeta:
    fonte_id: str
    status: str  # "verificado" (ja existia, hash conferido) ou "baixado"
    caminho: Path


def baixar_url(url: str, destino: Path) -> None:
    """Baixa para <destino>.parcial e renomeia ao final (aceita https:// e file://)."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_name(destino.name + ".parcial")
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r, open(parcial, "wb") as f:
        while bloco := r.read(BLOCO_BYTES):
            f.write(bloco)
    os.replace(parcial, destino)


def _obter(fonte: Fonte, raiz: Path, baixar: Callable[[str, Path], None]) -> ResultadoColeta:
    destino = raiz / fonte.caminho_relativo
    if destino.exists():
        verificar_sha256(destino, fonte.sha256, fonte.tamanho_bytes)
        return ResultadoColeta(fonte.id, "verificado", destino)
    novo = destino.with_name(destino.name + ".novo")
    try:
        baixar(fonte.url, novo)
        verificar_sha256(novo, fonte.sha256, fonte.tamanho_bytes)
    except BaseException:
        novo.unlink(missing_ok=True)
        raise
    os.replace(novo, destino)
    return ResultadoColeta(fonte.id, "baixado", destino)


def coletar(manifesto: Manifesto, raiz: Path, baixar: Callable[[str, Path], None] = baixar_url) -> list[ResultadoColeta]:
    """Garante todas as fontes sob `raiz`; qualquer hash divergente interrompe a execucao."""
    return [_obter(f, Path(raiz), baixar) for f in manifesto.fontes]


def _nome_legivel(info: zipfile.ZipInfo) -> str:
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode("cp437").decode("cp850")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename


def _recusar_caminho_inseguro(nome: str) -> None:
    ascii_ = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode("ascii")
    partes = [p for p in ascii_.replace("\\", "/").split("/") if p]
    if any(p in ("..", ".") for p in partes) or nome.startswith("/"):
        raise ValueError(f"Membro com caminho inseguro: {nome!r}")


def extrair_membros(fonte: Fonte, raiz: Path) -> list[Path]:
    """Extrai so os membros listados no manifesto e confere o SHA-256 de cada um (idempotente)."""
    raiz = Path(raiz)
    pedidos = {m.nome: m for m in fonte.membros}
    saidas: list[Path] = []
    with zipfile.ZipFile(raiz / fonte.caminho_relativo) as z:
        for info in z.infolist():
            nome = _nome_legivel(info)
            if info.is_dir() or nome not in pedidos:
                continue
            _recusar_caminho_inseguro(nome)
            membro = pedidos.pop(nome)
            destino = raiz / membro.destino_relativo
            pronto = (destino.exists() and destino.stat().st_size == membro.tamanho_bytes
                      and sha256_arquivo(destino) == membro.sha256)
            if not pronto:
                destino.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(destino, "wb") as dst:
                    while bloco := src.read(BLOCO_BYTES):
                        dst.write(bloco)
            verificar_sha256(destino, membro.sha256, membro.tamanho_bytes)
            saidas.append(destino)
    if pedidos:
        raise FileNotFoundError(f"Membros ausentes em {fonte.id}: {sorted(pedidos)}")
    return saidas
