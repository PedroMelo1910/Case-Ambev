"""Entrada unica da coleta e das leituras da POF: cache local, hash e registro de aquisicao.

A coleta da POF 2008-2009 e 2017-2018 grava em `<CASE_AMBEV_RAW_ROOT>/pof_*/` pelo manifesto selado, com SHA-256 de
cada zip e de cada membro extraido, e registra a aquisicao em `<raiz>/aquisicoes/pof.json`. Tambem le os microdados de
largura fixa pelo dicionario oficial e as planilhas xlsx da documentacao.
"""

import argparse
import hashlib
import io
import json
import os
import re
import sys
import unicodedata
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree

import duckdb
import xlrd

# ======== origem: occasion (fontes, microdados, eda_dados, pipeline_pof) ========
# So as definicoes que o produto usa, movidas com o texto original em 24/11/2026 (renomes registrados no protocolo);
# o restante do pacote e a execucao
# historica completa (com o manifesto congelado) fica na copia verificada fora do repositorio.

# ---------- origem: occasion/pipeline_pof.py ----------

COLUNAS = {
    "consumo_2008": ["COD_UF", "NUM_SEQ", "NUM_DV", "NUM_DOM", "NUM_UC", "COD_INFORMANTE", "ESTRATO_POF", "PESO_FINAL",
                     "NUM_QUADRO", "V9018", "V9015", "V9001", "RENDA_TOTAL", "DIA_SEMANA"],
    "morador_2008": ["COD_UF", "NUM_SEQ", "NUM_DV", "NUM_DOM", "NUM_UC", "COD_INFORMANTE", "PESO_FINAL", "IDADE_ANOS",
                     "V0405", "TIPO_SITUACAO_REG", "NIVEL_INSTRUCAO_MORADOR"],
    "domicilio_2008": ["COD_UF", "NUM_SEQ", "NUM_DV", "NUM_DOM", "PERIODO_REAL"],
    "consumo_2017": ["UF", "ESTRATO_POF", "TIPO_SITUACAO_REG", "COD_UPA", "NUM_DOM", "NUM_UC", "COD_INFORMANTE", "QUADRO",
                     "SEQ", "V9005", "V9007", "V9001", "V9015", "V9017", "V9018", "PESO_FINAL", "RENDA_TOTAL",
                     "DIA_SEMANA", "DIA_ATIPICO"],
    "morador_2017": ["UF", "COD_UPA", "NUM_DOM", "NUM_UC", "COD_INFORMANTE", "V0306", "V0403", "V0404", "PESO_FINAL", "NIVEL_INSTRUCAO"],
}


# (membro do zip de dados, aba do dicionario) por tabela
ORIGEM_OFICIAL = {
    "consumo_2008": ("T_CONSUMO_S.txt", "Consumo Alimentar"), "morador_2008": ("T_MORADOR_S.txt", "Morador"),
    "domicilio_2008": ("T_DOMICILIO_S.txt", "Domicílio"),
    "consumo_2017": ("CONSUMO_ALIMENTAR.txt", "Consumo Alimentar"), "morador_2017": ("MORADOR.txt", "Morador"),
}


# ---------- origem: occasion/fontes.py ----------

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


# ---------- origem: occasion/microdados.py ----------

MARCA_CABECALHO = "Posição Inicial"


AMOSTRA_BYTES = 65536


@dataclass(frozen=True)
class Variavel:
    posicao: int
    tamanho: int
    codigo: str
    descricao: str
    categorias: tuple[str, ...]


@dataclass(frozen=True)
class Layout:
    campos: dict[str, tuple[int, int]]  # codigo -> (posicao inicial base 1, tamanho)
    comprimento: int


class LayoutInvalido(ValueError):
    """Linha com comprimento diferente do dicionario ou coluna inexistente."""


# --- dicionario oficial ---------------------------------------------------------------------------
def _texto(valor) -> str:
    return str(int(valor)) if isinstance(valor, float) and valor.is_integer() else str(valor).strip()


def extrair_variaveis(linhas: list[list]) -> tuple[str, list[Variavel]]:
    """Converte as linhas de uma aba em variaveis; linhas sem posicao continuam a anterior."""
    registro = next((v.strip() for lin in linhas[:5] for v in lin if isinstance(v, str) and "REGISTRO" in v.upper()), "")
    inicio = next((i + 1 for i, lin in enumerate(linhas)
                   if any(isinstance(v, str) and v.strip().startswith(MARCA_CABECALHO) for v in lin)), None)
    if inicio is None:
        return registro, []
    brutas: list[dict] = []
    for linha in linhas[inicio:]:
        pos, tam, _dec, cod, desc, cat = ([_texto(v) for v in linha[:6]] + [""] * 6)[:6]
        if pos and cod:
            brutas.append({"posicao": int(float(pos)), "tamanho": int(float(tam)) if tam else 0, "codigo": cod,
                           "descricao": desc, "categorias": [cat] if cat else []})
        elif brutas:
            if cat:
                brutas[-1]["categorias"].append(cat)
            if desc:
                brutas[-1]["descricao"] += " " + desc
    return registro, [Variavel(b["posicao"], b["tamanho"], b["codigo"], b["descricao"], tuple(b["categorias"]))
                      for b in brutas]


def ler_dicionario_xls(caminho: Path) -> dict[str, list[Variavel]]:
    """Retorna {nome_da_aba_sem_espacos_nas_pontas: variaveis} do xls oficial."""
    abas = {}
    for folha in xlrd.open_workbook(str(caminho)).sheets():
        linhas = [[folha.cell_value(r, c) for c in range(folha.ncols)] for r in range(folha.nrows)]
        abas[folha.name.strip()] = extrair_variaveis(linhas)[1]
    return abas


def layout_de(variaveis: Iterable[Variavel]) -> Layout:
    campos = {v.codigo: (v.posicao, v.tamanho) for v in variaveis}
    return Layout(campos=campos, comprimento=max(p + t - 1 for p, t in campos.values()))


# --- leitor de largura fixa -----------------------------------------------------------------------
def _origem(arquivo: Path) -> str:
    with open(arquivo, "rb") as f:
        fim = "\\r\\n" if b"\r\n" in f.read(AMOSTRA_BYTES) else "\\n"
    # latin-1 mapeia 1 byte = 1 caractere, o que mantem as posicoes do dicionario exatas.
    return ("SELECT linha FROM read_csv(?, columns={'linha': 'VARCHAR'}, header=false, delim='\\x01', quote='', "
            f"escape='', auto_detect=false, encoding='latin-1', new_line='{fim}')")


def sql_colunas(arquivo: Path, layout: Layout, colunas: list[str]) -> str:
    """Subconsulta (com um `?` para o caminho) que fatia `colunas` do arquivo, sem materializar."""
    ausentes = [c for c in colunas if c not in layout.campos]
    if ausentes:
        raise LayoutInvalido(f"Colunas ausentes no dicionario: {ausentes}")
    expressoes = ", ".join(
        f"nullif(trim(substr(linha, {layout.campos[c][0]}, {layout.campos[c][1]})), '') AS \"{c}\"" for c in colunas)
    return f"SELECT {expressoes} FROM ({_origem(Path(arquivo))})"


def carregar_registro(con: duckdb.DuckDBPyConnection, tabela: str, arquivo: Path, layout: Layout,
                      colunas: list[str], where: str | None = None) -> int:
    """Cria `tabela` com `colunas` (texto, sem espacos nas pontas; vazio vira NULL).

    O comprimento de TODAS as linhas e conferido antes do filtro `where`. Retorna as linhas gravadas.
    """
    fatiada = sql_colunas(Path(arquivo), layout, colunas)
    fora = con.execute(f"SELECT count(*) FROM ({_origem(Path(arquivo))}) WHERE length(linha) <> {layout.comprimento}",
                       [str(arquivo)]).fetchone()[0]
    if fora:
        raise LayoutInvalido(f"{fora} linhas de {Path(arquivo).name} nao tem {layout.comprimento} posicoes")
    filtro = f" WHERE {where}" if where else ""
    con.execute(f"CREATE OR REPLACE TABLE {tabela} AS SELECT * FROM ({fatiada}){filtro}", [str(arquivo)])
    return con.execute(f"SELECT count(*) FROM {tabela}").fetchone()[0]


# ======== coleta da POF com registro de aquisicao ========
# Arquivo presente com o SHA-256 do manifesto vira cache (nenhuma chamada de rede); divergencia interrompe.

RAIZ_REPO = Path(__file__).resolve().parents[1]
MANIFESTO_POF = RAIZ_REPO / "configs" / "manifesto_fontes.json"


def _agora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _registrar(raiz: Path, fonte_id: str, registro: dict) -> dict:
    pasta = Path(raiz) / "aquisicoes"
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / f"{fonte_id}.json").write_text(json.dumps(registro, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return registro


def coletar_pof(raiz: Path) -> dict:
    """POF 2008-2009 e 2017-2018 pelo manifesto selado, com a mesma coleta do nucleo historico."""
    manifesto = carregar_manifesto(MANIFESTO_POF)
    resultados = coletar(manifesto, raiz)
    extraidos = [p for f in manifesto.fontes for p in extrair_membros(f, raiz)]
    return _registrar(raiz, "pof", {
        "fonte": "pof", "estados": {r.fonte_id: r.status for r in resultados},
        "arquivos": sorted(Path(p).relative_to(raiz).as_posix() for p in extraidos), "registrado_em": _agora(),
    })


# ======== leitura de planilhas xlsx (sem dependencia nova) ========
# A documentacao da POF 2017-2018 traz a classificacao oficial dos grupos de consumo alimentar em .xlsx; o leitor abaixo
# usa so zipfile e ElementTree (strings compartilhadas, strings inline e valores), suficiente para planilhas de texto.

_NS_XLSX = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_REL_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


def _texto_xlsx(no) -> str:
    return "".join(t.text or "" for t in no.iter(f"{{{_NS_XLSX['m']}}}t"))


def _valor_xlsx(celula, compartilhadas: list[str]) -> str:
    tipo, valor = celula.get("t"), celula.find("m:v", _NS_XLSX)
    if tipo == "inlineStr":
        return _texto_xlsx(celula)
    if valor is None or valor.text is None:
        return ""
    return compartilhadas[int(valor.text)] if tipo == "s" else valor.text


def _caminhos_das_folhas(z: zipfile.ZipFile, livro) -> dict[str, str]:
    """Nome da aba -> caminho da folha, pelas relacoes do livro (ou pela ordem, sem o arquivo de relacoes)."""
    folhas = livro.find("m:sheets", _NS_XLSX)
    relacoes = "xl/_rels/workbook.xml.rels"
    if relacoes not in z.namelist():
        return {f.get("name"): f"xl/worksheets/sheet{i}.xml" for i, f in enumerate(folhas, 1)}
    alvos = {r.get("Id"): r.get("Target") for r in ElementTree.fromstring(z.read(relacoes))}
    return {f.get("name"): "xl/" + alvos[f.get(_REL_ID)].lstrip("/").removeprefix("xl/") for f in folhas}


def ler_xlsx(conteudo: bytes) -> dict[str, list[dict[str, str]]]:
    """{aba: [linha como {coluna: texto}]} na ordem da planilha; celulas vazias ficam fora do dicionario da linha."""
    with zipfile.ZipFile(io.BytesIO(conteudo)) as z:
        nomes = z.namelist()
        compartilhadas = ([_texto_xlsx(si) for si in ElementTree.fromstring(z.read("xl/sharedStrings.xml"))
                           .findall("m:si", _NS_XLSX)] if "xl/sharedStrings.xml" in nomes else [])
        livro = ElementTree.fromstring(z.read("xl/workbook.xml"))
        saida = {}
        for aba, caminho in _caminhos_das_folhas(z, livro).items():
            linhas = []
            for linha in ElementTree.fromstring(z.read(caminho)).iter(f"{{{_NS_XLSX['m']}}}row"):
                valores = {re.match(r"[A-Z]+", c.get("r")).group(): _valor_xlsx(c, compartilhadas)
                           for c in linha.findall("m:c", _NS_XLSX)}
                linhas.append({k: v for k, v in valores.items() if v != ""})
            saida[aba] = linhas
    return saida


# ======== linha de comando ========

FONTES = ("pof",)


def coletar_tudo(raiz: Path, fontes: tuple[str, ...] = FONTES) -> dict:
    """Coleta as fontes pedidas e devolve o registro de aquisicao de cada uma."""
    acoes = {"pof": lambda: coletar_pof(raiz)}
    return {fonte: acoes[fonte]() for fonte in fontes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Coleta as fontes do Case Ambev em CASE_AMBEV_RAW_ROOT, com cache, SHA-256 e registro de aquisicao.")
    parser.add_argument("fontes", nargs="*", metavar="fonte", help=f"uma ou mais de {', '.join(FONTES)} (padrao: todas)")
    args = parser.parse_args(argv)
    desconhecidas = sorted(set(args.fontes) - set(FONTES))
    if desconhecidas:
        parser.error(f"fontes desconhecidas: {desconhecidas}")
    resumo = coletar_tudo(raiz_bruta(), tuple(args.fontes) or FONTES)
    sys.stdout.write(json.dumps({f: r["estados"] for f, r in resumo.items()}, ensure_ascii=False, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
