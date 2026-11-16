"""Caminho de 2017-2018: elegibilidade (pizza), features brutas e conjunto selado. Nenhum rotulo e construido.

Este modulo e o unico que o desenvolvimento de 2017 usa, e por isso nao importa o de 2008 (onde estao as
listas de bebidas). Saidas: x_2017 (features, sem peso) vai para a pasta de trabalho; ids_pesos_2017
(identificadores e pesos) vai para a pasta selada, marcada como somente leitura e fora da arvore de trabalho.

O selo (chmod 0444/0555) protege contra ESCRITA, nao contra LEITURA: o arquivo continua legivel por qualquer processo
do mesmo usuario. O controle de acesso e o fluxo da avaliacao temporal (`occasion.avaliacao_temporal`), nao o chmod.
"""

import os
import shutil
import tempfile
from pathlib import Path

import duckdb

from occasion.crosswalk import conferir_chave_unica, registrar_juncao, sql_hora
from occasion.fontes import sha256_arquivo
from occasion.verificacao import escrever_csv

EDICAO = "2017-2018"
# Pizza estrita (auditoria_pof/scripts/listas_codigos.py); 2008-2009 acrescenta versoes light (pof2008_comparavel.py).
PIZZA_ESTRITA = (
    "8500903", "8500914", "8500915", "8500916", "8500917", "8500924", "8500927", "8500928", "8500929",
    "8500930", "8500932", "8500934", "8500935", "8500936", "8500937", "8500939", "8508017",
)
PIZZA_EXTRA_2008 = ("8508003", "8508014", "8508015", "8508016", "8508017")
COLUNAS_X_2017 = (
    "episodio_id", "edicao", "uf", "cod_upa", "num_dom", "num_uc", "cod_informante", "quadro", "hora", "estrato_pof",
    "tipo_situacao_reg", "dia_semana", "dia_atipico", "ocasiao_v9017", "local_v9018", "idade", "sexo", "nivel_instrucao", "renda_total",
)
COLUNAS_IDS_PESOS_2017 = ("episodio_id", "edicao", "cod_upa", "num_dom", "peso_final")
ARQ_FEATURES, ARQ_SELADO = "x_2017.csv", "ids_pesos_2017.csv"
_MODO_ARQUIVO, _MODO_PASTA = 0o444, 0o555


def pizza(edicao: str) -> tuple[str, ...]:
    if edicao == "2017-2018":
        return PIZZA_ESTRITA
    if edicao == "2008-2009":
        return tuple(sorted(set(PIZZA_ESTRITA) | set(PIZZA_EXTRA_2008)))
    raise ValueError(f"edicao desconhecida: {edicao!r}")


def construir_e2_pizza_2017(con: duckdb.DuckDBPyConnection) -> dict:
    """Cria x_2017 e ids_pesos_2017 a partir de consumo_2017_pizza (so linhas de pizza) e morador_2017."""
    con.execute(f"""
        CREATE OR REPLACE TABLE ep_2017 AS
        SELECT UF uf, COD_UPA cod_upa, CAST(NUM_DOM AS INT) num_dom, CAST(NUM_UC AS INT) num_uc,
               CAST(COD_INFORMANTE AS INT) cod_informante, CAST(QUADRO AS INT) quadro,
               {sql_hora(EDICAO, 'V9015')} hora, min(ESTRATO_POF) estrato_pof,
               min(TIPO_SITUACAO_REG) tipo_situacao_reg, min(DIA_SEMANA) dia_semana, min(DIA_ATIPICO) dia_atipico,
               min(V9017) ocasiao_v9017, min(V9018) local_v9018, min(CAST(RENDA_TOTAL AS DOUBLE)) renda_total,
               min(CAST(PESO_FINAL AS DOUBLE)) peso_final
        FROM consumo_2017_pizza WHERE NOT (V9005 IS NULL AND V9007 IS NULL) GROUP BY ALL""")
    con.execute("""
        CREATE OR REPLACE TABLE morador_2017_norm AS
        SELECT COD_UPA cod_upa, CAST(NUM_DOM AS INT) num_dom, CAST(NUM_UC AS INT) num_uc,
               CAST(COD_INFORMANTE AS INT) cod_informante, CAST(V0403 AS INT) idade, V0404 sexo,
               NIVEL_INSTRUCAO nivel_instrucao
        FROM morador_2017""")
    juncoes = {"chaves_unicas": {
        "morador_2017": conferir_chave_unica(con, "morador_2017_norm", ["cod_upa", "num_dom", "num_uc", "cod_informante"])}}
    antes = con.execute("SELECT count(*) FROM ep_2017").fetchone()[0]
    con.execute("""
        CREATE OR REPLACE TABLE ep_2017_pessoa AS
        SELECT e.*, m.idade, m.sexo, m.nivel_instrucao
        FROM ep_2017 e LEFT JOIN morador_2017_norm m USING (cod_upa, num_dom, num_uc, cod_informante)""")
    registrar_juncao(juncoes, "episodios_com_morador_2017", antes,
                     con.execute("SELECT count(*) FROM ep_2017_pessoa").fetchone()[0])
    sem = con.execute("SELECT count(*) FROM ep_2017_pessoa WHERE idade IS NULL").fetchone()[0]
    if sem:
        raise ValueError(f"{sem} episodios de 2017-2018 sem correspondencia em MORADOR")
    con.execute(f"""
        CREATE OR REPLACE TABLE base_2017 AS
        SELECT format('{EDICAO}|{{}}|{{}}|{{}}|{{}}|{{}}|{{}}', cod_upa, lpad(num_dom::VARCHAR, 2, '0'), num_uc,
                      lpad(cod_informante::VARCHAR, 2, '0'), quadro, lpad(hora::VARCHAR, 2, '0')) episodio_id,
               '{EDICAO}' edicao, * FROM ep_2017_pessoa""")
    con.execute(f"CREATE OR REPLACE TABLE x_2017 AS SELECT {', '.join(COLUNAS_X_2017)} FROM base_2017")
    con.execute(f"CREATE OR REPLACE TABLE ids_pesos_2017 AS SELECT {', '.join(COLUNAS_IDS_PESOS_2017)} FROM base_2017")
    return {"episodios_com_pizza": con.execute("SELECT count(*) FROM x_2017").fetchone()[0], "juncoes": juncoes}


# --- conjunto selado ------------------------------------------------------------------------------
class SeloNaoAplicado(RuntimeError):
    """O sistema de arquivos ignorou o chmod: o conjunto selado continuaria gravavel."""


class PastasSobrepostas(ValueError):
    """Pasta de trabalho e pasta selada iguais, aninhadas ou em link simbolico: o selo atingiria o trabalho."""


def _conferir_pastas(trabalho: Path, selado: Path) -> None:
    for nome, pasta in (("trabalho", trabalho), ("selado", selado)):
        if pasta.is_symlink():
            raise PastasSobrepostas(f"pasta {nome} e um link simbolico: chmod, copia e conferencia seguiriam o alvo do link")
    real_t, real_s = trabalho.resolve(), selado.resolve()
    if real_t == real_s or real_s.is_relative_to(real_t) or real_t.is_relative_to(real_s):
        raise PastasSobrepostas("pasta de trabalho e pasta selada devem ser distintas e sem aninhamento: o selo "
                                "tornaria a pasta de trabalho somente leitura")


def estado_do_selo(selado: Path) -> dict:
    """O que o selo garante hoje. Serve para registrar, nunca para decidir acesso: leitura nao e bloqueada."""
    selado = Path(selado)
    arquivo = selado / ARQ_SELADO
    protegido = arquivo.exists() and not (os.access(arquivo, os.W_OK) or os.access(selado, os.W_OK))
    return {"protege_escrita": protegido, "bloqueia_leitura": False,
            "legivel_por_este_processo": arquivo.exists() and os.access(arquivo, os.R_OK),
            "observacao": ("chmod 0444 nao impede leitura; o controle de acesso e o fluxo da avaliacao temporal "
                           "(autorizacao, manifesto, controles), nao o modo do arquivo")}


def _selar(arquivo: Path, pasta: Path) -> None:
    os.chmod(arquivo, _MODO_ARQUIVO)
    os.chmod(pasta, _MODO_PASTA)
    if os.access(arquivo, os.W_OK) or os.access(pasta, os.W_OK):
        raise SeloNaoAplicado(
            f"{pasta} continua gravavel apos o chmod (sistema de arquivos sem permissoes POSIX, ou execucao como root). "
            "Aponte CASE_AMBEV_SELADO_ROOT para um sistema de arquivos que respeite chmod (ex.: ext4 no WSL).")


def _gravar_selado(con: duckdb.DuckDBPyConnection, selado: Path) -> str:
    """Grava o conjunto selado; o selo e (re)aplicado e conferido SEMPRE, inclusive quando o hash ja coincide
    (cobre uma execucao anterior que falhou entre a gravacao e a selagem)."""
    with tempfile.TemporaryDirectory() as tmp:
        novo = escrever_csv(con, "ids_pesos_2017", Path(tmp) / ARQ_SELADO, "episodio_id")
        hash_novo, destino = sha256_arquivo(novo), selado / ARQ_SELADO
        if destino.exists():
            if sha256_arquivo(destino) != hash_novo:
                raise FileExistsError(f"{destino} ja esta selado com conteudo diferente; nao sera sobrescrito")
        else:
            selado.mkdir(parents=True, exist_ok=True)
            parcial = destino.with_name(f"{destino.name}.{os.getpid()}.parcial")  # nome por processo
            try:
                shutil.copyfile(novo, parcial)
                os.replace(parcial, destino)
            except BaseException:
                parcial.unlink(missing_ok=True)
                raise
        _selar(destino, selado)
    return hash_novo


def _gravar_features(con: duckdb.DuckDBPyConnection, trabalho: Path, substituir: bool) -> str:
    """Grava x_2017.csv sem sobrescrever em silencio um arquivo existente com conteudo diferente."""
    destino = trabalho / ARQ_FEATURES
    with tempfile.TemporaryDirectory() as tmp:
        novo = escrever_csv(con, "x_2017", Path(tmp) / ARQ_FEATURES, "episodio_id")
        hash_novo = sha256_arquivo(novo)
        if destino.exists() and sha256_arquivo(destino) != hash_novo:
            if not substituir:
                raise FileExistsError(f"{destino} ja existe com conteudo diferente; nao sera sobrescrito "
                                      "(remova o arquivo ou use substituir_features=True de forma deliberada)")
        if not destino.exists() or sha256_arquivo(destino) != hash_novo:
            trabalho.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(novo, destino)
    return hash_novo


def gravar_conjuntos_2017(con: duckdb.DuckDBPyConnection, trabalho: Path, selado: Path,
                          substituir_features: bool = False) -> dict[str, str]:
    """Grava x_2017.csv em `trabalho` e ids_pesos_2017.csv em `selado`; devolve {arquivo: sha256}.

    Recusa pastas iguais, aninhadas ou em link simbolico, e conteudo diferente do ja gravado em qualquer dos dois
    arquivos (o das features so se `substituir_features` for pedido de forma explicita)."""
    trabalho, selado = Path(trabalho), Path(selado)
    _conferir_pastas(trabalho, selado)
    return {ARQ_FEATURES: _gravar_features(con, trabalho, substituir_features), ARQ_SELADO: _gravar_selado(con, selado)}
