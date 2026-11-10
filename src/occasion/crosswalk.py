"""K-PREP-03: correspondencia de nomes, codigos e unidades entre a POF 2008-2009 e a 2017-2018.

A tabela `configs/crosswalk.json` registra, por conceito, a variavel de cada edicao, a unidade e se o
conceito e comum. Aqui ficam os mapeamentos testados, a conferencia da tabela contra os dicionarios
oficiais, as conferencias de chave unica e de contagem de episodios nos joins, e a harmonizacao das
features comuns. Nenhum rotulo de alvo e lido ou construido.
"""

import csv
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path

import duckdb
import xlrd

from occasion.fontes import EDICOES, sha256_arquivo
from occasion.microdados import Variavel
from occasion.verificacao import UFS_BRASIL

COLUNAS_FEATURES_COMUNS = (
    "episodio_id", "edicao", "dia_coleta", "dia_semana", "hora", "uf", "grande_regiao", "urbano_rural", "area_comparavel",
    "idade", "sexo", "escolaridade", "pct_urbano_uf", "pct_pop_0_14_uf", "pct_pop_15_19_uf",
    "pct_domicilios_com_computador_com_internet_uf",
)
_COLUNAS_CONTEXTO = COLUNAS_FEATURES_COMUNS[-4:]
_TIPOS_ESTRATO = ("capital", "resto_rm", "resto_uf", "rural")  # ordem das colunas da planilha oficial
_AREA = {"capital": "capital", "resto_rm": "urbano_nao_capital", "resto_uf": "urbano_nao_capital", "rural": "rural"}
_QUADROS = {"2008-2009": {"71": 1, "72": 2}, "2017-2018": {"72": 1, "73": 2}}
_DIAS = {"segunda-feira": 1, "terca-feira": 2, "quarta-feira": 3, "quinta-feira": 4, "sexta-feira": 5, "sabado": 6, "domingo": 7}
_REGIOES = {"1": "Norte", "2": "Nordeste", "3": "Sudeste", "4": "Sul", "5": "Centro-Oeste"}
_IDADE_MAXIMA = 111  # dicionario de 2017: "valores inteiros de 0 a 111"


class CrosswalkInvalido(ValueError):
    """A tabela de correspondencia ou um dos mapeamentos nao e valido para os dados."""


class ChaveDuplicada(ValueError):
    """Chave que deveria ser unica aparece mais de uma vez (o join multiplicaria episodios)."""


class JuncaoAlteraLinhas(ValueError):
    """O numero de episodios mudou depois de um join."""


# --- mapeamentos de codigo ------------------------------------------------------------------------
def _sem_acento(texto: str) -> str:
    return unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode("ascii").strip().lower()


def _inteiro(valor, minimo: int, maximo: int, rotulo: str) -> int:
    texto = str(valor).strip()
    if not re.fullmatch(r"\d+", texto) or not minimo <= int(texto) <= maximo:
        raise ValueError(f"{rotulo} invalido: {valor!r}")
    return int(texto)


def quadro_para_dia_coleta(edicao: str, quadro) -> int:
    """Primeiro e segundo dia de coleta: 71/72 em 2008-2009 e 72/73 em 2017-2018."""
    try:
        return _QUADROS[edicao][str(quadro).strip()]
    except KeyError:
        raise ValueError(f"quadro {quadro!r} fora do dominio de {edicao}") from None


def dia_semana_numero(edicao: str, valor) -> int:
    """1 = segunda ... 7 = domingo. 2008: codigo numerico; 2017: nome (os microdados vem sem acento)."""
    if edicao == "2008-2009":
        return _inteiro(valor, 1, 7, "dia da semana")
    try:
        return _DIAS[_sem_acento(valor)]
    except KeyError:
        raise ValueError(f"dia da semana invalido: {valor!r}") from None


def hora_inteira(edicao: str, valor) -> int:
    """Hora inteira 0..23. 2008: '09'; 2017: '9H' ou '09H'."""
    texto = str(valor).strip()
    if edicao == "2017-2018":
        achado = re.fullmatch(r"(\d{1,2})H", texto)
        if not achado:
            raise ValueError(f"hora invalida em {edicao}: {valor!r}")
        texto = achado.group(1)
    return _inteiro(texto, 0, 23, "hora")


def sql_hora(edicao: str, coluna: str) -> str:
    """Expressao SQL equivalente a `hora_inteira` (para agregar episodios na propria consulta)."""
    return f"CAST(replace({coluna}, 'H', '') AS INT)" if edicao == "2017-2018" else f"CAST({coluna} AS INT)"


def uf_padrao(edicao: str, valor) -> str:
    """Codigo IBGE de 2 digitos (igual nas duas edicoes: COD_UF em 2008, UF em 2017)."""
    uf = str(valor).strip().zfill(2)
    if uf not in UFS_BRASIL:
        raise ValueError(f"UF invalida em {edicao}: {valor!r}")
    return uf


def grande_regiao(uf: str) -> str:
    return _REGIOES[uf_padrao("", uf)[0]]


def _codigo(valor, tabela: Mapping[str, str], rotulo: str) -> str:
    try:
        return tabela[str(valor).strip()]
    except KeyError:
        raise ValueError(f"{rotulo} invalido: {valor!r}") from None


def sexo_padrao(edicao: str, codigo) -> str:  # V0405 (2008) e V0404 (2017): 1 e 2 nas duas
    return _codigo(codigo, {"1": "masculino", "2": "feminino"}, f"sexo em {edicao}")


def urbano_rural(edicao: str, codigo) -> str:  # TIPO_SITUACAO_REG: 1 urbano, 2 rural
    return _codigo(codigo, {"1": "urbano", "2": "rural"}, f"situacao do domicilio em {edicao}")


def escolaridade_nivel(edicao: str, codigo) -> int:  # 7 niveis com os mesmos rotulos nas duas edicoes
    return _inteiro(codigo, 1, 7, f"nivel de instrucao em {edicao}")


def idade_anos(edicao: str, valor) -> int:
    return _inteiro(valor, 0, _IDADE_MAXIMA, f"idade em {edicao}")


def area_comparavel(tipo: str) -> str:
    """Capital / urbano nao capital / rural: o limite da RM mudou entre as edicoes, entao resto da RM e resto da UF se unem."""
    return _AREA[tipo]


# --- estratos geograficos (planilhas oficiais) ----------------------------------------------------
def expandir_codigos(valor) -> list[int]:
    """Expande celulas como '1 e 2', '3 a 6', 1.0, '1       a       7', '-' e vazio."""
    if isinstance(valor, float):
        return [int(valor)]
    texto = str(valor).strip()
    if texto in ("", "-"):
        return []
    codigos: list[int] = []
    for pedaco in re.split(r"\s*(?:,|\be\b)\s*", texto):
        intervalo = re.fullmatch(r"(\d+)\s+a\s+(\d+)", pedaco.strip())
        if intervalo:
            codigos += range(int(intervalo.group(1)), int(intervalo.group(2)) + 1)
        elif re.fullmatch(r"\d+", pedaco.strip()):
            codigos.append(int(pedaco))
        else:
            raise ValueError(f"celula de estrato nao reconhecida: {valor!r}")
    return codigos


def ler_planilha_estratos(caminho: Path) -> list[list]:
    folha = xlrd.open_workbook(str(caminho)).sheet_by_index(0)
    return [[folha.cell_value(r, c) for c in range(folha.ncols)] for r in range(folha.nrows)]


def nomes_uf_do_dicionario(variavel_uf: Variavel) -> dict[str, str]:
    """{'rondonia': '11', ...} a partir das categorias '11 – Rondônia' do dicionario."""
    nomes = {_sem_acento(c.split("–", 1)[1]): c.split("–", 1)[0].strip() for c in variavel_uf.categorias if "–" in c}
    if len(nomes) != len(UFS_BRASIL):
        raise CrosswalkInvalido(f"dicionario traz {len(nomes)} UFs; esperado {len(UFS_BRASIL)}")
    return nomes


def decodificar_estratos(linhas: list[list], nomes_uf: Mapping[str, str], edicao: str) -> dict[tuple[str, int], str]:
    """{(uf, codigo_do_estrato): tipo}. Em 2008 o codigo (2 digitos) se repete entre UFs; em 2017 ele leva o prefixo da UF."""
    inicio = next((i + 1 for i, lin in enumerate(linhas) if any(_sem_acento(v) == "da uf" for v in lin)), None)
    if inicio is None:
        raise ValueError("planilha de estratos sem o cabecalho 'da UF'")
    estratos: dict[tuple[str, int], str] = {}
    for lin in linhas[inicio:]:
        nome = _sem_acento(lin[0])
        if not nome or nome.startswith("regiao"):
            continue
        if nome not in nomes_uf:
            raise ValueError(f"UF desconhecida na planilha de estratos: {lin[0]!r}")
        uf = nomes_uf[nome]
        for tipo, celula in zip(_TIPOS_ESTRATO, lin[1:5]):
            for codigo in expandir_codigos(celula):
                if edicao == "2017-2018" and codigo // 100 != int(uf):
                    raise ValueError(f"estrato {codigo} com prefixo diferente da UF {uf} ({lin[0]})")
                if (uf, codigo) in estratos and estratos[(uf, codigo)] != tipo:
                    raise ValueError(f"estrato {codigo} da UF {uf} em mais de um tipo")
                estratos[(uf, codigo)] = tipo
    return estratos


# --- tabela de correspondencia --------------------------------------------------------------------
def carregar_crosswalk(caminho: Path) -> dict:
    tabela = json.loads(Path(caminho).read_text(encoding="utf-8"))
    colunas = [c["coluna_comum"] for c in tabela["conceitos"] if c["entra_em_features_comuns"]]
    esperado = ["episodio_id", "edicao", *colunas, *tabela["contexto_uf"]["colunas"]]
    if list(COLUNAS_FEATURES_COMUNS) != esperado:
        raise CrosswalkInvalido(f"COLUNAS_FEATURES_COMUNS difere da tabela: {esperado}")
    return tabela


def validar_contra_dicionarios(tabela: Mapping, dicionarios: Mapping[str, Mapping[str, list[Variavel]]]) -> int:
    """Confere nome, posicao, tamanho e (quando declaradas) categorias de cada variavel contra o dicionario oficial."""
    conferidas = 0
    for conceito in tabela["conceitos"]:
        for edicao, entrada in conceito["edicoes"].items():
            for v in entrada.get("variaveis", []):
                no_dicionario = {x.codigo: x for x in dicionarios[edicao].get(v["registro"], [])}
                if v["variavel"] not in no_dicionario:
                    raise CrosswalkInvalido(f"{conceito['conceito']}: {v['variavel']} ausente em {v['registro']} de {edicao}")
                real = no_dicionario[v["variavel"]]
                if (real.posicao, real.tamanho) != (v["posicao"], v["tamanho"]):
                    raise CrosswalkInvalido(
                        f"{conceito['conceito']}: posicao/tamanho de {v['variavel']} em {edicao} e {(real.posicao, real.tamanho)}, "
                        f"tabela diz {(v['posicao'], v['tamanho'])}")
                if "categorias" in v and tuple(v["categorias"]) != real.categorias:
                    raise CrosswalkInvalido(f"{conceito['conceito']}: categorias de {v['variavel']} em {edicao} diferem da tabela")
                conferidas += 1
    return conferidas


# --- chaves unicas e contagens nos joins ----------------------------------------------------------
def conferir_chave_unica(con: duckdb.DuckDBPyConnection, tabela: str, colunas: list[str]) -> dict:
    chave = ", ".join(colunas)
    linhas, distintas = con.execute(f"SELECT count(*), count(DISTINCT ({chave})) FROM {tabela}").fetchone()
    if linhas != distintas:
        raise ChaveDuplicada(f"{tabela}: {linhas - distintas} chave(s) duplicada(s) em ({chave})")
    return {"linhas": linhas, "chaves_distintas": distintas}


def registrar_juncao(registro: dict, nome: str, antes: int, depois: int) -> dict:
    if antes != depois:
        raise JuncaoAlteraLinhas(f"{nome}: {antes} episodios antes do join e {depois} depois")
    registro[nome] = {"episodios_antes": antes, "episodios_depois": depois}
    return registro


def ler_contexto_uf(caminho: Path, sha256_esperado: str) -> list[dict]:
    if sha256_arquivo(caminho) != sha256_esperado:
        raise CrosswalkInvalido(f"{Path(caminho).name} diverge do SHA-256 registrado em crosswalk.json")
    with open(caminho, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def conferir_contexto_uf(linhas: Iterable[Mapping], ufs_esperadas: Iterable[str]) -> int:
    """Chave edicao x UF unica e completa (2 edicoes x UFs esperadas = 54 no Brasil)."""
    chaves = [(r["edicao_pof"], r["uf"]) for r in linhas]
    if len(chaves) != len(set(chaves)):
        raise ChaveDuplicada(f"contexto_uf: {len(chaves) - len(set(chaves))} chave(s) edicao x UF duplicada(s)")
    esperado = {(e, uf) for e in EDICOES for uf in ufs_esperadas}
    if set(chaves) != esperado:
        raise CrosswalkInvalido(f"contexto_uf: faltam {sorted(esperado - set(chaves))}, sobram {sorted(set(chaves) - esperado)}")
    return len(chaves)


# --- harmonizacao das features comuns -------------------------------------------------------------
_FONTES = {  # edicao -> (tabela de episodios, colunas de origem na ordem usada por _linha)
    "2008-2009": ("e2_2008", "2008", ("uf", "estrato_pof", "tipo_situacao", "dia_semana", "quadro", "hora", "idade_anos", "sexo",
                                        "nivel_instrucao")),
    "2017-2018": ("x_2017", "2017", ("uf", "estrato_pof", "tipo_situacao_reg", "dia_semana", "quadro", "hora", "idade", "sexo",
                                       "nivel_instrucao")),
}


def _linha(edicao: str, episodio_id: str, estratos: Mapping, valores: tuple) -> tuple:
    uf, estrato, situacao, dia_semana, quadro, hora, idade, sexo, nivel = valores
    chave = (uf_padrao(edicao, uf), int(estrato))
    if chave not in estratos:
        raise ValueError(f"estrato {chave} do episodio {episodio_id} sem correspondencia na planilha de {edicao}")
    if not 0 <= hora <= 23:
        raise ValueError(f"hora fora do dominio no episodio {episodio_id}")
    return (episodio_id, edicao, quadro_para_dia_coleta(edicao, quadro), dia_semana_numero(edicao, dia_semana), hora,
            chave[0], grande_regiao(chave[0]), urbano_rural(edicao, situacao), area_comparavel(estratos[chave]),
            idade_anos(edicao, idade), sexo_padrao(edicao, sexo), escolaridade_nivel(edicao, nivel))


def harmonizar_features(con: duckdb.DuckDBPyConnection, estratos: Mapping[str, Mapping], contexto_uf: list[dict]) -> dict:
    """Cria features_comuns_2008 e features_comuns_2017 (mesmas colunas) e confere episodios antes e depois do join."""
    contexto = [(r["edicao_pof"], r["uf"], *(float(r[c]) for c in _COLUNAS_CONTEXTO)) for r in contexto_uf]
    con.execute(f"CREATE OR REPLACE TABLE contexto_uf(edicao_pof VARCHAR, uf VARCHAR, "
                f"{', '.join(c + ' DOUBLE' for c in _COLUNAS_CONTEXTO)})")
    con.executemany(f"INSERT INTO contexto_uf VALUES (?, ?, {', '.join('?' * len(_COLUNAS_CONTEXTO))})", contexto)
    juncoes: dict = {"chaves_unicas": {"contexto_uf": conferir_chave_unica(con, "contexto_uf", ["edicao_pof", "uf"])}}
    base = COLUNAS_FEATURES_COMUNS[:12]
    tipos = ["VARCHAR", "VARCHAR", "INT", "INT", "INT", "VARCHAR", "VARCHAR", "VARCHAR", "VARCHAR", "INT", "VARCHAR", "INT"]
    for edicao, (tabela, ano, colunas) in _FONTES.items():
        linhas = con.execute(f"SELECT episodio_id, {', '.join(colunas)} FROM {tabela} ORDER BY episodio_id").fetchall()
        con.execute(f"CREATE OR REPLACE TABLE feat_base_{ano}({', '.join(f'{c} {t}' for c, t in zip(base, tipos))})")
        con.executemany(f"INSERT INTO feat_base_{ano} VALUES ({', '.join('?' * len(base))})",
                        [_linha(edicao, r[0], estratos[edicao], r[1:]) for r in linhas])
        con.execute(f"""CREATE OR REPLACE TABLE features_comuns_{ano} AS
            SELECT b.*, {', '.join('c.' + c for c in _COLUNAS_CONTEXTO)}
            FROM feat_base_{ano} b LEFT JOIN contexto_uf c ON c.edicao_pof = b.edicao AND c.uf = b.uf""")
        sem_contexto = con.execute(f"SELECT count(*) FROM features_comuns_{ano} WHERE {_COLUNAS_CONTEXTO[0]} IS NULL").fetchone()[0]
        if sem_contexto:
            raise CrosswalkInvalido(f"{sem_contexto} episodios de {edicao} sem contexto_uf")
        depois = con.execute(f"SELECT count(*) FROM features_comuns_{ano}").fetchone()[0]
        registrar_juncao(juncoes, f"features_comuns_{ano}", len(linhas), depois)
    return juncoes
