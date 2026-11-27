"""Padronizacao, validacao, producao das tabelas de escolhas da POF e auditoria das refeicoes.

Reune o contrato das entradas da aplicacao (campos e dominios, usado igual no treino e na inferencia), a validacao
comum (`EntradaInvalida`, `conferir_campos`, `conferir_artefatos`), a producao das tabelas em Parquet e a auditoria que
reconstroi as refeicoes dos registros brutos ate a tabela do alvo (cobertura, ausencias e sensibilidade do alvo). A
harmonizacao da POF (crosswalk e regras de variaveis) e as listas de codigos vieram de `occasion/` em 24/11/2026.
"""

import argparse
import hashlib
import json
import os
import re
import reprlib
import sys
import unicodedata
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from functools import cache
from pathlib import Path
from types import MappingProxyType

import duckdb
import sklearn
import xlrd

import coleta
from coleta import COLUNAS as COLUNAS_POF
from coleta import EDICOES, ORIGEM_OFICIAL, Variavel, carregar_registro, layout_de, ler_dicionario_xls, raiz_bruta

# ======== origem: occasion (crosswalk, variaveis, episodios_2008, episodios_2017, verificacao) ========
# So as definicoes que o produto usa, movidas com o texto original em 24/11/2026 (renomes registrados no protocolo);
# o restante do pacote e a execucao
# historica completa (com o manifesto congelado) fica na copia verificada fora do repositorio.

# ---------- origem: occasion/verificacao.py ----------

UFS_BRASIL = (
    "11", "12", "13", "14", "15", "16", "17", "21", "22", "23", "24", "25", "26", "27", "28", "29",
    "31", "32", "33", "35", "41", "42", "43", "50", "51", "52", "53",
)


# ---------- origem: occasion/variaveis.py ----------

CONFIG_VARIAVEIS = Path(__file__).resolve().parents[1] / "configs" / "variaveis.json"


class RegrasInvalidas(ValueError):
    """`configs/variaveis.json` nao e coerente com a especificacao de 13 variaveis e 28 colunas."""


def _congelar(valor):
    if isinstance(valor, dict):
        return MappingProxyType({k: _congelar(v) for k, v in valor.items()})
    if isinstance(valor, list):
        return tuple(_congelar(v) for v in valor)
    return valor


def _conferir_cobertura(regras: Mapping) -> None:
    horas = [h for hs in regras["faixas_hora"].values() for h in hs]
    niveis = [n for ns in regras["escolaridade_3"].values() for n in ns]
    if sorted(horas) != list(range(24)) or sorted(niveis) != list(range(1, 8)):
        raise RegrasInvalidas("faixas de hora (0..23) e de escolaridade (1..7) devem cobrir o dominio sem sobreposicao")
    faixas = regras["faixas_etarias"]
    contiguas = all(a["max"] is not None and b["min"] == a["max"] + 1 for a, b in zip(faixas, faixas[1:]))
    if not contiguas or faixas[-1]["max"] is not None or faixas[0]["min"] != 10:
        raise RegrasInvalidas("faixas etarias devem ser contiguas, de 10 anos ate o limite superior aberto")


@cache
def carregar_regras(caminho: Path = CONFIG_VARIAVEIS) -> Mapping:
    """Regras do JSON, imutaveis (o resultado e compartilhado entre chamadas)."""
    regras = json.loads(Path(caminho).read_text(encoding="utf-8"))
    colunas = regras["colunas_logistica"]
    declaradas = [c for var in regras["variaveis"] for c in var["colunas"]]
    if len(regras["variaveis"]) != 13 or len(colunas) != 28 or sorted(declaradas) != sorted(colunas):
        raise RegrasInvalidas("esperado 13 variaveis e 28 colunas coerentes com as variaveis declaradas")
    _conferir_cobertura(regras)
    return _congelar(regras)


# --- validacao -----------------------------------------------------------------------------------
def _inteiro_regra(valor, rotulo: str) -> int:
    if isinstance(valor, bool) or not isinstance(valor, int):
        raise ValueError(f"{rotulo} invalido: {valor!r}")
    return valor


def _dentro(idade: int, minimo: int, maximo: int | None) -> bool:
    return idade >= minimo and (maximo is None or idade <= maximo)


# --- regras por variavel -------------------------------------------------------------------------
def faixa_hora(hora) -> str:
    hora = _inteiro_regra(hora, "hora")
    for nome, horas in carregar_regras()["faixas_hora"].items():
        if hora in horas:
            return nome
    raise ValueError(f"hora fora de 0..23: {hora}")


def faixa_etaria(idade) -> str | None:
    """None para idade desconhecida ou abaixo de 10 anos (fora da populacao do consumo individual)."""
    if idade is None:
        return None
    idade = _inteiro_regra(idade, "idade")
    if idade < 0:
        raise ValueError(f"idade negativa: {idade}")
    for faixa in carregar_regras()["faixas_etarias"]:
        if _dentro(idade, faixa["min"], faixa["max"]):
            return faixa["nome"]
    return None


def escolaridade_3(nivel) -> str:
    nivel = _inteiro_regra(nivel, "nivel de instrucao")
    for nome, niveis in carregar_regras()["escolaridade_3"].items():
        if nivel in niveis:
            return nome
    raise ValueError(f"nivel de instrucao fora de 1..7: {nivel}")


# ---------- origem: occasion/crosswalk.py ----------

_TIPOS_ESTRATO = ("capital", "resto_rm", "resto_uf", "rural")  # ordem das colunas da planilha oficial


_AREA = {"capital": "capital", "resto_rm": "urbano_nao_capital", "resto_uf": "urbano_nao_capital", "rural": "rural"}


_DIAS = {"segunda-feira": 1, "terca-feira": 2, "quarta-feira": 3, "quinta-feira": 4, "sexta-feira": 5, "sabado": 6, "domingo": 7}


_REGIOES = {"1": "Norte", "2": "Nordeste", "3": "Sudeste", "4": "Sul", "5": "Centro-Oeste"}


_IDADE_MAXIMA = 111  # dicionario de 2017: "valores inteiros de 0 a 111"


class CrosswalkInvalido(ValueError):
    """A tabela de correspondencia ou um dos mapeamentos nao e valido para os dados."""


# --- mapeamentos de codigo ------------------------------------------------------------------------
def _sem_acento(texto: str) -> str:
    return unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode("ascii").strip().lower()


def _inteiro(valor, minimo: int, maximo: int, rotulo: str) -> int:
    texto = str(valor).strip()
    if not re.fullmatch(r"\d+", texto) or not minimo <= int(texto) <= maximo:
        raise ValueError(f"{rotulo} invalido: {valor!r}")
    return int(texto)


def dia_semana_numero(edicao: str, valor) -> int:
    """1 = segunda ... 7 = domingo. 2008: codigo numerico; 2017: nome (os microdados vem sem acento)."""
    if edicao == "2008-2009":
        return _inteiro(valor, 1, 7, "dia da semana")
    try:
        return _DIAS[_sem_acento(valor)]
    except KeyError:
        raise ValueError(f"dia da semana invalido: {valor!r}") from None


def sql_hora(edicao: str, coluna: str) -> str:
    """Expressao SQL da hora inteira do consumo (V9015), para agregar episodios na propria consulta."""
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


# --- chaves unicas e contagens nos joins ----------------------------------------------------------
# ---------- origem: occasion/episodios_2017.py ----------

# Pizza estrita (auditoria_pof/scripts/listas_codigos.py); 2008-2009 acrescenta versoes light (pof2008_comparavel.py).
PIZZA_ESTRITA = (
    "8500903", "8500914", "8500915", "8500916", "8500917", "8500924", "8500927", "8500928", "8500929",
    "8500930", "8500932", "8500934", "8500935", "8500936", "8500937", "8500939", "8508017",
)


PIZZA_EXTRA_2008 = ("8508003", "8508014", "8508015", "8508016", "8508017")


def pizza(edicao: str) -> tuple[str, ...]:
    if edicao == "2017-2018":
        return PIZZA_ESTRITA
    if edicao == "2008-2009":
        return tuple(sorted(set(PIZZA_ESTRITA) | set(PIZZA_EXTRA_2008)))
    raise ValueError(f"edicao desconhecida: {edicao!r}")


# ---------- origem: occasion/episodios_2008.py ----------

# ATENCAO (portao do alvo de 2017, parecer da fase 2, criterio 9): as listas de bebidas ficam NESTE modulo,
# que o caminho de desenvolvimento de 2017 nunca importa. Fonte: auditoria_pof/scripts/listas_codigos.py.
GUARANA = ("8200301", "8200302", "8201601", "8201602", "8207401", "8207402")


REFRIGERANTE = (
    "8200101", "8200102", "8200202", "8200208", "8200211", "8200301", "8200302", "8200407", "8200411", "8200505",
    "8200507", "8201017", "8201802", "8203501", "8200701", "8200702", "8200801", "8201511", "8201601", "8201602",
    "8207401", "8207402", "8207501", "8207601", "8207701", "8201502", "8201902", "8205402", "8209102", "8207705",
)


# ======== origem: aplicacao ========

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


# ======== contrato das entradas da aplicacao ========
CAMPOS = ("dia_semana", "hora", "grande_regiao", "area", "faixa_etaria", "sexo", "escolaridade")
CAMPOS_PUBLICO = ("faixa_etaria", "sexo", "escolaridade")
TODOS = "todos"
_REGRAS = carregar_regras()
PERIODOS = tuple(_REGRAS["faixas_hora"])
REGIOES = tuple(_REGRAS["regioes"])
AREAS = tuple(_REGRAS["areas"])
FAIXAS_ETARIAS = tuple(f["nome"] for f in _REGRAS["faixas_etarias"])
SEXOS = tuple(_REGRAS["sexos"])
ESCOLARIDADES = tuple(_REGRAS["escolaridade_3"])
VALORES_PUBLICO = MappingProxyType({"faixa_etaria": FAIXAS_ETARIAS, "sexo": SEXOS, "escolaridade": ESCOLARIDADES})
def _colunas_dia(representacao: str) -> tuple[str, ...]:
    if representacao == "D7":
        return tuple(f"dia_{d}" for d in range(1, 8))
    if representacao == "D4":
        return ("dia_segunda_a_quinta", "dia_sexta", "dia_sabado", "dia_domingo")
    if representacao == "D2":
        return ("sexta_a_domingo", "sexta_a_domingo_x_noite")
    raise ValueError(f"representacao do dia desconhecida: {representacao!r}")


def colunas(representacao: str) -> tuple[str, ...]:
    """Colunas codificadas na ordem fixa: periodo, dia, regiao, area, faixa etaria, sexo, escolaridade."""
    return (tuple(f"hora_{p}" for p in PERIODOS) + _colunas_dia(representacao)
            + tuple(f"regiao_{r.lower().replace('-', '_')}" for r in REGIOES) + tuple(f"area_{a}" for a in AREAS)
            + tuple(f"idade_{f}" for f in FAIXAS_ETARIAS) + ("sexo_feminino",)
            + tuple(f"escolaridade_{e}" for e in ESCOLARIDADES))


def _categoria(valor: object, campo: str, validos: tuple[str, ...]) -> str:
    if type(valor) is not str or valor not in validos:
        raise EntradaInvalida(f"{campo} deve ser um de {list(validos)}, recebido {recebido(valor)}")
    return valor


def campos_informados(contexto: Mapping[str, object], campos: tuple[str, ...]) -> tuple[str, ...]:
    """Campos esperados com `periodo` no lugar de `hora` quando o contexto traz o periodo; recusa os dois ou nenhum."""
    if ("hora" in contexto) == ("periodo" in contexto):
        raise EntradaInvalida("informe exatamente um entre hora e periodo")
    if "periodo" not in contexto:
        return campos
    return tuple("periodo" if campo == "hora" else campo for campo in campos)


def periodo_informado(contexto: Mapping[str, object]) -> tuple[str, int | None]:
    """Periodo do contexto e a hora, se veio: da `hora` pela regra congelada ou do `periodo` validado."""
    if "periodo" in contexto:
        return _categoria(contexto["periodo"], "periodo", PERIODOS), None
    hora = conferir_inteiro(contexto["hora"], "hora", 0, 23)
    return faixa_hora(hora), hora


def preparar(contexto: Mapping[str, object]) -> dict:
    """Contexto validado na forma canonica (com `periodo`); nao altera a entrada e nao imputa nada."""
    if not isinstance(contexto, Mapping):
        raise EntradaInvalida(f"contexto deve ser um dicionario com os campos {list(CAMPOS)}")
    conferir_campos(contexto, campos_informados(contexto, CAMPOS), "contexto",
                    " (renda, composicao da casa, clima e alvo nao fazem parte desta versao)")
    periodo, hora = periodo_informado(contexto)
    canonico = {"dia_semana": conferir_inteiro(contexto["dia_semana"], "dia_semana", 1, 7), "periodo": periodo,
                "grande_regiao": _categoria(contexto["grande_regiao"], "grande_regiao", REGIOES),
                "area": _categoria(contexto["area"], "area", AREAS)}
    for campo in CAMPOS_PUBLICO:
        canonico[campo] = _categoria(contexto[campo], campo, (*VALORES_PUBLICO[campo], TODOS))
    return canonico if hora is None else {**canonico, "hora": hora}


# ======== cadastro de produtos: o que nao conta como comida ========

GRUPOS_BEBIDA = ("82", "83")  # bebidas nao alcoolicas e alcoolicas no cadastro de 2008
DESCRICOES_BEBIDA_ADICAO = (  # fora dos grupos 82/83: bebidas preparadas e adicoes, pela descricao do cadastro
    r"^CAFE( |$)", r"^CAFE COM LEITE", r"^SUCO( |$)", r"^REFRESCO", r"^VITAMINA( |$)", r"^CHA( |$)",
    r"^LEITE (DE VACA|DE CABRA|INTEGRAL|DESNATADO|SEMIDESNATADO|EM PO|FERMENTADO|AROMATIZADO|DE SOJA)",
    r"^ACUCAR", r"^ADOCANTE", r"^ACHOCOLATADO", r"^XAROPE")


def cadastro_produtos(doc_zip: Path) -> list[tuple[str, str]]:
    """(codigo de 7 digitos, descricao) de cada item do cadastro de produtos do consumo alimentar, na ordem da planilha."""
    with zipfile.ZipFile(doc_zip) as z:
        nome = next(n for n in z.namelist() if "Produtos do Consumo Alimentar" in n)
        folha = xlrd.open_workbook(file_contents=z.read(nome)).sheet_by_index(0)
    pares = ((str(folha.cell_value(r, 0)).split(".")[0].strip(), str(folha.cell_value(r, 1)).strip())
             for r in range(1, folha.nrows))
    return [(codigo, descricao) for codigo, descricao in pares if len(codigo) == 7 and codigo.isdigit()]


def bebida_ou_adicao(doc_zip: Path) -> list[str]:
    """Codigos que nao contam como comida no cadastro da edicao: grupos de bebida e bebida preparada ou adicao."""
    padroes = [re.compile(p) for p in DESCRICOES_BEBIDA_ADICAO]
    return sorted(codigo for codigo, descricao in cadastro_produtos(doc_zip)
                  if codigo[:2] in GRUPOS_BEBIDA or any(p.search(descricao) for p in padroes))


# Auditoria de 24/11/2026: a POF 2017-2018 publica a classificacao oficial dos grupos ("Classificacao dos grupos de
# Consumo Alimentar.xlsx", 1.593 itens). A regra v1 (cadastro de 2008) erra em dois sentidos contra ela: trata como
# comida as bebidas lacteas (Toddynho, leite achocolatado) e o guarana em po, e trata como bebida o "cafe com pao", o
# "cafe com bolo/biscoito" e o "cafe com farinha" (preparacoes mistas). A v2 corrige so esses pontos; o resto da v1
# (leite fluido, vitaminas, acucar, adocante, achocolatado em po, xarope e codigos 82/83 sem classificacao) continua.
# 2008-2009 nao tem classificacao oficial e segue a v1, que reproduz o universo validado da tabela t1.
REGRA_NAO_COMIDA = MappingProxyType({"2008-2009": "v1_cadastro", "2017-2018": "v2_classificacao_oficial_2017"})
GRUPO_BEBIDAS_OFICIAL = "Bebidas"
GRUPO_PREPARACOES_MISTAS = "Preparações mistas"
ARQUIVO_CLASSIFICACAO = "Classifica"  # "Classificação dos grupos de Consumo Alimentar.xlsx" (nome com acento no zip)


def classificacao_dos_grupos(linhas: Sequence[Mapping[str, str]]) -> dict[str, tuple[str, str]]:
    """Achata a aba da classificacao oficial: {codigo: (grupo, subgrupo)}. A coluna A abre um grupo, uma linha so com a
    coluna C abre um subgrupo e uma linha com codigo de 7 digitos na coluna B e um item."""
    grupo = subgrupo = None
    mapa = {}
    for linha in linhas:
        a, b, c = (str(linha.get(k, "")).strip() for k in ("A", "B", "C"))
        if b.isdigit() and len(b) == 7:
            mapa[b] = (grupo, subgrupo)
        elif a and not b and not c:
            grupo, subgrupo = a, None
        elif c and not a and not b and grupo:
            subgrupo = c
    return mapa


def classificacao_oficial(doc_zip: Path) -> dict[str, tuple[str, str]]:
    with zipfile.ZipFile(doc_zip) as z:
        nome = next(n for n in z.namelist() if ARQUIVO_CLASSIFICACAO in n and n.endswith(".xlsx"))
        folhas = coleta.ler_xlsx(z.read(nome))
    return classificacao_dos_grupos(next(iter(folhas.values())))


def regra_nao_comida_v2(nao_v1: Sequence[str], classificacao: Mapping[str, tuple[str, str]]) -> list[str]:
    """Nao-comida v2: bebidas da classificacao oficial + itens da v1 fora delas, menos as preparacoes mistas."""
    bebidas = {c for c, (grupo, _) in classificacao.items() if grupo == GRUPO_BEBIDAS_OFICIAL}
    mistas = {c for c, (grupo, _) in classificacao.items() if grupo == GRUPO_PREPARACOES_MISTAS}
    return sorted((set(nao_v1) | bebidas) - mistas)


def particao(domicilio_id: str) -> str:
    """Divisao do protocolo v2: teste reservado quando os 8 primeiros hex do SHA-256 do domicilio_id, como inteiro, tem
    resto 0 na divisao por 5 (cerca de 20% dos domicilios); o restante e desenvolvimento."""
    return "teste" if int(hashlib.sha256(domicilio_id.encode("utf-8")).hexdigest()[:8], 16) % 5 == 0 else "desenvolvimento"


# ======== tabelas do protocolo novo: escolhas da POF ========
# Saidas em Parquet (DuckDB) sob CASE_AMBEV_PROCESSADO_ROOT (padrao: data/processed/v2 no repositorio, ignorado pelo
# Git). Cada tabela grava ao lado um JSON de contrato com populacao, colunas, contagens e SHA-256.

RAIZ_REPO = Path(__file__).resolve().parents[1]
VAR_PROCESSADO = "CASE_AMBEV_PROCESSADO_ROOT"
SABORES = ("guarana", "cola", "outro", "desconhecido", "sem_refrigerante")
# Codigos do cadastro oficial de produtos do consumo alimentar com descricao "COLA" (iguais em 2008 e 2017).
COLA_POF = ("8200101", "8200102", "8200701", "8200702", "8200801")
# "REFRIGERANTE NAO ESPECIFICADO": e refrigerante, mas o sabor nao foi registrado. Fica fora de "outro" (sabor conhecido).
SABOR_DESCONHECIDO_POF = ("8203501",)
OUTROS_REFRIGERANTES = tuple(sorted(set(REFRIGERANTE) - set(GUARANA) - set(COLA_POF) - set(SABOR_DESCONHECIDO_POF)))
# Prioridade do sabor no episodio: guarana > cola > outro sabor conhecido > sabor desconhecido > sem refrigerante.
SQL_SABOR = ("CASE WHEN guarana = 1 THEN 'guarana' WHEN cola = 1 THEN 'cola' WHEN outro_conhecido = 1 THEN 'outro' "
             "WHEN desconhecido = 1 THEN 'desconhecido' ELSE 'sem_refrigerante' END")


def raiz_processado(env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    return Path(env.get(VAR_PROCESSADO) or RAIZ_REPO / "data" / "processed" / "v2")


def _lista_sql(codigos) -> str:
    return "(" + ",".join(f"'{c}'" for c in codigos) + ")"


def _gravar_parquet(con, tabela: str, destino: Path, contrato: dict) -> dict:
    """Grava a tabela em Parquet (ordem estavel) e o contrato ao lado; devolve o contrato com contagem e hash."""
    ordem = ", ".join(contrato["chave"])
    unica = con.execute(f"SELECT count(*) = count(DISTINCT ({ordem})) FROM {tabela}").fetchone()[0]
    if not unica:
        raise EntradaInvalida(f"{tabela}: chave {contrato['chave']} repetida")
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_name(destino.name + ".parcial")
    con.execute(f"COPY (SELECT * FROM {tabela} ORDER BY {ordem}) TO '{parcial.as_posix()}' (FORMAT PARQUET)")
    os.replace(parcial, destino)
    descricoes = contrato.pop("descricoes", {})
    final = {**contrato, "arquivo": destino.name, "linhas": con.execute(f"SELECT count(*) FROM {tabela}").fetchone()[0],
             "colunas": [{"nome": c[0], "tipo": c[1], "descricao": descricoes.get(c[0], "")}
                         for c in con.execute(f"DESCRIBE {tabela}").fetchall()],
             "sha256_conteudo": _hash_do_conteudo(con, tabela, ordem),
             "sha256": hashlib.sha256(destino.read_bytes()).hexdigest()}
    destino.with_suffix(".contrato.json").write_text(json.dumps(final, ensure_ascii=False, indent=1) + "\n",
                                                     encoding="utf-8")
    return final


def _hash_do_conteudo(con, tabela: str, ordem: str) -> str:
    """SHA-256 das linhas em ordem estavel: igual entre execucoes mesmo que o Parquet mude de metadados."""
    h = hashlib.sha256()
    for (linha,) in con.execute(f"SELECT CAST(t AS VARCHAR) FROM (SELECT * FROM {tabela} ORDER BY {ordem}) t").fetchall():
        h.update(linha.encode("utf-8") + b"\n")
    return h.hexdigest()


def _tabela_de_mapa(con, nome: str, colunas: tuple[str, ...], linhas: list[tuple]) -> None:
    """Tabela pequena de traducao montada pelas funcoes de harmonizacao do nucleo (implementacao unica)."""
    con.execute(f"CREATE OR REPLACE TEMP TABLE {nome} ({', '.join(f'{c} VARCHAR' for c in colunas)})")
    if linhas:
        con.executemany(f"INSERT INTO {nome} VALUES ({', '.join('?' * len(colunas))})",
                        [tuple(None if v is None else str(v) for v in linha) for linha in linhas])


def _mapa_periodo(con) -> None:
    _tabela_de_mapa(con, "mapa_periodo", ("hora", "periodo"), [(h, faixa_hora(h)) for h in range(24)])


# ---------- escolhas: refeicoes da POF (universo com e sem pizza) ----------

_FONTES_POF = {
    "2008-2009": {"pasta": "pof_2008_2009", "doc_zip": "Documentacao_20231009.zip",
                  "estratos": "Estratos_Geograficos_POF_2008-2009.xls"},
    "2017-2018": {"pasta": "pof_2017_2018", "doc_zip": "Documentacao_20230713.zip",
                  "estratos": "Estratos_POF_2017-2018.xls"},
}


def nao_comida(edicao: str, raiz_bruta: Path) -> list[str]:
    """Codigos que nao contam como comida pela regra da edicao (REGRA_NAO_COMIDA): v1 pelo cadastro (grupos 82/83 e
    descricoes de bebida ou adicao) e, em 2017-2018, v2 com a classificacao oficial dos grupos."""
    fonte = _FONTES_POF[edicao]
    doc_zip = Path(raiz_bruta) / fonte["pasta"] / "downloads" / fonte["doc_zip"]
    v1 = bebida_ou_adicao(doc_zip)
    if REGRA_NAO_COMIDA[edicao] == "v1_cadastro":
        return v1
    return regra_nao_comida_v2(v1, classificacao_oficial(doc_zip))


def _estratos(edicao: str, raiz_bruta: Path, dicionario: dict) -> dict:
    uf = next(v for aba in dicionario.values() for v in aba if v.codigo in ("UF", "COD_UF") and v.categorias)
    base = Path(raiz_bruta) / _FONTES_POF[edicao]["pasta"] / "extraido" / "documentacao"
    return decodificar_estratos(ler_planilha_estratos(base / _FONTES_POF[edicao]["estratos"]),
                                nomes_uf_do_dicionario(uf), edicao)


def _itens_sql(edicao: str, nao: list[str]) -> str:
    """Uma linha por item consumido (sem as linhas de adicao de 2017), com a chave da pessoa e as marcas do grafo."""
    marcas = (f"(V9001 NOT IN {_lista_sql(nao)})::INT comida, (V9001 IN {_lista_sql(pizza(edicao))})::INT pizza, "
              f"(V9001 IN {_lista_sql(REFRIGERANTE)})::INT refrigerante, "
              f"(V9001 IN {_lista_sql(GUARANA)})::INT guarana, (V9001 IN {_lista_sql(COLA_POF)})::INT cola, "
              f"(V9001 IN {_lista_sql(OUTROS_REFRIGERANTES)})::INT outro_conhecido, "
              f"(V9001 IN {_lista_sql(SABOR_DESCONHECIDO_POF)})::INT desconhecido")
    hora = sql_hora(edicao, "V9015")
    if edicao == "2008-2009":
        return f"""SELECT lpad(COD_UF, 2, '0') uf, lpad(COD_UF, 2, '0') || lpad(NUM_SEQ, 3, '0') || NUM_DV upa,
            CAST(NUM_DOM AS INT) num_dom, CAST(NUM_UC AS INT) num_uc, CAST(COD_INFORMANTE AS INT) informante,
            CAST(NUM_QUADRO AS INT) quadro, {hora} hora, CAST(ESTRATO_POF AS INT) estrato,
            DIA_SEMANA dia_semana_bruto, CAST(NULL AS VARCHAR) ocasiao_v9017, V9018 local_v9018,
            CAST(NULL AS VARCHAR) dia_atipico, CAST(PESO_FINAL AS DOUBLE) peso, {marcas} FROM consumo_2008"""
    return f"""SELECT UF uf, COD_UPA upa, CAST(NUM_DOM AS INT) num_dom, CAST(NUM_UC AS INT) num_uc,
        CAST(COD_INFORMANTE AS INT) informante, CAST(QUADRO AS INT) quadro, {hora} hora, CAST(ESTRATO_POF AS INT) estrato,
        DIA_SEMANA dia_semana_bruto, V9017 ocasiao_v9017, V9018 local_v9018, DIA_ATIPICO dia_atipico,
        CAST(PESO_FINAL AS DOUBLE) peso, {marcas}
        FROM consumo_2017 WHERE NOT (V9005 IS NULL AND V9007 IS NULL)"""


def _morador_sql(edicao: str) -> str:
    if edicao == "2008-2009":
        return """SELECT lpad(COD_UF, 2, '0') || lpad(NUM_SEQ, 3, '0') || NUM_DV upa, CAST(NUM_DOM AS INT) num_dom,
            CAST(NUM_UC AS INT) num_uc, CAST(COD_INFORMANTE AS INT) informante, IDADE_ANOS idade_bruta,
            V0405 sexo_bruto, NIVEL_INSTRUCAO_MORADOR nivel_bruto FROM morador_2008"""
    return """SELECT COD_UPA upa, CAST(NUM_DOM AS INT) num_dom, CAST(NUM_UC AS INT) num_uc,
        CAST(COD_INFORMANTE AS INT) informante, V0403 idade_bruta, V0404 sexo_bruto, NIVEL_INSTRUCAO nivel_bruto
        FROM morador_2017"""


def _distintos(con, coluna: str, tabela: str) -> list:
    return [r[0] for r in con.execute(f"SELECT DISTINCT {coluna} FROM {tabela} WHERE {coluna} IS NOT NULL").fetchall()]


def _mapas_de_harmonizacao(con, edicao: str, estratos: dict) -> None:
    """Traduz valores brutos distintos pelas funcoes do nucleo (crosswalk e variaveis), sem reescreve-las em SQL."""
    _mapa_periodo(con)
    _tabela_de_mapa(con, "mapa_dia", ("bruto", "dia_semana"),
                    [(v, dia_semana_numero(edicao, v)) for v in _distintos(con, "dia_semana_bruto", "itens")])
    _tabela_de_mapa(con, "mapa_area", ("uf", "estrato", "grande_regiao", "area"),
                    [(uf, est, grande_regiao(uf), area_comparavel(estratos[(uf, est)]))
                     for uf, est in con.execute("SELECT DISTINCT uf, estrato FROM itens").fetchall()])
    _tabela_de_mapa(con, "mapa_idade", ("bruto", "faixa_etaria"),
                    [(v, faixa_etaria(idade_anos(edicao, v))) for v in _distintos(con, "idade_bruta", "pessoas")])
    _tabela_de_mapa(con, "mapa_sexo", ("bruto", "sexo"),
                    [(v, sexo_padrao(edicao, v)) for v in _distintos(con, "sexo_bruto", "pessoas")])
    _tabela_de_mapa(con, "mapa_nivel", ("bruto", "escolaridade"),
                    [(v, escolaridade_3(escolaridade_nivel(edicao, v))) for v in _distintos(con, "nivel_bruto", "pessoas")])


_SQL_ESCOLHAS = """CREATE TEMP TABLE escolhas AS
    WITH ep AS (
        SELECT upa, num_dom, num_uc, informante, quadro, hora, min(uf) uf, min(estrato) estrato,
               min(dia_semana_bruto) dia_semana_bruto, min(ocasiao_v9017) ocasiao_v9017, min(local_v9018) local_v9018,
               min(dia_atipico) dia_atipico, min(peso) peso, max(comida) comida, max(pizza) pizza,
               max(refrigerante) refrigerante, max(guarana) guarana, max(cola) cola,
               max(outro_conhecido) outro_conhecido, max(desconhecido) desconhecido
        FROM itens GROUP BY upa, num_dom, num_uc, informante, quadro, hora)
    SELECT '{edicao}' edicao,
           concat_ws('-', ep.upa, ep.num_dom, ep.num_uc, ep.informante, ep.quadro, ep.hora) episodio_id,
           concat_ws('-', ep.upa, ep.num_dom) domicilio_id, pt.particao, ep.upa, ep.uf, ep.estrato, a.grande_regiao, a.area,
           CAST(d.dia_semana AS INT) dia_semana, m.periodo, ep.hora, ep.quadro, ep.ocasiao_v9017, ep.local_v9018,
           ep.dia_atipico, i.faixa_etaria, s.sexo, n.escolaridade, ep.peso, ep.pizza, ep.refrigerante, ep.guarana,
           ep.cola, {sabor} sabor
    FROM ep
    LEFT JOIN mapa_particao pt ON pt.domicilio_id = concat_ws('-', ep.upa, ep.num_dom)
    LEFT JOIN pessoas p USING (upa, num_dom, num_uc, informante)
    LEFT JOIN mapa_dia d ON d.bruto = ep.dia_semana_bruto
    LEFT JOIN mapa_periodo m ON m.hora = CAST(ep.hora AS VARCHAR)
    LEFT JOIN mapa_area a ON a.uf = ep.uf AND a.estrato = CAST(ep.estrato AS VARCHAR)
    LEFT JOIN mapa_idade i ON i.bruto = p.idade_bruta
    LEFT JOIN mapa_sexo s ON s.bruto = p.sexo_bruto
    LEFT JOIN mapa_nivel n ON n.bruto = p.nivel_bruto
    WHERE ep.comida = 1"""


def _contagens_escolhas(con) -> dict:
    """Contagens do contrato: linhas e domicilios das duas particoes; desfechos (pizza, refrigerante, sabor) so do
    desenvolvimento, para o teste reservado nao ser inspecionado antes da escolha dos modelos."""
    total, domicilios, sem_harmonizacao = con.execute("""SELECT count(*), count(DISTINCT domicilio_id),
        count(*) FILTER (WHERE dia_semana IS NULL OR periodo IS NULL OR area IS NULL OR sexo IS NULL) FROM escolhas""").fetchone()
    por_particao = {p: {"refeicoes": n, "domicilios": d} for p, n, d in con.execute(
        "SELECT particao, count(*), count(DISTINCT domicilio_id) FROM escolhas GROUP BY 1 ORDER BY 1").fetchall()}
    dev = con.execute("""SELECT sum(pizza), sum(refrigerante) FILTER (WHERE pizza = 1), sum(guarana) FILTER (WHERE pizza = 1),
        sum(cola) FILTER (WHERE pizza = 1), count(*) FILTER (WHERE pizza = 1 AND sabor = 'outro'),
        count(*) FILTER (WHERE pizza = 1 AND sabor = 'desconhecido') FROM escolhas WHERE particao = 'desenvolvimento'""").fetchone()
    por_particao["desenvolvimento"] |= dict(zip(
        ("com_pizza", "pizza_com_refrigerante", "pizza_com_guarana", "pizza_com_cola", "pizza_com_outro_sabor",
         "pizza_com_sabor_desconhecido"), dev, strict=True))
    return {"refeicoes": total, "domicilios": domicilios, "linhas_sem_harmonizacao": sem_harmonizacao,
            "por_particao": por_particao}


def escolhas_pof(edicao: str, raiz_bruta: Path, saida: Path) -> dict:
    """Refeicoes elegiveis (episodio E2 com comida) com as escolhas do grafo: pizza -> refrigerante -> sabor."""
    raiz_bruta = Path(raiz_bruta)
    base = raiz_bruta / _FONTES_POF[edicao]["pasta"] / "extraido"
    dicionario = ler_dicionario_xls(base / "documentacao" / "Dicionarios_de_variaveis.xls")
    sufixo = edicao[:4]
    con = duckdb.connect()
    con.execute("SET memory_limit = '2500MB'")
    for tabela in (f"consumo_{sufixo}", f"morador_{sufixo}"):
        arquivo, aba = ORIGEM_OFICIAL[tabela]
        carregar_registro(con, tabela, base / "dados" / arquivo, layout_de(dicionario[aba]), COLUNAS_POF[tabela])
    con.execute(f"CREATE TEMP TABLE itens AS {_itens_sql(edicao, nao_comida(edicao, raiz_bruta))}")
    con.execute(f"CREATE TEMP TABLE pessoas AS {_morador_sql(edicao)}")
    _mapas_de_harmonizacao(con, edicao, _estratos(edicao, raiz_bruta, dicionario))
    _tabela_de_mapa(con, "mapa_particao", ("domicilio_id", "particao"),
                    [(d, particao(d)) for (d,) in con.execute(
                        "SELECT DISTINCT concat_ws('-', upa, num_dom) FROM itens").fetchall()])
    con.execute(_SQL_ESCOLHAS.format(edicao=edicao, sabor=SQL_SABOR))
    return _gravar_parquet(con, "escolhas", Path(saida) / f"escolhas_pof_{sufixo}.parquet", {
        "tabela": "escolhas", "versao": "escolhas_v2", "edicao": edicao, "chave": ["episodio_id"],
        "unidade": "episodio E2: pessoa x dia de registro (quadro) x hora inteira, com ao menos um item de comida",
        "populacao": "refeicoes elegiveis (com e sem pizza); a populacao condicional e o subconjunto pizza = 1",
        "regra_nao_comida": REGRA_NAO_COMIDA[edicao],
        "grafo": "pizza ou outra comida -> com ou sem refrigerante -> guarana, cola, outro sabor ou sabor desconhecido",
        "grupo_de_divisao": "domicilio_id (todos os episodios de um domicilio ficam na mesma divisao)",
        "particao": "coluna particao: teste reservado pelo SHA-256 do domicilio_id (protocolo v2); desfechos so do "
                    "desenvolvimento aparecem nas contagens",
        "contagens": _contagens_escolhas(con),
        "datas": "sem data de calendario na edicao; dia_semana e quadro (dia 1 ou 2 de registro) sao os unicos marcadores",
        "limitacoes": ["sem marca, canal ou delivery", "moradores de 10 anos ou mais que registraram o consumo alimentar",
                       "2008-2009 sem V9017 e sem linhas de adicao; em 2017-2018 as linhas de adicao ficam fora",
                       "contagens sem peso; use a coluna peso para estimativas populacionais"],
        "descricoes": {"peso": "PESO_FINAL do IBGE (fator de expansao)",
                       "sabor": "guarana > cola > outro sabor conhecido > desconhecido (refrigerante nao especificado)",
                       "particao": "desenvolvimento ou teste (reservado; aberto uma unica vez)",
                       "ocasiao_v9017": "1 cafe, 2 almoco, 3 lanche, 4 jantar, 5 ceia, 6 outra, 9 indeterminado (so 2017)",
                       "local_v9018": "local da refeicao conforme o dicionario da edicao",
                       "quadro": "dia de registro (2008: 71/72; 2017: 72/73)",
                       "domicilio_id": "chave de agrupamento para as divisoes"}})


# ======== auditoria das refeicoes: dos registros brutos ate a tabela do alvo ========
# Reconstroi as refeicoes a partir de TODOS os registros brutos (inclusive adicoes, agua, codigos fora do cadastro e dias
# sem comida) para medir cobertura, ausencias e a sensibilidade do alvo. Desfechos (pizza, guarana) de 2017 so aparecem
# para a particao de desenvolvimento; a particao de teste mostra apenas contagens sem o alvo.

EDICOES_AUDITORIA = EDICOES
# Agua pura ou aromatizada. Tonica, coco e agua com sumo de fruta ficam em "outra bebida".
AGUA_2017 = ("8201004", "8201005", "8201007", "8216301", "8280001")
COLUNAS_AUDITORIA = {
    "2008-2009": ["COD_UF", "NUM_SEQ", "NUM_DV", "NUM_DOM", "NUM_UC", "COD_INFORMANTE", "ESTRATO_POF", "PESO_FINAL",
                  "NUM_QUADRO", "V9018", "V9015", "V9005", "V9007", "V9001", "V9016", "QTD_FINAL", "DIA_SEMANA"],
    "2017-2018": ["UF", "ESTRATO_POF", "COD_UPA", "NUM_DOM", "NUM_UC", "COD_INFORMANTE", "QUADRO", "SEQ", "V9005",
                  "V9007", "V9001", "V9015", "V9017", "V9018", "PESO_FINAL", "DIA_SEMANA", "DIA_ATIPICO"],
}
_UPA_2008 = "lpad(COD_UF,2,'0')||lpad(NUM_SEQ,3,'0')||NUM_DV upa"
_ALVO_AUDITORIA_SQL = """CASE WHEN pizza = 0 THEN 'sem_pizza' WHEN guarana = 1 THEN 'pizza_com_guarana'
    WHEN refri_ne = 1 THEN 'pizza_inconclusivo_refri_nao_especificado' ELSE 'pizza_sem_guarana' END"""


def _carregar_auditoria(con, edicao: str, raiz_bruta: Path) -> None:
    base = Path(raiz_bruta) / _FONTES_POF[edicao]["pasta"] / "extraido"
    dic = ler_dicionario_xls(base / "documentacao" / "Dicionarios_de_variaveis.xls")
    suf = edicao[:4]
    arq, aba = ORIGEM_OFICIAL[f"consumo_{suf}"]
    carregar_registro(con, "bruto", base / "dados" / arq, layout_de(dic[aba]), COLUNAS_AUDITORIA[edicao])
    arq, aba = ORIGEM_OFICIAL[f"morador_{suf}"]
    cols = (["COD_UF", "NUM_SEQ", "NUM_DV", "NUM_DOM", "NUM_UC", "COD_INFORMANTE", "IDADE_ANOS"] if suf == "2008"
            else ["COD_UPA", "NUM_DOM", "NUM_UC", "COD_INFORMANTE", "V0403"])
    carregar_registro(con, "morador", base / "dados" / arq, layout_de(dic[aba]), cols)
    if suf == "2008":
        arq, aba = ORIGEM_OFICIAL["domicilio_2008"]
        carregar_registro(con, "domicilio", base / "dados" / arq, layout_de(dic[aba]),
                          ["COD_UF", "NUM_SEQ", "NUM_DV", "NUM_DOM", "PERIODO_REAL"])


def sql_linhas_auditoria(edicao: str, cadastro: Iterable[str], nao: Sequence[str]) -> str:
    """Uma linha por registro bruto da tabela `bruto`, com a chave da pessoa e a classe do item."""
    agua = AGUA_2017 if edicao == "2017-2018" else ("-",)
    adicao = "(V9005 IS NULL AND V9007 IS NULL)" if edicao == "2017-2018" else "FALSE"
    classe = (f"CASE WHEN V9001 IS NULL THEN 'sem_codigo' WHEN {adicao} THEN 'adicao' "
              f"WHEN V9001 NOT IN {_lista_sql(cadastro)} THEN 'fora_do_cadastro' WHEN V9001 IN {_lista_sql(agua)} THEN 'agua' "
              f"WHEN V9001 IN {_lista_sql(nao)} THEN 'outra_bebida_ou_nao_comida' ELSE 'comida' END")
    marcas = (f"(V9001 IN {_lista_sql(pizza(edicao))})::INT pizza, (V9001 IN {_lista_sql(GUARANA)})::INT guarana, "
              f"(V9001 IN {_lista_sql(COLA_POF)})::INT cola, (V9001 IN {_lista_sql(OUTROS_REFRIGERANTES)})::INT outro_refri, "
              f"(V9001 IN {_lista_sql(SABOR_DESCONHECIDO_POF)})::INT refri_ne")
    if edicao == "2008-2009":
        pessoa = (f"lpad(COD_UF,2,'0') uf, {_UPA_2008}, CAST(NUM_DOM AS INT) dom, "
                  "CAST(NUM_UC AS INT) uc, CAST(COD_INFORMANTE AS INT) inf, CAST(NUM_QUADRO AS INT) quadro, "
                  "TRY_CAST(V9015 AS INT) hora")
        extra = "NULL::VARCHAR ocasiao, NULL::VARCHAR atipico"
        chave = ("concat_ws('|', COD_UF, NUM_SEQ, NUM_DV, NUM_DOM, NUM_UC, COD_INFORMANTE, NUM_QUADRO, V9015, V9001, "
                 "V9005, V9007, V9016, QTD_FINAL)")
    else:
        pessoa = ("UF uf, COD_UPA upa, CAST(NUM_DOM AS INT) dom, CAST(NUM_UC AS INT) uc, CAST(COD_INFORMANTE AS INT) inf, "
                  "CAST(QUADRO AS INT) quadro, TRY_CAST(replace(V9015,'H','') AS INT) hora")
        extra = "V9017 ocasiao, DIA_ATIPICO atipico"
        chave = "concat_ws('|', COD_UPA, NUM_DOM, NUM_UC, COD_INFORMANTE, QUADRO, SEQ)"
    return (f"SELECT {pessoa}, V9015 hora_bruta, DIA_SEMANA dia_bruto, CAST(ESTRATO_POF AS INT) estrato, "
            f"CAST(PESO_FINAL AS DOUBLE) peso, {extra}, V9001 codigo, {classe} classe, {marcas}, "
            f"{chave} chave_bruta FROM bruto")


def _um(con, sql: str):
    return con.execute(sql).fetchone()[0]


def _grupos(con, sql: str) -> dict:
    return {str(k): v for k, v in con.execute(sql).fetchall()}


def _contagens_linhas(con) -> dict:
    return {
        "linhas_brutas": _um(con, "SELECT count(*) FROM linhas"),
        "por_classe": _grupos(con, "SELECT classe, count(*) FROM linhas GROUP BY 1 ORDER BY 2 DESC"),
        "hora_invalida_ou_vazia": _um(con, "SELECT count(*) FROM linhas WHERE hora IS NULL OR hora NOT BETWEEN 0 AND 23"),
        "valores_de_hora_invalidos": _grupos(con, "SELECT hora_bruta, count(*) FROM linhas "
                                                  "WHERE hora IS NULL OR hora NOT BETWEEN 0 AND 23 GROUP BY 1"),
        "dia_semana_vazio": _um(con, "SELECT count(*) FROM linhas WHERE dia_bruto IS NULL"),
        "duplicatas_pela_chave": _um(con, "SELECT count(*) - count(DISTINCT chave_bruta) FROM linhas"),
        "ocasiao_v9017": _grupos(con, "SELECT coalesce(ocasiao,'(vazio)'), count(*) FROM linhas GROUP BY 1 ORDER BY 1"),
        "dia_atipico": _grupos(con, "SELECT coalesce(atipico,'(vazio)'), count(*) FROM linhas GROUP BY 1 ORDER BY 1"),
        "pessoas": _um(con, "SELECT count(DISTINCT (upa,dom,uc,inf)) FROM linhas"),
        "domicilios": _um(con, "SELECT count(DISTINCT (upa,dom)) FROM linhas"),
    }


def episodios_auditoria(con) -> None:
    """Tabela `ep`: um episodio por pessoa x quadro x hora valida, com o tipo (com comida, so agua...) e a particao."""
    con.execute("""CREATE TEMP TABLE ep AS SELECT upa, dom, uc, inf, quadro, hora, min(uf) uf, min(estrato) estrato,
        min(peso) peso, min(dia_bruto) dia_bruto,
        max((classe='comida')::INT) comida, max((classe='agua')::INT) agua,
        max((classe='outra_bebida_ou_nao_comida')::INT) bebida, max((classe='adicao')::INT) adicao,
        max(pizza) pizza, max(guarana) guarana, max(cola) cola, max(outro_refri) outro_refri, max(refri_ne) refri_ne
        FROM linhas WHERE hora BETWEEN 0 AND 23 GROUP BY upa, dom, uc, inf, quadro, hora""")
    _tabela_de_mapa(con, "mapa_part", ("dom_id", "particao"),
                    [(d, particao(d)) for (d,) in con.execute("SELECT DISTINCT upa || '-' || dom FROM ep").fetchall()])
    periodo = " ".join(f"WHEN hora = {h} THEN '{faixa_hora(h)}'" for h in range(24))
    regiao = " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in _REGIOES.items())
    con.execute(f"""CREATE TEMP TABLE ep2 AS SELECT ep.*, m.particao,
        CASE WHEN comida = 1 THEN 'com_comida' WHEN agua = 1 AND bebida = 0 THEN 'so_agua'
             WHEN bebida = 1 THEN 'so_bebidas_sem_comida' WHEN adicao = 1 THEN 'so_adicao'
             ELSE 'so_codigo_sem_classe' END tipo,
        CASE {periodo} END periodo, CASE substr(uf,1,1) {regiao} END regiao
        FROM ep JOIN mapa_part m ON m.dom_id = ep.upa || '-' || ep.dom""")
    con.execute("DROP TABLE ep; ALTER TABLE ep2 RENAME TO ep")


def _tabela_episodios(con) -> dict:
    out = {}
    for part, tipo, n, p in con.execute("SELECT particao, tipo, count(*), sum(peso) FROM ep GROUP BY 1, 2 ORDER BY 1, 2").fetchall():
        out.setdefault(part, {})[tipo] = {"n": n, "peso": p}
    for d in out.values():
        tn, tp = sum(v["n"] for v in d.values()), sum(v["peso"] for v in d.values())
        for v in d.values():
            v["pct"], v["pct_ponderado"] = 100 * v["n"] / tn, 100 * v["peso"] / tp
        d["total"] = {"n": tn, "peso": tp}
    return out


def _pessoas_e_dias(con, edicao: str) -> dict:
    """Moradores elegiveis (10+ anos) nos domicilios com consumo e quantos dias de registro cada um tem."""
    idade = "TRY_CAST(IDADE_ANOS AS INT)" if edicao == "2008-2009" else "TRY_CAST(V0403 AS INT)"
    chave = _UPA_2008 if edicao == "2008-2009" else "COD_UPA upa"
    con.execute(f"""CREATE TEMP TABLE mor AS SELECT {chave}, CAST(NUM_DOM AS INT) dom, CAST(NUM_UC AS INT) uc,
        CAST(COD_INFORMANTE AS INT) inf, {idade} idade FROM morador""")
    r = con.execute("""WITH d AS (SELECT DISTINCT upa, dom FROM linhas),
        dias AS (SELECT upa, dom, uc, inf, count(DISTINCT quadro) nd FROM linhas GROUP BY ALL)
        SELECT count(*) FILTER (WHERE m.idade >= 10), count(*) FILTER (WHERE m.idade >= 10 AND dias.nd IS NULL),
               count(*) FILTER (WHERE dias.nd = 1), count(*) FILTER (WHERE dias.nd = 2),
               count(*) FILTER (WHERE m.idade < 10 AND dias.nd IS NOT NULL)
        FROM mor m JOIN d USING (upa, dom) LEFT JOIN dias USING (upa, dom, uc, inf)""").fetchone()
    sem_morador = _um(con, """SELECT count(*) FROM (SELECT DISTINCT upa, dom, uc, inf FROM linhas) l
        ANTI JOIN mor USING (upa, dom, uc, inf)""")
    return {"moradores_10_mais_nos_domicilios_com_consumo": r[0], "elegiveis_sem_nenhum_registro": r[1],
            "pessoas_com_1_dia": r[2], "pessoas_com_2_dias": r[3], "registros_de_menores_de_10": r[4],
            "pessoas_do_consumo_sem_par_no_morador": sem_morador,
            "pessoas_por_quadro": _grupos(con, "SELECT quadro, count(DISTINCT (upa,dom,uc,inf)) FROM linhas "
                                               "GROUP BY 1 ORDER BY 1"),
            "quadro_das_pessoas_com_1_dia": _grupos(con, """WITH dias AS (SELECT upa, dom, uc, inf, min(quadro) q,
                count(DISTINCT quadro) nd FROM linhas GROUP BY ALL) SELECT q, count(*) FROM dias WHERE nd = 1 GROUP BY 1"""),
            "pessoas_dia": _um(con, "SELECT count(DISTINCT (upa,dom,uc,inf,quadro)) FROM linhas"),
            "pessoas_dia_sem_comida": _um(con, """SELECT count(*) FROM (SELECT upa,dom,uc,inf,quadro,
                max((classe='comida')::INT) c FROM linhas GROUP BY ALL) WHERE c = 0""")}


def tabela_alvo(con, filtro: str) -> dict:
    """Refeicoes com comida por desfecho: sem pizza, pizza com guarana, inconclusivo e pizza sem guarana."""
    rows = con.execute(f"""SELECT {_ALVO_AUDITORIA_SQL} c, count(*), sum(peso) FROM ep
        WHERE tipo = 'com_comida' AND {filtro} GROUP BY 1 ORDER BY 1""").fetchall()
    tn, tp = sum(r[1] for r in rows), sum(r[2] for r in rows)
    return {"denominador_refeicoes_com_comida": tn,
            "linhas": {c: {"n": n, "pct": 100 * n / tn, "pct_ponderado": 100 * p / tp} for c, n, p in rows}}


def _bebidas_com_pizza(con, filtro: str) -> dict:
    """Bebidas nas refeicoes com pizza (categorias sobrepostas: uma refeicao pode cair em mais de uma)."""
    r = con.execute(f"""SELECT count(*), sum(agua), sum((agua=0 AND bebida=0)::INT), sum(guarana), sum(cola),
        sum(outro_refri), sum(refri_ne), sum((guarana+cola+outro_refri+refri_ne >= 2)::INT),
        sum((bebida=1 AND guarana+cola+outro_refri+refri_ne = 0)::INT), sum((guarana=1 AND refri_ne=1)::INT),
        sum((guarana=1 AND cola=1)::INT)
        FROM ep WHERE tipo='com_comida' AND pizza=1 AND {filtro}""").fetchone()
    nomes = ("refeicoes_com_pizza", "com_agua", "nenhuma_bebida_registrada", "guarana", "cola", "outro_refri_sabor_conhecido",
             "refri_sabor_nao_especificado", "dois_ou_mais_tipos_de_refri", "so_bebidas_que_nao_sao_refri",
             "guarana_e_refri_nao_especificado_juntos", "guarana_e_cola_juntos")
    return dict(zip(nomes, r, strict=True))


def sensibilidade_do_alvo(con, filtro: str) -> dict:
    """Taxa estrita, sem os inconclusivos, limite superior e imputacao proporcional ao sabor conhecido."""
    n, w, g, gw, ne, conhecido = con.execute(f"""SELECT count(*), sum(peso),
        sum((pizza=1 AND guarana=1)::INT), sum(peso*(pizza=1 AND guarana=1)::INT),
        sum((pizza=1 AND guarana=0 AND refri_ne=1)::INT),
        sum((pizza=1 AND (guarana+cola+outro_refri)>0)::INT)
        FROM ep WHERE tipo='com_comida' AND {filtro}""").fetchone()
    share = g / conhecido if conhecido else 0.0
    return {"refeicoes": n, "positivos_estritos": g, "inconclusivos": ne,
            "taxa_estrita_pct": 100 * g / n, "taxa_estrita_ponderada_pct": 100 * gw / w,
            "taxa_excluindo_inconclusivos_pct": 100 * g / (n - ne),
            "taxa_limite_superior_pct": 100 * (g + ne) / n,
            "parcela_guarana_entre_pizza_com_refri_de_sabor_conhecido": share,
            "taxa_imputada_proporcional_pct": 100 * (g + share * ne) / n}


def _concentracao(con, filtro: str, eixo: str) -> list:
    return con.execute(f"""SELECT {eixo}, count(*) FILTER (WHERE tipo='com_comida') refeicoes,
        count(*) FILTER (WHERE tipo='com_comida' AND pizza=1) com_pizza,
        count(*) FILTER (WHERE tipo='com_comida' AND pizza=1 AND guarana=0 AND refri_ne=1) inconclusivos,
        count(*) FILTER (WHERE tipo='so_agua') so_agua
        FROM ep WHERE {filtro} GROUP BY 1 ORDER BY 1""").fetchall()


def _um_dia_por_regiao(con) -> list:
    return con.execute("""WITH dias AS (SELECT upa, dom, uc, inf, min(uf) uf, count(DISTINCT quadro) nd FROM linhas
        GROUP BY ALL) SELECT substr(uf,1,1) eixo, count(*), round(100.0*avg((nd=1)::INT), 2) FROM dias
        GROUP BY 1 ORDER BY 1""").fetchall()


def _semanas_2008(con) -> dict:
    """Semana de campo (PERIODO_REAL) dos domicilios de 2008: cobertura e refeicoes e inconclusivos por trimestre."""
    con.execute(f"""CREATE TEMP TABLE sem AS SELECT {_UPA_2008},
        CAST(NUM_DOM AS INT) dom, TRY_CAST(PERIODO_REAL AS INT) semana FROM domicilio""")
    faixa = con.execute("SELECT min(semana), max(semana), count(DISTINCT semana), count(*) FILTER (WHERE semana IS NULL), "
                        "count(*) FROM sem").fetchone()
    sem_par = _um(con, "SELECT count(*) FROM (SELECT DISTINCT upa, dom FROM ep) ANTI JOIN sem USING (upa, dom)")
    tri = con.execute("""SELECT (s.semana-1)//13 + 1 tri, count(*) FILTER (WHERE tipo='com_comida'),
        count(*) FILTER (WHERE tipo='com_comida' AND pizza=1 AND guarana=0 AND refri_ne=1)
        FROM ep JOIN sem s USING (upa, dom) GROUP BY 1 ORDER BY 1""").fetchall()
    return {"min_max_distintas_nulas_total": faixa, "domicilios_do_consumo_sem_semana": sem_par,
            "refeicoes_e_inconclusivos_por_trimestre_de_campo": tri}


def auditar_refeicoes(edicao: str, raiz_bruta: Path, temp: Path | None = None) -> dict:
    """Auditoria de uma edicao: linhas brutas por classe, pessoas e dias, episodios, alvo, bebidas e sensibilidade."""
    con = duckdb.connect()
    con.execute("SET memory_limit = '2500MB'; SET threads = 2")
    if temp:
        Path(temp).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = '{Path(temp).as_posix()}'")
    fonte = _FONTES_POF[edicao]
    cadastro = {codigo for codigo, _ in cadastro_produtos(Path(raiz_bruta) / fonte["pasta"] / "downloads" / fonte["doc_zip"])}
    _carregar_auditoria(con, edicao, raiz_bruta)
    con.execute(f"CREATE TEMP TABLE linhas AS {sql_linhas_auditoria(edicao, cadastro, nao_comida(edicao, raiz_bruta))}")
    con.execute("DROP TABLE bruto")
    out = {"edicao": edicao, "itens_no_cadastro": len(cadastro), "linhas": _contagens_linhas(con),
           "pessoas": _pessoas_e_dias(con, edicao)}
    episodios_auditoria(con)
    out["episodios"] = _tabela_episodios(con)
    filtros = ({"desenvolvimento": "particao = 'desenvolvimento'"} if edicao == "2017-2018"
               else {"todas": "TRUE", "desenvolvimento": "particao = 'desenvolvimento'"})
    out["alvo"] = {k: tabela_alvo(con, f) for k, f in filtros.items()}
    out["bebidas_nas_refeicoes_com_pizza"] = {k: _bebidas_com_pizza(con, f) for k, f in filtros.items()}
    out["sensibilidade"] = {k: sensibilidade_do_alvo(con, f) for k, f in filtros.items()}
    dev = filtros["desenvolvimento"]
    out["concentracao_por_regiao"] = _concentracao(con, dev, "regiao")
    out["concentracao_por_periodo"] = _concentracao(con, dev, "periodo")
    out["um_dia_por_regiao"] = _um_dia_por_regiao(con)
    if edicao == "2008-2009":
        out["periodo_real"] = _semanas_2008(con)
    return out


def gravar_auditorias(raiz_bruta: Path, saida: Path, temp: Path | None = None) -> list[str]:
    """Grava auditoria_<ano>.json de cada edicao em `saida`; devolve os nomes gravados."""
    Path(saida).mkdir(parents=True, exist_ok=True)
    nomes = []
    for edicao in EDICOES_AUDITORIA:
        nome = f"auditoria_{edicao[:4]}.json"
        (Path(saida) / nome).write_text(json.dumps(auditar_refeicoes(edicao, raiz_bruta, temp), ensure_ascii=False,
                                                   indent=1, default=str), encoding="utf-8")
        nomes.append(nome)
    return nomes


# ---------- linha de comando ----------

TABELAS = ("escolhas_2008", "escolhas_2017")
AUDITORIA = "auditoria"


def preparar_tudo(raiz_bruta: Path, saida: Path, tabelas: Sequence[str] = TABELAS) -> dict:
    acoes = {
        "escolhas_2008": lambda: escolhas_pof("2008-2009", raiz_bruta, saida),
        "escolhas_2017": lambda: escolhas_pof("2017-2018", raiz_bruta, saida),
    }
    contratos = {t: acoes[t]() for t in tabelas}
    caminho = Path(saida) / "manifesto_tabelas.json"
    anteriores = json.loads(caminho.read_text(encoding="utf-8"))["tabelas"] if caminho.exists() else {}
    atuais = {t: {k: c[k] for k in ("arquivo", "linhas", "sha256", "sha256_conteudo")} for t, c in contratos.items()}
    caminho.write_text(json.dumps({"versao": "tabelas_v2", "tabelas": {**anteriores, **atuais}}, ensure_ascii=False,
                                  indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return contratos


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepara as tabelas de escolhas em Parquet e, com o alvo 'auditoria', "
                                                 "audita as refeicoes dos registros brutos ate a tabela do alvo. "
                                                 f"Le CASE_AMBEV_RAW_ROOT e grava em {VAR_PROCESSADO}.")
    parser.add_argument("alvos", nargs="*", metavar="alvo",
                        help=f"uma ou mais de {', '.join(TABELAS)}, {AUDITORIA} (padrao: todas as tabelas)")
    parser.add_argument("--temp", type=Path, help="pasta temporaria do DuckDB para a auditoria (opcional)")
    args = parser.parse_args(argv)
    desconhecidos = sorted(set(args.alvos) - {*TABELAS, AUDITORIA})
    if desconhecidos:
        parser.error(f"alvos desconhecidos: {desconhecidos}")
    tabelas = tuple(a for a in args.alvos if a in TABELAS)
    resumo = {}
    if tabelas or not args.alvos:
        contratos = preparar_tudo(raiz_bruta(), raiz_processado(), tabelas or TABELAS)
        resumo = {t: {"linhas": c["linhas"], "sha256_conteudo": c["sha256_conteudo"]} for t, c in contratos.items()}
    if AUDITORIA in args.alvos:
        resumo[AUDITORIA] = gravar_auditorias(raiz_bruta(), raiz_processado(), args.temp)
    sys.stdout.write(json.dumps(resumo, ensure_ascii=False, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
