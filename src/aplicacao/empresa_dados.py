"""Dados da empresa: leitura do arquivo enviado, diagnostico e codificacao das variaveis.

Tres blocos, em ordem: formato esperado e leitura (sem nome, documento ou contato); diagnostico, que recusa o treino
quando o alvo nao pode ser observado e aponta os problemas por linha; e codificacao das variaveis, aprendida so com o
treino e aplicada igual na avaliacao e na inferencia.
"""

import csv
import hashlib
import io
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType

import numpy as np

from aplicacao import EntradaInvalida, conferir_inteiro, previsao_consumo, recebido
from occasion.variaveis import carregar_regras, faixa_hora


# ---------- leitura do arquivo ----------
# Formato esperado e leitura segura do CSV enviado: so texto tabular, sem executar nada.
#
# Formato: papeis das colunas (obrigatorios e opcionais), tipos de registro e um exemplo ficticio para baixar.
# Leitura: previa, mapeamento sugerido e colunas com aparencia de dado pessoal. Recusa arquivo vazio ou grande demais,
# bytes nulos, cabecalho vazio ou repetido e linhas com numero de colunas diferente do cabecalho. Codificacao UTF-8 (com
# ou sem BOM); se falhar, Latin-1 declarada no resultado. Separador "," ou ";".

# ---------- formato esperado ----------

COLUNAS_OBRIGATORIAS = MappingProxyType({
    "identificador": "Identificador anônimo do pedido ou da ocasião (sem nome, documento ou contato)",
    "data_hora": "Data e hora do pedido (AAAA-MM-DD HH:MM ou DD/MM/AAAA HH:MM)",
    "localizacao": "UF (sigla) ou grande região",
    "pizza": "O pedido tem pizza? (sim/não ou 1/0)",
    "guarana": "O mesmo pedido tem guaraná? (sim/não ou 1/0)"})
COLUNAS_OPCIONAIS = MappingProxyType({
    "grupo": "Loja ou estabelecimento (usado para separar a avaliação, não como variável)",
    "canal": "Canal do pedido (por exemplo, delivery ou salão)",
    "promocao": "Havia promoção no pedido? (sim/não ou 1/0)",
    "preco": "Preço da pizza ou do pedido, em reais",
    "temperatura": "Temperatura no local e horário, em °C"})
VARIAVEIS_DECLARAVEIS = ("canal", "promocao", "preco", "temperatura")
TIPOS_DE_REGISTRO = MappingProxyType({
    "pedido_consumidor": "Pedidos de consumidores (delivery, salão, balcão)",
    "registro_consumo": "Registros de ocasiões de consumo (pesquisa, diário)",
    "venda_a_distribuidor": "Vendas da indústria a distribuidores",
    "abastecimento_estabelecimento": "Abastecimento de estabelecimentos (sell-in)"})
TIPOS_ACEITOS = ("pedido_consumidor", "registro_consumo")
DESCRICAO = ("Exemplo fictício, só para mostrar a estrutura: uma linha por pedido, com pizza e guaraná indicados no "
             "mesmo pedido. Não contém dados reais.")
_EXEMPLO = (
    ("id_pedido", "data_hora", "uf", "tem_pizza", "tem_guarana", "loja", "canal", "promocao"),
    ("A0001", "2025-03-07 20:15", "SP", "sim", "sim", "L01", "delivery", "0"),
    ("A0002", "2025-03-07 20:40", "SP", "sim", "nao", "L01", "salao", "1"),
    ("A0003", "2025-03-08 13:05", "RJ", "nao", "sim", "L02", "delivery", "0"),
    ("A0004", "2025-03-09 21:30", "MG", "sim", "nao", "L03", "delivery", "0"))


def exemplo_csv() -> str:
    return "\n".join(",".join(linha) for linha in _EXEMPLO) + "\n"


# ---------- leitura ----------

MAXIMO_BYTES = 20 * 1024 * 1024
MAXIMO_LINHAS = 500_000
MAXIMO_COLUNAS = 60
LINHAS_DA_PREVIA = 10
_NOME_PESSOAL = re.compile(r"(^|_)(nome|email|e_mail|telefone|celular|fone|cpf|cnpj|rg|endereco|endereço|cep)($|_)")
_VALOR_PESSOAL = (re.compile(r"[^@\s]+@[^@\s]+\.[a-z]{2,}", re.I), re.compile(r"\d{3}\.\d{3}\.\d{3}-\d{2}"),
                  re.compile(r"\(?\d{2}\)?\s?9\d{4}-?\d{4}"))
_SUGESTOES = {
    "identificador": ("id_pedido", "pedido", "id", "id_ocasiao", "codigo_pedido"),
    "data_hora": ("data_hora", "datahora", "data", "momento", "timestamp", "criado_em"),
    "localizacao": ("uf", "estado", "regiao", "grande_regiao"),
    "pizza": ("tem_pizza", "pizza", "com_pizza"),
    "guarana": ("tem_guarana", "guarana", "com_guarana"),
    "grupo": ("loja", "estabelecimento", "id_loja", "restaurante"),
    "canal": ("canal",), "promocao": ("promocao", "promo", "em_promocao"),
    "preco": ("preco", "valor", "preco_pizza"), "temperatura": ("temperatura", "temp_c")}


@dataclass(frozen=True)
class Tabela:
    colunas: list
    linhas: list
    sha256: str
    codificacao: str
    separador: str

    def previa(self, n: int = LINHAS_DA_PREVIA) -> dict:
        return {"colunas": list(self.colunas), "linhas": [list(l) for l in self.linhas[:n]], "total_de_linhas": len(self.linhas)}

    def coluna(self, nome: str) -> list[str]:
        indice = self.colunas.index(nome)
        return [linha[indice] for linha in self.linhas]


def _decodificar(conteudo: bytes) -> tuple[str, str]:
    try:
        return conteudo.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        return conteudo.decode("latin-1"), "latin-1"


def ler(conteudo: bytes) -> Tabela:
    if not conteudo:
        raise EntradaInvalida("o arquivo está vazio")
    if len(conteudo) > MAXIMO_BYTES:
        raise EntradaInvalida(f"o arquivo passa de {MAXIMO_BYTES // (1024 * 1024)} MB")
    if b"\x00" in conteudo:
        raise EntradaInvalida("o arquivo não parece ser um CSV de texto (contém bytes nulos)")
    texto, codificacao = _decodificar(conteudo)
    if not texto.strip():
        raise EntradaInvalida("o arquivo está vazio")
    primeira = texto.split("\n", 1)[0]
    separador = ";" if primeira.count(";") > primeira.count(",") else ","
    leitor = csv.reader(io.StringIO(texto), delimiter=separador)
    try:
        cabecalho = [c.strip() for c in next(leitor)]
        linhas = []
        for numero, linha in enumerate(leitor, start=2):
            if not any(c.strip() for c in linha):
                continue
            if len(linha) != len(cabecalho):
                raise EntradaInvalida(f"a linha {numero} tem {len(linha)} colunas; o cabeçalho tem {len(cabecalho)}")
            linhas.append([c.strip() for c in linha])
            if len(linhas) > MAXIMO_LINHAS:
                raise EntradaInvalida(f"o arquivo passa de {MAXIMO_LINHAS} linhas")
    except csv.Error as erro:
        raise EntradaInvalida(f"o CSV não pôde ser lido: {erro}") from erro
    if not cabecalho or any(not c for c in cabecalho) or len(set(cabecalho)) != len(cabecalho):
        raise EntradaInvalida("o cabeçalho precisa de nomes de coluna preenchidos e diferentes entre si")
    if len(cabecalho) > MAXIMO_COLUNAS:
        raise EntradaInvalida(f"o arquivo passa de {MAXIMO_COLUNAS} colunas")
    if not linhas:
        raise EntradaInvalida("o arquivo não tem linhas de dados")
    return Tabela(cabecalho, linhas, hashlib.sha256(conteudo).hexdigest(), codificacao, separador)


def _normalizar(nome: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", nome.lower()).strip("_")


def colunas_pessoais(tabela: Tabela) -> list[str]:
    """Colunas com nome ou conteudo de dado pessoal identificavel (amostra das primeiras linhas)."""
    suspeitas = []
    for indice, nome in enumerate(tabela.colunas):
        amostra = [linha[indice] for linha in tabela.linhas[:200]]
        if _NOME_PESSOAL.search(_normalizar(nome)) or any(p.search(v) for v in amostra for p in _VALOR_PESSOAL):
            suspeitas.append(nome)
    return suspeitas


def sugerir_mapeamento(colunas: list[str]) -> dict:
    normalizadas = {_normalizar(c): c for c in colunas}
    sugestao = {}
    for papel, nomes in _SUGESTOES.items():
        encontrado = next((normalizadas[n] for n in nomes if n in normalizadas), None)
        if encontrado and encontrado not in sugestao.values():
            sugestao[papel] = encontrado
    return sugestao


# ---------- diagnostico e plano ----------
# Diagnostico do arquivo mapeado: unidade, alvo observavel, variaveis disponiveis antes da previsao e problemas.
#
# O alvo e observavel so quando pizza e guarana estao indicados na MESMA unidade (pedido ou ocasiao) de um tipo de registro
# de consumo. Valor invalido em campo obrigatorio bloqueia, com as linhas apontadas: nada e descartado em silencio. Variavel
# opcional so entra se quem envia declarar que ela e conhecida antes da previsao.

MINIMO_OCASIOES = 300
MINIMO_POSITIVOS = 30
MINIMO_NEGATIVOS = 30
MAXIMO_CATEGORIAS_CANAL = 20
LINHAS_APONTADAS = 5
_FORMATOS_DATA = ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y %H:%M",
                  "%d/%m/%Y %H:%M:%S")
_SIM = {"sim", "s", "1", "true", "verdadeiro", "yes", "y"}
_NAO = {"nao", "não", "n", "0", "false", "falso", "no"}
UF_REGIAO = {
    **dict.fromkeys(("AC", "AP", "AM", "PA", "RO", "RR", "TO"), "Norte"),
    **dict.fromkeys(("AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"), "Nordeste"),
    **dict.fromkeys(("ES", "MG", "RJ", "SP"), "Sudeste"), **dict.fromkeys(("PR", "RS", "SC"), "Sul"),
    **dict.fromkeys(("DF", "GO", "MS", "MT"), "Centro-Oeste")}
REGIOES = ("Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste")
_UNIDADES = {"pedido_consumidor": "Cada linha é um pedido de consumidor; o alvo é se um pedido com pizza também tem guaraná.",
             "registro_consumo": "Cada linha é uma ocasião de consumo registrada; o alvo é se a ocasião com pizza também tem guaraná."}


def data_hora(texto: str) -> datetime | None:
    for padrao in _FORMATOS_DATA:
        try:
            return datetime.strptime(texto, padrao)
        except ValueError:
            continue
    return None


def booleano(texto: str) -> int | None:
    valor = texto.strip().lower()
    return 1 if valor in _SIM else 0 if valor in _NAO else None


def regiao(texto: str) -> str | None:
    valor = texto.strip()
    return UF_REGIAO.get(valor.upper()) or next((r for r in REGIOES if r.lower() == valor.lower()), None)


def numero(texto: str) -> float | None:
    try:
        valor = float(texto.replace(",", "."))
    except ValueError:
        return None
    return valor if math.isfinite(valor) else None


_CONVERSORES = {"data_hora": data_hora, "localizacao": regiao, "pizza": booleano, "guarana": booleano,
                "promocao": booleano, "preco": numero, "temperatura": numero}


def _problema(campo: str, mensagem: str, bloqueia: bool = True) -> dict:
    return {"campo": campo, "mensagem": mensagem, "bloqueia": bloqueia}


def _conferir_mapeamento(tabela, mapeamento: dict, declaracao: dict) -> list[dict]:
    problemas = []
    tipo = declaracao.get("tipo_de_registro")
    if tipo not in TIPOS_DE_REGISTRO:
        problemas.append(_problema("tipo_de_registro", "Informe o que cada linha representa (tipo de registro)."))
    elif tipo not in TIPOS_ACEITOS:
        problemas.append(_problema("tipo_de_registro", (
            f"{TIPOS_DE_REGISTRO[tipo]} não mostram o que o consumidor levou junto com a pizza: vendas a "
            "distribuidor ou abastecimento não permitem observar se o guaraná acompanhou a pizza na mesma ocasião.")))
    for papel, coluna in mapeamento.items():
        if papel not in {**COLUNAS_OBRIGATORIAS, **COLUNAS_OPCIONAIS}:
            problemas.append(_problema(papel, f"Papel de coluna desconhecido: {papel}."))
        elif coluna not in tabela.colunas:
            problemas.append(_problema(papel, f"A coluna '{coluna}' indicada para {papel} não existe no arquivo."))
    for papel in COLUNAS_OBRIGATORIAS:
        if papel not in mapeamento:
            mensagem = f"Falta indicar a coluna de {papel}: {COLUNAS_OBRIGATORIAS[papel]}."
            if papel == "pizza":
                mensagem = ("Falta o indicador de pizza na mesma unidade. Sem ele não é possível observar se o guaraná "
                            "acompanhou a pizza, e o treino não pode ser feito com este arquivo.")
            problemas.append(_problema(papel, mensagem))
    pessoais = set(colunas_pessoais(tabela))
    for papel, coluna in mapeamento.items():
        if coluna in pessoais:
            problemas.append(_problema(papel, f"A coluna '{coluna}' parece conter dado pessoal; use um identificador anônimo."))
    if "preco" in mapeamento:
        problemas.append(_problema("preco", "Confira se o preço é só da pizza: um total do pedido que já inclui o "
                                            "guaraná revela o alvo e invalida o treino.", bloqueia=False))
    for coluna in sorted(pessoais - set(mapeamento.values())):
        problemas.append(_problema(coluna, f"A coluna '{coluna}' parece conter dado pessoal; ela não será usada. "
                                           "Prefira removê-la antes de enviar.", bloqueia=False))
    return problemas


def _converter(tabela, mapeamento: dict) -> tuple[dict, list[dict]]:
    valores, problemas = {}, []
    for papel, coluna in mapeamento.items():
        if coluna not in tabela.colunas:
            continue
        brutos = tabela.coluna(coluna)
        conversor = _CONVERSORES.get(papel)
        convertidos = [conversor(v) if conversor else (v or None) for v in brutos]
        invalidos = [i + 2 for i, v in enumerate(convertidos) if v is None]
        if invalidos:
            linhas = ", ".join(map(str, invalidos[:LINHAS_APONTADAS]))
            bloqueia = papel in COLUNAS_OBRIGATORIAS
            problemas.append(_problema(papel, f"{papel}: {len(invalidos)} valor(es) vazio(s) ou inválido(s) na coluna "
                                              f"'{coluna}' (linhas {linhas}{'…' if len(invalidos) > LINHAS_APONTADAS else ''}).",
                                       bloqueia))
        valores[papel] = convertidos
    return valores, problemas


def _variaveis(mapeamento: dict, declaracao: dict, registros: list[dict]) -> dict:
    usadas = [{"campo": "dia_semana", "origem": "data_hora"}, {"campo": "periodo", "origem": "data_hora"},
              {"campo": "grande_regiao", "origem": "localizacao"}]
    excluidas = []
    conhecidos = declaracao.get("conhecidos_antes", {})
    for campo in VARIAVEIS_DECLARAVEIS:
        if campo not in mapeamento:
            continue
        if conhecidos.get(campo) is not True:
            excluidas.append({"campo": campo, "motivo": "não declarado como conhecido antes da previsão"})
        elif campo == "canal" and len({r["canal"] for r in registros}) > MAXIMO_CATEGORIAS_CANAL:
            excluidas.append({"campo": campo, "motivo": f"mais de {MAXIMO_CATEGORIAS_CANAL} categorias"})
        elif any(r[campo] is None for r in registros):
            excluidas.append({"campo": campo, "motivo": "há valores vazios ou inválidos nas ocasiões com pizza"})
        else:
            usadas.append({"campo": campo, "origem": mapeamento[campo]})
    return {"usadas": usadas, "excluidas": excluidas,
            "ausentes_nesta_versao": "clima, feriado e preço só entram se vierem no arquivo e forem conhecidos antes da previsão"}


def _registros(valores: dict, n: int) -> list[dict]:
    registros = []
    for i in range(n):
        if valores["pizza"][i] != 1:
            continue
        momento = valores["data_hora"][i]
        registro = {"momento": momento, "dia_semana": momento.isoweekday(), "hora": momento.hour,
                    "periodo": faixa_hora(momento.hour), "grande_regiao": valores["localizacao"][i],
                    "grupo": valores.get("grupo", [None] * n)[i], "y": valores["guarana"][i]}
        for campo in VARIAVEIS_DECLARAVEIS:
            registro[campo] = valores[campo][i] if campo in valores else None
        registros.append(registro)
    return registros


def diagnosticar(tabela, mapeamento: dict, declaracao: dict) -> dict:
    if not isinstance(mapeamento, dict) or not isinstance(declaracao, dict):
        raise EntradaInvalida("mapeamento e declaração devem ser objetos")
    if not all(isinstance(p, str) and isinstance(c, str) for p, c in mapeamento.items()):
        raise EntradaInvalida("o mapeamento deve ligar cada papel ao nome de uma coluna (texto)")
    if not isinstance(declaracao.get("tipo_de_registro"), str):
        raise EntradaInvalida("informe o tipo de registro como texto")
    conhecidos = declaracao.get("conhecidos_antes", {})
    if not isinstance(conhecidos, dict) or not all(isinstance(v, bool) for v in conhecidos.values()):
        raise EntradaInvalida("conhecidos_antes deve indicar, para cada variável, verdadeiro ou falso")
    problemas = _conferir_mapeamento(tabela, mapeamento, declaracao)
    valores, invalidos = _converter(tabela, mapeamento)
    problemas += invalidos
    registros = []
    if not any(p["bloqueia"] for p in problemas):
        ids = Counter(valores["identificador"])
        repetidos = [i for i, n in ids.items() if n > 1]
        if repetidos:
            problemas.append(_problema("identificador", f"{len(repetidos)} identificador(es) repetido(s), por exemplo "
                                                        f"'{repetidos[0]}': cada linha precisa ser uma unidade única."))
        registros = _registros(valores, len(tabela.linhas))
    positivos = sum(r["y"] for r in registros)
    if registros or not any(p["bloqueia"] for p in problemas):
        for falta, minimo, texto in ((len(registros), MINIMO_OCASIOES, "ocasiões com pizza"),
                                     (positivos, MINIMO_POSITIVOS, "ocasiões com pizza e guaraná"),
                                     (len(registros) - positivos, MINIMO_NEGATIVOS, "ocasiões com pizza sem guaraná")):
            if falta < minimo:
                problemas.append(_problema("volume", f"Há {falta} {texto}; o mínimo para treinar é {minimo}."))
    tipo = declaracao.get("tipo_de_registro")
    momentos = [r["momento"] for r in registros]
    return {
        "pode_treinar": not any(p["bloqueia"] for p in problemas), "problemas": problemas,
        "unidade": {"tipo": tipo, "descricao": _UNIDADES.get(tipo, "Tipo de registro não aceito para esta pergunta.")},
        "alvo": {"observavel": "pizza" in mapeamento and "guarana" in mapeamento and tipo in TIPOS_ACEITOS,
                 "definicao": "guaraná na mesma unidade, entre as unidades com pizza",
                 "ocasioes_com_pizza": len(registros), "com_guarana": positivos, "linhas_no_arquivo": len(tabela.linhas)},
        "periodo_coberto": {"inicio": min(momentos).isoformat(), "fim": max(momentos).isoformat()} if momentos else None,
        "variaveis": _variaveis(mapeamento, declaracao, registros), "dados_sha256": tabela.sha256,
        "mapeamento": dict(mapeamento), "declaracao": dict(declaracao), "registros": registros}


# ---------- codificacao das variaveis ----------
# Codificacao das variaveis da empresa, aprendida so com o treino e aplicada igual na avaliacao e na inferencia.
#
# Sempre: periodo (4), dia da semana (7), grande regiao (5). Opcionais declarados: canal (categorias do treino),
# promocao (0/1), preco e temperatura (numericos; a escala e aprendida dentro do pipeline do modelo).

PERIODOS = tuple(carregar_regras()["faixas_hora"])
OPCIONAIS = ("canal", "promocao", "preco", "temperatura")


@dataclass(frozen=True)
class Codificador:
    campos: tuple
    canais: tuple

    def colunas(self) -> list[str]:
        nomes = [f"hora_{p}" for p in PERIODOS] + [f"dia_{d}" for d in range(1, 8)] + [f"regiao_{r}" for r in REGIOES]
        if "canal" in self.campos:
            nomes += [f"canal_{c}" for c in self.canais]
        return nomes + [c for c in ("promocao", "preco", "temperatura") if c in self.campos]

    def linha(self, r: Mapping) -> list[float]:
        if r["periodo"] not in PERIODOS or r["grande_regiao"] not in REGIOES or not 1 <= r["dia_semana"] <= 7:
            raise ValueError(f"registro fora do vocabulário: {r['dia_semana']}, {r['periodo']}, {r['grande_regiao']}")
        if "canal" in self.campos and r["canal"] not in self.canais:
            raise ValueError(f"canal não visto no treino: {r['canal']}")
        valores = [float(r["periodo"] == p) for p in PERIODOS] + [float(r["dia_semana"] == d) for d in range(1, 8)]
        valores += [float(r["grande_regiao"] == g) for g in REGIOES]
        if "canal" in self.campos:
            valores += [float(r["canal"] == c) for c in self.canais]
        return valores + [float(r[c]) for c in ("promocao", "preco", "temperatura") if c in self.campos]

    def matriz(self, registros: Sequence[Mapping]) -> np.ndarray:
        return np.array([self.linha(r) for r in registros], dtype=float)

    def entrada(self, modelo: str, registros: Sequence[Mapping]) -> np.ndarray:
        """Entrada de cada modelo, igual no treino e na inferencia: celula dia x periodo para a frequencia,
        indicadores para os demais."""
        if modelo == "frequencia":
            return np.array([[self.celula(r)] for r in registros], dtype=object)
        return self.matriz(registros)

    @staticmethod
    def celula(r: Mapping) -> str:
        return f"{r['dia_semana']}|{r['periodo']}"

    @staticmethod
    def celula_de_suporte(r: Mapping) -> str:
        return f"{r['dia_semana']}|{r['periodo']}|{r['grande_regiao']}"


def ajustar(registros: Sequence[Mapping], campos_usados: Sequence[str]) -> Codificador:
    opcionais = tuple(c for c in OPCIONAIS if c in campos_usados)
    canais = tuple(sorted({str(r["canal"]) for r in registros})) if "canal" in opcionais else ()
    return Codificador(("dia_semana", "periodo", "grande_regiao", *opcionais), canais)


def preparar(contexto: Mapping, codificador: Codificador) -> dict:
    """Contexto da inferencia na forma do treino; recusa campos que o modelo nao usa ou valores fora do treino."""
    if not isinstance(contexto, Mapping):
        raise EntradaInvalida("o contexto deve ser um dicionário")
    base = ("dia_semana", "hora", "grande_regiao", *[c for c in OPCIONAIS if c in codificador.campos])
    esperados = list(previsao_consumo.campos_informados(contexto, base))  # `periodo` no lugar de `hora`, nunca os dois
    if set(contexto) != set(esperados):
        raise EntradaInvalida(f"este modelo usa exatamente os campos {esperados}")
    periodo, hora = previsao_consumo.periodo_informado(contexto)
    if contexto["grande_regiao"] not in REGIOES:
        raise EntradaInvalida(f"grande_regiao deve ser uma de {list(REGIOES)}")
    registro = {"dia_semana": conferir_inteiro(contexto["dia_semana"], "dia_semana", 1, 7), "periodo": periodo,
                "grande_regiao": contexto["grande_regiao"], **({} if hora is None else {"hora": hora})}
    if "canal" in codificador.campos:
        if contexto["canal"] not in codificador.canais:
            raise EntradaInvalida(f"canal deve ser um dos vistos no treino: {list(codificador.canais)}")
        registro["canal"] = contexto["canal"]
    if "promocao" in codificador.campos:
        registro["promocao"] = conferir_inteiro(contexto["promocao"], "promocao", 0, 1)
    for campo in ("preco", "temperatura"):
        if campo in codificador.campos:
            valor = contexto[campo]
            try:
                numero = float(valor) if not isinstance(valor, bool) and isinstance(valor, (int, float)) else math.nan
            except OverflowError:
                numero = math.nan
            if not math.isfinite(numero):
                raise EntradaInvalida(f"{campo} deve ser um número finito, recebido {recebido(valor)}")
            registro[campo] = numero
    return registro
