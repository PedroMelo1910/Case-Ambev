"""Entender a mensagem e validar o cenario: campos com proveniencia, correcoes e perguntas so pelo essencial.

Cada campo guarda {valor, estado, origem, turno} e, quando uma fonte conferiu, `fonte_id`. Estados: declarado (a pessoa
disse), hipotese (cenario condicional ou aceito como suposicao), confirmado (fonte conferiu), contradito (fonte afirma
outra coisa) e nao_confirmado (a fonte consultada nao lista). Regras primeiro; nada aqui calcula probabilidade.
Ordem: tempo (pela data e fuso da sessao) -> municipio -> feriado e territorio, que dependem dos dois anteriores.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta

from aplicacao import EntradaInvalida, previsao_consumo
from aplicacao.personas import extrair
from personas import referencias as ref

CAMPOS_DO_CALCULO = ("dia_semana", "periodo", "grande_regiao", "area")
ESTADOS = ("declarado", "hipotese", "confirmado", "contradito", "nao_confirmado")
ORIGENS = ("regra", "llm", "fonte")
CLIMAS = ("chuva", "frio", "calor")
PRODUTOS = ("pizza", "guarana", "cola", "refrigerante")
_DIAS = {"segunda": 1, "terca": 2, "quarta": 3, "quinta": 4, "sexta": 5, "sabado": 6, "domingo": 7}
_DIA = r"(segunda|terca|quarta|quinta|sexta|sabado|domingo)(?: feira)?"
_CONDICIONAL = r" (?:e se|se|caso|tivesse|houvesse|fosse|imagin\w*|supon\w*) "
_HIPOTETICO = r" (?:considere|considera|suponha|supondo|imagine|hipotes\w*|mesmo assim|faz de conta) "
_CORRECAO = r" (?:na verdade|corrigindo|errei|quis dizer|alias) "


@dataclass(frozen=True)
class Leitura:
    contexto: dict
    historico: list = field(default_factory=list)
    perguntas: list = field(default_factory=list)
    temas: frozenset = frozenset()


def _campo(valor, estado: str, turno: int, origem: str = "regra", **extra) -> dict:
    return {"valor": valor, "estado": estado, "origem": origem, "turno": turno, **extra}


# ---------- tempo ----------

def _proxima(hoje: date, dia: int) -> date:
    return hoje + timedelta(days=(dia - hoje.isoweekday()) % 7 or 7)


def _data_explicita(mensagem: str, hoje: date) -> date | None:
    m = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{4}))?\b", mensagem)
    if not m:
        return None
    try:
        return date(int(m.group(3) or hoje.year), int(m.group(2)), int(m.group(1)))
    except ValueError:
        return None


def _dia_do_mes(texto: str, hoje: date) -> date | None:
    m = re.search(r" dia (\d{1,2}) ", texto)
    if not m:
        return None
    dia = int(m.group(1))
    ano, mes = (hoje.year, hoje.month) if dia >= hoje.day else ((hoje.year + hoje.month // 12), hoje.month % 12 + 1)
    try:
        return date(ano, mes, dia)
    except ValueError:
        return None


def resolver_data(texto: str, mensagem: str, hoje: date, data_anterior: date | None) -> tuple[date | None, bool]:
    """Data citada e se ficou ambigua (dia da semana igual ao de hoje, sem qualificador)."""
    for achar in (lambda: _data_explicita(mensagem, hoje), lambda: _dia_do_mes(texto, hoje)):
        achada = achar()
        if achada:
            return achada, False
    for termo, dias in ((" depois de amanha ", 2), (" amanha ", 1), (" hoje ", 0)):
        if termo in texto:
            return hoje + timedelta(days=dias), False
    m = re.search(rf"(?: proxima )?{_DIA}", texto)
    if not m:
        return None, False
    dia = _DIAS[m.group(1)]
    if re.search(rf"{_DIA} (?:da semana que vem|da proxima semana)", texto):
        return hoje + timedelta(days=7 - hoje.weekday() + dia - 1), False
    if re.search(rf"proxima {_DIA}|{_DIA} que vem", texto):
        return _proxima(hoje, dia), False
    if data_anterior is not None:  # "na verdade e sabado", "e no domingo?": mesma semana da data ja combinada
        return data_anterior + timedelta(days=dia - data_anterior.isoweekday()), False
    if dia == hoje.isoweekday():
        return None, True
    return _proxima(hoje, dia), False


# ---------- interpretacao ----------

class _Montagem:
    def __init__(self, anterior: Mapping, turno: int):
        self.ctx = {k: dict(v) for k, v in anterior.items()}
        self.turno = turno
        self.historico: list[dict] = []
        self.perguntas: list[str] = []
        self.temas: set[str] = set()

    def definir(self, nome: str, novo: dict) -> None:
        antigo = self.ctx.get(nome)
        if antigo is not None and antigo["valor"] != novo["valor"]:
            self.historico.append({"campo": nome, "de": antigo["valor"], "para": novo["valor"], "turno": self.turno})
        if antigo is None or antigo["valor"] != novo["valor"] or antigo["estado"] != novo["estado"]:
            self.ctx[nome] = novo

    def perguntar(self, codigo: str) -> None:
        if codigo not in self.perguntas:
            self.perguntas.append(codigo)


def _tempo(m: _Montagem, texto: str, mensagem: str, hoje: date, base: dict) -> None:
    anterior = date.fromisoformat(m.ctx["data"]["valor"]) if "data" in m.ctx else None
    data, ambigua = resolver_data(texto, mensagem, hoje, anterior)
    if data is not None:
        m.definir("data", _campo(data.isoformat(), "declarado", m.turno))
        m.definir("dia_semana", _campo(data.isoweekday(), "confirmado", m.turno, "fonte", fonte_id="CALENDARIO"))
        return
    if ambigua:
        m.perguntar("data")
    if "dia_semana" in base:
        m.definir("dia_semana", _campo(base["dia_semana"], "declarado", m.turno))


def _municipio_citado(texto: str, base: dict) -> tuple[str | None, str | None, bool]:
    """(codigo do municipio, UF citada sem municipio, se "Sao Paulo" ficou ambiguo)."""
    if re.search(r" (?:estado de|no estado de) (?:sao paulo|sp) ", texto):
        return None, "SP", False
    tem_sp = " sao paulo " in texto or " capital paulista " in texto
    marcador = re.search(r" (?:cidade de|municipio de|capital) sao paulo | sao paulo capital | capital paulista ", texto)
    if tem_sp and (marcador or base.get("area") == "capital"):
        return ref.SP, None, False
    capitais = ref.capitais_no_texto(texto)
    if len(capitais) == 1:
        return capitais[0], None, False
    return None, None, tem_sp


def _lugar(m: _Montagem, texto: str, base: dict) -> None:
    codigo, uf, ambiguo = _municipio_citado(texto, base)
    if ambiguo:
        m.perguntar("cidade_ou_estado")
    if codigo:
        dados = ref.municipio(codigo)
        fonte = "IBGE-LOCALIDADES"
        m.definir("municipio", _campo(codigo, "confirmado", m.turno, "fonte", fonte_id=fonte, nome=dados["nome"]))
        m.definir("uf", _campo(dados["uf"], "confirmado", m.turno, "fonte", fonte_id=fonte))
        m.definir("grande_regiao", _campo(dados["grande_regiao"], "confirmado", m.turno, "fonte", fonte_id=fonte))
        area = "capital" if dados["capital"] else "urbano_nao_capital"
        m.definir("area", _campo(area, "confirmado" if dados["capital"] else "declarado", m.turno, "fonte", fonte_id=fonte))
        if codigo != ref.SP:
            m.ctx.pop("distrito", None)
        return
    if uf:
        m.definir("uf", _campo(uf, "declarado", m.turno))
        m.definir("grande_regiao", _campo(ref.REGIAO_DA_UF[uf], "confirmado", m.turno, "fonte", fonte_id="IBGE-LOCALIDADES"))
    for nome in ("grande_regiao", "area"):
        if nome in base and not (uf and nome == "grande_regiao"):
            m.definir(nome, _campo(base[nome], "declarado", m.turno))


def _territorio(m: _Montagem, texto: str) -> None:
    pede = re.search(r" (?:bairro|distrito|regiao da cidade|zona) ", texto)
    if re.search(r" (?:renda|pobre\w*|ric[oa]s?|classe|salario\w*) ", texto):
        m.temas.add("renda")
    if pede:
        m.temas.add("territorio")
    if m.ctx.get("municipio", {}).get("valor") != ref.SP:
        return
    achados = ref.distritos_no_texto(texto)
    if len(achados) == 1:
        codigo = achados[0]
        m.definir("distrito", _campo(codigo, "confirmado", m.turno, "fonte", fonte_id="IBGE-CENSO2022-RENDA-RESP",
                                     nome=ref.territorio(codigo)["distrito"]["nome"]))
        m.temas.add("territorio")
    elif pede:
        m.perguntar("bairro")


def _feriado(m: _Montagem, texto: str) -> None:
    citado = " feriad" in texto
    anterior = m.ctx.get("feriado")
    if citado:
        m.temas.add("feriado")
    if "data" not in m.ctx:
        if citado:
            m.definir("feriado", _campo(True, "declarado", m.turno))
        return
    oficial = ref.feriado(date.fromisoformat(m.ctx["data"]["valor"]), m.ctx.get("municipio", {}).get("valor"))
    extra = {"fontes_consultadas": oficial["fontes_consultadas"], "cobertura_municipal": oficial["cobertura_municipal"]}
    if oficial["feriado"]:
        if not (anterior and anterior["estado"] == "confirmado"):
            m.temas.add("feriado")
        m.definir("feriado", _campo(True, "confirmado", m.turno, "fonte", nome=oficial["nome"], esfera=oficial["esfera"],
                                    **extra))
    elif citado or (anterior and anterior["estado"] != "confirmado"):
        hipotetico = re.search(_HIPOTETICO, texto) or (anterior or {}).get("estado") == "hipotese"
        m.definir("feriado", _campo(True, "hipotese" if hipotetico else "nao_confirmado", m.turno, **extra))
    elif anterior and anterior["estado"] == "confirmado":
        m.ctx.pop("feriado")


def _condicional(texto: str) -> bool:
    return bool(re.search(_CONDICIONAL, texto))


def _clima_oferta_produtos(m: _Montagem, texto: str, mensagem: str) -> None:
    clima = "chuva" if re.search(r" (?:chuv\w*|chov\w*) ", texto) else ("frio" if " frio " in texto else
                                                                      ("calor" if re.search(r" calor\w* ", texto) else None))
    if clima:
        m.definir("clima", _campo(clima, "hipotese" if _condicional(texto) else "declarado", m.turno))
        m.temas.add("clima")
    pct = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:%|por cento)", mensagem.lower())
    if pct or re.search(r" (?:descont\w*|promo\w*|combos?|cupo\w*|ofert\w*) ", texto):
        valor = {"percentual": float(pct.group(1).replace(",", ".")) if pct else None,
                 "tipo": "combo" if " combo" in texto else "item"}
        m.definir("oferta", _campo(valor, "hipotese" if _condicional(texto) else "declarado", m.turno))
        m.temas.add("desconto")
    achados = [p for p, termos in (("pizza", (" pizza",)), ("guarana", (" guarana",)), ("cola", (" coca", " cola ")),
                                   ("refrigerante", (" refri",))) if any(t in texto for t in termos)]
    if achados:
        atuais = list(m.ctx.get("produtos", {}).get("valor", []))
        m.definir("produtos", _campo(sorted(set(atuais) | set(achados)), "declarado", m.turno))
    if re.search(r" (?:coca\w*|antarctica|marca\w*) ", texto):
        m.temas.add("marca")
    if re.search(r" (?:compra\w*|compraria|pedi\w*|vend\w*) ", texto):
        m.temas.add("compra")


def interpretar(mensagem: str, anterior: Mapping, turno: int, hoje: date) -> Leitura:
    texto = ref.normalizar(mensagem)
    base, ambiguos = extrair(mensagem, hoje)
    m = _Montagem(anterior, turno)
    for campo in ambiguos:
        m.perguntar(campo)
    _tempo(m, texto, mensagem, hoje, base)
    if "periodo" in base:
        m.definir("periodo", _campo(base["periodo"], "declarado", turno))
    _lugar(m, texto, base)
    _territorio(m, texto)
    _feriado(m, texto)
    _clima_oferta_produtos(m, texto, mensagem)
    if re.search(_CORRECAO, texto):
        m.temas.add("correcao")
    return Leitura(m.ctx, m.historico, m.perguntas, frozenset(m.temas))


def faltantes(ctx: Mapping) -> list[str]:
    return [c for c in CAMPOS_DO_CALCULO if c not in ctx]


# ---------- validacao do estado devolvido pela pagina ----------

def _iso(valor) -> bool:
    try:
        date.fromisoformat(valor)
        return True
    except (TypeError, ValueError):
        return False


def _oferta_valida(v) -> bool:
    if not isinstance(v, dict) or set(v) != {"percentual", "tipo"} or v["tipo"] not in ("combo", "item"):
        return False
    pct = v["percentual"]  # 5.0 volta da pagina como 5: o JSON do navegador nao distingue inteiro de decimal
    return pct is None or (type(pct) in (int, float) and 0 < pct <= 100)


_VALIDOS = {
    "data": _iso, "dia_semana": lambda v: type(v) is int and 1 <= v <= 7,
    "periodo": lambda v: v in previsao_consumo.PERIODOS, "grande_regiao": lambda v: v in previsao_consumo.REGIOES,
    "area": lambda v: v in previsao_consumo.AREAS, "municipio": ref.existe_municipio,
    "uf": lambda v: v in ref.REGIAO_DA_UF, "distrito": ref.existe_distrito, "feriado": lambda v: v is True,
    "clima": lambda v: v in CLIMAS, "oferta": _oferta_valida,
    "produtos": lambda v: isinstance(v, list) and set(v) <= set(PRODUTOS),
}


def validar(ctx: object) -> dict:
    """Recusa campo desconhecido, valor fora do vocabulario ou proveniencia invalida (sessao adulterada)."""
    if not isinstance(ctx, Mapping):
        raise EntradaInvalida("contexto deve ser um objeto")
    for nome, item in ctx.items():
        if nome not in _VALIDOS or not isinstance(item, Mapping):
            raise EntradaInvalida(f"campo de contexto desconhecido: {nome}")
        if not _VALIDOS[nome](item.get("valor")) or item.get("estado") not in ESTADOS or item.get("origem") not in ORIGENS:
            raise EntradaInvalida(f"campo de contexto inválido: {nome}")
        if type(item.get("turno")) is not int or item["turno"] < 0:
            raise EntradaInvalida(f"turno inválido em {nome}")
        if "fonte_id" in item and not (isinstance(item["fonte_id"], str) and _FONTE_ID.fullmatch(item["fonte_id"])):
            raise EntradaInvalida(f"fonte inválida em {nome}")
    limpo = {k: {c: v[c] for c in _CHAVES if c in v} for k, v in ctx.items()}
    return _derivados(limpo)


_CHAVES = ("valor", "estado", "origem", "turno", "fonte_id")
_FONTE_ID = re.compile(r"[A-Z0-9-]{1,40}")


def _derivados(ctx: dict) -> dict:
    """Nomes e dados de fonte refeitos pelo servidor: o estado da pagina nunca traz texto livre para a conversa."""
    if "municipio" in ctx:
        ctx["municipio"]["nome"] = ref.municipio(ctx["municipio"]["valor"])["nome"]
    if "distrito" in ctx:
        ctx["distrito"]["nome"] = ref.territorio(ctx["distrito"]["valor"])["distrito"]["nome"]
    if "feriado" in ctx and "data" in ctx:
        oficial = ref.feriado(date.fromisoformat(ctx["data"]["valor"]), ctx.get("municipio", {}).get("valor"))
        ctx["feriado"].update(fontes_consultadas=oficial["fontes_consultadas"],
                              cobertura_municipal=oficial["cobertura_municipal"])
        if ctx["feriado"]["estado"] == "confirmado" and oficial["feriado"]:
            ctx["feriado"].update(nome=oficial["nome"], esfera=oficial["esfera"])
    return ctx
