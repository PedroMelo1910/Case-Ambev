"""Regras de leitura dos resultados, comuns ao modo historico e ao modo da empresa.

Tres blocos, em ordem: rotulos em linguagem comum (so texto exibido), comparacao com o habitual (faixa, evidencia e
fatores por campo) e sugestao de priorizacao (regras explicitas sobre a faixa). Os identificadores gravados nas
respostas (`comparacao_v2`, `priorizacao_v2`) sao versoes das regras e nao dependem do nome do arquivo.
"""

from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType

import numpy as np

from aplicacao import EntradaInvalida, recebido


# ---------- rotulos em linguagem comum ----------
# Rotulos em linguagem comum para campos, valores, modelos e faixas exibidos ao usuario.

CAMPOS = MappingProxyType({
    "dia_semana": "Dia da semana", "periodo": "Período", "grande_regiao": "Região", "area": "Tipo de área",
    "faixa_etaria": "Faixa etária", "sexo": "Sexo", "escolaridade": "Escolaridade", "canal": "Canal",
    "uf": "UF", "preco": "Preço", "promocao": "Promoção", "temperatura": "Temperatura"})
DIAS = MappingProxyType({1: "segunda-feira", 2: "terça-feira", 3: "quarta-feira", 4: "quinta-feira", 5: "sexta-feira",
                         6: "sábado", 7: "domingo"})
PERIODOS = MappingProxyType({"manha": "manhã (5h–10h)", "almoco": "almoço (11h–14h)", "tarde": "tarde (15h–18h)",
                             "noite": "noite (19h–4h)"})
AREAS = MappingProxyType({"capital": "capital", "urbano_nao_capital": "urbana fora da capital", "rural": "rural"})
FAIXAS_ETARIAS = MappingProxyType({"10_17": "10 a 17 anos", "18_29": "18 a 29 anos", "30_44": "30 a 44 anos",
                           "45_59": "45 a 59 anos", "60_mais": "60 anos ou mais"})
SEXOS = MappingProxyType({"masculino": "masculino", "feminino": "feminino"})
ESCOLARIDADES = MappingProxyType({"ate_fundamental": "até o fundamental", "medio": "ensino médio", "superior": "superior"})
MODELOS = MappingProxyType({"habitual": "taxa habitual", "frequencia": "frequência por dia e período",
                            "logistica": "regressão logística", "boosting": "gradient boosting"})
GRUPOS_DIA = MappingProxyType({"sexta_a_domingo": "sexta a domingo", "segunda_a_quinta": "segunda a quinta",
                               "sexta": "sexta-feira", "sabado": "sábado", "domingo": "domingo",
                               **{f"dia_{d}": nome for d, nome in DIAS.items()}})
PROMOCAO = MappingProxyType({0: "sem promoção", 1: "com promoção"})
FAIXAS_COMPARACAO = MappingProxyType({"acima_do_habitual": "Acima do habitual",
                                      "sem_diferenca_demonstrada": "Sem diferença demonstrada",
                                      "abaixo_do_habitual": "Abaixo do habitual",
                                      "evidencia_insuficiente": "Evidência insuficiente"})
CATEGORIAS = MappingProxyType({"candidato_a_priorizar": "Candidato a priorizar",
                               "candidato_exploratorio": "Hipótese a testar (estimativa exploratória)",
                               "sem_destaque": "Sem destaque", "baixa_prioridade": "Baixa prioridade",
                               "evidencia_insuficiente": "Não usar para priorizar"})
MOTIVOS = MappingProxyType({"suporte_insuficiente": "Pouca amostra no treino",
                            "incerteza_nao_mensuravel": "Incerteza não mensurável",
                            "resultado_instavel": "Resultado instável"})
PAPEIS_ROTULOS = MappingProxyType({"selecionado": "selecionado na avaliação",
                           "exploratorio": "exploratório (vantagem sobre a taxa habitual não demonstrada)"})
_POR_CAMPO = MappingProxyType({"dia_semana": DIAS, "periodo": PERIODOS, "area": AREAS, "faixa_etaria": FAIXAS_ETARIAS,
                               "sexo": SEXOS, "escolaridade": ESCOLARIDADES, "promocao": PROMOCAO})


def valor(campo: str, bruto: object) -> str:
    return str(_POR_CAMPO.get(campo, {}).get(bruto, bruto))


def filtro(campo: str, bruto: object) -> str:
    """Filtro em linguagem comum, por exemplo 'tipo de área: rural'."""
    return f"{CAMPOS.get(campo, campo).lower()}: {valor(campo, bruto)}"


_DO_PERIODO = MappingProxyType({"manha": "da manhã", "almoco": "do almoço", "tarde": "da tarde", "noite": "da noite"})
_DA_AREA = MappingProxyType({"capital": "de capital", "urbano_nao_capital": "urbana fora da capital", "rural": "rural"})
# Sujeito da frase (nome comum do valor) e o conjunto com que ele e comparado.
_SUJEITOS = MappingProxyType({
    "dia_semana": (lambda v: f"{'O' if v in (6, 7) else 'A'} {DIAS.get(v, v)}", "dos dias da semana"),
    "periodo": (lambda v: f"O período {_DO_PERIODO.get(v, v)}", "dos períodos"),
    "grande_regiao": (lambda v: f"A região {v}", "das regiões"),
    "area": (lambda v: f"A área {_DA_AREA.get(v, v)}", "dos tipos de área"),
    "faixa_etaria": (lambda v: f"A faixa de {FAIXAS_ETARIAS.get(v, v)}", "das faixas etárias"),
    "sexo": (lambda v: f"O sexo {SEXOS.get(v, v)}", "dos dois sexos"),
    "escolaridade": (lambda v: f"A escolaridade {ESCOLARIDADES.get(v, v)}", "dos níveis de escolaridade"),
    "canal": (lambda v: f"O canal {v}", "dos canais"),
    "uf": (lambda v: f"A UF {v}", "das UFs"),
    "promocao": (lambda v: f"Pedido {PROMOCAO.get(v, v)}", "com e sem promoção"),
    "preco": (lambda v: "O preço informado", "dos preços observados"),
    "temperatura": (lambda v: "A temperatura informada", "das temperaturas observadas")})


def _sujeito(fator: dict) -> tuple[str, str]:
    sujeito, conjunto = _SUJEITOS.get(fator["campo"], (lambda v: f"{CAMPOS.get(fator['campo'], fator['campo'])} {v}",
                                                         "desse campo"))
    return sujeito(fator["valor"]), conjunto


def sujeito_do_fator(fator: dict) -> str:
    """Nome comum do valor, em minusculas, por exemplo 'o período da noite'."""
    nome = _sujeito(fator)[0]
    return nome[:1].lower() + nome[1:]


def frase_do_fator(fator: dict) -> str:
    """Frase em nomes comuns, por exemplo: 'O período da noite eleva a estimativa em relação à média dos períodos, no
    mesmo modelo.' Descreve o calculo de `fatores_por_campo` (troca so esse campo), nao causa."""
    nome, conjunto = _sujeito(fator)
    if fator["direcao"] == "neutro":
        return f"{nome} praticamente não altera a estimativa."
    verbo = "eleva" if fator["direcao"] == "eleva" else "reduz"
    return f"{nome} {verbo} a estimativa em relação à média {conjunto}, no mesmo modelo."


def fatores_exibidos(fatores: Sequence[dict], constante: bool = False) -> list[dict]:
    """Fatores que a tela mostra: nenhum para uma taxa constante; senao, so os que movem a estimativa."""
    return [] if constante else [f for f in fatores if f["direcao"] != "neutro"]


# ---------- leitura do cenario em linguagem comum ----------
# Quatro estados distintos, a partir da faixa da comparacao: acima, abaixo, diferenca incerta e evidencia insuficiente.

LEITURAS = MappingProxyType({"acima_do_habitual": "Sinal acima da referência",
                             "abaixo_do_habitual": "Sinal abaixo da referência",
                             "sem_diferenca_demonstrada": "Sem sinal claro para priorizar este cenário",
                             "evidencia_insuficiente": "Dados insuficientes para ler este cenário"})
ESTADOS_DA_LEITURA = MappingProxyType({"acima_do_habitual": "acima", "abaixo_do_habitual": "abaixo",
                                       "sem_diferenca_demonstrada": "incerta",
                                       "evidencia_insuficiente": "insuficiente"})
AREAS_DO_RECORTE = MappingProxyType({"capital": "capitais", "urbano_nao_capital": "áreas urbanas fora das capitais",
                                     "rural": "áreas rurais"})
PROXIMOS_PASSOS = MappingProxyType({
    "candidato_a_priorizar": "Próximo passo: testar uma ação neste cenário com grupo de controle antes de ampliar.",
    "candidato_exploratorio": "Próximo passo: confirmar com dados próprios num teste com grupo de controle; a "
                              "estimativa por cenário é exploratória.",
    "sem_destaque": "Próximo passo: não priorizar este cenário por este número. Se houver uma ação em mente, como um "
                    "desconto, meça num teste com grupo de controle (veja Recomendações).",
    "baixa_prioridade": "Próximo passo: olhar outros cenários; este não sustenta prioridade.",
    "evidencia_insuficiente": "Próximo passo: não usar este número para decidir; reunir dados próprios deste cenário."})
LEITURA_DA_CATEGORIA = MappingProxyType({
    "candidato_a_priorizar": "candidato a priorizar, a confirmar com teste", "candidato_exploratorio": "hipótese a testar",
    "sem_destaque": "sem prioridade demonstrada", "baixa_prioridade": "abaixo da referência, sem base para priorizar",
    "evidencia_insuficiente": "sem base para priorizar"})


def texto_pct(proporcao: float) -> str:
    casas = 2 if 0 < proporcao < 0.01 else 1
    return f"{proporcao * 100:.{casas}f}%".replace(".", ",")


def texto_pp(pontos: float) -> str:
    sinal = "+" if pontos >= 0.05 else "−" if pontos <= -0.05 else ""
    return f"{sinal}{abs(pontos):.1f} p.p.".replace(".", ",", 1)


def recorte(grande_regiao: str, area: str) -> str:
    """Recorte que a estimativa usa, por exemplo 'Sudeste, capitais' (a pesquisa nao identifica cidade nem bairro)."""
    return f"{grande_regiao}, {AREAS_DO_RECORTE[area]}"


def leitura_do_cenario(comparacao: Mapping, medida: str, referencia: str, exploratorio: bool) -> dict:
    """Titulo, linha da medida e uma frase sobre a incerteza, em linguagem comum."""
    faixa = comparacao["faixa"]
    linha = (f"{medida} · {texto_pct(comparacao['estimativa'])} neste cenário · "
             f"{texto_pct(comparacao['referencia'])} {referencia}")
    ic = comparacao.get("intervalo_pp")
    faixa_txt = f"de {texto_pp(ic[0])} a {texto_pp(ic[1])}" if ic else ""
    if faixa == "sem_diferenca_demonstrada":
        explicacao = (f"A diferença é incerta: a faixa de 95% da diferença vai {faixa_txt} e inclui zero, então os dados "
                      "não mostram se este cenário fica acima ou abaixo da referência.")
    elif faixa in ("acima_do_habitual", "abaixo_do_habitual"):
        lado = "acima" if faixa == "acima_do_habitual" else "abaixo"
        explicacao = f"A faixa de 95% da diferença, {faixa_txt}, fica toda {lado} de zero."
        if exploratorio:
            explicacao += " A estimativa por cenário é exploratória: trate como hipótese a testar."
    else:
        nota = comparacao.get("nota")
        explicacao = "Não há base suficiente para comparar com a referência" + (f": {nota}." if nota else ".")
    return {"estado": ESTADOS_DA_LEITURA[faixa], "titulo": LEITURAS[faixa], "linha": linha, "explicacao": explicacao}


# ---------- comparacao com o habitual e fatores ----------
# Leitura de uma estimativa: comparacao com o habitual (diferenca, intervalo, faixa) e fatores por campo.
#
# Faixas (versao `comparacao_v2`):
# - `acima_do_habitual` / `abaixo_do_habitual`: o intervalo inteiro da diferenca (95%, modelos reamostrados) fica acima /
#   abaixo de zero;
# - `sem_diferenca_demonstrada`: o intervalo inclui zero. Nao demonstra proximidade nem igualdade com o habitual;
# - `evidencia_insuficiente`, com os motivos que se aplicam: `suporte_insuficiente` (contagem do treino abaixo dos minimos
#   nos campos contados), `incerteza_nao_mensuravel` (a estrutura dos dados nao permite reamostrar) e `resultado_instavel`
#   (a diferenca pontual cai fora do intervalo das reamostras).
#
# O suporte conta so os campos declarados em `alcance["considerados"]`; filtros fora dessa contagem ficam em
# `alcance["nao_cobertos"]`. Os minimos sao um piso para tentar a comparacao, nao garantia de confiabilidade. Sem
# incerteza mensuravel, o intervalo calculado nao tem validade e nao e devolvido.
#
# Fatores por campo: a estimativa do contexto e comparada com a media das estimativas obtidas trocando so esse campo por
# cada valor observado no treino, ponderada pela frequencia desses valores. Efeito em pontos percentuais; descreve
# associacao no modelo, nao causa. Funciona com qualquer modelo que devolva probabilidades.

VERSAO_COMPARACAO = "comparacao_v2"
MINIMO_EPISODIOS = 30
MINIMO_POSITIVOS = 3
NIVEL = 0.95
FAIXAS = ("acima_do_habitual", "sem_diferenca_demonstrada", "abaixo_do_habitual", "evidencia_insuficiente")
LIMITES = (
    # textos simples; o sentido tecnico (sem garantia simultanea; reamostragem com modelo, variaveis e
    # hiperparametro fixos, que nao mede sobreajuste) fica na documentacao
    "cada cenário é avaliado sozinho: ao consultar muitos cenários, alguns vão aparecer acima ou abaixo do habitual "
    "por acaso",
    "a faixa de incerteza mostra quanto a estimativa varia entre amostras dos mesmos dados, sempre com o mesmo "
    "modelo; ela não mede se o modelo se ajusta demais a esses dados",
    "associação não comprova efeito de uma ação, aumento de vendas nem retorno financeiro")
SIGNIFICADO_DO_MINIMO = "piso para tentar a comparação; atingir o mínimo não garante que a estimativa seja confiável"


def intervalo(valores, nivel: float = NIVEL) -> tuple[float, float]:
    """Intervalo percentil das diferencas obtidas com os modelos reamostrados."""
    cauda = (1 - nivel) / 2
    baixo, alto = np.quantile(np.asarray(valores, dtype=float), [cauda, 1 - cauda])
    return float(baixo), float(alto)


def suficiente(suporte: dict, minimo_episodios: int = MINIMO_EPISODIOS, minimo_positivos: int = MINIMO_POSITIVOS) -> bool:
    return suporte["episodios"] >= minimo_episodios and suporte["positivos"] >= minimo_positivos


def somar_suporte(suporte: dict, prefixo: str) -> dict:
    """Contagem de uma celula mais larga: soma as celulas cuja chave comeca com `prefixo` (por exemplo, todas as
    regioes de um dia e periodo, quando o modelo nao usa regiao)."""
    partes = [v for chave, v in suporte.items() if chave.startswith(prefixo)]
    return {"episodios": sum(v["episodios"] for v in partes), "positivos": sum(v["positivos"] for v in partes)}


def _motivos(suporte: dict, contados: list, dentro: bool, incerteza_disponivel: bool,
             minimo_episodios: int, minimo_positivos: int) -> list[tuple[str, str]]:
    motivos = []
    if not suficiente(suporte, minimo_episodios, minimo_positivos):
        onde = f" com {', '.join(contados)}" if contados else ""
        motivos.append(("suporte_insuficiente",
                        f"{suporte['episodios']} ocasiões no treino{onde}, {suporte['positivos']} com guaraná; o mínimo "
                        f"para comparar é {minimo_episodios} e {minimo_positivos}"))
    if not incerteza_disponivel:
        motivos.append(("incerteza_nao_mensuravel", "a estrutura dos dados não permite medir a incerteza da diferença"))
    elif not dentro:
        motivos.append(("resultado_instavel",
                        "a diferença estimada cai fora do intervalo das reamostras: resultado instável, ou reamostragem "
                        "pouco confiável para este contexto"))
    return motivos


def _faixa(baixo: float, alto: float) -> str:
    if baixo > 0:
        return "acima_do_habitual"
    if alto < 0:
        return "abaixo_do_habitual"
    return "sem_diferenca_demonstrada"


def comparar(estimativa: float, referencia: float, ic: tuple[float, float], suporte: dict, alcance: dict | None = None,
             minimo_episodios: int = MINIMO_EPISODIOS, minimo_positivos: int = MINIMO_POSITIVOS,
             incerteza_disponivel: bool = True) -> dict:
    baixo, alto = ic
    diferenca = estimativa - referencia
    contados = list((alcance or {}).get("considerados", []))
    dentro = bool(baixo <= diferenca <= alto)
    motivos = _motivos(suporte, contados, dentro, incerteza_disponivel, minimo_episodios, minimo_positivos)
    faixa = "evidencia_insuficiente" if motivos else _faixa(baixo, alto)
    return {"versao": VERSAO_COMPARACAO, "faixa": faixa, "motivos": [codigo for codigo, _ in motivos],
            "nota": "; ".join(texto for _, texto in motivos) or None,
            "estimativa": float(estimativa), "referencia": float(referencia), "diferenca_pp": diferenca * 100,
            "diferenca_relativa": diferenca / referencia if referencia > 0 else None,
            "intervalo_pp": [baixo * 100, alto * 100] if incerteza_disponivel else None, "nivel_do_intervalo": NIVEL,
            "incerteza_disponivel": incerteza_disponivel,
            "diferenca_dentro_do_intervalo": dentro if incerteza_disponivel else None,
            "suporte": {"episodios": suporte["episodios"], "positivos": suporte["positivos"],
                        "campos_considerados": contados,
                        "filtros_nao_cobertos": list((alcance or {}).get("nao_cobertos", []))},
            "minimos_de_suporte": {"episodios": minimo_episodios, "positivos": minimo_positivos,
                                   "significado": SIGNIFICADO_DO_MINIMO},
            "limites": list(LIMITES)}


# ---------- fatores por campo ----------

LIMIAR_NEUTRO_PP = 0.1
MAXIMO_FATORES = 4  # fatores mostrados por resposta


def _media_prevista(prever: Callable[[list[dict]], np.ndarray], registros: Sequence[dict], pesos: Sequence[float]) -> float:
    return float(np.dot(pesos, prever(list(registros))))


def fatores_por_campo(prever: Callable[[list[dict]], np.ndarray], registros: Sequence[dict], pesos: Sequence[float],
              campos: Sequence[str], distribuicoes: Mapping[str, Mapping[object, int]]) -> list[dict]:
    base = _media_prevista(prever, registros, pesos)
    saida = []
    for campo in campos:
        contagens = distribuicoes[campo]
        total = sum(contagens.values())
        alternativa = sum(n / total * _media_prevista(prever, [{**r, campo: valor} for r in registros], pesos)
                          for valor, n in contagens.items())
        efeito = (base - alternativa) * 100
        direcao = "neutro" if abs(efeito) < LIMIAR_NEUTRO_PP else ("eleva" if efeito > 0 else "reduz")
        saida.append({"campo": campo, "valor": registros[0][campo], "efeito_pp": efeito, "direcao": direcao})
    return sorted(saida, key=lambda f: -abs(f["efeito_pp"]))


# ---------- sugestao de priorizacao ----------
# Sugestao de priorizacao a partir da faixa em relacao ao habitual e do papel do modelo contextual.
#
# Regras explicitas, em ordem; a primeira que vale define a saida. Nao estimam retorno, volume nem efeito de campanha.
# A justificativa recebe os motivos reais da evidencia insuficiente e os filtros que a contagem de suporte nao cobre.

VERSAO_PRIORIZACAO = "priorizacao_v2"
PAPEIS = ("selecionado", "exploratorio")
_ACIMA = ("Cada cenário é avaliado sozinho, e a faixa de incerteza mantém o mesmo modelo. Associação não comprova "
          "efeito de uma ação nem retorno.")

REGRAS = (
    ("P1", lambda f, p: f == "evidencia_insuficiente", "evidencia_insuficiente",
     "Não use esta estimativa para priorizar."),
    ("P2", lambda f, p: f == "acima_do_habitual" and p == "selecionado", "candidato_a_priorizar",
     "A chance estimada está acima do habitual, com o intervalo da diferença inteiro acima de zero. "
     "Contexto candidato a uma ação, a ser testada antes de ampliar. " + _ACIMA),
    ("P3", lambda f, p: f == "acima_do_habitual", "candidato_exploratorio",
     "A estimativa por contexto está acima do habitual, mas vem de um modelo exploratório: na avaliação, sua vantagem "
     "sobre a taxa habitual não foi demonstrada, e o intervalo não mede o quanto ele se ajusta demais aos dados. "
     "Trate como hipótese a verificar com dados próprios. " + _ACIMA),
    ("P4", lambda f, p: f == "sem_diferenca_demonstrada", "sem_destaque",
     "O intervalo da diferença inclui zero: os dados não demonstram diferença em relação ao habitual, o que não quer "
     "dizer que a chance seja igual. Não há base para destacar este contexto."),
    ("P5", lambda f, p: f == "abaixo_do_habitual", "baixa_prioridade",
     "A chance estimada está abaixo do habitual, com o intervalo da diferença inteiro abaixo de zero. "
     "Os dados não sustentam priorizar este contexto; isso não prova ausência de oportunidade."),
)


def _complementos(identificador: str, papel: str, motivos: Sequence[str], nao_cobertos: Sequence[str]) -> str:
    if identificador == "P1":
        return f" Motivo: {'; '.join(motivos)}." if motivos else ""
    extras = ""
    if identificador == "P5" and papel == "exploratorio":
        extras += " A estimativa vem de um modelo exploratório."
    if identificador in ("P2", "P3", "P5"):
        extras += " Atingir o mínimo de ocasiões no treino não garante confiabilidade."
    if nao_cobertos and identificador in ("P2", "P3", "P5"):
        extras += (f" A contagem de ocasiões do treino não filtra {', '.join(nao_cobertos)}; para esse recorte, o "
                   "resultado depende do modelo.")
    return extras


def decidir(faixa: str, papel: str, motivos: Sequence[str] = (), nao_cobertos: Sequence[str] = ()) -> dict:
    if papel not in PAPEIS:
        raise EntradaInvalida(f"papel do modelo deve ser um de {list(PAPEIS)}, recebido {recebido(papel)}")
    for identificador, condicao, categoria, justificativa in REGRAS:
        if condicao(faixa, papel):
            return {"regra": identificador, "categoria": categoria, "versao": VERSAO_PRIORIZACAO,
                    "justificativa": justificativa + _complementos(identificador, papel, motivos, nao_cobertos)}
    raise EntradaInvalida(f"faixa desconhecida para a priorização: {recebido(faixa)}")
