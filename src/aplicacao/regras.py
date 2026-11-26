"""Decision Engine e leitura dos resultados do Occasion Score observado (pizza e guarana na mesma refeicao).

Tres blocos, em ordem: rotulos em linguagem comum (so texto exibido), comparacao com a media (faixa, evidencia e
frase dos fatores) e a categoria de ativacao (regras explicitas sobre a faixa e os dois componentes do score). Os
identificadores gravados nas respostas (`comparacao_v2`, `ativacao_v3`) sao versoes das regras.
"""

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

import preparo

# ---------- rotulos em linguagem comum ----------

CAMPOS = MappingProxyType({
    "dia_semana": "Dia da semana", "periodo": "Período", "horario": "Horário", "grande_regiao": "Região", "area": "Tipo de área",
    "faixa_etaria": "Faixa etária", "sexo": "Sexo", "escolaridade": "Escolaridade"})
DIAS = MappingProxyType({1: "segunda-feira", 2: "terça-feira", 3: "quarta-feira", 4: "quinta-feira", 5: "sexta-feira",
                         6: "sábado", 7: "domingo"})
PERIODOS = MappingProxyType({"manha": "manhã (5h–10h)", "almoco": "almoço (11h–14h)", "tarde": "tarde (15h–18h)",
                             "noite": "noite (19h–4h)"})
AREAS = MappingProxyType({"capital": "capital", "urbano_nao_capital": "urbana fora da capital", "rural": "rural"})
FAIXAS_ETARIAS = MappingProxyType({"10_17": "10 a 17 anos", "18_29": "18 a 29 anos", "30_44": "30 a 44 anos",
                                   "45_59": "45 a 59 anos", "60_mais": "60 anos ou mais"})
SEXOS = MappingProxyType({"masculino": "masculino", "feminino": "feminino"})
ESCOLARIDADES = MappingProxyType({"ate_fundamental": "até o fundamental", "medio": "ensino médio", "superior": "superior"})
MODELOS = MappingProxyType({"global": "taxa média", "dia_periodo": "frequência por dia e período",
                            "logistica": "regressão logística", "dois_passos": "modelo em dois passos (pizza × guaraná)",
                            "boosting": "gradient boosting"})
FAIXAS_COMPARACAO = MappingProxyType({"acima_do_habitual": "Acima da média",
                                      "sem_diferenca_demonstrada": "Sem diferença demonstrada",
                                      "abaixo_do_habitual": "Abaixo da média",
                                      "evidencia_insuficiente": "Evidência insuficiente"})
CATEGORIAS = MappingProxyType({"testar_combo_no_contexto": "Testar um combo de pizza com guaraná neste contexto",
                               "sem_destaque": "Sem destaque", "baixa_prioridade": "Baixa prioridade",
                               "nao_priorizar": "Não usar para priorizar"})
MOTIVOS = MappingProxyType({"suporte_insuficiente": "Pouca amostra no treino",
                            "incerteza_nao_mensuravel": "Incerteza não mensurável",
                            "resultado_instavel": "Resultado instável",
                            "recorte_raro": "Combinação rara no treino"})
_POR_CAMPO = MappingProxyType({"dia_semana": DIAS, "periodo": PERIODOS, "area": AREAS, "faixa_etaria": FAIXAS_ETARIAS,
                               "sexo": SEXOS, "escolaridade": ESCOLARIDADES})
AVISOS_OBSERVADO = (
    "Estimativa de consumo observado na POF/IBGE 2017–2018 (refeições com comida de moradores de 10 anos ou mais): não "
    "mede pedidos, vendas, canal, preço nem a marca Guaraná Antarctica.",
    "Refrigerante registrado sem sabor conta como guaraná não identificado (zero).",
    "A média e os percentuais são da amostra, sem o peso de expansão do IBGE; ponderada, a taxa média do treino sobe "
    "cerca de um terço (ver documentação). Use os números para comparar contextos, não como tamanho de mercado.")


def valor(campo: str, bruto: object) -> str:
    return str(_POR_CAMPO.get(campo, {}).get(bruto, bruto))


def filtro(campo: str, bruto: object) -> str:
    """Filtro em linguagem comum, por exemplo 'tipo de área: rural'."""
    return f"{CAMPOS.get(campo, campo).lower()}: {valor(campo, bruto)}"


_DO_PERIODO = MappingProxyType({"manha": "da manhã", "almoco": "do almoço", "tarde": "da tarde", "noite": "da noite"})
_DA_AREA = MappingProxyType({"capital": "de capital", "urbano_nao_capital": "urbana fora da capital", "rural": "rural"})
_SUJEITOS = MappingProxyType({
    "dia_semana": (lambda v: f"{'O' if v in (6, 7) else 'A'} {DIAS.get(v, v)}", "dos dias da semana"),
    "periodo": (lambda v: f"O período {_DO_PERIODO.get(v, v)}", "dos períodos"),
    "grande_regiao": (lambda v: f"A região {v}", "das regiões"),
    "area": (lambda v: f"A área {_DA_AREA.get(v, v)}", "dos tipos de área"),
    "faixa_etaria": (lambda v: f"A faixa de {FAIXAS_ETARIAS.get(v, v)}", "das faixas etárias"),
    "sexo": (lambda v: f"O sexo {SEXOS.get(v, v)}", "dos dois sexos"),
    "escolaridade": (lambda v: f"A escolaridade {ESCOLARIDADES.get(v, v)}", "dos níveis de escolaridade")})


_PERIODO_CURTO = MappingProxyType({"manha": "manhã", "almoco": "almoço", "tarde": "tarde", "noite": "noite"})


def _sujeito(fator: dict) -> tuple[str, str]:
    if fator["campo"] == "horario":  # hora exata: a troca muda hora e periodo juntos (ver modelagem.fatores)
        return (f"O horário das {int(fator['valor']):02d}h ({_PERIODO_CURTO.get(fator['periodo'], fator['periodo'])})",
                "dos horários de todos os períodos (hora e período juntos)")
    sujeito, conjunto = _SUJEITOS.get(fator["campo"], (lambda v: f"{CAMPOS.get(fator['campo'], fator['campo'])} {v}",
                                                         "desse campo"))
    return sujeito(fator["valor"]), conjunto


def frase_do_fator(fator: dict) -> str:
    """Frase em nomes comuns, por exemplo: 'O período da noite eleva a estimativa em relação à média dos períodos, no
    mesmo modelo.' Descreve a troca so desse campo no modelo, nao causa."""
    nome, conjunto = _sujeito(fator)
    if fator["direcao"] == "neutro":
        return f"{nome} praticamente não altera a estimativa."
    verbo = "eleva" if fator["direcao"] == "eleva" else "reduz"
    return f"{nome} {verbo} a estimativa em relação à média {conjunto}, no mesmo modelo."


# ---------- leitura do cenario em linguagem comum ----------

LEITURAS = MappingProxyType({"acima_do_habitual": "Acima da média neste contexto",
                             "abaixo_do_habitual": "Abaixo da média neste contexto",
                             "sem_diferenca_demonstrada": "Diferença não demonstrada para este contexto",
                             "evidencia_insuficiente": "Use a média como ponto de partida"})
ESTADOS_DA_LEITURA = MappingProxyType({"acima_do_habitual": "acima", "abaixo_do_habitual": "abaixo",
                                       "sem_diferenca_demonstrada": "incerta", "evidencia_insuficiente": "insuficiente"})
AREAS_DO_RECORTE = MappingProxyType({"capital": "capitais", "urbano_nao_capital": "áreas urbanas fora das capitais",
                                     "rural": "áreas rurais"})
PROXIMOS_PASSOS = MappingProxyType({
    "testar_combo_no_contexto": "Próximo passo: oferecer o combo a uma parte do público deste contexto e comparar com "
                                "um grupo de controle sem a oferta, medindo pizza com guaraná nos dois grupos antes "
                                "de ampliar.",
    "sem_destaque": "Próximo passo: não priorizar este contexto por este número.",
    "baixa_prioridade": "Próximo passo: olhar outros contextos; este não sustenta prioridade.",
    "nao_priorizar": "Próximo passo: não usar este número para decidir; reunir dados próprios deste contexto."})


def texto_pct(proporcao: float) -> str:
    casas = 3 if 0 < proporcao < 0.001 else 2 if proporcao < 0.01 else 1
    return f"{proporcao * 100:.{casas}f}%".replace(".", ",")


def texto_pp(pontos: float) -> str:
    sinal = "+" if pontos > 0 else "−" if pontos < 0 else ""
    return f"{sinal}{abs(pontos):.3f} p.p.".replace(".", ",", 1)


def recorte(grande_regiao: str, area: str) -> str:
    """Recorte que a estimativa usa, por exemplo 'Sudeste, capitais' (a pesquisa nao identifica cidade nem bairro)."""
    return f"{grande_regiao}, {AREAS_DO_RECORTE[area]}"


def leitura_do_cenario(comparacao: Mapping, medida: str, referencia: str, exploratorio: bool) -> dict:
    """Leitura para a tela: titulo, destaque (qual numero e o principal), linha curta, mensagem e detalhe tecnico."""
    faixa = comparacao["faixa"]
    est, ref = texto_pct(comparacao["estimativa"]), texto_pct(comparacao["referencia"])
    ic = comparacao.get("intervalo_pp")
    faixa_txt = f"de {texto_pp(ic[0])} a {texto_pp(ic[1])}" if ic else ""
    nota = comparacao.get("nota")
    vezes = (f" ({comparacao['estimativa'] / comparacao['referencia']:.1f}× a média)".replace(".", ",")
             if comparacao["referencia"] > 0 else "")
    if faixa in ("acima_do_habitual", "abaixo_do_habitual"):
        lado = "acima" if faixa == "acima_do_habitual" else "abaixo"
        destaque, detalhe = "estimativa", ""
        linha = f"{medida}: {est} das refeições neste contexto{vezes}, contra {ref} {referencia}."
        mensagem = (f"Neste contexto, a chance estimada fica {lado} da média ({ref}). A faixa de 95% da diferença, "
                    f"{faixa_txt}, fica toda {lado} de zero.")
        if exploratorio:
            mensagem += " A estimativa por contexto é exploratória: trate como hipótese a testar."
    elif faixa == "sem_diferenca_demonstrada":
        destaque = "estimativa"
        linha = f"{medida}: {est} das refeições neste contexto, contra {ref} {referencia}."
        mensagem = (f"Os dados não demonstram diferença em relação à média ({ref}), o que não quer dizer que a chance "
                    "seja igual. Use a média como ponto de partida.")
        detalhe = f"A faixa de 95% da diferença vai {faixa_txt} e inclui zero." if faixa_txt else ""
    else:
        destaque = "referencia"
        linha = f"{ref} das refeições com comida da amostra de treino (POF 2017–2018) tiveram pizza e guaraná juntos."
        mensagem = "Para esse contexto, vale usar a média como ponto de partida."
        detalhe = ("Não há base suficiente para ler só este contexto" + (f": {nota}." if nota else ".")
                   + f" A estimativa seria {est}, mas o próprio sistema não a sustenta aqui.")
    return {"estado": ESTADOS_DA_LEITURA[faixa], "titulo": LEITURAS[faixa], "destaque": destaque, "linha": linha,
            "mensagem": mensagem, "detalhe": detalhe, "explicacao": mensagem}


# ---------- comparacao com a media ----------
# Faixas (versao `comparacao_v2`): `acima_do_habitual` / `abaixo_do_habitual` quando o intervalo de 95% da diferenca
# (modelos reajustados em reamostras de UPAs) fica todo acima / abaixo de zero; `sem_diferenca_demonstrada` quando
# inclui zero; `evidencia_insuficiente` com os motivos `suporte_insuficiente` (contagem do dia x periodo no treino
# abaixo dos minimos), `incerteza_nao_mensuravel` e `resultado_instavel` (diferenca pontual fora das reamostras).

VERSAO_COMPARACAO = "comparacao_v2"
MINIMO_EPISODIOS = 30
MINIMO_POSITIVOS = 3
NIVEL = 0.95
LIMIAR_NEUTRO_PP_OBSERVADO = 0.006  # cerca de 10% da taxa media (0,063%): efeitos menores sao lidos como neutros
MAXIMO_FATORES = 4
LIMITES = (
    "cada contexto é avaliado sozinho: ao consultar muitos contextos, alguns vão aparecer acima ou abaixo da média "
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


def _milhar(n: int) -> str:
    """Inteiro com ponto de milhar, como no resto da tela (10.425)."""
    return f"{int(n):,}".replace(",", ".")


def suficiente(suporte: dict, minimo_episodios: int = MINIMO_EPISODIOS, minimo_positivos: int = MINIMO_POSITIVOS) -> bool:
    return suporte["episodios"] >= minimo_episodios and suporte["positivos"] >= minimo_positivos


def _motivos(suporte: dict, contados: list, dentro: bool, incerteza_disponivel: bool,
             minimo_episodios: int, minimo_positivos: int) -> list[tuple[str, str]]:
    motivos = []
    if not suficiente(suporte, minimo_episodios, minimo_positivos):
        onde = f" com {', '.join(contados)}" if contados else ""
        motivos.append(("suporte_insuficiente",
                        f"{_milhar(suporte['episodios'])} refeições no treino{onde}, {_milhar(suporte['positivos'])} "
                        f"com pizza e guaraná; para comparar, são precisas pelo menos {_milhar(minimo_episodios)} "
                        f"refeições e {_milhar(minimo_positivos)} com pizza e guaraná"))
    if not incerteza_disponivel:
        motivos.append(("incerteza_nao_mensuravel", "a estrutura dos dados não permite medir a incerteza da diferença"))
    elif not dentro:
        motivos.append(("resultado_instavel", "a diferença estimada cai fora do intervalo das reamostras"))
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


def marcar_recorte_raro(comparacao: Mapping, refeicoes: int, minimo: int = MINIMO_EPISODIOS) -> dict:
    """Combinacao de regiao, area e publico com menos de `minimo` refeicoes no treino: a estimativa vem de efeitos
    aprendidos em outras combinacoes, entao a leitura passa a evidencia insuficiente (nova copia, sem alterar a
    original)."""
    texto = (f"{refeicoes} refeições no treino com essa região, área e público; abaixo de {minimo}, a estimativa soma "
             "efeitos aprendidos em outras combinações")
    nota = "; ".join(t for t in (comparacao.get("nota"), texto) if t)
    return {**comparacao, "faixa": "evidencia_insuficiente", "motivos": [*comparacao["motivos"], "recorte_raro"],
            "nota": nota}


# ---------- categoria de ativacao (Decision Engine) ----------
# Regras explicitas, em ordem; a primeira que vale define a saida. O score diz se o contexto e relevante para a
# ocasiao; nao estima retorno, volume, canal nem efeito de campanha.
#   A1 evidencia insuficiente     -> nao_priorizar             (usar a media; reunir dados proprios)
#   A2 acima da media             -> testar_combo_no_contexto  (hipotese a testar com grupo de controle)
#   A4 diferenca nao demonstrada  -> sem_destaque
#   A5 abaixo da media            -> baixa_prioridade
# v3 (25/11/2026), definida depois da abertura do teste e sem treino novo: a v2 rotulava A2 como "combo para consumo
# em casa" citando que 68% dos positivos foram em casa, mas ~90% de todas as refeicoes e ~73% das refeicoes com pizza
# tambem foram em casa; a comparacao nao sustenta priorizar o canal domestico. A3 (awareness: guarana abaixo da media
# entre as pizzas) segue inativa: a POF nao mede a marca e o componente P(guarana | pizza) nao generalizou no teste.
VERSAO_ATIVACAO = "ativacao_v3"
REGRAS_INATIVAS = MappingProxyType({
    "A3": "awareness ou afinidade com a marca: exige medir a marca (a POF não mede) e um componente guaraná | pizza "
          "que generalize (não generalizou no teste reservado)"})
_SEM_CAUSA = " Associação observada não comprova efeito de uma ação nem retorno."


def _motivo_acima(comparacao: Mapping, componentes: Mapping | None) -> str:
    est, ref = comparacao["estimativa"], comparacao["referencia"]
    vezes = f" ({est / ref:.1f}× a média)".replace(".", ",") if ref > 0 else ""
    ic = comparacao.get("intervalo_pp")
    faixa = f", com a faixa de 95% da diferença de {texto_pp(ic[0])} a {texto_pp(ic[1])}, inteira acima de zero" if ic else ""
    pizza = (f" No mesmo modelo, a pizza aparece em {texto_pct(componentes['pizza'])} das refeições deste contexto, "
             f"contra {texto_pct(componentes['pizza_referencia'])} na média.") if componentes else ""
    return (f"Neste contexto, a chance estimada de pizza e guaraná na mesma refeição é {texto_pct(est)}{vezes}, contra "
            f"{texto_pct(ref)} na média{faixa}.{pizza} Isso sustenta testar uma oferta aqui, não escolher o canal nem "
            "prever quanto ela venderá." + _SEM_CAUSA)


def decidir_ativacao(comparacao: Mapping, componentes: Mapping | None) -> dict:
    """Categoria e justificativa a partir da comparacao do proprio contexto (faixa, estimativa, media e intervalo)."""
    faixa = comparacao["faixa"]
    if faixa == "evidencia_insuficiente":
        regra, categoria = "A1", "nao_priorizar"
        nota = comparacao.get("nota")
        texto = ("Não use esta estimativa para priorizar" + (f" ({nota})" if nota else "")
                 + ". Use a média das refeições como ponto de partida para este contexto.")
    elif faixa == "acima_do_habitual":
        regra, categoria, texto = "A2", "testar_combo_no_contexto", _motivo_acima(comparacao, componentes)
    elif faixa == "sem_diferenca_demonstrada":
        regra, categoria = "A4", "sem_destaque"
        texto = ("O intervalo da diferença inclui zero: os dados não demonstram diferença em relação à média, o que não "
                 "quer dizer que a chance seja igual. Não há base para destacar este contexto.")
    elif faixa == "abaixo_do_habitual":
        regra, categoria = "A5", "baixa_prioridade"
        texto = ("A chance estimada está abaixo da média, com o intervalo da diferença inteiro abaixo de zero. Os dados "
                 "não sustentam priorizar este contexto; isso não prova ausência de oportunidade.")
    else:
        raise preparo.EntradaInvalida(f"faixa desconhecida para a ativação: {preparo.recebido(faixa)}")
    return {"regra": regra, "categoria": categoria, "versao": VERSAO_ATIVACAO, "justificativa": texto}
