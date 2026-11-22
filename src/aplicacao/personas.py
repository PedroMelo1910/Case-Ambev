"""Conversa das personas: regras e textos montados sobre os resultados de `previsao_consumo` (POF 2008-2009).

A conversa nao calcula nada por conta propria. O numero vem de `previsao_consumo.avaliar` com publico "todos": a
persona escolhida muda so o texto, porque o modelo atual nao usa composicao da casa e nao ha modelo por perfil
validado. O servidor nao guarda estado; a pagina devolve, a cada mensagem, o contexto ja reunido.

Ordem de uma resposta: perguntas fora do dado (preco, promocao, clima, canal), distribuicao das bebidas, campos que
faltam e, com o contexto completo, o recorte do perfil e a estimativa. As etapas da jornada (`JORNADA`) organizam o que
e medido e o que e hipotese; so a etapa da bebida junto da pizza tem desfecho observado. A simulacao completa da
jornada fica adiada.
"""

import re
import unicodedata
from collections.abc import Mapping
from datetime import date
from types import MappingProxyType

from aplicacao import EntradaInvalida, conferir_inteiro, previsao_consumo, recebido, regras
from occasion.variaveis import faixa_hora

VERSAO = "conversa_personas_v1"
MAXIMO_MENSAGEM = 500
CAMPOS = ("dia_semana", "periodo", "grande_regiao", "area")  # campos de contexto que a avaliacao exige
PERGUNTAS = MappingProxyType({
    "dia_semana": "o dia da semana", "periodo": "o período (manhã, almoço, tarde ou noite)",
    "grande_regiao": "a região (Norte, Nordeste, Centro-Oeste, Sudeste ou Sul) ou o estado",
    "area": "o tipo de área (capital, interior ou zona rural)"})

# Contagens de episodios E2 (pessoa x dia x hora) com pizza na POF 2008-2009, auditoria `episodios_todas_comidas`:
# moradores contados na tabela de moradores da unidade de consumo, todas as idades e todas as condicoes na unidade.
FONTE_DOS_PERFIS = "POF 2008–2009 (IBGE), ocasiões com pizza"
PERSONAS = MappingProxyType({
    "pai_de_familia": MappingProxyType({
        "nome": "Pai de família",
        "definicao": "homem de 30 a 59 anos em unidade de consumo com três ou mais moradores",
        "aproximacao": "O nome é uma aproximação: a contagem não confirma paternidade nem exige criança na casa.",
        "suporte": MappingProxyType({"episodios": 186, "positivos": 23, "pessoas": 178})}),
    "jovem_sozinho": MappingProxyType({
        "nome": "Jovem adulto que mora sozinho",
        "definicao": "pessoa de 18 a 29 anos em unidade de consumo com um único morador",
        "aproximacao": "Nos 8 casos com pizza, o domicílio também tem um só morador.",
        "suporte": MappingProxyType({"episodios": 8, "positivos": 0, "pessoas": 8})}),
})

# Rotulo hierarquico das bebidas nas 1.168 ocasioes com pizza (auditoria k_prep_01 v2). Agua nao foi coletada em 2008.
BEBIDAS = (("Guaraná", 101), ("Outro refrigerante", 554), ("Outra bebida", 245), ("Nenhuma bebida registrada", 268))

# Etapas da jornada no grafo de estados da especificacao. So a bebida junto da pizza tem desfecho na POF.
JORNADA = MappingProxyType({
    "decidir": ("decidir entre cozinhar, comer fora ou pedir", "hipotese"),
    "canal": ("escolher o canal do pedido", "hipotese"),
    "oferta": ("ver vitrine, preço e ofertas", "hipotese"),
    "pizza": ("escolher pizza entre outros pratos", "frequencia_de_consumo"),
    "bebida": ("bebida consumida junto da pizza", "observado"),
    "concluir": ("concluir, trocar ou abandonar o pedido", "hipotese")})

# Perguntas sobre efeitos que os dados nao medem: gatilhos, etapa da jornada, o que falta na POF e o que ajudaria.
FORA_DO_DADO = (
    ("preco", (r"precos?", r"descont\w*", r"barat\w*", r"mais car[oa]s?", r"caros?", r"cupo\w*", r"valor\w*"), "oferta",
     "a POF não registra o preço pago nem os preços das bebidas oferecidas",
     "um teste em que o preço do guaraná é sorteado entre lojas ou semanas, no mesmo dia e horário, com um grupo sem "
     "mudança; um histórico de pedidos com preço, na aba Dados da empresa, mostra associação, não efeito"),
    ("promocao", (r"promo\w*", r"combos?", r"ofert\w*", r"brindes?", r"campanhas?"), "oferta",
     "a POF não registra promoção, combo nem o que estava em oferta no momento",
     "um teste com e sem o combo, sorteado por praça ou semana, comparando a proporção de pedidos de pizza com guaraná"),
    ("clima", (r"chuv\w*", r"chov\w*", r"clima\w*", r"frio", r"calor\w*", r"temperaturas?", r"tempo nublado"), None,
     "a POF não registra o clima do dia de cada ocasião",
     "pedidos com data e local cruzados com a observação meteorológica do mesmo dia, comparando dias parecidos "
     "(mesmo dia da semana e horário); isso mede associação, porque o clima não pode ser sorteado"),
    ("canal", (r"canal", r"canais", r"aplicativos?", r"apps?", r"delivery", r"entregas?", r"ifood", r"whatsapp",
               r"telefone\w*"), "canal",
     "a POF não registra o canal do pedido, e consumo fora de casa não inclui entrega em domicílio",
     "pedidos registrados por canal; para medir o efeito de trocar de canal, a mesma oferta testada em dois canais, "
     "com sorteio de quem recebe cada versão"),
)
_BEBIDAS = ("bebida", "bebidas", "refrigerante", "refrigerantes", "o que bebe", "o que bebem")
_RECOMECAR = ("recomecar", "nova conversa", "comecar de novo")
_SAUDACOES = ("bom dia", "boa tarde", "boa noite")
_SEM_CAMPO = (r"mais tarde", r"mais cedo", r"zona (?:sul|norte|leste|oeste)")  # expressoes que nao sao periodo nem regiao
_ANTES_DE_DURACAO = ("em", "daqui", "a", "de", "por", "uns", "umas")  # "em 3 horas" e duracao, nao horario

_DIAS = (("segunda", 1), ("terca", 2), ("quarta", 3), ("quinta", 4), ("sexta", 5), ("sabado", 6), ("domingo", 7))
_PERIODOS = (("cafe da manha", "manha"), ("manha", "manha"), ("almoco", "almoco"), ("meio dia", "almoco"),
             ("tarde", "tarde"), ("noite", "noite"), ("jantar", "noite"), ("janta", "noite"), ("madrugada", "noite"))
_REGIOES = (("centro oeste", "Centro-Oeste"), ("nordeste", "Nordeste"), ("sudeste", "Sudeste"), ("norte", "Norte"),
            ("sul", "Sul"))
_ESTADOS = {
    "Norte": ("acre", "amapa", "amazonas", "rondonia", "roraima", "tocantins"),
    "Nordeste": ("alagoas", "bahia", "ceara", "maranhao", "paraiba", "pernambuco", "piaui", "rio grande do norte",
                 "sergipe"),
    "Sudeste": ("espirito santo", "minas gerais", "rio de janeiro", "sao paulo"),
    "Sul": ("parana", "rio grande do sul", "santa catarina"),
    "Centro-Oeste": ("distrito federal", "brasilia", "goias", "mato grosso do sul", "mato grosso")}
_SIGLAS_UF = {"Norte": "AC AP AM PA RO RR TO", "Nordeste": "AL BA CE MA PB PE PI RN SE",
              "Sudeste": "ES MG RJ SP", "Sul": "PR RS SC", "Centro-Oeste": "DF GO MS MT"}
_AREAS = (("fora da capital", "urbano_nao_capital"), ("interior", "urbano_nao_capital"), ("zona rural", "rural"),
          ("rural", "rural"), ("roca", "rural"), ("sitio", "rural"), ("fazenda", "rural"), ("capital", "capital"))


def _normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode("ascii")
    return " " + re.sub(r"[^a-z0-9]+", " ", sem_acento).strip() + " "


def _tem(texto: str, termos) -> bool:
    return any(f" {termo} " in texto for termo in termos)


def _casa(texto: str, padroes) -> bool:
    """Algum padrao aparece como palavra inteira (os padroes aceitam variacoes, como plural e conjugacao)."""
    return any(re.search(rf" (?:{padrao}) ", texto) for padrao in padroes)


def _achados(texto: str, termos) -> list:
    """Valores cujos termos aparecem como palavras inteiras; o termo encontrado sai do texto (o mais longo primeiro)."""
    valores = []
    for termo, valor in sorted(termos, key=lambda par: -len(par[0])):
        if f" {termo} " in texto:
            texto = texto.replace(f" {termo} ", " ")
            if valor not in valores:
                valores.append(valor)
    return valores


def _regioes(original: str, texto: str) -> list:
    estados = [(nome, regiao) for regiao, nomes in _ESTADOS.items() for nome in nomes]
    if "pará" in original.lower():  # sem acento, "para" e preposicao
        estados.append(("para", "Norte"))
    achadas = _achados(texto, [*estados, *_REGIOES])
    letras = [c for c in original if c.isalpha()]
    if letras and sum(c.isupper() for c in letras) / len(letras) > 0.6:
        return achadas  # mensagem toda em maiusculas: "SE", "MA" e "PE" seriam palavras, nao siglas
    sem_hora = re.sub(r"\d\s*(?:AM|PM)\b", " ", original)
    siglas = set(re.findall(r"\b[A-Z]{2}\b", sem_hora))  # so em maiusculas: "se", "es" e "ma" sao palavras comuns
    return achadas + [r for r, lista in _SIGLAS_UF.items() if siglas & set(lista.split()) and r not in achadas]


def _horas(texto: str) -> list[int]:
    """Horarios como "20h", "20h30", "as 20" e "12 horas"; "em 3 horas" e "daqui a 2 horas" sao duracao."""
    horas = [int(h) for anterior, h in re.findall(r"(\w+)? (\d{1,2}) ?(?:h|hs|h\d{2}|hora|horas)(?= )", texto)
             if anterior not in _ANTES_DE_DURACAO]
    horas += [int(h) for h in re.findall(r" (?:as|pelas|das) (\d{1,2})(?= )", texto)]
    return [h for h in horas if h <= 23]


def extrair(mensagem: str, hoje: date | None = None) -> tuple[dict, list[str]]:
    """Campos reconhecidos na mensagem e os que vieram ambiguos (mais de um valor). Nada e inventado."""
    texto = _normalizar(mensagem)
    for saudacao in _SAUDACOES:
        texto = texto.replace(f" {saudacao} ", " ")
    for expressao in _SEM_CAMPO:
        texto = re.sub(rf" {expressao} ", " ", texto)
    dias = _achados(texto, _DIAS)
    for palavra, deslocamento in ((" depois de amanha ", 2), (" hoje ", 0), (" amanha ", 1)):
        if palavra in texto:
            texto = texto.replace(palavra, " ")
            dia = ((hoje or date.today()).isoweekday() - 1 + deslocamento) % 7 + 1
            dias += [] if dia in dias else [dia]
    periodos = _achados(texto, _PERIODOS)
    for hora in _horas(texto):
        if faixa_hora(hora) not in periodos:
            periodos.append(faixa_hora(hora))
    encontrados = {"dia_semana": dias, "periodo": periodos, "grande_regiao": _regioes(mensagem, texto),
                   "area": _achados(texto, _AREAS)}
    campos = {campo: valores[0] for campo, valores in encontrados.items() if len(valores) == 1}
    return campos, [campo for campo, valores in encontrados.items() if len(valores) > 1]


def inteiro(valor: int) -> str:
    return f"{valor:,}".replace(",", ".")


def _pct(valor: float) -> str:
    return f"{valor * 100:.1f}%".replace(".", ",")


def _pp(valor: float) -> str:
    sinal = "+" if valor >= 0.05 else ("−" if valor <= -0.05 else "")
    return f"{sinal}{abs(valor):.1f}".replace(".", ",") + " p.p."


def _lista(itens: list[str]) -> str:
    return itens[0] if len(itens) == 1 else f"{', '.join(itens[:-1])} e {itens[-1]}"


def _bloco(tipo: str, titulo: str, *linhas: str) -> dict:
    return {"tipo": tipo, "titulo": titulo, "linhas": [linha for linha in linhas if linha]}


def _validar(pedido: object) -> tuple[str, str, dict]:
    if not isinstance(pedido, Mapping) or set(pedido) != {"persona", "mensagem", "contexto"}:
        raise EntradaInvalida('o pedido deve ter exatamente os campos "persona", "mensagem" e "contexto"')
    persona, mensagem, contexto = pedido["persona"], pedido["mensagem"], pedido["contexto"]
    if type(persona) is not str or persona not in PERSONAS:
        raise EntradaInvalida(f"persona deve ser uma de {list(PERSONAS)}, recebido {recebido(persona)}")
    if type(mensagem) is not str or len(mensagem) > MAXIMO_MENSAGEM:
        raise EntradaInvalida(f"mensagem deve ser um texto de até {MAXIMO_MENSAGEM} caracteres")
    if not isinstance(contexto, Mapping) or not set(contexto) <= set(CAMPOS):
        raise EntradaInvalida(f"contexto aceita só os campos {list(CAMPOS)}")
    vocabulario = {"periodo": previsao_consumo.PERIODOS, "grande_regiao": previsao_consumo.REGIOES,
                   "area": previsao_consumo.AREAS}
    validado = {}
    for campo, valor in contexto.items():
        if campo == "dia_semana":
            validado[campo] = conferir_inteiro(valor, "dia_semana", 1, 7)
        elif type(valor) is not str or valor not in vocabulario[campo]:
            raise EntradaInvalida(f"{campo} deve ser um de {list(vocabulario[campo])}, recebido {recebido(valor)}")
        else:
            validado[campo] = valor
    return persona, mensagem, validado


def _descrever(contexto: dict) -> str:
    return (f"{regras.DIAS[contexto['dia_semana']]}, {regras.PERIODOS[contexto['periodo']]}, "
            f"{contexto['grande_regiao']}, área {regras.AREAS[contexto['area']]}")


def _perfil(persona: str) -> dict:
    dados = PERSONAS[persona]
    s = dados["suporte"]
    if not regras.suficiente(s):
        com_guarana = "nenhuma" if s["positivos"] == 0 else str(s["positivos"])
        return _bloco("sem_suporte", "Falta de dados para este perfil",
                      f"Na POF 2008–2009 há só {s['episodios']} ocasiões com pizza de {dados['definicao']}, "
                      f"{com_guarana} com guaraná.",
                      "É pouco para uma estimativa própria. Nenhuma ocasião com guaraná nesse recorte não quer dizer "
                      "chance zero, e o app não calcula uma faixa para tão poucos casos.",
                      "A estimativa abaixo é a geral do contexto, sem recorte de perfil.")
    return _bloco("perfil", "Recorte descritivo do perfil",
                  f"{dados['nome']} ({dados['definicao']}): {s['episodios']} ocasiões com pizza na POF 2008–2009, "
                  f"{s['positivos']} com guaraná, de {s['pessoas']} pessoas. {dados['aproximacao']}",
                  "Essa contagem é descritiva, sem faixa de incerteza, e não entra no cálculo da estimativa.")


def _estimativa(contexto: dict) -> tuple[dict, dict]:
    pedido = {**contexto, **{campo: previsao_consumo.TODOS for campo in previsao_consumo.CAMPOS_PUBLICO}}
    r = previsao_consumo.avaliar(pedido)
    c, m = r["comparacao"], r["model_metadata"]
    resumo = previsao_consumo.carregar().resumo
    exploratoria = m["papel_do_modelo_contextual"] == "exploratorio"
    nome = "Estimativa exploratória por contexto" if exploratoria else "Estimativa por contexto"
    linhas = [f"Referência histórica: {_pct(c['referencia'])} das ocasiões com pizza tiveram guaraná "
              f"({inteiro(resumo['positivos'])} de {inteiro(resumo['episodios'])}, POF 2008–2009)."]
    if c["incerteza_disponivel"]:
        baixo, alto = c["intervalo_estimativa"]
        linhas += [f"{nome}: {_pct(c['estimativa'])} (faixa de 95% da estimativa: {_pct(baixo)} a {_pct(alto)}).",
                   f"Diferença para o habitual: {_pp(c['diferenca_pp'])} (faixa de 95% da diferença: "
                   f"{_pp(c['intervalo_pp'][0])} a {_pp(c['intervalo_pp'][1])})."]
    else:
        linhas.append(f"{nome}: {_pct(c['estimativa'])}; a incerteza não pôde ser medida.")
    nota = f" {c['nota'][0].upper()}{c['nota'][1:]}." if c["nota"] else ""
    linhas.append(f"Leitura: {regras.FAIXAS_COMPARACAO[c['faixa']]}.{nota} "
                  f"Sugestão por regra: {regras.CATEGORIAS[r['recommendation_category']]}.")
    if exploratoria:
        linhas.append(f"Modelo da estimativa: {m['modelo_contextual']}, exploratório. Na avaliação de 2008–2009, "
                      "nenhum modelo por contexto superou a taxa habitual com segurança; trate como hipótese a testar.")
    linhas.append("Cálculo para o contexto geral (público: todos). A persona não entra no cálculo: o modelo atual não "
                  "usa a composição da casa.")
    numero = {"origem": "previsao_consumo.avaliar", "versao": m["versao"], "modelo": m["modelo_contextual"],
              "papel": m["papel_do_modelo_contextual"], "modelo_de_referencia": m["modelo_selecionado"],
              "contexto_enviado": pedido, "estimativa": c["estimativa"], "faixa": c["faixa"]}
    return _bloco("estimativa", f"Estimativa para {_descrever(contexto)}", *linhas), numero


def _fora_do_dado(texto: str) -> list[dict]:
    return [_bloco("fora_do_dado", "Fora do alcance dos dados",
                   f"Não estimo esse efeito, nem para mais nem para menos: {falta}.",
                   (f"Etapa da jornada: {JORNADA[etapa][0]}. Ela não tem medida nos dados" if etapa else
                    "O clima é uma condição da ocasião, não uma etapa da jornada, e não tem medida nos dados")
                   + "; só a bebida consumida junto da pizza tem.",
                   f"O que ajudaria a medir: {ajuda}.")
            for _, gatilhos, etapa, falta, ajuda in FORA_DO_DADO if _casa(texto, gatilhos)]


def _bebidas() -> dict:
    total = sum(n for _, n in BEBIDAS)
    return _bloco("bebidas", f"O que acompanhou a pizza em {inteiro(total)} ocasiões (POF 2008–2009)",
                  *[f"{nome}: {inteiro(n)} ({_pct(n / total)})" for nome, n in BEBIDAS],
                  "Cada ocasião conta uma vez: guaraná primeiro, depois outro refrigerante, depois outra bebida. Em "
                  "2008 a água não foi registrada, então nenhuma bebida registrada pode incluir quem bebeu água. "
                  "Contagem geral, sem recorte de perfil.")


def _abertura(persona: str) -> dict:
    dados = PERSONAS[persona]
    return _bloco("abertura", f"Persona: {dados['nome']}",
                  f"Recorte na POF 2008–2009: {dados['definicao']}. {dados['aproximacao']}",
                  "Diga o dia, o período, a região e o tipo de área da ocasião com pizza. Também explico o que os "
                  "dados não medem, como preço, promoção, clima e canal.")


def _pergunta(contexto: dict, ambiguos: list[str]) -> dict:
    faltando = [PERGUNTAS[c] for c in CAMPOS if c not in contexto]
    repetidos = _lista([regras.CAMPOS[c].lower() for c in ambiguos]) if ambiguos else ""
    situacao = (f"Para estimar, falta {_lista(faltando)}." if faltando else
                f"Nada foi recalculado; continua valendo {_descrever(contexto)}.")
    return _bloco("pergunta", "Informação que falta" if faltando else "Valor a confirmar",
                  f"Encontrei mais de um valor para {repetidos}; informe só um." if ambiguos else "", situacao)


def _nao_reconhecida(contexto: dict) -> dict:
    return _bloco("pergunta", "Mensagem não reconhecida",
                  "Não encontrei dia, período, região ou tipo de área nessa mensagem, nem uma pergunta sobre preço, "
                  "promoção, clima, canal ou bebidas.",
                  f"O contexto atual é {_descrever(contexto)}. Para outra ocasião, diga o valor que muda.")


def _resposta(persona, contexto, blocos, numero=None) -> dict:
    return {"versao": VERSAO, "persona": persona, "contexto": contexto,
            "faltando": [campo for campo in CAMPOS if campo not in contexto], "mensagens": blocos, "numero": numero}


def conversar(pedido: object, hoje: date | None = None) -> dict:
    """Uma resposta por mensagem. A estimativa so e calculada quando a mensagem traz um campo novo, sem ambiguidade,
    e o contexto fica completo; valor ambiguo nunca recalcula com o valor antigo."""
    persona, mensagem, contexto = _validar(pedido)
    texto = _normalizar(mensagem)
    if mensagem.strip() == "" or _tem(texto, _RECOMECAR):
        return _resposta(persona, {}, [_abertura(persona)])
    novos, ambiguos = extrair(mensagem, hoje)
    contexto = {**contexto, **novos}
    blocos = _fora_do_dado(texto) + ([_bebidas()] if _tem(texto, _BEBIDAS) else [])
    completo = all(campo in contexto for campo in CAMPOS)
    if ambiguos or (not completo and (novos or not blocos)):
        return _resposta(persona, contexto, [*blocos, _pergunta(contexto, ambiguos)])
    if novos:
        estimativa, numero = _estimativa(contexto)
        return _resposta(persona, contexto, [*blocos, _perfil(persona), estimativa], numero)
    return _resposta(persona, contexto, blocos or [_nao_reconhecida(contexto)])


def descricoes() -> list[dict]:
    """Perfis para a pagina: definicao, aproximacao e suporte (contagens descritivas)."""
    return [{"id": chave, "nome": p["nome"], "definicao": p["definicao"], "aproximacao": p["aproximacao"],
             "suporte": dict(p["suporte"]), "suficiente": regras.suficiente(p["suporte"]), "fonte": FONTE_DOS_PERFIS}
            for chave, p in PERSONAS.items()]
