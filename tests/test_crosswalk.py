"""K-PREP-03: correspondencia entre as edicoes da POF, um teste por mapeamento (sem tocar no alvo de 2017)."""

import duckdb
import pytest

import preparo as cw
from coleta import Variavel

# --- mapeamentos de codigo ------------------------------------------------------------------------


def test_dia_da_semana_2008_numerico_1_segunda_a_7_domingo():
    assert [cw.dia_semana_numero("2008-2009", str(i)) for i in range(1, 8)] == [1, 2, 3, 4, 5, 6, 7]


@pytest.mark.parametrize("texto,numero", [("Segunda-Feira", 1), ("Terca-Feira", 2), ("Terça-Feira", 2), ("Quarta-Feira", 3),
                                          ("Quinta-Feira", 4), ("Sexta-Feira", 5), ("Sabado", 6), ("Sábado", 6),
                                          ("Domingo", 7), ("  domingo ", 7)])
def test_dia_da_semana_2017_texto_com_ou_sem_acento(texto, numero):
    assert cw.dia_semana_numero("2017-2018", texto) == numero


@pytest.mark.parametrize("edicao,valor", [("2008-2009", "0"), ("2008-2009", "8"), ("2008-2009", "Domingo"),
                                          ("2017-2018", "Feriado"), ("2017-2018", "7")])
def test_dia_da_semana_invalido_e_recusado(edicao, valor):
    with pytest.raises(ValueError):
        cw.dia_semana_numero(edicao, valor)


@pytest.mark.parametrize("edicao", ["2008-2009", "2017-2018"])
def test_sexo_codigos_1_e_2_nas_duas_edicoes(edicao):
    assert (cw.sexo_padrao(edicao, "1"), cw.sexo_padrao(edicao, "2")) == ("masculino", "feminino")
    with pytest.raises(ValueError):
        cw.sexo_padrao(edicao, "3")


@pytest.mark.parametrize("edicao", ["2008-2009", "2017-2018"])
def test_escolaridade_sete_niveis_identicos(edicao):
    assert [cw.escolaridade_nivel(edicao, str(i)) for i in range(1, 8)] == [1, 2, 3, 4, 5, 6, 7]
    for ruim in ("0", "8", "88"):
        with pytest.raises(ValueError):
            cw.escolaridade_nivel(edicao, ruim)


@pytest.mark.parametrize("edicao", ["2008-2009", "2017-2018"])
def test_uf_codigo_ibge_igual_nas_duas_edicoes(edicao):
    assert cw.uf_padrao(edicao, "35") == "35" and cw.uf_padrao(edicao, 11) == "11"
    with pytest.raises(ValueError):
        cw.uf_padrao(edicao, "99")


@pytest.mark.parametrize("edicao", ["2008-2009", "2017-2018"])
def test_idade_em_anos_inteira_e_dentro_do_dominio(edicao):
    assert cw.idade_anos(edicao, "34") == 34 and cw.idade_anos(edicao, "10") == 10
    for ruim in ("-1", "112", "abc", ""):
        with pytest.raises(ValueError):
            cw.idade_anos(edicao, ruim)


def test_grande_regiao_pela_uf():
    assert [cw.grande_regiao(u) for u in ("11", "29", "35", "43", "53")] == [
        "Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"]
    with pytest.raises(ValueError):
        cw.grande_regiao("99")


# --- estratos geograficos (planilhas oficiais) ----------------------------------------------------
NOMES_UF = {"rondonia": "11", "acre": "12", "amazonas": "13", "sao paulo": "35", "rio grande do sul": "43",
            "distrito federal": "53"}
CABECALHO = [["", "ESTRATOS  ", "", "", ""], ["ÁREAS ", "URBANO", "", "", "RURAL"], ["DA PESQUISA", "Município", "Resto ", "Resto ", ""],
             ["", "da Capital", "da RM", "da UF", ""], ["", "", "", "", ""]]
PLANILHA_2008 = CABECALHO + [
    ["Região Norte", "", "", "", ""], ["Rondônia", "1 e 2", "-", "3 a 6", "7 a 11"], ["Acre", 1.0, "-", 2.0, "3 e 4"],
    ["Amazonas", "1 a 4", "-", "5 a 8", "9 a 13"], ["", "", "", "", ""], ["Região Sudeste", "", "", "", ""],
    ["São Paulo", "1 a 9", "10 a 15", "16 a 30", "31 a 51"], ["Região Sul", "", "", "", ""],
    ["Rio Grande Do Sul", "1 a 6", "7 a 9", "10 a 18", "19 a 30"], ["Distrito Federal", "1       a       7", "", "", "8 a 9"]]
PLANILHA_2017 = CABECALHO + [
    ["Região Norte", "", "", "", ""], ["Rondônia", "1101 e 1102", "-", "1103 a 1107 ", "1108 a 1111"],
    ["Acre", 1201.0, "-", 1202.0, "1203 e 1204"], ["Amazonas", "1301 a 1306", 1307.0, "1308 a 1310", "1311 a 1316"],
    ["São Paulo", "3501 a 3509 ", "3510 a 3515", "3516 a 3536", "3537 a 3553"],
    ["Rio Grande Do Sul", "4301 a 4306", "4307 a 4309", "4310 a 4324", "4325 a 4335"],
    ["Distrito Federal", "5301 a 5306", "", "", "5307 e 5308"]]


def test_expande_codigos_de_estrato():
    assert cw.expandir_codigos("1 e 2") == [1, 2] and cw.expandir_codigos("3 a 6") == [3, 4, 5, 6]
    assert cw.expandir_codigos(1.0) == [1] and cw.expandir_codigos("1103 a 1107 ") == [1103, 1104, 1105, 1106, 1107]
    assert cw.expandir_codigos("1       a       7") == list(range(1, 8))
    assert cw.expandir_codigos("-") == [] and cw.expandir_codigos("") == []


def test_estratos_2008_sao_por_uf_e_se_repetem_entre_ufs():
    est = cw.decodificar_estratos(PLANILHA_2008, NOMES_UF, "2008-2009")
    assert est[("11", 1)] == "capital" and est[("12", 1)] == "capital"  # mesmo codigo, UFs diferentes
    assert est[("11", 3)] == "resto_uf" and est[("11", 7)] == "rural" and est[("35", 10)] == "resto_rm"
    assert est[("53", 7)] == "capital" and est[("53", 8)] == "rural" and ("11", 12) not in est


def test_estratos_2017_codigo_traz_a_uf_e_confere_com_o_nome_da_planilha():
    est = cw.decodificar_estratos(PLANILHA_2017, NOMES_UF, "2017-2018")
    assert est[("13", 1307)] == "resto_rm" and est[("35", 3501)] == "capital" and est[("43", 4325)] == "rural"
    assert est[("12", 1202)] == "resto_uf"


def test_estrato_2017_com_prefixo_de_outra_uf_e_recusado():
    ruim = CABECALHO + [["Rondônia", "1201 e 1202", "-", "1103 a 1107", "1108 a 1111"]]
    with pytest.raises(ValueError, match="prefixo"):
        cw.decodificar_estratos(ruim, NOMES_UF, "2017-2018")


def test_codigo_de_estrato_em_dois_tipos_e_recusado():
    ruim = CABECALHO + [["Rondônia", "1 a 3", "-", "3 a 6", "7 a 11"]]
    with pytest.raises(ValueError, match="mais de um tipo"):
        cw.decodificar_estratos(ruim, NOMES_UF, "2008-2009")


def test_nome_de_uf_desconhecido_e_recusado():
    with pytest.raises(ValueError, match="UF"):
        cw.decodificar_estratos(CABECALHO + [["Atlantida", "1", "-", "2", "3"]], NOMES_UF, "2008-2009")


def test_area_comparavel_une_resto_da_rm_e_resto_da_uf():
    assert [cw.area_comparavel(t) for t in ("capital", "resto_rm", "resto_uf", "rural")] == [
        "capital", "urbano_nao_capital", "urbano_nao_capital", "rural"]




# --- tabela de correspondencia (configs/crosswalk.json) -------------------------------------------


# --- validacao contra os dicionarios oficiais -----------------------------------------------------
def _var(cod, pos, tam, cats=(), desc=""):
    return Variavel(pos, tam, cod, desc, tuple(cats))


DICIONARIOS_OK = {
    "2008-2009": {"Consumo Alimentar": [_var("DIA_SEMANA", 152, 1)], "Morador": [_var("V0405", 76, 2, ["1 – Masculino", "2 – Feminino"])]},
    "2017-2018": {"Consumo Alimentar": [_var("DIA_SEMANA", 529, 15)], "Morador": [_var("V0404", 36, 1, ["1 – Homem", "2 – Mulher"])]},
}
MINI = {"conceitos": [
    {"conceito": "dia_semana", "comum": "sim", "edicoes": {
        "2008-2009": {"variaveis": [{"registro": "Consumo Alimentar", "variavel": "DIA_SEMANA", "posicao": 152, "tamanho": 1}]},
        "2017-2018": {"variaveis": [{"registro": "Consumo Alimentar", "variavel": "DIA_SEMANA", "posicao": 529, "tamanho": 15}]}}},
    {"conceito": "sexo", "comum": "sim", "edicoes": {
        "2008-2009": {"variaveis": [{"registro": "Morador", "variavel": "V0405", "posicao": 76, "tamanho": 2,
                                     "categorias": ["1 – Masculino", "2 – Feminino"]}]},
        "2017-2018": {"variaveis": [{"registro": "Morador", "variavel": "V0404", "posicao": 36, "tamanho": 1,
                                     "categorias": ["1 – Homem", "2 – Mulher"]}]}}}]}
