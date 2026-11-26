import duckdb
import pytest

from sinteticos_pof import COLUNAS, escrever_largura_fixa, layout_sintetico
from coleta import extrair_variaveis
from coleta import Layout, LayoutInvalido, carregar_registro, layout_de

CABECALHO = ["Posição Inicial", "Tamanho", "Decimais", "Código da variável", "Descrição", "Categorias"]


def test_extrai_variaveis_e_categorias_do_dicionario_oficial():
    linhas = [
        ["REGISTRO – CONSUMO ALIMENTAR", "", "", "", "", ""],
        CABECALHO,
        [1.0, 2.0, "", "UF", "Unidade da federação", "11 – Rondônia"],
        ["", "", "", "", "", "12 – Acre"],
        [3.0, 4.0, "", "ESTRATO_POF", "Estrato", ""],
        ["", "", "", "", "continuação da descrição", ""],
    ]
    registro, vars_ = extrair_variaveis(linhas)
    assert registro == "REGISTRO – CONSUMO ALIMENTAR"
    assert [(v.codigo, v.posicao, v.tamanho) for v in vars_] == [("UF", 1, 2), ("ESTRATO_POF", 3, 4)]
    assert vars_[0].categorias == ("11 – Rondônia", "12 – Acre")
    assert "continuação" in vars_[1].descricao


def test_aba_sem_cabecalho_retorna_vazio():
    assert extrair_variaveis([["nada"]]) == ("", [])


def test_layout_de_variaveis():
    _, vars_ = extrair_variaveis([CABECALHO, [1.0, 2.0, "", "A", "d", ""], [3.0, 5.0, "", "B", "d", ""]])
    lay = layout_de(vars_)
    assert lay.campos == {"A": (1, 2), "B": (3, 5)} and lay.comprimento == 7


def _lay(tabela):
    campos, comp = layout_sintetico(tabela)
    return Layout(campos=campos, comprimento=comp)


@pytest.mark.parametrize("fim", ["\r\n", "\n"])
def test_le_largura_fixa_com_trim_e_nulos(tmp_path, fim):
    arq = escrever_largura_fixa(tmp_path / "m.txt", "morador_2017", [
        {"UF": "35", "COD_UPA": "350000001", "NUM_DOM": "01", "NUM_UC": "1", "COD_INFORMANTE": "01", "V0306": "", "V0403": "34",
         "V0404": "1", "PESO_FINAL": "10.5"},
        {"UF": "11", "COD_UPA": "110000009", "NUM_DOM": "02", "NUM_UC": "1", "COD_INFORMANTE": "02", "V0306": "05", "V0403": "9",
         "V0404": "2", "PESO_FINAL": "7"}], fim_de_linha=fim)
    con = duckdb.connect()
    n = carregar_registro(con, "mor", arq, _lay("morador_2017"), ["UF", "COD_UPA", "V0306", "V0403"])
    assert n == 2
    assert con.execute("SELECT UF, COD_UPA, V0306, V0403 FROM mor ORDER BY UF").fetchall() == [
        ("11", "110000009", "05", "9"), ("35", "350000001", None, "34")]


def test_linha_com_comprimento_errado_interrompe(tmp_path):
    arq = tmp_path / "x.txt"
    arq.write_text("curta\r\n", encoding="latin-1", newline="")
    with pytest.raises(LayoutInvalido):
        carregar_registro(duckdb.connect(), "t", arq, _lay("morador_2017"), ["UF"])


def test_coluna_fora_do_dicionario_interrompe(tmp_path):
    arq = escrever_largura_fixa(tmp_path / "m.txt", "morador_2017", [{"UF": "35"}])
    with pytest.raises(LayoutInvalido):
        carregar_registro(duckdb.connect(), "t", arq, _lay("morador_2017"), ["NAO_EXISTE"])


def test_filtro_na_leitura_so_materializa_linhas_pedidas(tmp_path):
    arq = escrever_largura_fixa(tmp_path / "c.txt", "consumo_2017", [
        {"UF": "35", "V9001": "8500903"}, {"UF": "35", "V9001": "8200301"}, {"UF": "11", "V9001": "8500914"}])
    con = duckdb.connect()
    n = carregar_registro(con, "c", arq, _lay("consumo_2017"), ["UF", "V9001"], where="V9001 IN ('8500903','8500914')")
    assert n == 2
    assert {r[0] for r in con.execute("SELECT V9001 FROM c").fetchall()} == {"8500903", "8500914"}


def test_conferencia_de_comprimento_ocorre_antes_do_filtro(tmp_path):
    arq = escrever_largura_fixa(tmp_path / "c.txt", "consumo_2017", [{"UF": "35", "V9001": "8200301"}])
    with open(arq, "a", encoding="latin-1", newline="") as f:
        f.write("linha invalida\r\n")
    with pytest.raises(LayoutInvalido):
        carregar_registro(duckdb.connect(), "c", arq, _lay("consumo_2017"), ["UF", "V9001"], where="V9001 = '8500903'")


def test_fabrica_cobre_todas_as_tabelas():
    assert set(COLUNAS) == {"consumo_2008", "morador_2008", "domicilio_2008", "consumo_2017", "morador_2017"}
