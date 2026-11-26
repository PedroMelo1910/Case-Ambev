"""Auditoria do denominador da POF (rodada de 24/11/2026): regra de nao-comida v2, sabor desconhecido e particao
reservada por domicilio. Tudo com dados sinteticos; nenhum teste abre microdados."""

import hashlib
import io
import zipfile

import duckdb

import coleta
import preparo


def _xlsx(linhas: list[list[str]]) -> bytes:
    """Planilha xlsx minima (strings inline) para o leitor sem dependencias."""
    def celula(col: int, r: int, valor: str) -> str:
        return f'<c r="{chr(65 + col)}{r}" t="inlineStr"><is><t>{valor}</t></is></c>' if valor else ""
    corpo = "".join(f'<row r="{r}">' + "".join(celula(c, r, v) for c, v in enumerate(linha)) + "</row>"
                    for r, linha in enumerate(linhas, 1))
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    rel = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
    saida = io.BytesIO()
    with zipfile.ZipFile(saida, "w") as z:
        z.writestr("xl/workbook.xml", f'<workbook {ns} {rel}><sheets><sheet name="Gruali" sheetId="1" r:id="rId1"/>'
                                      "</sheets></workbook>")
        z.writestr("xl/worksheets/sheet1.xml", f"<worksheet {ns}><sheetData>{corpo}</sheetData></worksheet>")
    return saida.getvalue()


CLASSIFICACAO = [
    ["Apêndice 1 - Classificação dos grupos de alimentos e preparações"],
    ["Grupo de alimentos", "Código do item", "Descrição do item"],
    ["Bebidas"], ["", "", "Água"], ["", "8216301", "Agua"],
    ["", "", "Bebidas lácteas"], ["", "6900822", "Toddynho"],
    ["Preparações mistas"], ["", "", "Preparações mistas"], ["", "8581001", "Cafe com Pão"],
    ["Pizzas, salgados e sanduíches"], ["", "", "Pizzas"], ["", "8500903", "Pizza"],
]


def test_leitor_xlsx_devolve_as_linhas_por_coluna():
    folhas = coleta.ler_xlsx(_xlsx(CLASSIFICACAO))
    assert list(folhas) == ["Gruali"]
    assert folhas["Gruali"][4] == {"B": "8216301", "C": "Agua"}


def test_classificacao_oficial_achata_grupo_e_subgrupo():
    mapa = preparo.classificacao_dos_grupos(coleta.ler_xlsx(_xlsx(CLASSIFICACAO))["Gruali"])
    assert mapa == {"8216301": ("Bebidas", "Água"), "6900822": ("Bebidas", "Bebidas lácteas"),
                    "8581001": ("Preparações mistas", "Preparações mistas"),
                    "8500903": ("Pizzas, salgados e sanduíches", "Pizzas")}


def test_regra_v2_soma_bebidas_oficiais_e_devolve_preparacoes_mistas_a_comida():
    classificacao = {"8216301": ("Bebidas", "Água"), "6900822": ("Bebidas", "Bebidas lácteas"),
                     "8581001": ("Preparações mistas", "Preparações mistas"), "7900101": ("Laticínios", "Leite integral"),
                     "8500903": ("Pizzas, salgados e sanduíches", "Pizzas")}
    # regra v1 (cadastro): leite fluido e cafe com pao fora; Toddynho dentro; um codigo de bebida sem classificacao fora
    v1 = ["7900101", "8581001", "8216301", "8209102"]
    assert preparo.regra_nao_comida_v2(v1, classificacao) == ["6900822", "7900101", "8209102", "8216301"]


def test_sabor_separa_refrigerante_de_sabor_desconhecido():
    con = duckdb.connect()
    con.execute("""CREATE TABLE ep AS SELECT * FROM (VALUES (1, 1, 1, 0, 1), (1, 0, 1, 0, 1), (1, 0, 0, 1, 1),
        (1, 0, 0, 0, 1), (0, 0, 0, 0, 0)) v(refrigerante, guarana, cola, outro_conhecido, desconhecido)""")
    assert [r[0] for r in con.execute(f"SELECT {preparo.SQL_SABOR} FROM ep").fetchall()] == [
        "guarana", "cola", "outro", "desconhecido", "sem_refrigerante"]
    assert preparo.SABORES == ("guarana", "cola", "outro", "desconhecido", "sem_refrigerante")
    assert preparo.SABOR_DESCONHECIDO_POF == ("8203501",)


def test_particao_segue_o_protocolo_e_reserva_cerca_de_um_quinto():
    ids = [f"{11000000 + i:09d}-{i % 7 + 1}" for i in range(5000)]
    for d in ids[:50]:
        esperado = "teste" if int(hashlib.sha256(d.encode()).hexdigest()[:8], 16) % 5 == 0 else "desenvolvimento"
        assert preparo.particao(d) == esperado
    fracao = sum(preparo.particao(d) == "teste" for d in ids) / len(ids)
    assert 0.18 < fracao < 0.22


def test_regras_por_edicao_ficam_registradas():
    assert preparo.REGRA_NAO_COMIDA == {"2008-2009": "v1_cadastro", "2017-2018": "v2_classificacao_oficial_2017"}
