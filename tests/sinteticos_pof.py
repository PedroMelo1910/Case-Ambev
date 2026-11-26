"""Fabrica de dados sinteticos: arquivos de largura fixa e tabelas minimas."""

from pathlib import Path

COLUNAS = {
    "consumo_2008": [("COD_UF", 2), ("NUM_SEQ", 3), ("NUM_DV", 1), ("NUM_DOM", 2), ("NUM_UC", 1), ("COD_INFORMANTE", 2),
                     ("ESTRATO_POF", 2), ("PESO_FINAL", 15), ("NUM_QUADRO", 2), ("V9018", 1), ("V9015", 2),
                     ("V9001", 7), ("RENDA_TOTAL", 16), ("DIA_SEMANA", 1)],
    "morador_2008": [("COD_UF", 2), ("NUM_SEQ", 3), ("NUM_DV", 1), ("NUM_DOM", 2), ("NUM_UC", 1), ("COD_INFORMANTE", 2),
                     ("PESO_FINAL", 14), ("IDADE_ANOS", 3), ("V0405", 2), ("TIPO_SITUACAO_REG", 1),
                     ("NIVEL_INSTRUCAO_MORADOR", 1)],
    "domicilio_2008": [("COD_UF", 2), ("NUM_SEQ", 3), ("NUM_DV", 1), ("NUM_DOM", 2), ("PERIODO_REAL", 4)],
    "consumo_2017": [("UF", 2), ("ESTRATO_POF", 4), ("TIPO_SITUACAO_REG", 1), ("COD_UPA", 9), ("NUM_DOM", 2), ("NUM_UC", 1),
                     ("COD_INFORMANTE", 2), ("QUADRO", 2), ("SEQ", 2), ("V9005", 4), ("V9007", 2), ("V9001", 7),
                     ("V9015", 3), ("V9017", 1), ("V9018", 1), ("PESO_FINAL", 15), ("RENDA_TOTAL", 10),
                     ("DIA_SEMANA", 15), ("DIA_ATIPICO", 1)],
    "morador_2017": [("UF", 2), ("COD_UPA", 9), ("NUM_DOM", 2), ("NUM_UC", 1), ("COD_INFORMANTE", 2), ("V0306", 2),
                     ("V0403", 3), ("V0404", 1), ("PESO_FINAL", 14), ("NIVEL_INSTRUCAO", 1)],
}


def layout_sintetico(tabela):
    """Retorna {codigo: (posicao_1_base, tamanho)} e o comprimento da linha."""
    campos, pos = {}, 1
    for nome, tam in COLUNAS[tabela]:
        campos[nome] = (pos, tam)
        pos += tam
    return campos, pos - 1


def escrever_largura_fixa(caminho: Path, tabela: str, linhas: list[dict], fim_de_linha="\r\n"):
    """Grava linhas (dict coluna->valor texto) alinhadas a esquerda, preenchidas com espacos."""
    larguras = COLUNAS[tabela]
    with open(caminho, "w", encoding="latin-1", newline="") as f:
        for linha in linhas:
            f.write("".join(str(linha.get(n, "")).ljust(t)[:t] for n, t in larguras) + fim_de_linha)
    return caminho


def c08(uf, seq, dv, dom, uc, inf, quadro, hora, cod, **extra):
    base = {"COD_UF": f"{uf:02d}", "NUM_SEQ": f"{seq:03d}", "NUM_DV": str(dv), "NUM_DOM": f"{dom:02d}", "NUM_UC": str(uc),
            "COD_INFORMANTE": f"{inf:02d}", "ESTRATO_POF": "01", "PESO_FINAL": "100.5", "NUM_QUADRO": str(quadro),
            "V9018": "1", "V9015": f"{hora:02d}", "V9001": cod, "RENDA_TOTAL": "2000.00", "DIA_SEMANA": "6"}
    base.update(extra)
    return base


def m08(uf, seq, dv, dom, uc, inf, idade=30, sexo="1", **extra):
    base = {"COD_UF": f"{uf:02d}", "NUM_SEQ": f"{seq:3d}", "NUM_DV": str(dv), "NUM_DOM": f"{dom:2d}", "NUM_UC": str(uc),
            "COD_INFORMANTE": f"{inf:2d}", "PESO_FINAL": "100.5", "IDADE_ANOS": str(idade), "V0405": sexo, "TIPO_SITUACAO_REG": "1",
            "NIVEL_INSTRUCAO_MORADOR": "5"}
    base.update(extra)
    return base


def d08(uf, seq, dv, dom, periodo=12):
    return {"COD_UF": f"{uf:02d}", "NUM_SEQ": f"{seq:3d}", "NUM_DV": str(dv), "NUM_DOM": f"{dom:2d}", "PERIODO_REAL": str(periodo)}


def c17(uf, upa, dom, uc, inf, quadro, seq, hora, cod, adicao=False, **extra):
    base = {"UF": str(uf), "ESTRATO_POF": f"{uf}01", "TIPO_SITUACAO_REG": "1", "COD_UPA": str(upa), "NUM_DOM": f"{dom:02d}",
            "NUM_UC": str(uc), "COD_INFORMANTE": f"{inf:02d}", "QUADRO": str(quadro), "SEQ": f"{seq:02d}",
            "V9005": "" if adicao else "100", "V9007": "" if adicao else "1", "V9001": cod, "V9015": f"{hora:02d}H",
            "V9017": "3", "V9018": "1", "PESO_FINAL": "200.25", "RENDA_TOTAL": "3000.00", "DIA_SEMANA": "Sabado",
            "DIA_ATIPICO": "2"}
    base.update(extra)
    return base


def m17(uf, upa, dom, uc, inf, idade=25, sexo="2", v0306="1", **extra):
    base = {"UF": str(uf), "COD_UPA": str(upa), "NUM_DOM": f"{dom:02d}", "NUM_UC": str(uc), "COD_INFORMANTE": f"{inf:02d}",
            "V0306": v0306, "V0403": str(idade), "V0404": sexo, "PESO_FINAL": "200.25", "NIVEL_INSTRUCAO": "5"}
    base.update(extra)
    return base


def cenario_2008():
    """6 episodios E2 (3 pessoas); 4 com pizza, 1 com guarana, 2 com algum refrigerante."""
    pizza, pizza_light, guarana, cola, agua, arroz = "8500903", "8508003", "8200301", "8200101", "6001001", "1234567"
    consumo = [
        c08(35, 1, 0, 1, 1, 1, 71, 12, arroz),
        c08(35, 1, 0, 1, 1, 1, 71, 20, pizza), c08(35, 1, 0, 1, 1, 1, 71, 20, guarana),
        c08(35, 1, 0, 1, 1, 1, 71, 21, pizza_light), c08(35, 1, 0, 1, 1, 1, 71, 21, cola),
        c08(35, 1, 0, 1, 1, 1, 72, 20, pizza),
        c08(35, 1, 0, 1, 1, 2, 71, 20, guarana),
        c08(11, 2, 5, 3, 1, 1, 71, 19, pizza), c08(11, 2, 5, 3, 1, 1, 71, 19, agua),
    ]
    morador = [m08(35, 1, 0, 1, 1, 1, idade=34, sexo="1"), m08(35, 1, 0, 1, 1, 2, idade=12, sexo="2"),
               m08(11, 2, 5, 3, 1, 1, idade=55, sexo="2")]
    domicilio = [d08(35, 1, 0, 1, 12), d08(11, 2, 5, 3, 40)]
    return consumo, morador, domicilio


def cenario_2017():
    """Pizza em 3 episodios E2 de 2 pessoas; guarana e refrigerante presentes nos brutos (nao devem vazar)."""
    pizza, guarana, cola, arroz = "8500903", "8200301", "8200101", "1234567"
    consumo = [
        c17(35, 350000001, 1, 1, 1, 72, 1, 20, pizza), c17(35, 350000001, 1, 1, 1, 72, 2, 20, "8500914"),
        c17(35, 350000001, 1, 1, 1, 72, 3, 20, guarana), c17(35, 350000001, 1, 1, 1, 72, 4, 20, cola),
        c17(35, 350000001, 1, 1, 1, 73, 1, 21, pizza),
        c17(35, 350000001, 1, 1, 1, 73, 2, 12, arroz),
        c17(35, 350000001, 1, 1, 1, 73, 3, 12, "8500903", adicao=True),
        c17(11, 110000009, 2, 1, 2, 72, 1, 19, pizza), c17(11, 110000009, 2, 1, 2, 72, 2, 19, "8500908"),
        c17(11, 110000009, 2, 1, 3, 72, 1, 13, guarana),
    ]
    morador = [m17(35, 350000001, 1, 1, 1, idade=40), m17(11, 110000009, 2, 1, 2, idade=17, sexo="1"),
               m17(11, 110000009, 2, 1, 3, idade=9)]
    return consumo, morador
