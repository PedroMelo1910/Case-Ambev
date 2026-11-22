"""Regenera a tabela de pizza por cenario (`t1` de `src/personas/dados/estimativas.json`) a partir da POF 2008-2009.

Uso (na raiz Case-Ambev, com os microdados de 2008 baixados pela coleta em data/raw/pof_2008_2009):
  PYTHONPATH=src python -m treinar_pizza --saida <pasta fora do repositorio> [--atualizar]

O que faz, com o codigo congelado de `src/occasion` (leitura, episodios E2 e crosswalk) e o de `aplicacao.aprendizado`:
1. monta as refeicoes com comida (episodio pessoa x dia x hora com ao menos um item que nao e bebida nem adicao, pelo
   cadastro de produtos de 2008);
2. reajusta o modelo publicado (logistica ridge, com o hiperparametro registrado na tabela atual) em todas elas;
3. calcula os 420 cenarios (dia x periodo x regiao x tipo de area), com publico "todos" (media sobre a composicao
   observada), e a faixa de 95% em 200 reajustes sobre reamostras de UPAs;
4. grava `t1_pizza.json` em --saida e compara com a tabela publicada. Com --atualizar, troca so a parte `t1` do
   arquivo da aplicacao; a checagem da estimativa composta e a analise de renda e sabor ficam como registradas.
A selecao do modelo (validacao cruzada agrupada por UPA) esta documentada em `estimativas.json["t1"]["validacao"]`;
esta entrada refaz o ajuste e a tabela, nao a selecao. Nada de 2017 e lido. Linha a linha nao e gravado.
"""

import argparse
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter
from itertools import product
from pathlib import Path

import duckdb
import numpy as np
import xlrd

from aplicacao import aprendizado as ex, previsao_consumo as pc
from occasion.crosswalk import (area_comparavel, decodificar_estratos, dia_semana_numero, escolaridade_nivel,
                                grande_regiao, ler_planilha_estratos, nomes_uf_do_dicionario, sexo_padrao, sql_hora)
from occasion.episodios_2008 import construir_e2_2008
from occasion.microdados import carregar_registro, layout_de, ler_dicionario_xls
from occasion.pipeline_pof import COLUNAS
from occasion.variaveis import escolaridade_3, faixa_etaria, faixa_hora

RAIZ = Path(__file__).resolve().parents[1]
BASE = RAIZ / "data" / "raw" / "pof_2008_2009" / "extraido"
DOC_ZIP = RAIZ / "data" / "raw" / "pof_2008_2009" / "downloads" / "Documentacao_20231009.zip"
PUBLICADO = RAIZ / "src" / "personas" / "dados" / "estimativas.json"
SEMENTE = 20261116
REPLICAS = 200
CAMPOS = ("dia_semana", "periodo", "grande_regiao", "area", "faixa_etaria", "sexo", "escolaridade")
PUBLICO = ("faixa_etaria", "sexo", "escolaridade")
CATEGORIAS = {"dia_semana": tuple(range(1, 8)), "periodo": pc.PERIODOS, "grande_regiao": pc.REGIOES, "area": pc.AREAS,
              "faixa_etaria": pc.FAIXAS_ETARIAS, "sexo": pc.SEXOS, "escolaridade": pc.ESCOLARIDADES}
CELULAS = tuple(f"{d}|{p}" for d in range(1, 8) for p in pc.PERIODOS)
GRUPOS_BEBIDA = ("82", "83")  # bebidas nao alcoolicas e alcoolicas no cadastro de 2008
DESCRICOES_BEBIDA_ADICAO = (  # fora dos grupos 82/83: bebidas preparadas e adicoes, pela descricao do cadastro
    r"^CAFE( |$)", r"^CAFE COM LEITE", r"^SUCO( |$)", r"^REFRESCO", r"^VITAMINA( |$)", r"^CHA( |$)",
    r"^LEITE (DE VACA|DE CABRA|INTEGRAL|DESNATADO|SEMIDESNATADO|EM PO|FERMENTADO|AROMATIZADO|DE SOJA)",
    r"^ACUCAR", r"^ADOCANTE", r"^ACHOCOLATADO", r"^XAROPE")
_PESSOA = "cod_uf, seq, dv, num_dom, num_uc, cod_inf"


def bebida_ou_adicao() -> list[str]:
    """Codigos de 2008 que nao contam como comida: grupos de bebida e descricoes de bebida preparada ou adicao."""
    with zipfile.ZipFile(DOC_ZIP) as z:
        nome = next(n for n in z.namelist() if "Produtos do Consumo Alimentar" in n)
        folha = xlrd.open_workbook(file_contents=z.read(nome)).sheet_by_index(0)
    padroes = [re.compile(p) for p in DESCRICOES_BEBIDA_ADICAO]
    codigos = []
    for r in range(1, folha.nrows):
        codigo = str(folha.cell_value(r, 0)).split(".")[0].strip()
        descricao = str(folha.cell_value(r, 1)).strip()
        if len(codigo) == 7 and codigo.isdigit() and (codigo[:2] in GRUPOS_BEBIDA or any(p.search(descricao) for p in padroes)):
            codigos.append(codigo)
    return sorted(codigos)


def carregar(con) -> dict:
    dic = ler_dicionario_xls(BASE / "documentacao" / "Dicionarios_de_variaveis.xls")
    for tabela, (arquivo, aba) in {"consumo_2008": ("T_CONSUMO_S.txt", "Consumo Alimentar"),
                                   "morador_2008": ("T_MORADOR_S.txt", "Morador"),
                                   "domicilio_2008": ("T_DOMICILIO_S.txt", "Domicílio")}.items():
        carregar_registro(con, tabela, BASE / "dados" / arquivo, layout_de(dic[aba]), COLUNAS[tabela])
    uf = next(v for aba in dic.values() for v in aba if v.codigo in ("UF", "COD_UF") and v.categorias)
    linhas = ler_planilha_estratos(BASE / "documentacao" / "Estratos_Geograficos_POF_2008-2009.xls")
    return decodificar_estratos(linhas, nomes_uf_do_dicionario(uf), "2008-2009")


def refeicoes_com_comida(con, estratos: dict, nao_comida: list[str]) -> dict:
    """Uma linha por episodio E2 com comida: os sete campos na forma canonica, pizza e UPA."""
    construir_e2_2008(con)
    lista = "(" + ",".join(f"'{c}'" for c in nao_comida) + ")"
    con.execute(f"""CREATE OR REPLACE TABLE comida_2008 AS
        SELECT CAST(COD_UF AS INT) cod_uf, CAST(NUM_SEQ AS INT) seq, CAST(NUM_DV AS INT) dv, CAST(NUM_DOM AS INT) num_dom,
               CAST(NUM_UC AS INT) num_uc, CAST(COD_INFORMANTE AS INT) cod_inf, CAST(NUM_QUADRO AS INT) quadro,
               {sql_hora('2008-2009', 'V9015')} hora, max((V9001 NOT IN {lista})::INT) comida
        FROM consumo_2008 GROUP BY ALL""")
    linhas = con.execute(f"""
        SELECT e.cod_uf, e.seq, e.dv, e.estrato_pof, e.dia_semana, e.hora, e.p, n.idade_anos, n.sexo, n.nivel_instrucao
        FROM ep_2008 e JOIN morador_2008_norm n USING ({_PESSOA}) JOIN comida_2008 c USING ({_PESSOA}, quadro, hora)
        WHERE c.comida = 1
        ORDER BY e.cod_uf, e.seq, e.dv, e.num_dom, e.num_uc, e.cod_inf, e.quadro, e.hora""").fetchall()
    colunas = {c: [] for c in (*CAMPOS, "p", "upa")}
    for uf, seq, dv, estrato, dia, hora, p, idade, sexo, nivel in linhas:
        uf2 = f"{uf:02d}"
        canonico = pc.canonico_de_nativas({
            "dia_semana": dia_semana_numero("2008-2009", dia), "faixa_hora": faixa_hora(hora),
            "grande_regiao": grande_regiao(uf2), "area": area_comparavel(estratos[(uf2, int(estrato))]),
            "faixa_etaria": faixa_etaria(idade), "sexo": sexo_padrao("2008-2009", sexo),
            "escolaridade": escolaridade_3(escolaridade_nivel("2008-2009", nivel))})
        for campo in CAMPOS:
            colunas[campo].append(canonico[campo])
        colunas["p"].append(p)
        colunas["upa"].append(f"{uf2}{seq:03d}{dv}")
    return {c: np.asarray(v) for c, v in colunas.items()}


def matriz(campos: dict) -> np.ndarray:
    """Indicadores dos sete campos e das celulas dia x periodo (a mesma entrada do modelo validado)."""
    blocos = [np.column_stack([(np.asarray(campos[c]) == v) for v in CATEGORIAS[c]]) for c in CATEGORIAS]
    cel = np.char.add(np.char.add(np.asarray(campos["dia_semana"]).astype(str), "|"), np.asarray(campos["periodo"]).astype(str))
    blocos.append(np.column_stack([cel == c for c in CELULAS]))
    return np.hstack(blocos).astype(np.float32)


def _grade(base: dict) -> tuple[list[tuple], np.ndarray, np.ndarray]:
    composicao = Counter(zip(*(base[c].tolist() for c in PUBLICO), strict=True))
    combinacoes = sorted(composicao)
    pesos = np.array([composicao[c] for c in combinacoes], dtype=float) / sum(composicao.values())
    contextos = list(product(range(1, 8), pc.PERIODOS, pc.REGIOES, pc.AREAS))
    linhas = {c: [] for c in CAMPOS}
    for ctx in contextos:
        for publico in combinacoes:
            for campo, valor in zip(CAMPOS, (*ctx, *publico), strict=True):
                linhas[campo].append(valor)
    return contextos, matriz({c: np.asarray(v) for c, v in linhas.items()}), pesos


def _marginal(modelo, x: np.ndarray, pesos: np.ndarray, n: int) -> np.ndarray:
    return (modelo.predict_proba(x)[:, 1].reshape(n, len(pesos)) * pesos).sum(axis=1)


def tabela(base: dict, hiperparametros: dict) -> dict:
    candidato = next(c for c in ex.candidatos_padrao(False) if c.nome == "logistica")
    x, y = matriz(base), base["p"].astype(int)
    estimador = candidato.construir(SEMENTE)
    estimador.set_params(**ex._parametros_internos(hiperparametros, estimador.get_params()))
    final = estimador.fit(x, y)
    contextos, xg, pesos = _grade(base)
    pontual = _marginal(final, xg, pesos, len(contextos))
    reamostras = ex.reamostrar(candidato, hiperparametros, x, y, base["upa"], REPLICAS, SEMENTE)
    amostras = np.vstack([_marginal(m, xg, pesos, len(contextos)) for m, _ in reamostras])
    chave = ["|".join(map(str, linha)) for linha in zip(*(base[c] for c in CAMPOS[:4]), strict=True)]
    n, k = Counter(chave), Counter(c for c, v in zip(chave, y, strict=True) if v)
    saida = {}
    for i, ctx in enumerate(contextos):
        nome = "|".join(map(str, ctx))
        saida[nome] = {"estimativa": float(pontual[i]),
                       "intervalo": [float(v) for v in np.quantile(amostras[:, i], [0.025, 0.975])],
                       "dp_log": float(np.std(np.log(np.clip(amostras[:, i], 1e-12, 1)))),
                       "refeicoes": n[nome], "com_pizza": k[nome]}
    return {"refeicoes": int(len(y)), "com_pizza": int(y.sum()), "taxa_geral": float(y.mean()), "contextos": saida}


def comparar(nova: dict, publicada: dict) -> dict:
    a, b = nova["contextos"], publicada["contextos"]
    dif = [abs(a[c]["estimativa"] - b[c]["estimativa"]) for c in b]
    faixa = [max(abs(x - y) for x, y in zip(a[c]["intervalo"], b[c]["intervalo"], strict=True)) for c in b]
    contagens = all(a[c]["refeicoes"] == b[c]["refeicoes"] and a[c]["com_pizza"] == b[c]["com_pizza"] for c in b)
    return {"mesmos_cenarios": set(a) == set(b), "mesmas_contagens": contagens,
            "refeicoes": [nova["refeicoes"], publicada["refeicoes"]], "com_pizza": [nova["com_pizza"], publicada["com_pizza"]],
            "maior_diferenca_estimativa": max(dif), "maior_diferenca_faixa": max(faixa)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--saida", type=Path, required=True, help="pasta fora do repositório para a tabela regenerada")
    parser.add_argument("--atualizar", action="store_true", help="troca a parte t1 do arquivo da aplicação")
    args = parser.parse_args()
    publicado = json.loads(PUBLICADO.read_text(encoding="utf-8"))
    con = duckdb.connect()
    base = refeicoes_com_comida(con, carregar(con), bebida_ou_adicao())
    nova = tabela(base, publicado["t1"]["hiperparametros"])
    t1 = {**publicado["t1"], **nova}
    texto = json.dumps(t1, ensure_ascii=False, separators=(",", ":"))
    args.saida.mkdir(parents=True, exist_ok=True)
    (args.saida / "t1_pizza.json").write_text(texto, encoding="utf-8")
    relatorio = {"sha256": hashlib.sha256(texto.encode()).hexdigest(), **comparar(nova, publicado["t1"])}
    if args.atualizar:
        PUBLICADO.write_text(json.dumps({**publicado, "t1": t1}, ensure_ascii=False, separators=(",", ":")),
                             encoding="utf-8")
        relatorio["atualizado"] = str(PUBLICADO.relative_to(RAIZ))
    sys.stdout.write(json.dumps(relatorio, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
