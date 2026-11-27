"""Criterios 7 e 8: nada de caminho absoluto ou microdados na entrega versionavel; licenca e citacao no README.

Dados locais em data/raw e data/processed sao permitidos quando ignorados pelo Git. Arquivos ja rastreados,
mesmo ignorados depois, continuam sendo conferidos.
"""

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

FRASE = "licença específica dos microdados não confirmada; aceite de risco residual em 07/11/2026"
CACHES = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}  # nao versionados (.gitignore)
EXT_DADOS = {".zip", ".parquet", ".xls", ".xlsx", ".txt", ".csv", ".db", ".duckdb", ".sqlite"}
PADROES_ABSOLUTOS = [r"/mnt/[a-z]/", r"/home/\w+", r"[A-Za-z]:\\", r"/Users/\w+"]
# Variaveis do projeto que precisam ficar AUSENTES para que estes testes provem que o comando
# funciona sem elas (a configuracao se descobre pela ajuda, nao por um estado de ambiente herdado).
VARS_PROJETO = ("CASE_AMBEV_RAW_ROOT", "CASE_AMBEV_PROCESSADO_ROOT", "CASE_AMBEV_SAIDA_ROOT",
                "CASE_AMBEV_SELADO_ROOT")


def _ambiente_subprocesso(repo):
    """Ambiente isolado para rodar os modulos num subprocesso.

    Parte do ambiente do SO (precisa de SystemRoot etc. para o Winsock/asyncio no Windows; sem eles,
    importar sklearn/joblib quebra com WinError 10106), remove explicitamente as variaveis do projeto
    para preservar o isolamento que o teste verifica, e fixa so PYTHONPATH e COLUMNS.
    """
    env = {k: v for k, v in os.environ.items() if k not in VARS_PROJETO}
    env["PYTHONPATH"] = str(repo / "src")
    env["COLUMNS"] = "200"
    return env


def _arquivos(repo):
    resultado = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        check=True, capture_output=True,
    )
    for nome in sorted(set(resultado.stdout.decode("utf-8").split("\0")) - {""}):
        p = repo / nome
        if p.is_file() and not CACHES & set(p.parts):
            yield p


def test_dados_locais_ignorados_nao_entram_na_entrega_mas_vazamentos_sao_detectados(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("data/raw/\ndata/processed/\n", encoding="utf-8")
    for pasta in ("raw", "processed"):
        destino = tmp_path / "data" / pasta
        destino.mkdir(parents=True)
        (destino / "local.csv").write_text("exemplo\n", encoding="utf-8")
    (tmp_path / "vazamento.csv").write_text("exemplo\n", encoding="utf-8")
    rastreado = "data/raw/rastreado.csv"
    (tmp_path / rastreado).write_text("exemplo\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-f", rastreado], check=True)
    arquivos = {p.relative_to(tmp_path).as_posix() for p in _arquivos(tmp_path)}
    assert "data/raw/local.csv" not in arquivos and "data/processed/local.csv" not in arquivos
    assert "vazamento.csv" in arquivos and rastreado in arquivos
    with pytest.raises(pytest.fail.Exception, match="arquivo de dados no repositorio"):
        test_sem_dado_bruto_ou_derivado_versionavel(tmp_path)


def _texto_a_varrer(repo, p):
    return p.read_text(encoding="utf-8", errors="ignore")


def test_sem_caminho_absoluto_em_codigo_configs_e_docs(repo):
    for p in _arquivos(repo):
        if p.suffix in {".py", ".json", ".md", ".toml", ".example", ""} or p.name == ".gitignore":
            texto = _texto_a_varrer(repo, p)
            for padrao in PADROES_ABSOLUTOS:
                assert not re.search(padrao, texto), f"{p.relative_to(repo)} contem caminho absoluto ({padrao})"


def test_sem_dado_bruto_ou_derivado_versionavel(repo):
    for p in _arquivos(repo):
        if p.suffix in EXT_DADOS:
            pytest.fail(f"arquivo de dados no repositorio: {p.relative_to(repo)}")


def test_gitignore_barra_microdados_e_derivados(repo):
    texto = (repo / ".gitignore").read_text(encoding="utf-8")
    for padrao in ["*.parquet", "*.zip", "data/raw/", "data/processed/", ".venv/"]:
        assert padrao in texto


def test_caminho_dos_dados_vem_de_variavel_de_ambiente():
    from coleta import AmbienteInvalido, raiz_bruta
    from preparo import raiz_processado
    with pytest.raises(AmbienteInvalido, match="CASE_AMBEV_RAW_ROOT"):
        raiz_bruta({})
    assert raiz_bruta({"CASE_AMBEV_RAW_ROOT": "/qualquer"}) == Path("/qualquer")
    assert raiz_processado({"CASE_AMBEV_PROCESSADO_ROOT": "/p"}) == Path("/p")


def test_ajuda_da_linha_de_comando_lista_as_variaveis_de_ambiente(repo):
    """Sem .env.example, a configuracao se descobre pela ajuda do comando (e pelo erro quando falta a variavel)."""
    import subprocess
    import sys
    env = _ambiente_subprocesso(repo)
    for modulo, variaveis in (("coleta", ("CASE_AMBEV_RAW_ROOT",)),
                              ("preparo", ("CASE_AMBEV_RAW_ROOT", "CASE_AMBEV_PROCESSADO_ROOT"))):
        r = subprocess.run([sys.executable, "-m", modulo, "--help"], capture_output=True, text=True, env=env, cwd=repo)
        assert r.returncode == 0, r.stderr
        for var in variaveis:
            assert var in r.stdout.replace("\n", " "), (modulo, var)


def test_dotenv_segue_ignorado_pelo_git(repo):
    linhas = (repo / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in linhas


def test_manifesto_guarda_origem_edicao_url_datas_e_licenca(repo):
    manifesto = json.loads((repo / "configs" / "manifesto_fontes.json").read_text(encoding="utf-8"))
    assert manifesto["licenca_nota"] == FRASE and "IBGE" in manifesto["fonte_dados"]
    for f in manifesto["fontes"]:
        assert f["edicao"] and f["url"] and f["baixado_em"] and f["sha256"]
        assert "IBGE" in f["citacao"] and f["url"] in f["citacao"]


def test_data_do_zip_recuperado_distingue_aquisicao_e_recuperacao(repo):
    manifesto = json.loads((repo / "configs" / "manifesto_fontes.json").read_text(encoding="utf-8"))
    recuperados = [f for f in manifesto["fontes"] if "recuperado_em" in f]
    assert {f["id"] for f in recuperados} == {"Documentacao_20230713.zip"}  # Tradutores nao e lido pelo pipeline
    for f in recuperados:
        assert f["baixado_em"] == "2026-10-27" and f["recuperado_em"] == "2026-11-09" and f["evidencia_datas"]


def test_sem_print_ou_depuracao_no_produto(repo):
    for p in (repo / "src").rglob("*.py"):
        texto = p.read_text(encoding="utf-8")
        assert "breakpoint(" not in texto and "import pdb" not in texto, p.name
        assert not re.search(r"^\s*print\(", texto, re.M), f"print em {p.name}"


def test_nada_exclusivo_de_tendencias_fica_na_entrega(repo):
    """Retirada de 20/11/2026: codigo, configuracao e dados do Trends e do clima foram arquivados fora da entrega."""
    for relativo in ("src/aplicacao/tendencias.py", "src/aplicacao/web/tendencias.js", ".env.example",
                     *(f"src/rotinas/{m}.py" for m in ("coleta", "preparo", "previsao", "previsao_piloto", "atualizacao")),
                     *(f"data/{c}/{f}" for c in ("raw", "processed") for f in ("google_trends", "clima_sp"))):
        assert not (repo / relativo).exists(), relativo
    for arquivo in (repo / "src").rglob("*"):
        if arquivo.suffix in {".py", ".js", ".html", ".css"}:
            texto = arquivo.read_text(encoding="utf-8")
            # "Open-Meteo" e "BrasilAPI" voltaram a lista em 25/11/2026: clima e feriados foram avaliados e retirados.
            for termo in ("APIFY", "Apify", "Google Trends", "aplicacao.tendencias", "rotinas.previsao", "Open-Meteo",
                          "BrasilAPI"):
                assert termo not in texto, (arquivo.name, termo)


# ---- revisao tecnica de 24/11/2026: a arvore so traz o produto vigente (alvo observado) e a sua reproducao ----
# Protocolos de versoes abandonadas, fontes descartadas (Takeaway Londres, DHRD) e o codigo das funcionalidades retiradas
# (personas, CSV da empresa, contexto reduzido, pipeline E2 de 2008) sairam; a copia verificavel fica fora do repositorio.

# Retirada de 25/11/2026: clima, feriados, Delivery Center e contexto por UF foram avaliados e ficaram fora do modelo;
# os scripts, os resultados e as configuracoes exclusivas foram copiados (com SHA-256) para fora do repositorio.
CONFIGS_ATIVAS = {
    "crosswalk.json",  # correspondencia 2008 x 2017
    "manifesto_fontes.json",  # coleta da POF com SHA-256
    "protocolo_v3_alvo_observado.json",  # protocolo fixado antes da comparacao do modelo publicado
    "variaveis.json",  # faixas e categorias usadas no preparo e na inferencia
}


def test_configs_tem_so_os_arquivos_ativos_e_nenhuma_subpasta_de_versao(repo):
    assert {p.name for p in (repo / "configs").iterdir()} == CONFIGS_ATIVAS


def test_fontes_e_tabelas_descartadas_nao_sao_mais_coletadas_nem_preparadas():
    import coleta
    import preparo
    assert coleta.FONTES == ("pof",)
    assert preparo.TABELAS == ("escolhas_2008", "escolhas_2017")


def test_codigo_das_funcionalidades_retiradas_saiu_dos_modulos():
    import coleta
    import preparo
    retirados = {
        coleta: ("conferir_entradas", "carregar_episodios", "feriado_nacional", "clima", "coletar_takeaway",
                 "coletar_dhrd", "coletar_clima", "coletar_feriados", "coletar_delivery_center", "coletar_arquivo",
                 "carregar_catalogo", "LOCAIS_CLIMA"),
        preparo: ("construir_e2_2008", "diagnosticar", "preparar_empresa", "codificar", "celula_de_suporte",
                  "cestas_londres", "volume_delivery_center", "contexto", "feriados_locais", "pascoa"),
    }
    for modulo, nomes in retirados.items():
        assert [n for n in nomes if hasattr(modulo, n)] == [], modulo.__name__


def test_pasta_experimentos_saiu_e_nenhum_comando_depende_dela_nem_do_backup(repo):
    assert not (repo / "experimentos").exists()
    for p in [*(repo / "src").rglob("*.py"), *(repo / "configs").iterdir(), repo / "pyproject.toml",
              repo / ".gitignore", repo / ".gitattributes"]:
        texto = p.read_text(encoding="utf-8")
        for termo in ("experimentos", "_backups"):
            assert termo not in texto, (p.relative_to(repo).as_posix(), termo)


def test_auditoria_e_eda_tem_comando_nos_modulos_responsaveis(repo):
    """A auditoria das refeicoes (preparo) e a EDA do alvo (modelagem) reproduzem a analise exploratoria do TAPI."""
    import subprocess
    import sys
    env = _ambiente_subprocesso(repo)
    for modulo, comando in (("preparo", "auditoria"), ("modelagem", "eda")):
        r = subprocess.run([sys.executable, "-m", modulo, "--help"], capture_output=True, text=True, env=env, cwd=repo)
        assert r.returncode == 0, r.stderr
        assert comando in r.stdout, (modulo, comando)
