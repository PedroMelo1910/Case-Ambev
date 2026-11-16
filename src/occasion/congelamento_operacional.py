"""Manifesto operacional: congela a avaliacao sem retreinar e sem afrouxar o congelamento do treinamento.

O gerador original (`congelamento.manifesto_da_execucao`) exige que TODO o codigo atual seja o da execucao de origem. Aqui
essa exigencia continua valendo para tudo o que a execucao registrou, exceto os arquivos que o protocolo operacional
declara como codigo operacional, posteriores a execucao e fora do caminho de treinamento. O protocolo de treinamento
continua sendo o que a execucao usou; o protocolo operacional so pode tratar da conducao da avaliacao.

O manifesto mantem o formato lido por `congelamento.verificar_manifesto`, que segue conferindo todos os arquivos.
"""

import ast
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from occasion import congelamento as cg
from occasion import modelagem as m

MODULO_DE_TREINAMENTO = "modelagem_2008"
PACOTE = "occasion"
CAMPOS_PRESERVADOS = ("selecionado_em_2008", "comparadores_previamente_definidos", "papel", "sem_reajuste",
                      "controles_registrados", "autorizacao", "rotulo_obrigatorio", "relato")
BLOCOS_OPERACIONAIS = {"versao", "data", "tipo", "descricao", "protocolo_de_treinamento", "avaliacao_temporal", "congelamento",
                       "registro_de_aberturas"}
_SHA256 = re.compile(r"[0-9a-f]{64}")
Invalido = cg.CongelamentoInvalido


def _importados(arvore: ast.AST) -> list[str]:
    """Nomes dos modulos do pacote importados por um arquivo (imports absolutos ou relativos)."""
    nomes = []
    for no in ast.walk(arvore):
        if isinstance(no, ast.Import):
            nomes += [a.name.split(".")[1] for a in no.names if a.name.startswith(PACOTE + ".")]
        elif isinstance(no, ast.ImportFrom):
            partes = (no.module or "").split(".")
            if no.level:
                nomes += [partes[0]] if no.module else [a.name for a in no.names]
            elif partes[0] == PACOTE:
                nomes += [partes[1]] if len(partes) > 1 else [a.name for a in no.names]
    return nomes


def fecho_do_treinamento(repo: Path) -> set[str]:
    """Arquivos do pacote alcancados por import a partir do modulo de treinamento (caminhos relativos ao repositorio)."""
    pacote = Path(repo) / "src" / PACOTE
    vistos, pendentes = set(), [MODULO_DE_TREINAMENTO]
    while pendentes:
        nome = pendentes.pop()
        arquivo = pacote / f"{nome}.py"
        if nome in vistos or not arquivo.is_file():
            continue
        vistos.add(nome)
        pendentes += _importados(ast.parse(arquivo.read_text(encoding="utf-8")))
    return {f"src/{PACOTE}/__init__.py", *(f"src/{PACOTE}/{nome}.py" for nome in vistos)}


def _escopos(repo: Path, usado: Mapping[str, str], atual: Mapping[str, str], operacionais: set[str]) -> dict:
    """Separa o codigo em treinamento (igual ao da execucao, sem excecao) e operacional (declarado). Recusa o resto."""
    no_caminho = sorted(operacionais & fecho_do_treinamento(repo))
    if no_caminho:
        raise Invalido(f"arquivo declarado operacional esta no caminho de treinamento: {no_caminho}")
    ausentes = sorted(operacionais - atual.keys())
    if ausentes:
        raise Invalido(f"arquivo operacional declarado e ausente do repositorio: {ausentes}")
    treinamento = {caminho: sha for caminho, sha in usado.items() if caminho not in operacionais}
    divergentes = sorted(caminho for caminho, sha in treinamento.items() if atual.get(caminho) != sha)
    if divergentes:
        raise Invalido(f"codigo do treinamento difere do da execucao de origem: {divergentes}")
    novos = sorted(caminho for caminho in atual if caminho not in usado and caminho not in operacionais)
    if novos:
        raise Invalido(f"arquivo novo, ausente da execucao de treinamento e nao declarado como operacional: {novos}")
    return {"treinamento": treinamento,
            "operacionais": {caminho: {"na_execucao": usado.get(caminho), "no_manifesto": atual[caminho]}
                             for caminho in sorted(operacionais)}}


def _protocolos(repo: Path, protocolos: Mapping[str, str], treinamento: str, operacional: str, reg: dict) -> tuple[dict, dict, dict]:
    """(hashes de todos os protocolos, protocolo de treinamento, protocolo operacional), com as conferencias entre eles."""
    blocos = {versao: {"arquivo": rel, "sha256": m.sha256_arquivo(repo / rel)} for versao, rel in protocolos.items()}
    prot = m.carregar_protocolo(repo / protocolos[treinamento])
    if prot.get("versao") != treinamento or reg.get("protocolo_sha256") != blocos[treinamento]["sha256"]:
        raise Invalido(f"a execucao nao usou o protocolo de treinamento {treinamento} registrado")
    oper = m.carregar_protocolo(repo / protocolos[operacional])
    referencia = oper.get("protocolo_de_treinamento") or {}
    if (oper.get("versao") != operacional or referencia.get("versao") != treinamento
            or referencia.get("arquivo") != protocolos[treinamento] or referencia.get("sha256") != blocos[treinamento]["sha256"]):
        raise Invalido(f"protocolo operacional {operacional} nao identifica o protocolo de treinamento {treinamento} pelo hash")
    fora = sorted(set(oper) - BLOCOS_OPERACIONAIS)
    if fora:
        raise Invalido(f"protocolo operacional traz bloco fora do seu escopo: {fora}")
    mudados = [campo for campo in CAMPOS_PRESERVADOS if oper["avaliacao_temporal"].get(campo) != prot["avaliacao_temporal"].get(campo)]
    if mudados:
        raise Invalido(f"protocolo operacional altera decisao do protocolo de treinamento: {mudados}")
    return blocos, prot, oper


def manifesto_operacional(*, repo: Path, base: Path, execucao: str, protocolos: Mapping[str, str], protocolo_treinamento: str,
                          protocolo_operacional: str, registro_de_aberturas: Path, entradas_avaliacao: Mapping[str, str],
                          leituras_revisadas: dict | None = None, execucoes_anteriores: dict | None = None,
                          registros_extra: Mapping[str, str] | None = None, controles: dict | None = None) -> dict:
    """Monta o manifesto a partir de uma execucao ja gravada. Recusa se algum arquivo do treinamento difere do da
    execucao, se o protocolo operacional toca em decisao do treinamento ou se algum hash nao confere."""
    repo, base = Path(repo), Path(base)
    pasta = base / execucao
    reg, res = cg._ler_json(pasta / "registro.json"), cg._ler_json(pasta / "resultados.json")
    blocos, prot, oper = _protocolos(repo, protocolos, protocolo_treinamento, protocolo_operacional, reg)
    usado = reg.get("codigo") or {}
    if usado.get("inalterado") is not True:
        raise Invalido("o codigo mudou durante a execucao ou o hash do codigo nao foi registrado")
    codigo = cg.hashes_do_codigo(repo)
    escopos = _escopos(repo, usado.get("antes") or {}, codigo, set(oper["congelamento"]["codigo_operacional"]))
    if prot["classes"]["comparacao_numerica"]["casas_decimais"] != m.CASAS_EMPATE:
        raise Invalido("precisao numerica do protocolo difere da do codigo")
    temporal, escolhido = prot["avaliacao_temporal"], res["escolha_provisoria"]["escolhido"]
    if escolhido != temporal["selecionado_em_2008"]:
        raise Invalido(f"selecionado na execucao ({escolhido}) difere do protocolo ({temporal['selecionado_em_2008']})")
    comparadores = list(temporal["comparadores_previamente_definidos"])
    artefatos = cg._artefatos(base, execucao, res)
    if not set(comparadores) <= set(artefatos):
        raise Invalido(f"artefato ausente para comparadores: {sorted(set(comparadores) - set(artefatos))}")
    invalidas = sorted(nome for nome, sha in entradas_avaliacao.items() if not isinstance(sha, str) or not _SHA256.fullmatch(sha))
    if invalidas:
        raise Invalido(f"entrada da avaliacao sem sha256 valido: {invalidas}")
    ref = prot["entradas"]["taxa_referencia"]
    colunas = cg._ler_json(repo / "configs" / "variaveis.json")["colunas_logistica"]
    anteriores = {nome: {**{k: x for k, x in v.items() if k != "arquivos"},
                         "arquivos_sha256": {rel: cg._hash_relativo(base, rel) for rel in v.get("arquivos", [])}}
                  for nome, v in (execucoes_anteriores or {}).items()}
    registros = {f"{execucao}/{a}": cg._hash_relativo(base, f"{execucao}/{a}") for a in cg._ARQUIVOS_EXECUCAO}
    registros.update({rel: cg._hash_relativo(base, rel) for rel in (registros_extra or {}).values()})
    operacional = oper["avaliacao_temporal"]
    manifesto = {
        "versao_manifesto": cg.VERSAO_MANIFESTO, "gerado_em_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocolo_vigente": protocolo_operacional, "protocolos": blocos,
        "protocolo_de_treinamento": {"versao": protocolo_treinamento, **blocos[protocolo_treinamento]},
        "protocolo_operacional": {"versao": protocolo_operacional, **blocos[protocolo_operacional]},
        "entradas_sha256": reg.get("entradas_sha256", {}),
        "entradas_avaliacao_sha256": dict(entradas_avaliacao),
        "codigo": {"raiz": "repositorio", "sha256": codigo, **escopos, "regra": oper["congelamento"]["codigo_de_treinamento"]},
        "execucao_de_origem": {"pasta": execucao, "codigo_inalterado": True, "codigo_de_treinamento_igual_ao_da_execucao": True,
                               "codigo_igual_ao_do_manifesto": usado.get("antes") == codigo,
                               "protocolo_sha256": reg["protocolo_sha256"]},
        "execucoes_anteriores": anteriores,
        "ambiente": {**cg.versoes_do_ambiente(), "uv_lock_sha256": codigo.get("uv.lock"), "threads": reg.get("threads")},
        "artefatos": artefatos,
        "selecao": {"selecionado": escolhido, "comparadores": comparadores, "m_estrela_2008": res["m_estrela"],
                    "regra": prot["avaliacao"]["regra_escolha"], "papel_dos_comparadores": temporal["papel"],
                    "motivo": "parcimonia: ganho nao demonstrado, que nao comprova equivalencia"},
        "constantes": {"p0": cg._ler_json(cg._caminho_relativo(base, artefatos["B0"]["arquivo"]))["p0"],
                       "b": ref["positivos"] / ref["episodios"], "b_fracao": f"{ref['positivos']}/{ref['episodios']}",
                       "casas_comparacao": m.CASAS_EMPATE, "ordem_classes": prot["classes"]["ordem"],
                       "coberturas": prot["limiar"]["coberturas"], "cobertura_operacional": prot["limiar"]["operacional"]},
        "modelos": {nome: cg._bloco_modelo(nome, res, prot["candidatos"]["G"], colunas) for nome in comparadores},
        "registros": registros,
        "registro_de_aberturas": {"raiz": str(Path(registro_de_aberturas).resolve()), **oper["registro_de_aberturas"]},
        "avaliacao_temporal": {"controles": {k: (controles or temporal["controles_registrados"])[k] for k in cg._CONTROLES},
                               "autorizacao_formato": temporal["autorizacao"]["formato"], "comparadores": comparadores,
                               "selecionado": escolhido, "papel": temporal["papel"], "sem_reajuste": temporal["sem_reajuste"],
                               "rotulo_obrigatorio": temporal["rotulo_obrigatorio"], "relato": temporal["relato"],
                               "bootstrap": prot["avaliacao"]["bootstrap"],
                               "retomada_formato": operacional["retomada"]["formato"], "estados": operacional["estados"],
                               "divergencia_de_controle": operacional["divergencia_de_controle"]}}
    if leituras_revisadas is not None:
        manifesto["leituras_corrigidas"] = cg._leituras(base, execucao, res, leituras_revisadas, anteriores)
    return manifesto
