"""Operacoes do modo "dados da empresa" usadas pela interface: enviar, diagnosticar, planejar, treinar em segundo
plano, acompanhar, disponibilizar e avaliar. Linhas enviadas nunca saem daqui para a interface, exceto a previa.

O modo tem tres modulos: `empresa_dados` (formato, leitura, diagnostico e codificacao das variaveis, a mesma no treino
e na inferencia), `empresa_modelo` (treino com selecao e inferencia) e este (armazenamento e operacoes). Nada aqui
presume que a empresa tenha os dados nem que eles sustentem a pergunta: o diagnostico recusa o treino quando o alvo
(guarana na mesma unidade de uma ocasiao com pizza) nao pode ser observado. Dados e modelos ficam fora do
repositorio, em CASE_AMBEV_EMPRESA_ROOT.

Armazenamento fora do repositorio: CASE_AMBEV_EMPRESA_ROOT ou ~/.case_ambev_empresa. Identificadores sao hexadecimais
gerados aqui; qualquer outro valor e recusado antes de virar caminho. O treino roda numa linha de execucao com etapas
gravadas em estado.json; a falha vira estado legivel; um treino por vez. A pasta e confiavel por premissa (uso local):
o hash de `modelo.joblib` e guardado tambem em disponivel.json, mas quem reescreve a pasta inteira reescreve os dois.
"""

import json
import os
import re
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from aplicacao import ArtefatoInvalido, EntradaInvalida, conferir_artefatos, empresa_dados, empresa_modelo, recebido

# ---------- armazenamento ----------

VARIAVEL = "CASE_AMBEV_EMPRESA_ROOT"
_ID = re.compile(r"[0-9a-f]{32}")
_TRAVA = threading.Lock()


def raiz() -> Path:
    pasta = Path(os.environ.get(VARIAVEL) or Path.home() / ".case_ambev_empresa")
    repositorio = Path(__file__).resolve().parents[2]
    if pasta.resolve() == repositorio or repositorio in pasta.resolve().parents:
        raise EntradaInvalida(f"{VARIAVEL} não pode apontar para dentro do repositório de entrega")
    return pasta


def novo_id() -> str:
    return uuid.uuid4().hex


def conferir_id(valor: object) -> str:
    if type(valor) is not str or not _ID.fullmatch(valor):
        raise EntradaInvalida(f"identificador inválido: {recebido(valor)}")
    return valor


def pasta_de(tipo: str, identificador: str) -> Path:
    return raiz() / tipo / conferir_id(identificador)


def gravar_json(caminho: Path, conteudo: dict) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_suffix(".tmp")
    with _TRAVA:
        temporario.write_text(json.dumps(conteudo, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        temporario.replace(caminho)


def ler_json(caminho: Path) -> dict:
    try:
        return json.loads(caminho.read_text(encoding="utf-8"))
    except FileNotFoundError as erro:
        raise EntradaInvalida(f"registro não encontrado: {caminho.parent.name}") from erro
    except ValueError as erro:
        raise EntradaInvalida(f"registro ilegível: {caminho.parent.name}/{caminho.name}") from erro


# ---------- treino em segundo plano ----------

NOMES = {"validacao": "Validação do arquivo e do plano", "preparacao": "Preparação das variáveis",
         "ajuste_e_selecao": "Ajuste e escolha do modelo", "teste_final": "Avaliação no período final",
         "reamostragem": "Cálculo da incerteza", "gravacao": "Gravação do modelo"}
_ATIVAS: dict = {}
_INICIO = threading.Lock()


def _agora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _caminho(identificador: str):
    return pasta_de("experimentos", identificador) / "estado.json"


def estado_do_treino(identificador: str) -> dict:
    return ler_json(_caminho(identificador))


def _marcar(identificador: str, nome: str) -> None:
    atual = estado_do_treino(identificador)
    etapas = [{**e, "situacao": "concluida", "fim": e.get("fim") or _agora()} if e["situacao"] == "em_andamento" else e
              for e in atual["etapas"]]
    etapas.append({"nome": nome, "descricao": NOMES[nome], "situacao": "em_andamento", "inicio": _agora(), "fim": None})
    gravar_json(_caminho(identificador), {**atual, "etapas": etapas})


def _encerrar(identificador: str, situacao: str, erro: str | None = None) -> None:
    atual = estado_do_treino(identificador)
    final = "concluida" if situacao == "concluido" else "falhou"
    etapas = [{**e, "situacao": final, "fim": _agora()} if e["situacao"] == "em_andamento" else e for e in atual["etapas"]]
    gravar_json(_caminho(identificador), {**atual, "situacao": situacao, "etapas": etapas, "erro": erro, "fim": _agora()})


def _estados_gravados() -> list[tuple[str, dict]]:
    """Estados legiveis dos experimentos; pastas com nome fora do padrao ou estado ilegivel sao ignoradas."""
    pasta = raiz() / "experimentos"
    saida = []
    for item in (sorted(pasta.iterdir()) if pasta.is_dir() else []):
        if not _ID.fullmatch(item.name) or not (item / "estado.json").is_file():
            continue
        try:
            saida.append((item.name, ler_json(item / "estado.json")))
        except EntradaInvalida:
            continue
    return saida


def marcar_orfaos() -> int:
    """Treinos que ficaram "em_andamento" de um processo anterior viram falha com motivo (chamado ao subir o servidor)."""
    marcados = 0
    for identificador, gravado in _estados_gravados():
        if identificador not in _ATIVAS and gravado.get("situacao") == "em_andamento":
            _encerrar(identificador, "falhou", "treino interrompido: o servidor foi encerrado antes do fim")
            marcados += 1
    return marcados


def _mensagem(erro: Exception) -> str:
    """Motivo legivel sem caminhos locais."""
    texto = re.sub(r"(/[^\s'\"]+)+", "<caminho>", str(erro))[:300]
    return f"{type(erro).__name__}: {texto}"


def iniciar(identificador: str, trabalho: Callable[[Callable[[str], None]], dict]) -> None:
    with _INICIO:
        if _ATIVAS:
            raise EntradaInvalida("já há um treino em andamento; aguarde o fim dele para iniciar outro")
        _ATIVAS[identificador] = None
    try:
        gravar_json(_caminho(identificador), {"id": identificador, "situacao": "em_andamento", "inicio": _agora(),
                                              "etapas": [], "erro": None, "fim": None})
        _marcar(identificador, "validacao")
    except BaseException:
        _ATIVAS.pop(identificador, None)  # sem isso, uma falha aqui bloquearia todo treino seguinte
        raise

    def executar() -> None:
        try:
            trabalho(lambda nome: _marcar(identificador, nome))
            _encerrar(identificador, "concluido")
        except Exception as erro:  # qualquer falha do treino precisa aparecer na interface, nao sumir na thread
            _encerrar(identificador, "falhou", _mensagem(erro))
        finally:
            _ATIVAS.pop(identificador, None)

    linha = threading.Thread(target=executar, name=f"treino-{identificador[:8]}", daemon=True)
    _ATIVAS[identificador] = linha
    try:
        linha.start()
    except BaseException:
        _ATIVAS.pop(identificador, None)
        raise


def em_andamento() -> list[str]:
    return list(_ATIVAS)


# ---------- operacoes ----------

DISPONIVEL = "disponivel.json"


def _publico(diagnostico: dict) -> dict:
    return {k: v for k, v in diagnostico.items() if k != "registros"}


def _tabela(envio: str) -> empresa_dados.Tabela:
    try:
        conteudo = (pasta_de("envios", envio) / "dados.csv").read_bytes()
    except FileNotFoundError as erro:
        raise EntradaInvalida("envio não encontrado: envie o arquivo de novo") from erro
    return empresa_dados.ler(conteudo)


def receber_arquivo(conteudo: bytes) -> dict:
    tabela = empresa_dados.ler(conteudo)
    envio = novo_id()
    destino = pasta_de("envios", envio)
    destino.mkdir(parents=True)
    (destino / "dados.csv").write_bytes(conteudo)
    registro = {"id": envio, "sha256": tabela.sha256, "colunas": tabela.colunas, "linhas": len(tabela.linhas),
                "codificacao": tabela.codificacao, "separador": tabela.separador,
                "recebido_em": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    gravar_json(destino / "envio.json", registro)
    return {**registro, "previa": tabela.previa(), "sugestao": empresa_dados.sugerir_mapeamento(tabela.colunas),
            "colunas_pessoais": empresa_dados.colunas_pessoais(tabela)}


def diagnosticar(envio: str, mapeamento: dict, declaracao: dict) -> dict:
    diagnostico = empresa_dados.diagnosticar(_tabela(envio), mapeamento, declaracao)
    saida = _publico(diagnostico)
    if diagnostico["pode_treinar"]:
        saida["plano"] = empresa_modelo.montar_plano(diagnostico)
    return saida


def iniciar_treino(envio: str, mapeamento: dict, declaracao: dict) -> dict:
    diagnostico = empresa_dados.diagnosticar(_tabela(envio), mapeamento, declaracao)
    plano_do_experimento = empresa_modelo.montar_plano(diagnostico)  # recusa com motivo se o arquivo nao sustenta a pergunta
    experimento = novo_id()
    pasta = pasta_de("experimentos", experimento)
    gravar_json(pasta / "plano.json", {**plano_do_experimento, "envio": envio})  # registrado antes de treinar
    iniciar(experimento, lambda etapa: empresa_modelo.executar(plano_do_experimento, diagnostico, pasta / "artefatos", etapa))
    return {"id": experimento, "plano": plano_do_experimento}


def resultado(experimento: str) -> dict:
    pasta = pasta_de("experimentos", experimento)
    estado = estado_do_treino(experimento)
    saida = {"estado": estado, "plano": ler_json(pasta / "plano.json")}
    if estado["situacao"] == "concluido":
        saida["resultado"] = ler_json(pasta / "artefatos" / "resultado.json")
    return saida


def listar() -> list[dict]:
    itens = [{"id": identificador, "situacao": gravado.get("situacao", "desconhecida"), "inicio": gravado.get("inicio", "")}
             for identificador, gravado in _estados_gravados()]
    return sorted(itens, key=lambda i: str(i["inicio"]), reverse=True)


def disponibilizar(experimento: str) -> dict:
    if estado_do_treino(experimento)["situacao"] != "concluido":
        raise EntradaInvalida("só um treino concluído pode ser disponibilizado")
    meta = conferir_artefatos(pasta_de("experimentos", experimento) / "artefatos", "modelo da empresa")
    registro = {"experimento": conferir_id(experimento), "modelo_sha256": meta["sha256"]["modelo.joblib"],
                "disponibilizado_em": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    gravar_json(raiz() / DISPONIVEL, registro)
    return registro


def retirar() -> None:
    (raiz() / DISPONIVEL).unlink(missing_ok=True)


def modelo_disponivel() -> dict | None:
    caminho = raiz() / DISPONIVEL
    if not caminho.is_file():
        return None
    try:
        registro = ler_json(caminho)
        _, resultado_do_treino, meta = empresa_modelo.carregar(pasta_de("experimentos", registro["experimento"]) / "artefatos")
        if meta["sha256"]["modelo.joblib"] != registro["modelo_sha256"]:
            raise ArtefatoInvalido("modelo diferente do disponibilizado")
    except (ArtefatoInvalido, EntradaInvalida, OSError, ValueError, KeyError) as erro:
        return {"indisponivel": True, "motivo": f"o modelo disponibilizado não pôde ser carregado: {type(erro).__name__}"}
    return {**registro, "campos": resultado_do_treino["campos"], "canais": resultado_do_treino["canais"],
            "validacao": resultado_do_treino["validacao"]}


def avaliar(contexto: dict) -> dict:
    disponivel = modelo_disponivel()
    if disponivel is None:
        raise EntradaInvalida("nenhum modelo treinado com dados da empresa foi disponibilizado")
    if disponivel.get("indisponivel"):
        raise ArtefatoInvalido(disponivel["motivo"])
    pasta = pasta_de("experimentos", disponivel["experimento"]) / "artefatos"
    return empresa_modelo.avaliar(contexto, pasta, disponivel["experimento"])
