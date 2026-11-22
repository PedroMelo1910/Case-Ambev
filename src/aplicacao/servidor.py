"""Interface web local da ferramenta de oportunidade por contexto, so com a biblioteca padrao.

Uso: `PYTHONPATH=src python -m aplicacao.servidor` e abrir o endereco impresso. O servidor nao calcula nada: repassa
para `previsao_consumo.avaliar` (demonstracao historica) ou `empresa_servico` (dados da empresa) e
devolve {"success": true, "data": ...} ou {"success": false, "error": {"type", "message"}}. Entrada invalida vira 422, pedido mal
formado vira 400, falha de integridade vira 500. Escuta so em 127.0.0.1 por padrao: demonstracao local, sem autenticacao.
"""

import argparse
import json
import math
import os
import re
import sys
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from aplicacao import (
    ArtefatoInvalido,
    EntradaInvalida,
    empresa_dados,
    empresa_servico,
    personas,
    previsao_consumo,
    regras,
)
from occasion.variaveis import faixa_hora
from personas import ao_vivo as ao_vivo_personas
from personas import evidencias as acervo_personas
from personas import fluxo as conversa_personas
from personas import llm as llm_personas

WEB = Path(__file__).resolve().parent / "web"
ESTATICOS = {"/": ("index.html", "text/html; charset=utf-8"), "/estilo.css": ("estilo.css", "text/css; charset=utf-8"),
             "/app.js": ("app.js", "text/javascript; charset=utf-8"),
             "/personas.js": ("personas.js", "text/javascript; charset=utf-8")}
RECURSOS = {f"/recursos/{nome}": (f"recursos/{nome}", tipo) for nome, tipo in (
    ("lata-guarana.webp", "image/webp"), ("manrope.woff2", "font/woff2"))}
MARCAS = {"/marca/ambev.svg": ("marca/ambev.svg", "image/svg+xml"), "/marca/ambev.png": ("marca/ambev.png", "image/png")}
MAXIMO_JSON = 16 * 1024
MAXIMO_CSV = 20 * 1024 * 1024
PORTA_PADRAO = 8000
TEMPO_LIMITE = 30
CABECALHOS = {
    "Content-Security-Policy": "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
                               "img-src 'self'; font-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store"}
_TREINO = re.compile(r"^/api/empresa/treinos/([^/]+)$")
_DISPONIBILIZAR = re.compile(r"^/api/empresa/treinos/([^/]+)/disponibilizar$")
FALHAS_INTERNAS = (ArtefatoInvalido, OSError)


class PedidoInvalido(ValueError):
    """Corpo ausente, grande demais, de outro tipo ou fora do formato esperado."""


def _opcao(valor: object, rotulo: str) -> dict:
    return {"valor": valor, "rotulo": rotulo}


def _com_todos(valores, rotulo_de) -> list[dict]:
    return [_opcao(previsao_consumo.TODOS, "Todos"), *[_opcao(v, rotulo_de(v)) for v in valores]]


def opcoes() -> dict:
    pacote = previsao_consumo.carregar()
    resumo = pacote.resumo
    return {
        "dias": [_opcao(d, regras.DIAS[d]) for d in range(1, 8)],
        "periodos": [_opcao(p, regras.PERIODOS[p]) for p in previsao_consumo.PERIODOS],
        "horas": [{**_opcao(h, f"{h:02d}h"), "periodo": regras.PERIODOS[faixa_hora(h)]} for h in range(24)],
        "regioes": [_opcao(r, r) for r in previsao_consumo.REGIOES],
        "areas": [_opcao(a, regras.AREAS[a]) for a in previsao_consumo.AREAS],
        "faixas_etarias": _com_todos(previsao_consumo.FAIXAS_ETARIAS, regras.FAIXAS_ETARIAS.__getitem__),
        "sexos": _com_todos(previsao_consumo.SEXOS, regras.SEXOS.__getitem__),
        "escolaridades": _com_todos(previsao_consumo.ESCOLARIDADES, regras.ESCOLARIDADES.__getitem__),
        "historico": {"versao": previsao_consumo.VERSAO, "fonte": resumo["fonte"], "episodios": resumo["episodios"],
                      "positivos": resumo["positivos"], "referencia": pacote.referencia,
                      "modelo_selecionado": regras.MODELOS[resumo["selecao"]["selecionado"]],
                      "modelo_contextual": regras.MODELOS[pacote.contextual], "papel_do_modelo_contextual": pacote.papel,
                      "dia": previsao_consumo.REPRESENTACOES_DIA[pacote.representacao],
                      "alcance": resumo["alcance_da_avaliacao"]},
        "rotulos": {"faixas": dict(regras.FAIXAS_COMPARACAO), "categorias": dict(regras.CATEGORIAS),
                    "papeis": dict(regras.PAPEIS_ROTULOS), "modelos": dict(regras.MODELOS), "campos": dict(regras.CAMPOS),
                    "motivos": dict(regras.MOTIVOS)},
        "personas": personas.descricoes(),
        "empresa": empresa_servico.modelo_disponivel(),
        "formato_empresa": {"obrigatorios": dict(empresa_dados.COLUNAS_OBRIGATORIAS),
                            "opcionais": dict(empresa_dados.COLUNAS_OPCIONAIS),
                            "tipos_de_registro": dict(empresa_dados.TIPOS_DE_REGISTRO),
                            "descricao": empresa_dados.DESCRICAO}}


def avaliar(pedido: object) -> dict:
    if not isinstance(pedido, dict) or set(pedido) != {"base", "contexto"}:
        raise PedidoInvalido('o pedido deve ter exatamente os campos "base" e "contexto"')
    if pedido["base"] == "historica":
        return previsao_consumo.avaliar(pedido["contexto"])
    if pedido["base"] == "empresa":
        return empresa_servico.avaliar(pedido["contexto"])
    raise EntradaInvalida('base deve ser "historica" ou "empresa"')


def _pedido_da_empresa(pedido: object) -> tuple[str, dict, dict]:
    if not isinstance(pedido, dict) or set(pedido) != {"envio", "mapeamento", "declaracao"}:
        raise PedidoInvalido('o pedido deve ter os campos "envio", "mapeamento" e "declaracao"')
    return pedido["envio"], pedido["mapeamento"], pedido["declaracao"]


def _recusar_constante(texto: str) -> float:
    raise PedidoInvalido(f"número não aceito no pedido: {texto}")


def ler_json(bruto: bytes) -> object:
    try:
        return json.loads(bruto.decode("utf-8"), parse_constant=_recusar_constante)
    except PedidoInvalido:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as erro:
        raise PedidoInvalido("o corpo do pedido não é JSON válido") from erro


def _finitos(valor: object) -> object:
    if isinstance(valor, float) and not math.isfinite(valor):
        return None
    if isinstance(valor, dict):
        return {chave: _finitos(item) for chave, item in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_finitos(item) for item in valor]
    return valor


class Pagina(BaseHTTPRequestHandler):
    server_version = "OportunidadePorContexto"
    sys_version = ""
    timeout = TEMPO_LIMITE

    def _enviar(self, estado: HTTPStatus, corpo: bytes, tipo: str, extras: dict | None = None) -> None:
        self.send_response(estado)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(corpo)))
        for nome, valor in {**CABECALHOS, **(extras or {})}.items():
            self.send_header(nome, valor)
        try:
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(corpo)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _sucesso(self, dados: object, estado: HTTPStatus = HTTPStatus.OK) -> None:
        corpo = json.dumps({"success": True, "data": _finitos(dados)}, ensure_ascii=False, allow_nan=False, default=str)
        self._enviar(estado, corpo.encode("utf-8"), "application/json; charset=utf-8")

    def _erro(self, estado: HTTPStatus, tipo: str, mensagem: str) -> None:
        corpo = json.dumps({"success": False, "error": {"type": tipo, "message": mensagem}}, ensure_ascii=False)
        self._enviar(estado, corpo.encode("utf-8"), "application/json; charset=utf-8")

    def _executar(self, funcao, estado: HTTPStatus = HTTPStatus.OK) -> None:
        try:
            self._sucesso(funcao(), estado)
        except PedidoInvalido as erro:
            self.close_connection = True
            self._erro(HTTPStatus.BAD_REQUEST, "pedido_invalido", str(erro))
        except EntradaInvalida as erro:
            self._erro(HTTPStatus.UNPROCESSABLE_ENTITY, "entrada_invalida", str(erro))
        except FALHAS_INTERNAS as erro:
            self._erro(HTTPStatus.INTERNAL_SERVER_ERROR, "integridade",
                       f"falha de integridade dos arquivos da ferramenta ({type(erro).__name__})")
        except Exception:  # ultima barreira: o cliente recebe resposta e o detalhe vai para o terminal do servidor
            traceback.print_exc(file=sys.stderr)
            self._erro(HTTPStatus.INTERNAL_SERVER_ERROR, "interna", "falha interna inesperada; veja o terminal do servidor")

    def _host_valido(self) -> bool:
        """Recusa nomes de host externos (protecao contra DNS rebinding): so endereco local com a porta em uso."""
        porta = self.server.server_address[1]
        if self.headers.get("Host", "") in {f"127.0.0.1:{porta}", f"localhost:{porta}"}:
            return True
        self._erro(HTTPStatus.FORBIDDEN, "host_recusado", "endereço de acesso não permitido")
        return False

    def do_GET(self) -> None:
        if not self._host_valido():
            return None
        caminho = self.path.split("?", 1)[0]
        if caminho == "/api/opcoes":
            return self._executar(opcoes)
        if caminho == "/api/empresa/exemplo.csv":
            return self._enviar(HTTPStatus.OK, empresa_dados.exemplo_csv().encode("utf-8"), "text/csv; charset=utf-8",
                                {"Content-Disposition": 'attachment; filename="exemplo_pedidos_ficticio.csv"'})
        if caminho == "/api/empresa/treinos":
            return self._executar(empresa_servico.listar)
        if treino := _TREINO.match(caminho):
            return self._executar(lambda: empresa_servico.resultado(treino.group(1)))
        arquivo = ESTATICOS.get(caminho) or RECURSOS.get(caminho) or MARCAS.get(caminho)
        if arquivo is None or not (WEB / arquivo[0]).is_file():
            return self._erro(HTTPStatus.NOT_FOUND, "nao_encontrado", "endereço inexistente")
        self._enviar(HTTPStatus.OK, (WEB / arquivo[0]).read_bytes(), arquivo[1])

    do_HEAD = do_GET

    def _corpo(self, tipo: str, maximo: int) -> bytes:
        if self.headers.get_content_type() != tipo:
            raise PedidoInvalido(f"o pedido deve ter Content-Type {tipo}")
        declarado = self.headers.get("Content-Length", "")
        if not (declarado.isascii() and declarado.isdigit()) or len(declarado) > 9:
            raise PedidoInvalido("Content-Length ausente ou inválido")
        if not 0 < int(declarado) <= maximo:
            raise PedidoInvalido(f"o corpo do pedido deve ter de 1 a {maximo} bytes")
        return self.rfile.read(int(declarado))

    def _json(self) -> object:
        return ler_json(self._corpo("application/json", MAXIMO_JSON))

    def do_POST(self) -> None:
        if not self._host_valido():
            return None
        caminho = self.path.split("?", 1)[0]
        if caminho == "/api/avaliar":
            return self._executar(lambda: avaliar(self._json()))
        if caminho == "/api/personas/conversa":
            return self._executar(lambda: personas.conversar(self._json()))
        if caminho == "/api/conversa":
            return self._executar(lambda: conversa_personas.conversar(
                self._json(), redator=self.server.redator,
                clima_ao_vivo=self.server.clima_ao_vivo, feriado_ao_vivo=self.server.feriado_ao_vivo))
        if caminho == "/api/empresa/envios":
            return self._executar(lambda: empresa_servico.receber_arquivo(self._corpo("text/csv", MAXIMO_CSV)))
        if caminho == "/api/empresa/diagnostico":
            return self._executar(lambda: empresa_servico.diagnosticar(*_pedido_da_empresa(self._json())))
        if caminho == "/api/empresa/treinos":
            return self._executar(lambda: empresa_servico.iniciar_treino(*_pedido_da_empresa(self._json())), HTTPStatus.ACCEPTED)
        if disponibilizar := _DISPONIBILIZAR.match(caminho):
            return self._executar(lambda: empresa_servico.disponibilizar(disponibilizar.group(1)))
        if caminho == "/api/empresa/retirar":
            return self._executar(lambda: empresa_servico.retirar() or {"retirado": True})
        self._erro(HTTPStatus.NOT_FOUND, "nao_encontrado", "endereço inexistente")

    def log_message(self, formato_: str, *args: object) -> None:  # o contexto e os arquivos enviados nao sao registrados
        return


def criar_servidor(host: str = "127.0.0.1", porta: int = PORTA_PADRAO) -> ThreadingHTTPServer:
    """Confere os artefatos da versao reduzida antes de aceitar pedidos; falha aqui impede a subida."""
    previsao_consumo.carregar()
    acervo_personas.carregar()  # acervo e indice FTS5 em memoria: cartao fora do formato impede a subida
    empresa_servico.marcar_orfaos()
    instancia = ThreadingHTTPServer((host, porta), Pagina)
    instancia.redator = llm_personas.configurado()  # None: modo deterministico (padrao); a chave fica so aqui
    # Coleta ao vivo de clima/feriado para a fala direcional das personas: ligada por CASE_AMBEV_AO_VIVO=1.
    # Desligada por padrao (nenhuma chamada de rede); a degradacao graciosa protege contra falha quando ligada.
    if os.environ.get("CASE_AMBEV_AO_VIVO", "").strip() in ("1", "true", "sim"):
        instancia.clima_ao_vivo = ao_vivo_personas.clima
        instancia.feriado_ao_vivo = ao_vivo_personas.feriado_nacional
    else:
        instancia.clima_ao_vivo = None
        instancia.feriado_ao_vivo = None
    return instancia


def main() -> None:
    parser = argparse.ArgumentParser(description="Ferramenta local: oportunidade por contexto (pizza + guaraná).")
    parser.add_argument("--host", default="127.0.0.1", help="endereço de escuta (padrão: só esta máquina)")
    parser.add_argument("--porta", type=int, default=PORTA_PADRAO, help=f"porta (padrão: {PORTA_PADRAO})")
    argumentos = parser.parse_args()
    servidor = criar_servidor(argumentos.host, argumentos.porta)
    sys.stdout.write(f"Ferramenta disponível em http://{argumentos.host}:{servidor.server_address[1]}  (Ctrl+C encerra)\n")
    modo = f"Groq ({servidor.redator.modelo})" if servidor.redator else "determinístico (sem modelo de linguagem)"
    sys.stdout.write(f"Redação das conversas: {modo}\n")
    sys.stdout.flush()
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        servidor.server_close()


if __name__ == "__main__":
    main()
