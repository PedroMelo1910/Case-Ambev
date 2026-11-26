"""API e servico da interface web (HTML, CSS e JavaScript em `web/`).

Comando: python -m aplicacao.servidor [--porta 8000]. Rotas: GET /api/opcoes e POST /api/avaliar (corpo
{"contexto": {...}}), que chama `modelagem.avaliar` com o Occasion Score observado. Resposta no envelope
{"success": bool, "data" | "error"}.
"""

import argparse
import json
import math
import sys
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import modelagem
import preparo
from aplicacao import regras

WEB = Path(__file__).resolve().parent / "web"
ESTATICOS = {"/": ("index.html", "text/html; charset=utf-8"), "/estilo.css": ("estilo.css", "text/css; charset=utf-8"),
             "/app.js": ("app.js", "text/javascript; charset=utf-8")}
RECURSOS = {f"/recursos/{nome}": (f"recursos/{nome}", tipo) for nome, tipo in (("manrope.woff2", "font/woff2"),)}
MAXIMO_JSON = 16 * 1024
PORTA_PADRAO = 8000
TEMPO_LIMITE = 30
CABECALHOS = {
    "Content-Security-Policy": "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
                               "img-src 'self'; font-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store"}
FALHAS_INTERNAS = (preparo.ArtefatoInvalido, OSError)


class PedidoInvalido(ValueError):
    """Corpo ausente, grande demais, de outro tipo ou fora do formato esperado."""


def _opcao(valor: object, rotulo: str) -> dict:
    return {"valor": valor, "rotulo": rotulo}


def _com_todos(valores, rotulo_de) -> list[dict]:
    return [_opcao(preparo.TODOS, "Público geral"), *[_opcao(v, rotulo_de(v)) for v in valores]]


def opcoes() -> dict:
    pacote = modelagem.carregar()
    resumo = pacote.resumo
    return {
        "dias": [_opcao(d, regras.DIAS[d]) for d in range(1, 8)],
        "periodos": [_opcao(p, regras.PERIODOS[p]) for p in preparo.PERIODOS],
        "horas": [{**_opcao(h, f"{h:02d}h"), "periodo": preparo.faixa_hora(h)} for h in range(24)],
        "regioes": [_opcao(r, r) for r in preparo.REGIOES],
        "areas": [_opcao(a, regras.AREAS[a]) for a in preparo.AREAS],
        "faixas_etarias": _com_todos(preparo.FAIXAS_ETARIAS, regras.FAIXAS_ETARIAS.__getitem__),
        "sexos": _com_todos(preparo.SEXOS, regras.SEXOS.__getitem__),
        "escolaridades": _com_todos(preparo.ESCOLARIDADES, regras.ESCOLARIDADES.__getitem__),
        "modelo": {"versao": modelagem.VERSAO, "fonte": modelagem.BASE_DA_ESTIMATIVA, "alvo": modelagem.MEDIDA,
                   "nome": regras.MODELOS[resumo["spec"]["familia"]], "refeicoes": resumo["refeicoes_treino"],
                   "positivos": resumo["positivos_treino"], "referencia": resumo["taxa_global_treino"],
                   "avaliacao_teste": resumo["avaliacao"],
                   "importancia": {c: v["sustentada"] for c, v in pacote.importancia_geral.items()},
                   "origem_da_importancia": modelagem.ORIGEM_IMPORTANCIA},
        "rotulos": {"faixas": dict(regras.FAIXAS_COMPARACAO), "categorias": dict(regras.CATEGORIAS),
                    "modelos": dict(regras.MODELOS), "campos": dict(regras.CAMPOS), "motivos": dict(regras.MOTIVOS)}}


def avaliar(pedido: object) -> dict:
    if not isinstance(pedido, dict) or set(pedido) != {"contexto"}:
        raise PedidoInvalido('o pedido deve ter exatamente o campo "contexto"')
    return modelagem.avaliar(pedido["contexto"])


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
    server_version = "OccasionScore"
    sys_version = ""
    timeout = TEMPO_LIMITE

    def _enviar(self, estado: HTTPStatus, corpo: bytes, tipo: str) -> None:
        self.send_response(estado)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(corpo)))
        for nome, valor in CABECALHOS.items():
            self.send_header(nome, valor)
        try:
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(corpo)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _sucesso(self, dados: object) -> None:
        corpo = json.dumps({"success": True, "data": _finitos(dados)}, ensure_ascii=False, allow_nan=False, default=str)
        self._enviar(HTTPStatus.OK, corpo.encode("utf-8"), "application/json; charset=utf-8")

    def _erro(self, estado: HTTPStatus, tipo: str, mensagem: str) -> None:
        corpo = json.dumps({"success": False, "error": {"type": tipo, "message": mensagem}}, ensure_ascii=False)
        self._enviar(estado, corpo.encode("utf-8"), "application/json; charset=utf-8")

    def _executar(self, funcao) -> None:
        try:
            self._sucesso(funcao())
        except PedidoInvalido as erro:
            self.close_connection = True
            self._erro(HTTPStatus.BAD_REQUEST, "pedido_invalido", str(erro))
        except preparo.EntradaInvalida as erro:
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
        arquivo = ESTATICOS.get(caminho) or RECURSOS.get(caminho)
        if arquivo is None or not (WEB / arquivo[0]).is_file():
            return self._erro(HTTPStatus.NOT_FOUND, "nao_encontrado", "endereço inexistente")
        self._enviar(HTTPStatus.OK, (WEB / arquivo[0]).read_bytes(), arquivo[1])

    do_HEAD = do_GET

    def _json(self) -> object:
        if self.headers.get_content_type() != "application/json":
            raise PedidoInvalido("o pedido deve ter Content-Type application/json")
        declarado = self.headers.get("Content-Length", "")
        if not (declarado.isascii() and declarado.isdigit()) or len(declarado) > 9:
            raise PedidoInvalido("Content-Length ausente ou inválido")
        if not 0 < int(declarado) <= MAXIMO_JSON:
            raise PedidoInvalido(f"o corpo do pedido deve ter de 1 a {MAXIMO_JSON} bytes")
        return ler_json(self.rfile.read(int(declarado)))

    def do_POST(self) -> None:
        if not self._host_valido():
            return None
        if self.path.split("?", 1)[0] == "/api/avaliar":
            return self._executar(lambda: avaliar(self._json()))
        self._erro(HTTPStatus.NOT_FOUND, "nao_encontrado", "endereço inexistente")

    def log_message(self, formato_: str, *args: object) -> None:  # os contextos consultados nao sao registrados
        return


def criar_servidor(host: str = "127.0.0.1", porta: int = PORTA_PADRAO) -> ThreadingHTTPServer:
    """Confere os artefatos do modelo antes de aceitar pedidos; falha aqui impede a subida."""
    modelagem.carregar()
    return ThreadingHTTPServer((host, porta), Pagina)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ferramenta local: Occasion Score pizza + guaraná.")
    parser.add_argument("--host", default="127.0.0.1", help="endereço de escuta (padrão: só esta máquina)")
    parser.add_argument("--porta", type=int, default=PORTA_PADRAO, help=f"porta (padrão: {PORTA_PADRAO})")
    argumentos = parser.parse_args()
    servidor = criar_servidor(argumentos.host, argumentos.porta)
    sys.stdout.write(f"Ferramenta disponível em http://{argumentos.host}:{servidor.server_address[1]}  (Ctrl+C encerra)\n")
    sys.stdout.flush()
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        servidor.server_close()


if __name__ == "__main__":
    main()
