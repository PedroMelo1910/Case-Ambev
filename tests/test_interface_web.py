"""Contrato estatico da pagina de Ocasioes (src/aplicacao/web): estrutura, recursos locais e entrada segura no DOM.

O comportamento na tela (valores, foco, aviso de desatualizado) e conferido no navegador pelas cenas da rodada;
aqui ficam as garantias que nao dependem de navegador.
"""

import re
from html.parser import HTMLParser

import pytest

from aplicacao import servidor


class _Coletor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.urls, self.botoes, self.legendas, self.nomes = set(), [], [], [], set()
        self._em_botao = self._em_legenda = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.add(a["id"])
        if "name" in a:
            self.nomes.add(a["name"])
        self.urls += [a[k] for k in ("src", "href", "action") if a.get(k)]
        if tag == "button":
            self._em_botao = {"type": a.get("type", "submit"), "texto": ""}
        if tag == "legend":
            self._em_legenda = ""

    def handle_data(self, data):
        if self._em_botao is not None:
            self._em_botao["texto"] += data
        if self._em_legenda is not None:
            self._em_legenda += data

    def handle_endtag(self, tag):
        if tag == "button" and self._em_botao is not None:
            self.botoes.append({**self._em_botao, "texto": self._em_botao["texto"].strip()})
            self._em_botao = None
        if tag == "legend" and self._em_legenda is not None:
            self.legendas.append(" ".join(self._em_legenda.split()))
            self._em_legenda = None


@pytest.fixture(scope="module")
def web(repo):
    pasta = repo / "src" / "aplicacao" / "web"
    html = (pasta / "index.html").read_text(encoding="utf-8")
    coletor = _Coletor()
    coletor.feed(html)
    return {"html": html, "js": (pasta / "app.js").read_text(encoding="utf-8"),
            "css": (pasta / "estilo.css").read_text(encoding="utf-8"), "dom": coletor}


def test_todo_id_usado_pelo_script_existe_na_pagina(web):
    usados = set(re.findall(r'\$\("([\w-]+)"\)', web["js"]))
    assert usados, "o script deveria localizar elementos por id"
    assert usados <= web["dom"].ids, sorted(usados - web["dom"].ids)


def test_pagina_so_usa_recursos_servidos_pelo_proprio_servidor(web):
    servidos = set(servidor.ESTATICOS) | set(servidor.RECURSOS)
    for url in web["dom"].urls:
        assert url.startswith("#") or url in servidos, url
    for arquivo in ("html", "css", "js"):
        texto = web[arquivo]
        assert "runable" not in texto.lower() and "googleapis" not in texto and "gstatic" not in texto
        assert not re.search(r"https?://(?!www\.w3\.org/2000/svg)", texto), arquivo


def test_script_nao_injeta_html_nem_executa_texto(web):
    for proibido in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert proibido not in web["js"], proibido


def test_script_so_conversa_com_as_duas_rotas_da_api(web):
    rotas = set(re.findall(r'"(/api/[\w/]*)"', web["js"]))
    assert rotas == {"/api/opcoes", "/api/avaliar"}
    assert web["js"].count("fetch(") == 1  # uma unica funcao de acesso, com tratamento de erro


def test_formulario_tem_os_grupos_quando_onde_para_quem_e_um_unico_botao_primario(web):
    dom = web["dom"]
    assert "Quando pizza e guaraná aparecem juntos?" in web["html"]
    assert {"Quando", "Onde", "Para quem"} <= set(dom.legendas)
    envios = [b for b in dom.botoes if b["type"] == "submit"]
    assert [b["texto"] for b in envios] == ["Avaliar ocasião"]


def test_escolha_entre_periodo_e_hora_e_explicita(web):
    dom = web["dom"]
    assert "momento" in dom.nomes  # um grupo unico decide se vai periodo ou hora no pedido
    assert {"periodo", "hora"} <= dom.ids


def test_formulario_nao_tem_campos_que_o_modelo_nao_usa(web):
    nomes = " ".join(sorted(web["dom"].nomes | web["dom"].ids)).lower()
    for termo in ("canal", "marca", "desconto", "clima", "chuva", "feriado", "classe"):
        assert termo not in nomes, termo


def test_resultado_tem_a_ordem_chance_media_fatores_acao_e_os_detalhes(web):
    html = web["html"]
    posicoes = [html.index(f'id="{i}"') for i in ("num-estimativa", "comparacao", "fatores", "categoria")]
    assert posicoes == sorted(posicoes)
    assert "Como o modelo calculou a estimativa" in html
    assert {"estado-vazio", "falha", "desatualizado", "resumo-erros"} <= web["dom"].ids


def test_movimento_reduzido_e_foco_visivel_tem_regra_no_estilo(web):
    assert "prefers-reduced-motion" in web["css"] and ":focus-visible" in web["css"]


def test_servidor_so_publica_recursos_que_a_pagina_usa_e_cada_um_tem_licenca(web, repo):
    """Retirada da lata 3D do topo: a imagem e a licenca dela nao ficam servidas nem empacotadas sem uso."""
    pagina = web["html"] + web["css"] + web["js"]
    for rota in servidor.RECURSOS:
        assert rota in pagina, f"{rota} servido sem uso na página"
    pasta = repo / "src" / "aplicacao" / "web" / "recursos"
    midias = {p.name for p in pasta.iterdir() if p.suffix != ".md"}
    assert {rota.rsplit("/", 1)[1] for rota in servidor.RECURSOS} == midias
    licencas = {p.name for p in pasta.glob("*.md")}
    assert all(any(lic.startswith(m.split(".")[0]) for lic in licencas) for m in midias)
    assert all(any(m.startswith(lic.split("-")[0]) for m in midias) for lic in licencas), "licença sem recurso"
