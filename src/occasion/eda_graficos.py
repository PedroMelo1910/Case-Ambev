"""Figuras SVG da EDA (biblioteca padrao: o ambiente do projeto nao tem matplotlib).

Cada linha da figura guarda `data-nivel`, `data-n` e `data-k`, lidos de volta por `conferir_figura` para provar que
os totais do grafico sao os da tabela de origem.
"""

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

LARGURA, ALTURA_LINHA, MARGEM_ESQ, MARGEM_DIR, TOPO = 760, 26, 250, 30, 54
COR_PONTO, COR_IC, COR_REF, COR_BAIXO = "#1f5f8b", "#7a9db8", "#b5472f", "#9a9a9a"


def _x(valor: float, maximo: float) -> float:
    return MARGEM_ESQ + (LARGURA - MARGEM_ESQ - MARGEM_DIR) * valor / maximo


def _linha(i: int, l: dict, maximo: float) -> str:
    y = TOPO + i * ALTURA_LINHA + ALTURA_LINHA / 2
    zero = l["n"] > 0 and l["k"] == 0  # zero observado nao e probabilidade zero: sem ponto nem barra
    todos = l["n"] > 0 and l["k"] == l["n"]
    nota = ", zero observado" if zero else ", so positivos" if todos else ""
    rotulo = escape(f"{l['nivel']}  (n={l['n']}, k={l['k']}{nota})")
    partes = [f'<g class="linha" data-nivel={quoteattr(str(l["nivel"]))} data-n="{l["n"]}" data-k="{l["k"]}">',
              f'<text x="{MARGEM_ESQ - 8}" y="{y + 4}" text-anchor="end" font-size="12">{rotulo}</text>']
    if l["taxa"] is not None and not zero:
        cor = COR_PONTO if l["suporte"] == "ok" else COR_BAIXO
        if l.get("boot_inf") is not None:  # bootstrap degenerado (k = 0 ou k = n) fica sem barra
            partes.append(f'<line x1="{_x(l["boot_inf"], maximo):.1f}" x2="{_x(l["boot_sup"], maximo):.1f}" y1="{y}" y2="{y}" '
                          f'stroke="{COR_IC}" stroke-width="3"/>')
        partes.append(f'<circle cx="{_x(l["taxa"], maximo):.1f}" cy="{y}" r="5" fill="{cor}"/>')
    partes.append("</g>")
    return "".join(partes)


def figura_taxas(titulo: str, linhas: Sequence[dict], caminho: Path, referencia: float | None = None,
                 subtitulo: str = "Ponto = taxa observada; barra = IC 95% por bootstrap de UPAs (sem barra se o bootstrap e degenerado); cinza = suporte baixo") -> None:
    """Grafico de pontos com intervalo, um nivel por linha, com n e k no rotulo. `referencia` desenha a taxa global."""
    maximo = max([l["boot_sup"] for l in linhas if l.get("boot_sup") is not None] + [referencia or 0, 0.05]) * 1.05
    altura = TOPO + len(linhas) * ALTURA_LINHA + 36
    corpo = [_linha(i, l, maximo) for i, l in enumerate(linhas)]
    eixo = [f'<text x="{_x(t, maximo):.1f}" y="{altura - 12}" text-anchor="middle" font-size="11">{t:.0%}</text>'
            for t in [i / 20 for i in range(int(maximo * 20) + 1)]]
    ref = ""
    if referencia is not None:
        xr = _x(referencia, maximo)
        ref = (f'<line x1="{xr:.1f}" x2="{xr:.1f}" y1="{TOPO}" y2="{altura - 28}" stroke="{COR_REF}" stroke-dasharray="4 3"/>'
               f'<text x="{xr:.1f}" y="{TOPO - 6}" text-anchor="middle" font-size="11" fill="{COR_REF}">global {referencia:.1%}</text>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{LARGURA}" height="{altura}" font-family="sans-serif">'
           f'<rect width="100%" height="100%" fill="#ffffff"/><text x="12" y="20" font-size="15" font-weight="bold">{escape(titulo)}</text>'
           f'<text x="12" y="38" font-size="11" fill="#555">{escape(subtitulo)}</text>{ref}{"".join(corpo)}{"".join(eixo)}</svg>')
    Path(caminho).write_text(svg, encoding="utf-8")


def conferir_figura(caminho: Path, linhas: Sequence[dict]) -> dict:
    """Compara (nivel, n, k) de cada linha da figura com a tabela de origem e os totais de n e k."""
    raiz = ET.parse(caminho).getroot()
    figura = [(g.get("data-nivel"), int(g.get("data-n")), int(g.get("data-k"))) for g in raiz.iter("{http://www.w3.org/2000/svg}g")
              if g.get("class") == "linha"]
    tabela = [(str(l["nivel"]), l["n"], l["k"]) for l in linhas]
    divergencias = [f"{a} != {b}" for a, b in zip(figura, tabela) if a != b]
    if len(figura) != len(tabela):
        divergencias.append(f"{len(figura)} linhas na figura, {len(tabela)} na tabela")
    return {"ok": not divergencias, "divergencias": divergencias, "n_figura": sum(n for _, n, _ in figura),
            "n_tabela": sum(n for _, n, _ in tabela), "k_figura": sum(k for _, _, k in figura), "k_tabela": sum(k for _, _, k in tabela)}
