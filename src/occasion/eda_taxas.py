"""Estatistica descritiva da EDA de 2008-2009: taxas por nivel com intervalo, suporte e redundancias.

Funcoes puras, so biblioteca padrao. Dois intervalos de 95%, sempre em colunas separadas e nunca misturados:
  boot_*    percentil do bootstrap de UPAs inteiras (a dependencia que importa e a de domicilio e UPA); ignora o estrato;
  wilson_*  intervalo de Wilson, que supoe episodios independentes.
Nenhum dos dois e limite garantido do outro. Quando todos os episodios do nivel tem o mesmo resultado (k = 0 ou k = n), toda
replica do bootstrap repete esse valor: o bootstrap fica degenerado, nao estima variancia e seu intervalo nao e publicado.
Nenhuma funcao seleciona variaveis por associacao com o alvo: isso pertence a modelagem.
"""

import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

MIN_N, MIN_POSITIVOS, CELULA_MINIMA = 30, 5, 5


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de Wilson (episodios independentes). Para n = 0 devolve (0, 1)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    meia = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centro - meia), min(1.0, centro + meia)


def rotulo_contagem(n: int):
    """Mascara contagens de 1 a 4 (celulas que poderiam identificar poucas pessoas); zero e informativo e aparece."""
    return "<5" if 0 < n < CELULA_MINIMA else n


def suporte(n: int, k: int) -> str:
    """`ok` so com n >= 30 e pelo menos 5 positivos; abaixo disso a leitura e apenas descritiva."""
    return "ok" if n >= MIN_N and k >= MIN_POSITIVOS else "baixo"


def quantil(valores: Sequence[float], p: float) -> float:
    """Quantil com interpolacao linear entre posicoes ordenadas (posicao p * (n - 1), a definicao usual 'tipo 7')."""
    if not valores:
        raise ValueError("quantil de lista vazia")
    ordenados = sorted(valores)
    posicao = p * (len(ordenados) - 1)
    abaixo = math.floor(posicao)
    acima = min(abaixo + 1, len(ordenados) - 1)
    return ordenados[abaixo] + (ordenados[acima] - ordenados[abaixo]) * (posicao - abaixo)


def _por_cluster(obs: Sequence[Mapping], niveis: Sequence) -> dict:
    """cluster -> nivel -> [n, k, soma_w, soma_wy]."""
    tabela: dict = defaultdict(lambda: {n: [0, 0, 0.0, 0.0] for n in niveis})
    for o in obs:
        if o["nivel"] not in niveis:
            continue
        cel = tabela[o["cluster"]][o["nivel"]]
        w = o.get("w", 1.0)
        cel[0] += 1
        cel[1] += o["y"]
        cel[2] += w
        cel[3] += w * o["y"]
    return tabela


def _replicas(tabela: dict, niveis: Sequence, replicas: int, semente: int, ponderar: bool) -> dict:
    clusters = list(tabela)
    rng = random.Random(semente)
    esparso = {nivel: [(i, *tabela[c][nivel]) for i, c in enumerate(clusters) if tabela[c][nivel][0]] for nivel in niveis}
    saida = {n: [] for n in niveis}
    for _ in range(replicas):
        sorteio = Counter(rng.choices(range(len(clusters)), k=len(clusters)))
        for nivel in niveis:
            n = k = sw = swy = 0.0
            for i, cn, ck, csw, cswy in esparso[nivel]:
                m = sorteio.get(i, 0)
                if m:
                    n, k, sw, swy = n + m * cn, k + m * ck, sw + m * csw, swy + m * cswy
            if n > 0:
                saida[nivel].append((swy / sw) if ponderar else (k / n))
    return saida


def taxas_por_nivel(obs: Sequence[Mapping], niveis: Sequence, replicas: int = 1000, semente: int = 20261113,
                    ponderar: bool = False) -> list[dict]:
    """Uma linha por nivel: n, k, taxa sem peso e os dois intervalos em colunas proprias (`boot_*` e `wilson_*`).

    `boot_replicas` conta as replicas em que o nivel apareceu; `boot_descartadas` conta as demais (o nivel nao foi sorteado,
    taxa indefinida: o intervalo e condicional a presenca do nivel). `boot_degenerado` e verdadeiro quando k = 0 ou k = n; nesse caso `boot_inf` e `boot_sup` ficam
    vazios. Com `ponderar`, entram `w1_taxa`, `w1_boot_inf` e `w1_boot_sup` (mesmas reamostras, mesma regra).
    Niveis sem episodios ficam com n = 0 e taxa None."""
    total = sum(1 for o in obs if o["nivel"] in niveis)
    tabela = _por_cluster(obs, niveis)
    bruto = _replicas(tabela, niveis, replicas, semente, ponderar=False)
    pond = _replicas(tabela, niveis, replicas, semente, ponderar=True) if ponderar else {}
    linhas = []
    for nivel in niveis:
        sel = [o for o in obs if o["nivel"] == nivel]
        n, k = len(sel), sum(o["y"] for o in sel)
        degenerado = n > 0 and k in (0, n)
        tem_boot = n > 0 and not degenerado
        linha = {"nivel": nivel, "n": n, "k": k, "pct_episodios": n / total if total else None, "taxa": k / n if n else None,
                 "boot_inf": quantil(bruto[nivel], 0.025) if tem_boot else None,
                 "boot_sup": quantil(bruto[nivel], 0.975) if tem_boot else None,
                 "boot_replicas": len(bruto[nivel]), "boot_descartadas": replicas - len(bruto[nivel]), "boot_degenerado": degenerado,
                 "wilson_inf": wilson(k, n)[0] if n else None, "wilson_sup": wilson(k, n)[1] if n else None,
                 "suporte": suporte(n, k), "clusters": sum(1 for c in tabela if tabela[c][nivel][0])}
        if ponderar:
            sw, swy = sum(o.get("w", 1.0) for o in sel), sum(o.get("w", 1.0) * o["y"] for o in sel)
            linha["w1_taxa"] = swy / sw if sw else None
            linha["w1_boot_inf"] = quantil(pond[nivel], 0.025) if tem_boot else None
            linha["w1_boot_sup"] = quantil(pond[nivel], 0.975) if tem_boot else None
        linhas.append(linha)
    return linhas


def posto(linhas: Sequence[Sequence[float]], tolerancia: float = 1e-8) -> int:
    """Posto de uma matriz (eliminacao de Gauss com pivo parcial). Serve para separar dependencia linear exata
    (indicadores completos) de mera correlacao alta (uma interacao e produto, nao combinacao linear)."""
    m = [list(map(float, linha)) for linha in linhas]
    achado = 0
    for coluna in range(len(m[0]) if m else 0):
        pivo = max(range(achado, len(m)), key=lambda r: abs(m[r][coluna]), default=None)
        if pivo is None or abs(m[pivo][coluna]) < tolerancia:
            continue
        m[achado], m[pivo] = m[pivo], m[achado]
        base = [v / m[achado][coluna] for v in m[achado]]
        m[achado] = base
        for r in range(len(m)):
            if r != achado and abs(m[r][coluna]) > tolerancia:
                fator = m[r][coluna]
                m[r] = [a - fator * b for a, b in zip(m[r], base)]
        achado += 1
        if achado == len(m):
            break
    return achado


def diferenca_por_upa(obs: Sequence[Mapping], nivel_a, nivel_b, replicas: int = 1000, semente: int = 20261113) -> dict:
    """Taxa(a) - taxa(b) com intervalo percentil de 95% por bootstrap de UPAs inteiras (sem peso). As duas taxas saem da
    mesma reamostra; replicas em que algum dos niveis some sao descartadas e contadas em `replicas_validas`."""
    tabela = _por_cluster(obs, [nivel_a, nivel_b])
    clusters = list(tabela)
    for nivel in (nivel_a, nivel_b):
        if not any(tabela[c][nivel][0] for c in clusters):
            raise ValueError(f"nivel {nivel!r} sem episodios: contraste indefinido")
    rng = random.Random(semente)

    def taxa(sorteio: Counter, nivel) -> float | None:
        n = sum(m * tabela[clusters[i]][nivel][0] for i, m in sorteio.items())
        return sum(m * tabela[clusters[i]][nivel][1] for i, m in sorteio.items()) / n if n else None

    difs = []
    for _ in range(replicas):
        sorteio = Counter(rng.choices(range(len(clusters)), k=len(clusters)))
        a, b = taxa(sorteio, nivel_a), taxa(sorteio, nivel_b)
        if a is not None and b is not None:
            difs.append(a - b)
    if not difs:
        raise ValueError("nenhuma replica com os dois niveis: contraste sem intervalo")
    todos = Counter(range(len(clusters)))
    return {"diferenca": taxa(todos, nivel_a) - taxa(todos, nivel_b), "boot_inf": quantil(difs, 0.025), "boot_sup": quantil(difs, 0.975),
            "replicas_validas": len(difs), "replicas": replicas,
            "n_a": sum(tabela[c][nivel_a][0] for c in clusters), "n_b": sum(tabela[c][nivel_b][0] for c in clusters)}


def pearson(x: Sequence, y: Sequence) -> float | None:
    """Correlacao de Pearson ignorando pares com valor vazio; None se algum lado for constante."""
    pares = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pares) < 2:
        return None
    ma, mb = sum(a for a, _ in pares) / len(pares), sum(b for _, b in pares) / len(pares)
    sa = sum((a - ma) ** 2 for a, _ in pares)
    sb = sum((b - mb) ** 2 for _, b in pares)
    if sa == 0 or sb == 0:
        return None
    return sum((a - ma) * (b - mb) for a, b in pares) / math.sqrt(sa * sb)


def cramer_v(x: Sequence, y: Sequence) -> float | None:
    """V de Cramer entre duas categoricas (sem correcao de vies; descritivo)."""
    pares = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    n = len(pares)
    linhas, colunas = sorted({a for a, _ in pares}), sorted({b for _, b in pares})
    if n == 0 or min(len(linhas), len(colunas)) < 2:
        return None
    obs = Counter(pares)
    ra, cb = Counter(a for a, _ in pares), Counter(b for _, b in pares)
    qui = sum((obs.get((a, b), 0) - ra[a] * cb[b] / n) ** 2 / (ra[a] * cb[b] / n) for a in linhas for b in colunas)
    return math.sqrt(qui / (n * (min(len(linhas), len(colunas)) - 1)))


def colunas_redundantes(matriz: Mapping[str, Sequence], limiar: float = 0.9) -> list[dict]:
    """Pares de colunas com |correlacao| >= limiar, do maior para o menor."""
    nomes = list(matriz)
    pares = []
    for i, a in enumerate(nomes):
        for b in nomes[i + 1:]:
            r = pearson(matriz[a], matriz[b])
            if r is not None and abs(r) >= limiar:
                pares.append({"a": a, "b": b, "correlacao": r})
    return sorted(pares, key=lambda p: -abs(p["correlacao"]))


def grupos_completos(matriz: Mapping[str, Sequence], grupos: Mapping[str, Sequence[str]]) -> dict[str, bool]:
    """True se, em todos os episodios, as colunas do grupo somam 1 (indicadores completos: colineares com o intercepto)."""
    resultado = {}
    for nome, colunas in grupos.items():
        n = len(matriz[colunas[0]])
        resultado[nome] = all(sum(matriz[c][i] or 0 for c in colunas) == 1 for i in range(n))
    return resultado
