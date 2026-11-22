"""Treino da rede neural da jornada (RUR-Net) na POF 2008-2009, com validação agrupada por UPA.

Uso (no WSL, venv do projeto):
    PYTHONPATH=src .venv/bin/python -m treinar_personas --base <base_ampliada.npz> --artefatos artifacts/personas_rede_v1

Lê a base real de 2008 (fora do repositório). Treina duas cabeças com rótulo observado:
- comida: pizza vs. outra refeição, nas refeições com comida (recorte S1);
- bebida: guaraná / outro refrigerante / outra bebida / nenhuma, nas refeições com pizza (cabeça hierárquica).

Compara a rede com duas referências sem deep learning, nas MESMAS dobras agrupadas por UPA:
- taxa histórica (probabilidade constante = taxa do treino);
- regressão logística (as mesmas features, sem resíduo neural).

Relata o resultado real, inclusive se a rede não superar a referência (log-loss e calibração por dobra). Grava o
artefato (pesos da rede reajustada em todo o treino + padronizador + métricas) e um resumo agregado. Nenhuma linha dos
microdados é gravada. A avaliação de 2017 não é tocada.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from personas import rede

SEMENTE = 20261122
DOBRAS = 5
EPOCAS_COMIDA = 8        # cabeça comida: 254 mil episódios em mini-batches; converge rápido com tanto dado
EPOCAS_BEBIDA = 300      # cabeça bebida: 1.168 episódios em lote cheio; barata, mais épocas
LOTE = 4096
LARGURA = 16
DROPOUT = 0.2
DECAIMENTO = 1e-2
TAXA = 0.01
EPS = 1e-7


# ---------- carga da base ----------

def carregar_base(caminho: Path) -> dict:
    """Lê a base .npz e monta os registros com as features que a rede usa, mais rótulos e grupo (UPA).

    Os arrays são materializados uma vez (indexar o NpzFile dentro do loop redescomprimiria a cada acesso)."""
    npz = np.load(caminho, allow_pickle=True)
    col = {nome: npz[nome] for nome in ("grande_regiao", "area", "faixa_etaria", "sexo", "escolaridade", "periodo",
                                        "dia_semana", "n_todos", "n_menores", "renda_pc_todos")}
    n = len(npz["p"])
    registros = []
    for i in range(n):
        registros.append({
            "grande_regiao": str(col["grande_regiao"][i]), "area": str(col["area"][i]),
            "faixa_etaria": str(col["faixa_etaria"][i]), "sexo": str(col["sexo"][i]),
            "escolaridade": str(col["escolaridade"][i]), "periodo": str(col["periodo"][i]),
            "dia_semana": int(col["dia_semana"][i]),
            "n_moradores": int(col["n_todos"][i]), "n_menores": int(col["n_menores"][i]),
            "renda_pc": float(col["renda_pc_todos"][i]),
        })
    return {
        "registros": registros,
        "comida": np.asarray(npz["comida"], dtype=int), "pizza": np.asarray(npz["p"], dtype=int),
        "guarana": np.asarray(npz["g"], dtype=int), "refrigerante": np.asarray(npz["r"], dtype=int),
        "cola": np.asarray(npz["cola"], dtype=int),
        "upa": np.asarray(npz["upa"]), "peso": np.asarray(npz["peso"], dtype=float),
    }


def rotulo_bebida(base: dict) -> np.ndarray:
    """Índice da folha de bebida nas ocasiões com pizza, com três folhas OBSERVÁVEIS na base:
    0 = guaraná, 1 = outro refrigerante, 2 = sem refrigerante (outra bebida ou nenhuma, agrupadas).

    A POF 2008 não registrou água e a base não distingue uma bebida não-refrigerante de nenhuma bebida. Em vez de uma
    divisão provisória 50/50 entre 'outra bebida' e 'nenhuma', as duas ficam agrupadas em 'sem refrigerante' até haver
    rótulos observados — assim a cabeça só aprende contrastes que os dados sustentam (guaraná × outro refri × resto)."""
    g, r = base["guarana"], base["refrigerante"]
    folha = np.full(len(g), 2, dtype=int)  # sem refrigerante (agrupado) por padrão
    folha[(r == 1) & (g == 0)] = 1  # outro refrigerante
    folha[g == 1] = 0  # guaraná (dentro de refrigerante)
    return folha


# ---------- perdas e métricas ----------

def _log_loss_binaria(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _calibracao_no_todo(y: np.ndarray, p: np.ndarray) -> float:
    """Razão observado/esperado (1,0 = calibrado no agregado)."""
    return float(y.sum() / max(p.sum(), EPS))


# ---------- treino de uma cabeça ----------

def _particoes(y: np.ndarray, grupos: np.ndarray, semente: int):
    from sklearn.model_selection import StratifiedGroupKFold
    return list(StratifiedGroupKFold(DOBRAS, shuffle=True, random_state=semente).split(np.zeros(len(y)), y, grupos))


def _treinar(X, alvos_comida, alvos_bebida, mascara_bebida, semente, epocas_comida, epocas_bebida):
    """Treina a RUR-Net. As cabeças têm parâmetros próprios e otimizadores separados: treinar a bebida NÃO altera
    nenhum peso da comida (pizza). A cabeça comida usa mini-batches (dados grandes, poucas épocas); a cabeça bebida usa
    lote cheio das pizzas (dados pequenos, mais épocas).

    `alvos_comida` pode ser None (não treinar comida, ex.: avaliação só da bebida) e idem `alvos_bebida`. A ordem é
    sempre comida primeiro, depois bebida — a mesma sequência do artefato servido."""
    import torch

    torch.manual_seed(semente)
    r = rede.construir_rede(largura=LARGURA, dropout=DROPOUT, semente=semente)
    perda_ce = torch.nn.CrossEntropyLoss()
    xt = torch.from_numpy(X.astype(np.float32))
    rng = np.random.default_rng(semente)
    r.train()

    if alvos_comida is not None:
        otim_comida = torch.optim.AdamW(r.comida.parameters(), lr=TAXA, weight_decay=DECAIMENTO)
        yc = torch.from_numpy(np.where(alvos_comida == 1, 0, 1).astype(np.int64))  # pizza=0, outra=1
        n = len(xt)
        for _ in range(epocas_comida):
            ordem = rng.permutation(n)
            for ini in range(0, n, LOTE):
                lote = torch.from_numpy(ordem[ini:ini + LOTE])
                otim_comida.zero_grad()
                perda_ce(r.comida(xt[lote]), yc[lote]).backward()
                otim_comida.step()

    if alvos_bebida is not None:
        # congela a cabeça de comida: nenhum gradiente a altera durante o treino da bebida
        for p in r.comida.parameters():
            p.requires_grad_(False)
        otim_bebida = torch.optim.AdamW(r.bebida.parameters(), lr=TAXA, weight_decay=DECAIMENTO)
        mb = np.flatnonzero(mascara_bebida)
        xb = xt[torch.from_numpy(mb)]
        yb = torch.from_numpy(alvos_bebida[mb].astype(np.int64))
        for _ in range(epocas_bebida):
            otim_bebida.zero_grad()
            perda_ce(r.bebida(xb), yb).backward()
            otim_bebida.step()
        for p in r.comida.parameters():
            p.requires_grad_(True)  # restaura para salvar/servir normalmente

    r.eval()
    return r


def prever_rede(r, X: np.ndarray) -> dict:
    import torch

    with torch.no_grad():
        saida = r(torch.from_numpy(X.astype(np.float32)))
        p_pizza = torch.softmax(saida["comida"], dim=1)[:, 0].numpy()  # ação 0 = pizza
        p_bebida = torch.softmax(saida["bebida"], dim=1).numpy()  # 4 folhas
    return {"pizza": p_pizza, "bebida": p_bebida}


# ---------- referências sem deep learning ----------

def prever_logistica(Xtr, ytr, Xte, multinomial: bool = False):
    from sklearn.linear_model import LogisticRegression

    modelo = LogisticRegression(max_iter=1000, C=1.0)
    modelo.fit(Xtr, ytr)
    if multinomial:
        return modelo.predict_proba(Xte)
    return modelo.predict_proba(Xte)[:, 1]


# ---------- avaliação cruzada ----------

def avaliar(base: dict) -> dict:
    """Validação agrupada por UPA nas duas cabeças, comparando rede, logística e taxa histórica por dobra."""
    idx_comida = np.flatnonzero(base["comida"] == 1)
    idx_pizza = np.flatnonzero(base["pizza"] == 1)
    return {"comida": _avaliar_comida(base, idx_comida), "bebida": _avaliar_bebida(base, idx_pizza)}


def _montar_X(base: dict, indices: np.ndarray, pad: rede.Padronizador) -> np.ndarray:
    return rede.matriz_features([base["registros"][i] for i in indices], pad)


def _avaliar_comida(base: dict, idx: np.ndarray) -> dict:
    y = (base["pizza"][idx] == 1).astype(int)  # pizza entre refeições com comida
    grupos = base["upa"][idx]
    perdas = {"rede": [], "logistica": [], "historica": []}
    for k, (tr, te) in enumerate(_particoes(y, grupos, SEMENTE)):
        itr, ite = idx[tr], idx[te]
        pad = rede.Padronizador.ajustar(_numericas(base, itr))
        Xtr, Xte = _montar_X(base, itr, pad), _montar_X(base, ite, pad)
        ytr, yte = y[tr], y[te]
        taxa = float(ytr.mean())
        perdas["historica"].append(_log_loss_binaria(yte, np.full(len(yte), taxa)))
        perdas["logistica"].append(_log_loss_binaria(yte, prever_logistica(Xtr, ytr, Xte)))
        r = _treinar(Xtr, base["pizza"][itr], None, np.zeros(len(itr), bool), SEMENTE,
                     EPOCAS_COMIDA, EPOCAS_BEBIDA)
        perdas["rede"].append(_log_loss_binaria(yte, prever_rede(r, Xte)["pizza"]))
        print(f"  [comida] dobra {k + 1}/{DOBRAS} ok", flush=True)
    return _resumo_perdas(perdas, int(y.sum()), len(y))


def _avaliar_bebida(base: dict, idx: np.ndarray) -> dict:
    """Contraste primário guaraná × não-guaraná (presença de guaraná), entre ocasiões com pizza."""
    y = (base["guarana"][idx] == 1).astype(int)
    grupos = base["upa"][idx]
    folha = rotulo_bebida(base)[idx]
    perdas = {"rede": [], "logistica": [], "historica": []}
    for k, (tr, te) in enumerate(_particoes(y, grupos, SEMENTE)):
        itr, ite = idx[tr], idx[te]
        pad = rede.Padronizador.ajustar(_numericas(base, itr))
        Xtr, Xte = _montar_X(base, itr, pad), _montar_X(base, ite, pad)
        ytr, yte = y[tr], y[te]
        taxa = float(ytr.mean())
        perdas["historica"].append(_log_loss_binaria(yte, np.full(len(yte), taxa)))
        perdas["logistica"].append(_log_loss_binaria(yte, prever_logistica(Xtr, ytr, Xte)))
        r = _treinar(Xtr, None, folha[tr], np.ones(len(itr), bool), SEMENTE, EPOCAS_COMIDA, EPOCAS_BEBIDA)
        p_g = prever_rede(r, Xte)["bebida"][:, 0]  # folha 0 = guaraná
        perdas["rede"].append(_log_loss_binaria(yte, p_g))
        print(f"  [bebida] dobra {k + 1}/{DOBRAS} ok", flush=True)
    return _resumo_perdas(perdas, int(y.sum()), len(y))


def _numericas(base: dict, indices: np.ndarray) -> np.ndarray:
    return np.array([rede.vetor_bruto(base["registros"][i])[1] for i in indices], dtype=np.float32)


def _resumo_perdas(perdas: dict, positivos: int, total: int) -> dict:
    media = {k: float(np.mean(v)) for k, v in perdas.items()}
    return {"log_loss_medio": media, "por_dobra": {k: [float(x) for x in v] for k, v in perdas.items()},
            "positivos": positivos, "episodios": total,
            "rede_vence_historica": media["rede"] < media["historica"],
            "rede_vence_logistica": media["rede"] < media["logistica"]}


# ---------- ajuste final e gravação ----------

def ajustar_final(base: dict):
    """Reajusta a rede em todo o conjunto (comida sobre refeições com comida; bebida sobre pizzas) e grava o artefato."""
    idx_comida = np.flatnonzero(base["comida"] == 1)
    pad = rede.Padronizador.ajustar(_numericas(base, idx_comida))
    # treino conjunto: usamos todas as refeições com comida; a cabeça de bebida só recebe perda nas pizzas
    X = _montar_X(base, idx_comida, pad)
    y_comida = base["pizza"][idx_comida]
    folha = rotulo_bebida(base)[idx_comida]
    mascara_bebida = base["pizza"][idx_comida] == 1
    r = _treinar(X, y_comida, folha, mascara_bebida, SEMENTE, EPOCAS_COMIDA, EPOCAS_BEBIDA)
    return r, pad


def gravar(artefatos: Path, r, pad: rede.Padronizador, avaliacao: dict, base: dict) -> None:
    import torch

    artefatos.mkdir(parents=True, exist_ok=True)
    torch.save(r.state_dict(), artefatos / "rede.pt")
    com_pizza = base["pizza"] == 1
    folha_pizza = rotulo_bebida(base)[com_pizza]
    meta = {
        "versao": rede.VERSAO, "largura": LARGURA,
        "padronizador": {"media": list(pad.media), "desvio": list(pad.desvio)},
        "features": {"categoricas": {k: list(v) for k, v in rede.CATEGORICAS.items()},
                     "binarias": list(rede.BINARIAS), "numericas": list(rede.NUMERICAS)},
        "treino": {"semente": SEMENTE, "dobras": DOBRAS, "epocas_comida": EPOCAS_COMIDA,
                   "epocas_bebida": EPOCAS_BEBIDA, "lote": LOTE, "decaimento": DECAIMENTO, "taxa": TAXA,
                   "torch": torch.__version__},
        "contagens": {"episodios_com_comida": int((base["comida"] == 1).sum()),
                      "episodios_com_pizza": int(com_pizza.sum()),
                      "pizza_com_guarana": int((base["guarana"][com_pizza]).sum())},
        # distribuição histórica observada das 3 folhas de bebida entre as ocasiões com pizza (referência do chat)
        "bebida_historica": {
            "guarana": int((folha_pizza == 0).sum()),
            "outro_refrigerante": int((folha_pizza == 1).sum()),
            "sem_refrigerante": int((folha_pizza == 2).sum()),
        },
    }
    (artefatos / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (artefatos / "avaliacao.json").write_text(json.dumps(avaliacao, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino da rede neural da jornada (POF 2008).")
    parser.add_argument("--base", required=True, type=Path, help="base_ampliada.npz (fora do repositório)")
    parser.add_argument("--artefatos", required=True, type=Path)
    args = parser.parse_args()
    base = carregar_base(args.base)
    print(f"base: {len(base['registros'])} episódios; comida={int((base['comida']==1).sum())}; "
          f"pizza={int((base['pizza']==1).sum())}; guaraná|pizza={int(base['guarana'][base['pizza']==1].sum())}",
          flush=True)
    avaliacao = avaliar(base)
    for no, res in avaliacao.items():
        m = res["log_loss_medio"]
        print(f"[{no}] log-loss  rede={m['rede']:.5f}  logística={m['logistica']:.5f}  histórica={m['historica']:.5f}"
              f"  | rede>hist={res['rede_vence_historica']} rede>log={res['rede_vence_logistica']}", flush=True)
    r, pad = ajustar_final(base)
    gravar(args.artefatos, r, pad, avaliacao, base)
    print(f"artefato gravado em {args.artefatos}", flush=True)


if __name__ == "__main__":
    main()
