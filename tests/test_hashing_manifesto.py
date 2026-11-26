import json
import os
import re
from pathlib import Path

import pytest

from coleta import HashDivergente, sha256_arquivo, verificar_sha256
from coleta import ManifestoInvalido, carregar_manifesto

FRASE = "licença específica dos microdados não confirmada; aceite de risco residual em 07/11/2026"
SHA_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_sha256_de_arquivo_conhecido(tmp_path):
    arq = tmp_path / "a.bin"
    arq.write_bytes(b"abc")
    assert sha256_arquivo(arq) == SHA_ABC


def test_hash_divergente_interrompe(tmp_path):
    arq = tmp_path / "a.bin"
    arq.write_bytes(b"abd")
    with pytest.raises(HashDivergente):
        verificar_sha256(arq, SHA_ABC)


def test_tamanho_divergente_interrompe(tmp_path):
    arq = tmp_path / "a.bin"
    arq.write_bytes(b"abc")
    with pytest.raises(HashDivergente):
        verificar_sha256(arq, SHA_ABC, tamanho_bytes=4)


def test_manifesto_do_repositorio_e_completo(repo):
    m = carregar_manifesto(repo / "configs" / "manifesto_fontes.json")
    assert len(m.fontes) == 4  # so as fontes que o pipeline usa
    assert m.licenca_nota == FRASE
    for f in m.fontes:
        assert re.fullmatch(r"[0-9a-f]{64}", f.sha256)
        assert f.url.startswith("https://") and "ibge.gov.br" in f.url
        assert not os.path.isabs(f.caminho_relativo)
        assert f.edicao in {"2008-2009", "2017-2018"}
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", f.baixado_em)
        assert "IBGE" in f.citacao and f.url in f.citacao
    assert {f.edicao for f in m.fontes} == {"2008-2009", "2017-2018"}
    assert {f.id for f in m.fontes} == {"Dados_20231009.zip", "Documentacao_20231009.zip", "Dados_20230713.zip",
                                       "Documentacao_20230713.zip"}


def test_manifesto_membros_extraidos_tem_hash(repo):
    m = carregar_manifesto(repo / "configs" / "manifesto_fontes.json")
    nomes = {mb.nome for f in m.fontes for mb in f.membros}
    assert {"CONSUMO_ALIMENTAR.txt", "MORADOR.txt", "T_CONSUMO_S.txt", "T_MORADOR_S.txt", "T_DOMICILIO_S.txt"} <= nomes
    for f in m.fontes:
        for mb in f.membros:
            assert re.fullmatch(r"[0-9a-f]{64}", mb.sha256)
            assert not os.path.isabs(mb.destino_relativo)


def test_manifesto_coincide_com_o_da_auditoria(repo):
    """Conferencia opcional contra o manifesto da auditoria de downloads (fora do repositorio)."""
    caminho = os.environ.get("CASE_AMBEV_AUDITORIA_MANIFESTO")
    auditoria = Path(caminho) if caminho else None
    if auditoria is None or not auditoria.exists():
        pytest.skip("manifesto da auditoria indisponivel (defina CASE_AMBEV_AUDITORIA_MANIFESTO)")
    ref = {e["url"]: (e["sha256"], e["tamanho_bytes"]) for e in json.loads(auditoria.read_text(encoding="utf-8"))}
    m = carregar_manifesto(repo / "configs" / "manifesto_fontes.json")
    assert {f.url: (f.sha256, f.tamanho_bytes) for f in m.fontes}.items() <= ref.items()  # subconjunto do manifesto da auditoria


def test_manifesto_rejeita_caminho_absoluto(tmp_path):
    ruim = tmp_path / "m.json"
    ruim.write_text(json.dumps({"licenca_nota": FRASE, "fontes": [{
        "id": "x", "edicao": "2008-2009", "url": "https://ftp.ibge.gov.br/x.zip", "caminho_relativo": "/abs/x.zip",
        "tamanho_bytes": 1, "sha256": "0" * 64, "baixado_em": "2026-10-30", "membros": [],
        "citacao": "IBGE https://ftp.ibge.gov.br/x.zip", "last_modified_remoto": None}]}), encoding="utf-8")
    with pytest.raises(ManifestoInvalido):
        carregar_manifesto(ruim)
