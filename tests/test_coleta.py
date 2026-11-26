import hashlib
import zipfile

import pytest

from coleta import baixar_url, coletar, extrair_membros
from coleta import HashDivergente, sha256_arquivo
from coleta import Fonte, Manifesto, Membro

FRASE = "licença específica dos microdados não confirmada; aceite de risco residual em 07/11/2026"


def _fonte(tmp_path, nome="dados.zip", conteudo=b"conteudo", sha=None, membros=()):
    origem = tmp_path / "origem" / nome
    origem.parent.mkdir(exist_ok=True)
    origem.write_bytes(conteudo)
    return Fonte(id=nome, edicao="2008-2009", url=origem.as_uri(), caminho_relativo=f"pof/downloads/{nome}",
                 tamanho_bytes=len(conteudo), sha256=sha or hashlib.sha256(conteudo).hexdigest(),
                 baixado_em="2026-10-30", last_modified_remoto=None, membros=tuple(membros),
                 citacao=f"IBGE {origem.as_uri()}")


def _manifesto(*fontes):
    return Manifesto(licenca_nota=FRASE, fontes=tuple(fontes))


def test_baixa_quando_ausente_e_verifica_hash(tmp_path):
    raiz = tmp_path / "raw"
    res = coletar(_manifesto(_fonte(tmp_path)), raiz)
    assert [r.status for r in res] == ["baixado"]
    assert (raiz / "pof/downloads/dados.zip").read_bytes() == b"conteudo"


def test_segunda_execucao_e_idempotente_e_nao_baixa(tmp_path):
    raiz = tmp_path / "raw"
    m = _manifesto(_fonte(tmp_path))
    coletar(m, raiz)
    chamadas = []
    res = coletar(m, raiz, baixar=lambda *a, **k: chamadas.append(a))
    assert [r.status for r in res] == ["verificado"] and chamadas == []


def test_arquivo_existente_com_hash_divergente_interrompe_sem_sobrescrever(tmp_path):
    raiz = tmp_path / "raw"
    f = _fonte(tmp_path)
    destino = raiz / f.caminho_relativo
    destino.parent.mkdir(parents=True)
    destino.write_bytes(b"corrompido")
    with pytest.raises(HashDivergente):
        coletar(_manifesto(f), raiz)
    assert destino.read_bytes() == b"corrompido"


def test_download_com_hash_errado_nao_deixa_arquivo(tmp_path):
    raiz = tmp_path / "raw"
    f = _fonte(tmp_path, sha="0" * 64)
    with pytest.raises(HashDivergente):
        coletar(_manifesto(f), raiz)
    assert not list(raiz.rglob("dados.zip*"))


def test_interrompe_no_primeiro_hash_ruim_sem_processar_seguintes(tmp_path):
    raiz = tmp_path / "raw"
    ruim = _fonte(tmp_path, "a.zip", sha="0" * 64)
    bom = _fonte(tmp_path, "b.zip")
    with pytest.raises(HashDivergente):
        coletar(_manifesto(ruim, bom), raiz)
    assert not (raiz / "pof/downloads/b.zip").exists()


def _zip_com(tmp_path, membros: dict[str, bytes]):
    caminho = tmp_path / "origem" / "pacote.zip"
    caminho.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(caminho, "w") as z:
        for nome, dados in membros.items():
            z.writestr(nome, dados)
    return caminho


def _fonte_zip(tmp_path, membros_zip, pedidos):
    caminho = _zip_com(tmp_path, membros_zip)
    membros = [Membro(nome=n, destino_relativo=f"pof/extraido/{n}", sha256=hashlib.sha256(membros_zip[n]).hexdigest(),
                      tamanho_bytes=len(membros_zip[n])) for n in pedidos]
    return Fonte(id="pacote.zip", edicao="2008-2009", url=caminho.as_uri(), caminho_relativo="pof/downloads/pacote.zip",
                 tamanho_bytes=caminho.stat().st_size, sha256=sha256_arquivo(caminho), baixado_em="2026-10-30",
                 last_modified_remoto=None, membros=tuple(membros), citacao=f"IBGE {caminho.as_uri()}")


def test_extrai_somente_membros_do_manifesto_e_confere_hash(tmp_path):
    raiz = tmp_path / "raw"
    f = _fonte_zip(tmp_path, {"A.txt": b"aaa", "B.txt": b"bbb"}, ["A.txt"])
    coletar(_manifesto(f), raiz)
    saidas = extrair_membros(f, raiz)
    assert [p.name for p in saidas] == ["A.txt"] and saidas[0].read_bytes() == b"aaa"
    assert not (raiz / "pof/extraido/B.txt").exists()
    assert extrair_membros(f, raiz) == saidas  # idempotente


def test_extracao_detecta_hash_de_membro_divergente(tmp_path):
    raiz = tmp_path / "raw"
    f = _fonte_zip(tmp_path, {"A.txt": b"aaa"}, ["A.txt"])
    f = Fonte(**{**f.__dict__, "membros": (Membro("A.txt", "pof/extraido/A.txt", "0" * 64, 3),)})
    coletar(_manifesto(f), raiz)
    with pytest.raises(HashDivergente):
        extrair_membros(f, raiz)


def test_extracao_recusa_caminho_inseguro(tmp_path):
    raiz = tmp_path / "raw"
    f = _fonte_zip(tmp_path, {"../fora.txt": b"x"}, ["../fora.txt"])
    coletar(_manifesto(f), raiz)
    with pytest.raises(ValueError):
        extrair_membros(f, raiz)


def test_baixar_url_aceita_file_uri(tmp_path):
    origem = tmp_path / "x.bin"
    origem.write_bytes(b"123")
    destino = tmp_path / "d" / "x.bin"
    baixar_url(origem.as_uri(), destino)
    assert destino.read_bytes() == b"123"


def test_extrai_membro_com_acento_para_destino_ascii(tmp_path):
    raiz = tmp_path / "raw"
    nome = "Dicionários de váriaveis.xls"
    caminho = tmp_path / "origem" / "doc.zip"
    caminho.parent.mkdir()
    with zipfile.ZipFile(caminho, "w") as z:
        z.writestr(zipfile.ZipInfo(nome), b"dic")
        z.filelist[0].flag_bits |= 0x800  # nome marcado como UTF-8
    membro = Membro(nome, "pof/extraido/Dicionarios_de_variaveis.xls", hashlib.sha256(b"dic").hexdigest(), 3)
    f = Fonte(id="doc.zip", edicao="2017-2018", url=caminho.as_uri(), caminho_relativo="pof/downloads/doc.zip",
              tamanho_bytes=caminho.stat().st_size, sha256=sha256_arquivo(caminho), baixado_em="2026-10-27",
              last_modified_remoto=None, membros=(membro,), citacao=f"IBGE {caminho.as_uri()}")
    coletar(_manifesto(f), raiz)
    assert [p.name for p in extrair_membros(f, raiz)] == ["Dicionarios_de_variaveis.xls"]
