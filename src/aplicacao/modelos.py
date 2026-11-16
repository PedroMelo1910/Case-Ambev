"""Artefatos congelados empacotados (modelo de taxa histórica e regressão logística), papel de cada modelo e verificacao de integridade.

Integridade dos artefatos NAO e reproducao cientifica: aqui se confere o manifesto pelo SHA-256 fixado no codigo, os
arquivos de modelo de taxa histórica e regressão logística e seus metadados pelos hashes do manifesto e as versoes do ambiente. Nao se treina, nao se le
microdado e nao se confere o codigo cientifico nem os demais artefatos (isso e a verificacao completa do manifesto, que
exige o repositorio de pesquisa). O papel de cada modelo vem do manifesto e de metadados da aplicacao, nunca do usuario.
"""

import functools
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from occasion import congelamento as cg
from occasion import variaveis as transformacao
from occasion.variaveis import COLUNAS_LOGISTICA

SHA256_MANIFESTO = "e36ed4c1b508f94badb82992ba111a3ca3e0707181af4a0de618a01230a8a2ee"
NOME_MANIFESTO = "manifesto_congelamento_v1_3.json"
BASE = Path(__file__).resolve().parents[2] / "artifacts" / "modelagem_2008"
MODELOS = ("B0", "L")
SELECIONADO = "B0"
USO_EXPLORATORIO = frozenset({"L"})  # decisao da aplicacao: a logistica so e mostrada como analise exploratoria
_AMBIENTE = ("scikit-learn", "numpy", "scipy", "joblib")
ESCOPO = ("Integridade dos artefatos empacotados (manifesto, modelo de taxa histórica e regressão logística e seus metadados), da transformação que a inferência "
          "executa (configs/variaveis.json e src/occasion/variaveis.py) e das versões do ambiente; não reproduz o "
          "treinamento, não confere o restante do código científico nem os demais artefatos e não usa microdados. "
          "A verificação em cache vale por processo: verificar_integridade sempre reconfere os arquivos.")


class IntegridadeInvalida(RuntimeError):
    """Manifesto, artefato, metadado ou ambiente diferente do que foi congelado. Nunca vira inferencia."""


@dataclass(frozen=True)
class Papel:
    """selecionado: escolhido com 2008; ordena_contextos: distingue contextos; uso_exploratorio: so analise exploratoria."""
    selecionado: bool
    ordena_contextos: bool
    uso_exploratorio: bool


@dataclass(frozen=True)
class Pacote:
    """Estado verificado e compartilhado pelo processo. Mapeamentos somente leitura; `manifesto` devolve uma copia nova
    a cada acesso. Os estimadores do scikit-learn continuam objetos mutaveis (nao ha como congela-los)."""
    base: Path
    manifesto_json: str
    manifesto_sha256: str
    modelos: Mapping
    papeis: Mapping
    artefatos_sha256: Mapping
    transformacao_sha256: Mapping

    @property
    def manifesto(self) -> dict:
        return json.loads(self.manifesto_json)


def _arquivos_da_transformacao() -> dict:
    """Arquivos realmente carregados que definem faixas, fator de renda e colunas, por caminho relativo ao repositorio."""
    return {"configs/variaveis.json": Path(transformacao.CONFIG), "src/occasion/variaveis.py": Path(transformacao.__file__)}


def _conferir_transformacao(manifesto: dict) -> dict:
    obtidos = {}
    for relativo, caminho in _arquivos_da_transformacao().items():
        try:
            obtido = hashlib.sha256(caminho.read_bytes()).hexdigest()
        except OSError as erro:
            raise IntegridadeInvalida(f"transformacao: {relativo} ausente ou ilegivel") from erro
        if obtido != manifesto["codigo"]["sha256"].get(relativo):
            raise IntegridadeInvalida(f"transformacao: {relativo} diferente do registrado no manifesto")
        obtidos[relativo] = obtido
    return obtidos


def _arquivo(base: Path, relativo: str) -> Path:
    posix = PurePosixPath(str(relativo))
    if not relativo or posix.is_absolute() or ".." in posix.parts or "\\" in relativo:
        raise IntegridadeInvalida(f"caminho invalido no manifesto: {relativo!r}")
    return base.joinpath(*posix.parts)


def _sha256(base: Path, relativo: str, rotulo: str) -> str:
    try:
        return hashlib.sha256(_arquivo(base, relativo).read_bytes()).hexdigest()
    except OSError as erro:
        raise IntegridadeInvalida(f"{rotulo}: arquivo ausente ou ilegivel ({relativo})") from erro


def _ler_manifesto(base: Path) -> tuple[dict, str, str]:
    """Manifesto interpretado, o texto exato que foi hasheado e o SHA-256."""
    try:
        bruto = (base / NOME_MANIFESTO).read_bytes()
    except OSError as erro:
        raise IntegridadeInvalida(f"manifesto ausente ou ilegivel em {NOME_MANIFESTO}") from erro
    obtido = hashlib.sha256(bruto).hexdigest()
    if obtido != SHA256_MANIFESTO:
        raise IntegridadeInvalida(f"manifesto: SHA-256 {obtido[:12]} diferente do fixado {SHA256_MANIFESTO[:12]}")
    try:
        texto = bruto.decode("utf-8")
        manifesto = json.loads(texto)
    except ValueError as erro:
        raise IntegridadeInvalida(f"manifesto ilegivel: {erro}") from erro
    if manifesto.get("versao_manifesto") != cg.VERSAO_MANIFESTO:
        raise IntegridadeInvalida(f"manifesto: versao {manifesto.get('versao_manifesto')!r} nao suportada")
    return manifesto, texto, obtido


def _conferir_ambiente(manifesto: dict) -> None:
    atual, esperado = cg.versoes_do_ambiente(), manifesto["ambiente"]
    problemas = [f"ambiente: {chave} {esperado.get(chave)} no manifesto, {atual[chave]} instalado"
                 for chave in _AMBIENTE if atual[chave] != esperado.get(chave)]
    if ".".join(atual["python"].split(".")[:2]) != ".".join(str(esperado.get("python")).split(".")[:2]):
        problemas.append(f"ambiente: python {esperado.get('python')} no manifesto, {atual['python']} instalado")
    if problemas:
        raise IntegridadeInvalida("; ".join(problemas))


def _carregar_modelo(manifesto: dict, nome: str, base: Path):
    try:
        artefato = manifesto["artefatos"][nome]
    except KeyError as erro:
        raise IntegridadeInvalida(f"artefato {nome} ausente do manifesto") from erro
    if _sha256(base, artefato["meta"], f"artefato {nome}") != artefato["meta_sha256"]:
        raise IntegridadeInvalida(f"artefato {nome}: metadados diferentes do registrado no manifesto")
    if _sha256(base, artefato["arquivo"], f"artefato {nome}") != artefato["sha256"]:
        raise IntegridadeInvalida(f"artefato {nome}: SHA-256 diferente do registrado no manifesto")
    try:
        return cg.carregar_modelo_congelado(manifesto, nome, base)
    except cg.CongelamentoInvalido as erro:
        raise IntegridadeInvalida(f"artefato {nome}: {erro}") from erro
    except OSError as erro:  # so o tipo: a mensagem do sistema traria o caminho local
        raise IntegridadeInvalida(f"artefato {nome}: arquivo ilegivel ({type(erro).__name__})") from erro


def _papeis(manifesto: dict) -> dict:
    if manifesto["selecao"]["selecionado"] != SELECIONADO:
        raise IntegridadeInvalida(f"o modelo selecionado no manifesto nao e {SELECIONADO}: a aplicacao nao tem regras para ele")
    papeis = {}
    for nome in MODELOS:
        ordena = manifesto["modelos"][nome]["ordena_contextos"]
        if not isinstance(ordena, bool):
            raise IntegridadeInvalida(f"manifesto: ordena_contextos de {nome} nao e booleano")
        selecionado = nome == manifesto["selecao"]["selecionado"]
        papeis[nome] = Papel(selecionado=selecionado, ordena_contextos=ordena,
                             uso_exploratorio=nome in USO_EXPLORATORIO and not selecionado)
    if list(manifesto["modelos"]["L"]["entradas"]["colunas"]) != list(COLUNAS_LOGISTICA):
        raise IntegridadeInvalida("colunas da logistica no manifesto diferem das da transformacao da aplicacao")
    return papeis


def _somente_leitura(modelo):
    return MappingProxyType(modelo) if isinstance(modelo, dict) else modelo


def _verificar_e_carregar(base: Path) -> Pacote:
    manifesto, texto, sha = _ler_manifesto(base)
    _conferir_ambiente(manifesto)
    transformacoes = _conferir_transformacao(manifesto)
    carregados = {nome: _somente_leitura(_carregar_modelo(manifesto, nome, base)) for nome in MODELOS}
    return Pacote(base=base, manifesto_json=texto, manifesto_sha256=sha, modelos=MappingProxyType(carregados),
                  papeis=MappingProxyType(_papeis(manifesto)),
                  artefatos_sha256=MappingProxyType({nome: manifesto["artefatos"][nome]["sha256"] for nome in MODELOS}),
                  transformacao_sha256=MappingProxyType(transformacoes))


def _resolver_base(base: Path | str | None) -> Path:
    try:
        return Path(BASE if base is None else base).resolve()
    except (ValueError, OSError) as erro:
        raise IntegridadeInvalida("base dos artefatos invalida") from erro


@functools.cache
def _carregar(base: Path) -> Pacote:
    return _verificar_e_carregar(base)


def carregar(base: Path | str | None = None) -> Pacote:
    """Pacote verificado a partir de `base` (padrao: artifacts/modelagem_2008 da entrega). A verificacao e feita uma vez
    por processo e fica em cache; para reconferir os arquivos, use `verificar_integridade`."""
    return _carregar(_resolver_base(base))


def verificar_integridade(base: Path | str | None = None) -> dict:
    """Reconfere AGORA (sem cache) manifesto, artefatos, transformacao e ambiente; levanta IntegridadeInvalida se algo
    diverge. Nao e reproducao cientifica."""
    pacote = _verificar_e_carregar(_resolver_base(base))
    return {"ok": True, "manifesto_sha256": pacote.manifesto_sha256, "artefatos": dict(pacote.artefatos_sha256),
            "transformacao": dict(pacote.transformacao_sha256), "ambiente": dict(cg.versoes_do_ambiente()), "escopo": ESCOPO}
