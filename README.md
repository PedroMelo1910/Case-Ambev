# Case Ambev: Occasion Intelligence (pizza e guaraná)

Ferramenta local para a equipe de marketing comparar ocasiões de consumo. Para um dia, um período ou uma hora, uma região, um tipo de área e, se quiser, um público, ela estima a chance de uma refeição incluir pizza e guaraná juntos e compara essa chance com a média das refeições. O resultado indica se vale testar um combo naquele contexto.

A estimativa vem do consumo registrado na Pesquisa de Orçamentos Familiares (POF) do IBGE, edições 2017–2018 e 2008–2009. A POF não mede pedidos de delivery, vendas, a marca Guaraná Antarctica nem o efeito de uma campanha. A recomendação de testar um combo é uma decisão de marketing apoiada pela estimativa e precisa ser confirmada em um teste com grupo de controle.

## Como abrir a aplicação

Requisito: [uv](https://docs.astral.sh/uv/), que instala o Python 3.12 e as dependências fixadas em `uv.lock`. O modelo já está em `artifacts/ocasiao_observada_v1/`, então não é preciso baixar os microdados. A partir da raiz do projeto, no PowerShell, no Prompt de Comando ou em um terminal Linux/macOS:

```bash
uv sync
cd src
uv run python -m aplicacao.servidor
```

Abra `http://127.0.0.1:8000` no navegador. `Ctrl+C` encerra o servidor.

Testes, a partir da raiz: `uv run pytest`.

## Documentação

[`docs/documentacao.md`](docs/documentacao.md) explica o que o modelo prevê, de onde vem a evidência (desenvolvimento e teste reservado na POF 2017–2018, avaliação temporal com a POF 2008–2009), as limitações e as fontes que foram testadas e ficaram fora do modelo. A seção "Como executar" traz os comandos para refazer a coleta e o preparo com os microdados.

## Dados e licença

Fonte: IBGE, Pesquisa de Orçamentos Familiares 2008–2009 e 2017–2018, microdados. As URLs, as datas de download e os SHA-256 estão em [`configs/manifesto_fontes.json`](configs/manifesto_fontes.json). Licença específica dos microdados não confirmada; aceite de risco residual em 07/11/2026. Microdados e derivados linha a linha não fazem parte do repositório.

Projeto acadêmico do Inteli, sem vínculo oficial com a Ambev.
