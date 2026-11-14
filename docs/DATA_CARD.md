# Data Card — Occasion Intelligence: Pizza + Guaraná

**Desenvolvimento: POF 2008–2009. Avaliação temporal reservada: POF 2017–2018.** O alvo de 2017 fica selado e **não foi aberto** na preparação documentada, nem os pesos. Os resultados descritivos de 2008 estão na [EDA](EDA_2008.md), reproduzível por `python -m occasion.eda_2008`.

**O que estes dados são, em uma frase.** A POF registra consumo alimentar: o que cada pessoa disse ter comido e bebido, quando e onde. Ela **não comprova** pedidos de delivery, a marca Antarctica nem o efeito de uma ação de marketing; por isso nenhuma leitura deste projeto fala de pedido, marca ou causalidade.

## 1. Unidade de análise

O **episódio**: uma pessoa, em um dia de coleta, em uma hora, em que consumiu pizza (`pessoa × dia × hora`). Identificador `episodio_id`, único por edição. O instante de previsão é a refeição com pizza já definida e a bebida ainda não escolhida.

| Contagem em 2008–2009 | Valor |
|---|---|
| Episódios com pizza | 1.168 |
| Pessoas | 1.106 (1.046 com 1 episódio, 58 com 2 e 2 com 3) |
| Unidades de consumo | 715 (na amostra, cada domicílio tem uma unidade de consumo com pizza) |
| Domicílios | 715 |
| UPAs (unidades primárias de amostragem) | 644 |
| Estratos (UF × estrato) | 293, dos quais 144 com uma única UPA |
| UFs | 27 |

A hora é arredondada pelo IBGE e não há data de calendário: só dia da semana e hora.

## 2. Target (ground truth ou proxy)

`y_guarana = 1` quando o **mesmo episódio** contém um dos seis códigos de refrigerante de guaraná da POF (`8200301`, `8200302`, `8201601`, `8201602`, `8207401`, `8207402`); caso contrário, 0. O tipo de ativação nunca é o alvo.

**É registro autodeclarado de consumo.** Pedido, canal e marca não são observados; o alvo não permite identificá-los. A classificação da POF não tem código de marca de guaraná. Parte do refrigerante é "não especificado" (132 dos 655 episódios com refrigerante), então a taxa identificada pode subestimar o guaraná consumido.

| Medida (2008–2009) | Numerador / denominador | Taxa |
|---|---|---|
| Guaraná entre episódios com pizza | 101 / 1.168 | 8,65% |
| Algum refrigerante entre episódios com pizza | 655 / 1.168 | 56,1% |
| Guaraná entre episódios com refrigerante | 101 / 655 | 15,4% |

**Interpretação do score:** frequência histórica estimada de guaraná em episódios com pizza de contextos parecidos, na amostra de 2008–2009. Não é probabilidade de pedir, preferência pela marca Antarctica, chance de compra nem efeito de promoção.

## 3. Dicionário de variáveis

A coluna "Disponível na aplicação" distingue entradas que **o usuário pode informar** (`sim`), informações indisponíveis no instante da consulta (`não`) e dados somente de diagnóstico. O sistema não observa automaticamente renda ou composição familiar. A especificação inicial tem 13 variáveis derivadas e 28 colunas, definidas em [variaveis.json](../configs/variaveis.json). `log_renda_pc` usa reais de jan/2009; a conversão da renda atual ainda precisa ser implementada na inferência.

| Variável | Descrição e derivação | Origem 2008–09 | Origem 2017–18 | Papel | Disponível na aplicação | Limitação |
|---|---|---|---|---|---|---|
| `hora_manha` | faixa de hora 5–10 h | `V9015` | `V9015` | treino, B1, boosting | sim | hora arredondada |
| `hora_almoco` | faixa de hora 11–14 h | `V9015` | `V9015` | treino, B1, boosting | sim | hora arredondada |
| `hora_tarde` | faixa de hora 15–18 h | `V9015` | `V9015` | treino, B1, boosting | sim | hora arredondada |
| `hora_noite` | faixa de hora 19–4 h (madrugada entra na noite do mesmo dia) | `V9015` | `V9015` | treino, B1, boosting | sim | 2 episódios de 0–4 h em 2008 |
| `sexta_a_domingo` | dia da semana em sexta, sábado ou domingo | `DIA_SEMANA` | `DIA_SEMANA` | treino | sim | composição de dias vem do calendário de coleta |
| `regiao_norte` | grande região, a partir da UF | `COD_UF` | `UF` | treino, B2, segmento | sim | — |
| `regiao_nordeste` | grande região, a partir da UF | `COD_UF` | `UF` | treino, B2, segmento | sim | — |
| `regiao_sudeste` | grande região, a partir da UF | `COD_UF` | `UF` | treino, B2, segmento | sim | — |
| `regiao_sul` | grande região, a partir da UF | `COD_UF` | `UF` | treino, B2, segmento | sim | — |
| `regiao_centro_oeste` | grande região, a partir da UF | `COD_UF` | `UF` | treino, B2, segmento | sim | — |
| `area_capital` | tipo de área: capital | estrato + planilha | estrato + planilha | treino | sim | 3 níveis porque limites metropolitanos mudaram |
| `area_urbano_nao_capital` | tipo de área: urbano fora da capital | estrato + planilha | estrato + planilha | treino | sim | idem |
| `area_rural` | tipo de área: rural | estrato + planilha | estrato + planilha | treino | sim | idem |
| `idade_10_17` | faixa etária 10–17 | `IDADE_ANOS` | `V0403` | treino, segmento | sim | data de referência difere em menos de um ano |
| `idade_18_29` | faixa etária 18–29 | `IDADE_ANOS` | `V0403` | treino, segmento | sim | idem |
| `idade_30_44` | faixa etária 30–44 | `IDADE_ANOS` | `V0403` | treino, segmento | sim | idem |
| `idade_45_59` | faixa etária 45–59 | `IDADE_ANOS` | `V0403` | treino, segmento | sim | idem |
| `idade_60_mais` | faixa etária 60 ou mais | `IDADE_ANOS` | `V0403` | treino, segmento | sim | 52 episódios em 2008 |
| `sexo_feminino` | sexo feminino (1) ou masculino (0) | `V0405` | `V0404` | treino | sim | — |
| `escolaridade_ate_fundamental` | níveis 1–3 do IBGE | `NIVEL_INSTRUCAO_MORADOR` | `NIVEL_INSTRUCAO` | treino | sim | algoritmo de derivação do IBGE não documentado |
| `escolaridade_medio` | níveis 4–5 do IBGE | `NIVEL_INSTRUCAO_MORADOR` | `NIVEL_INSTRUCAO` | treino | sim | idem |
| `escolaridade_superior` | níveis 6–7 do IBGE | `NIVEL_INSTRUCAO_MORADOR` | `NIVEL_INSTRUCAO` | treino | sim | idem |
| `log_renda_pc` | log da renda total da unidade de consumo ÷ todos os moradores, em reais de jan/2009 | `RENDA_TOTAL` | `RENDA_TOTAL` × 0,5895 (IPCA) | treino, segmento | sim | inclui renda não monetária e variação patrimonial; em reais de jan/2009, sem conversão de renda atual ainda |
| `moradores_uc` | número de moradores da unidade de consumo | cadastro de moradores | cadastro de moradores | treino | sim | atributo compartilhado pelos episódios da mesma unidade de consumo |
| `presenca_crianca_0_11` | há morador de 0 a 11 anos | cadastro de moradores | cadastro de moradores | treino | sim | — |
| `presenca_adolescente_12_17` | há morador de 12 a 17 anos | cadastro de moradores | cadastro de moradores | treino | sim | associada à faixa 10–17 do próprio episódio |
| `presenca_idoso_60_mais` | há morador com 60 anos ou mais | cadastro de moradores | cadastro de moradores | treino | sim | — |
| `sexta_a_domingo_x_noite` | interação: sexta a domingo e faixa noite | derivada | derivada | treino | sim | igual ao produto de duas colunas acima |
| `dia_semana` | dia da semana em 7 níveis (nativa) | `DIA_SEMANA` | `DIA_SEMANA` | B1 e boosting | sim | composição pelo calendário de coleta |
| `idade` | idade em anos (nativa) | `IDADE_ANOS` | `V0403` | boosting, alternativa | sim | — |
| `uf` | unidade da federação, 27 níveis | `COD_UF` | `UF` | só relato por segmento | sim | nenhuma UF tem 30 positivos |
| `indicadores_uf_sidra` | 4 indicadores de Censo e PNAD por UF e edição | SIDRA 2000 e 2006 | SIDRA 2010 e 2015 | contexto descritivo, fora do modelo | somente diagnóstico | no treino são a UF reescrita |
| `cod_upa` | UPA do episódio | cadastro | cadastro | grupo da validação cruzada e do bootstrap | não | — |
| `peso_final` | peso da pessoa (calibrado por UF, sexo e idade) | `PESO_FINAL` | `PESO_FINAL` (selado) | só sensibilidade ponderada (W1) | não | ver item 8 |
| `dia_coleta` | primeiro ou segundo dia de coleta | quadro 71/72 | quadro 72/73 | só diagnóstico | não | artefato do instrumento; não existe na consulta |
| `local_refeicao_v9018` | local da refeição | `V9018` | `V9018` | nenhum | não | muda de conceito entre as edições |
| `ocasiao_consumo` | ocasião declarada | ausente | presente | nenhum | não | não comparável |
| `dia_atipico` | dia atípico declarado | ausente | presente | nenhum | não | não comparável |
| `y_guarana` | alvo: guaraná no mesmo episódio | itens de consumo | itens de consumo (selado) | alvo | não | consumo autodeclarado; ver item 2 |
| `y_refrigerante` | algum refrigerante no mesmo episódio | itens de consumo | itens de consumo | só descrição | não | parte do desfecho |

## 4. Fontes e versões

Única fonte de registros observados de consumo: microdados da **POF** do IBGE, baixados por código com verificação de SHA-256 (`configs/manifesto_fontes.json`). Nenhuma outra fonte tem chave comum com o episódio (veja o item 5).

| Arquivo | Edição | Baixado em |
|---|---|---|
| `Dados_20231009.zip` | 2008–2009 | 30/10/2026 |
| `Documentacao_20231009.zip` | 2008–2009 | 30/10/2026 |
| `Dados_20230713.zip` | 2017–2018 | 30/10/2026 |
| `Documentacao_20230713.zip` | 2017–2018 | 27/10/2026; recuperado em 09/11/2026 |

URLs completas, tamanhos e SHA-256 ficam no [manifesto de fontes](../configs/manifesto_fontes.json), usado pelo pipeline na verificação dos downloads.

URLs em `ftp.ibge.gov.br/Orcamentos_Familiares/`, pastas `Pesquisa_de_Orcamentos_Familiares_2008_2009` e `..._2017_2018` (subpasta `Microdados`).

- **Contexto por UF** (`configs/contexto_uf.csv`): quatro indicadores do SIDRA (Censo 2000 e 2010; PNAD 2006 e 2015), só descritivos.
- **Correção de renda:** IPCA de jan/2009 (2.906,74) e jan/2018 (4.930,72); fator 0,5895163 multiplica só a renda de 2017–2018.
- **Ambiente:** Python 3.12, `duckdb==2.0.0`, `xlrd==2.0.2`, versões fixadas em `uv.lock`.
- **Entradas definitivas de 2008:** `e2_2008.csv`, `features_comuns_2008.csv`, `identificadores_2008.csv`, `variaveis_nativas_2008.csv` e `matriz_logistica_2008.csv`. Caminho relativo à raiz dos dados: `pkg_f1_01/pe01_v2/execucao_2/saida`, informado por `CASE_AMBEV_SAIDA_ROOT` ou `--saida`. Os SHA-256 estão centralizados em [entradas_2008.json](../configs/entradas_2008.json) e são conferidos antes da leitura. A pasta antiga sem as matrizes é recusada. Duas execuções seguidas produziram os mesmos hashes.

## 5. Regras de integração

- **Chaves.** Pessoa = UPA, domicílio, unidade de consumo e informante dentro de cada edição; UPA = UF (2) + sequência (3) + dígito (1). As edições nunca são unidas por chave: elas só se comparam por conceitos harmonizados.
- **Junções de 2008** (consumo → moradores → domicílio): sem perda nem duplicação, 1.168 episódios antes e depois de cada junção; chaves únicas conferidas; 30 de 30 contagens de referência da auditoria do projeto reproduzidas (não são 30 estatísticas publicadas pelo IBGE).
- **Alinhamento por `episodio_id`.** `variaveis_nativas`, `matriz_logistica` e `identificadores` têm a mesma ordem de linhas, por `episodio_id`. A EDA conferiu 14 checagens (mesmos episódios, ordem, UPA, peso, dia, região, área, idade, sexo, faixa de hora, interação e alvo dentro de refrigerante): 0 divergências em 1.168 episódios.
- **Harmonização entre edições** (`configs/crosswalk.json`, `configs/variaveis.json`): dia da semana 1 = segunda … 7 = domingo; hora inteira 0–23; área em 3 níveis; escolaridade em 3 níveis; renda corrigida pelo IPCA. Uma única função (`derivar_nativas`) aplica as regras às duas edições.
- **Fontes que não se integram.** Clima, feriados, Google Trends e OpenStreetMap ficam fora do núcleo: a POF não tem data de calendário nem município, então nenhuma chave coerente existe. Fontes de granularidades incompatíveis não são combinadas como se fossem a mesma observação.

## 6. Estratégia anti-leakage

- **Instante de previsão:** a refeição com pizza está definida; a bebida ainda não. Toda variável precisa existir nesse instante.
- **Nunca entram:** `dia_coleta`, local da refeição, ocasião, dia atípico e qualquer indicador de refrigerante do episódio (tabela do item 3).
- **Nada é aprendido dos dados na preparação.** Imputação, padronização e cortes por quantil ficam dentro das dobras de treino. Os tercis de renda da EDA são só descritivos e não entram na matriz do modelo.
- **Mesmo processamento no treino e na inferência**, pela função única de derivação.
- **Validação:** divisão sempre agrupada por UPA (pessoas e domicílios do mesmo lado); incerteza por bootstrap de UPAs inteiras; seleção só com dados de 2008.
- **Separação temporal:** desenvolvimento em 2008–2009; avaliação em 2017–2018, aberta **uma única vez**, com protocolo, variáveis, modelos e limiares congelados antes. O alvo e os pesos de 2017 estão em pasta somente leitura, fora do repositório, e **não foram abertos**. O módulo de 2017 nunca importa as listas de bebidas.
- **Exposição prévia:** os agregados de 2017 já eram conhecidos antes do congelamento: 1.703 episódios com pizza e 181 positivos (10,6%; 9,5% ponderado). Portanto, a avaliação temporal não é cega e deve ser interpretada com essa limitação.

## 7. População coberta e excluída

**Coberta:** moradores com 10 anos ou mais da subamostra de consumo alimentar da POF, em domicílios particulares, que consumiram pizza em algum dia de registro.

**Excluídos:** menores de 10 anos; domicílios coletivos; pessoas sem registro de consumo; quem não comeu pizza nos dias de registro.

| Passo (2008–2009) | Contagem |
|---|---|
| Pessoas com consumo registrado | 34.003 |
| Episódios (pessoa × dia × hora) no total | 287.680 |
| Episódios com pizza (população do estudo) | 1.168, de 1.106 pessoas |
| 2017–2018: episódios elegíveis (alvo selado) | 1.703 (contagem já verificada; sem nova tabulação) |

Não cobre: quem não consome pizza, menores de 10 anos, domicílios coletivos, consumo fora da janela de registro, nem qualquer comportamento de compra.

## 8. Vieses e limitações

- **Consumo registrado, não compra.** Sem informação de quem comprou, quanto pagou ou se foi combo. A POF registra consumo alimentar e **não comprova** pedidos de delivery, a marca Antarctica nem o efeito de uma ação de marketing.
- **Sem canal.** Delivery, restaurante e aplicativo não existem no bloco de consumo.
- **Sem marca.** Nenhum código de marca de guaraná; o alvo é um limite inferior de guaraná.
- **Instrumentos diferentes.** 2008–09 é registro alimentar preenchido pela pessoa; 2017–18 é recordatório de 24 h por entrevista. A diferença de taxa entre edições mistura tempo e instrumento e não é tendência. A água só é registrada em 2017–18.
- **Calendário de coleta.** Domingo é 13,4% dos episódios de pizza em 2008 e 32,2% em 2017; 41% dos episódios de 2008 estão no 1º dia de coleta e 59% no 2º. Nem pesos nem modelo removem esse efeito de desenho.
- **Dependência entre episódios.** 752 dos 1.168 episódios (64%) estão em domicílios com 2 ou mais episódios; ICC ≈ 0,65 entre moradores do mesmo domicílio. Os 101 positivos estão em 75 domicílios e 70 UPAs; **não são 101 observações independentes**. Os intervalos das tabelas e figuras são percentis do bootstrap de UPAs; o intervalo de Wilson aparece só como comparação e supõe episódios independentes. Com todos os resultados iguais (k = 0 ou k = n), o bootstrap é degenerado e não gera intervalo.
- **Grupos pequenos.** Manhã (65 episódios, 2 positivos) e almoço (89, 2) têm suporte baixo; nenhuma UF tem 30 positivos; 144 de 293 estratos têm uma só UPA. Zero observado não é probabilidade zero.
- **Pesos.** O IBGE fornece o peso da pessoa (W0), constante dentro da pessoa. O peso W1 (peso da pessoa ÷ dias de registro) é **derivado** e usado somente em análise de sensibilidade; a leitura principal é sem pesos. Pesos variam de 349 (percentil 1) a 33.901 (percentil 99); dois episódios têm peso 1,0.
- **Colinearidade.** As 28 colunas incluem 5 conjuntos completos de indicadores (hora, região, área, idade, escolaridade) que somam 1 e são colineares com o intercepto; exigem penalização, como ridge, ou um nível de referência.
- **Representatividade.** Dados de 2008–2009, com mais de 15 anos; os contextos de hoje podem ser outros.
- **Sem causalidade.** Nenhuma leitura pode afirmar que uma ativação venderá mais.

## 9. Licença e cuidados com LGPD

- **Licença.** A licença específica dos microdados do IBGE **não foi confirmada**. Microdados e derivados linha a linha não são redistribuídos.
- **Redistribuição.** Microdados e derivados linha a linha **nunca** entram no repositório (`.gitignore`); só tabelas agregadas e figuras são publicadas. Convenção implementada (do projeto, sem garantia): em tabelas e figuras, níveis com 1 a 4 episódios saem com n, k e taxas suprimidos; níveis com n ≥ 5 saem completos, inclusive com k de 1 a 4. A prosa dos documentos cita algumas contagens pequenas de estrutura e não segue essa convenção.
- **LGPD.** Os microdados tratam de pessoas reais. O projeto **não conferiu** campo a campo a ausência de identificadores diretos nem o risco de reidentificação por combinação de variáveis; a disponibilidade pública dos microdados não prova ausência de risco, e o limiar n < 5 não é garantia de anonimização. Por precaução, são tratados como potencialmente reidentificáveis. Não há tentativa de reidentificação nem de cruzamento com outras bases pessoais.
