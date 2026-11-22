"""Conversa compartilhada das abas Personas e Recomendacoes.

Fluxo: entender a mensagem -> validar o cenario -> recuperar evidencia -> calcular -> responder e atualizar o grafico.
Os numeros vem do servidor (`aplicacao.previsao_consumo` e a tabela validada de `dados/estimativas.json`); o texto e o
grafico leem o mesmo contrato (`resposta_conversa_v1`). O LLM e opcional, so redige e nunca calcula.
"""

VERSAO = "resposta_conversa_v1"
