"use strict";
// Ocasioes: pizza e guarana na mesma refeicao. Nenhum valor e calculado aqui: a pagina monta o pedido e mostra a resposta
// do servidor (GET /api/opcoes, POST /api/avaliar). Todo texto entra por textContent; rotulos de faixas, categorias,
// campos e motivos vem do servidor (opcoes.rotulos).

const $ = (id) => document.getElementById(id);
const estado = { opcoes: null, temResultado: false, avaliando: false, consulta: null };
const PUBLICO = { faixa_etaria: "faixas_etarias", sexo: "sexos", escolaridade: "escolaridades" };
const QUALQUER = { faixa_etaria: "Qualquer faixa", sexo: "Qualquer sexo", escolaridade: "Qualquer escolaridade" };
const TODOS = "todos";
const SIMBOLOS = { acima_do_habitual: "▲", abaixo_do_habitual: "▼", sem_diferenca_demonstrada: "≈", evidencia_insuficiente: "!" };
const DIRECOES = { eleva: "▲ eleva", reduz: "▼ reduz", neutro: "• neutro" };
const CATEGORIA_EM_DESTAQUE = "testar_combo_no_contexto";
const PASSOS_DA_ESCALA = [1, 2, 2.5, 5, 7.5, 10, 20];
const ESCALA_MINIMA = 0.001; // 0,1%: piso do eixo quando os valores sao zero // a escala termina num valor redondo logo acima do maior numero mostrado
const MOVIMENTO_REDUZIDO = "(prefers-reduced-motion: reduce)";
const MARGEM_DA_JANELA = 120; // px: resultado com o topo abaixo disso fica fora da vista

const maiuscula = (texto) => texto.charAt(0).toUpperCase() + texto.slice(1);
const rotulos = () => estado.opcoes.rotulos;
const nome = (campo) => rotulos().campos[campo] || campo;

function el(tag, propriedades, ...filhos) {
  const no = document.createElement(tag);
  Object.entries(propriedades || {}).forEach(([chave, valor]) => {
    if (chave === "texto") no.textContent = valor;
    else if (chave === "classe") no.className = valor;
    else no.setAttribute(chave, valor);
  });
  filhos.forEach((f) => no.append(f));
  return no;
}
const limpar = (no) => no.replaceChildren();
const inteiro = (v) => v.toLocaleString("pt-BR");
const decimal = (v, casas) => v.toLocaleString("pt-BR", { minimumFractionDigits: casas, maximumFractionDigits: casas });
const numero = (v, casas = 1) => decimal(Math.abs(v), casas);
const casasDe = (v) => (v > 0 && v < 0.001 ? 3 : v < 0.01 ? 2 : 1);
const pct = (v) => `${numero(v * 100, casasDe(Math.abs(v)))}%`;
const sinal = (v, limiar) => (v >= limiar ? "+" : v <= -limiar ? "−" : "");
const pp = (v) => `${sinal(v, 0.0005)}${numero(v, 3)} p.p.`;
const pares = (dl, linhas) => {
  limpar(dl);
  linhas.forEach(([rotulo, valor]) => dl.append(el("dt", { texto: rotulo }), el("dd", { texto: String(valor) })));
};
const rotuloDe = (lista, valor) => (lista.find((x) => String(x.valor) === String(valor)) || { rotulo: String(valor) }).rotulo;

// ---------- acesso a API ----------
function falha(mensagem, tipo) {
  const erro = new Error(mensagem);
  erro.tipo = tipo;
  return erro;
}

async function api(caminho, opcoes) {
  let resposta;
  try {
    resposta = await fetch(caminho, opcoes);
  } catch (erro) {
    throw falha("Não foi possível falar com o servidor local. Confira se ele continua em execução.", "conexao");
  }
  let corpo;
  try {
    corpo = await resposta.json();
  } catch (erro) {
    throw falha(`O servidor respondeu de forma inesperada (código ${resposta.status}).`, "conexao");
  }
  if (!corpo || typeof corpo !== "object" || corpo.success !== true) {
    const detalhe = corpo && corpo.error && typeof corpo.error === "object" ? corpo.error : {};
    throw falha(typeof detalhe.message === "string" ? detalhe.message : "Falha sem mensagem.", detalhe.type || "desconhecido");
  }
  if (!resposta.ok || !corpo.data || typeof corpo.data !== "object") {
    throw falha(`O servidor respondeu de forma inesperada (código ${resposta.status}).`, "conexao");
  }
  return corpo.data;
}
const enviarJson = (caminho, dados) => api(caminho, {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(dados)
});

// ---------- campos ----------
// Grupo de chips: radios nativos (setas mudam a escolha, Tab passa ao proximo grupo).
function chips(container, opcoes, padrao) {
  limpar(container);
  opcoes.forEach((o) => {
    const entrada = el("input", { type: "radio", name: container.id, value: String(o.valor) });
    entrada.checked = String(o.valor) === String(padrao);
    const texto = el("span", { texto: o.texto });
    if (o.detalhe) texto.append(el("small", { texto: o.detalhe }));
    container.append(el("label", {}, entrada, texto));
  });
}
const escolhido = (nomeDoGrupo) => {
  const marcado = document.querySelector(`input[name="${nomeDoGrupo}"]:checked`);
  return marcado ? marcado.value : null;
};

function preencher(select, opcoes, padrao) {
  limpar(select);
  opcoes.forEach((o) => select.append(el("option", { value: String(o.valor), texto: o.rotulo })));
  if (padrao !== undefined) select.value = String(padrao);
}

// "noite (19h–4h)" vira chip "Noite" com o intervalo em letra menor.
function partesDoPeriodo(rotulo) {
  const partes = /^(.*?)\s*\((.*)\)$/.exec(rotulo);
  return partes ? { texto: maiuscula(partes[1]), detalhe: partes[2] } : { texto: maiuscula(rotulo) };
}
const nomeCurto = (rotulo) => maiuscula(rotulo.split("-")[0]);

function montarCampos(o) {
  chips($("dia_semana"), o.dias.map((d) => ({ valor: d.valor, texto: nomeCurto(d.rotulo) })), 5);
  chips($("periodo"), o.periodos.map((p) => ({ valor: p.valor, ...partesDoPeriodo(p.rotulo) })), "noite");
  const periodoDe = (valor) => partesDoPeriodo(rotuloDe(o.periodos, valor)).texto.toLowerCase();
  preencher($("hora"), o.horas.map((h) => ({ valor: h.valor, rotulo: `${h.rotulo} · ${periodoDe(h.periodo)}` })));
  chips($("grande_regiao"), o.regioes.map((r) => ({ valor: r.valor, texto: r.rotulo })), "Sudeste");
  chips($("area"), o.areas.map((a) => ({ valor: a.valor, texto: maiuscula(a.rotulo) })), o.areas[0].valor);
  Object.entries(PUBLICO).forEach(([campo, lista]) => {
    const opcoes = o[lista].map((x) => (x.valor === TODOS ? { valor: TODOS, rotulo: QUALQUER[campo] } : { ...x, rotulo: maiuscula(x.rotulo) }));
    preencher($(campo), opcoes, TODOS);
  });
}

// Periodo OU hora exata: so o campo escolhido fica visivel e vai no pedido.
function aplicarMomento() {
  const porHora = escolhido("momento") === "hora";
  $("campo-periodo").hidden = porHora;
  $("campo-hora").hidden = !porHora;
  if (!porHora) return;
  const periodo = escolhido("periodo");
  const atual = estado.opcoes.horas.find((h) => String(h.valor) === $("hora").value);
  if (!atual || atual.periodo !== periodo) {
    const primeira = estado.opcoes.horas.find((h) => h.periodo === periodo);
    if (primeira) $("hora").value = String(primeira.valor);
  }
}

function aplicarPublico() {
  const geral = escolhido("publico") !== "escolhido";
  $("campos-publico").hidden = geral;
  if (geral) Object.keys(PUBLICO).forEach((campo) => { $(campo).value = TODOS; });
}

function contexto() {
  const momento = escolhido("momento") === "hora" ? { hora: Number($("hora").value) } : { periodo: escolhido("periodo") };
  const geral = escolhido("publico") !== "escolhido";
  const publico = Object.fromEntries(Object.keys(PUBLICO).map((campo) => [campo, geral ? TODOS : $(campo).value]));
  return { dia_semana: Number(escolhido("dia_semana")), ...momento, grande_regiao: escolhido("grande_regiao"),
    area: escolhido("area"), ...publico };
}

// ---------- envio e estados ----------
const rolagem = () => (window.matchMedia(MOVIMENTO_REDUZIDO).matches ? "auto" : "smooth");
function foraDaJanela(no) {
  const topo = no.getBoundingClientRect().top;
  return topo < document.querySelector(".topo").offsetHeight || topo > window.innerHeight - MARGEM_DA_JANELA;
}

function marcarDesatualizado() {
  if (!estado.temResultado) return;
  if ($("desatualizado").hidden) $("anuncio").textContent = $("desatualizado").textContent; // anuncia so na transicao
  $("desatualizado").hidden = false;
  $("aviso-botao").hidden = false; // no celular o resultado fica abaixo do formulario: o aviso aparece tambem junto ao botao
  $("resultado").classList.add("esta-desatualizado");
}

function mostrarMensagem(no, mensagem) {
  no.textContent = mensagem;
  no.hidden = false;
}

// O botao fica com aria-disabled (e nao disabled) enquanto avalia: assim o foco do teclado nao se perde.
function marcarAvaliando(ativo) {
  const botao = $("avaliar-botao");
  estado.avaliando = ativo;
  botao.textContent = ativo ? "Avaliando…" : "Avaliar ocasião";
  if (ativo) botao.setAttribute("aria-disabled", "true"); else botao.removeAttribute("aria-disabled");
  $("resultado").toggleAttribute("data-ocupado", ativo);
  $("resultado").setAttribute("aria-busy", String(ativo));
}

async function avaliar(evento) {
  evento.preventDefault();
  if (estado.avaliando) return;
  $("resumo-erros").hidden = true;
  $("falha").hidden = true;
  $("anuncio").textContent = "Avaliando a ocasião…";
  marcarAvaliando(true);
  try {
    const pedido = { contexto: contexto() };
    const dados = await enviarJson("/api/avaliar", pedido);
    estado.consulta = pedido;
    mostrarResultado(dados);
    if (JSON.stringify(pedido.contexto) !== JSON.stringify(contexto())) marcarDesatualizado();
    if (foraDaJanela($("resultado"))) $("resultado").scrollIntoView({ block: "start", behavior: rolagem() });
  } catch (erro) {
    $("anuncio").textContent = "";
    if (!erro.tipo || erro.tipo === "entrada_invalida") {
      $("conteudo-resultado").hidden = true;
      $("estado-vazio").hidden = false;
      $("aviso-botao").hidden = true;
      $("resultado").classList.remove("esta-desatualizado");
      estado.temResultado = false;
    }
    if (erro.tipo === "entrada_invalida") {
      mostrarMensagem($("resumo-erros"), `Revise a ocasião: ${erro.message}`);
      $("resumo-erros").focus();
    } else {
      mostrarMensagem($("falha"), erro.tipo ? erro.message : "Não foi possível mostrar o resultado. Recarregue a página e tente de novo.");
      if (foraDaJanela($("falha"))) $("falha").scrollIntoView({ block: "center", behavior: rolagem() });
    }
  } finally {
    marcarAvaliando(false);
  }
}

// ---------- resultado ----------
function descreverCenario(c) {
  const o = estado.opcoes;
  const momento = c.hora !== undefined ? `às ${rotuloDe(o.horas, c.hora)}` : partesDoPeriodo(rotuloDe(o.periodos, c.periodo)).texto.toLowerCase();
  const partes = [maiuscula(rotuloDe(o.dias, c.dia_semana)), momento, c.grande_regiao, `área ${rotuloDe(o.areas, c.area)}`];
  const publico = Object.entries(PUBLICO).filter(([campo]) => c[campo] !== TODOS).map(([campo, lista]) => (campo === "escolaridade" ? `escolaridade ${rotuloDe(o[lista], c[campo])}` : rotuloDe(o[lista], c[campo])));
  partes.push(publico.length ? publico.join(", ") : "público geral");
  return partes.join(" · ");
}

function escalaRedonda(maximo) {
  if (!(maximo > 0)) return ESCALA_MINIMA;
  const alvo = maximo * 1.05;
  const ordem = 10 ** Math.floor(Math.log10(alvo));
  return (PASSOS_DA_ESCALA.find((p) => p * ordem >= alvo) || 10) * ordem;
}

const largura = (no, inicio, fim, escala) => {
  no.style.left = `${Math.max(0, Math.min(100, (inicio / escala) * 100))}%`;
  no.style.width = `${Math.max(0.6, Math.min(100, ((fim - inicio) / escala) * 100))}%`;
};

function mostrarBarras(c) {
  const [baixo, alto] = c.intervalo_estimativa;
  const escala = escalaRedonda(Math.max(alto, c.estimativa, c.referencia));
  largura($("barra-estimativa"), 0, c.estimativa, escala);
  largura($("barra-faixa"), Math.max(0, baixo), alto, escala);
  largura($("barra-media"), 0, c.referencia, escala);
  $("barra-estimativa-valor").textContent = pct(c.estimativa);
  $("barra-media-valor").textContent = pct(c.referencia);
  $("escala-meio").textContent = pct(escala / 2);
  $("escala-maximo").textContent = pct(escala);
}

function mostrarFatores(r) {
  limpar($("fatores"));
  r.top_factors.forEach((f) => $("fatores").append(el("li", {},
    el("span", { classe: "fator-direcao", "data-direcao": f.direcao, texto: DIRECOES[f.direcao] || f.direcao }),
    el("span", { classe: "fator-campo", texto: nome(f.campo) }),
    el("span", { classe: "fator-texto", texto: f.texto }))));
  if (!r.top_factors.length) $("fatores").append(el("li", {}, el("span", { classe: "fator-texto", texto: "Nenhum campo confirmado mudou a estimativa de forma perceptível." })));
  const semEfeito = r.fatores_avaliados.filter((f) => !f.ajuda_o_modelo).map((f) => nome(f.campo).toLowerCase());
  $("sem-efeito").textContent = semEfeito.length
    ? `Sem ganho para o modelo na validação (não usados para explicar): ${semEfeito.join(", ")}.` : "";
  $("sem-efeito").hidden = !semEfeito.length;
}

function mostrarAcao(r) {
  const destaque = r.recommendation_category === CATEGORIA_EM_DESTAQUE;
  $("cartao-acao").classList.toggle("acao-destaque", destaque);
  $("acao-rotulo").textContent = destaque ? "Ação que vale testar" : "Orientação para este cenário";
  $("categoria").textContent = rotulos().categorias[r.recommendation_category] || r.recommendation_category;
  $("proximo-passo").textContent = r.proximo_passo;
  $("justificativa").textContent = r.recommendation_reason;
}

function mostrarPlacar(r, insuficiente) {
  const c = r.comparacao;
  $("num-estimativa-rotulo").textContent = insuficiente ? r.leitura.titulo : "Chance neste cenário";
  $("num-estimativa").textContent = pct(r.occasion_score);
  $("num-medida").textContent = "das refeições com comida têm pizza e guaraná juntos";
  $("faixa-simbolo").textContent = SIMBOLOS[c.faixa] || "";
  $("faixa-texto").textContent = rotulos().faixas[c.faixa] || c.faixa;
  $("num-media").textContent = `Média das refeições: ${pct(c.referencia)}`;
  $("linha-medida").textContent = r.leitura.linha;
  $("linha-medida").hidden = insuficiente;
  $("leitura").textContent = r.leitura.mensagem;
  $("barras").hidden = insuficiente;
  if (!insuficiente) mostrarBarras(c);
}

function mostrarResultado(r) {
  const c = r.comparacao;
  const insuficiente = c.faixa === "evidencia_insuficiente";
  $("conteudo-resultado").setAttribute("data-faixa", c.faixa);
  $("cenario").textContent = descreverCenario(estado.consulta.contexto);
  mostrarPlacar(r, insuficiente);
  $("titulo-fatores").textContent = insuficiente ? "O que pesou na estimativa exploratória" : "O que mais pesou";
  mostrarFatores(r);
  mostrarAcao(r);
  mostrarDetalhes(r);
  // so depois de tudo preenchido o resultado aparece (uma falha no meio nao deixa tela parcial)
  estado.temResultado = true;
  $("estado-vazio").hidden = true;
  $("desatualizado").hidden = true;
  $("aviso-botao").hidden = true;
  $("resultado").classList.remove("esta-desatualizado");
  $("conteudo-resultado").hidden = false;
  $("conteudo-resultado").classList.remove("surge"); // entrada suave a cada resultado novo (sem efeito com movimento reduzido)
  void $("conteudo-resultado").offsetWidth;
  $("conteudo-resultado").classList.add("surge");
  anunciar(`${$("num-estimativa-rotulo").textContent}: ${$("num-estimativa").textContent}, ${$("faixa-texto").textContent}. `
    + `${$("acao-rotulo").textContent}: ${$("categoria").textContent}.`);
}

// Limpa e regrava a regiao de status: o mesmo texto repetido (reavaliar o mesmo cenario) volta a ser anunciado.
function anunciar(texto) {
  $("anuncio").textContent = "";
  window.setTimeout(() => { $("anuncio").textContent = texto; }, 60);
}

function mostrarDetalhes(r) {
  const c = r.comparacao;
  const s = c.suporte;
  const k = r.componentes;
  const nivel = numero(c.nivel_do_intervalo * 100, 0);
  $("leitura-detalhe").textContent = r.leitura.detalhe || "";
  $("leitura-detalhe").hidden = !r.leitura.detalhe;
  const numeros = [
    ["Estimativa do modelo", `${pct(c.estimativa)} (faixa de ${nivel}%: ${pct(c.intervalo_estimativa[0])} a ${pct(c.intervalo_estimativa[1])})`],
    ["Diferença para a média", c.intervalo_pp
      ? `${pp(c.diferenca_pp)} (faixa de ${nivel}%: ${pp(c.intervalo_pp[0])} a ${pp(c.intervalo_pp[1])})`
      : `${pp(c.diferenca_pp)} (faixa indisponível)`],
    [c.referencia_nome ? maiuscula(c.referencia_nome) : "Média", pct(c.referencia)],
    ["Regra simples (dia × período)", pct(c.regra_dia_periodo)]];
  if (k) {
    numeros.push(["Pizza na ocasião", `${pct(k.pizza)} das refeições (média ${pct(k.pizza_referencia)})`]);
    numeros.push(["Guaraná quando há pizza", `${pct(k.guarana_dado_pizza)} das refeições com pizza (média ${pct(k.guarana_dado_pizza_referencia)})`
      + (k.guarana_sustentado ? "" : " · variação entre contextos não confirmada no teste")]);
  }
  pares($("numeros"), numeros);
  const suporte = [
    ["Refeições contadas", `${inteiro(s.episodios)} no treino, ${inteiro(s.positivos)} com pizza e guaraná`],
    ["Campos da contagem", s.campos_considerados.join(" · ")]];
  if (s.filtros_nao_cobertos.length) suporte.push(["A contagem não filtra (o modelo usa)", s.filtros_nao_cobertos.join("; ")]);
  const minimo = c.minimos_de_suporte;
  suporte.push(["Mínimo para comparar", `${minimo.episodios} refeições e ${minimo.positivos} positivas (${minimo.significado})`]);
  pares($("suporte"), suporte);
  limpar($("avisos"));
  [...c.limites.map((limite) => `${maiuscula(limite)}.`), ...r.model_metadata.avisos]
    .forEach((texto) => $("avisos").append(el("li", { texto })));
  const m = r.model_metadata;
  const t = m.avaliacao_teste;
  const captura = (n) => `${numero(t[n].captura_20 * 100, 0)}% dos positivos nos 20% priorizados`;
  pares($("detalhes"), [
    ["Base", m.fonte], ["Alvo", `${m.alvo}; unidade: ${m.unidade}`], ["Modelo", `${m.modelo} · ${m.versao}`],
    ["Teste reservado", `${inteiro(t.escolhido.n)} refeições, ${inteiro(t.escolhido.positivos)} positivas`],
    ["Priorização no teste", `modelo: ${captura("escolhido")}; regra dia × período: ${captura("dia_periodo")}`],
    ["Público", r.contexto.faixa_etaria === TODOS && r.contexto.sexo === TODOS && r.contexto.escolaridade === TODOS
      ? "público geral (composição observada na região e área)" : "público informado"],
    ["Horário", r.contexto.hora_informada ? `${r.contexto.hora}h` : "média das horas do período"],
    ["Regra de ativação", `${m.regra.id} (${m.regra.versao})`]]);
}

// ---------- inicio ----------
function ligarFormulario() {
  $("formulario").addEventListener("submit", avaliar);
  $("formulario").addEventListener("change", (evento) => {
    if (evento.target.name === "momento" || evento.target.name === "periodo") aplicarMomento();
    if (evento.target.name === "publico") aplicarPublico();
    marcarDesatualizado();
  });
}

async function iniciar() {
  $("formulario").addEventListener("submit", (evento) => evento.preventDefault()); // nunca o envio nativo do navegador
  try {
    estado.opcoes = await api("/api/opcoes");
    montarCampos(estado.opcoes);
    const modelo = estado.opcoes.modelo;
    $("taxa-inicial").textContent = pct(modelo.referencia);
    $("fonte-inicial").textContent = `POF/IBGE 2017–2018 · ${inteiro(modelo.refeicoes)} refeições no desenvolvimento`;
    $("referencia-inicial").hidden = false;
    aplicarMomento();
    aplicarPublico();
    ligarFormulario();
    $("avaliar-botao").disabled = false;
  } catch (erro) {
    const motivo = erro.tipo ? erro.message : "a lista de opções veio incompleta.";
    mostrarMensagem($("falha"), `A ferramenta não pôde ser carregada: ${motivo} Recarregue a página.`);
  }
}

document.addEventListener("DOMContentLoaded", iniciar);
