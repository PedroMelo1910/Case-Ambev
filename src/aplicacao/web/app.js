"use strict";
// Ferramenta de oportunidade por contexto. Nenhum valor e calculado aqui: a pagina monta o pedido, mostra a resposta
// do servidor e acompanha o treino com dados da empresa. Todo texto entra por textContent. Os rotulos de faixas,
// categorias, papeis, modelos e campos vem do servidor (opcoes.rotulos).

const $ = (id) => document.getElementById(id);
const estado = { opcoes: null, temResultado: false, avaliando: false, envio: null, envios: 0, experimento: null,
  liberarTreino: null, consulta: null };
const SECOES_DE_OCASIOES = ["avaliar", "empresa", "como"];
const SECOES = [...SECOES_DE_OCASIOES, "personas"];
const NOME_DA_FERRAMENTA = "Oportunidade por contexto";
const SITUACOES = { concluida: "Concluída", em_andamento: "Em andamento", falhou: "Falhou" };
const VALIDACOES = { temporal: "Temporal (treino antes do corte, avaliação depois)", grupos: "Lojas fora do treino",
  estratificada: "Divisões estratificadas" };
const DECLARAVEIS = ["canal", "promocao", "preco", "temperatura"];
const PUBLICO = { faixa_etaria: "faixas_etarias", sexo: "sexos", escolaridade: "escolaridades" };
const COLUNAS = { identificador: "Identificador do pedido", data_hora: "Data e hora", localizacao: "UF ou região",
  pizza: "Tem pizza", guarana: "Tem guaraná", grupo: "Loja" };
const INTERVALO_DO_TREINO = 1500; // ms entre consultas do andamento
const MAXIMO_CONSULTAS_DO_TREINO = 1200; // 30 minutos
const MAXIMO_FALHAS_SEGUIDAS = 3; // falhas de conexao toleradas antes de desistir do acompanhamento
const PASSOS_SEGUINTES = ["passo-colunas", "passo-diagnostico", "passo-treino"];
const MOVIMENTO_REDUZIDO = "(prefers-reduced-motion: reduce)";
const APARELHO_LIMITADO = { nucleos: 2, memoriaGb: 2 }; // abaixo disso a ambientacao some
const MARGEM_DA_JANELA = 120; // px: resultado com o topo abaixo disso fica fora da vista
const rolagem = () => (window.matchMedia(MOVIMENTO_REDUZIDO).matches ? "auto" : "smooth");

const maiuscula = (texto) => texto.charAt(0).toUpperCase() + texto.slice(1);
const rotulos = () => estado.opcoes.rotulos;
function nome(chave) {
  const r = rotulos();
  if (COLUNAS[chave]) return COLUNAS[chave];
  if (r.campos[chave]) return r.campos[chave];
  return r.modelos[chave] ? maiuscula(r.modelos[chave]) : chave;
}

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
const pct = (v) => `${numero(v * 100)}%`;
const sinal = (v, limiar) => (v >= limiar ? "+" : v <= -limiar ? "−" : "");
const pp = (v) => `${sinal(v, 0.05)}${numero(v)} p.p.`;
const relativo = (v) => (v === null || v === undefined ? "sem diferença relativa definida" : `${sinal(v, 0.005)}${numero(v * 100, 0)}% em termos relativos`);
const pares = (dl, linhas) => {
  limpar(dl);
  linhas.forEach(([rotulo, valor]) => dl.append(el("dt", { texto: rotulo }), el("dd", { texto: String(valor) })));
};

function falhaDeConexao(mensagem) {
  const falha = new Error(mensagem);
  falha.tipo = "conexao";
  return falha;
}

async function api(caminho, opcoes) {
  let resposta;
  try {
    resposta = await fetch(caminho, opcoes);
  } catch (erro) {
    throw falhaDeConexao("Não foi possível falar com o servidor local. Confira se ele continua em execução.");
  }
  let corpo;
  try {
    corpo = await resposta.json();
  } catch (erro) {
    throw falhaDeConexao(`O servidor respondeu de forma inesperada (código ${resposta.status}).`);
  }
  if (!corpo.success) {
    const falha = new Error(corpo.error ? corpo.error.message : "Falha sem mensagem.");
    falha.tipo = corpo.error ? corpo.error.type : "desconhecido";
    throw falha;
  }
  return corpo.data;
}
const enviarJson = (caminho, dados) => api(caminho, {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(dados)
});

function preencher(select, opcoes, padrao) {
  limpar(select);
  opcoes.forEach((o) => select.append(el("option", { value: String(o.valor), texto: o.rotulo })));
  if (padrao !== undefined) select.value = String(padrao);
}

// ---------- navegacao e secoes ----------
function secaoDoEndereco() {
  const id = window.location.hash.slice(1);
  if (id === "") return SECOES[0]; // endereco sem ancora (por exemplo, ao voltar no navegador): primeira secao
  if (id === "recomendacoes") return "personas";
  return SECOES.includes(id) ? id : null;
}

function mostrarSecao(id, focar) {
  const deOcasioes = SECOES_DE_OCASIOES.includes(id);
  SECOES.forEach((secao) => { $(secao).hidden = secao !== id; });
  $("secoes").hidden = !deOcasioes; // o menu de secoes e das Ocasioes
  $("secoes").querySelectorAll("a").forEach((link) => {
    if (link.getAttribute("href") !== `#${id}`) { link.removeAttribute("aria-current"); return; }
    link.setAttribute("aria-current", "page");
    document.title = `${link.textContent} · ${NOME_DA_FERRAMENTA}`;
  });
  const ferramenta = deOcasioes ? "#avaliar" : `#${id}`; // no menu superior, as tres secoes de Ocasioes valem como uma
  $("navegacao").querySelectorAll("a").forEach((link) => {
    if (link.getAttribute("href") !== ferramenta) { link.removeAttribute("aria-current"); return; }
    link.setAttribute("aria-current", "page");
    if (!deOcasioes) document.title = `${link.textContent} · ${NOME_DA_FERRAMENTA}`;
  });
  document.dispatchEvent(new CustomEvent("secao-aberta", { detail: id }));
  if (!focar) return;
  window.scrollTo(0, 0);
  $(id).querySelector("h1").focus({ preventScroll: true });
}

// Cabecalho fixo: a altura real vira --altura-topo (as ancoras param abaixo dele) e a sombra so aparece apos rolar.
function ligarCabecalho() {
  const topo = document.querySelector(".topo");
  const medir = () => document.documentElement.style.setProperty("--altura-topo", `${topo.offsetHeight}px`);
  if ("ResizeObserver" in window) new ResizeObserver(medir).observe(topo);
  medir();
  const sombra = () => topo.classList.toggle("rolou", window.scrollY > 4);
  window.addEventListener("scroll", sombra, { passive: true });
  sombra();
}

function ligarNavegacao() {
  document.addEventListener("click", (evento) => { // link para a secao ja aberta: o endereco nao muda
    const link = evento.target.closest('a[href^="#"]');
    if (link && SECOES.includes(link.hash.slice(1)) && link.hash === window.location.hash) {
      mostrarSecao(link.hash.slice(1), true);
    }
  });
  document.querySelector(".pular").addEventListener("click", (evento) => { // sem mudar o endereco: a secao atual fica
    evento.preventDefault();
    $("conteudo").focus();
  });
  window.addEventListener("hashchange", () => {
    const id = secaoDoEndereco();
    if (id) mostrarSecao(id, true);
  });
  const inicial = secaoDoEndereco();
  if (inicial && inicial !== SECOES[0]) mostrarSecao(inicial, false);
}

// ---------- ambientacao (decorativa) ----------
// Pausa quando a aba fica oculta; some em aparelhos limitados ou com economia de dados; uma imagem que falha e
// escondida. Movimento reduzido fica a cargo do CSS (versao estatica). Nada aqui afeta a ferramenta.
function aparelhoLimitado() {
  const nucleos = navigator.hardwareConcurrency || 8;
  const memoria = navigator.deviceMemory || 8;
  const economia = navigator.connection && navigator.connection.saveData;
  return Boolean(economia) || nucleos <= APARELHO_LIMITADO.nucleos || memoria <= APARELHO_LIMITADO.memoriaGb;
}

const CHAVE_PARADO = "fundo-parado"; // preferencia do visitante, so neste navegador

function lerParado() {
  try { return window.localStorage.getItem(CHAVE_PARADO) === "1"; } catch (erro) { return false; }
}

function gravarParado(parado) {
  try { window.localStorage.setItem(CHAVE_PARADO, parado ? "1" : "0"); } catch (erro) { /* sem armazenamento: vale so nesta visita */ }
}

// Botao "Pausar animacao do fundo" (WCAG 2.2.2): para as latas e as folhas de vez, em posicao estatica.
function ligarPausaDoFundo(raiz) {
  const botao = $("pausar-fundo");
  const aplicar = (parado) => {
    raiz.classList.toggle("parado", parado);
    botao.setAttribute("aria-pressed", String(parado));
  };
  aplicar(lerParado());
  botao.addEventListener("click", () => {
    const parado = botao.getAttribute("aria-pressed") !== "true";
    aplicar(parado);
    gravarParado(parado);
  });
}

function ligarAmbiente() {
  const raiz = document.documentElement;
  if (aparelhoLimitado()) raiz.classList.add("ambiente-leve");
  else raiz.classList.add("ambiente-ativo"); // sem esta classe (aparelho limitado ou JS que falhou) nada decorativo carrega
  ligarPausaDoFundo(raiz);
  const pausar = () => raiz.classList.toggle("pausado", document.hidden || window.matchMedia(MOVIMENTO_REDUZIDO).matches);
  document.addEventListener("visibilitychange", pausar);
  window.matchMedia(MOVIMENTO_REDUZIDO).addEventListener("change", pausar);
  pausar();
  document.querySelectorAll(".ambiente img").forEach((imagem) => {
    imagem.addEventListener("error", () => { imagem.hidden = true; });
  });
}

function carregarMarca() {
  const fontes = ["/marca/ambev.svg", "/marca/ambev.png"];
  const tentar = (i) => {
    if (i >= fontes.length) return;
    const teste = new Image();
    teste.onload = () => { $("logo").src = fontes[i]; $("logo").hidden = false; };
    teste.onerror = () => tentar(i + 1);
    teste.src = fontes[i];
  };
  tentar(0);
}

// ---------- avaliar ----------
function base() { return document.querySelector('input[name="base"]:checked').value; }

function aplicarBase() {
  const atual = base();
  const campos = estado.opcoes.empresa ? estado.opcoes.empresa.campos : [];
  document.querySelectorAll("[data-base]").forEach((no) => {
    const campo = no.getAttribute("data-campo");
    no.hidden = no.getAttribute("data-base") !== atual || (campo !== null && !campos.includes(campo));
  });
  marcarDesatualizado();
}

function contexto() {
  const valorInteiro = (id) => Number($(id).value);
  // O modelo distingue so o periodo: a pagina envia o periodo escolhido, sem inventar uma hora representativa.
  const comum = { dia_semana: valorInteiro("dia_semana"), periodo: $("periodo").value, grande_regiao: $("grande_regiao").value };
  if (base() === "historica") {
    return { ...comum, area: $("area").value, faixa_etaria: $("faixa_etaria").value, sexo: $("sexo").value,
      escolaridade: $("escolaridade").value };
  }
  const extras = {};
  estado.opcoes.empresa.campos.forEach((campo) => {
    if (campo === "canal") extras.canal = $("canal").value;
    if (campo === "promocao") extras.promocao = valorInteiro("promocao");
    if (campo === "preco" || campo === "temperatura") {
      if ($(campo).value.trim() === "") {
        const falta = new Error(`Informe ${nome(campo).toLowerCase()}.`);
        falta.campo = campo;
        throw falta;
      }
      extras[campo] = Number($(campo).value);
    }
  });
  return { ...comum, ...extras };
}

function marcarDesatualizado() {
  if (!estado.temResultado) return;
  if ($("desatualizado").hidden) $("anuncio").textContent = $("desatualizado").textContent; // anuncia so na transicao
  $("desatualizado").hidden = false;
  $("resultado").classList.add("esta-desatualizado");
}

function mostrarMensagem(no, mensagem) {
  no.textContent = mensagem;
  no.hidden = false;
}

// Campo que falta: o resumo de erros leva ate ele e o campo fica marcado ate ser alterado.
function mostrarErroDeCampo(erro) {
  const campo = $(erro.campo);
  const link = el("a", { href: `#${erro.campo}`, texto: erro.message });
  link.addEventListener("click", (evento) => { evento.preventDefault(); campo.focus(); });
  campo.setAttribute("aria-invalid", "true");
  campo.setAttribute("aria-describedby", "resumo-erros");
  $("resumo-erros").replaceChildren("Revise a ocasião: ", link);
  $("resumo-erros").hidden = false;
}

// O botao fica com aria-disabled (e nao disabled) enquanto avalia: assim o foco do teclado nao se perde. O painel
// recebe so um atributo de estilo: aria-busy calaria os anuncios feitos dentro dele.
function marcarAvaliando(ativo) {
  const botao = $("avaliar-botao");
  estado.avaliando = ativo;
  botao.textContent = ativo ? "Avaliando…" : "Avaliar ocasião";
  if (ativo) botao.setAttribute("aria-disabled", "true"); else botao.removeAttribute("aria-disabled");
  $("resultado").toggleAttribute("data-ocupado", ativo);
}

async function avaliar(evento) {
  evento.preventDefault();
  if (estado.avaliando) return;
  const foraDaJanela = (no) => { // o cabecalho fixo cobre o alto da janela
    const topo = no.getBoundingClientRect().top;
    return topo < document.querySelector(".topo").offsetHeight || topo > window.innerHeight - MARGEM_DA_JANELA;
  };
  $("resumo-erros").hidden = true;
  $("falha").hidden = true;
  $("anuncio").textContent = "Avaliando a ocasião…";
  marcarAvaliando(true);
  try {
    const pedido = { base: base(), contexto: contexto() };
    const dados = await enviarJson("/api/avaliar", pedido);
    estado.consulta = pedido;
    mostrarResultado(dados);
    if (foraDaJanela($("resultado"))) $("resultado").scrollIntoView({ block: "start", behavior: rolagem() });
  } catch (erro) {
    $("anuncio").textContent = "";
    if (erro.campo) {
      mostrarErroDeCampo(erro);
      $("resumo-erros").focus();
    } else if (erro.tipo === "entrada_invalida") {
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

function posicionar(no, inicio, fim, escala) {
  no.style.left = `${Math.max(0, Math.min(100, (inicio / escala) * 100))}%`;
  if (fim !== undefined) no.style.width = `${Math.max(0.5, ((fim - inicio) / escala) * 100)}%`;
}

function descreverCenario(pedido) {
  const o = estado.opcoes;
  const c = pedido.contexto;
  const rotulo = (lista, valor) => (lista.find((x) => String(x.valor) === String(valor)) || { rotulo: String(valor) }).rotulo;
  const partes = [maiuscula(rotulo(o.dias, c.dia_semana)), rotulo(o.periodos, c.periodo), c.grande_regiao];
  if (c.area) partes.push(`área ${rotulo(o.areas, c.area)}`);
  Object.entries(PUBLICO).forEach(([campo, lista]) => {
    if (c[campo] && c[campo] !== "todos") partes.push(rotulo(o[lista], c[campo]));
  });
  if (c.canal) partes.push(`canal ${c.canal}`);
  if (c.promocao !== undefined) partes.push(c.promocao ? "com promoção" : "sem promoção");
  if (c.preco !== undefined) partes.push(`preço R$ ${decimal(c.preco, 2)}`);
  if (c.temperatura !== undefined) partes.push(`${decimal(c.temperatura, 1)} °C`);
  return partes.join(partes.some((parte) => /\d,\d/.test(parte)) ? "; " : ", "); // ";" evita confundir com a virgula decimal
}

function mostrarRegua(c) {
  const nivel = numero(c.nivel_do_intervalo * 100, 0);
  const intervalo = c.incerteza_disponivel ? c.intervalo_pp : [0, 0];
  const baixo = Math.max(0, c.referencia + intervalo[0] / 100);
  const alto = c.referencia + intervalo[1] / 100;
  const escala = Math.max(0.2, alto, c.estimativa, c.referencia) * 1.15;
  posicionar($("regua-intervalo"), baixo, alto, escala);
  $("regua-intervalo").hidden = !c.incerteza_disponivel;
  $("chave-intervalo").hidden = !c.incerteza_disponivel;
  $("chave-intervalo").textContent = `Faixa de incerteza (${nivel}%)`;
  posicionar($("regua-habitual"), c.referencia, undefined, escala);
  posicionar($("regua-estimativa"), c.estimativa, undefined, escala);
  $("regua-maximo").textContent = pct(escala);
  $("incerteza").textContent = c.incerteza_disponivel
    ? `Diferença entre ${pp(intervalo[0])} e ${pp(intervalo[1])} (${nivel}%)`
    : "Não pôde ser medida com estes dados; sem intervalo.";
}

function mostrarSuporte(c) {
  const s = c.suporte;
  const minimo = c.minimos_de_suporte;
  const linhas = [
    ["Ocasiões contadas", `${inteiro(s.episodios)} no treino, ${inteiro(s.positivos)} com guaraná`],
    ["Campos da contagem", s.campos_considerados.join(" · ") || "todas as ocasiões do treino"]];
  if (s.filtros_nao_cobertos.length) linhas.push(["A contagem não filtra", s.filtros_nao_cobertos.join("; ")]);
  linhas.push(["Mínimo para comparar", `${minimo.episodios} ocasiões e ${minimo.positivos} com guaraná (${minimo.significado})`]);
  pares($("suporte"), linhas);
  const onde = s.campos_considerados.length ? ` · ${s.campos_considerados.join(", ")}` : "";
  $("base-resumo").textContent = `${inteiro(s.episodios)} ocasiões no treino, ${inteiro(s.positivos)} com guaraná${onde}`;
}

// Sem diferenca demonstrada: a justificativa do servidor ja diz que isso nao e igualdade. Evidencia insuficiente: cada motivo
// (pouca amostra, incerteza nao mensuravel, instabilidade) aparece pelo nome, a partir do codigo do servidor, e a nota
// do servidor segue inteira logo abaixo (o texto de um motivo pode conter ";", entao a nota nao e dividida).
function mostrarLeitura(c, explicacao) {
  limpar($("motivos"));
  $("motivos").hidden = c.motivos.length === 0;
  c.motivos.forEach((motivo) => $("motivos").append(el("li", {}, el("strong", { texto: rotulos().motivos[motivo] || motivo }))));
  $("leitura").textContent = explicacao;
  $("leitura").hidden = !explicacao;
}

function mostrarResultado(r) {
  const c = r.comparacao;
  const exploratoria = r.model_metadata.papel_do_modelo_contextual === "exploratorio";
  estado.temResultado = true;
  $("estado-vazio").hidden = true;
  $("conteudo-resultado").hidden = false;
  $("desatualizado").hidden = true;
  $("resultado").classList.remove("esta-desatualizado");
  $("conteudo-resultado").setAttribute("data-faixa", c.faixa);
  $("conteudo-resultado").classList.remove("surge"); // entrada suave a cada resultado novo (sem efeito com movimento reduzido)
  void $("conteudo-resultado").offsetWidth;
  $("conteudo-resultado").classList.add("surge");
  $("cenario").textContent = descreverCenario(estado.consulta);
  // Leitura em linguagem comum primeiro (titulo, linha da medida, uma frase sobre a incerteza); a faixa tecnica fica
  // no detalhe. Tudo vem pronto do servidor.
  $("faixa").textContent = r.leitura.titulo;
  $("linha-medida").textContent = r.leitura.linha;
  mostrarLeitura(c, r.leitura.explicacao);
  $("base-estimativa").textContent = [r.leitura.base, r.leitura.recorte ? `recorte ${r.leitura.recorte}` : ""]
    .filter(Boolean).join(" · ");
  $("exploratorio").hidden = !exploratoria;
  $("exploratorio-texto").textContent = "Trate como hipótese a testar.";
  limpar($("num-estimativa"));
  $("num-estimativa").append(pct(c.estimativa));
  $("num-habitual").textContent = pct(c.referencia);
  limpar($("num-diferenca"));
  $("num-diferenca").append(pp(c.diferenca_pp), el("small", { texto: relativo(c.diferenca_relativa) }));
  mostrarRegua(c);
  mostrarSuporte(c);
  $("categoria").textContent = rotulos().categorias[r.recommendation_category] || r.recommendation_category;
  $("justificativa").textContent = r.recommendation_reason;
  $("proximo-passo").textContent = r.proximo_passo;
  limpar($("fatores"));
  const fatores = r.top_factors.filter((f) => f.direcao !== "neutro"); // so os que movem a estimativa
  if (!r.top_factors.length) $("fatores").append(el("li", { "data-direcao": "neutro", texto: "A estimativa é uma taxa constante: nenhum campo a altera." }));
  else if (!fatores.length) $("fatores").append(el("li", { "data-direcao": "neutro", texto: "Nenhum campo mudou a estimativa de forma perceptível." }));
  fatores.forEach((f) => $("fatores").append(el("li", { "data-direcao": f.direcao, texto: f.texto })));
  limpar($("avisos"));
  [...c.limites.map((limite) => `${maiuscula(limite)}.`), ...r.model_metadata.avisos]
    .forEach((texto) => $("avisos").append(el("li", { texto })));
  mostrarDetalhes(r);
  $("anuncio").textContent = `Resultado: ${r.leitura.titulo}. ${r.leitura.linha}.`;
}

function mostrarDetalhes(r) {
  const m = r.model_metadata;
  const c = r.comparacao;
  const linhas = [
    ["Base", m.fonte], ["Versão", m.versao], ["Alcance da avaliação", m.alcance_da_avaliacao],
    ["Modelo selecionado", m.modelo_selecionado],
    ["Modelo da estimativa por contexto", `${m.modelo_contextual}, ${rotulos().papeis[m.papel_do_modelo_contextual]}`],
    ["Classificação técnica", `${rotulos().faixas[c.faixa] || c.faixa} (pela faixa de 95% da diferença)`],
    ["Habitual", `${c.referencia_nome}: ${pct(c.referencia)}`],
    ["Público", r.contexto.publico], ["Como o dia entra no modelo", r.contexto.dia.como_o_modelo_usa],
    ["Regra de priorização", `${m.regra.id} (${m.regra.versao})`]];
  // Metricas de validacao e identificacao do modelo seguem em model_metadata (API) e na documentacao, fora da tela.
  pares($("detalhes"), linhas);
}

// ---------- dados da empresa ----------
const avisar = (texto) => { $("anuncio-empresa").textContent = texto; };
const ocupado = (botao) => botao.getAttribute("aria-disabled") === "true";

// Botao em andamento: aria-disabled (o foco nao se perde) e rotulo provisorio. Devolve a funcao que o libera.
function ocupar(botao, rotulo) {
  const original = botao.textContent;
  botao.setAttribute("aria-disabled", "true");
  botao.textContent = rotulo;
  return () => { botao.removeAttribute("aria-disabled"); botao.textContent = original; };
}

function liberarTreino() {
  if (estado.liberarTreino) estado.liberarTreino();
  estado.liberarTreino = null;
}

function esconderPassos(ids) {
  const visiveis = ids.filter((id) => !$(id).hidden);
  visiveis.forEach((id) => { $(id).hidden = true; });
  return visiveis.length > 0;
}

function focarPasso(id) { $(id).querySelector("h2").focus(); }

function mostrarFormato() {
  const f = estado.opcoes.formato_empresa;
  limpar($("formato-colunas"));
  Object.entries(f.obrigatorios).forEach(([papel, texto]) => $("formato-colunas").append(
    el("tr", {}, el("td", {}, el("strong", { texto: nome(papel) })), el("td", { texto: `${texto} (obrigatória)` }))));
  Object.entries(f.opcionais).forEach(([papel, texto]) => $("formato-colunas").append(
    el("tr", {}, el("td", { texto: nome(papel) }), el("td", { texto: `${texto} (opcional)` }))));
  preencher($("tipo_de_registro"), Object.entries(f.tipos_de_registro).map(([valor, rotulo]) => ({ valor, rotulo })), "pedido_consumidor");
}

function mostrarPrevia(envio) {
  const previa = envio.previa;
  $("previa-resumo").textContent = `${inteiro(previa.total_de_linhas)} linhas e ${previa.colunas.length} colunas. `
    + (envio.colunas_pessoais.length ? `Colunas que parecem ter dado pessoal e não devem ser usadas: ${envio.colunas_pessoais.join(", ")}.` : "Nenhuma coluna com aparência de dado pessoal.");
  const tabela = $("previa");
  limpar(tabela);
  tabela.append(el("thead", {}, el("tr", {}, ...previa.colunas.map((c) => el("th", { scope: "col", texto: c })))));
  tabela.append(el("tbody", {}, ...previa.linhas.map((linha) => el("tr", {}, ...linha.map((v) => el("td", { texto: v }))))));
  const f = estado.opcoes.formato_empresa;
  limpar($("mapeamento"));
  [...Object.keys(f.obrigatorios), ...Object.keys(f.opcionais)].forEach((papel) => {
    const id = `mapa-${papel}`;
    const select = el("select", { id });
    preencher(select, [{ valor: "", rotulo: "— não usar —" }, ...envio.colunas.map((c) => ({ valor: c, rotulo: c }))],
      envio.sugestao[papel] || "");
    const obrigatoria = papel in f.obrigatorios ? " (obrigatória)" : "";
    $("mapeamento").append(el("div", { classe: "campo" }, el("label", { for: id, texto: `${nome(papel)}${obrigatoria}` }), select));
  });
  limpar($("conhecidos"));
  DECLARAVEIS.forEach((campo) => {
    const id = `conhecido-${campo}`;
    $("conhecidos").append(el("label", { for: id }, el("input", { type: "checkbox", id }), nome(campo)));
  });
  $("passo-colunas").hidden = false;
}

async function enviarArquivo() {
  const arquivo = $("arquivo").files[0];
  const pedido = ++estado.envios; // respostas de um envio anterior deixam de valer
  $("envio-erro").hidden = true;
  estado.envio = null;
  esconderPassos(PASSOS_SEGUINTES);
  if (!arquivo) return;
  avisar("Lendo o arquivo…");
  try {
    const envio = await api("/api/empresa/envios", { method: "POST", headers: { "Content-Type": "text/csv" }, body: arquivo });
    if (pedido !== estado.envios) return;
    estado.envio = envio;
    mostrarPrevia(envio);
    avisar(`Arquivo lido: ${inteiro(envio.previa.total_de_linhas)} linhas. Passo 2, prévia e colunas, disponível.`);
  } catch (erro) {
    if (pedido !== estado.envios) return;
    avisar("");
    mostrarMensagem($("envio-erro"), `O arquivo não pôde ser usado: ${erro.message}`);
  }
}

function pedidoDaEmpresa() {
  const f = estado.opcoes.formato_empresa;
  const mapeamento = {};
  [...Object.keys(f.obrigatorios), ...Object.keys(f.opcionais)].forEach((papel) => {
    const valor = $(`mapa-${papel}`).value;
    if (valor) mapeamento[papel] = valor;
  });
  const conhecidos = {};
  DECLARAVEIS.forEach((campo) => { conhecidos[campo] = $(`conhecido-${campo}`).checked; });
  return { envio: estado.envio.id, mapeamento,
    declaracao: { tipo_de_registro: $("tipo_de_registro").value, conhecidos_antes: conhecidos } };
}

function lista(itens, classe) { return el("ul", { classe: classe || "problemas" }, ...itens); }
function resumo(linhas) {
  const dl = el("dl", { classe: "pares" });
  pares(dl, linhas);
  return dl;
}

function mostrarDiagnostico(d) {
  const alvo = d.alvo;
  const blocos = [resumo([
    ["Cada linha", d.unidade.descricao],
    ["Ocasiões com pizza", `${inteiro(alvo.ocasioes_com_pizza)} (${inteiro(alvo.com_guarana)} com guaraná)`],
    ["Período coberto", d.periodo_coberto ? `${d.periodo_coberto.inicio.slice(0, 10)} a ${d.periodo_coberto.fim.slice(0, 10)}` : "—"],
    ["Variáveis usadas", d.variaveis.usadas.map((v) => nome(v.campo)).join(", ")],
    ["Fora do modelo", d.variaveis.excluidas.map((v) => `${nome(v.campo)} (${v.motivo})`).join("; ") || "nenhuma"]])];
  if (d.problemas.length) {
    blocos.push(el("h3", { texto: d.pode_treinar ? "Avisos" : "O que impede o treino" }),
      lista(d.problemas.map((p) => el("li", { classe: p.bloqueia ? "problema-bloqueia" : "",
        texto: `${p.bloqueia ? "Impede o treino" : "Aviso"}: ${p.mensagem}` }))));
  }
  if (d.plano) {
    const p = d.plano;
    blocos.push(el("h3", { texto: "Plano do experimento (registrado antes do treino)" }), resumo([
      ["Validação", `${VALIDACOES[p.validacao.tipo]}. ${p.validacao.limitacao}`],
      ["Habitual", p.referencia.definicao],
      ["Modelos comparados", p.candidatos.map(nome).join(", ")],
      ["Escolha", p.metrica_de_escolha]]));
  }
  limpar($("diagnostico"));
  $("diagnostico").append(...blocos);
  $("treinar").hidden = !d.pode_treinar;
  $("passo-diagnostico").hidden = false;
}

async function diagnosticar() {
  const botao = $("diagnosticar");
  if (ocupado(botao)) return;
  const liberar = ocupar(botao, "Verificando…");
  try {
    const d = await enviarJson("/api/empresa/diagnostico", pedidoDaEmpresa());
    mostrarDiagnostico(d);
    avisar(d.pode_treinar ? "Diagnóstico pronto: o arquivo pode ser usado no treino."
      : "O diagnóstico encontrou problemas que impedem o treino.");
  } catch (erro) {
    limpar($("diagnostico"));
    $("diagnostico").append(el("p", { classe: "caixa caixa-erro", role: "alert", texto: erro.message }));
    $("passo-diagnostico").hidden = false;
    $("treinar").hidden = true;
  } finally {
    liberar();
  }
  focarPasso("passo-diagnostico");
}

// Atualiza a lista no lugar (so o que mudou): quem esta lendo as etapas nao perde a posicao a cada consulta.
function mostrarEtapas(estadoDoTreino) {
  const ul = $("etapas");
  estadoDoTreino.etapas.forEach((etapa, i) => {
    if (!ul.children[i]) ul.append(el("li", {}, el("span", { classe: "etapa-situacao" }), el("span")));
    const [situacao, descricao] = ul.children[i].children;
    if (situacao.getAttribute("data-situacao") !== etapa.situacao) {
      situacao.setAttribute("data-situacao", etapa.situacao);
      situacao.textContent = SITUACOES[etapa.situacao];
    }
    if (descricao.textContent !== etapa.descricao) descricao.textContent = etapa.descricao;
  });
  while (ul.children.length > estadoDoTreino.etapas.length) ul.lastChild.remove();
}

function mostrarTreino(r) {
  limpar($("treino-resultado"));
  const s = r.selecao;
  const final = r.teste_final;
  const cabecalho = ["Modelo", "Log-loss (validação)", "AUC (validação)", "Log-loss (período final)", "AUC (período final)"];
  const formatar = (v, casas) => (v === null || v === undefined ? "—" : numero(v, casas));
  const linhas = Object.keys(r.candidatos).map((modelo) => {
    const v = r.candidatos[modelo].metricas;
    const t = final.disponivel ? final.metricas[modelo] : {};
    const rotulo = modelo === s.selecionado ? `${nome(modelo)} (selecionado)` : nome(modelo);
    return el("tr", {}, el("td", { texto: rotulo }), el("td", { texto: formatar(v.log_loss, 4) }), el("td", { texto: formatar(v.auc, 3) }),
      el("td", { texto: formatar(t.log_loss, 4) }), el("td", { texto: formatar(t.auc, 3) }));
  });
  $("treino-resultado").append(
    el("p", { texto: `Selecionado: ${nome(s.selecionado)}. Estimativa por contexto: ${nome(r.modelo_contextual.nome)}, ${rotulos().papeis[r.modelo_contextual.papel]}. A escolha usou só os dados mais antigos do treino; o período final apenas confirma.` }),
    el("details", { classe: "detalhes" }, el("summary", { texto: "Comparação técnica dos modelos" }),
      el("div", { classe: "rolagem-tabela", tabindex: "0", role: "group", "aria-label": "Comparação dos modelos, com rolagem horizontal" },
        el("table", { classe: "tabela tabela-compacta" },
          el("thead", {}, el("tr", {}, ...cabecalho.map((c) => el("th", { scope: "col", texto: c })))), el("tbody", {}, ...linhas)))),
    el("p", { texto: r.incerteza.disponivel ? `Incerteza medida com ${r.incerteza.blocos} blocos (${r.incerteza.unidade}).`
      : "A estrutura dos dados não permite medir a incerteza: os cenários aparecerão como evidência insuficiente." }));
  $("disponibilizar").hidden = false;
}

function falhaDoTreino(mensagem) {
  liberarTreino();
  $("treino-resultado").replaceChildren(el("p", { classe: "caixa caixa-erro", role: "alert", texto: mensagem }));
}

async function acompanhar(experimento, consulta = 0, falhas = 0) {
  if (experimento !== estado.experimento) return; // outro treino comecou: esta consulta deixa de valer
  let r;
  try {
    r = await api(`/api/empresa/treinos/${experimento}`);
  } catch (erro) {
    if (experimento !== estado.experimento) return;
    if (erro.tipo === "conexao" && falhas < MAXIMO_FALHAS_SEGUIDAS) {
      setTimeout(() => acompanhar(experimento, consulta + 1, falhas + 1), INTERVALO_DO_TREINO);
    } else {
      falhaDoTreino(erro.message);
    }
    return;
  }
  if (experimento !== estado.experimento) return;
  mostrarEtapas(r.estado);
  if (r.estado.situacao === "em_andamento") {
    if (consulta >= MAXIMO_CONSULTAS_DO_TREINO) falhaDoTreino("O treino passou de 30 minutos. Confira o terminal do servidor.");
    else setTimeout(() => acompanhar(experimento, consulta + 1), INTERVALO_DO_TREINO);
    return;
  }
  if (r.estado.situacao === "falhou") {
    falhaDoTreino(`O treino falhou: ${r.estado.erro}`);
    return;
  }
  mostrarTreino(r.resultado);
  liberarTreino();
  avisar("Treino concluído. Compare os modelos e decida se quer usar o selecionado.");
}

async function treinar() {
  const botao = $("treinar");
  if (ocupado(botao)) return;
  estado.liberarTreino = ocupar(botao, "Treinando…"); // liberado quando o treino termina, falha ou nao comeca
  $("diagnostico").querySelectorAll(".erro-do-treino").forEach((no) => no.remove());
  try {
    const inicio = await enviarJson("/api/empresa/treinos", pedidoDaEmpresa());
    estado.experimento = inicio.id;
    $("passo-treino").hidden = false;
    $("disponibilizar").hidden = true;
    $("disponibilizado").hidden = true;
    $("disponibilizar-erro").hidden = true;
    limpar($("etapas"));
    limpar($("treino-resultado"));
    avisar("Treino iniciado. As etapas aparecem no passo 4.");
    focarPasso("passo-treino");
    acompanhar(inicio.id);
  } catch (erro) {
    liberarTreino();
    $("diagnostico").append(el("p", { classe: "caixa caixa-erro erro-do-treino", role: "alert", texto: erro.message }));
  }
}

async function disponibilizar() {
  const botao = $("disponibilizar");
  if (ocupado(botao)) return;
  const liberar = ocupar(botao, "Disponibilizando…");
  $("disponibilizar-erro").hidden = true;
  try {
    await enviarJson(`/api/empresa/treinos/${estado.experimento}/disponibilizar`, {});
    await carregarOpcoes();
    aplicarBase();
    marcarDesatualizado();
    mostrarMensagem($("disponibilizado"), "Modelo disponível. Em “Avaliar uma ocasião”, escolha “Modelo da empresa”.");
    avisar("Modelo disponível para avaliar ocasiões.");
  } catch (erro) {
    $("disponibilizado").hidden = true;
    mostrarMensagem($("disponibilizar-erro"), erro.message);
  } finally {
    liberar();
  }
}

// ---------- inicio ----------
async function carregarOpcoes() {
  const o = await api("/api/opcoes");
  estado.opcoes = o;
  const radio = $("base-empresa");
  const valido = o.empresa && !o.empresa.indisponivel ? o.empresa : null;
  $("base-empresa-nota").textContent = valido
    ? "Treinado com o arquivo enviado"
    : (o.empresa && o.empresa.motivo) || "Disponível depois de treinar um modelo com seus dados";
  o.empresa = valido;
  radio.disabled = !valido;
  if (!valido && base() === "empresa") $("base-historica").checked = true;
  preencher($("canal"), valido ? valido.canais.map((c) => ({ valor: c, rotulo: c })) : []);
  return o;
}

function ligarFormularios() {
  $("formulario").addEventListener("submit", avaliar);
  $("formulario").addEventListener("change", (e) => {
    if (e.target.getAttribute("aria-invalid")) {
      e.target.removeAttribute("aria-invalid");
      if (e.target.getAttribute("aria-describedby") === "resumo-erros") e.target.removeAttribute("aria-describedby");
    }
    if (e.target.name !== "base") { marcarDesatualizado(); return; }
    const avisoDesatualizado = estado.temResultado && $("desatualizado").hidden ? `${$("desatualizado").textContent} ` : "";
    aplicarBase(); // um unico anuncio: o aviso de desatualizado (se houver) e os campos que mudaram
    $("anuncio").textContent = avisoDesatualizado + (base() === "empresa"
      ? "Campos do modelo da empresa exibidos: canal, preço, temperatura ou promoção, conforme o treino."
      : "Campos dos dados públicos exibidos, com o público opcional.");
  });
  $("arquivo").addEventListener("change", enviarArquivo);
  $("diagnosticar").addEventListener("click", diagnosticar);
  $("passo-colunas").addEventListener("change", () => { // mapeamento mudou: diagnostico e plano anteriores deixam de valer
    if (esconderPassos(["passo-diagnostico", "passo-treino"])) avisar("Mapeamento alterado: verifique o arquivo de novo.");
  });
  $("treinar").addEventListener("click", treinar);
  $("disponibilizar").addEventListener("click", disponibilizar);
}

async function iniciar() {
  ligarCabecalho();
  ligarNavegacao();
  ligarAmbiente();
  carregarMarca();
  try {
    const o = await carregarOpcoes();
    preencher($("dia_semana"), o.dias, 5);
    preencher($("periodo"), o.periodos.map((p) => ({ valor: p.valor, rotulo: maiuscula(p.rotulo) })), "noite");
    preencher($("grande_regiao"), o.regioes, o.regioes[2].valor);
    preencher($("area"), o.areas, o.areas[0].valor);
    preencher($("faixa_etaria"), o.faixas_etarias, "todos");
    preencher($("sexo"), o.sexos, "todos");
    preencher($("escolaridade"), o.escolaridades, "todos");
    mostrarFormato();
    aplicarBase();
    ligarFormularios();
  } catch (erro) {
    const mensagem = `A ferramenta não pôde ser carregada: ${erro.message} Recarregue a página.`;
    mostrarMensagem($("falha"), mensagem);
    mostrarMensagem($("envio-erro"), mensagem);
    $("avaliar-botao").disabled = true;
    $("arquivo").disabled = true;
  }
}

document.addEventListener("DOMContentLoaded", iniciar);
