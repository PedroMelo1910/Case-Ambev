"use strict";

const CHAVE_CONVERSA = "conversa-personas-v4";
const conversa = { estado: {}, falas: [], pedido: null };
const PERFIS = { pai_de_familia: "Pai de família", jovem_sozinho: "Jovem que mora sozinho" };

function perfilSelecionado() {
  return document.querySelector('input[name="persona"]:checked').value;
}

function gravarConversa() {
  try {
    sessionStorage.setItem(CHAVE_CONVERSA, JSON.stringify({
      persona: perfilSelecionado(), estado: conversa.estado, falas: conversa.falas,
    }));
  } catch { /* a conversa segue nesta visita */ }
}

function retomarConversa() {
  try {
    const salva = JSON.parse(sessionStorage.getItem(CHAVE_CONVERSA));
    if (!salva || !Object.hasOwn(PERFIS, salva.persona) || !salva.estado || typeof salva.estado !== "object"
        || !Array.isArray(salva.falas) || !salva.falas.every((f) => f
          && typeof f.usuario === "boolean" && Array.isArray(f.textos) && f.textos.every((t) => typeof t === "string"))) return;
    document.querySelector(`input[name="persona"][value="${salva.persona}"]`).checked = true;
    conversa.estado = salva.estado;
    salva.falas.forEach(acrescentarFala);
  } catch { /* um histórico indisponível não impede uma conversa nova */ }
}

function acrescentarFala(fala) {
  const item = el("div", { classe: fala.usuario ? "fala fala-usuario" : "fala fala-assistente" },
    el("p", { classe: "fala-autor", texto: fala.usuario ? "Você" : PERFIS[perfilSelecionado()] }),
    ...fala.textos.map((texto) => el("p", { texto })));
  $("conversa").append(item);
  conversa.falas.push(fala);
  $("conversa-sugestoes").hidden = true;
  return item;
}

function marcarEnvio(ativo) {
  $("enviar").disabled = ativo;
  $("enviar").textContent = ativo ? "Enviando…" : "Enviar";
  $("conversa-status").textContent = ativo ? "A persona está respondendo." : "";
}

function reiniciarConversa() {
  conversa.pedido?.abort();
  conversa.pedido = null;
  conversa.estado = {};
  conversa.falas = [];
  limpar($("conversa"));
  $("conversa-sugestoes").hidden = false;
  $("conversa-erro").hidden = true;
  $("mensagem").value = "";
  marcarEnvio(false);
  gravarConversa();
}

async function enviarMensagem(texto) {
  if (conversa.pedido || !texto) return;
  const pedido = new AbortController();
  conversa.pedido = pedido;
  $("conversa-erro").hidden = true;
  marcarEnvio(true);
  const fala = acrescentarFala({ usuario: true, textos: [texto] });
  fala.scrollIntoView({ block: "nearest", behavior: "instant" });
  try {
    const resposta = await api("/api/conversa", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ aba: "personas", persona: perfilSelecionado(), mensagem: texto, estado: conversa.estado }),
      signal: pedido.signal,
    });
    if (conversa.pedido !== pedido) return;
    conversa.estado = resposta.estado;
    const item = acrescentarFala({ usuario: false, textos: resposta.falas });
    gravarConversa();
    item.scrollIntoView({ block: "nearest", behavior: rolagem() });
  } catch (erro) {
    if (conversa.pedido !== pedido) return;
    if (erro.tipo === "entrada_invalida" && Object.keys(conversa.estado).length) {
      reiniciarConversa();
    } else {
      conversa.falas.pop();
      fala.remove();
      $("conversa-sugestoes").hidden = conversa.falas.length > 0;
    }
    if (!$("mensagem").value) $("mensagem").value = texto;
    $("conversa-erro").textContent = "A mensagem não foi enviada. Tente de novo.";
    $("conversa-erro").hidden = false;
  } finally {
    if (conversa.pedido === pedido) {
      conversa.pedido = null;
      marcarEnvio(false);
    }
  }
}

document.querySelectorAll('input[name="persona"]').forEach((radio) => radio.addEventListener("change", reiniciarConversa));
$("nova-conversa").addEventListener("click", () => {
  reiniciarConversa();
  $("mensagem").focus();
});
$("conversa-formulario").addEventListener("submit", (evento) => {
  evento.preventDefault();
  const campo = $("mensagem");
  if (!campo.value.trim() || conversa.pedido) { campo.focus(); return; }
  const texto = campo.value.trim();
  campo.value = "";
  campo.focus();
  enviarMensagem(texto);
});
$("conversa-sugestoes").addEventListener("click", (evento) => {
  const sugestao = evento.target.closest(".sugestao");
  if (sugestao) {
    $("mensagem").focus();
    enviarMensagem(sugestao.textContent);
  }
});
retomarConversa();
