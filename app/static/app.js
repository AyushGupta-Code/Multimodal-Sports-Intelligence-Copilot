let sessionId = null;

const body = document.body;
const defaultUseLlm = body.dataset.defaultUseLlm === "true";
const chatLog = document.getElementById("chatLog");
const emptyState = document.getElementById("emptyState");
const queryText = document.getElementById("queryText");
const sendButton = document.getElementById("sendButton");
const resetButton = document.getElementById("resetButton");
const sessionMeta = document.getElementById("sessionMeta");
const evidencePre = document.getElementById("evidence");
const tracePre = document.getElementById("trace");
const llmBadge = document.getElementById("llmBadge");
const traceMode = document.getElementById("traceMode");
const failureReason = document.getElementById("failureReason");
const promptButtons = Array.from(document.querySelectorAll("[data-prompt]"));

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function autosizeTextarea() {
  queryText.style.height = "auto";
  queryText.style.height = `${Math.min(queryText.scrollHeight, 220)}px`;
}

function answerSource(turn) {
  const trace = turn.trace || {};
  if (trace.use_llm && trace.generation_mode === "llm" && !trace.llm_fallback) {
    return "Ollama";
  }
  if (trace.use_llm && trace.llm_fallback) {
    return "Template Fallback";
  }
  return "Template";
}

function renderHistory(history) {
  if (!history || history.length === 0) {
    chatLog.innerHTML = "";
    chatLog.appendChild(emptyState);
    return;
  }

  chatLog.innerHTML = history.map((turn) => `
    <article class="message user">
      <div class="label">User</div>
      <div class="bubble">${escapeHtml(turn.user_message)}</div>
    </article>
    <article class="message assistant">
      <div class="label">Assistant · ${escapeHtml(answerSource(turn))}</div>
      <div class="bubble">${escapeHtml(turn.answer)}</div>
    </article>
  `).join("");
  chatLog.scrollTop = chatLog.scrollHeight;
}

function renderTraceState(trace) {
  if (!trace) {
    traceMode.textContent = "Awaiting response";
    failureReason.textContent = "No LLM issues reported.";
    return;
  }

  if (trace.generation_mode === "llm" && !trace.llm_fallback) {
    traceMode.textContent = "Ollama response";
    failureReason.textContent = "LLM response validated successfully.";
    return;
  }

  if (trace.llm_fallback) {
    traceMode.textContent = "Template fallback";
    failureReason.textContent = trace.llm_failure_reason || "Ollama failed and the app fell back to the template path.";
    return;
  }

  traceMode.textContent = "Template response";
  failureReason.textContent = trace.use_llm
    ? "The chat stayed on the deterministic template path."
    : "OLLAMA_URL was not enabled when the server started.";
}

async function ensureSession() {
  if (sessionId) {
    return sessionId;
  }
  const response = await fetch("/sessions", { method: "POST" });
  const data = await response.json();
  sessionId = data.session_id;
  sessionMeta.textContent = `Session: ${sessionId}`;
  llmBadge.textContent = defaultUseLlm ? "Ollama enabled" : "Template mode";
  return sessionId;
}

async function resetSession() {
  if (!sessionId) {
    await ensureSession();
    return;
  }
  resetButton.disabled = true;
  try {
    await fetch(`/sessions/${sessionId}/reset`, { method: "POST" });
    renderHistory([]);
    evidencePre.textContent = "";
    tracePre.textContent = "";
    renderTraceState(null);
    sessionMeta.textContent = `Session reset: ${sessionId}`;
    queryText.focus();
  } finally {
    resetButton.disabled = false;
  }
}

async function runChat(event) {
  event.preventDefault();
  const message = queryText.value.trim();
  if (!message) {
    return;
  }

  sendButton.disabled = true;
  const currentSessionId = await ensureSession();
  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: currentSessionId,
        message,
        use_llm: defaultUseLlm,
        llm_required: false
      })
    });
    const data = await response.json();
    renderHistory(data.history || []);
    evidencePre.textContent = JSON.stringify(data.evidence || [], null, 2);
    tracePre.textContent = JSON.stringify(data.trace || {}, null, 2);
    renderTraceState(data.trace || null);
    queryText.value = "";
    autosizeTextarea();
    queryText.focus();
  } finally {
    sendButton.disabled = false;
  }
}

queryText.addEventListener("input", autosizeTextarea);
queryText.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    runChat(event);
  }
});

document.getElementById("composer").addEventListener("submit", runChat);
resetButton.addEventListener("click", resetSession);
promptButtons.forEach((button) => {
  button.addEventListener("click", () => {
    queryText.value = button.dataset.prompt || "";
    autosizeTextarea();
    queryText.focus();
  });
});

autosizeTextarea();
ensureSession();
renderTraceState(null);
