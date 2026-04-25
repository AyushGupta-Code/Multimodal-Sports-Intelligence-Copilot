"""FastAPI app for the local match chatbot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from pipeline import SESSION_STORE, build_index, build_sequences, load_event_files, normalize_events, run_chat_turn, run_query
from schemas import ChatRequest, IngestRequest, QueryRequest, SessionResponse

app = FastAPI(title="Local Soccer Intelligence Copilot")
DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "open-data" / "data" / "events"

STATE: dict[str, Any] = {
    "raw_events": [],
    "events": [],
    "sequences": [],
    "index": None,
    "trace": {
        "files_ingested": 0,
        "sequences_built": 0,
        "sequences_indexed": 0,
    },
}


def _load_default_data() -> None:
    """Load the default local dataset and build sequences."""

    try:
        raw_events, file_count = load_event_files(str(DEFAULT_DATASET_PATH))
        events = normalize_events(raw_events)
        sequences = build_sequences(events)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    STATE["raw_events"] = raw_events
    STATE["events"] = events
    STATE["sequences"] = sequences
    STATE["index"] = None
    STATE["trace"] = {
        "files_ingested": file_count,
        "sequences_built": len(sequences),
        "sequences_indexed": 0,
    }


def _auto_prepare() -> None:
    """Load data and build the search index automatically when needed."""

    if not STATE["events"]:
        _load_default_data()
    if STATE["index"] is None:
        try:
            STATE["index"] = build_index(STATE["events"], STATE["sequences"])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        STATE["trace"]["sequences_indexed"] = len(STATE["sequences"])


def _require_ingested() -> None:
    """Raise an API error when ingestion has not happened yet."""

    if not STATE["events"]:
        raise HTTPException(status_code=400, detail="No data loaded yet. Run /ingest first.")


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    """Serve a chat-style built-in UI."""

    return """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Local Soccer Intelligence Copilot</title>
      <style>
        :root {
          --bg: #f3efe7;
          --panel: #fbf8f2;
          --ink: #1d1d1b;
          --muted: #6c685f;
          --line: #d8d0c3;
          --user: #1d5c4b;
          --assistant: #ffffff;
          --accent: #b35c2e;
        }
        * { box-sizing: border-box; }
        body {
          margin: 0;
          font-family: Georgia, "Times New Roman", serif;
          background: radial-gradient(circle at top, #fffaf0 0%, var(--bg) 55%);
          color: var(--ink);
        }
        .shell {
          max-width: 1100px;
          margin: 0 auto;
          min-height: 100vh;
          display: grid;
          grid-template-columns: minmax(0, 1.8fr) 320px;
          gap: 18px;
          padding: 24px;
        }
        .chat-panel, .side-panel {
          background: rgba(251, 248, 242, 0.88);
          border: 1px solid var(--line);
          border-radius: 24px;
          backdrop-filter: blur(8px);
        }
        .chat-panel {
          display: flex;
          flex-direction: column;
          overflow: hidden;
          min-height: calc(100vh - 48px);
        }
        .hero {
          padding: 24px 24px 16px;
          border-bottom: 1px solid var(--line);
          background: linear-gradient(135deg, rgba(179, 92, 46, 0.08), rgba(29, 92, 75, 0.08));
        }
        .hero h1 {
          margin: 0 0 8px;
          font-size: 32px;
          line-height: 1.1;
        }
        .hero p {
          margin: 0;
          color: var(--muted);
          max-width: 70ch;
        }
        .chat-log {
          flex: 1;
          overflow-y: auto;
          padding: 24px;
          display: flex;
          flex-direction: column;
          gap: 18px;
        }
        .empty {
          border: 1px dashed var(--line);
          border-radius: 18px;
          padding: 18px;
          color: var(--muted);
          background: rgba(255,255,255,0.5);
        }
        .message {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        .label {
          font-size: 12px;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: var(--muted);
        }
        .bubble {
          padding: 16px 18px;
          border-radius: 18px;
          border: 1px solid var(--line);
          white-space: pre-wrap;
          line-height: 1.55;
        }
        .message.user { align-items: flex-end; }
        .message.user .bubble {
          background: var(--user);
          color: #f7f4ed;
          border-color: rgba(29, 92, 75, 0.5);
          max-width: 78%;
        }
        .message.assistant .bubble {
          background: var(--assistant);
          max-width: 92%;
        }
        .composer {
          border-top: 1px solid var(--line);
          padding: 18px;
          background: rgba(255,255,255,0.7);
        }
        .composer-box {
          display: flex;
          gap: 12px;
          align-items: flex-end;
        }
        textarea {
          width: 100%;
          min-height: 68px;
          max-height: 220px;
          resize: vertical;
          border: 1px solid var(--line);
          border-radius: 18px;
          padding: 14px 16px;
          font: inherit;
          background: #fffdf8;
          color: var(--ink);
        }
        button {
          border: 0;
          border-radius: 999px;
          padding: 14px 18px;
          background: var(--accent);
          color: #fff9f2;
          font: inherit;
          cursor: pointer;
        }
        button:disabled {
          opacity: 0.6;
          cursor: wait;
        }
        .side-panel {
          padding: 18px;
          display: flex;
          flex-direction: column;
          gap: 16px;
          min-height: calc(100vh - 48px);
        }
        .side-panel h2 {
          margin: 0 0 8px;
          font-size: 16px;
        }
        .card {
          border: 1px solid var(--line);
          border-radius: 16px;
          padding: 14px;
          background: rgba(255,255,255,0.62);
        }
        .scroll-card {
          max-height: 320px;
          overflow-y: auto;
        }
        pre {
          margin: 0;
          white-space: pre-wrap;
          word-break: break-word;
          font-size: 12px;
          line-height: 1.45;
        }
        .meta {
          color: var(--muted);
          font-size: 13px;
        }
        @media (max-width: 900px) {
          .shell {
            grid-template-columns: 1fr;
            padding: 14px;
          }
          .chat-panel, .side-panel {
            min-height: auto;
          }
          .message.user .bubble, .message.assistant .bubble {
            max-width: 100%;
          }
        }
      </style>
    </head>
    <body>
      <div class="shell">
        <section class="chat-panel">
          <div class="hero">
            <h1>Local Soccer Intelligence Copilot</h1>
            <p>Ask grounded questions about the match, then keep going with follow-ups. The chat keeps session context and prefers LLM answers automatically.</p>
          </div>
          <div id="chatLog" class="chat-log">
            <div class="empty" id="emptyState">
              Try: <strong>Who won this match?</strong><br />
              Then follow with: <strong>Who scored?</strong> or <strong>What about Messi in the second half?</strong>
            </div>
          </div>
          <div class="composer">
            <div class="composer-box">
              <textarea id="queryText" placeholder="Message the match chatbot..."></textarea>
              <button id="sendButton" onclick="runChat()">Send</button>
            </div>
          </div>
        </section>
        <aside class="side-panel">
         <div class="card scroll-card">
            <h2>Session</h2>
            <div id="sessionMeta" class="meta">Starting chat session…</div>
          </div>
          <div class="card scroll-card">
            <h2>Latest Evidence</h2>
            <pre id="evidence"></pre>
          </div>
          <div class="card scroll-card">
            <h2>Latest Trace</h2>
            <pre id="trace"></pre>
          </div>
        </aside>
      </div>
      <script>
        let sessionId = null;

        async function ensureSession() {
          if (sessionId) return sessionId;
          const res = await fetch('/sessions', {method: 'POST'});
          const data = await res.json();
          sessionId = data.session_id;
          document.getElementById('sessionMeta').textContent = 'Session: ' + sessionId;
          return sessionId;
        }

        function escapeHtml(text) {
          return String(text)
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;');
        }

        function renderHistory(history) {
          const chatLog = document.getElementById('chatLog');
          const emptyState = document.getElementById('emptyState');
          if (!history || history.length === 0) {
            chatLog.innerHTML = '';
            chatLog.appendChild(emptyState);
            return;
          }
          function answerSource(turn) {
            const trace = turn.trace || {};
            if (trace.use_llm && trace.generation_mode === 'llm' && !trace.llm_fallback) {
              return 'Ollama';
            }
            if (trace.use_llm && trace.llm_fallback) {
              return 'Template fallback';
            }
            return 'Template';
          }
          chatLog.innerHTML = history.map(turn => `
            <div class="message user">
              <div class="label">User</div>
              <div class="bubble">${escapeHtml(turn.user_message)}</div>
            </div>
            <div class="message assistant">
              <div class="label">Assistant · ${escapeHtml(answerSource(turn))}</div>
              <div class="bubble">${escapeHtml(turn.answer)}</div>
            </div>
          `).join('');
          chatLog.scrollTop = chatLog.scrollHeight;
        }

        async function runChat() {
          const textarea = document.getElementById('queryText');
          const button = document.getElementById('sendButton');
          const message = textarea.value.trim();
          if (!message) return;
          button.disabled = true;
          const id = await ensureSession();
          try {
            const res = await fetch('/chat', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({
                session_id: id,
                message: message,
                use_llm: true,
                llm_required: false
              })
            });
            const data = await res.json();
            renderHistory(data.history || []);
            document.getElementById('evidence').textContent = JSON.stringify(data.evidence || data, null, 2);
            document.getElementById('trace').textContent = JSON.stringify(data.trace || {}, null, 2);
            textarea.value = '';
          } finally {
            button.disabled = false;
            textarea.focus();
          }
        }

        document.getElementById('queryText').addEventListener('keydown', function(event) {
          if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            runChat();
          }
        });

        ensureSession();
      </script>
    </body>
    </html>
    """


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict[str, Any]:
    """Load local event files, normalize them, and build sequences."""

    try:
        raw_events, file_count = load_event_files(request.dataset_path)
        events = normalize_events(raw_events)
        sequences = build_sequences(events)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    STATE["raw_events"] = raw_events
    STATE["events"] = events
    STATE["sequences"] = sequences
    STATE["index"] = None
    STATE["trace"] = {
        "files_ingested": file_count,
        "sequences_built": len(sequences),
        "sequences_indexed": 0,
    }
    return {
        "answer": f"Ingested {len(events)} events from {file_count} file(s) and built {len(sequences)} sequences.",
        "evidence": [{"summary": sequence.summary} for sequence in sequences[:5]],
        "trace": STATE["trace"],
    }


@app.post("/build-index")
def build_index_route() -> dict[str, Any]:
    """Build the in-memory retrieval index."""

    _require_ingested()
    try:
        STATE["index"] = build_index(STATE["events"], STATE["sequences"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    STATE["trace"]["sequences_indexed"] = len(STATE["sequences"])
    return {
        "answer": f"Indexed {len(STATE['sequences'])} sequences with TF-IDF.",
        "evidence": [],
        "trace": STATE["trace"],
    }


@app.post("/sessions", response_model=SessionResponse)
def create_session() -> SessionResponse:
    """Create a new chat session."""

    session = SESSION_STORE.create_session()
    return SessionResponse(session_id=session.session_id, history=[])


@app.post("/sessions/{session_id}/reset", response_model=SessionResponse)
def reset_session(session_id: str) -> SessionResponse:
    """Reset an existing chat session."""

    try:
        session = SESSION_STORE.reset_session(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return SessionResponse(session_id=session.session_id, history=[])


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    """Return the stored chat session history."""

    try:
        session = SESSION_STORE.get_session(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"session_id": session_id, "history": [turn.model_dump() for turn in session.turns]}


@app.post("/query")
def query(request: QueryRequest) -> dict[str, Any]:
    """Run a single grounded retrieval query."""

    _auto_prepare()
    try:
        response = run_query(
            query=request.query,
            index_data=STATE["index"],
            trace=STATE["trace"],
            top_k=request.top_k,
            use_llm=request.use_llm,
            llm_required=request.llm_required,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return response.model_dump()


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    """Run one chat turn with session memory."""

    _auto_prepare()
    try:
        session = SESSION_STORE.get_session(request.session_id)
        response = run_chat_turn(
            session=session,
            message=request.message,
            index_data=STATE["index"],
            trace=STATE["trace"],
            top_k=request.top_k,
            use_llm=request.use_llm,
            llm_required=request.llm_required,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return response
