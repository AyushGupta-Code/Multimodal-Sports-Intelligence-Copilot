"""HTML shell for the local testing UI."""

from __future__ import annotations


def render_home(default_use_llm: bool) -> str:
    """Return the local testing page."""

    return f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Local Soccer Intelligence Copilot</title>
      <link rel="preconnect" href="https://fonts.googleapis.com" />
      <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
      <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet" />
      <link rel="stylesheet" href="/static/styles.css" />
    </head>
    <body data-default-use-llm="{str(default_use_llm).lower()}">
      <main class="app-shell">
        <aside class="icon-rail" aria-label="Primary navigation">
          <button class="rail-button active" type="button" aria-label="Chat">💬</button>
          <button class="rail-button" type="button" aria-label="Settings">⚙️</button>
          <button class="rail-button" type="button" aria-label="Prompts">✏️</button>
          <button class="rail-button" type="button" aria-label="Files">📄</button>
        </aside>

        <aside class="chat-sidebar">
          <div class="workspace">Select workspace...</div>
          <button id="resetButton" class="new-chat" type="button">+ New Chat</button>
          <label class="search-wrap" for="querySearch">
            <span class="sr-only">Search chats</span>
            <input id="querySearch" type="text" placeholder="Search chats..." disabled />
          </label>
          <p class="no-chats">No chats.</p>

          <details class="quick-settings" open>
            <summary>Quick Settings</summary>
            <p><span id="llmBadge" class="status-badge">Checking Ollama</span></p>
            <p><strong>Trace mode:</strong> <span id="traceMode">Awaiting response</span></p>
            <p id="failureReason" class="notice-text">No LLM issues reported.</p>
            <p id="sessionMeta" class="session-chip mono-text">Starting local session...</p>
            <h4>Evidence</h4>
            <pre id="evidence" class="mono-text"></pre>
            <h4>Trace</h4>
            <pre id="trace" class="mono-text"></pre>
          </details>
        </aside>

        <section class="chat-stage">
          <header class="topbar">
            <p>Quick Settings</p>
            <h2>Local Match Model</h2>
          </header>

          <section id="chatLog" class="chat-log">
            <article class="welcome-card" id="emptyState">
              <div class="logo-badge">UI</div>
              <h1>Soccer Chatbot UI</h1>
              <p>Grounded answers with local match evidence.</p>
            </article>
          </section>

          <form id="composer" class="composer">
            <label class="sr-only" for="queryText">Message</label>
            <div class="composer-shell">
              <textarea id="queryText" name="message" placeholder="Send a message..." rows="1"></textarea>
              <button id="sendButton" type="submit">➤</button>
            </div>
          </form>
        </section>
      </main>
      <script src="/static/app.js"></script>
    </body>
    </html>
    """
