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
      <link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet" />
      <link rel="stylesheet" href="/static/styles.css" />
    </head>
    <body data-default-use-llm="{str(default_use_llm).lower()}">
      <div class="ambient ambient-a"></div>
      <div class="ambient ambient-b"></div>
      <main class="shell">
        <aside class="nav-rail">
          <div class="brand-block">
            <div class="brand-mark">MS</div>
            <div>
              <p class="brand-kicker">Match Studio</p>
              <h1>Soccer Copilot</h1>
            </div>
          </div>
          <section class="nav-panel">
            <p class="panel-label">Workspace</p>
            <button class="nav-item nav-item-active" type="button">Live Match Chat</button>
            <button class="nav-item" type="button">Grounded Retrieval</button>
            <button class="nav-item" type="button">Evidence Review</button>
          </section>
          <section class="nav-panel">
            <p class="panel-label">Prompt Starters</p>
            <button class="suggestion-chip" type="button" data-prompt="Who won this match?">Who won this match?</button>
            <button class="suggestion-chip" type="button" data-prompt="How did Barcelona create chances?">How did Barcelona create chances?</button>
            <button class="suggestion-chip" type="button" data-prompt="What about Messi in the second half?">What about Messi in the second half?</button>
            <button class="suggestion-chip" type="button" data-prompt="Show the strongest goal evidence.">Show the strongest goal evidence.</button>
          </section>
          <section class="nav-panel status-panel">
            <p class="panel-label">Model</p>
            <span id="llmBadge" class="status-badge">Checking Ollama</span>
            <p class="status-copy">Local mode with grounded evidence and session memory.</p>
          </section>
        </aside>
        <section class="chat-stage">
          <header class="topbar">
            <div>
              <p class="eyebrow">Local Match Intelligence</p>
              <h2>Ask grounded tactical questions with a Chatbot UI-inspired workspace.</h2>
            </div>
            <div class="topbar-actions">
              <p id="sessionMeta" class="session-chip mono-text">Starting local session...</p>
              <button id="resetButton" class="secondary-button" type="button">Reset Session</button>
            </div>
          </header>
          <section class="hero-card">
            <div>
              <p class="hero-kicker">Realtime Analysis</p>
              <h3>Test the copilot locally with Ollama and retrieved match evidence.</h3>
            </div>
            <p class="hero-copy">The UI stays local, the answers stay grounded, and the trace is always visible while you iterate.</p>
          </section>
          <section id="chatLog" class="chat-log">
            <article class="welcome-card" id="emptyState">
              <span class="welcome-badge">Ready</span>
              <h3>Start with a concrete match question</h3>
              <p>Try “Who won this match?”, “How did Barcelona create chances?”, or “What about Messi in the second half?”.</p>
            </article>
          </section>
          <form id="composer" class="composer">
            <label class="composer-label" for="queryText">Ask the local copilot</label>
            <div class="composer-shell">
              <textarea id="queryText" name="message" placeholder="Ask about tactics, players, scoreline, or a follow-up." rows="1"></textarea>
              <div class="composer-actions">
                <span class="hint mono-text">Enter to send</span>
                <button id="sendButton" type="submit">Send</button>
              </div>
            </div>
          </form>
        </section>
        <aside class="inspector-rail">
          <section class="panel">
            <div class="panel-head">
              <h2>Trace Status</h2>
              <span id="traceMode" class="tiny-badge">Awaiting response</span>
            </div>
            <p id="failureReason" class="notice-text">No LLM issues reported.</p>
          </section>
          <section class="panel">
            <h2>Latest Evidence</h2>
            <pre id="evidence" class="mono-text"></pre>
          </section>
          <section class="panel">
            <h2>Trace</h2>
            <pre id="trace" class="mono-text"></pre>
          </section>
        </aside>
      </main>
      <script src="/static/app.js"></script>
    </body>
    </html>
    """
