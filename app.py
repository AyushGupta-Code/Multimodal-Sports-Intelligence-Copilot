"""FastAPI app for the Local Soccer Intelligence Copilot MVP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from pipeline import build_index, build_sequences, load_event_files, normalize_events, run_query
from schemas import IngestRequest, QueryRequest

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
    """Load the default local dataset and build sequences when needed."""

    # Read the local StatsBomb files from the project folder so the UI does not need a path input.
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

    # Prepare the app on demand so the user can go straight from opening the page to asking a question.
    if not STATE["events"]:
        _load_default_data()
    if STATE["index"] is None:
        try:
            STATE["index"] = build_index(STATE["sequences"])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        STATE["trace"]["sequences_indexed"] = len(STATE["sequences"])


def _require_ingested() -> None:
    """Raise an API error when ingestion has not happened yet."""

    # Stop the request early if no event data has been loaded yet.
    if not STATE["events"]:
        raise HTTPException(status_code=400, detail="No data loaded yet. Run /ingest first.")


def _require_index() -> None:
    """Raise an API error when the retrieval index has not been built yet."""

    # Stop the request early if the search index has not been created yet.
    if STATE["index"] is None:
        raise HTTPException(status_code=400, detail="Index not built yet. Run /build-index first.")


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    """Serve a minimal built-in HTML UI for local ingestion and querying."""

    # Return a very small HTML page directly from FastAPI so the project stays simple.
    return """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Local Soccer Intelligence Copilot</title>
    </head>
    <body style="font-family: sans-serif; max-width: 900px; margin: 40px auto; line-height: 1.5;">
      <h1>Local Soccer Intelligence Copilot</h1>
      <p>Ask a grounded question over the local StatsBomb data stored in the project.</p>
      <label>Query</label><br />
      <textarea id="queryText" style="width: 100%; min-height: 100px; padding: 8px;" placeholder="How does this team create chances?"></textarea>
      <p><button onclick="runQuery()">Run Query</button></p>
      <h2>Answer</h2>
      <pre id="answer" style="white-space: pre-wrap; background: #f5f5f5; padding: 12px;"></pre>
      <h2>Evidence</h2>
      <pre id="evidence" style="white-space: pre-wrap; background: #f5f5f5; padding: 12px;"></pre>
      <h2>Trace</h2>
      <pre id="trace" style="white-space: pre-wrap; background: #f5f5f5; padding: 12px;"></pre>
      <script>
        async function runQuery() {
          const res = await fetch('/query', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({query: document.getElementById('queryText').value})
          });
          const data = await res.json();
          render(data);
        }
        function render(data) {
          document.getElementById('answer').textContent = data.answer || data.detail || '';
          document.getElementById('evidence').textContent = JSON.stringify(data.evidence || data, null, 2);
          document.getElementById('trace').textContent = JSON.stringify(data.trace || {}, null, 2);
        }
      </script>
    </body>
    </html>
    """


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict[str, Any]:
    """Load local event files, normalize them, and build attacking sequences."""

    # Clear the old index after loading new data so search results stay correct.
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
    """Build the in-memory retrieval index from the currently loaded sequences."""

    # Build the search index in a separate step after data has been ingested.
    _require_ingested()
    try:
        STATE["index"] = build_index(STATE["sequences"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    STATE["trace"]["sequences_indexed"] = len(STATE["sequences"])
    return {
        "answer": f'Indexed {len(STATE["sequences"])} sequences with TF-IDF.',
        "evidence": [],
        "trace": STATE["trace"],
    }


@app.post("/query")
def query(request: QueryRequest) -> dict[str, Any]:
    """Run a grounded retrieval query over the current in-memory index."""

    # Load the local data and build the index automatically before running the search.
    _auto_prepare()
    try:
        response = run_query(
            query=request.query,
            index_data=STATE["index"],
            trace=STATE["trace"],
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return response.model_dump()
