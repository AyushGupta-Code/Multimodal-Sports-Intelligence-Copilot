# Local Soccer Intelligence Copilot

Local-first FastAPI app for ingesting StatsBomb-style soccer event JSON, building attacking-sequence retrieval, and testing a grounded chatbot UI against either deterministic templates or a local Ollama model.

## Current Structure

```text
app/
  main.py           FastAPI entrypoint and routes
  state.py          Shared dataset/index state and auto-prepare helpers
  api/schemas.py    Request and response DTOs
  models/domain.py  Internal event, fact, and chat models
  services/         Data loading, retrieval, generation, and chat logic
  ui/page.py        HTML shell for the local test UI
  static/           Frontend JS and CSS served by FastAPI
data/               Local dataset folder used by default
tests/              Smoke tests for API, state, and chat flow
```

## What It Does

- Loads one local JSON file or a folder of JSON files.
- Normalizes a small subset of StatsBomb-style event fields.
- Groups events into simple team-possession attacking sequences.
- Builds deterministic sequence summaries for retrieval.
- Uses an in-memory TF-IDF index for local search.
- Automatically ingests data and builds the index when you start chatting.
- Supports local Ollama rewriting while keeping retrieval and evidence grounded.

## Dependencies

- Python 3.10+
- `fastapi`
- `uvicorn`
- `scikit-learn`
- `pydantic`

## Install

```bash
conda create -n local-soccer-copilot python=3.10
conda activate local-soccer-copilot
python -m pip install -r requirements.txt
```

## Optional Local LLM With Ollama

The app can use a local Ollama model for answer phrasing while leaving retrieval unchanged.

### 1. Install and run Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```

### 2. Pull a model

```bash
ollama pull gemma4:26b
```

### 3. Point the app at Ollama

```bash
export OLLAMA_URL=http://127.0.0.1:11434
export OLLAMA_MODEL=gemma4:26b
```

If `OLLAMA_URL` is unset, the UI and API stay on the deterministic template-answer path.

## Local Dataset

Create the local data folder and download one sample StatsBomb event file:

```bash
mkdir -p data/events
python3 -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/15946.json', 'data/events/15946.json')"
```

The app loads data in this order:

- `DATASET_PATH` environment variable, if set
- `data/events`

## Run Locally

```bash
python -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

## Local UI

The test UI is now split out of `app/main.py`:

- `app/ui/page.py` renders the page shell
- `app/static/styles.css` controls layout and visuals
- `app/static/app.js` handles sessions, chat turns, reset, evidence, and trace panes

This is intended for local iteration only. There is no deployment-specific logic in the current app flow.

## API

### `POST /ingest`

Request body:

```json
{
  "dataset_path": "/path/to/events_folder"
}
```

### `POST /build-index`

No request body.

### `POST /query`

Request body:

```json
{
  "query": "How does this team attack in transition?",
  "top_k": 5,
  "use_llm": true
}
```

### `POST /chat`

Request body:

```json
{
  "session_id": "existing-session-id",
  "message": "What about Messi in the second half?",
  "top_k": 5,
  "use_llm": true
}
```

Response fields:

- `answer`: grounded answer
- `evidence`: retrieved sequence summaries with scores
- `trace`: ingestion, indexing, retrieval, and generation metadata
- `history`: session turn history

## Example Questions

- `Who won this match?`
- `Who scored?`
- `How did Barcelona create chances?`
- `What about Messi in the second half?`
- `How does this team attack in transition?`

## Notes

- Retrieval works without any generation model.
- Answers stay grounded in retrieved evidence.
- Ollama is optional and local-only.
- The app is organized as a single Python package for local testing before any future frontend/backend split.
