# Local Soccer Intelligence Copilot

Minimal local-first MVP for ingesting StatsBomb-style soccer event JSON, converting events into simple attacking sequences, indexing those sequence summaries with TF-IDF, and answering grounded tactical questions through a FastAPI API and tiny built-in HTML UI.

## What It Does

- Loads one local JSON file or a folder of JSON files.
- Normalizes a small subset of StatsBomb-style event fields.
- Groups events into simple team-possession attacking sequences.
- Builds deterministic sequence summaries for retrieval.
- Uses an in-memory TF-IDF index for local search.
- Automatically ingests data and builds the index when you run a query in the UI.
- Returns grounded answers with evidence snippets and trace metadata.

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
pip install -r requirements.txt
```

## Optional LLM Mode (Ollama + Gemma 4)

The app can use a local Ollama model for answer writing while keeping TF-IDF retrieval and evidence formatting unchanged.

### 1. Install Ollama

On Linux, install Ollama with:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start Ollama:

```bash
ollama serve
```

### 2. Download a Local Model

Pull Gemma 4 locally:

```bash
ollama pull gemma4:26b
```

You can also test the model directly from the terminal:

```bash
ollama run gemma4:26b
```

### 3. Set the Model Name for the App

In the terminal where you run FastAPI:

```bash
export OLLAMA_MODEL=gemma4:26b
```

Optional custom Ollama URL:

```bash
export OLLAMA_URL=http://127.0.0.1:11434
```

### 4. Use the Local LLM

The LLM path is optional. Send `use_llm: true` when you want Gemma 4 to rewrite the grounded answer from the retrieved evidence.

Example `POST /query` body:

```json
{
  "query": "How does this team create chances?",
  "top_k": 5,
  "use_llm": true
}
```

If Ollama is not running or the model is unavailable, the app falls back to the template answer automatically.
The answers remain grounded in the retrieved evidence; the LLM only improves phrasing.

## Download Local StatsBomb Data

Create the local data folder and download one sample StatsBomb event file:

```bash
mkdir -p open-data/data/events
python3 -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/15946.json', 'open-data/data/events/15946.json')"
```

This gives you a local event file at:

```text
/mnt/d/Projects/Multimodal-Sports-Intelligence-Copilot/open-data/data/events/15946.json
```

## Run

```bash
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`.

## Point The App To Local StatsBomb JSON

- The UI now reads local data automatically from:
  `/mnt/d/Projects/Multimodal-Sports-Intelligence-Copilot/open-data/data/events`
- You do not need to enter a dataset path in the page.
- If that folder contains one file, the app uses that file.
- If that folder contains multiple `.json` files, the app loads all of them.

The app expects each JSON file to contain a top-level list of StatsBomb-style event dictionaries.

## UI Flow

1. Download the sample data into `open-data/data/events`.
2. Start the FastAPI server.
3. Open the UI.
4. Enter a question and click `Run Query`.

When you click `Run Query`, the app automatically:

- loads the local event data
- builds attacking sequences
- builds the TF-IDF index
- retrieves evidence
- returns a grounded answer

The built-in page works with the template answer path by default. Use the API request above when you want Ollama-based phrasing.

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
  "top_k": 5
}
```

Response fields:

- `answer`: grounded template-based answer
- `evidence`: retrieved sequence summaries with scores
- `trace`: ingestion, indexing, retrieval, and generation metadata

The `/query` route automatically loads the local dataset and builds the index if they are not already ready.

Example with optional Ollama generation:

```json
{
  "query": "How does this team create chances?",
  "top_k": 5,
  "use_llm": true
}
```

## Example Questions

- `How does this team create chances?`
- `What are this player's common attacking patterns?`
- `Find sequences similar to a cutback attack.`
- `How does this team attack in transition?`
- `Show examples of through balls into the half-space.`
- `Summarize this winger's chance-creation style.`

One simple sample question to try first:

- `How does this team create chances?`

## Notes

- Retrieval works without any generation model.
- Answers are grounded only in retrieved evidence.
- Optional LLM generation uses a local Ollama model and falls back to the template answer on failure.
- Phase 1 is event-data only and keeps everything in memory for simplicity.
