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
- Phase 1 is event-data only and keeps everything in memory for simplicity.
