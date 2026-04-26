"""Shared application state and local dataset preparation helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from app.services.data_loader import load_event_files, normalize_events
from app.services.match_features import build_sequences
from app.services.retrieval import build_index

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "events"

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


def _candidate_paths() -> list[Path]:
    dataset_path = Path(os.getenv("DATASET_PATH", "")).expanduser() if os.getenv("DATASET_PATH") else None
    if dataset_path:
        return [dataset_path]
    return [DEFAULT_DATASET_PATH]


def ingest_dataset(dataset_path: str) -> dict[str, Any]:
    """Load local event files into app state."""

    raw_events, file_count = load_event_files(dataset_path)
    events = normalize_events(raw_events)
    sequences = build_sequences(events)
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


def load_default_data() -> None:
    """Load the default local dataset and build sequences."""

    last_error: ValueError | None = None
    for candidate_path in _candidate_paths():
        try:
            ingest_dataset(str(candidate_path))
            break
        except ValueError as exc:
            last_error = exc
    else:
        detail = str(last_error) if last_error else "No dataset path could be loaded."
        raise HTTPException(status_code=400, detail=detail)


def auto_prepare() -> None:
    """Load data and build the search index automatically when needed."""

    if not STATE["events"]:
        load_default_data()
    if STATE["index"] is None:
        try:
            STATE["index"] = build_index(STATE["events"], STATE["sequences"])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        STATE["trace"]["sequences_indexed"] = len(STATE["sequences"])


def require_ingested() -> None:
    """Raise an API error when ingestion has not happened yet."""

    if not STATE["events"]:
        raise HTTPException(status_code=400, detail="No data loaded yet. Run /ingest first.")
