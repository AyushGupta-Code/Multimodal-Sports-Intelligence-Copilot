"""Compact data pipeline for local soccer event ingestion and retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from schemas import EventRecord, QueryResponse, SequenceRecord


def _safe_name(value: Any, fallback: str | None = "Unknown") -> str | None:
    """Return a readable string for nested StatsBomb name fields."""

    # Some fields are nested like {"name": "Pass"}, so this helper pulls out the name safely.
    if isinstance(value, dict):
        nested_value = value.get("name")
        return str(nested_value) if nested_value not in (None, "") else fallback
    if value in (None, ""):
        return fallback
    return str(value)


def _safe_location(value: Any) -> list[float] | None:
    """Return a normalized 2D location when one is available."""

    # Keep only the first two coordinates and ignore bad values instead of failing.
    if not isinstance(value, list) or len(value) < 2:
        return None
    try:
        return [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        return None


def _event_text(event: EventRecord) -> str:
    """Return a short event phrase for sequence summaries."""

    # Build a short event label that is easy to read and easy to index for search.
    if event.event_type.lower() == "pass" and event.pass_end_location:
        return f"pass to {int(event.pass_end_location[0])},{int(event.pass_end_location[1])}"
    if event.event_type.lower() == "shot" and event.shot_outcome:
        return f"shot ({event.shot_outcome})"
    return event.event_type.lower()


def _query_text(query: str) -> str:
    """Expand common soccer questions into simple retrieval-friendly keywords."""

    # Add a few plain keyword hints so broad questions match the short sequence summaries better.
    lowered = query.strip().lower()
    extra_terms: list[str] = []
    if "chance" in lowered or "create chances" in lowered:
        extra_terms.extend(["shot", "attack", "opportunity", "final third"])
    if "transition" in lowered:
        extra_terms.extend(["progressed", "attack", "forward"])
    if "through ball" in lowered:
        extra_terms.extend(["pass", "progressed", "attacking third"])
    if "winger" in lowered:
        extra_terms.extend(["pass", "shot", "attack"])
    if "cutback" in lowered:
        extra_terms.extend(["pass", "shot", "attacking third"])
    return " ".join([lowered, *extra_terms]).strip()


def _index_text(sequence: SequenceRecord) -> str:
    """Build the text used for TF-IDF indexing for one sequence."""

    # Include a few simple keywords so broad tactical questions can still find relevant sequences.
    keywords = ["team", "attack", "possession", "sequence"]
    if sequence.ended_in_shot:
        keywords.extend(["shot", "chance", "chance_creation", "opportunity"])
    if any("pass" in event for event in sequence.event_chain):
        keywords.extend(["pass", "combination_play"])
    if "progressed from" in sequence.progression:
        keywords.extend(["progression", "forward", "transition"])
    if "attacking third" in sequence.progression:
        keywords.extend(["final_third", "attacking_third"])
    return " ".join([sequence.summary, " ".join(keywords)])


def _zone_name(x: float | None) -> str:
    """Map an x coordinate into a broad progression zone."""

    # Split the pitch into rough thirds so progression can be described simply.
    if x is None:
        return "unknown"
    if x < 40:
        return "defensive third"
    if x < 80:
        return "middle third"
    return "attacking third"


def _progression_label(events: list[EventRecord]) -> str:
    """Describe broad field progression from the first to last usable location."""

    # Compare the start and end of the sequence to describe how far the attack moved.
    start_x = next((e.location[0] for e in events if e.location), None)
    end_x = next(
        (
            (e.pass_end_location or e.location)[0]
            for e in reversed(events)
            if e.pass_end_location or e.location
        ),
        None,
    )
    start_zone = _zone_name(start_x)
    end_zone = _zone_name(end_x)
    if start_x is None or end_x is None:
        return "unknown progression"
    if end_x - start_x >= 25:
        return f"progressed from {start_zone} to {end_zone}"
    if end_zone == "attacking third":
        return f"sustained attack in {end_zone}"
    return f"circulated between {start_zone} and {end_zone}"


def _sequence_summary(sequence_id: str, events: list[EventRecord]) -> SequenceRecord:
    """Build one deterministic attacking-sequence summary from normalized events."""

    # Create one short summary that will be shown to the user and also used for search.
    first = events[0]
    players = list(dict.fromkeys(e.player_name for e in events if e.player_name != "Unknown"))
    event_chain = [_event_text(event) for event in events[:8]]
    ended_in_shot = any(event.event_type.lower() == "shot" for event in events)
    progression = _progression_label(events)
    player_text = ", ".join(players[:5]) if players else "unknown players"
    chain_text = " -> ".join(event_chain) if event_chain else "no clear event chain"
    shot_text = "ended in a shot" if ended_in_shot else "did not end in a shot"
    summary = (
        f"{first.team_name} sequence with {player_text}. "
        f"Chain: {chain_text}. {shot_text}. {progression}."
    )
    return SequenceRecord(
        sequence_id=sequence_id,
        match_id=first.match_id,
        team_name=first.team_name,
        possession_id=first.possession_id,
        players=players,
        event_chain=event_chain,
        ended_in_shot=ended_in_shot,
        progression=progression,
        summary=summary,
    )


def _load_json_file(path: Path) -> list[dict[str, Any]]:
    """Load and validate one StatsBomb-style event JSON file."""

    # Check that the file exists, is valid JSON, and looks like a list of events.
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc.msg}") from exc
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of events in {path}")
    for index, item in enumerate(data[:5]):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid event structure in {path} at item {index}")
    return data


def load_event_files(dataset_path: str) -> tuple[list[dict[str, Any]], int]:
    """Load StatsBomb-style JSON events from one file or a directory."""

    # Allow the user to point to one file or a folder of JSON files.
    path = Path(dataset_path).expanduser()
    if not path.exists():
        raise ValueError(f"Dataset path does not exist: {path}")
    files = [path] if path.is_file() else sorted(path.glob("*.json"))
    if not files:
        raise ValueError(f"No JSON files found at: {path}")
    all_events: list[dict[str, Any]] = []
    for file_path in files:
        all_events.extend(_load_json_file(file_path))
    return all_events, len(files)


def normalize_events(raw_events: list[dict[str, Any]]) -> list[EventRecord]:
    """Normalize raw StatsBomb-style events into compact event records."""

    # Keep only the small set of event fields needed for this MVP.
    normalized: list[EventRecord] = []
    for raw in raw_events:
        event_type = _safe_name(raw.get("type"))
        pass_data = raw.get("pass") if isinstance(raw.get("pass"), dict) else {}
        shot_data = raw.get("shot") if isinstance(raw.get("shot"), dict) else {}
        normalized.append(
            EventRecord(
                match_id=str(raw.get("match_id")) if raw.get("match_id") is not None else None,
                team_name=_safe_name(raw.get("team")),
                player_name=_safe_name(raw.get("player")),
                event_type=event_type,
                minute=int(raw.get("minute") or 0),
                second=int(raw.get("second") or 0),
                possession_id=raw.get("possession"),
                play_pattern=_safe_name(raw.get("play_pattern"), None) if raw.get("play_pattern") else None,
                location=_safe_location(raw.get("location")),
                pass_end_location=_safe_location(pass_data.get("end_location")),
                shot_outcome=_safe_name(shot_data.get("outcome"), None) if shot_data.get("outcome") else None,
            )
        )
    return normalized


def build_sequences(events: list[EventRecord]) -> list[SequenceRecord]:
    """Group normalized events into simple team-possession attacking sequences."""

    # Group events by match, team, and possession to form simple attacking sequences.
    grouped: dict[tuple[str | None, str, int | None], list[EventRecord]] = {}
    for index, event in enumerate(events):
        if event.team_name == "Unknown":
            continue
        key = (
            event.match_id,
            event.team_name,
            event.possession_id if event.possession_id is not None else -index,
        )
        grouped.setdefault(key, []).append(event)
    sequences: list[SequenceRecord] = []
    for index, possession_events in enumerate(grouped.values(), start=1):
        possession_events.sort(key=lambda item: (item.minute, item.second))
        if len(possession_events) < 2:
            continue
        sequences.append(_sequence_summary(f"seq-{index}", possession_events))
    return sequences


def build_index(sequences: list[SequenceRecord]) -> dict[str, Any]:
    """Build an in-memory TF-IDF retrieval index over sequence summaries."""

    # Build a local search index from the sequence summaries.
    if not sequences:
        raise ValueError("No sequences available to index")
    texts = [_index_text(sequence) for sequence in sequences]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    return {"vectorizer": vectorizer, "matrix": matrix, "sequences": sequences}


def retrieve(query: str, index_data: dict[str, Any], top_k: int = 5) -> list[dict[str, Any]]:
    """Retrieve the most relevant sequence summaries for a natural-language query."""

    # Return the best matching summaries and their scores for the user's question.
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("Query must not be empty")
    expanded_query = _query_text(cleaned_query)
    vectorizer: TfidfVectorizer = index_data["vectorizer"]
    matrix = index_data["matrix"]
    sequences: list[SequenceRecord] = index_data["sequences"]
    query_vector = vectorizer.transform([expanded_query])
    scores = linear_kernel(query_vector, matrix).flatten()
    ranked = scores.argsort()[::-1]
    results: list[dict[str, Any]] = []
    for idx in ranked[: max(top_k, 1)]:
        score = float(scores[idx])
        if score <= 0:
            continue
        sequence = sequences[idx]
        results.append(
            {
                "sequence_id": sequence.sequence_id,
                "score": round(score, 4),
                "team_name": sequence.team_name,
                "players": sequence.players,
                "ended_in_shot": sequence.ended_in_shot,
                "summary": sequence.summary,
            }
        )
    return results


def compose_grounded_answer(query: str, evidence: list[dict[str, Any]]) -> str:
    """Compose a deterministic grounded answer from retrieved evidence only."""

    # Build a careful answer that only refers to what appears in the retrieved evidence.
    if not evidence:
        return f'No grounded evidence was found for "{query}".'
    teams = sorted({item["team_name"] for item in evidence if item.get("team_name")})
    players = []
    for item in evidence:
        players.extend(item.get("players") or [])
    unique_players = list(dict.fromkeys(players))
    shot_count = sum(1 for item in evidence if item.get("ended_in_shot"))
    progression_hits = [
        item["summary"]
        for item in evidence
        if "progressed from" in item["summary"] or "sustained attack" in item["summary"]
    ]
    answer_parts = [
        f'The answer is grounded in {len(evidence)} retrieved sequences for "{query}".',
        f"Teams in evidence: {', '.join(teams) if teams else 'unknown'}.",
        f"Common players involved: {', '.join(unique_players[:6]) if unique_players else 'not clear from evidence'}.",
        f"{shot_count} of the retrieved sequences ended in a shot.",
        "The evidence most often shows forward progression."
        if progression_hits
        else "The evidence does not show a strong progression pattern.",
    ]
    return " ".join(answer_parts)


def run_query(query: str, index_data: dict[str, Any], trace: dict[str, Any], top_k: int = 5) -> QueryResponse:
    """Run retrieval and grounded answer generation for one user query."""

    # Run search first, then build the final response with evidence and trace data.
    evidence = retrieve(query=query, index_data=index_data, top_k=top_k)
    answer = compose_grounded_answer(query=query, evidence=evidence)
    return QueryResponse(
        answer=answer,
        evidence=evidence,
        trace={
            **trace,
            "retrieval_results": len(evidence),
            "generation_mode": "template",
        },
    )
