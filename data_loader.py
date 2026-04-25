"""Data loading and normalization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from domain_models import EventRecord


def _safe_name(value: Any, fallback: str | None = "Unknown") -> str | None:
    """Return a readable string for nested StatsBomb name fields."""

    if isinstance(value, dict):
        nested_value = value.get("name")
        return str(nested_value) if nested_value not in (None, "") else fallback
    if value in (None, ""):
        return fallback
    return str(value)


def _safe_location(value: Any) -> list[float] | None:
    """Return a normalized 2D location when one is available."""

    if not isinstance(value, list) or len(value) < 2:
        return None
    try:
        return [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    """Return a float value when possible."""

    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _load_json_file(path: Path) -> list[dict[str, Any]]:
    """Load and validate one StatsBomb-style event JSON file."""

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
    """Load StatsBomb-style JSON events from one file or directory."""

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
    """Normalize raw StatsBomb events into compact records."""

    normalized: list[EventRecord] = []
    for raw in raw_events:
        event_type = _safe_name(raw.get("type"))
        pass_data = raw.get("pass") if isinstance(raw.get("pass"), dict) else {}
        shot_data = raw.get("shot") if isinstance(raw.get("shot"), dict) else {}
        bad_behaviour = raw.get("bad_behaviour") if isinstance(raw.get("bad_behaviour"), dict) else {}
        foul_committed = raw.get("foul_committed") if isinstance(raw.get("foul_committed"), dict) else {}
        substitution = raw.get("substitution") if isinstance(raw.get("substitution"), dict) else {}
        normalized.append(
            EventRecord(
                match_id=str(raw.get("match_id")) if raw.get("match_id") is not None else None,
                team_name=_safe_name(raw.get("team")),
                possession_team_name=_safe_name(raw.get("possession_team")),
                player_name=_safe_name(raw.get("player")),
                event_type=event_type,
                minute=int(raw.get("minute") or 0),
                second=int(raw.get("second") or 0),
                period=int(raw.get("period") or 1),
                possession_id=raw.get("possession"),
                play_pattern=_safe_name(raw.get("play_pattern"), None) if raw.get("play_pattern") else None,
                location=_safe_location(raw.get("location")),
                pass_end_location=_safe_location(pass_data.get("end_location")),
                pass_recipient_name=_safe_name(pass_data.get("recipient"), None) if pass_data.get("recipient") else None,
                shot_outcome=_safe_name(shot_data.get("outcome"), None) if shot_data.get("outcome") else None,
                shot_xg=_safe_float(shot_data.get("statsbomb_xg")),
                duration=_safe_float(raw.get("duration")),
                position_name=_safe_name(raw.get("position"), None) if raw.get("position") else None,
                card_name=_safe_name(bad_behaviour.get("card"), None) or _safe_name(foul_committed.get("card"), None),
                replacement_player_name=_safe_name(substitution.get("replacement"), None)
                if substitution
                else None,
            )
        )
    return normalized
