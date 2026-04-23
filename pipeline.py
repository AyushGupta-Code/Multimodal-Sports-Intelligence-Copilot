"""Generalized local match QA pipeline for grounded soccer analysis."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from schemas import DocumentRecord, EventRecord, QueryResponse, SequenceRecord


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


def _normalize_text(value: str) -> str:
    """Normalize text for lightweight entity matching."""

    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _name_in_query(name: str, lowered_query: str) -> bool:
    """Return whether a team or player name appears in the query."""

    lowered_name = name.lower()
    if lowered_name in lowered_query:
        return True
    normalized_query = _normalize_text(lowered_query)
    normalized_name = _normalize_text(name)
    if normalized_name and re.search(rf"\b{re.escape(normalized_name)}\b", normalized_query):
        return True
    for token in normalized_name.split():
        if len(token) >= 4 and re.search(rf"\b{re.escape(token)}(?:'s)?\b", normalized_query):
            return True
    return False


def _top_counter_items(counter: Counter[str], limit: int = 5) -> list[tuple[str, int]]:
    """Return the most common counter items in deterministic order."""

    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]


def _event_text(event: EventRecord) -> str:
    """Return a short event phrase for sequence summaries."""

    if event.event_type.lower() == "pass":
        recipient = f" to {event.pass_recipient_name}" if event.pass_recipient_name else ""
        if event.pass_end_location:
            return f"pass{recipient} toward {int(event.pass_end_location[0])},{int(event.pass_end_location[1])}"
        return f"pass{recipient}"
    if event.event_type.lower() == "shot":
        if event.shot_outcome:
            return f"shot ({event.shot_outcome})"
        return "shot"
    return event.event_type.lower()


def _query_text(query: str) -> str:
    """Expand common football prompts into retrieval-friendly keywords."""

    lowered = query.strip().lower()
    extra_terms: list[str] = []
    keyword_map = {
        "chance": ["shot", "attack", "final third", "opportunity"],
        "create": ["progression", "pass", "carry"],
        "transition": ["progressed", "carry", "forward"],
        "build up": ["pass", "possession", "combination play"],
        "through ball": ["pass", "progression"],
        "cross": ["pass", "wide", "attacking third"],
        "press": ["ball recovery", "duel", "carry"],
        "defend": ["ball recovery", "duel"],
        "shot": ["finish", "attempt"],
        "xg": ["shot", "chance"],
        "player": ["involvement", "sequence"],
        "team": ["match", "possession"],
        "when": ["minute", "period"],
        "first half": ["period 1"],
        "second half": ["period 2"],
    }
    for phrase, terms in keyword_map.items():
        if phrase in lowered:
            extra_terms.extend(terms)
    return " ".join([lowered, *extra_terms]).strip()


def _index_text(document: DocumentRecord) -> str:
    """Build the text used for TF-IDF indexing for one document."""

    parts = [
        document.summary,
        document.text,
        " ".join(document.keywords),
        document.doc_type.replace("_", " "),
        document.team_name or "",
        document.player_name or "",
        " ".join(document.players),
    ]
    return " ".join(part for part in parts if part)


def _zone_name(x: float | None) -> str:
    """Map an x coordinate into a broad progression zone."""

    if x is None:
        return "unknown"
    if x < 40:
        return "defensive third"
    if x < 80:
        return "middle third"
    return "attacking third"


def _progression_label(events: list[EventRecord]) -> str:
    """Describe broad field progression from the first to last usable location."""

    start_x = next((event.location[0] for event in events if event.location), None)
    end_x = next(
        (
            (event.pass_end_location or event.location)[0]
            for event in reversed(events)
            if event.pass_end_location or event.location
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
    """Build one deterministic possession summary from normalized events."""

    first = events[0]
    players = list(dict.fromkeys(event.player_name for event in events if event.player_name != "Unknown"))
    event_chain = [_event_text(event) for event in events[:8]]
    ended_in_shot = any(event.event_type.lower() == "shot" for event in events)
    progression = _progression_label(events)
    play_patterns = list(
        dict.fromkeys(event.play_pattern for event in events if event.play_pattern not in (None, "Unknown"))
    )
    player_text = ", ".join(players[:5]) if players else "unknown players"
    chain_text = " -> ".join(event_chain) if event_chain else "no clear event chain"
    shot_text = "ended in a shot" if ended_in_shot else "did not end in a shot"
    pattern_text = f" Play patterns: {', '.join(play_patterns[:3])}." if play_patterns else ""
    summary = (
        f"{first.team_name} sequence in period {first.period} from {first.minute}:{first.second:02d} "
        f"to {events[-1].minute}:{events[-1].second:02d} with {player_text}. "
        f"Chain: {chain_text}. {shot_text}. {progression}.{pattern_text}"
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
        start_minute=first.minute,
        end_minute=events[-1].minute,
        period=first.period,
        play_patterns=play_patterns,
    )


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
    """Load StatsBomb-style JSON events from one file or a directory."""

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

    normalized: list[EventRecord] = []
    for raw in raw_events:
        event_type = _safe_name(raw.get("type"))
        pass_data = raw.get("pass") if isinstance(raw.get("pass"), dict) else {}
        shot_data = raw.get("shot") if isinstance(raw.get("shot"), dict) else {}
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
            )
        )
    return normalized


def build_sequences(events: list[EventRecord]) -> list[SequenceRecord]:
    """Group normalized events into simple team-possession sequences."""

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
        possession_events.sort(key=lambda item: (item.period, item.minute, item.second))
        if len(possession_events) < 2:
            continue
        sequences.append(_sequence_summary(f"seq-{index}", possession_events))
    return sequences


def build_match_context(events: list[EventRecord], sequences: list[SequenceRecord]) -> dict[str, Any]:
    """Build reusable match-, team-, and player-level stats."""

    if not events:
        raise ValueError("No events available to analyze")

    team_stats: dict[str, dict[str, Any]] = {}
    player_stats: dict[str, dict[str, Any]] = {}
    play_pattern_counts: Counter[str] = Counter()
    event_type_counts: Counter[str] = Counter()
    period_counts: Counter[str] = Counter()
    minute_min = min(event.minute for event in events)
    minute_max = max(event.minute for event in events)
    match_ids = sorted({event.match_id for event in events if event.match_id})

    for event in events:
        team_name = event.team_name
        player_name = event.player_name
        event_type = event.event_type.lower()
        play_pattern = event.play_pattern or "Unknown"

        team_entry = team_stats.setdefault(
            team_name,
            {
                "events": 0,
                "shots": 0,
                "passes": 0,
                "carries": 0,
                "recoveries": 0,
                "xg": 0.0,
                "players": set(),
                "periods": Counter(),
                "play_patterns": Counter(),
                "minutes": [],
                "sequences": 0,
            },
        )
        team_entry["events"] += 1
        team_entry["players"].add(player_name)
        team_entry["periods"][str(event.period)] += 1
        team_entry["play_patterns"][play_pattern] += 1
        team_entry["minutes"].append(event.minute)

        if event_type == "shot":
            team_entry["shots"] += 1
            team_entry["xg"] += event.shot_xg or 0.0
        elif event_type == "pass":
            team_entry["passes"] += 1
        elif event_type == "carry":
            team_entry["carries"] += 1
        elif event_type == "ball recovery":
            team_entry["recoveries"] += 1

        if player_name != "Unknown":
            player_entry = player_stats.setdefault(
                player_name,
                {
                    "team_name": team_name,
                    "events": 0,
                    "shots": 0,
                    "passes": 0,
                    "carries": 0,
                    "recoveries": 0,
                    "xg": 0.0,
                    "minutes": [],
                    "periods": Counter(),
                    "play_patterns": Counter(),
                    "sequences": 0,
                },
            )
            player_entry["events"] += 1
            player_entry["minutes"].append(event.minute)
            player_entry["periods"][str(event.period)] += 1
            player_entry["play_patterns"][play_pattern] += 1
            if event_type == "shot":
                player_entry["shots"] += 1
                player_entry["xg"] += event.shot_xg or 0.0
            elif event_type == "pass":
                player_entry["passes"] += 1
            elif event_type == "carry":
                player_entry["carries"] += 1
            elif event_type == "ball recovery":
                player_entry["recoveries"] += 1

        play_pattern_counts[play_pattern] += 1
        event_type_counts[event_type] += 1
        period_counts[str(event.period)] += 1

    for sequence in sequences:
        if sequence.team_name in team_stats:
            team_stats[sequence.team_name]["sequences"] += 1
        for player in sequence.players:
            if player in player_stats:
                player_stats[player]["sequences"] += 1

    for team_name, team_entry in team_stats.items():
        team_entry["players"] = sorted(player for player in team_entry["players"] if player != "Unknown")
        team_entry["minute_min"] = min(team_entry["minutes"]) if team_entry["minutes"] else None
        team_entry["minute_max"] = max(team_entry["minutes"]) if team_entry["minutes"] else None
        team_entry["top_play_patterns"] = _top_counter_items(team_entry["play_patterns"], limit=3)

    for player_name, player_entry in player_stats.items():
        player_entry["minute_min"] = min(player_entry["minutes"]) if player_entry["minutes"] else None
        player_entry["minute_max"] = max(player_entry["minutes"]) if player_entry["minutes"] else None
        player_entry["top_play_patterns"] = _top_counter_items(player_entry["play_patterns"], limit=3)

    return {
        "match_ids": match_ids,
        "team_names": sorted(team_stats.keys()),
        "player_names": sorted(player_stats.keys()),
        "team_stats": team_stats,
        "player_stats": player_stats,
        "total_events": len(events),
        "total_sequences": len(sequences),
        "total_shots": event_type_counts.get("shot", 0),
        "total_passes": event_type_counts.get("pass", 0),
        "total_carries": event_type_counts.get("carry", 0),
        "total_recoveries": event_type_counts.get("ball recovery", 0),
        "total_xg": round(sum(team_entry["xg"] for team_entry in team_stats.values()), 3),
        "play_patterns": _top_counter_items(play_pattern_counts, limit=5),
        "event_types": _top_counter_items(event_type_counts, limit=8),
        "period_counts": dict(period_counts),
        "minute_min": minute_min,
        "minute_max": minute_max,
        "top_players_by_events": _top_counter_items(
            Counter({player: stats["events"] for player, stats in player_stats.items()}), limit=5
        ),
        "top_players_by_shots": _top_counter_items(
            Counter({player: stats["shots"] for player, stats in player_stats.items() if stats["shots"] > 0}), limit=5
        ),
        "top_players_by_passes": _top_counter_items(
            Counter({player: stats["passes"] for player, stats in player_stats.items() if stats["passes"] > 0}), limit=5
        ),
    }


def _format_top_items(items: list[tuple[str, int]], fallback: str = "none") -> str:
    """Render a short top-items string."""

    if not items:
        return fallback
    return ", ".join(f"{name} ({count})" for name, count in items)


def build_documents(sequences: list[SequenceRecord], context: dict[str, Any]) -> list[DocumentRecord]:
    """Build a mixed document set for match, team, player, and sequence retrieval."""

    documents: list[DocumentRecord] = []
    overview_summary = (
        f"Match overview for {', '.join(context['team_names'])}: {context['total_events']} events, "
        f"{context['total_sequences']} indexed sequences, {context['total_shots']} shots, "
        f"{context['total_passes']} passes, {context['total_carries']} carries, "
        f"{context['total_recoveries']} ball recoveries, total xG {context['total_xg']:.2f}."
    )
    overview_text = (
        f"Teams: {', '.join(context['team_names'])}. "
        f"Minute range: {context['minute_min']} to {context['minute_max']}. "
        f"Top players by events: {_format_top_items(context['top_players_by_events'])}. "
        f"Top players by shots: {_format_top_items(context['top_players_by_shots'])}. "
        f"Top play patterns: {_format_top_items(context['play_patterns'])}. "
        f"Period counts: {context['period_counts']}."
    )
    documents.append(
        DocumentRecord(
            doc_id="match-overview",
            doc_type="match_overview",
            match_id=context["match_ids"][0] if context["match_ids"] else None,
            summary=overview_summary,
            text=overview_text,
            keywords=["match", "overview", "summary", "shots", "passes", "carries", "xg"],
        )
    )

    for team_name, team_stats in context["team_stats"].items():
        top_team_players = _top_counter_items(
            Counter(
                {
                    player: context["player_stats"][player]["events"]
                    for player in team_stats["players"]
                    if player in context["player_stats"]
                }
            ),
            limit=4,
        )
        documents.append(
            DocumentRecord(
                doc_id=f"team-{_normalize_text(team_name).replace(' ', '-')}",
                doc_type="team_summary",
                match_id=context["match_ids"][0] if context["match_ids"] else None,
                team_name=team_name,
                minute_start=team_stats["minute_min"],
                minute_end=team_stats["minute_max"],
                summary=(
                    f"{team_name} team summary: {team_stats['events']} events, {team_stats['sequences']} sequences, "
                    f"{team_stats['shots']} shots, {team_stats['passes']} passes, {team_stats['carries']} carries, "
                    f"{team_stats['recoveries']} ball recoveries, xG {team_stats['xg']:.2f}."
                ),
                text=(
                    f"Top involved players: {_format_top_items(top_team_players)}. "
                    f"Top play patterns: {_format_top_items(team_stats['top_play_patterns'])}. "
                    f"Period split: {dict(team_stats['periods'])}. "
                    f"Minute range: {team_stats['minute_min']} to {team_stats['minute_max']}."
                ),
                players=[player for player, _ in top_team_players],
                keywords=["team", team_name, "shots", "passes", "carries", "recoveries", "xg"],
            )
        )

    for player_name, player_stats in context["player_stats"].items():
        documents.append(
            DocumentRecord(
                doc_id=f"player-{_normalize_text(player_name).replace(' ', '-')}",
                doc_type="player_summary",
                match_id=context["match_ids"][0] if context["match_ids"] else None,
                team_name=player_stats["team_name"],
                player_name=player_name,
                minute_start=player_stats["minute_min"],
                minute_end=player_stats["minute_max"],
                summary=(
                    f"{player_name} player summary for {player_stats['team_name']}: {player_stats['events']} events, "
                    f"{player_stats['sequences']} sequences, {player_stats['shots']} shots, "
                    f"{player_stats['passes']} passes, {player_stats['carries']} carries, "
                    f"{player_stats['recoveries']} ball recoveries, xG {player_stats['xg']:.2f}."
                ),
                text=(
                    f"Minute range: {player_stats['minute_min']} to {player_stats['minute_max']}. "
                    f"Play patterns: {_format_top_items(player_stats['top_play_patterns'])}. "
                    f"Period split: {dict(player_stats['periods'])}."
                ),
                players=[player_name],
                keywords=["player", player_name, player_stats["team_name"], "shots", "passes", "carries", "xg"],
            )
        )

    for sequence in sequences:
        documents.append(
            DocumentRecord(
                doc_id=sequence.sequence_id,
                doc_type="sequence",
                match_id=sequence.match_id,
                team_name=sequence.team_name,
                period=sequence.period,
                minute_start=sequence.start_minute,
                minute_end=sequence.end_minute,
                summary=sequence.summary,
                text=(
                    f"Sequence for {sequence.team_name}. Players: {', '.join(sequence.players[:6])}. "
                    f"Progression: {sequence.progression}. Play patterns: {', '.join(sequence.play_patterns[:3]) or 'unknown'}."
                ),
                players=sequence.players,
                ended_in_shot=sequence.ended_in_shot,
                play_patterns=sequence.play_patterns,
                keywords=[
                    "sequence",
                    sequence.team_name,
                    "shot" if sequence.ended_in_shot else "no-shot",
                    "progression",
                    "possession",
                    *sequence.play_patterns[:3],
                ],
            )
        )

    return documents


def build_index(events: list[EventRecord], sequences: list[SequenceRecord]) -> dict[str, Any]:
    """Build an in-memory TF-IDF retrieval index over mixed match documents."""

    if not events:
        raise ValueError("No events available to index")
    context = build_match_context(events=events, sequences=sequences)
    documents = build_documents(sequences=sequences, context=context)
    texts = [_index_text(document) for document in documents]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    return {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "documents": documents,
        "context": context,
    }


def _question_profile(query: str) -> dict[str, bool]:
    """Extract lightweight intent flags from the user question."""

    lowered = query.strip().lower()
    return {
        "asks_who": bool(re.search(r"\bwho\b|\bwhich player\b|\bwhich team\b", lowered)),
        "asks_how": lowered.startswith("how"),
        "asks_why": lowered.startswith("why"),
        "asks_when": bool(re.search(r"\bwhen\b|\bminute\b|\bfirst half\b|\bsecond half\b|\bperiod\b", lowered)),
        "asks_count": bool(re.search(r"\bhow many\b|\bcount\b|\bnumber of\b", lowered)),
        "asks_team": "team" in lowered or "barcelona" in lowered or "alav" in lowered,
        "asks_player": "player" in lowered or "who" in lowered,
        "asks_shot": "shot" in lowered or "finish" in lowered or "chance" in lowered,
        "asks_pass": "pass" in lowered,
        "asks_carry": "carry" in lowered or "dribble" in lowered,
        "asks_recovery": "recovery" in lowered or "recover" in lowered or "press" in lowered,
        "asks_xg": "xg" in lowered or "expected goals" in lowered,
        "asks_summary": "summary" in lowered or "overview" in lowered or "overall" in lowered or "match" in lowered,
        "asks_compare": bool(re.search(r"\bcompare\b|\bbetter\b|\bmore than\b|\bvs\b|\bversus\b", lowered)),
        "uses_ambiguous_team_ref": "this team" in lowered or "that team" in lowered,
    }


def extract_query_entities(query: str, index_data: dict[str, Any]) -> dict[str, list[str]]:
    """Detect referenced teams and players from the indexed match context."""

    lowered_query = query.lower()
    context = index_data["context"]
    matched_teams = [team for team in context["team_names"] if _name_in_query(team, lowered_query)]
    matched_players = [player for player in context["player_names"] if _name_in_query(player, lowered_query)]
    return {"teams": matched_teams, "players": matched_players}


def _target_doc_types(profile: dict[str, bool]) -> set[str]:
    """Return preferred document types for the current question."""

    targets = {"sequence", "team_summary", "player_summary", "match_overview"}
    if profile["uses_ambiguous_team_ref"]:
        return {"team_summary", "sequence"}
    if profile["asks_how"] or profile["asks_why"]:
        return {"sequence", "team_summary"}
    if profile["asks_who"] and profile["asks_player"]:
        return {"player_summary", "team_summary"}
    if profile["asks_team"]:
        return {"team_summary", "sequence", "match_overview"}
    if profile["asks_when"]:
        return {"sequence", "player_summary", "team_summary"}
    if profile["asks_summary"]:
        return {"match_overview", "team_summary", "sequence"}
    return targets


def _doc_matches_entities(document: DocumentRecord, entities: dict[str, list[str]]) -> bool:
    """Return whether a document matches at least one explicit query entity."""

    if entities["teams"] and document.team_name in entities["teams"]:
        return True
    if entities["players"] and (
        document.player_name in entities["players"] or any(player in entities["players"] for player in document.players)
    ):
        return True
    return False


def retrieve(query: str, index_data: dict[str, Any], top_k: int = 5) -> list[dict[str, Any]]:
    """Retrieve the most relevant mixed evidence documents for a natural-language query."""

    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("Query must not be empty")

    expanded_query = _query_text(cleaned_query)
    profile = _question_profile(cleaned_query)
    entities = extract_query_entities(cleaned_query, index_data=index_data)
    target_types = _target_doc_types(profile)
    vectorizer: TfidfVectorizer = index_data["vectorizer"]
    matrix = index_data["matrix"]
    documents: list[DocumentRecord] = index_data["documents"]
    query_vector = vectorizer.transform([expanded_query])
    scores = linear_kernel(query_vector, matrix).flatten()

    ranked_results: list[tuple[float, DocumentRecord]] = []
    for index, document in enumerate(documents):
        adjusted_score = float(scores[index])
        if document.doc_type in target_types:
            adjusted_score += 0.12
        if profile["uses_ambiguous_team_ref"] and document.doc_type == "team_summary":
            adjusted_score += 0.26
        if profile["uses_ambiguous_team_ref"] and document.doc_type == "match_overview":
            adjusted_score -= 0.18
        if entities["players"] and document.doc_type == "player_summary":
            adjusted_score += 0.18
        if entities["teams"] and document.doc_type == "team_summary":
            adjusted_score += 0.14
        if profile["asks_when"] and document.doc_type == "sequence":
            adjusted_score += 0.16
        if document.doc_type == "match_overview" and (profile["asks_summary"] or not entities["teams"] and not entities["players"]):
            adjusted_score += 0.2
        if _doc_matches_entities(document, entities):
            adjusted_score += 0.22
        elif entities["teams"] or entities["players"]:
            adjusted_score -= 0.08
        if adjusted_score > 0:
            ranked_results.append((adjusted_score, document))

    ranked_results.sort(key=lambda item: item[0], reverse=True)
    results: list[dict[str, Any]] = []
    for adjusted_score, document in ranked_results[: max(top_k, 1)]:
        results.append(
            {
                "doc_id": document.doc_id,
                "doc_type": document.doc_type,
                "score": round(adjusted_score, 4),
                "match_id": document.match_id,
                "team_name": document.team_name,
                "player_name": document.player_name,
                "period": document.period,
                "minute_start": document.minute_start,
                "minute_end": document.minute_end,
                "players": document.players,
                "ended_in_shot": document.ended_in_shot,
                "play_patterns": document.play_patterns,
                "summary": document.summary,
            }
        )
    return results


def build_evidence_stats(evidence: list[dict[str, Any]], index_data: dict[str, Any], query: str) -> dict[str, Any]:
    """Build deterministic evidence stats for generic grounded QA."""

    sequence_evidence = [item for item in evidence if item["doc_type"] == "sequence"]
    team_docs = [item for item in evidence if item["doc_type"] == "team_summary"]
    player_docs = [item for item in evidence if item["doc_type"] == "player_summary"]
    player_counter: Counter[str] = Counter()
    team_counter: Counter[str] = Counter()
    for item in evidence:
        if item.get("team_name"):
            team_counter[item["team_name"]] += 1
        for player in item.get("players") or []:
            player_counter[player] += 1
        if item.get("player_name"):
            player_counter[item["player_name"]] += 1

    sequence_shots = sum(1 for item in sequence_evidence if item.get("ended_in_shot"))
    attacking_third_hits = sum(1 for item in sequence_evidence if "attacking third" in item["summary"].lower())
    pass_sequences = sum(1 for item in sequence_evidence if "pass" in item["summary"].lower())
    carry_sequences = sum(1 for item in sequence_evidence if "carry" in item["summary"].lower())
    minute_values = [
        minute
        for item in evidence
        for minute in [item.get("minute_start"), item.get("minute_end")]
        if isinstance(minute, int)
    ]
    context = index_data["context"]
    return {
        "query": query,
        "evidence_count": len(evidence),
        "sequence_count": len(sequence_evidence),
        "team_doc_count": len(team_docs),
        "player_doc_count": len(player_docs),
        "shot_sequences": sequence_shots,
        "attacking_third_hits": attacking_third_hits,
        "pass_sequences": pass_sequences,
        "carry_sequences": carry_sequences,
        "team_names": sorted(team_counter.keys()),
        "dominant_team": _top_counter_items(team_counter, limit=1)[0][0] if team_counter else None,
        "top_players": _top_counter_items(player_counter, limit=5),
        "minute_min": min(minute_values) if minute_values else context["minute_min"],
        "minute_max": max(minute_values) if minute_values else context["minute_max"],
        "context_total_events": context["total_events"],
        "context_total_sequences": context["total_sequences"],
        "context_total_shots": context["total_shots"],
        "context_total_passes": context["total_passes"],
        "context_total_carries": context["total_carries"],
        "context_total_recoveries": context["total_recoveries"],
        "context_total_xg": context["total_xg"],
    }


def _sequence_pattern_label(stats: dict[str, Any]) -> str:
    """Summarize tactical tendencies from retrieved sequence evidence."""

    if stats["sequence_count"] <= 0:
        return "mixed match evidence"
    majority = max(1, stats["sequence_count"] // 2)
    if stats["shot_sequences"] >= majority and stats["shot_sequences"] > 0:
        return "shot-leaning sequences"
    if stats["pass_sequences"] >= stats["carry_sequences"] and stats["pass_sequences"] > 0:
        return "pass-led progression"
    if stats["carry_sequences"] > 0:
        return "carry-led progression"
    return "mixed possession patterns"


def _team_style_label(team_stats: dict[str, Any]) -> str:
    """Summarize a team's broad chance-creation style from deterministic team stats."""

    if team_stats["passes"] >= team_stats["carries"] * 1.2:
        return "pass-led buildup"
    if team_stats["carries"] >= team_stats["passes"] * 0.8:
        return "carry-heavy progression"
    return "mixed buildup"


def _compare_team_chance_creation(context: dict[str, Any]) -> str:
    """Render an ambiguity-aware comparison when multiple teams are in scope."""

    team_names = context["team_names"]
    if len(team_names) < 2:
        team_name = team_names[0]
        team_stats = context["team_stats"][team_name]
        return (
            f"{team_name} created {team_stats['shots']} shots from {team_stats['passes']} passes, "
            f"{team_stats['carries']} carries, and {team_stats['xg']:.2f} xG."
        )

    ranked = sorted(
        team_names,
        key=lambda name: (
            -context["team_stats"][name]["shots"],
            -context["team_stats"][name]["xg"],
            name,
        ),
    )
    first_name, second_name = ranked[:2]
    first = context["team_stats"][first_name]
    second = context["team_stats"][second_name]
    first_style = _team_style_label(first)
    second_style = _team_style_label(second)
    return (
        f'The query is ambiguous because the dataset includes {", ".join(team_names)}. '
        f"{first_name} created more chances with {first['shots']} shots and {first['xg']:.2f} xG, driven by {first_style} "
        f"({first['passes']} passes, {first['carries']} carries). "
        f"{second_name} created fewer chances with {second['shots']} shots and {second['xg']:.2f} xG, also from {second_style} "
        f"({second['passes']} passes, {second['carries']} carries)."
    )


def _top_player_for_metric(context: dict[str, Any], metric: str, team_name: str | None = None) -> tuple[str, int] | None:
    """Return the leading player for a deterministic metric."""

    candidates = []
    for player_name, stats in context["player_stats"].items():
        if team_name and stats["team_name"] != team_name:
            continue
        value = int(stats.get(metric, 0))
        if value > 0:
            candidates.append((player_name, value))
    return sorted(candidates, key=lambda item: (-item[1], item[0]))[0] if candidates else None


def _entity_focus_summary(
    profile: dict[str, bool],
    entities: dict[str, list[str]],
    context: dict[str, Any],
    stats: dict[str, Any],
) -> str:
    """Build a direct answer using deterministic match and evidence stats."""

    team_name = entities["teams"][0] if entities["teams"] else None
    player_name = entities["players"][0] if entities["players"] else None
    pattern = _sequence_pattern_label(stats)

    if profile["uses_ambiguous_team_ref"] and not team_name and not player_name:
        if profile["asks_how"] or profile["asks_shot"] or profile["asks_summary"]:
            return _compare_team_chance_creation(context)
        return (
            f'The team reference is ambiguous because the indexed dataset includes {", ".join(context["team_names"])}.'
        )

    if profile["asks_summary"] and not team_name and not player_name:
        return (
            f"The indexed match data covers {', '.join(context['team_names'])} with {context['total_events']} events, "
            f"{context['total_shots']} shots, and {context['total_xg']:.2f} total xG."
        )

    if profile["asks_who"] and profile["asks_shot"]:
        top_shooter = _top_player_for_metric(context=context, metric="shots", team_name=team_name)
        if top_shooter:
            scope = f" for {team_name}" if team_name else ""
            return f"{top_shooter[0]} recorded the most shots{scope} ({top_shooter[1]})."
    if profile["asks_who"] and profile["asks_pass"]:
        top_passer = _top_player_for_metric(context=context, metric="passes", team_name=team_name)
        if top_passer:
            scope = f" for {team_name}" if team_name else ""
            return f"{top_passer[0]} recorded the most passes{scope} ({top_passer[1]})."
    if profile["asks_who"] and profile["asks_carry"]:
        top_carrier = _top_player_for_metric(context=context, metric="carries", team_name=team_name)
        if top_carrier:
            scope = f" for {team_name}" if team_name else ""
            return f"{top_carrier[0]} recorded the most carries{scope} ({top_carrier[1]})."
    if profile["asks_who"] and profile["asks_recovery"]:
        top_recoverer = _top_player_for_metric(context=context, metric="recoveries", team_name=team_name)
        if top_recoverer:
            scope = f" for {team_name}" if team_name else ""
            return f"{top_recoverer[0]} recorded the most ball recoveries{scope} ({top_recoverer[1]})."
    if profile["asks_who"]:
        top_involved = _top_player_for_metric(context=context, metric="events", team_name=team_name)
        if top_involved:
            scope = f" for {team_name}" if team_name else ""
            return f"{top_involved[0]} had the highest event involvement{scope} ({top_involved[1]} events)."

    if player_name and player_name in context["player_stats"]:
        player_stats = context["player_stats"][player_name]
        if profile["asks_shot"]:
            return f"{player_name} recorded {player_stats['shots']} shots in the indexed match data."
        if profile["asks_pass"]:
            return f"{player_name} recorded {player_stats['passes']} passes in the indexed match data."
        if profile["asks_carry"]:
            return f"{player_name} recorded {player_stats['carries']} carries in the indexed match data."
        if profile["asks_recovery"]:
            return f"{player_name} recorded {player_stats['recoveries']} ball recoveries in the indexed match data."
        if profile["asks_when"]:
            return (
                f"{player_name} appears in the indexed data from minute {player_stats['minute_min']} "
                f"to minute {player_stats['minute_max']}."
            )
        return (
            f"{player_name} was involved in {player_stats['events']} events and {player_stats['sequences']} indexed sequences "
            f"for {player_stats['team_name']}."
        )

    if team_name and team_name in context["team_stats"]:
        team_stats = context["team_stats"][team_name]
        team_style = _team_style_label(team_stats)
        if profile["asks_how"] or profile["asks_why"]:
            return (
                f"{team_name} creates chances mainly through {team_style}, producing {team_stats['shots']} shots and "
                f"{team_stats['xg']:.2f} xG from {team_stats['passes']} passes and {team_stats['carries']} carries."
            )
        if profile["asks_shot"]:
            return f"{team_name} recorded {team_stats['shots']} shots and {team_stats['xg']:.2f} xG in the indexed match data."
        if profile["asks_pass"]:
            return f"{team_name} recorded {team_stats['passes']} passes in the indexed match data."
        if profile["asks_carry"]:
            return f"{team_name} recorded {team_stats['carries']} carries in the indexed match data."
        if profile["asks_recovery"]:
            return f"{team_name} recorded {team_stats['recoveries']} ball recoveries in the indexed match data."
        if profile["asks_when"]:
            return f"{team_name}'s indexed events run from minute {team_stats['minute_min']} to minute {team_stats['minute_max']}."
        return (
            f"{team_name} produced {team_stats['sequences']} indexed sequences and the retrieved evidence points to {pattern}."
        )

    if profile["asks_team"] and not entities["teams"]:
        return f"The indexed match data includes {', '.join(context['team_names'])}."
    if profile["asks_xg"]:
        return f"The indexed match data totals {context['total_xg']:.2f} xG across both teams."
    if profile["asks_shot"]:
        return f"The indexed match data contains {context['total_shots']} shots overall."
    if profile["asks_pass"]:
        return f"The indexed match data contains {context['total_passes']} passes overall."
    if profile["asks_carry"]:
        return f"The indexed match data contains {context['total_carries']} carries overall."
    if profile["asks_recovery"]:
        return f"The indexed match data contains {context['total_recoveries']} ball recoveries overall."
    if profile["asks_when"]:
        return f"The retrieved evidence spans minutes {stats['minute_min']} to {stats['minute_max']}."
    if profile["asks_summary"] or profile["asks_how"] or profile["asks_why"]:
        return (
            f"The indexed match data suggests {pattern}, with {stats['shot_sequences']} of {max(stats['sequence_count'], 1)} "
            f"retrieved sequences ending in a shot."
        )
    return (
        f"The retrieved evidence covers {stats['evidence_count']} relevant documents across {', '.join(stats['team_names']) or 'the match'}."
    )


def _build_analysis_bundle(query: str, evidence: list[dict[str, Any]], index_data: dict[str, Any]) -> dict[str, Any]:
    """Build reusable generic analysis for grounded answers."""

    context = index_data["context"]
    entities = extract_query_entities(query, index_data=index_data)
    profile = _question_profile(query)
    stats = build_evidence_stats(evidence=evidence, index_data=index_data, query=query)
    evidence_snapshots = [item["summary"] for item in evidence[:4]]
    dominant_team = stats["dominant_team"] or (entities["teams"][0] if entities["teams"] else None)
    sequence_pattern = _sequence_pattern_label(stats)
    direct_answer = _entity_focus_summary(profile=profile, entities=entities, context=context, stats=stats)
    focus_entities = ", ".join([*entities["teams"], *entities["players"]]) or "none explicitly named"

    return {
        "query": query,
        "profile": profile,
        "entities": entities,
        "stats": stats,
        "direct_answer": direct_answer,
        "sequence_pattern": sequence_pattern,
        "dominant_team": dominant_team,
        "focus_entities": focus_entities,
        "evidence_snapshots": evidence_snapshots,
        "top_players": stats["top_players"],
        "context": context,
    }


def _render_analysis_answer(analysis: dict[str, Any]) -> str:
    """Render an in-depth deterministic answer from the analysis bundle."""

    stats = analysis["stats"]
    context = analysis["context"]
    top_players = analysis["top_players"]
    player_line = _format_top_items(top_players[:3], fallback="No clear player concentration")
    evidence_lines = (
        [f"- Evidence {index + 1}: {line}" for index, line in enumerate(analysis["evidence_snapshots"])]
        if analysis["evidence_snapshots"]
        else ["- No evidence snapshots are available."]
    )
    return "\n".join(
        [
            f"Direct Answer: {analysis['direct_answer']}",
            "Grounded Match Analysis:",
            f"- Query focus: {analysis['focus_entities']}.",
            f"- Match teams: {', '.join(context['team_names'])}.",
            f"- Retrieved document count: {stats['evidence_count']}.",
            f"- Retrieved sequence count: {stats['sequence_count']}.",
            f"- Sequence pattern: {analysis['sequence_pattern']}.",
            f"- Shot-ending sequences: {stats['shot_sequences']} of {stats['sequence_count']}.",
            f"- Attacking-third reach: {stats['attacking_third_hits']} of {stats['sequence_count']}.",
            f"- Pass-led sequences: {stats['pass_sequences']} of {stats['sequence_count']}.",
            f"- Carry-led sequences: {stats['carry_sequences']} of {stats['sequence_count']}.",
            f"- Match totals: {context['total_shots']} shots, {context['total_passes']} passes, {context['total_carries']} carries, {context['total_recoveries']} recoveries, {context['total_xg']:.2f} xG.",
            f"- Most represented players in retrieved evidence: {player_line}.",
            f"- Evidence minute range: {stats['minute_min']} to {stats['minute_max']}.",
            "Evidence Snapshots:",
            *evidence_lines,
            "Scope Note: This answer is grounded in indexed match data and retrieved evidence from this dataset only.",
        ]
    )


def compose_grounded_answer(query: str, evidence: list[dict[str, Any]], index_data: dict[str, Any]) -> str:
    """Compose a deterministic grounded answer from retrieved evidence only."""

    if not evidence:
        return f'No grounded evidence was found for "{query}".'
    analysis = _build_analysis_bundle(query=query, evidence=evidence, index_data=index_data)
    return _render_analysis_answer(analysis=analysis)


def build_grounded_prompt_json(
    query: str,
    evidence: list[dict[str, Any]],
    analysis: dict[str, Any],
    extra_instruction: str = "",
) -> str:
    """Build a strict JSON-only prompt for generic grounded match answers."""

    stats = analysis["stats"]
    context = analysis["context"]
    evidence_lines = []
    for item in evidence:
        evidence_lines.append(
            f"- doc_type={item['doc_type']}; team={item.get('team_name') or 'unknown'}; "
            f"player={item.get('player_name') or 'n/a'}; minute_range={item.get('minute_start')}:{item.get('minute_end')}; "
            f"summary={item['summary']}"
        )
    return "\n".join(
        [
            "You are a grounded football analyst answering questions about one indexed match dataset.",
            "Return ONLY valid JSON. No markdown. No extra text.",
            "Use ONLY the evidence and deterministic facts below.",
            "Do not use outside knowledge.",
            "Do not invent teams, players, scores, metrics, timelines, or conclusions not supported here.",
            "If the question says 'this team' or 'that team' and multiple teams are present without a named team, explicitly say the reference is ambiguous and compare the teams briefly instead of choosing one.",
            "Keep the direct answer aligned with the deterministic direct answer candidate when it already answers the question safely.",
            "Required JSON schema:",
            "{"
            '"direct_answer": string, '
            '"analysis": string, '
            '"evidence_points": [string, string, string], '
            '"scope_note": string'
            "}",
            'The "scope_note" value must be exactly: "This answer is grounded in indexed match data and retrieved evidence from this dataset only."',
            f"Question: {query}",
            f"Deterministic direct answer candidate: {analysis['direct_answer']}",
            f"Retrieved evidence stats: evidence_count={stats['evidence_count']}, sequence_count={stats['sequence_count']}, shot_sequences={stats['shot_sequences']}, attacking_third_hits={stats['attacking_third_hits']}, pass_sequences={stats['pass_sequences']}, carry_sequences={stats['carry_sequences']}, minute_min={stats['minute_min']}, minute_max={stats['minute_max']}.",
            f"Match totals: total_events={context['total_events']}, total_sequences={context['total_sequences']}, total_shots={context['total_shots']}, total_passes={context['total_passes']}, total_carries={context['total_carries']}, total_recoveries={context['total_recoveries']}, total_xg={context['total_xg']}.",
            f"Teams in match: {context['team_names']}.",
            f"Top players in evidence: {analysis['top_players']}.",
            extra_instruction,
            "Evidence:",
            *evidence_lines,
        ]
    )


def compose_llm_grounded_answer(
    query: str,
    evidence: list[dict[str, Any]],
    analysis: dict[str, Any],
    extra_instruction: str = "",
) -> str:
    """Compose a grounded JSON answer with an LLM using only the retrieved evidence."""

    if not evidence:
        return f'No grounded evidence was found for "{query}".'
    prompt = build_grounded_prompt_json(
        query=query,
        evidence=evidence,
        analysis=analysis,
        extra_instruction=extra_instruction,
    )
    request_body = json.dumps(
        {
            "model": os.getenv("OLLAMA_MODEL", "gemma4:26b"),
            "prompt": prompt,
            "format": "json",
            "stream": False,
        }
    ).encode("utf-8")
    ollama_url = f"{os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434').rstrip('/')}/api/generate"
    request = urllib.request.Request(
        url=ollama_url,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace").strip()
        detail = error_body or exc.reason
        raise ValueError(f"Ollama request failed at {ollama_url}: HTTP {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise ValueError(f"Ollama is not available at {ollama_url}: {reason}") from exc
    return str(payload.get("response", "")).strip()


def parse_and_validate_llm_json(raw_text: str) -> dict[str, Any]:
    """Parse and validate the generic LLM JSON payload."""

    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```json\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("LLM JSON parse failed: no JSON object found")
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM JSON parse failed: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("LLM JSON payload is not an object")

    payload.setdefault("direct_answer", "")
    payload.setdefault("analysis", "")
    payload.setdefault("evidence_points", [])
    payload.setdefault(
        "scope_note",
        "This answer is grounded in indexed match data and retrieved evidence from this dataset only.",
    )

    if not isinstance(payload["direct_answer"], str) or not payload["direct_answer"].strip():
        raise ValueError("LLM JSON direct_answer missing")
    if not isinstance(payload["analysis"], str) or not payload["analysis"].strip():
        raise ValueError("LLM JSON analysis missing")
    if not isinstance(payload["evidence_points"], list):
        raise ValueError("LLM JSON evidence_points is not a list")

    evidence_points = []
    for item in payload["evidence_points"][:4]:
        if isinstance(item, str) and item.strip():
            evidence_points.append(item.strip())
    if len(evidence_points) < 2:
        raise ValueError("LLM JSON evidence_points too short")
    payload["evidence_points"] = evidence_points

    blocked_patterns = ["outside the dataset", "league average", "season average"]
    lowered_fields = " ".join(
        [payload["direct_answer"], payload["analysis"], *payload["evidence_points"], payload["scope_note"]]
    ).lower()
    for pattern in blocked_patterns:
        if pattern in lowered_fields:
            raise ValueError(f"LLM JSON unsupported phrase: {pattern}")

    exact_scope = "This answer is grounded in indexed match data and retrieved evidence from this dataset only."
    if payload["scope_note"] != exact_scope:
        payload["scope_note"] = exact_scope
    return payload


def render_grounded_llm_answer(valid_json: dict[str, Any]) -> str:
    """Render a deterministic answer shell from validated generic LLM JSON."""

    evidence_lines = [f"- {item}" for item in valid_json["evidence_points"]]
    return "\n".join(
        [
            f"Answer: {valid_json['direct_answer']}",
            f"Grounded Reasoning: {valid_json['analysis']}",
            "Evidence Summary:",
            *evidence_lines,
            f"Scope Note: {valid_json['scope_note']}",
        ]
    )


def run_query(
    query: str,
    index_data: dict[str, Any],
    trace: dict[str, Any],
    top_k: int = 5,
    use_llm: bool = False,
    llm_required: bool = False,
) -> QueryResponse:
    """Run retrieval and grounded answer generation for one user query."""

    evidence = retrieve(query=query, index_data=index_data, top_k=top_k)
    generation_mode = "template"
    llm_validated = False
    llm_validation_errors: list[str] = []
    llm_retry_used = False
    llm_output_format: str | None = None
    llm_fallback = False
    llm_failure_reason: str | None = None
    entities = extract_query_entities(query, index_data=index_data)
    analysis = _build_analysis_bundle(query=query, evidence=evidence, index_data=index_data) if evidence else None

    if use_llm and evidence and analysis:
        try:
            raw_answer = compose_llm_grounded_answer(query=query, evidence=evidence, analysis=analysis)
            llm_output_format = "json"
            try:
                valid_json = parse_and_validate_llm_json(raw_text=raw_answer)
                answer = render_grounded_llm_answer(valid_json)
                llm_validated = True
            except ValueError as exc:
                llm_validation_errors = [str(exc)]
                llm_retry_used = True
                retry_instruction = "Return valid JSON only. Keep claims tightly grounded in the listed evidence."
                raw_answer = compose_llm_grounded_answer(
                    query=query,
                    evidence=evidence,
                    analysis=analysis,
                    extra_instruction=retry_instruction,
                )
                valid_json = parse_and_validate_llm_json(raw_text=raw_answer)
                answer = render_grounded_llm_answer(valid_json)
                llm_validated = True
                llm_validation_errors = []
            if llm_validated:
                generation_mode = "llm"
            else:
                llm_failure_reason = "; ".join(llm_validation_errors) or "llm validation failed"
                if llm_required:
                    raise ValueError(f"LLM generation failed validation and llm_required=true: {llm_failure_reason}")
                answer = compose_grounded_answer(query=query, evidence=evidence, index_data=index_data)
                llm_fallback = True
        except Exception as exc:
            llm_failure_reason = llm_failure_reason or str(exc)
            if llm_required:
                raise ValueError(f"LLM generation failed validation and llm_required=true: {llm_failure_reason}") from exc
            answer = compose_grounded_answer(query=query, evidence=evidence, index_data=index_data)
            llm_fallback = True
    else:
        answer = compose_grounded_answer(query=query, evidence=evidence, index_data=index_data)

    if not evidence:
        answer = f'No grounded evidence was found for "{query}".'

    return QueryResponse(
        answer=answer,
        evidence=evidence,
        trace={
            **trace,
            "retrieval_results": len(evidence),
            "use_llm": use_llm,
            "generation_mode": generation_mode,
            "llm_required": llm_required,
            "llm_output_format": llm_output_format,
            "llm_validated": llm_validated,
            "llm_validation_errors": llm_validation_errors,
            "llm_retry_used": llm_retry_used,
            "llm_fallback": llm_fallback,
            "llm_failure_reason": llm_failure_reason,
            "query_teams": entities["teams"],
            "query_players": entities["players"],
            "available_teams": index_data["context"]["team_names"],
            "total_match_events": index_data["context"]["total_events"],
            "total_match_sequences": index_data["context"]["total_sequences"],
        },
    )
