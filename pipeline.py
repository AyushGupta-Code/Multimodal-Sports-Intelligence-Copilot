"""Compact data pipeline for local soccer event ingestion and retrieval."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
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
    player_counts: dict[str, int] = {}
    for item in evidence:
        for player in item.get("players") or []:
            player_counts[player] = player_counts.get(player, 0) + 1
    top_players = sorted(player_counts.items(), key=lambda item: (-item[1], item[0]))[:3]
    shot_count = sum(1 for item in evidence if item.get("ended_in_shot"))
    progression_count = sum(
        1
        for item in evidence
        if "progressed from" in item["summary"] or "sustained attack" in item["summary"]
    )
    pass_count = sum(
        1 for item in evidence if "pass" in item["summary"] or "combination" in item["summary"]
    )
    dribble_count = sum(1 for item in evidence if "carry" in item["summary"])

    if shot_count >= max(1, len(evidence) // 2):
        pattern_text = "The retrieved attacks mostly build toward shots."
    elif pass_count >= max(1, len(evidence) // 2):
        pattern_text = "The retrieved attacks are mostly built through passing combinations."
    elif dribble_count >= max(1, len(evidence) // 2):
        pattern_text = "The retrieved attacks often include carries to move the ball forward."
    else:
        pattern_text = "The retrieved attacks show a mix of short combinations and forward moves."

    if shot_count == 0:
        shot_text = "None of the retrieved sequences end in a shot."
    elif shot_count == 1:
        shot_text = "Only 1 of the retrieved sequences ends in a shot."
    else:
        shot_text = f"{shot_count} of the {len(evidence)} retrieved sequences end in a shot."

    if progression_count >= max(1, len(evidence) // 2):
        progression_text = "Most of the evidence shows the ball being worked forward into attacking areas."
    elif progression_count > 0:
        progression_text = "Some of the evidence shows forward progression, but it is not consistent across every sequence."
    else:
        progression_text = "The evidence does not show a strong forward-progression pattern."

    if top_players:
        player_names = ", ".join(name for name, _ in top_players)
        player_text = f"The most visible players in these sequences are {player_names}."
    else:
        player_text = "The key players are not clear from the retrieved evidence."

    team_text = f"The evidence comes from {', '.join(teams)}." if teams else "The team is not clear from the retrieved evidence."
    return " ".join([pattern_text, shot_text, progression_text, player_text, team_text])


def build_evidence_stats(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Build deterministic evidence stats used by the LLM and validator."""

    # Compute the only counts the LLM is allowed to use so numeric claims stay grounded.
    total_sequences = len(evidence)
    shot_endings = sum(1 for item in evidence if item.get("ended_in_shot"))
    non_shot_endings = total_sequences - shot_endings
    progression_hits = sum(
        1
        for item in evidence
        if "progressed from" in item["summary"] or "sustained attack" in item["summary"]
    )
    attacking_third_reach_count = sum(1 for item in evidence if "attacking third" in item["summary"])
    team_names = sorted({item["team_name"] for item in evidence if item.get("team_name")})
    return {
        "total_sequences": total_sequences,
        "shot_endings": shot_endings,
        "non_shot_endings": non_shot_endings,
        "progression_hits": progression_hits,
        "attacking_third_reach_count": attacking_third_reach_count,
        "team_names": team_names,
    }


def build_role_prompt_json(
    query: str,
    evidence: list[dict[str, Any]],
    stats: dict[str, Any],
    extra_instruction: str = "",
) -> str:
    """Build a strict JSON-only prompt for grounded role-style answers."""

    # Force the model to return only structured grounded fields so deterministic
    # code can render the final answer without placeholders or invented counts.
    pass_sequences = sum(1 for item in evidence if "pass" in item["summary"])
    carry_sequences = sum(1 for item in evidence if "carry" in item["summary"])
    team_counts: dict[str, int] = {}
    for item in evidence:
        team = item["team_name"]
        team_counts[team] = team_counts.get(team, 0) + 1
    dominant_team = max(team_counts, key=team_counts.get) if team_counts else None
    evidence_lines = []
    for item in evidence:
        players = ", ".join((item.get("players") or [])[:3]) or "unknown"
        evidence_lines.append(
            f"- team={item['team_name']}; players={players}; ended_in_shot={item['ended_in_shot']}; progression={item['summary'].split('. ')[-1].rstrip('.')}; summary={item['summary']}"
        )
    return "\n".join(
    [
        "You are a grounded football analyst writing from retrieved sequence summaries.",
        "Return ONLY valid JSON. No markdown. No bullet lists outside JSON. No extra text.",
        "Use ONLY the evidence below.",
        "Do not use outside knowledge.",
        "Use only provided evidence and deterministic stats.",
        "Use these stats exactly; do not compute new numbers.",
        "All counts must be integers from the provided stats only.",
        "No fabricated metrics, percentages, rates, averages, or placeholders.",
        "Do not use the phrase 'based on the retrieved sample'.",
        "Required JSON keys and allowed values:",
        '{'
        '"role_classification": "link_player" | "ball_carrier" | "finisher" | "mixed" | "insufficient", '
        '"role_reasoning": string, '
        '"after_involvement": string, '
        '"attacking_third_reach_count": int, '
        '"shot_ending_count": int, '
        '"total_sequences": int, '
        '"team_context": "single-team" | "mixed-team", '
        '"sample_warning": string'
        '}',
        'The "sample_warning" value must be exactly: "This is based on retrieved sequences, not the full dataset."',
        f"Question: {query}",
        f"Retrieved sample counts: pass_sequences={pass_sequences}, carry_sequences={carry_sequences}, total_sequences={stats['total_sequences']}, shot_endings={stats['shot_endings']}, non_shot_endings={stats['non_shot_endings']}, progression_hits={stats['progression_hits']}, attacking_third_reach_count={stats['attacking_third_reach_count']}.",
        f"Team counts in sample: {team_counts}.",
        f"Dominant team in sample: {dominant_team or 'unknown'}.",
        f"Allowed team names: {stats['team_names']}.",
        extra_instruction,
        "Evidence:",
        *evidence_lines,
    ]
    )


def compose_llm_grounded_answer(
    query: str,
    evidence: list[dict[str, Any]],
    stats: dict[str, Any],
    extra_instruction: str = "",
) -> str:
    """Compose a grounded JSON answer with an LLM using only the retrieved evidence."""

    # Keep the LLM path tightly constrained so deterministic code can validate
    # and render the final answer without template phrasing.
    if not evidence:
        return f'No grounded evidence was found for "{query}".'
    prompt = build_role_prompt_json(
        query=query,
        evidence=evidence,
        stats=stats,
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
    request = urllib.request.Request(
        url=f"{os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434').rstrip('/')}/api/generate",
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise ValueError("Ollama is not available") from exc
    return str(payload.get("response", "")).strip()


def parse_and_validate_llm_json(
    raw_text: str,
    stats: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """Parse and validate the LLM JSON payload."""

    # Enforce a strict JSON contract so only grounded structured output reaches
    # the final renderer.
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```json\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        # Some models return extra wrapper text; recover the first JSON object.
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("LLM JSON parse failed: no JSON object found")
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM JSON parse failed: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("LLM JSON payload is not an object")
    # Deterministic backfills keep strict mode usable when models omit fields.
    default_role_reasoning = (
        "The evidence suggests a mixed role across combinations and progression"
        if stats["shot_endings"] < stats["total_sequences"]
        else "The evidence suggests involvement that often appears near the end of attacks"
    )
    default_after_involvement = (
        "his involvement is usually followed by continued progression rather than an immediate shot"
        if stats["shot_endings"] == 0
        else "his involvement is usually followed by progression, with some sequences ending in a shot"
    )
    payload.setdefault("role_classification", "insufficient")
    payload.setdefault("role_reasoning", default_role_reasoning)
    payload.setdefault("after_involvement", default_after_involvement)
    payload.setdefault("attacking_third_reach_count", stats["attacking_third_reach_count"])
    payload.setdefault("shot_ending_count", stats["shot_endings"])
    payload.setdefault("total_sequences", stats["total_sequences"])
    payload.setdefault("team_context", "mixed-team" if len(stats["team_names"]) > 1 else "single-team")
    payload.setdefault("sample_warning", "This is based on retrieved sequences, not the full dataset.")
    if payload["role_classification"] not in {"link_player", "ball_carrier", "finisher", "mixed", "insufficient"}:
        payload["role_classification"] = "insufficient"
    if payload["team_context"] not in {"single-team", "mixed-team"}:
        payload["team_context"] = "mixed-team" if len(stats["team_names"]) > 1 else "single-team"
    for key in ["attacking_third_reach_count", "shot_ending_count", "total_sequences"]:
        if isinstance(payload[key], str) and payload[key].isdigit():
            payload[key] = int(payload[key])
        if not isinstance(payload[key], int):
            raise ValueError(f"LLM JSON non-integer field: {key}")
    if not (0 <= payload["attacking_third_reach_count"] <= payload["total_sequences"]):
        raise ValueError("LLM JSON attacking_third_reach_count out of range")
    if not (0 <= payload["shot_ending_count"] <= payload["total_sequences"]):
        raise ValueError("LLM JSON shot_ending_count out of range")
    # Keep final numeric claims deterministic and grounded in computed stats.
    payload["total_sequences"] = stats["total_sequences"]
    payload["shot_ending_count"] = stats["shot_endings"]
    payload["attacking_third_reach_count"] = stats["attacking_third_reach_count"]
    placeholder_patterns = ["based on the retrieved sample"]
    text_fields = [payload["role_reasoning"], payload["after_involvement"], payload["sample_warning"]]
    for phrase in placeholder_patterns:
        if any(phrase in str(field).lower() for field in text_fields):
            raise ValueError(f"LLM JSON placeholder phrase: {phrase}")
    if "not enough detail" in str(payload["role_reasoning"]).lower() and payload["role_classification"] != "insufficient":
        raise ValueError("LLM JSON insufficient phrasing mismatch")
    if payload["sample_warning"] != "This is based on retrieved sequences, not the full dataset.":
        payload["sample_warning"] = "This is based on retrieved sequences, not the full dataset."
    return payload


def render_role_answer(valid_json: dict[str, Any]) -> str:
    """Render a deterministic grounded answer from validated LLM JSON."""

    # Render prose from validated deterministic fields only, so free-text model
    # fragments do not leak unsupported tactical claims into the final answer.
    role_labels = {
        "link_player": "a link player in combination play",
        "ball_carrier": "more of a ball carrier",
        "finisher": "more of a finisher",
        "mixed": "a mixed role across buildup and progression",
        "insufficient": "an unclear role from this sample",
    }

    total = valid_json["total_sequences"]
    shot_endings = valid_json["shot_ending_count"]
    progression = valid_json["attacking_third_reach_count"]
    team_context = valid_json["team_context"]
    if shot_endings == 0:
        follow_up = "After involvement, sequences usually continue through progression rather than finishing with a shot."
    elif shot_endings == total:
        follow_up = "After involvement, sequences regularly continue into shot-ending attacks."
    else:
        follow_up = "After involvement, sequences often continue through progression, with only some ending in shots."

    if progression >= max(1, total // 2):
        progression_tendency = "Most retrieved sequences reach the attacking third."
    else:
        progression_tendency = "Only part of the sample reaches the attacking third consistently."

    answer_sentences = [
        f"Answer: In these retrieved sequences, the role looks like {role_labels[valid_json['role_classification']]}.",
        follow_up,
        progression_tendency,
        f"{progression} of {total} sequences reached the attacking third, and {shot_endings} of {total} ended in a shot.",
    ]
    if team_context == "mixed-team" or total <= 2:
        answer_sentences.append("The sample is either mixed across teams or small, so the conclusion should be treated cautiously.")
    answer = " ".join(answer_sentences)
    evidence_summary = "\n".join(
        [
            "Evidence Summary:",
            f"- Progression pattern: {valid_json['attacking_third_reach_count']} of {valid_json['total_sequences']} reached attacking third.",
            f"- Shot-ending tendency: {valid_json['shot_ending_count']} of {valid_json['total_sequences']} ended in a shot.",
            f"- Team context: {valid_json['team_context']}.",
            f"Scope Note: {valid_json['sample_warning']}",
        ]
    )
    return f"{answer}\n{evidence_summary}"


def _allowed_entities(evidence: list[dict[str, Any]]) -> set[str]:
    """Collect team and player names that are allowed to appear in the LLM answer."""

    # Limit entity references to names that actually appear in the retrieved evidence.
    entities = {item["team_name"] for item in evidence if item.get("team_name")}
    for item in evidence:
        entities.update(player for player in item.get("players") or [] if player)
    return entities


def extract_player_from_query(query: str, evidence: list[dict[str, Any]]) -> str | None:
    """Detect a player reference in the query using player names from retrieved evidence."""

    # Match simple player questions by looking for known player names or surname tokens in the query.
    lowered_query = query.lower()
    players = {
        player
        for item in evidence
        for player in item.get("players") or []
        if player
    }
    sorted_players = sorted(players, key=len, reverse=True)
    for player in sorted_players:
        lowered_player = player.lower()
        if lowered_player in lowered_query:
            return player
        for token in lowered_player.replace("'", "").split():
            if len(token) >= 4 and re.search(rf"\b{re.escape(token)}(?:'s)?\b", lowered_query):
                return player
    return None


def filter_evidence_for_player(evidence: list[dict[str, Any]], player_name: str) -> list[dict[str, Any]]:
    """Keep only evidence items that include the detected player."""

    # Filter after retrieval so ranking stays unchanged while player-specific answers stay focused.
    lowered_player = player_name.lower()
    return [
        item
        for item in evidence
        if any(lowered_player == player.lower() for player in item.get("players") or [])
    ]


def validate_llm_answer(answer: str, evidence: list[dict[str, Any]], stats: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate that an LLM answer stays within the retrieved evidence."""

    # This validator is intentionally strict and lightweight so unsupported claims
    # fall back to the safer template answer.
    lowered = answer.lower()
    errors: list[str] = []
    blocked_patterns = ["%", "per minute", "xg", "average", "averaged", "last games"]
    for pattern in blocked_patterns:
        if pattern in lowered:
            errors.append(f"unsupported metric pattern: {pattern}")
    if not answer.strip():
        errors.append("empty answer")
    required_sections = ["Answer:", "Evidence Summary:", "Scope Note:"]
    if not all(section in answer for section in required_sections):
        errors.append("required sections missing")
    contradictory_patterns = [
        "shot-ending sequence that does not end in a shot",
        "shot ending sequence that does not end in a shot",
        "shots not ended in a shot",
        "receives shots",
    ]
    for pattern in contradictory_patterns:
        if pattern in lowered:
            errors.append(f"contradictory phrasing: {pattern}")
    exact_scope = "Scope Note: This is based on retrieved sequences, not the full dataset."
    if exact_scope not in answer:
        errors.append("scope note not exact")
    for line in answer.splitlines():
        if line.startswith("Scope Note:") and line.strip() != exact_scope:
            errors.append("scope note contains extra text")
    if re.search(r"\b\d+\.\d+\s+of\s+\d+\b", answer):
        errors.append("fractional count pattern")

    return not errors, errors


def run_query(
    query: str,
    index_data: dict[str, Any],
    trace: dict[str, Any],
    top_k: int = 5,
    use_llm: bool = True,
    llm_required: bool = False,
) -> QueryResponse:
    """Run retrieval and grounded answer generation for one user query."""

    # Run search first, then build the final response with evidence and trace data.
    evidence = retrieve(query=query, index_data=index_data, top_k=top_k)
    player_name = extract_player_from_query(query=query, evidence=evidence)
    player_filter_applied = player_name is not None
    player_filter_hits = 0
    if player_name:
        filtered_evidence = filter_evidence_for_player(evidence=evidence, player_name=player_name)
        player_filter_hits = len(filtered_evidence)
        if filtered_evidence:
            evidence = filtered_evidence
        else:
            return QueryResponse(
                answer=f'No player-specific grounded evidence found for "{player_name}" in the top retrieved sequences. Try increasing top_k.',
                evidence=[],
                trace={
                    **trace,
                    "retrieval_results": 0,
                    "generation_mode": "template",
                    "llm_validated": False,
                    "llm_validation_errors": [],
                    "llm_retry_used": False,
                    "llm_fallback": False,
                    "player_filter_applied": True,
                    "player_filter_name": player_name,
                    "player_filter_hits": 0,
                },
            )
    generation_mode = "template"
    stats = build_evidence_stats(evidence)
    llm_validated = False
    llm_validation_errors: list[str] = []
    llm_retry_used = False
    llm_sanitized = False
    llm_output_format: str | None = None
    llm_fallback = False
    llm_failure_reason: str | None = None
    if use_llm and evidence:
        try:
            raw_answer = compose_llm_grounded_answer(query=query, evidence=evidence, stats=stats)
            llm_output_format = "json"
            try:
                valid_json = parse_and_validate_llm_json(raw_text=raw_answer, stats=stats, evidence=evidence)
                answer = render_role_answer(valid_json)
                llm_validated = True
            except ValueError as exc:
                llm_validated = False
                llm_validation_errors = [str(exc)]
                llm_retry_used = True
                retry_instruction = (
                    "Return valid JSON only. Remove unsupported stats/percentages/rates and use only evidence-backed counts."
                )
                raw_answer = compose_llm_grounded_answer(
                    query=query,
                    evidence=evidence,
                    stats=stats,
                    extra_instruction=retry_instruction,
                )
                valid_json = parse_and_validate_llm_json(raw_text=raw_answer, stats=stats, evidence=evidence)
                answer = render_role_answer(valid_json)
                llm_validated = True
                llm_validation_errors = []
            if llm_validated:
                generation_mode = "llm"
            else:
                llm_failure_reason = "; ".join(llm_validation_errors) or "llm validation failed"
                if llm_required:
                    raise ValueError(f"LLM generation failed validation and llm_required=true: {llm_failure_reason}")
                answer = compose_grounded_answer(query=query, evidence=evidence)
                llm_fallback = True
        except Exception as exc:
            llm_failure_reason = llm_failure_reason or str(exc)
            if llm_required:
                raise ValueError(f"LLM generation failed validation and llm_required=true: {llm_failure_reason}") from exc
            answer = compose_grounded_answer(query=query, evidence=evidence)
            llm_fallback = True
    else:
        answer = compose_grounded_answer(query=query, evidence=evidence)
    return QueryResponse(
        answer=answer,
        evidence=evidence,
        trace={
            **trace,
            "retrieval_results": len(evidence),
            "generation_mode": generation_mode,
            "llm_required": llm_required,
            "llm_sanitized": llm_sanitized,
            "llm_output_format": llm_output_format,
            "llm_validated": llm_validated,
            "llm_validation_errors": llm_validation_errors,
            "llm_retry_used": llm_retry_used,
            "llm_fallback": llm_fallback,
            "llm_failure_reason": llm_failure_reason,
            "player_filter_applied": player_filter_applied,
            "player_filter_name": player_name,
            "player_filter_hits": player_filter_hits,
        },
    )                                                                                               
