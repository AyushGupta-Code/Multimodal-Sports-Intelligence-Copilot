"""Retrieval planning and ranking."""

from __future__ import annotations

import re
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from app.models.domain import DocumentRecord, EventRecord, MatchFacts, SequenceRecord
from app.services.match_features import build_documents, build_match_facts


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
        if len(token) >= 4 and re.search(rf"\b{re.escape(token)}(?:'s)?\b", lowered_query):
            return True
    return False


def _query_text(query: str) -> str:
    """Expand prompts into retrieval-friendly keywords."""

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
        "goal": ["score", "winner"],
        "winner": ["score", "goal", "result"],
        "score": ["goal", "winner", "result"],
        "xg": ["shot", "chance"],
        "player": ["involvement", "sequence"],
        "team": ["match", "possession"],
        "when": ["minute", "period", "timeline"],
        "timeline": ["minute", "period", "sequence"],
        "card": ["booking", "yellow", "red"],
        "substitution": ["replacement", "change"],
        "first half": ["period 1"],
        "second half": ["period 2"],
    }
    for phrase, terms in keyword_map.items():
        if phrase in lowered:
            extra_terms.extend(terms)
    return " ".join([lowered, *extra_terms]).strip()


def _index_text(document: DocumentRecord) -> str:
    """Build the text used for TF-IDF indexing."""

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


def build_query_profile(query: str) -> dict[str, bool]:
    """Extract lightweight intent flags from the user question."""

    lowered = query.strip().lower()
    return {
        "asks_who": bool(re.search(r"\bwho\b|\bwhich player\b|\bwhich team\b", lowered)),
        "asks_how": bool(re.search(r"\bhow\b", lowered)),
        "asks_why": bool(re.search(r"\bwhy\b", lowered)),
        "asks_when": bool(re.search(r"\bwhen\b|\bminute\b|\btimeline\b|\bfirst half\b|\bsecond half\b|\bperiod\b", lowered)),
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
        "asks_winner": "who won" in lowered or "winner" in lowered or "did " in lowered and " win" in lowered,
        "asks_score": "score" in lowered or "scoreline" in lowered,
        "asks_goal": "goal" in lowered or "scorer" in lowered,
        "asks_card": "card" in lowered or "yellow" in lowered or "red" in lowered,
        "asks_substitution": "substitution" in lowered or "sub" in lowered or "replacement" in lowered,
        "uses_ambiguous_team_ref": "this team" in lowered or "that team" in lowered,
        "uses_follow_up_ref": bool(re.search(r"\bthey\b|\bthem\b|\bhe\b|\bhim\b|\bhis\b|\bthat player\b|\bthose sequences\b", lowered)),
    }


def extract_query_entities(query: str, index_data: dict[str, Any]) -> dict[str, list[str]]:
    """Detect referenced teams and players from indexed match facts."""

    lowered_query = query.lower()
    facts: MatchFacts = index_data["facts"]
    matched_teams = [team for team in facts.team_names if _name_in_query(team, lowered_query)]
    matched_players = [player for player in facts.player_names if _name_in_query(player, lowered_query)]
    return {"teams": matched_teams, "players": matched_players}


def _target_doc_types(profile: dict[str, bool]) -> set[str]:
    """Return preferred document types for the current question."""

    if profile["asks_winner"] or profile["asks_score"]:
        return {"match_overview", "goal_event", "team_summary"}
    if profile["asks_goal"]:
        return {"goal_event", "player_summary", "team_summary", "sequence"}
    if profile["asks_card"]:
        return {"card_event", "player_summary", "team_summary"}
    if profile["asks_substitution"]:
        return {"substitution_event", "player_summary", "team_summary"}
    if profile["uses_ambiguous_team_ref"]:
        return {"team_summary", "sequence"}
    if profile["asks_how"] or profile["asks_why"]:
        return {"sequence", "team_summary", "player_summary"}
    if profile["asks_when"]:
        return {"goal_event", "card_event", "substitution_event", "sequence", "player_summary", "team_summary"}
    if profile["asks_who"] and profile["asks_player"]:
        return {"player_summary", "team_summary", "goal_event"}
    if profile["asks_summary"]:
        return {"match_overview", "team_summary", "sequence"}
    return {"sequence", "team_summary", "player_summary", "match_overview"}


def _doc_matches_entities(document: DocumentRecord, entities: dict[str, list[str]]) -> bool:
    """Return whether a document matches at least one explicit query entity."""

    if entities["teams"] and document.team_name in entities["teams"]:
        return True
    if entities["players"] and (
        document.player_name in entities["players"] or any(player in entities["players"] for player in document.players)
    ):
        return True
    return False


def build_index(events: list[EventRecord], sequences: list[SequenceRecord]) -> dict[str, Any]:
    """Build the in-memory retrieval index over mixed match documents."""

    if not events:
        raise ValueError("No events available to index")
    facts = build_match_facts(events=events, sequences=sequences)
    documents = build_documents(events=events, sequences=sequences, facts=facts)
    texts = [_index_text(document) for document in documents]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    return {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "documents": documents,
        "facts": facts,
    }


def retrieve(query: str, index_data: dict[str, Any], top_k: int = 5) -> list[dict[str, Any]]:
    """Retrieve the most relevant mixed evidence documents."""

    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("Query must not be empty")

    expanded_query = _query_text(cleaned_query)
    profile = build_query_profile(cleaned_query)
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
            adjusted_score += 0.14
        if profile["uses_ambiguous_team_ref"] and document.doc_type == "team_summary":
            adjusted_score += 0.26
        if profile["uses_ambiguous_team_ref"] and document.doc_type == "match_overview":
            adjusted_score -= 0.18
        if profile["asks_how"] and document.doc_type == "sequence":
            adjusted_score += 0.14
        if profile["asks_when"] and document.doc_type in {"goal_event", "card_event", "substitution_event", "sequence"}:
            adjusted_score += 0.18
        if entities["players"] and document.doc_type == "player_summary":
            adjusted_score += 0.18
        if entities["teams"] and document.doc_type == "team_summary":
            adjusted_score += 0.14
        if document.doc_type == "match_overview" and (profile["asks_summary"] or profile["asks_winner"] or profile["asks_score"]):
            adjusted_score += 0.22
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
                "metadata": document.metadata,
            }
        )
    return results
