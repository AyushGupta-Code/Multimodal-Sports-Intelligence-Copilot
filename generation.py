"""Deterministic and LLM-backed answer generation."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from typing import Any

from domain_models import MatchFacts
from retrieval_engine import build_query_profile, extract_query_entities, retrieve
from schemas import QueryResponse


def _top_counter_items(counter: Counter[str], limit: int = 5) -> list[tuple[str, int]]:
    """Return counter items in deterministic order."""

    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]


def _format_top_items(items: list[tuple[str, int]], fallback: str = "none") -> str:
    """Render a short top-items string."""

    if not items:
        return fallback
    return ", ".join(f"{name} ({count})" for name, count in items)


def build_evidence_stats(evidence: list[dict[str, Any]], index_data: dict[str, Any], query: str) -> dict[str, Any]:
    """Build deterministic evidence stats for grounded QA."""

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
    facts: MatchFacts = index_data["facts"]
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
        "minute_min": min(minute_values) if minute_values else facts.minute_min,
        "minute_max": max(minute_values) if minute_values else facts.minute_max,
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
    """Summarize a team's chance-creation style."""

    if team_stats["passes"] >= team_stats["carries"] * 1.2:
        return "pass-led buildup"
    if team_stats["carries"] >= team_stats["passes"] * 0.8:
        return "carry-heavy progression"
    return "mixed buildup"


def _compare_team_chance_creation(facts: MatchFacts) -> str:
    """Render an ambiguity-aware comparison when multiple teams are in scope."""

    team_names = facts.team_names
    if len(team_names) < 2:
        team_name = team_names[0]
        team_stats = facts.team_stats[team_name]
        return (
            f"{team_name} created {team_stats['shots']} shots from {team_stats['passes']} passes, "
            f"{team_stats['carries']} carries, and {team_stats['xg']:.2f} xG."
        )

    ranked = sorted(
        team_names,
        key=lambda name: (-facts.team_stats[name]["shots"], -facts.team_stats[name]["xg"], name),
    )
    first_name, second_name = ranked[:2]
    first = facts.team_stats[first_name]
    second = facts.team_stats[second_name]
    return (
        f'The query is ambiguous because the dataset includes {", ".join(team_names)}. '
        f"{first_name} created more chances with {first['shots']} shots and {first['xg']:.2f} xG, driven by {_team_style_label(first)} "
        f"({first['passes']} passes, {first['carries']} carries). "
        f"{second_name} created fewer chances with {second['shots']} shots and {second['xg']:.2f} xG, also from {_team_style_label(second)} "
        f"({second['passes']} passes, {second['carries']} carries)."
    )


def _top_player_for_metric(facts: MatchFacts, metric: str, team_name: str | None = None) -> tuple[str, int] | None:
    """Return the leading player for a deterministic metric."""

    candidates = []
    for player_name, stats in facts.player_stats.items():
        if team_name and stats["team_name"] != team_name:
            continue
        value = int(stats.get(metric, 0))
        if value > 0:
            candidates.append((player_name, value))
    return sorted(candidates, key=lambda item: (-item[1], item[0]))[0] if candidates else None


def _scoreline_text(facts: MatchFacts) -> str:
    """Render a scoreline string."""

    return ", ".join(f"{team} {score}" for team, score in facts.score_by_team.items()) or "unknown scoreline"


def _direct_answer(
    profile: dict[str, bool],
    entities: dict[str, list[str]],
    facts: MatchFacts,
    stats: dict[str, Any],
) -> str:
    """Build the direct answer using deterministic match and evidence stats."""

    team_name = entities["teams"][0] if entities["teams"] else None
    player_name = entities["players"][0] if entities["players"] else None
    pattern = _sequence_pattern_label(stats)

    if profile["asks_winner"]:
        if facts.is_draw:
            return f"The match finished as a draw with scoreline { _scoreline_text(facts) }."
        if facts.winner:
            return f"{facts.winner} won the match. Final scoreline: {_scoreline_text(facts)}."
        return "The winner could not be determined from the indexed match facts."
    if profile["asks_score"]:
        return f"The scoreline was {_scoreline_text(facts)}."
    if profile["asks_goal"]:
        if player_name and player_name in facts.player_stats:
            return f"{player_name} scored {facts.player_stats[player_name]['goals']} goals in the indexed match data."
        if team_name and team_name in facts.team_stats:
            return f"{team_name} scored {facts.team_stats[team_name]['goals']} goals in the indexed match data."
        if facts.goals:
            goal_text = ", ".join(
                f"{goal['player_name']} for {goal['team_name']} at {goal['minute']}'"
                for goal in facts.goals[:4]
                if goal.get("player_name")
            )
            return f"Goals in the match: {goal_text}."
        return "No goals were found in the indexed match data."
    if profile["asks_card"]:
        if not facts.cards:
            return "No cards were found in the indexed match data."
        if team_name:
            team_cards = [card for card in facts.cards if card["team_name"] == team_name]
            return f"{team_name} received {len(team_cards)} cards in the indexed match data."
        return f"The indexed match data contains {len(facts.cards)} cards."
    if profile["asks_substitution"]:
        if not facts.substitutions:
            return "No substitutions were found in the indexed match data."
        if team_name:
            team_subs = [sub for sub in facts.substitutions if sub["team_name"] == team_name]
            return f"{team_name} made {len(team_subs)} substitutions in the indexed match data."
        return f"The indexed match data contains {len(facts.substitutions)} substitutions."

    if profile["uses_ambiguous_team_ref"] and not team_name and not player_name:
        if profile["asks_how"] or profile["asks_shot"] or profile["asks_summary"]:
            return _compare_team_chance_creation(facts)
        return f'The team reference is ambiguous because the indexed dataset includes {", ".join(facts.team_names)}.'

    if profile["asks_summary"] and not team_name and not player_name:
        return (
            f"The indexed match data covers {', '.join(facts.team_names)} with scoreline {_scoreline_text(facts)}, "
            f"{facts.total_events} events, {facts.total_shots} shots, and {facts.total_xg:.2f} total xG."
        )

    if profile["asks_who"] and profile["asks_shot"]:
        top = _top_player_for_metric(facts=facts, metric="shots", team_name=team_name)
        if top:
            return f"{top[0]} recorded the most shots{f' for {team_name}' if team_name else ''} ({top[1]})."
    if profile["asks_who"] and profile["asks_pass"]:
        top = _top_player_for_metric(facts=facts, metric="passes", team_name=team_name)
        if top:
            return f"{top[0]} recorded the most passes{f' for {team_name}' if team_name else ''} ({top[1]})."
    if profile["asks_who"] and profile["asks_goal"]:
        top = _top_player_for_metric(facts=facts, metric="goals", team_name=team_name)
        if top:
            return f"{top[0]} scored the most goals{f' for {team_name}' if team_name else ''} ({top[1]})."
    if profile["asks_who"]:
        top = _top_player_for_metric(facts=facts, metric="events", team_name=team_name)
        if top:
            return f"{top[0]} had the highest event involvement{f' for {team_name}' if team_name else ''} ({top[1]} events)."

    if player_name and player_name in facts.player_stats:
        player_stats = facts.player_stats[player_name]
        if profile["asks_shot"]:
            return f"{player_name} recorded {player_stats['shots']} shots in the indexed match data."
        if profile["asks_pass"]:
            return f"{player_name} recorded {player_stats['passes']} passes in the indexed match data."
        if profile["asks_carry"]:
            return f"{player_name} recorded {player_stats['carries']} carries in the indexed match data."
        if profile["asks_when"]:
            return f"{player_name} appears in the indexed data from minute {player_stats['minute_min']} to minute {player_stats['minute_max']}."
        return (
            f"{player_name} was involved in {player_stats['events']} events, {player_stats['goals']} goals, and "
            f"{player_stats['sequences']} indexed sequences for {player_stats['team_name']}."
        )

    if team_name and team_name in facts.team_stats:
        team_stats = facts.team_stats[team_name]
        if profile["asks_how"] or profile["asks_why"]:
            return (
                f"{team_name} creates chances mainly through {_team_style_label(team_stats)}, producing {team_stats['shots']} shots and "
                f"{team_stats['xg']:.2f} xG from {team_stats['passes']} passes and {team_stats['carries']} carries."
            )
        if profile["asks_shot"]:
            return f"{team_name} recorded {team_stats['shots']} shots, {team_stats['goals']} goals, and {team_stats['xg']:.2f} xG in the indexed match data."
        if profile["asks_pass"]:
            return f"{team_name} recorded {team_stats['passes']} passes in the indexed match data."
        if profile["asks_carry"]:
            return f"{team_name} recorded {team_stats['carries']} carries in the indexed match data."
        if profile["asks_when"]:
            return f"{team_name}'s indexed events run from minute {team_stats['minute_min']} to minute {team_stats['minute_max']}."
        return f"{team_name} produced {team_stats['sequences']} indexed sequences and the retrieved evidence points to {pattern}."

    if profile["asks_team"] and not entities["teams"]:
        return f"The indexed match data includes {', '.join(facts.team_names)}."
    if profile["asks_xg"]:
        return f"The indexed match data totals {facts.total_xg:.2f} xG across both teams."
    if profile["asks_shot"]:
        return f"The indexed match data contains {facts.total_shots} shots overall."
    if profile["asks_pass"]:
        return f"The indexed match data contains {facts.total_passes} passes overall."
    if profile["asks_carry"]:
        return f"The indexed match data contains {facts.total_carries} carries overall."
    if profile["asks_when"]:
        return f"The retrieved evidence spans minutes {stats['minute_min']} to {stats['minute_max']}."
    if profile["asks_summary"] or profile["asks_how"] or profile["asks_why"]:
        return f"The indexed match data suggests {pattern}, with {stats['shot_sequences']} of {max(stats['sequence_count'], 1)} retrieved sequences ending in a shot."
    return f"The retrieved evidence covers {stats['evidence_count']} relevant documents across {', '.join(stats['team_names']) or 'the match'}."


def _build_analysis_bundle(query: str, evidence: list[dict[str, Any]], index_data: dict[str, Any]) -> dict[str, Any]:
    """Build reusable analysis for grounded answers."""

    facts: MatchFacts = index_data["facts"]
    entities = extract_query_entities(query, index_data=index_data)
    profile = build_query_profile(query)
    stats = build_evidence_stats(evidence=evidence, index_data=index_data, query=query)
    evidence_snapshots = [item["summary"] for item in evidence[:4]]
    direct_answer = _direct_answer(profile=profile, entities=entities, facts=facts, stats=stats)
    focus_entities = ", ".join([*entities["teams"], *entities["players"]]) or "none explicitly named"
    return {
        "query": query,
        "profile": profile,
        "entities": entities,
        "stats": stats,
        "direct_answer": direct_answer,
        "sequence_pattern": _sequence_pattern_label(stats),
        "focus_entities": focus_entities,
        "evidence_snapshots": evidence_snapshots,
        "top_players": stats["top_players"],
        "facts": facts,
    }


def _render_analysis_answer(analysis: dict[str, Any]) -> str:
    """Render an in-depth deterministic answer."""

    stats = analysis["stats"]
    facts: MatchFacts = analysis["facts"]
    player_line = _format_top_items(analysis["top_players"][:3], fallback="No clear player concentration")
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
            f"- Match teams: {', '.join(facts.team_names)}.",
            f"- Scoreline: {_scoreline_text(facts)}.",
            f"- Winner: {'draw' if facts.is_draw else facts.winner or 'unknown'}.",
            f"- Retrieved document count: {stats['evidence_count']}.",
            f"- Retrieved sequence count: {stats['sequence_count']}.",
            f"- Sequence pattern: {analysis['sequence_pattern']}.",
            f"- Shot-ending sequences: {stats['shot_sequences']} of {stats['sequence_count']}.",
            f"- Attacking-third reach: {stats['attacking_third_hits']} of {stats['sequence_count']}.",
            f"- Pass-led sequences: {stats['pass_sequences']} of {stats['sequence_count']}.",
            f"- Carry-led sequences: {stats['carry_sequences']} of {stats['sequence_count']}.",
            f"- Match totals: {facts.total_shots} shots, {facts.total_passes} passes, {facts.total_carries} carries, {facts.total_recoveries} recoveries, {facts.total_xg:.2f} xG.",
            f"- Most represented players in retrieved evidence: {player_line}.",
            f"- Evidence minute range: {stats['minute_min']} to {stats['minute_max']}.",
            "Evidence Snapshots:",
            *evidence_lines,
            "Scope Note: This answer is grounded in indexed match data and retrieved evidence from this dataset only.",
        ]
    )


def compose_grounded_answer(query: str, evidence: list[dict[str, Any]], index_data: dict[str, Any]) -> str:
    """Compose a deterministic grounded answer."""

    if not evidence:
        return f'No grounded evidence was found for "{query}".'
    analysis = _build_analysis_bundle(query=query, evidence=evidence, index_data=index_data)
    return _render_analysis_answer(analysis=analysis)


def build_grounded_prompt_json(
    query: str,
    evidence: list[dict[str, Any]],
    analysis: dict[str, Any],
    conversation_context: dict[str, Any] | None = None,
    extra_instruction: str = "",
) -> str:
    """Build a strict JSON-only prompt for generic grounded match answers."""

    stats = analysis["stats"]
    facts: MatchFacts = analysis["facts"]
    evidence_lines = [
        f"- doc_type={item['doc_type']}; team={item.get('team_name') or 'unknown'}; "
        f"player={item.get('player_name') or 'n/a'}; minute_range={item.get('minute_start')}:{item.get('minute_end')}; "
        f"summary={item['summary']}"
        for item in evidence
    ]
    conversation_lines = []
    if conversation_context and conversation_context.get("used_session_context"):
        conversation_lines = [
            f"Conversation teams in focus: {conversation_context.get('teams') or []}.",
            f"Conversation players in focus: {conversation_context.get('players') or []}.",
            f"Recent user turns: {conversation_context.get('recent_turns') or []}.",
        ]
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
            f"Match totals: total_events={facts.total_events}, total_sequences={facts.total_sequences}, total_shots={facts.total_shots}, total_passes={facts.total_passes}, total_carries={facts.total_carries}, total_recoveries={facts.total_recoveries}, total_xg={facts.total_xg}, scoreline={_scoreline_text(facts)}, winner={facts.winner or 'draw' if facts.is_draw else 'unknown'}.",
            f"Teams in match: {facts.team_names}.",
            f"Top players in evidence: {analysis['top_players']}.",
            *conversation_lines,
            extra_instruction,
            "Evidence:",
            *evidence_lines,
        ]
    )


def compose_llm_grounded_answer(
    query: str,
    evidence: list[dict[str, Any]],
    analysis: dict[str, Any],
    conversation_context: dict[str, Any] | None = None,
    extra_instruction: str = "",
) -> str:
    """Compose a grounded JSON answer with an LLM."""

    if not evidence:
        return f'No grounded evidence was found for "{query}".'
    prompt = build_grounded_prompt_json(
        query=query,
        evidence=evidence,
        analysis=analysis,
        conversation_context=conversation_context,
        extra_instruction=extra_instruction,
    )
    request_body = json.dumps(
        {
            "model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
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
    """Parse and validate generic LLM JSON payload."""

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

    evidence_points = [item.strip() for item in payload["evidence_points"][:4] if isinstance(item, str) and item.strip()]
    if len(evidence_points) < 2:
        raise ValueError("LLM JSON evidence_points too short")
    payload["evidence_points"] = evidence_points

    exact_scope = "This answer is grounded in indexed match data and retrieved evidence from this dataset only."
    if payload["scope_note"] != exact_scope:
        payload["scope_note"] = exact_scope
    return payload


def render_grounded_llm_answer(valid_json: dict[str, Any]) -> str:
    """Render a deterministic answer shell from validated LLM JSON."""

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
    conversation_context: dict[str, Any] | None = None,
) -> QueryResponse:
    """Run retrieval and grounded answer generation."""

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
            raw_answer = compose_llm_grounded_answer(
                query=query,
                evidence=evidence,
                analysis=analysis,
                conversation_context=conversation_context,
            )
            llm_output_format = "json"
            try:
                valid_json = parse_and_validate_llm_json(raw_text=raw_answer)
                answer = render_grounded_llm_answer(valid_json)
                llm_validated = True
            except ValueError as exc:
                llm_validation_errors = [str(exc)]
                llm_retry_used = True
                raw_answer = compose_llm_grounded_answer(
                    query=query,
                    evidence=evidence,
                    analysis=analysis,
                    conversation_context=conversation_context,
                    extra_instruction="Return valid JSON only. Keep claims tightly grounded in the listed evidence.",
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
            "available_teams": index_data["facts"].team_names,
            "score_by_team": index_data["facts"].score_by_team,
            "winner": index_data["facts"].winner,
            "is_draw": index_data["facts"].is_draw,
            "total_match_events": index_data["facts"].total_events,
            "total_match_sequences": index_data["facts"].total_sequences,
            "conversation_context_used": bool(conversation_context and conversation_context.get("used_session_context")),
        },
    )
