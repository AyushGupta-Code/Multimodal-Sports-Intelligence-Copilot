"""Feature extraction for sequences and match facts."""

from __future__ import annotations

from collections import Counter
from typing import Any

from app.models.domain import DocumentRecord, EventRecord, MatchFacts, SequenceRecord


def _top_counter_items(counter: Counter[str], limit: int = 5) -> list[tuple[str, int]]:
    """Return counter items in deterministic order."""

    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]


def _format_top_items(items: list[tuple[str, int]], fallback: str = "none") -> str:
    """Render a short top-items string."""

    if not items:
        return fallback
    return ", ".join(f"{name} ({count})" for name, count in items)


def _event_text(event: EventRecord) -> str:
    """Return a short event phrase for summaries."""

    lowered = event.event_type.lower()
    if lowered == "pass":
        recipient = f" to {event.pass_recipient_name}" if event.pass_recipient_name else ""
        if event.pass_end_location:
            return f"pass{recipient} toward {int(event.pass_end_location[0])},{int(event.pass_end_location[1])}"
        return f"pass{recipient}"
    if lowered == "shot":
        return f"shot ({event.shot_outcome})" if event.shot_outcome else "shot"
    if lowered == "substitution" and event.replacement_player_name:
        return f"substitution for {event.replacement_player_name}"
    return lowered


def _zone_name(x: float | None) -> str:
    """Map an x coordinate into a rough field zone."""

    if x is None:
        return "unknown"
    if x < 40:
        return "defensive third"
    if x < 80:
        return "middle third"
    return "attacking third"


def _progression_label(events: list[EventRecord]) -> str:
    """Describe broad field progression for a sequence."""

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
    """Build one possession summary."""

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
        start_minute=first.minute,
        end_minute=events[-1].minute,
        period=first.period,
        play_patterns=play_patterns,
        summary=summary,
    )


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


def build_match_facts(events: list[EventRecord], sequences: list[SequenceRecord]) -> MatchFacts:
    """Build reusable match-, team-, and player-level facts."""

    if not events:
        raise ValueError("No events available to analyze")

    team_stats: dict[str, dict[str, Any]] = {}
    player_stats: dict[str, dict[str, Any]] = {}
    play_pattern_counts: Counter[str] = Counter()
    event_type_counts: Counter[str] = Counter()
    period_counts: Counter[str] = Counter()
    goals: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    substitutions: list[dict[str, Any]] = []
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
                "goals": 0,
                "passes": 0,
                "carries": 0,
                "recoveries": 0,
                "cards": 0,
                "substitutions": 0,
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
            if event.shot_outcome == "Goal":
                team_entry["goals"] += 1
                goals.append(
                    {
                        "team_name": team_name,
                        "player_name": player_name,
                        "minute": event.minute,
                        "period": event.period,
                    }
                )
        elif event_type == "pass":
            team_entry["passes"] += 1
        elif event_type == "carry":
            team_entry["carries"] += 1
        elif event_type == "ball recovery":
            team_entry["recoveries"] += 1
        elif event_type == "own goal for":
            team_entry["goals"] += 1
            goals.append(
                {
                    "team_name": team_name,
                    "player_name": player_name,
                    "minute": event.minute,
                    "period": event.period,
                    "own_goal_for": True,
                }
            )

        if event.card_name:
            team_entry["cards"] += 1
            cards.append(
                {
                    "team_name": team_name,
                    "player_name": player_name,
                    "card_name": event.card_name,
                    "minute": event.minute,
                    "period": event.period,
                }
            )
        if event.event_type.lower() == "substitution":
            team_entry["substitutions"] += 1
            substitutions.append(
                {
                    "team_name": team_name,
                    "player_name": player_name,
                    "replacement_player_name": event.replacement_player_name,
                    "minute": event.minute,
                    "period": event.period,
                }
            )

        if player_name != "Unknown":
            player_entry = player_stats.setdefault(
                player_name,
                {
                    "team_name": team_name,
                    "events": 0,
                    "shots": 0,
                    "goals": 0,
                    "passes": 0,
                    "carries": 0,
                    "recoveries": 0,
                    "cards": 0,
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
                if event.shot_outcome == "Goal":
                    player_entry["goals"] += 1
            elif event_type == "pass":
                player_entry["passes"] += 1
            elif event_type == "carry":
                player_entry["carries"] += 1
            elif event_type == "ball recovery":
                player_entry["recoveries"] += 1
            if event.card_name:
                player_entry["cards"] += 1

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

    score_by_team = {team_name: team_stats[team_name]["goals"] for team_name in sorted(team_stats)}
    winner = None
    is_draw = False
    if score_by_team:
        ranked = sorted(score_by_team.items(), key=lambda item: (-item[1], item[0]))
        if len(ranked) >= 2 and ranked[0][1] == ranked[1][1]:
            is_draw = True
        else:
            winner = ranked[0][0]

    return MatchFacts(
        match_ids=match_ids,
        team_names=sorted(team_stats.keys()),
        player_names=sorted(player_stats.keys()),
        team_stats=team_stats,
        player_stats=player_stats,
        score_by_team=score_by_team,
        winner=winner,
        is_draw=is_draw,
        goals=goals,
        cards=cards,
        substitutions=substitutions,
        total_events=len(events),
        total_sequences=len(sequences),
        total_shots=event_type_counts.get("shot", 0),
        total_passes=event_type_counts.get("pass", 0),
        total_carries=event_type_counts.get("carry", 0),
        total_recoveries=event_type_counts.get("ball recovery", 0),
        total_xg=round(sum(team_entry["xg"] for team_entry in team_stats.values()), 3),
        play_patterns=_top_counter_items(play_pattern_counts, limit=5),
        event_types=_top_counter_items(event_type_counts, limit=8),
        period_counts=dict(period_counts),
        minute_min=minute_min,
        minute_max=minute_max,
        top_players_by_events=_top_counter_items(
            Counter({player: stats["events"] for player, stats in player_stats.items()}),
            limit=5,
        ),
        top_players_by_shots=_top_counter_items(
            Counter({player: stats["shots"] for player, stats in player_stats.items() if stats["shots"] > 0}),
            limit=5,
        ),
        top_players_by_passes=_top_counter_items(
            Counter({player: stats["passes"] for player, stats in player_stats.items() if stats["passes"] > 0}),
            limit=5,
        ),
    )


def build_documents(
    events: list[EventRecord],
    sequences: list[SequenceRecord],
    facts: MatchFacts,
) -> list[DocumentRecord]:
    """Build mixed retrieval documents for the chatbot."""

    scoreline = " - ".join(f"{team} {score}" for team, score in facts.score_by_team.items())
    winner_text = "The match finished level." if facts.is_draw else f"{facts.winner} won the match." if facts.winner else ""
    documents: list[DocumentRecord] = [
        DocumentRecord(
            doc_id="match-overview",
            doc_type="match_overview",
            match_id=facts.match_ids[0] if facts.match_ids else None,
            summary=(
                f"Match overview for {', '.join(facts.team_names)}: scoreline {scoreline or 'unknown'}, "
                f"{facts.total_events} events, {facts.total_sequences} indexed sequences, {facts.total_shots} shots, "
                f"{facts.total_passes} passes, {facts.total_carries} carries, {facts.total_recoveries} ball recoveries, "
                f"total xG {facts.total_xg:.2f}. {winner_text}"
            ),
            text=(
                f"Teams: {', '.join(facts.team_names)}. Minute range: {facts.minute_min} to {facts.minute_max}. "
                f"Top players by events: {_format_top_items(facts.top_players_by_events)}. "
                f"Top players by shots: {_format_top_items(facts.top_players_by_shots)}. "
                f"Top play patterns: {_format_top_items(facts.play_patterns)}. Period counts: {facts.period_counts}."
            ),
            keywords=["match", "overview", "winner", "score", "scoreline", "shots", "passes", "xg"],
            metadata={"score_by_team": facts.score_by_team, "winner": facts.winner, "is_draw": facts.is_draw},
        )
    ]

    for team_name, team_stats in facts.team_stats.items():
        top_team_players = _top_counter_items(
            Counter(
                {
                    player: facts.player_stats[player]["events"]
                    for player in team_stats["players"]
                    if player in facts.player_stats
                }
            ),
            limit=4,
        )
        documents.append(
            DocumentRecord(
                doc_id=f"team-{team_name.lower().replace(' ', '-')}",
                doc_type="team_summary",
                match_id=facts.match_ids[0] if facts.match_ids else None,
                team_name=team_name,
                minute_start=team_stats["minute_min"],
                minute_end=team_stats["minute_max"],
                summary=(
                    f"{team_name} team summary: {team_stats['events']} events, {team_stats['sequences']} sequences, "
                    f"{team_stats['goals']} goals, {team_stats['shots']} shots, {team_stats['passes']} passes, "
                    f"{team_stats['carries']} carries, {team_stats['recoveries']} ball recoveries, xG {team_stats['xg']:.2f}."
                ),
                text=(
                    f"Top involved players: {_format_top_items(top_team_players)}. "
                    f"Top play patterns: {_format_top_items(team_stats['top_play_patterns'])}. "
                    f"Period split: {dict(team_stats['periods'])}. Minute range: {team_stats['minute_min']} to {team_stats['minute_max']}."
                ),
                players=[player for player, _ in top_team_players],
                keywords=["team", team_name, "goals", "shots", "passes", "carries", "xg"],
                metadata={"team_stats": team_stats},
            )
        )

    for player_name, player_stats in facts.player_stats.items():
        documents.append(
            DocumentRecord(
                doc_id=f"player-{player_name.lower().replace(' ', '-')}",
                doc_type="player_summary",
                match_id=facts.match_ids[0] if facts.match_ids else None,
                team_name=player_stats["team_name"],
                player_name=player_name,
                minute_start=player_stats["minute_min"],
                minute_end=player_stats["minute_max"],
                summary=(
                    f"{player_name} player summary for {player_stats['team_name']}: {player_stats['events']} events, "
                    f"{player_stats['sequences']} sequences, {player_stats['goals']} goals, {player_stats['shots']} shots, "
                    f"{player_stats['passes']} passes, {player_stats['carries']} carries, "
                    f"{player_stats['recoveries']} ball recoveries, xG {player_stats['xg']:.2f}."
                ),
                text=(
                    f"Minute range: {player_stats['minute_min']} to {player_stats['minute_max']}. "
                    f"Play patterns: {_format_top_items(player_stats['top_play_patterns'])}. "
                    f"Period split: {dict(player_stats['periods'])}."
                ),
                players=[player_name],
                keywords=["player", player_name, player_stats["team_name"], "goals", "shots", "passes", "carries", "xg"],
                metadata={"player_stats": player_stats},
            )
        )

    for goal_index, goal in enumerate(facts.goals, start=1):
        documents.append(
            DocumentRecord(
                doc_id=f"goal-{goal_index}",
                doc_type="goal_event",
                match_id=facts.match_ids[0] if facts.match_ids else None,
                team_name=goal["team_name"],
                player_name=goal.get("player_name"),
                minute_start=goal["minute"],
                minute_end=goal["minute"],
                period=goal["period"],
                summary=f"Goal for {goal['team_name']} by {goal.get('player_name') or 'unknown player'} at minute {goal['minute']}.",
                text="Goal event in the match timeline.",
                players=[goal.get("player_name")] if goal.get("player_name") else [],
                keywords=["goal", "score", "winner", goal["team_name"]],
            )
        )

    for card_index, card in enumerate(facts.cards, start=1):
        documents.append(
            DocumentRecord(
                doc_id=f"card-{card_index}",
                doc_type="card_event",
                match_id=facts.match_ids[0] if facts.match_ids else None,
                team_name=card["team_name"],
                player_name=card.get("player_name"),
                minute_start=card["minute"],
                minute_end=card["minute"],
                period=card["period"],
                summary=f"{card['card_name']} card for {card.get('player_name') or 'unknown player'} of {card['team_name']} at minute {card['minute']}.",
                text="Card event in the match timeline.",
                players=[card.get("player_name")] if card.get("player_name") else [],
                keywords=["card", card["card_name"], card["team_name"]],
            )
        )

    for sub_index, sub in enumerate(facts.substitutions, start=1):
        documents.append(
            DocumentRecord(
                doc_id=f"sub-{sub_index}",
                doc_type="substitution_event",
                match_id=facts.match_ids[0] if facts.match_ids else None,
                team_name=sub["team_name"],
                player_name=sub.get("player_name"),
                minute_start=sub["minute"],
                minute_end=sub["minute"],
                period=sub["period"],
                summary=(
                    f"Substitution for {sub['team_name']} at minute {sub['minute']}: "
                    f"{sub.get('player_name') or 'unknown player'} replaced by {sub.get('replacement_player_name') or 'unknown player'}."
                ),
                text="Substitution event in the match timeline.",
                players=[name for name in [sub.get("player_name"), sub.get("replacement_player_name")] if name],
                keywords=["substitution", sub["team_name"]],
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
