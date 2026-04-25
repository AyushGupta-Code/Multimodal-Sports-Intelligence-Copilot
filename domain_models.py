"""Domain models for match analysis and chat state."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EventRecord(BaseModel):
    """Normalized StatsBomb event fields used by the app."""

    match_id: str | None = None
    team_name: str = "Unknown"
    possession_team_name: str = "Unknown"
    player_name: str = "Unknown"
    event_type: str = "Unknown"
    minute: int = 0
    second: int = 0
    period: int = 1
    possession_id: int | None = None
    play_pattern: str | None = None
    location: list[float] | None = None
    pass_end_location: list[float] | None = None
    pass_recipient_name: str | None = None
    shot_outcome: str | None = None
    shot_xg: float | None = None
    duration: float | None = None
    position_name: str | None = None
    card_name: str | None = None
    replacement_player_name: str | None = None


class SequenceRecord(BaseModel):
    """Simple attacking sequence built from a possession or short event chain."""

    sequence_id: str
    match_id: str | None = None
    team_name: str
    possession_id: int | None = None
    players: list[str] = Field(default_factory=list)
    event_chain: list[str] = Field(default_factory=list)
    ended_in_shot: bool = False
    progression: str = "unknown progression"
    start_minute: int = 0
    end_minute: int = 0
    period: int = 1
    play_patterns: list[str] = Field(default_factory=list)
    summary: str


class DocumentRecord(BaseModel):
    """Indexed evidence document used for retrieval."""

    doc_id: str
    doc_type: str
    match_id: str | None = None
    team_name: str | None = None
    player_name: str | None = None
    period: int | None = None
    minute_start: int | None = None
    minute_end: int | None = None
    summary: str
    text: str
    players: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    ended_in_shot: bool = False
    play_patterns: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MatchFacts(BaseModel):
    """Computed match-, team-, and player-level facts."""

    match_ids: list[str] = Field(default_factory=list)
    team_names: list[str] = Field(default_factory=list)
    player_names: list[str] = Field(default_factory=list)
    team_stats: dict[str, dict[str, Any]] = Field(default_factory=dict)
    player_stats: dict[str, dict[str, Any]] = Field(default_factory=dict)
    score_by_team: dict[str, int] = Field(default_factory=dict)
    winner: str | None = None
    is_draw: bool = False
    goals: list[dict[str, Any]] = Field(default_factory=list)
    cards: list[dict[str, Any]] = Field(default_factory=list)
    substitutions: list[dict[str, Any]] = Field(default_factory=list)
    total_events: int = 0
    total_sequences: int = 0
    total_shots: int = 0
    total_passes: int = 0
    total_carries: int = 0
    total_recoveries: int = 0
    total_xg: float = 0.0
    play_patterns: list[tuple[str, int]] = Field(default_factory=list)
    event_types: list[tuple[str, int]] = Field(default_factory=list)
    period_counts: dict[str, int] = Field(default_factory=dict)
    minute_min: int = 0
    minute_max: int = 0
    top_players_by_events: list[tuple[str, int]] = Field(default_factory=list)
    top_players_by_shots: list[tuple[str, int]] = Field(default_factory=list)
    top_players_by_passes: list[tuple[str, int]] = Field(default_factory=list)


class ChatTurn(BaseModel):
    """Stored chat turn for conversational follow-ups."""

    user_message: str
    resolved_query: str
    answer: str
    trace: dict[str, Any]
    teams: list[str] = Field(default_factory=list)
    players: list[str] = Field(default_factory=list)


class ChatSession(BaseModel):
    """In-memory chat session."""

    session_id: str
    turns: list[ChatTurn] = Field(default_factory=list)
