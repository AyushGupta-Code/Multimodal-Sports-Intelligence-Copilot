"""Small schemas for the Local Soccer Intelligence Copilot MVP."""

from typing import Any

from pydantic import BaseModel, Field


class EventRecord(BaseModel):
    """Normalized event fields used by the Phase 1 pipeline."""

    match_id: str | None = None
    team_name: str = "Unknown"
    player_name: str = "Unknown"
    event_type: str = "Unknown"
    minute: int = 0
    second: int = 0
    possession_id: int | None = None
    play_pattern: str | None = None
    location: list[float] | None = None
    pass_end_location: list[float] | None = None
    shot_outcome: str | None = None


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
    summary: str


class IngestRequest(BaseModel):
    """Request body for loading local StatsBomb-style event JSON."""

    dataset_path: str


class QueryRequest(BaseModel):
    """Request body for running a grounded retrieval query."""

    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    """Response body for grounded answers with supporting evidence and trace data."""

    answer: str
    evidence: list[dict[str, Any]]
    trace: dict[str, Any]
