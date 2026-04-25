"""API schemas for the match chatbot."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Request body for loading local StatsBomb-style event JSON."""

    dataset_path: str


class QueryRequest(BaseModel):
    """Request body for running a grounded retrieval query."""

    query: str
    top_k: int = 5
    use_llm: bool = True
    llm_required: bool = False


class ChatRequest(BaseModel):
    """Request body for one chat turn."""

    message: str
    session_id: str
    top_k: int = 5
    use_llm: bool = True
    llm_required: bool = False


class QueryResponse(BaseModel):
    """Response body for grounded answers with supporting evidence and trace data."""

    answer: str
    evidence: list[dict[str, Any]]
    trace: dict[str, Any]


class SessionResponse(BaseModel):
    """Response body for session creation/reset."""

    session_id: str
    history: list[dict[str, Any]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """Response body for one chat turn."""

    session_id: str
    answer: str
    evidence: list[dict[str, Any]]
    trace: dict[str, Any]
    history: list[dict[str, Any]] = Field(default_factory=list)
