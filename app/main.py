"""FastAPI entrypoint for the local soccer copilot."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api.schemas import ChatRequest, IngestRequest, QueryRequest, SessionResponse
from app.services.chat import ChatSessionStore, run_chat_turn
from app.services.generation import run_query
from app.services.retrieval import build_index
from app.state import STATE, auto_prepare, ingest_dataset, require_ingested
from app.ui.page import render_home

app = FastAPI(title="Local Soccer Intelligence Copilot")
SESSION_STORE = ChatSessionStore()
STATIC_DIR = Path(__file__).resolve().parent / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    """Serve the local-only test UI."""

    return render_home(default_use_llm=bool(os.getenv("OLLAMA_URL")))


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict[str, Any]:
    """Load local event files, normalize them, and build sequences."""

    try:
        return ingest_dataset(request.dataset_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/build-index")
def build_index_route() -> dict[str, Any]:
    """Build the in-memory retrieval index."""

    require_ingested()
    try:
        STATE["index"] = build_index(STATE["events"], STATE["sequences"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    STATE["trace"]["sequences_indexed"] = len(STATE["sequences"])
    return {
        "answer": f"Indexed {len(STATE['sequences'])} sequences with TF-IDF.",
        "evidence": [],
        "trace": STATE["trace"],
    }


@app.post("/sessions", response_model=SessionResponse)
def create_session() -> SessionResponse:
    """Create a new chat session."""

    session = SESSION_STORE.create_session()
    return SessionResponse(session_id=session.session_id, history=[])


@app.post("/sessions/{session_id}/reset", response_model=SessionResponse)
def reset_session(session_id: str) -> SessionResponse:
    """Reset an existing chat session."""

    try:
        session = SESSION_STORE.reset_session(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return SessionResponse(session_id=session.session_id, history=[])


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    """Return the stored chat session history."""

    try:
        session = SESSION_STORE.get_session(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"session_id": session_id, "history": [turn.model_dump() for turn in session.turns]}


@app.post("/query")
def query(request: QueryRequest) -> dict[str, Any]:
    """Run a single grounded retrieval query."""

    auto_prepare()
    try:
        response = run_query(
            query=request.query,
            index_data=STATE["index"],
            trace=STATE["trace"],
            top_k=request.top_k,
            use_llm=request.use_llm,
            llm_required=request.llm_required,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return response.model_dump()


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    """Run one chat turn with session memory."""

    auto_prepare()
    try:
        session = SESSION_STORE.get_session(request.session_id)
        response = run_chat_turn(
            session=session,
            message=request.message,
            index_data=STATE["index"],
            trace=STATE["trace"],
            top_k=request.top_k,
            use_llm=request.use_llm,
            llm_required=request.llm_required,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return response
