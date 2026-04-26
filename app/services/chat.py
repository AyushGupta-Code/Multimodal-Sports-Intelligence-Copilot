"""Conversation state and chat orchestration."""

from __future__ import annotations

import re
import uuid
from typing import Any

from app.models.domain import ChatSession, ChatTurn
from app.services.generation import run_query
from app.services.retrieval import extract_query_entities


class ChatSessionStore:
    """Simple in-memory session store for conversational follow-ups."""

    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}

    def create_session(self) -> ChatSession:
        """Create a new chat session."""

        session = ChatSession(session_id=str(uuid.uuid4()))
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> ChatSession:
        """Return an existing session."""

        if session_id not in self._sessions:
            raise ValueError(f"Unknown session_id: {session_id}")
        return self._sessions[session_id]

    def reset_session(self, session_id: str) -> ChatSession:
        """Clear an existing session."""

        session = self.get_session(session_id)
        session.turns.clear()
        return session


def _has_follow_up_reference(query: str) -> bool:
    """Return whether the query likely depends on prior context."""

    lowered = query.lower()
    patterns = [
        r"^\s*and\b",
        r"^\s*also\b",
        r"^\s*then\b",
        r"^\s*what about\b",
        r"^\s*how about\b",
        r"^\s*what else\b",
        r"^\s*and what\b",
        r"^\s*in the\b",
        r"\bthey\b",
        r"\bthem\b",
        r"\bhe\b",
        r"\bhim\b",
        r"\bhis\b",
        r"\bthis team\b",
        r"\bthat team\b",
        r"\bthat player\b",
        r"\bthose sequences\b",
        r"\bthese chances\b",
    ]
    return any(re.search(pattern, lowered) for pattern in patterns)


def _recent_context(session: ChatSession) -> dict[str, list[str]]:
    """Collect recent team/player context from the latest turns."""

    teams: list[str] = []
    players: list[str] = []
    for turn in reversed(session.turns[-3:]):
        for team in turn.teams:
            if team not in teams:
                teams.append(team)
        for player in turn.players:
            if player not in players:
                players.append(player)
    return {"teams": teams, "players": players}


def resolve_chat_query(query: str, session: ChatSession, index_data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Inject recent conversational context for follow-up queries."""

    if not session.turns:
        return query, {"used_session_context": False, "teams": [], "players": [], "recent_turns": []}
    current_entities = extract_query_entities(query, index_data=index_data)
    if current_entities["teams"] or current_entities["players"] or not _has_follow_up_reference(query):
        return query, {"used_session_context": False, "teams": current_entities["teams"], "players": current_entities["players"], "recent_turns": []}

    recent = _recent_context(session)
    context_parts: list[str] = []
    if recent["teams"]:
        context_parts.append(f"teams in focus: {', '.join(recent['teams'])}")
    if recent["players"]:
        context_parts.append(f"players in focus: {', '.join(recent['players'])}")
    if not context_parts:
        return query, {"used_session_context": False, "teams": [], "players": [], "recent_turns": []}

    recent_turns = [turn.user_message for turn in session.turns[-3:]]
    resolved_query = (
        f"Conversation context ({'; '.join(context_parts)}). "
        f"Recent user turns: {' | '.join(recent_turns)}. "
        f"Follow-up question: {query}"
    )
    return resolved_query, {
        "used_session_context": True,
        "teams": recent["teams"],
        "players": recent["players"],
        "recent_turns": recent_turns,
    }


def run_chat_turn(
    session: ChatSession,
    message: str,
    index_data: dict[str, Any],
    trace: dict[str, Any],
    top_k: int = 5,
    use_llm: bool = False,
    llm_required: bool = False,
) -> dict[str, Any]:
    """Run one conversational turn and persist it."""

    resolved_query, session_context = resolve_chat_query(message, session=session, index_data=index_data)
    response = run_query(
        query=resolved_query,
        index_data=index_data,
        trace=trace,
        top_k=top_k,
        use_llm=use_llm,
        llm_required=llm_required,
        conversation_context=session_context,
    )
    turn = ChatTurn(
        user_message=message,
        resolved_query=resolved_query,
        answer=response.answer,
        trace=response.trace,
        teams=response.trace.get("query_teams") or [],
        players=response.trace.get("query_players") or [],
    )
    session.turns.append(turn)
    return {
        "session_id": session.session_id,
        "answer": response.answer,
        "evidence": response.evidence,
        "trace": {
            **response.trace,
            "resolved_query": resolved_query,
            "session_context": session_context,
            "turn_count": len(session.turns),
        },
        "history": [stored_turn.model_dump() for stored_turn in session.turns],
    }
