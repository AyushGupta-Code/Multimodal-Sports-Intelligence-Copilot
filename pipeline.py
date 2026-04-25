"""Thin compatibility facade over the refactored chatbot modules."""

from __future__ import annotations

from chat_service import ChatSessionStore, run_chat_turn
from data_loader import load_event_files, normalize_events
from generation import run_query
from match_features import build_match_facts, build_sequences
from retrieval_engine import build_index

SESSION_STORE = ChatSessionStore()

__all__ = [
    "SESSION_STORE",
    "build_index",
    "build_match_facts",
    "build_sequences",
    "load_event_files",
    "normalize_events",
    "run_chat_turn",
    "run_query",
]
