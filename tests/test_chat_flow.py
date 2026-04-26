"""Conversational flow tests."""

from __future__ import annotations

import unittest

from app.models.domain import ChatSession
from app.services.chat import resolve_chat_query


class ChatFlowTests(unittest.TestCase):
    def test_follow_up_query_uses_recent_context_when_entities_missing(self) -> None:
        session = ChatSession(
            session_id="test-session",
            turns=[
                {
                    "user_message": "How did Barcelona create chances?",
                    "resolved_query": "How did Barcelona create chances?",
                    "answer": "Barcelona created chances through pass-led buildup.",
                    "trace": {},
                    "teams": ["Barcelona"],
                    "players": ["Lionel Messi"],
                }
            ],
        )
        index_data = {
            "facts": type("Facts", (), {"team_names": ["Barcelona"], "player_names": ["Lionel Messi"]})()
        }

        resolved_query, metadata = resolve_chat_query("What about in the second half?", session=session, index_data=index_data)

        self.assertIn("Conversation context", resolved_query)
        self.assertTrue(metadata["used_session_context"])
        self.assertEqual(metadata["teams"], ["Barcelona"])
