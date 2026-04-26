"""Basic API smoke tests."""

from __future__ import annotations

import os
import unittest

from fastapi.testclient import TestClient

from app.main import SESSION_STORE, app
from app.state import STATE


def _reset_app_state() -> None:
    STATE["raw_events"] = []
    STATE["events"] = []
    STATE["sequences"] = []
    STATE["index"] = None
    STATE["trace"] = {
        "files_ingested": 0,
        "sequences_built": 0,
        "sequences_indexed": 0,
    }
    SESSION_STORE._sessions.clear()


class ApiSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_ollama_url = os.environ.pop("OLLAMA_URL", None)
        self.client = TestClient(app)
        _reset_app_state()

    def tearDown(self) -> None:
        _reset_app_state()
        if self.original_ollama_url is not None:
            os.environ["OLLAMA_URL"] = self.original_ollama_url

    def test_home_page_loads(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Local Soccer Intelligence Copilot", response.text)

    def test_chat_auto_prepares_local_dataset(self) -> None:
        session_response = self.client.post("/sessions")
        session_id = session_response.json()["session_id"]

        response = self.client.post(
            "/chat",
            json={
                "session_id": session_id,
                "message": "Who won this match?",
                "top_k": 3,
                "use_llm": False,
            },
        )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertIn("answer", payload)
        self.assertTrue(payload["history"])
        self.assertEqual(payload["trace"]["turn_count"], 1)

