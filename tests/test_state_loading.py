"""Dataset loading and path handling tests."""

from __future__ import annotations

import os
import tempfile
import unittest

from fastapi import HTTPException

from app.state import STATE, auto_prepare, load_default_data


def _reset_state() -> None:
    STATE["raw_events"] = []
    STATE["events"] = []
    STATE["sequences"] = []
    STATE["index"] = None
    STATE["trace"] = {
        "files_ingested": 0,
        "sequences_built": 0,
        "sequences_indexed": 0,
    }


class StateLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_dataset_path = os.environ.pop("DATASET_PATH", None)
        _reset_state()

    def tearDown(self) -> None:
        _reset_state()
        if self.original_dataset_path is not None:
            os.environ["DATASET_PATH"] = self.original_dataset_path

    def test_load_default_data_uses_repo_data_folder(self) -> None:
        load_default_data()

        self.assertGreater(len(STATE["events"]), 0)
        self.assertGreater(STATE["trace"]["files_ingested"], 0)

    def test_auto_prepare_honors_dataset_path_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["DATASET_PATH"] = tmpdir

            with self.assertRaises(HTTPException) as error:
                auto_prepare()

        self.assertEqual(error.exception.status_code, 400)
        self.assertTrue(error.exception.detail)
