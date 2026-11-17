import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from synthetic_agent_pipeline.export import validate_dataset
from synthetic_agent_pipeline.mock_teacher import MockTeacher
from synthetic_agent_pipeline.runner import _run_task, run_pipeline, tree_fingerprint
from synthetic_agent_pipeline.teachers import TeacherError


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def run_quietly(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return run_pipeline(output=self.root, **kwargs)

    def test_complete_mock_accept_reject_export_and_reproduce(self):
        first = self.run_quietly(run_id="first", mode="mock", count=6, seed=73, mock_fail_every=3)
        second = self.run_quietly(run_id="second", mode="mock", count=6, seed=73)
        self.assertEqual((first["generated"], first["accepted"], first["rejected"]), (6, 4, 2))
        self.assertEqual(second["accepted"], 6)
        self.assertEqual(first["generated_sha256"], second["generated_sha256"])
        self.assertEqual(first["rejection_reasons"], {"result_mismatch": 2})
        run = self.root / "mock/second"
        self.assertFalse((self.root / "teacher").exists())
        self.assertEqual(validate_dataset(run / "exports/train.jsonl"), 4)
        self.assertEqual(validate_dataset(run / "exports/validation.jsonl"), 2)
        index = [json.loads(line) for line in (run / "exports/index.jsonl").read_text().splitlines()]
        train_families = {row["family"] for row in index if row["split"] == "train"}
        validation_families = {row["family"] for row in index if row["split"] == "validation"}
        self.assertEqual(validation_families, {"sales_summary"})
        self.assertFalse(train_families & validation_families)
        self.assertEqual(len(list((run / "traces").glob("*/events.jsonl"))), 6)
        for workspace in (run / "workspaces").iterdir():
            self.assertFalse((workspace / "references").exists())
            self.assertEqual(set(path.name for path in workspace.iterdir()), {"task.json", "data", "output"})
        with self.assertRaises(FileExistsError):
            self.run_quietly(run_id="second", mode="mock", count=6, seed=73)

    def test_step_limit_rejects_even_when_export_is_empty(self):
        report = self.run_quietly(run_id="steps", mode="mock", count=3, seed=7, max_steps=1)
        self.assertEqual(report["accepted"], 0)
        self.assertEqual(report["rejection_reasons"], {"step_limit": 3})
        self.assertEqual(validate_dataset(self.root / "mock/steps/exports/train.jsonl"), 0)
        self.assertEqual(validate_dataset(self.root / "mock/steps/exports/validation.jsonl"), 0)

    def test_parent_enforces_hard_deadline(self):
        workspace = self.root / "workspace"
        workspace.mkdir()
        (workspace / "data").mkdir()
        traces = self.root / "traces"
        started = time.monotonic()
        status = _run_task(workspace, traces, {"family": "sales_summary", "prompt": "test"},
                           "mock", 16, 0.001, False)
        self.assertEqual(status["reason"], "timeout")
        self.assertLess(time.monotonic() - started, 2)
        self.assertEqual(json.loads((traces / "status.json").read_text())["reason"], "timeout")

    def test_invalid_config_fails_before_output_creation(self):
        for values in ({"count": 2}, {"validation_families": set()}, {"validation_families": {"unknown"}},
                       {"max_steps": 0}, {"timeout": float("nan")}, {"run_id": "../outside"}):
            options = dict(run_id="bad", mode="mock", count=3, seed=7)
            options.update(values)
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.run_quietly(**options)
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(TeacherError):
            self.run_quietly(run_id="missing-key", mode="teacher", count=3, seed=7)
        self.assertFalse((self.root / "teacher").exists())

    def test_full_teacher_http_path_uses_only_public_observations(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(request)
                messages = request["messages"]
                prompt = messages[1]["content"]
                family = ("table_reconciliation" if "ledger.csv" in prompt else
                          "inventory_discrepancies" if "inventory.sqlite" in prompt else "sales_summary")
                teacher = MockTeacher(family)
                previous_calls = [call for message in messages for call in message.get("tool_calls", [])]
                teacher.turn = len(previous_calls)
                teacher.wrote_result = any(call["function"]["name"] == "write_result" for call in previous_calls)
                message = teacher.complete(messages, request["tools"], 10)
                payload = json.dumps({"choices": [{"message": message, "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                env = {"TEACHER_API_KEY": "local-test-only", "TEACHER_MODEL": "test-teacher",
                       "TEACHER_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1"}
                with patch.dict(os.environ, env):
                    report = self.run_quietly(run_id="http-test", mode="teacher", count=3, seed=73, timeout=10)
            finally:
                server.shutdown()
                thread.join(2)
        self.assertEqual(report["accepted"], 3)
        self.assertEqual(report["splits"], {"train": 2, "validation": 1})
        self.assertEqual(report["mode"], "teacher")
        self.assertFalse((self.root / "mock").exists())
        self.assertGreater(len(requests), 12)
        self.assertEqual({request["model"] for request in requests}, {"test-teacher"})
        self.assertNotIn("reference_path", json.dumps(requests))
        for path in (self.root / "teacher/http-test").rglob("*.json*"):
            self.assertNotIn("local-test-only", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
