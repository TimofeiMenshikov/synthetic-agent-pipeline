"""Exercise the real HTTP adapter against a local, deterministic test server."""

import io
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from urllib.error import HTTPError

from synthetic_agent_pipeline.teachers import MAX_RESPONSE_BYTES, OpenAICompatibleTeacher, TeacherError


class TeacherTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.body = {"choices": [{"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}]}
        self.status = 200
        self.delay = 0
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                data = self.rfile.read(int(self.headers["Content-Length"]))
                owner.requests.append({"path": self.path, "headers": dict(self.headers), "payload": json.loads(data)})
                time.sleep(owner.delay)
                self.send_response(owner.status)
                self.send_header("Content-Type", "application/json")
                if owner.status == 302:
                    self.send_header("Location", f"http://localhost:{owner.server.server_port}/redirected")
                self.end_headers()
                body = owner.body if isinstance(owner.body, bytes) else json.dumps(owner.body).encode()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                owner.requests.append({"method": "GET", "path": self.path})
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"choices": [{"message": {"role": "assistant", "content": "redirect followed"}}]}).encode())

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"
        self.teacher = OpenAICompatibleTeacher.from_env({
            "TEACHER_API_KEY": "unit-test-secret", "TEACHER_MODEL": "test-model", "TEACHER_BASE_URL": self.base_url,
        })

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def assert_failure(self, code, *, timeout=2):
        with self.assertRaises(TeacherError) as raised:
            self.teacher.complete([{"role": "user", "content": "test"}], [], timeout)
        self.assertEqual(raised.exception.code, code)
        self.assertNotIn("unit-test-secret", str(raised.exception))
        self.assertEqual(raised.exception.as_dict()["code"], code)

    def test_payload_and_clean_message(self):
        messages = [{"role": "user", "content": "Прочитай CSV"}]
        tools = [{"type": "function", "function": {"name": "read_csv", "parameters": {"type": "object"}}}]
        self.body["choices"][0]["message"]["provider_extra"] = "discard"
        result = self.teacher.complete(messages, tools, 2)
        self.assertEqual(result, {"role": "assistant", "content": "done"})
        captured = self.requests[0]
        self.assertEqual(captured["path"], "/v1/chat/completions")
        self.assertEqual(captured["payload"], {"model": "test-model", "messages": messages, "tools": tools, "tool_choice": "auto"})
        self.assertEqual(captured["headers"]["Authorization"], "Bearer unit-test-secret")
        self.assertNotIn("unit-test-secret", repr(self.teacher))

    def test_environment_precedence_and_defaults(self):
        teacher = OpenAICompatibleTeacher.from_env({
            "TEACHER_API_KEY": "first", "OPENAI_API_KEY": "second",
            "TEACHER_MODEL": "first-model", "OPENAI_MODEL": "second-model",
            "TEACHER_BASE_URL": "http://localhost:1234/v1/", "OPENAI_BASE_URL": "http://ignored/v1",
        })
        self.assertEqual((teacher.api_key, teacher.model, teacher.base_url), ("first", "first-model", "http://localhost:1234/v1"))
        alias = OpenAICompatibleTeacher.from_env({"OPENAI_API_KEY": "local", "OPENAI_MODEL": "alias"})
        self.assertEqual(alias.base_url, "https://api.openai.com/v1")
        self.assertEqual(alias.model, "alias")
        for env in ({}, {"TEACHER_API_KEY": "x"}, {"TEACHER_MODEL": "x"}, {
            "TEACHER_API_KEY": "x", "TEACHER_MODEL": "x", "TEACHER_BASE_URL": "http://secret@host/v1",
        }):
            with self.subTest(env=env), self.assertRaises(TeacherError):
                OpenAICompatibleTeacher.from_env(env)

    def test_tool_calls_are_preserved_and_filtered(self):
        call = {"id": "call_1", "type": "function", "function": {"name": "query_sql", "arguments": '{"sql":"SELECT 1"}', "extra": "discard"}, "extra": "discard"}
        self.body = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [call]}}]}
        result = self.teacher.complete([], [], 2)
        self.assertEqual(result["tool_calls"], [{"id": "call_1", "type": "function", "function": {"name": "query_sql", "arguments": '{"sql":"SELECT 1"}'}}])

    def test_malformed_json_and_response(self):
        for body in (b"not-json", [], {}, {"choices": []}, {"choices": [{"message": {"role": "user", "content": "x"}}]},
                     {"choices": [{"message": {"role": "assistant", "content": None}}]},
                     {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "x", "type": "function", "function": {"name": "f", "arguments": {}}}]}}]}):
            with self.subTest(body=body):
                self.body = body
                self.assert_failure("invalid_response")

    def test_http_errors_never_echo_response(self):
        self.status = 401
        self.body = b"unit-test-secret"
        self.assert_failure("http_error")

    def test_http_error_body_is_closed(self):
        body = io.BytesIO(b"unit-test-secret")
        failure = HTTPError(self.base_url, 401, "denied", {}, body)
        with patch("synthetic_agent_pipeline.teachers.request.build_opener") as opener:
            opener.return_value.open.side_effect = failure
            self.assert_failure("http_error")
        self.assertTrue(body.closed)

    def test_redirect_does_not_forward_credentials(self):
        self.status = 302
        self.assert_failure("http_error")
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.requests[0]["path"], "/v1/chat/completions")

    def test_time_budget_and_http_timeout(self):
        self.assert_failure("timeout", timeout=0)
        self.assertEqual(self.requests, [])
        self.delay = 0.2
        started = time.monotonic()
        self.assert_failure("timeout", timeout=0.03)
        self.assertLess(time.monotonic() - started, 0.15)

    def test_large_or_truncated_response(self):
        self.body = b"x" * (MAX_RESPONSE_BYTES + 1)
        self.assert_failure("response_too_large")
        self.body = {"choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": "partial"}}]}
        self.assert_failure("incomplete_response")


if __name__ == "__main__":
    unittest.main()
