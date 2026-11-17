"""Small, dependency-free adapter for an OpenAI-compatible teacher API."""

from __future__ import annotations

import json
import math
import os
import socket
from dataclasses import dataclass, field
from http.client import HTTPException
from typing import Mapping
from urllib import error, parse, request


MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # The configured API endpoint must be direct. Never forward its bearer
        # token to a redirect destination, including a different host.
        return None


class TeacherError(RuntimeError):
    """A safe, structured failure that never embeds response bodies or keys."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class OpenAICompatibleTeacher:
    api_key: str = field(repr=False)
    base_url: str
    model: str

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "OpenAICompatibleTeacher":
        env = os.environ if environ is None else environ

        def first(*names: str, default: str = "") -> str:
            return next((env[name].strip() for name in names if env.get(name, "").strip()), default)

        api_key = first("TEACHER_API_KEY", "OPENAI_API_KEY")
        model = first("TEACHER_MODEL", "OPENAI_MODEL")
        base_url = first(
            "TEACHER_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE",
            default="https://api.openai.com/v1",
        ).rstrip("/")
        if not api_key:
            raise TeacherError("configuration_error", "Set TEACHER_API_KEY or OPENAI_API_KEY.")
        if not model:
            raise TeacherError("configuration_error", "Set TEACHER_MODEL or OPENAI_MODEL.")
        try:
            parsed = parse.urlsplit(base_url)
            valid = (
                parsed.scheme in {"http", "https"}
                and bool(parsed.hostname)
                and parsed.username is None
                and parsed.password is None
                and not parsed.query
                and not parsed.fragment
            )
            _ = parsed.port
        except ValueError:
            valid = False
        if not valid:
            raise TeacherError("configuration_error", "Teacher base URL must be an HTTP(S) URL without credentials, query or fragment.")
        return cls(api_key=api_key, base_url=base_url, model=model)

    def complete(self, messages: list, tools: list, timeout: float) -> dict:
        """Return the assistant message; callers retain the overall task deadline.

        ``timeout`` applies to blocking HTTP operations. The runner must enforce
        its wall-clock deadline as well, including against a slowly streaming peer.
        No retries are made, so a failed call does not silently consume the budget.
        """
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise TeacherError("timeout", "Teacher time budget has expired.")
        if not isinstance(messages, list) or not isinstance(tools, list):
            raise TeacherError("invalid_request", "Messages and tools must be lists.")
        try:
            payload = json.dumps(
                {"model": self.model, "messages": messages, "tools": tools, "tool_choice": "auto"},
                ensure_ascii=False, allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError):
            raise TeacherError("invalid_request", "Teacher request is not valid JSON.") from None
        req = request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=payload,
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.build_opener(_NoRedirect()).open(req, timeout=timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except error.HTTPError as exc:
            code = exc.code
            exc.close()
            raise TeacherError("http_error", f"Teacher API returned HTTP {code}.") from None
        except (socket.timeout, TimeoutError):
            raise TeacherError("timeout", "Teacher API timed out.") from None
        except error.URLError as exc:
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                raise TeacherError("timeout", "Teacher API timed out.") from None
            raise TeacherError("connection_error", "Could not connect to teacher API.") from None
        except (OSError, ValueError, HTTPException):
            raise TeacherError("connection_error", "Teacher HTTP request failed.") from None
        if len(raw) > MAX_RESPONSE_BYTES:
            raise TeacherError("response_too_large", "Teacher response exceeds 2 MiB.")
        try:
            data = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
            raise TeacherError("invalid_response", "Teacher returned invalid JSON.") from None
        return _assistant_message(data)


def _assistant_message(data: object) -> dict:
    if not isinstance(data, dict):
        raise TeacherError("invalid_response", "Teacher response must be a JSON object.")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise TeacherError("invalid_response", "Teacher response has no completion choice.")
    choice = choices[0]
    if choice.get("finish_reason") in {"length", "content_filter"}:
        raise TeacherError("incomplete_response", "Teacher completion was truncated or filtered.")
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise TeacherError("invalid_response", "Teacher response has no assistant message.")
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise TeacherError("invalid_response", "Assistant content must be text or null.")
    result = {"role": "assistant", "content": content}
    calls = message.get("tool_calls")
    if calls is not None:
        if not isinstance(calls, list):
            raise TeacherError("invalid_response", "Assistant tool_calls must be a list.")
        clean_calls = []
        seen = set()
        for call in calls:
            if not isinstance(call, dict):
                raise TeacherError("invalid_response", "Malformed assistant tool call.")
            call_id = call.get("id")
            function = call.get("function")
            if (
                not isinstance(call_id, str) or not call_id.strip() or call_id in seen
                or call.get("type") != "function" or not isinstance(function, dict)
                or not isinstance(function.get("name"), str) or not function["name"].strip()
                or not isinstance(function.get("arguments"), str)
            ):
                raise TeacherError("invalid_response", "Malformed assistant tool call.")
            seen.add(call_id)
            clean_calls.append({
                "id": call_id, "type": "function",
                "function": {"name": function["name"], "arguments": function["arguments"]},
            })
        if clean_calls:
            result["tool_calls"] = clean_calls
    if not result.get("tool_calls") and not content:
        raise TeacherError("invalid_response", "Assistant message contains neither text nor tool calls.")
    return result
