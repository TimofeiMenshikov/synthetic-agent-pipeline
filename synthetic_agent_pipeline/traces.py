"""Convert local generation snapshots into one complete SFT conversation.

Conceptual sources, read from the unmodified local NexGAP archive:
``nex-archive-2026-11-17/github/NexGAP/converter/trace/get_trace.py:103``
records per-generation input/output, trace/span IDs and parentObservationId.
``converter/trace/convert_spans_to_chatcompletion.py:1637`` selects the last
generation per parent; its ``:1689`` conversion appends the assistant output
to that generation's full request history and retains tools/provenance.
``github/NexGAP/run_end_to_end.py:84`` separates raw and converted traces.

This is an independent, strict implementation for native JSON tool calls. It
does not import NexGAP, fabricate missing call IDs or rewrite tool arguments.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from .verification import strict_json_loads


def _invalid(reason: str) -> None:
    # export imports this converter too; defer importing its shared exception.
    from .export import TraceValidationError
    raise TraceValidationError(reason)


def _check_snapshot(span: object) -> None:
    if type(span) is not dict:
        _invalid("generation_span_must_be_object")
    for key in ("trace_id", "span_id", "parentObservationId"):
        if type(span.get(key)) is not str or not span[key].strip():
            _invalid("missing_generation_identity")
    if type(span.get("step")) is not int or span["step"] < 1:
        _invalid("invalid_generation_step")
    if span.get("span_type") != "GENERATION" or span.get("span_name") != "OpenAI-generation":
        _invalid("unsupported_generation_span")
    inputs, output, tools = span.get("input"), span.get("output"), span.get("tools")
    if type(inputs) is not list or len(inputs) < 2 or any(type(item) is not dict for item in inputs):
        _invalid("invalid_generation_input")
    if inputs[0].get("role") != "system" or inputs[1].get("role") != "user":
        _invalid("invalid_generation_input")
    if type(output) is not dict or output.get("role") != "assistant":
        _invalid("invalid_generation_output")
    calls = output.get("tool_calls", [])
    if type(calls) is not list:
        _invalid("invalid_generation_output")
    content = output.get("content")
    if content is not None and type(content) is not str:
        _invalid("invalid_generation_output")
    if not calls and (type(content) is not str or not content.strip()):
        _invalid("empty_generation_output")
    if type(tools) is not list:
        _invalid("invalid_generation_tools")
    try:
        json.dumps(span, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        _invalid("invalid_generation_json")


def generation_span(task_id: str, step: int, messages: list, assistant: dict, tools: list) -> dict:
    """Capture an immutable snapshot before the caller appends tool responses.

    Raw traces also retain unfinished generations for diagnosing rejected tasks.
    The converter validates a complete final conversation before SFT export.
    """
    if type(task_id) is not str or not task_id.strip():
        _invalid("invalid_generation_task_id")
    if type(step) is not int or step < 1:
        _invalid("invalid_generation_step")
    span = {
        "trace_id": task_id,
        "span_id": f"{task_id}:generation:{step:04d}",
        "parentObservationId": task_id,
        "span_type": "GENERATION",
        "span_name": "OpenAI-generation",
        "step": step,
        "input": deepcopy(messages),
        "output": deepcopy(assistant),
        "tools": deepcopy(tools),
    }
    _check_snapshot(span)
    return span


def convert_generation_spans(path: Path) -> dict:
    """Select one latest generation, preserving its history exactly once.

    One file belongs to a single task and parent agent. Reject mixed identities,
    duplicate steps/IDs, changing tools or inconsistent request histories rather
    than silently joining unrelated runs. Final tool-call linkage is checked by
    the same validator used for SFT JSONL records.
    """
    from .export import TraceValidationError, validate_messages

    spans, identities, steps, span_ids = [], set(), set(), set()
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                span = strict_json_loads(line)
            except (ValueError, UnicodeError, RecursionError):
                raise TraceValidationError(f"invalid_generation_json_line_{line_number}") from None
            _check_snapshot(span)
            identities.add((span["trace_id"], span["parentObservationId"]))
            if len(identities) != 1:
                _invalid("mixed_generation_tasks_or_parents")
            if span["step"] in steps or span["span_id"] in span_ids:
                _invalid("duplicate_generation_step_or_id")
            steps.add(span["step"])
            span_ids.add(span["span_id"])
            spans.append(span)
    if not spans:
        _invalid("empty_generation_trace")
    spans.sort(key=lambda span: span["step"])
    previous = None
    for span in spans:
        if span["tools"] != spans[0]["tools"]:
            _invalid("changing_generation_tools")
        if span["input"][:2] != spans[0]["input"][:2]:
            _invalid("mixed_generation_task_prompts")
        if previous is not None:
            history = previous["input"] + [previous["output"]]
            if span["input"][:len(history)] != history:
                _invalid("inconsistent_generation_history")
            if not previous["output"].get("tool_calls"):
                _invalid("generation_after_final_response")
        previous = span
    latest = spans[-1]
    messages = deepcopy(latest["input"]) + [deepcopy(latest["output"])]
    tools = deepcopy(latest["tools"])
    validate_messages(messages, tools)
    return {"messages": messages, "tools": tools,
            "trace_id": latest["trace_id"], "span_id": latest["span_id"]}
