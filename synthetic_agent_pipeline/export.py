"""Validate complete tool-call conversations and write only accepted runs."""

import hashlib
from copy import deepcopy
import json
import math
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from .verification import strict_json_loads


class TraceValidationError(ValueError):
    pass


def _json_value(value) -> None:
    """Reject Python-only values before they can be coerced by JSON encoding."""
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _json_value(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _json_value(item)
        return
    raise TraceValidationError("invalid_json_value")


def _tool_names(tools: list) -> set[str]:
    if type(tools) is not list:
        raise TraceValidationError("tools_must_be_list")
    _json_value(tools)
    names = set()
    for tool in tools:
        if type(tool) is not dict or tool.get("type") != "function" or type(tool.get("function")) is not dict:
            raise TraceValidationError("invalid_tool_schema")
        function = tool["function"]
        name, parameters = function.get("name"), function.get("parameters")
        if type(name) is not str or not name.strip() or name in names:
            raise TraceValidationError("duplicate_or_invalid_tool_name")
        if "description" in function and type(function["description"]) is not str:
            raise TraceValidationError("invalid_tool_schema")
        if type(parameters) is not dict or parameters.get("type") != "object":
            raise TraceValidationError("invalid_tool_parameters_schema")
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])
        if (type(properties) is not dict or any(type(schema) is not dict for schema in properties.values())
                or type(required) is not list or any(type(key) is not str for key in required)
                or len(required) != len(set(required)) or not set(required).issubset(properties)):
            raise TraceValidationError("invalid_tool_parameters_schema")
        if "additionalProperties" in parameters and type(parameters["additionalProperties"]) not in {dict, bool}:
            raise TraceValidationError("invalid_tool_parameters_schema")
        names.add(name)
    return names


def validate_messages(messages: list, tools: list, require_actions: bool = True) -> None:
    if type(messages) is not list or len(messages) < 3:
        raise TraceValidationError("incomplete_conversation")
    if any(type(message) is not dict for message in messages):
        raise TraceValidationError("messages_must_be_objects")
    try:
        _json_value(messages)
        names = _tool_names(tools)
    except RecursionError as exc:
        raise TraceValidationError("json_nesting_too_deep") from exc
    if messages[0].get("role") != "system" or messages[1].get("role") != "user":
        raise TraceValidationError("missing_system_or_user")
    pending, seen, used = {}, set(), []
    for message in messages:
        role = message.get("role")
        if type(role) is not str or role not in {"system", "user", "assistant", "tool"}:
            raise TraceValidationError("invalid_role")
        if "name" in message and (type(message["name"]) is not str or not message["name"].strip()):
            raise TraceValidationError("invalid_message_name")
        if "tool_call_id" in message and role != "tool":
            raise TraceValidationError("tool_call_id_outside_tool")
        calls = message.get("tool_calls", [])
        if type(calls) is not list:
            raise TraceValidationError("tool_calls_must_be_list")
        if calls and role != "assistant":
            raise TraceValidationError("calls_outside_assistant")
        if role == "tool":
            call_id = message.get("tool_call_id")
            if type(call_id) is not str or call_id not in pending:
                raise TraceValidationError("orphan_or_duplicate_tool_response")
            if "name" in message and message["name"] != pending[call_id]:
                raise TraceValidationError("tool_response_name_mismatch")
            del pending[call_id]
            if not isinstance(message.get("content"), str):
                raise TraceValidationError("tool_content_must_be_string")
            try:
                result = strict_json_loads(message["content"])
                _json_value(result)
            except (ValueError, TypeError, RecursionError) as exc:
                raise TraceValidationError("invalid_tool_json") from exc
            if not isinstance(result, dict) or "error" in result:
                raise TraceValidationError("failed_tool_call")
            continue
        if pending:
            raise TraceValidationError("missing_tool_responses")
        if not calls and not isinstance(message.get("content"), str):
            raise TraceValidationError("message_content_must_be_string")
        if calls and message.get("content") is not None and not isinstance(message["content"], str):
            raise TraceValidationError("invalid_assistant_content")
        for call in calls:
            if type(call) is not dict:
                raise TraceValidationError("invalid_tool_call")
            call_id = call.get("id")
            if not isinstance(call_id, str) or not call_id.strip() or call_id in seen:
                raise TraceValidationError("duplicate_or_invalid_call_id")
            fn = call.get("function", {})
            if (call.get("type") != "function" or type(fn) is not dict
                    or type(fn.get("name")) is not str or fn["name"] not in names):
                raise TraceValidationError("unknown_tool")
            try:
                if not isinstance(fn.get("arguments"), str):
                    raise ValueError("arguments")
                arguments = strict_json_loads(fn["arguments"])
                if type(arguments) is not dict:
                    raise ValueError("arguments")
                _json_value(arguments)
            except (ValueError, TypeError, RecursionError) as exc:
                raise TraceValidationError("invalid_tool_arguments") from exc
            seen.add(call_id)
            pending[call_id] = fn["name"]
            used.append(fn["name"])
    if pending:
        raise TraceValidationError("missing_tool_responses")
    final = messages[-1]
    if final.get("role") != "assistant" or final.get("tool_calls") or not final.get("content", "").strip():
        raise TraceValidationError("missing_final_assistant_answer")
    reads = sum(name in {"read_csv", "query_sql"} for name in used)
    if require_actions and (reads < 2 or "write_result" not in used):
        raise TraceValidationError("insufficient_tool_actions")


def serialize_record(messages: list, tools: list, task_id: str, mode: str) -> dict:
    """Project native API messages to the sampled Nex agent-sft wire format.

    API traces remain unchanged: empty ``tool_calls`` and text content are added
    only to a copy written to the dataset.
    """
    validate_messages(messages, tools)
    if type(task_id) is not str or not task_id.strip():
        raise TraceValidationError("invalid_task_id")
    if type(mode) is not str or mode not in {"mock", "teacher", "unknown"}:
        raise TraceValidationError("invalid_export_mode")
    normalized = deepcopy(messages)
    for message in normalized:
        message.setdefault("tool_calls", [])
        if message.get("content") is None:
            message["content"] = ""
    return {
        "messages": normalized,
        "tools": deepcopy(tools),
        "uuid": str(uuid5(NAMESPACE_URL, "synthetic-agent-pipeline:" + task_id)),
        "generator": "synthetic-agent-pipeline/" + mode,
    }


def validate_dataset(path: Path) -> int:
    count = 0
    with Path(path).open("rb") as stream:
        for line_no, line in enumerate(stream, 1):
            try:
                record = strict_json_loads(line.decode("utf-8"))
                if (type(record) is not dict or not {"messages", "tools"}.issubset(record)
                        or set(record) - {"messages", "tools", "uuid", "generator"}):
                    raise TraceValidationError("invalid_record_keys")
                for key in ("uuid", "generator"):
                    if key in record and (type(record[key]) is not str or not record[key].strip()):
                        raise TraceValidationError("invalid_record_metadata")
                validate_messages(record["messages"], record["tools"])
            except (ValueError, TypeError, KeyError, RecursionError) as exc:
                raise TraceValidationError(f"{path}:{line_no}: {exc}") from exc
            count += 1
    return count


def export_accepted(records: list, destination: Path, validation_families: set, tools: list) -> dict:
    if type(records) is not list:
        raise TraceValidationError("records_must_be_list")
    _tool_names(tools)
    for record in records:
        if (type(record) is not dict or type(record.get("verification")) is not dict
                or type(record["verification"].get("accepted")) is not bool):
            raise TraceValidationError("invalid_verification_record")
    destination.mkdir(parents=True, exist_ok=False)
    counts = {"train": 0, "validation": 0}
    family_splits = {}
    with (destination / "train.jsonl").open("w", encoding="utf-8") as train, \
         (destination / "validation.jsonl").open("w", encoding="utf-8") as validation, \
         (destination / "index.jsonl").open("w", encoding="utf-8") as index:
        for record in records:
            if record["verification"]["accepted"] is not True:
                continue
            try:
                messages = strict_json_loads(Path(record["trajectory_path"]).read_text(encoding="utf-8"))
            except (ValueError, TypeError, KeyError, RecursionError) as exc:
                raise TraceValidationError("invalid_trajectory") from exc
            trace_id, span_id = record["task_id"], None
            if record.get("generation_spans_path"):
                from .traces import convert_generation_spans
                converted = convert_generation_spans(Path(record["generation_spans_path"]))
                if (converted["messages"] != messages or converted["tools"] != tools
                        or converted["trace_id"] != record["task_id"]):
                    raise TraceValidationError("generation_snapshot_mismatch")
                messages = converted["messages"]
                trace_id, span_id = converted["trace_id"], converted["span_id"]
            validate_messages(messages, tools)
            family = record["family"]
            split = "validation" if family in validation_families else "train"
            family_splits.setdefault(family, set()).add(split)
            serialized = serialize_record(messages, tools, record["task_id"], record.get("mode", "unknown"))
            line = json.dumps(serialized, ensure_ascii=False, allow_nan=False)
            (train if split == "train" else validation).write(line + "\n")
            counts[split] += 1
            index.write(json.dumps({
                key: record[key] for key in ("task_id", "family", "difficulty", "seed")
            } | {"split": split, "line": counts[split], "sha256": hashlib.sha256(line.encode()).hexdigest(),
                 "trace_id": trace_id, "span_id": span_id, "uuid": serialized["uuid"],
                 "generator": serialized["generator"]},
                ensure_ascii=False) + "\n")
    if any(len(splits) > 1 for splits in family_splits.values()):
        raise TraceValidationError("family_leakage")
    for split in counts:
        if validate_dataset(destination / f"{split}.jsonl") != counts[split]:
            raise TraceValidationError("export_count_mismatch")
    return counts
