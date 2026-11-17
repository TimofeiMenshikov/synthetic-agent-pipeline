"""Read-only source inspection with bounded streaming dataset samples."""

import hashlib
import json
from pathlib import Path


MAX_SAMPLE_LINE_BYTES = 8_000_000


def stream_sample(path: Path, count: int = 2):
    """Read only a few bounded lines, even for multi-gigabyte source datasets."""
    with Path(path).open("rb") as stream:
        for _ in range(count):
            line = stream.readline(MAX_SAMPLE_LINE_BYTES + 1)
            if not line:
                break
            if len(line) > MAX_SAMPLE_LINE_BYTES:
                raise ValueError("sample_line_too_large")
            yield json.loads(line)


def wire_shape_errors(row: object) -> list[str]:
    """Wire conventions observed in the local Nex-AGI tool_calling sample."""
    if not isinstance(row, dict):
        return ["record_must_be_object"]
    errors = []
    if set(row) != {"messages", "tools", "uuid", "generator"}:
        errors.append("record_keys_must_match_tool_calling_sample")
    for name in ("uuid", "generator"):
        if not isinstance(row.get(name), str) or not row[name]:
            errors.append(name + "_must_be_nonempty_string")
    if not isinstance(row.get("tools"), list):
        errors.append("tools_must_be_list")
    if not isinstance(row.get("messages"), list):
        return errors + ["messages_must_be_list"]
    for message in row["messages"]:
        if not isinstance(message, dict):
            errors.append("message_must_be_object")
            continue
        if not isinstance(message.get("content"), str):
            errors.append("content_must_be_string")
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            errors.append("every_message_needs_tool_calls_list")
            continue
        if message.get("role") != "assistant" and calls:
            errors.append("only_assistant_can_call_tools")
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                errors.append("invalid_tool_call")
                continue
            if not isinstance(call["function"].get("arguments"), str):
                errors.append("arguments_must_be_json_string")
        if message.get("role") == "tool" and not isinstance(message.get("tool_call_id"), str):
            errors.append("tool_response_needs_call_id")
    return sorted(set(errors))


def compare_reference_format(dataset: Path, archive: Path) -> dict:
    """Validate all exported rows against conventions from two reference rows.

    This verifies serialization, not statistical similarity or model quality.
    Source records are never copied into an export or passed to the teacher.
    """
    reference = Path(archive) / "huggingface/dataset/nex-agi/agent-sft/tool_calling.jsonl"
    if not reference.is_file():
        return {"verified": False, "reason": "reference_dataset_missing"}
    reference_rows = list(stream_sample(reference))
    if not reference_rows or any(wire_shape_errors(row) for row in reference_rows):
        return {"verified": False, "reason": "unrecognized_reference_sample"}
    count, failures = 0, []
    with Path(dataset).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            count += 1
            errors = wire_shape_errors(json.loads(line))
            if errors:
                failures.append({"line": line_no, "errors": errors})
    return {"verified": not failures, "reference_sample_rows": len(reference_rows),
            "exported_rows": count, "failures": failures,
            "scope": "serialization shape from bounded local sample; tool-call links checked separately"}


def audit_sources(archive: Path) -> dict:
    archive = Path(archive).resolve()
    if not archive.is_dir():
        return {"available": False, "archive": str(archive), "reason": "source_archive_missing",
                "format_compatibility": "unverified; standard OpenAI messages/tools schema used"}
    result = {"available": True, "archive": str(archive), "sources": [], "samples": {}}
    required = ["github/NexGAP/Gyrfalcon/core/pipeline.py", "github/NexGAP/run_end_to_end.py",
                "github/NexGAP/Gyrfalcon/llm_interface/query_generator.py"]
    paths = [archive / path for path in required]
    for relative in ["github/NexGAP/converter/trace"]:
        paths.extend(sorted((archive / relative).rglob("*.py"))[:5])
    for framework in ("menu_margin_pricing_system", "pitlane_proof"):
        for filename in ("framework_config.yaml", "config.json", "problem_types.json"):
            paths.append(archive / "github/NexGAP/Gyrfalcon/frameworks" / framework / filename)
    for path in paths:
        if not path.is_file():
            result["sources"].append({"path": str(path.relative_to(archive)), "missing": True})
            continue
        with path.open("rb") as stream:
            sample = stream.read(512_001)
        result["sources"].append({"path": str(path.relative_to(archive)), "sample_sha256": hashlib.sha256(sample).hexdigest(),
                                  "bytes_read": len(sample), "truncated": len(sample) > 512_000,
                                  "preview": sample[:4000].decode("utf-8", errors="replace")})
    for name in ("tool_calling.jsonl", "agent.jsonl"):
        path = archive / "huggingface/dataset/nex-agi/agent-sft" / name
        samples = []
        if not path.is_file():
            result["samples"][name] = {"missing": True}
            continue
        try:
            for row in stream_sample(path):
                messages = row.get("messages", [])
                samples.append({"keys": sorted(row), "field_types": {key: type(value).__name__ for key, value in row.items()},
                                "message_roles": [m.get("role") for m in messages if isinstance(m, dict)],
                                "message_keys": sorted({key for message in messages for key in message}),
                                "content_types": sorted({type(m.get("content")).__name__ for m in messages}),
                                "argument_types": sorted({type(c["function"]["arguments"]).__name__ for m in messages for c in m.get("tool_calls", [])}),
                                "tool_definition_keys": sorted({key for tool in row.get("tools", []) for key in tool}),
                                "wire_shape_errors": wire_shape_errors(row) if name == "tool_calling.jsonl" else None})
        except (ValueError, TypeError, AttributeError, KeyError):
            samples.append({"error": "unrecognized_or_oversize_sample"})
        result["samples"][name] = samples
    result["format_compatibility"] = "observed local sample; use check-reference to compare an export"
    return result
