"""Strict deterministic result verification, without model-based judging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MAX_RESULT_BYTES = 1_000_000


class DuplicateKeyError(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def strict_json_loads(text: str) -> Any:
    """Reject duplicate keys and non-JSON NaN/Infinity constants."""
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_nonfinite)


def _mismatch(expected: Any, actual: Any, path: str = "$") -> str | None:
    # bool is a subclass of int in Python; comparing types prevents True == 1.
    if type(expected) is not type(actual):
        return f"{path}: expected {type(expected).__name__}, got {type(actual).__name__}"
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            return f"{path}: missing keys={sorted(set(expected) - set(actual))}, extra keys={sorted(set(actual) - set(expected))}"
        for key in sorted(expected):
            problem = _mismatch(expected[key], actual[key], f"{path}.{key}")
            if problem:
                return problem
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path}: expected {len(expected)} rows, got {len(actual)}"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            problem = _mismatch(expected_item, actual_item, f"{path}[{index}]")
            if problem:
                return problem
    elif expected != actual:
        return f"{path}: incorrect value"
    return None


def verify_result(reference_path: Path, result_path: Path) -> dict:
    """Return a stable rejection code and diagnostic details for a result file."""
    reference_path, result_path = Path(reference_path), Path(result_path)
    reference = strict_json_loads(reference_path.read_text(encoding="utf-8"))
    if not result_path.exists():
        return {"accepted": False, "reason": "missing_result", "details": "output/result.json was not written"}
    if result_path.is_symlink() or not result_path.is_file():
        return {"accepted": False, "reason": "invalid_result_path", "details": "result must be a regular file, not a symlink"}
    if result_path.stat().st_size > MAX_RESULT_BYTES:
        return {"accepted": False, "reason": "result_too_large", "details": f"maximum {MAX_RESULT_BYTES} bytes"}
    try:
        actual = strict_json_loads(result_path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError, RecursionError) as error:
        return {"accepted": False, "reason": "invalid_json", "details": str(error)}
    problem = _mismatch(reference["expected"], actual)
    if problem:
        return {"accepted": False, "reason": "result_mismatch", "details": problem}
    return {"accepted": True, "reason": "verified", "details": "all fields, types, values and row order match the reference"}
