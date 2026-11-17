from copy import deepcopy
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from uuid import UUID

from synthetic_agent_pipeline.export import (
    TraceValidationError, export_accepted, serialize_record, validate_dataset, validate_messages,
)
from synthetic_agent_pipeline.tools import TOOLS


def call(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments),
    }}


def valid_messages():
    return [
        {"role": "system", "content": "Use the tools."},
        {"role": "user", "content": "Compare the data and write the result."},
        {"role": "assistant", "content": None, "tool_calls": [
            call("read1", "read_csv", {"path": "data/a.csv"}),
            call("read2", "query_sql", {"path": "data/a.sqlite", "query": "SELECT 1 AS n"}),
        ]},
        {"role": "tool", "tool_call_id": "read2", "content": '{"rows":[{"n":1}]}'},
        {"role": "tool", "tool_call_id": "read1", "content": '{"rows":[]}'},
        {"role": "assistant", "content": None, "tool_calls": [call("write", "write_result", {"result": {"total": 1}})]},
        {"role": "tool", "tool_call_id": "write", "content": '{"path":"output/result.json"}'},
        {"role": "assistant", "content": "Written output/result.json."},
    ]


class ExportTest(unittest.TestCase):
    def assert_invalid(self, messages, tools=None, reason=None):
        with self.assertRaises(TraceValidationError) as caught:
            validate_messages(messages, TOOLS if tools is None else tools)
        if reason:
            self.assertIn(reason, str(caught.exception))

    def test_parallel_calls_and_reverse_responses_are_valid(self):
        validate_messages(valid_messages(), TOOLS)

    def test_serializer_matches_reference_wire_shape_without_mutating_api_trace(self):
        messages = valid_messages()
        original = deepcopy(messages)
        record = serialize_record(messages, TOOLS, "sample-task", "mock")
        reference = json.loads((Path(__file__).resolve().parents[1] / "reference-format.json").read_text())
        expected_fields = reference["tool_calling_wire_format"]["top_level"]
        self.assertEqual(set(record), set(expected_fields))
        self.assertEqual(messages, original)
        self.assertIsNone(messages[2]["content"])
        self.assertNotIn("tool_calls", messages[0])
        for message in record["messages"]:
            self.assertIsInstance(message["content"], str)
            self.assertIsInstance(message["tool_calls"], list)
        self.assertEqual(record["messages"][2]["content"], "")
        self.assertEqual(record["messages"][0]["tool_calls"], [])
        self.assertEqual(record["generator"], "synthetic-agent-pipeline/mock")
        self.assertEqual(UUID(record["uuid"]).version, 5)
        teacher = serialize_record(messages, TOOLS, "sample-task", "teacher")
        self.assertEqual(teacher["uuid"], record["uuid"])
        self.assertEqual(teacher["generator"], "synthetic-agent-pipeline/teacher")
        self.assertNotEqual(serialize_record(messages, TOOLS, "other-task", "mock")["uuid"], record["uuid"])
        validate_messages(record["messages"], record["tools"])

    def test_empty_calls_on_all_roles_are_allowed_but_nonempty_are_assistant_only(self):
        messages = serialize_record(valid_messages(), TOOLS, "sample-task", "mock")["messages"]
        validate_messages(messages, TOOLS)
        for index in (0, 1, 3):
            bad = deepcopy(messages)
            bad[index]["tool_calls"] = [call("illegal", "list_files", {})]
            self.assert_invalid(bad, reason="calls_outside_assistant")
            bad[index]["tool_calls"] = None
            self.assert_invalid(bad, reason="tool_calls_must_be_list")

    def test_first_two_reference_records_streamed_when_archive_is_available(self):
        path = (Path(__file__).resolve().parents[2] / "nex-archive-2026-11-17" /
                "huggingface/dataset/nex-agi/agent-sft/tool_calling.jsonl")
        if not path.exists():
            self.skipTest("Local source archive is optional for unit tests")
        with path.open("rb") as stream:
            for index in range(2):
                record = json.loads(stream.readline())
                with self.subTest(record=index + 1):
                    self.assertEqual(set(record), {"messages", "tools", "uuid", "generator"})
                    self.assertIsInstance(record["tools"], list)
                    self.assertIsInstance(record["uuid"], str)
                    self.assertIsInstance(record["generator"], str)
                    validate_messages(record["messages"], record["tools"], require_actions=False)

    def test_orphan_duplicate_and_missing_responses(self):
        messages = valid_messages()
        messages[3]["tool_call_id"] = "orphan"
        self.assert_invalid(messages, reason="orphan_or_duplicate")
        messages = valid_messages()
        messages.insert(5, deepcopy(messages[4]))
        self.assert_invalid(messages, reason="orphan_or_duplicate")
        messages = valid_messages()
        del messages[4]
        self.assert_invalid(messages, reason="missing_tool_responses")
        self.assert_invalid(valid_messages()[:3], reason="missing_tool_responses")

    def test_global_duplicate_ids_and_interleaved_parallel_calls(self):
        messages = valid_messages()
        messages[5]["tool_calls"][0]["id"] = "read1"
        self.assert_invalid(messages, reason="duplicate_or_invalid_call_id")
        messages = valid_messages()
        messages[2]["tool_calls"][1]["id"] = "read1"
        self.assert_invalid(messages, reason="duplicate_or_invalid_call_id")
        messages = valid_messages()
        messages[4], messages[5] = messages[5], messages[4]
        self.assert_invalid(messages, reason="missing_tool_responses")

    def test_unknown_names_and_bad_function_arguments(self):
        messages = valid_messages()
        messages[2]["tool_calls"][0]["function"]["name"] = "shell"
        self.assert_invalid(messages, reason="unknown_tool")
        for bad in ("[]", "null", "not json", '{"x":1,"x":2}', '{"x":NaN}', '{"x":1e309}', {}):
            with self.subTest(arguments=bad):
                messages = valid_messages()
                messages[2]["tool_calls"][0]["function"]["arguments"] = bad
                self.assert_invalid(messages, reason="invalid_tool_arguments")
        messages = valid_messages()
        messages[3]["name"] = "read_csv"
        self.assert_invalid(messages, reason="tool_response_name_mismatch")

    def test_scalar_messages_and_nested_calls_raise_validation_errors(self):
        for bad in (None, 1, "messages", {}, ["system", "user", "assistant"]):
            with self.subTest(messages=bad):
                self.assert_invalid(bad)
        for value in (None, 1, "calls", {}, [None], [1], ["call"]):
            with self.subTest(tool_calls=value):
                messages = valid_messages()
                messages[2]["tool_calls"] = value
                self.assert_invalid(messages)
        for value in (None, 1, [], "function"):
            with self.subTest(function=value):
                messages = valid_messages()
                messages[2]["tool_calls"][0]["function"] = value
                self.assert_invalid(messages)
        messages = valid_messages()
        messages[3]["tool_call_id"] = []
        self.assert_invalid(messages)
        messages = valid_messages()
        messages[0]["role"] = []
        self.assert_invalid(messages)
        messages = valid_messages()
        messages[5]["role"] = []
        self.assert_invalid(messages)

    def test_malformed_tool_schemas_raise_validation_errors(self):
        bad_tools = [None, 1, "tools", {}, [None], [1],
                     [{"type": "function", "function": []}],
                     [{"type": "function", "function": {"name": "x", "parameters": 1}}]]
        for field, value in (("properties", 1), ("properties", {"x": 1}), ("required", 1),
                             ("required", [[]]), ("required", ["missing"]), ("additionalProperties", [])):
            tool = deepcopy(TOOLS[1])
            tool["function"]["parameters"][field] = value
            bad_tools.append([tool])
        duplicate = deepcopy(TOOLS)
        duplicate.append(deepcopy(duplicate[0]))
        bad_tools.append(duplicate)
        for tools in bad_tools:
            with self.subTest(tools=tools):
                with self.assertRaises(TraceValidationError):
                    validate_messages(valid_messages(), tools)

    def test_failed_tools_and_missing_actions_are_rejected(self):
        for content in ('{"error":{"code":"timeout"}}', "[]", "null", "not json", '{"x":1,"x":2}'):
            with self.subTest(content=content):
                messages = valid_messages()
                messages[3]["content"] = content
                self.assert_invalid(messages)
        messages = valid_messages()[:2] + [{"role": "assistant", "content": "Done."}]
        self.assert_invalid(messages, reason="insufficient_tool_actions")
        validate_messages(messages, TOOLS, require_actions=False)
        messages = valid_messages()
        messages[-1]["content"] = " "
        self.assert_invalid(messages, reason="missing_final_assistant_answer")

    def test_only_accepted_export_and_whole_family_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory = root / "messages.json"
            trajectory.write_text(json.dumps(valid_messages()), encoding="utf-8")
            records = []
            for task_id, family in (("a1", "reconcile"), ("a2", "reconcile"), ("b1", "summary"), ("b2", "summary")):
                records.append({"task_id": task_id, "family": family, "difficulty": "easy", "seed": 42, "mode": "mock",
                                "trajectory_path": str(trajectory), "verification": {"accepted": True}})
            records.append({"verification": {"accepted": False}, "trajectory_path": "/does/not/exist"})
            counts = export_accepted(records, root / "exports", {"summary"}, TOOLS)
            self.assertEqual(counts, {"train": 2, "validation": 2})
            self.assertEqual(validate_dataset(root / "exports/train.jsonl"), 2)
            self.assertEqual(validate_dataset(root / "exports/validation.jsonl"), 2)
            first = json.loads((root / "exports/train.jsonl").read_text().splitlines()[0])
            self.assertEqual(set(first), {"messages", "tools", "uuid", "generator"})
            self.assertEqual(first["generator"], "synthetic-agent-pipeline/mock")
            index = [json.loads(line) for line in (root / "exports/index.jsonl").read_text().splitlines()]
            self.assertEqual(len(index), 4)
            self.assertEqual({item["family"] for item in index if item["split"] == "train"}, {"reconcile"})
            self.assertEqual({item["family"] for item in index if item["split"] == "validation"}, {"summary"})
            self.assertTrue(all(len(item["sha256"]) == 64 for item in index))

    def test_dataset_checks_scalar_records_and_annotates_line_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            good_line = json.dumps({"messages": valid_messages(), "tools": TOOLS}) + "\n"
            for bad_record in (None, 1, "record", [], {"messages": 1, "tools": TOOLS}, {"messages": valid_messages(), "tools": 1}):
                with self.subTest(record=bad_record):
                    path.write_text(good_line + json.dumps(bad_record) + "\n", encoding="utf-8")
                    with self.assertRaises(TraceValidationError) as caught:
                        validate_dataset(path)
                    self.assertIn(":2:", str(caught.exception))
            for bad_line in (b"not json\n", b"\xff\n", b'{"messages":[],"tools":[],"tools":[]}\n'):
                with self.subTest(line=bad_line):
                    path.write_bytes(bad_line)
                    with self.assertRaises(TraceValidationError):
                        validate_dataset(path)

    def test_dataset_accepts_optional_metadata_only_with_string_types(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            record = serialize_record(valid_messages(), TOOLS, "sample-task", "mock")
            path.write_text(json.dumps(record) + "\n")
            self.assertEqual(validate_dataset(path), 1)
            for field in ("uuid", "generator"):
                for value in (None, 1, [], ""):
                    with self.subTest(field=field, value=value):
                        bad = deepcopy(record)
                        bad[field] = value
                        path.write_text(json.dumps(bad) + "\n")
                        with self.assertRaises(TraceValidationError):
                            validate_dataset(path)

    def test_check_jsonl_cli_reports_bad_schema_without_traceback(self):
        from synthetic_agent_pipeline.cli import main

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(json.dumps({"messages": valid_messages(), "tools": 42}) + "\n")
            stderr = io.StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
                main(["check-jsonl", str(path)])
            self.assertEqual(caught.exception.code, 2)
            self.assertIn("tools_must_be_list", json.loads(stderr.getvalue())["error"])


if __name__ == "__main__":
    unittest.main()
