import copy
import json
from pathlib import Path
import tempfile
import unittest

from synthetic_agent_pipeline.export import TraceValidationError
from synthetic_agent_pipeline.tools import TOOLS
from synthetic_agent_pipeline.traces import convert_generation_spans, generation_span


class GenerationTraceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "generation_spans.jsonl"
        self.messages = [{"role": "system", "content": "Use tools."}, {"role": "user", "content": "Compare the two CSV files."}]
        self.spans = []
        plan = [
            ("read_csv", {"path": "data/a.csv"}),
            ("read_csv", {"path": "data/b.csv"}),
            ("write_result", {"result": {"discrepancies": []}}),
        ]
        for step, (name, arguments) in enumerate(plan, 1):
            assistant = {"role": "assistant", "content": None, "tool_calls": [{
                "id": f"call_{step}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)},
            }]}
            self.spans.append(generation_span("task-a", step, self.messages, assistant, TOOLS))
            self.messages.extend([assistant, {"role": "tool", "tool_call_id": f"call_{step}", "content": '{}'}])
        final = {"role": "assistant", "content": "Saved output/result.json."}
        self.spans.append(generation_span("task-a", 4, self.messages, final, TOOLS))
        self.messages.append(final)

    def write(self, spans):
        self.path.write_text("".join(json.dumps(span) + "\n" for span in spans), encoding="utf-8")

    def test_last_generation_exports_history_without_duplicate_prefixes(self):
        # File order is not trusted: selection uses explicit generation step.
        self.write(list(reversed(self.spans)))
        record = convert_generation_spans(self.path)
        self.assertEqual(record["messages"], self.messages)
        self.assertEqual(len(record["messages"]), 9)
        self.assertEqual(sum(message["role"] == "user" for message in record["messages"]), 1)
        self.assertEqual(record["tools"], TOOLS)
        self.assertEqual((record["trace_id"], record["span_id"]), ("task-a", "task-a:generation:0004"))

    def test_unfinished_last_turn_is_rejected(self):
        self.write(self.spans[:-1])
        with self.assertRaises(TraceValidationError):
            convert_generation_spans(self.path)

    def test_mixed_tasks_or_parent_agents_are_rejected(self):
        for field in ("trace_id", "parentObservationId"):
            with self.subTest(field=field):
                spans = copy.deepcopy(self.spans)
                spans[-1][field] = "unrelated-task"
                self.write(spans)
                with self.assertRaisesRegex(TraceValidationError, "mixed_generation_tasks_or_parents"):
                    convert_generation_spans(self.path)

    def test_snapshots_do_not_mutate_when_caller_appends(self):
        self.assertEqual(len(self.spans[0]["input"]), 2)
        messages = copy.deepcopy(self.messages[:2])
        assistant = copy.deepcopy(self.messages[-1])
        tools = copy.deepcopy(TOOLS)
        span = generation_span("snapshot", 1, messages, assistant, tools)
        messages[1]["content"] = "changed"
        assistant["content"] = "changed"
        tools[0]["function"]["name"] = "changed"
        self.assertEqual(span["input"][1]["content"], "Compare the two CSV files.")
        self.assertEqual(span["output"]["content"], "Saved output/result.json.")
        self.assertEqual(span["tools"][0]["function"]["name"], "list_files")

    def test_malformed_snapshot_and_duplicate_steps_are_rejected(self):
        for field, value in (("input", {}), ("output", []), ("tools", {}), ("step", True), ("span_type", "EVENT")):
            with self.subTest(field=field):
                spans = copy.deepcopy(self.spans)
                spans[0][field] = value
                self.write(spans)
                with self.assertRaises(TraceValidationError):
                    convert_generation_spans(self.path)
        self.write(self.spans + [self.spans[-1]])
        with self.assertRaisesRegex(TraceValidationError, "duplicate_generation_step_or_id"):
            convert_generation_spans(self.path)

    def test_inconsistent_history_and_tool_schemas_are_rejected(self):
        spans = copy.deepcopy(self.spans)
        spans[-1]["input"][2]["tool_calls"][0]["function"]["arguments"] = '{"path":"data/forged.csv"}'
        self.write(spans)
        with self.assertRaisesRegex(TraceValidationError, "inconsistent_generation_history"):
            convert_generation_spans(self.path)
        spans = copy.deepcopy(self.spans)
        spans[-1]["tools"] = []
        self.write(spans)
        with self.assertRaisesRegex(TraceValidationError, "changing_generation_tools"):
            convert_generation_spans(self.path)

    def test_final_chain_validation_checks_tool_call_linkage(self):
        spans = copy.deepcopy(self.spans[-1:])
        spans[0]["input"][3]["tool_call_id"] = "orphan"
        self.write(spans)
        with self.assertRaisesRegex(TraceValidationError, "orphan_or_duplicate_tool_response"):
            convert_generation_spans(self.path)

    def test_empty_or_invalid_json_trace_is_rejected(self):
        for text in ("", "not-json\n", "{}\n"):
            with self.subTest(text=text):
                self.path.write_text(text, encoding="utf-8")
                with self.assertRaises(TraceValidationError):
                    convert_generation_spans(self.path)


if __name__ == "__main__":
    unittest.main()
