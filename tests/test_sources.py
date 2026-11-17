import json
from pathlib import Path
import tempfile
import unittest

from synthetic_agent_pipeline.sources import compare_reference_format, stream_sample, wire_shape_errors


def wire_record():
    return {"messages": [{"role": "system", "content": "system", "tool_calls": []},
                         {"role": "user", "content": "task", "tool_calls": []},
                         {"role": "assistant", "content": "done", "tool_calls": []}],
            "tools": [], "uuid": "synthetic-id", "generator": "unit-test"}


class SourceTests(unittest.TestCase):
    def test_stream_sample_stops_before_bad_third_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.jsonl"
            path.write_text((json.dumps(wire_record()) + "\n") * 2 + "invalid-json-do-not-read\n")
            self.assertEqual(len(list(stream_sample(path))), 2)

    def test_shape_rejects_our_previous_native_api_serialization(self):
        row = wire_record()
        self.assertEqual(wire_shape_errors(row), [])
        row["messages"][2]["content"] = None
        del row["messages"][0]["tool_calls"]
        del row["generator"]
        errors = wire_shape_errors(row)
        self.assertIn("content_must_be_string", errors)
        self.assertIn("every_message_needs_tool_calls_list", errors)
        self.assertIn("generator_must_be_nonempty_string", errors)

    def test_compare_export_and_missing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export = root / "export.jsonl"
            export.write_text(json.dumps(wire_record()) + "\n")
            self.assertFalse(compare_reference_format(export, root)["verified"])
            source = root / "huggingface/dataset/nex-agi/agent-sft/tool_calling.jsonl"
            source.parent.mkdir(parents=True)
            source.write_text((json.dumps(wire_record()) + "\n") * 2)
            self.assertEqual(compare_reference_format(export, root)["exported_rows"], 1)
            self.assertTrue(compare_reference_format(export, root)["verified"])
            invalid = wire_record()
            invalid["messages"][2]["content"] = None
            export.write_text(json.dumps(invalid) + "\n")
            self.assertFalse(compare_reference_format(export, root)["verified"])


if __name__ == "__main__":
    unittest.main()
