from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from synthetic_agent_pipeline.generation import CONFIG_PATH, FAMILIES, generate_batch, load_framework_config
from synthetic_agent_pipeline.verification import verify_result


class GenerationTests(unittest.TestCase):
    def test_declarative_framework_export_has_context_without_reference(self):
        config = load_framework_config()
        self.assertEqual(config["generation"]["family_order"], list(FAMILIES))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = generate_batch(root / "batch", 9, 12)
            queries = [json.loads(line) for line in (root / "batch" / "queries.jsonl").read_text().splitlines()]
            self.assertEqual(len(queries), len(tasks))
            for query, task in zip(queries, tasks):
                self.assertEqual(query["metadata"]["task_id"], task["task_id"])
                self.assertEqual(query["difficulty"], task["difficulty"])
                self.assertEqual(query["framework"], "local_csv_sqlite")
                self.assertEqual(query["trace_context"][-1], query["problem_type"])
                self.assertEqual(len(query["trace_context"]), 3)
                self.assertTrue(query["requires_local_files"])
                self.assertFalse(query["used_web_search"])
                self.assertFalse(query["fuzzified"])
                self.assertNotIn("expected", query)
                self.assertNotIn("reference_path", query["metadata"])
                prompt = json.loads((root / "batch" / task["public_dir"] / "task.json").read_text())["prompt"]
                self.assertEqual(query["query"], prompt)
            config["problem_types"]["tree"]["children"][1]["children"][0]["id"] = FAMILIES[0]
            bad_config = root / "bad.json"
            bad_config.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_framework_config(bad_config)
            self.assertTrue(CONFIG_PATH.is_file())

    def test_framework_adaptation_preserves_seeded_inputs_and_references(self):
        # This fixture was recorded before moving the scheduling parameters into
        # config. Normalize SQLite rows so it does not pin a SQLite file format.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshots = []
            for entry in generate_batch(root, 3, 20261117):
                public = root / entry["public_dir"]
                record = {"manifest": entry, "reference": json.loads((root / entry["reference_path"]).read_text()), "public": {}}
                for file in sorted(public.rglob("*")):
                    if not file.is_file():
                        continue
                    if file.suffix == ".sqlite":
                        connection = sqlite3.connect(file)
                        try:
                            tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
                            value = {table: connection.execute('SELECT * FROM "' + table + '" ORDER BY 1').fetchall() for table in tables}
                        finally:
                            connection.close()
                    else:
                        value = file.read_text()
                    record["public"][file.relative_to(public).as_posix()] = value
                snapshots.append(record)
            fingerprint = hashlib.sha256(json.dumps(snapshots, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            self.assertEqual(fingerprint, "a1af3a9c7a02e98a1204256e365c4b3a8318048dfb7931bd73a10749ef8b0909")

    def test_seed_reproduces_all_files_including_sqlite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second, different = root / "first", root / "second", root / "different"
            tasks = generate_batch(first, 20, 42)
            self.assertEqual(tasks, generate_batch(second, 20, 42))
            def hashes(directory):
                return {path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.rglob("*") if path.is_file()}
            self.assertEqual(hashes(first), hashes(second))
            self.assertNotEqual(tasks, generate_batch(different, 20, 43))
            self.assertEqual({entry["family"] for entry in tasks}, set(FAMILIES))
            self.assertEqual({entry["difficulty"] for entry in tasks}, {"easy", "medium", "hard"})
            for entry in tasks:
                public = first / entry["public_dir"]
                self.assertFalse((first / entry["reference_path"]).is_relative_to(public))
                self.assertNotIn("expected", json.loads((public / "task.json").read_text()))

    def test_reference_results_pass_and_wrong_results_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for entry in generate_batch(root / "batch", 3, 9):
                reference_path = root / "batch" / entry["reference_path"]
                expected = json.loads(reference_path.read_text())["expected"]
                result = root / "result.json"
                result.write_text(json.dumps(expected))
                self.assertTrue(verify_result(reference_path, result)["accepted"])
                key = next(iter(expected))
                bad = copy.deepcopy(expected)
                bad[key].pop()
                result.write_text(json.dumps(bad))
                self.assertFalse(verify_result(reference_path, result)["accepted"])
                bad = copy.deepcopy(expected)
                numeric = next(field for field, value in bad[key][0].items() if type(value) is int)
                bad[key][0][numeric] += 1
                result.write_text(json.dumps(bad))
                self.assertFalse(verify_result(reference_path, result)["accepted"])
                bad[key][0][numeric] = True
                result.write_text(json.dumps(bad))
                self.assertFalse(verify_result(reference_path, result)["accepted"])
                bad = copy.deepcopy(expected)
                bad[key].append(bad[key][0])
                result.write_text(json.dumps(bad))
                self.assertFalse(verify_result(reference_path, result)["accepted"])
                bad = copy.deepcopy(expected)
                bad["extra"] = 1
                result.write_text(json.dumps(bad))
                self.assertFalse(verify_result(reference_path, result)["accepted"])

    def test_invalid_json_duplicate_keys_and_missing_file_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry = generate_batch(root / "batch", 1, 0)[0]
            reference = root / "batch" / entry["reference_path"]
            result = root / "result.json"
            self.assertEqual(verify_result(reference, result)["reason"], "missing_result")
            for content in ('{"discrepancies":[],"discrepancies":[]}', '{"discrepancies":NaN}', '```json\n{}\n```'):
                result.write_text(content)
                self.assertEqual(verify_result(reference, result)["reason"], "invalid_json")

    def test_refuses_nonempty_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate_batch(root, 1, 2)
            with self.assertRaises(FileExistsError):
                generate_batch(root, 1, 2)


if __name__ == "__main__":
    unittest.main()
