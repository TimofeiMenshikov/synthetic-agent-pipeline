import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest

from synthetic_agent_pipeline.tools import ToolRuntime, TOOLS


class ToolsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name) / "workspace"
        (self.workspace / "data").mkdir(parents=True)
        (self.workspace / "data/items.csv").write_text("id,amount\na,12\nb,34\n", encoding="utf-8")
        with closing(sqlite3.connect(self.workspace / "data/items.sqlite")) as connection, connection:
            connection.execute("CREATE TABLE items (id TEXT PRIMARY KEY, amount INTEGER NOT NULL)")
            connection.executemany("INSERT INTO items VALUES (?, ?)", [("a", 12), ("b", 34)])
        (self.workspace / "reference.json").write_text('{"secret":true}', encoding="utf-8")
        self.runtime = ToolRuntime(self.workspace, time.monotonic() + 5)

    def tearDown(self):
        self.temp.cleanup()

    def test_normal_data_and_output_flow(self):
        self.assertEqual({tool["function"]["name"] for tool in TOOLS},
                         {"list_files", "read_csv", "inspect_database", "query_sql", "write_result"})
        files = self.runtime.execute("list_files", {})["files"]
        self.assertEqual([f["path"] for f in files], ["data/items.csv", "data/items.sqlite"])
        csv_result = self.runtime.execute("read_csv", {"path": "data/items.csv", "limit": 1})
        self.assertEqual(csv_result, {"columns": ["id", "amount"], "rows": [{"id": "a", "amount": "12"}], "total_rows": 2, "truncated": True})
        self.assertEqual(self.runtime.execute("read_csv", {"path": "data/items.csv", "offset": 1})["rows"], [{"id": "b", "amount": "34"}])
        schema = self.runtime.execute("inspect_database", {"path": "data/items.sqlite"})
        self.assertEqual(schema["tables"][0]["name"], "items")
        self.assertEqual([c["name"] for c in schema["tables"][0]["columns"]], ["id", "amount"])
        query = self.runtime.execute("query_sql", {"path": "data/items.sqlite", "query": "SELECT SUM(amount) AS total FROM items"})
        self.assertEqual(query, {"columns": ["total"], "rows": [{"total": 46}], "truncated": False})
        written = self.runtime.execute("write_result", {"result": {"total": 46}})
        self.assertEqual(written["path"], "output/result.json")
        self.assertEqual(json.loads((self.workspace / written["path"]).read_text()), {"total": 46})

    def test_paths_and_symlinks_are_rejected(self):
        outside = Path(self.temp.name) / "outside.csv"
        outside.write_text("secret\n1\n", encoding="utf-8")
        (self.workspace / "data/link.csv").symlink_to(outside)
        (self.workspace / "data/linkdir").symlink_to(outside.parent, target_is_directory=True)
        for path in ("../outside.csv", str(outside), "reference.json", "data/link.csv", "data/linkdir/outside.csv", "data/../reference.json"):
            with self.subTest(path=path):
                self.assertIn("error", self.runtime.execute("read_csv", {"path": path}))
        self.assertEqual(len(self.runtime.execute("list_files", {})["files"]), 2)
        (self.workspace / "output").symlink_to(outside.parent, target_is_directory=True)
        self.assertEqual(self.runtime.execute("write_result", {"result": {}})["error"]["code"], "invalid_path")
        self.assertFalse((outside.parent / "result.json").exists())

    def test_output_file_symlink_is_rejected(self):
        output = self.workspace / "output"
        output.mkdir()
        target = Path(self.temp.name) / "unmodified.txt"
        target.write_text("keep")
        (output / "result.json").symlink_to(target)
        self.assertIn("error", self.runtime.execute("write_result", {"result": {}}))
        self.assertEqual(target.read_text(), "keep")

    def test_sql_cannot_mutate_or_access_extensions(self):
        queries = [
            "DELETE FROM items", "PRAGMA table_info(items)", "ATTACH DATABASE ':memory:' AS other",
            "SELECT * FROM items; DELETE FROM items", "SELECT load_extension('anything')",
            "WITH x AS (SELECT 1) DELETE FROM items", "SELECT readfile('/etc/passwd')",
        ]
        for query in queries:
            with self.subTest(query=query):
                self.assertIn("error", self.runtime.execute("query_sql", {"path": "data/items.sqlite", "query": query}))
        with closing(sqlite3.connect(self.workspace / "data/items.sqlite")) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM items").fetchone()[0], 2)

    def test_strict_arguments_and_result_limits(self):
        calls = [
            ("unknown", {}), ("list_files", {"path": "x"}), ("list_files", []),
            ("read_csv", {"path": "data/items.csv", "limit": True}),
            ("read_csv", {"path": "data/items.csv", "offset": -1}),
            ("read_csv", {"path": "data/items.csv", "limit": 1001}),
            ("write_result", {"result": []}), ("write_result", {"result": {"nan": float("nan")}}),
            ("write_result", {"result": {"big": "x" * (1024 * 1024)}}),
        ]
        for name, arguments in calls:
            with self.subTest(name=name, arguments=str(arguments)[:100]):
                self.assertIn("error", self.runtime.execute(name, arguments))
        self.assertFalse((self.workspace / "output/result.json").exists())
        large_query = "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<1001) SELECT x FROM n"
        self.assertEqual(self.runtime.execute("query_sql", {"path": "data/items.sqlite", "query": large_query})["error"]["code"], "row_limit")
        dup_query = self.runtime.execute("query_sql", {"path": "data/items.sqlite", "query": "SELECT id,id FROM items"})
        self.assertEqual(dup_query["error"]["code"], "invalid_sql_result")

    def test_bad_csv_and_file_size(self):
        (self.workspace / "data/bad.csv").write_text("id,id\na,b\n")
        self.assertEqual(self.runtime.execute("read_csv", {"path": "data/bad.csv"})["error"]["code"], "invalid_csv")
        (self.workspace / "data/bad.csv").write_text("a,b\n1\n")
        self.assertEqual(self.runtime.execute("read_csv", {"path": "data/bad.csv"})["error"]["code"], "invalid_csv")
        (self.workspace / "data/huge.csv").write_text("x" * (2 * 1024 * 1024 + 1))
        self.assertEqual(self.runtime.execute("read_csv", {"path": "data/huge.csv"})["error"]["code"], "size_limit")

    def test_deadline_stops_query_and_later_calls(self):
        runtime = ToolRuntime(self.workspace, time.monotonic() + 0.03)
        query = "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n) SELECT SUM(x) AS total FROM n"
        start = time.monotonic()
        result = runtime.execute("query_sql", {"path": "data/items.sqlite", "query": query})
        self.assertEqual(result["error"]["code"], "timeout")
        self.assertLess(time.monotonic() - start, 1)
        self.assertEqual(runtime.execute("list_files", {})["error"]["code"], "timeout")


if __name__ == "__main__":
    unittest.main()
