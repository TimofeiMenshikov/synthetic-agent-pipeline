"""Small capability-based data tools. The model never receives a shell or Python."""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import time
from typing import Any


def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object", "properties": properties,
                "required": required, "additionalProperties": False,
            },
        },
    }


TOOLS = [
    _schema("list_files", "List available CSV and SQLite input files in data/.", {}, []),
    _schema("read_csv", "Read a page of a UTF-8 CSV; rows contain strings.", {
        "path": {"type": "string"},
        "offset": {"type": "integer", "minimum": 0},
        "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
    }, ["path"]),
    _schema("inspect_database", "Inspect table and column names in a SQLite input database.", {
        "path": {"type": "string"},
    }, ["path"]),
    _schema("query_sql", "Run one read-only SELECT or WITH query; at most 1000 result rows.", {
        "path": {"type": "string"}, "query": {"type": "string"},
    }, ["path", "query"]),
    _schema("write_result", "Write the final JSON object to output/result.json.", {
        "result": {"type": "object"},
    }, ["result"]),
]


class ToolError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ToolRuntime:
    """Only public data/ is readable; only output/result.json is writable.

    Path checks defend against model-provided paths and static symlinks. The task
    runner must own the workspace and not allow concurrent untrusted local writes.
    """

    CSV_BYTES = 2 * 1024 * 1024
    DB_BYTES = 10 * 1024 * 1024
    RESULT_BYTES = 1024 * 1024

    def __init__(self, workspace: Path, deadline: float):
        workspace = Path(workspace)
        if workspace.is_symlink():
            raise ValueError("Workspace must not be a symlink")
        self.workspace = workspace.resolve(strict=True)
        if not self.workspace.is_dir():
            raise ValueError("Workspace must be a directory")
        self.deadline = deadline

    def _check_time(self) -> None:
        if time.monotonic() >= self.deadline:
            raise ToolError("timeout", "Task deadline exceeded")

    @staticmethod
    def _arguments(arguments: dict, required: set[str], optional: set[str] | None = None) -> None:
        if type(arguments) is not dict:
            raise ToolError("invalid_arguments", "Arguments must be an object")
        if not required.issubset(arguments) or set(arguments) - required - (optional or set()):
            raise ToolError("invalid_arguments", "Missing required or unknown arguments")

    def _data_path(self, path: str, suffixes: set[str], max_bytes: int) -> Path:
        if type(path) is not str or not path or "\\" in path or "\x00" in path:
            raise ToolError("invalid_path", "Expected a relative data/ path")
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != "data":
            raise ToolError("invalid_path", "Only files inside data/ are readable")
        candidate = self.workspace
        for part in relative.parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise ToolError("invalid_path", "Symlinks are not allowed")
        if not candidate.is_file():
            raise ToolError("not_found", "Input file not found")
        if candidate.suffix.lower() not in suffixes:
            raise ToolError("invalid_path", "Unsupported input file extension")
        if not candidate.resolve().is_relative_to(self.workspace / "data"):
            raise ToolError("invalid_path", "Input path leaves data/")
        if candidate.stat().st_size > max_bytes:
            raise ToolError("size_limit", "Input file exceeds size limit")
        return candidate

    def execute(self, name: str, arguments: dict) -> dict:
        try:
            self._check_time()
            handlers = {
                "list_files": self._list_files, "read_csv": self._read_csv,
                "inspect_database": self._inspect_database, "query_sql": self._query_sql,
                "write_result": self._write_result,
            }
            if name not in handlers:
                raise ToolError("unknown_tool", "Unknown tool")
            result = handlers[name](arguments)
            self._check_time()
            return result
        except ToolError as exc:
            return {"error": {"code": exc.code, "message": str(exc)}}
        except (OSError, UnicodeError, csv.Error) as exc:
            return {"error": {"code": "input_error", "message": str(exc).replace(str(self.workspace), "<workspace>")}}
        except (sqlite3.Error, sqlite3.Warning) as exc:
            # Python 3.10 raises Warning for multiple statements; it is not an Error subclass.
            code = "timeout" if time.monotonic() >= self.deadline else "sql_error"
            return {"error": {"code": code, "message": "Task deadline exceeded" if code == "timeout" else str(exc)}}
        except (ValueError, TypeError, OverflowError, RecursionError):
            return {"error": {"code": "invalid_arguments", "message": "Invalid or non-JSON tool arguments"}}

    def _list_files(self, arguments: dict) -> dict:
        self._arguments(arguments, set())
        data_dir = self.workspace / "data"
        if data_dir.is_symlink():
            raise ToolError("invalid_path", "Symlinks are not allowed")
        files = []
        if data_dir.exists():
            for path in sorted(data_dir.rglob("*")):
                self._check_time()
                if path.is_symlink() or not path.is_file() or path.suffix.lower() not in {".csv", ".sqlite", ".db"}:
                    continue
                if any(p.is_symlink() for p in path.parents if p != self.workspace and p.is_relative_to(self.workspace)):
                    continue
                files.append({"path": path.relative_to(self.workspace).as_posix(), "size": path.stat().st_size})
                if len(files) > 1000:
                    raise ToolError("size_limit", "Too many input files")
        return {"files": files}

    def _read_csv(self, arguments: dict) -> dict:
        self._arguments(arguments, {"path"}, {"offset", "limit"})
        offset, limit = arguments.get("offset", 0), arguments.get("limit", 200)
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 1000:
            raise ToolError("invalid_arguments", "offset must be nonnegative; limit must be 1..1000")
        path = self._data_path(arguments["path"], {".csv"}, self.CSV_BYTES)
        raw = path.read_bytes()
        if len(raw) > self.CSV_BYTES:
            raise ToolError("size_limit", "Input file exceeds size limit")
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
        columns = reader.fieldnames
        if not columns or len(columns) != len(set(columns)) or any(not c for c in columns):
            raise ToolError("invalid_csv", "CSV must have unique nonempty column names")
        rows, total = [], 0
        for row in reader:
            if total % 100 == 0:
                self._check_time()
            if None in row or any(v is None for v in row.values()):
                raise ToolError("invalid_csv", "CSV row width does not match header")
            if offset <= total < offset + limit:
                rows.append(row)
            total += 1
        return {"columns": columns, "rows": rows, "total_rows": total, "truncated": offset + len(rows) < total}

    def _connect(self, path: Path, *, inspect: bool = False) -> sqlite3.Connection:
        connection = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True, timeout=max(0.001, min(1.0, self.deadline - time.monotonic())))
        connection.row_factory = sqlite3.Row
        if hasattr(connection, "setlimit"):
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, self.CSV_BYTES)
            connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 100_000)
            connection.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 100)
            connection.setlimit(sqlite3.SQLITE_LIMIT_COLUMN, 128)
        allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}

        def authorize(action: int, arg1: str | None, arg2: str | None, _database: str | None, _source: str | None) -> int:
            if action == sqlite3.SQLITE_FUNCTION and (arg2 or arg1 or "").lower() in {"load_extension", "writefile", "readfile"}:
                return sqlite3.SQLITE_DENY
            if inspect and action == sqlite3.SQLITE_PRAGMA and (arg1 or "").lower() == "table_info":
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

        connection.set_authorizer(authorize)
        connection.set_progress_handler(lambda: int(time.monotonic() >= self.deadline), 1000)
        return connection

    def _inspect_database(self, arguments: dict) -> dict:
        self._arguments(arguments, {"path"})
        path = self._data_path(arguments["path"], {".sqlite", ".db"}, self.DB_BYTES)
        connection = self._connect(path, inspect=True)
        try:
            names = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchmany(101)]
            if len(names) > 100:
                raise ToolError("size_limit", "Too many database tables")
            tables = []
            for name in names:
                self._check_time()
                columns = [{"name": row[1], "type": row[2], "not_null": bool(row[3]), "primary_key": bool(row[5])}
                           for row in connection.execute('PRAGMA table_info("' + name.replace('"', '""') + '")')]
                tables.append({"name": name, "columns": columns})
            return {"tables": tables}
        finally:
            connection.close()

    def _query_sql(self, arguments: dict) -> dict:
        self._arguments(arguments, {"path", "query"})
        query = arguments["query"]
        if type(query) is not str or len(query) > 100_000 or not re.match(r"^\s*(SELECT|WITH)\b", query, re.I):
            raise ToolError("invalid_arguments", "Expected one SELECT or WITH query")
        path = self._data_path(arguments["path"], {".sqlite", ".db"}, self.DB_BYTES)
        connection = self._connect(path)
        try:
            cursor = connection.execute(query)
            columns = [item[0] for item in cursor.description or []]
            if not columns or len(columns) != len(set(columns)):
                raise ToolError("invalid_sql_result", "Use unique names or aliases for all result columns")
            rows, encoded_size = [], 2
            for row in cursor:
                self._check_time()
                if len(rows) >= 1000:
                    raise ToolError("row_limit", "Query returned over 1000 rows; aggregate or use LIMIT")
                row = dict(row)
                try:
                    encoded_size += len(json.dumps(row, ensure_ascii=False, allow_nan=False).encode("utf-8")) + 2
                except (ValueError, TypeError):
                    raise ToolError("invalid_sql_result", "Query results must contain finite JSON values") from None
                if encoded_size > self.CSV_BYTES:
                    raise ToolError("size_limit", "Query result exceeds size limit")
                rows.append(row)
            return {"columns": columns, "rows": rows, "truncated": False}
        finally:
            connection.close()

    def _write_result(self, arguments: dict) -> dict:
        self._arguments(arguments, {"result"})
        if type(arguments["result"]) is not dict:
            raise ToolError("invalid_arguments", "result must be a JSON object")
        encoded = (json.dumps(arguments["result"], ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
        if len(encoded) > self.RESULT_BYTES:
            raise ToolError("size_limit", "Result exceeds 1 MiB")
        output_dir = self.workspace / "output"
        target = output_dir / "result.json"
        if output_dir.is_symlink() or target.is_symlink():
            raise ToolError("invalid_path", "Output symlinks are not allowed")
        if output_dir.exists() and not output_dir.is_dir():
            raise ToolError("invalid_path", "Output directory is not a directory")
        output_dir.mkdir(exist_ok=True)
        self._check_time()
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=output_dir, prefix=".result-", delete=False) as stream:
                temp_path = Path(stream.name)
                stream.write(encoded)
            self._check_time()
            os.replace(temp_path, target)
            return {"path": "output/result.json", "size": len(encoded)}
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
