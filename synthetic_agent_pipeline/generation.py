"""Seeded local CSV/SQLite tasks with references kept outside public data.

No teacher is used to generate references: each answer is computed directly from
the generated Python records before they are serialized to the agent's inputs.

The framework/persona -> problem-type path -> difficulty organization and query
export fields adapt the approach inspected in NexGAP/Gyrfalcon/core/pipeline.py,
frameworks/{menu_margin_pricing_system,pitlane_proof}/ and
llm_interface/query_generator.py:QueryExporter. This implementation uses a fixed
declarative tree and seeded data builders rather than LLM-generated questions or
mutable tag sampling. The downloaded NexGAP files are never imported or changed.
"""

from __future__ import annotations

import csv
from contextlib import closing
import hashlib
import json
from pathlib import Path
import random
import sqlite3
from typing import Any


GENERATOR_VERSION = "local-csv-sqlite-v1"
CONFIG_PATH = Path(__file__).parent / "configs" / "local_csv_sqlite.json"


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _database(path: Path, statements: list[tuple[str, list[tuple] | None]]) -> None:
    # A fresh database, fixed SQL order and fixed insertion order make SQLite
    # bytes repeatable for the same SQLite build as well as the logical records.
    if path.exists():
        path.unlink()
    with closing(sqlite3.connect(path)) as connection:
        with connection:
            connection.execute("PRAGMA page_size = 4096")
            for statement, records in statements:
                if records is None:
                    connection.execute(statement)
                else:
                    connection.executemany(statement, records)


def _reconciliation(data: Path, rng: random.Random, size: int, changes: int) -> tuple[str, dict]:
    ledger = [
        {"invoice_id": f"INV-{i:04d}", "customer_id": f"C-{rng.randrange(1, 9):03d}", "amount_cents": rng.randrange(100, 100_001)}
        for i in range(1, size + 1)
    ]
    expected = {row["invoice_id"]: row["amount_cents"] for row in ledger}
    actual = dict(expected)
    changed_ids = rng.sample(sorted(actual), changes)
    del actual[changed_ids[0]]
    for invoice_id in changed_ids[1:]:
        actual[invoice_id] += rng.choice((-1, 1)) * rng.randrange(1, 100)
    actual[f"INV-{size + 1:04d}"] = rng.randrange(100, 10_001)
    payments = [{"invoice_id": invoice_id, "paid_cents": amount} for invoice_id, amount in actual.items()]
    rng.shuffle(ledger)
    rng.shuffle(payments)
    _csv(data / "ledger.csv", ["invoice_id", "customer_id", "amount_cents"], ledger)
    _csv(data / "payments.csv", ["invoice_id", "paid_cents"], payments)
    discrepancies = []
    for invoice_id in sorted(expected.keys() | actual.keys()):
        if invoice_id not in actual:
            issue = "missing_payment"
        elif invoice_id not in expected:
            issue = "unexpected_payment"
        elif expected[invoice_id] != actual[invoice_id]:
            issue = "amount_mismatch"
        else:
            continue
        discrepancies.append({
            "invoice_id": invoice_id,
            "expected_cents": expected.get(invoice_id, 0),
            "actual_cents": actual.get(invoice_id, 0),
            "delta_cents": actual.get(invoice_id, 0) - expected.get(invoice_id, 0),
            "issue": issue,
        })
    prompt = """Сверь счета из data/ledger.csv с платежами из data/payments.csv. В ledger ожидаемая сумма — amount_cents, в payments фактическая — paid_cents. invoice_id уникален в каждом файле, порядок строк произвольный. Рассмотри объединение invoice_id обоих файлов. Для отсутствующей записи сумма равна 0. Включи только отсутствующие счета/платежи и несовпадающие суммы. issue: missing_payment, если платежа нет; unexpected_payment, если нет счета; иначе amount_mismatch. delta_cents = actual_cents - expected_cents. Все суммы — целые копейки.
Изучи оба исходных файла инструментами и запиши output/result.json ровно в формате:
{"discrepancies":[{"invoice_id":"INV-0001","expected_cents":100,"actual_cents":90,"delta_cents":-10,"issue":"amount_mismatch"}]}
Отсортируй discrepancies лексикографически по invoice_id. Не добавляй совпадающие записи или другие поля. Если расхождений нет, верни пустой список."""
    return prompt, {"discrepancies": discrepancies}


def _inventory(data: Path, rng: random.Random, size: int, changes: int) -> tuple[str, dict]:
    opening = [(f"SKU-{i:04d}", rng.randrange(60, 200)) for i in range(1, size + 1)]
    movements: list[tuple] = []
    quantities = dict(opening)
    for sku, _ in opening:
        for _ in range(rng.randrange(1, 5)):
            delta = rng.choice((-1, 1)) * rng.randrange(1, 16)
            movements.append((len(movements) + 1, sku, delta))
            quantities[sku] += delta
    recorded = dict(quantities)
    for sku in rng.sample(sorted(recorded), changes):
        recorded[sku] += rng.choice((-1, 1)) * rng.randrange(1, 8)
    snapshot = [{"sku": sku, "recorded_qty": quantity} for sku, quantity in recorded.items()]
    rng.shuffle(snapshot)
    _csv(data / "snapshot.csv", ["sku", "recorded_qty"], snapshot)
    _database(data / "inventory.sqlite", [
        ("CREATE TABLE opening (sku TEXT PRIMARY KEY, opening_qty INTEGER NOT NULL)", None),
        ("CREATE TABLE movements (movement_id INTEGER PRIMARY KEY, sku TEXT NOT NULL, delta_qty INTEGER NOT NULL)", None),
        ("INSERT INTO opening VALUES (?, ?)", opening),
        ("INSERT INTO movements VALUES (?, ?, ?)", movements),
    ])
    discrepancies = [
        {"sku": sku, "expected_qty": quantities[sku], "recorded_qty": recorded[sku], "delta_qty": recorded[sku] - quantities[sku]}
        for sku in sorted(quantities) if quantities[sku] != recorded[sku]
    ]
    prompt = """Найди расхождения в складских остатках. data/inventory.sqlite содержит opening(sku, opening_qty) и movements(movement_id, sku, delta_qty). Правильный остаток expected_qty = opening_qty + сумма всех delta_qty для sku; если движений нет, прибавь 0. data/snapshot.csv содержит зафиксированный остаток recorded_qty для каждого sku из opening; sku уникален, порядок строк произвольный. delta_qty = recorded_qty - expected_qty.
Изучи CSV и схему базы инструментами, выполни SQL для расчёта остатков и запиши output/result.json ровно в формате:
{"discrepancies":[{"sku":"SKU-0001","expected_qty":100,"recorded_qty":97,"delta_qty":-3}]}
Включи только ненулевые расхождения, отсортируй список лексикографически по sku, все количества — целые числа. Не добавляй другие поля. При отсутствии расхождений верни пустой список."""
    return prompt, {"discrepancies": discrepancies}


def _sales(data: Path, rng: random.Random, size: int, changes: int) -> tuple[str, dict]:
    del changes
    regions = ("Central", "East", "North", "South", "West")
    customers = [(f"C-{i:03d}", regions[(i - 1) % len(regions)]) for i in range(1, 11)]
    customer_regions = dict(customers)
    orders = [
        {
            "order_id": f"ORD-{i:04d}",
            "customer_id": rng.choice(customers)[0],
            "amount_cents": rng.randrange(100, 100_001),
            "status": rng.choice(("paid", "paid", "pending", "refunded")),
        }
        for i in range(1, size + 1)
    ]
    # Include at least one paid order in every instance, without constraining
    # which regions have zero sales.
    orders[0]["status"] = "paid"
    counts = {region: 0 for region in regions}
    revenue = {region: 0 for region in regions}
    for order in orders:
        if order["status"] == "paid":
            region = customer_regions[order["customer_id"]]
            counts[region] += 1
            revenue[region] += order["amount_cents"]
    rng.shuffle(orders)
    _csv(data / "orders.csv", ["order_id", "customer_id", "amount_cents", "status"], orders)
    _database(data / "customers.sqlite", [
        ("CREATE TABLE customers (customer_id TEXT PRIMARY KEY, region TEXT NOT NULL)", None),
        ("INSERT INTO customers VALUES (?, ?)", customers),
    ])
    summary = [{"region": region, "paid_order_count": counts[region], "revenue_cents": revenue[region]} for region in sorted(regions)]
    prompt = """Построй сводку оплаченных продаж по регионам. data/orders.csv содержит order_id, customer_id, amount_cents, status; order_id уникален. data/customers.sqlite содержит customers(customer_id, region); каждый клиент из CSV есть в базе. Учитывай только заказы со status = paid; pending и refunded исключи целиком. paid_order_count — количество оплаченных заказов, revenue_cents — сумма их amount_cents (целые копейки). Включи все различные region из customers, даже если оплаченных заказов нет: тогда оба значения равны 0.
Прочитай заказы и схему базы инструментами, получи клиентов SQL-запросом и запиши output/result.json ровно в формате:
{"summary":[{"region":"Central","paid_order_count":2,"revenue_cents":12345}]}
Отсортируй summary лексикографически по region. Количество и выручка — целые числа. Не добавляй другие поля."""
    return prompt, {"summary": summary}


_BUILDERS = {"table_reconciliation": _reconciliation, "inventory_discrepancies": _inventory, "sales_summary": _sales}


def load_framework_config(path: Path | None = None) -> dict:
    """Load and validate the local framework, problem tree and difficulty plan."""
    config = json.loads((CONFIG_PATH if path is None else Path(path)).read_text(encoding="utf-8"))
    families = config["generation"]["family_order"]
    difficulties = config["generation"]["difficulty_order"]
    if not isinstance(families, list) or len(families) != len(set(families)) or set(families) != set(_BUILDERS):
        raise ValueError("framework family_order must list each registered builder exactly once")
    if not isinstance(difficulties, list) or len(difficulties) != 3 or set(difficulties) != {"easy", "medium", "hard"}:
        raise ValueError("difficulty_order must contain easy, medium and hard exactly once")
    if config["generation"]["family_strategy"] != "round_robin" or config["generation"]["difficulty_strategy"] != "per_family_round_robin":
        raise ValueError("unsupported generation schedule")
    for difficulty in difficulties:
        parameters = config["generation"]["difficulty_parameters"][difficulty]
        size, changes = parameters["rows"], parameters["changes"]
        if type(size) is not int or type(changes) is not int or not 0 < changes < size <= 200:
            raise ValueError("difficulty parameters require 0 < changes < rows <= 200")
    if not set(config["validation_families"]).issubset(families):
        raise ValueError("unknown validation family")
    nodes, leaves = set(), set()

    def visit(node: dict) -> None:
        node_id = node["id"]
        if not isinstance(node_id, str) or not node_id or node_id in nodes or not isinstance(node["label"], str):
            raise ValueError("problem tree needs unique nonempty ids and text labels")
        nodes.add(node_id)
        if node.get("children"):
            for child in node["children"]:
                visit(child)
        else:
            if node_id not in families or not isinstance(node.get("persona"), str) or not node["persona"]:
                raise ValueError("problem tree leaves need a registered family id and persona")
            leaves.add(node_id)

    visit(config["problem_types"]["tree"])
    if leaves != set(families):
        raise ValueError("problem tree must cover every family")
    return config


FRAMEWORK_CONFIG = load_framework_config()
FAMILIES = tuple(FRAMEWORK_CONFIG["generation"]["family_order"])
DIFFICULTIES = tuple(FRAMEWORK_CONFIG["generation"]["difficulty_order"])
VALIDATION_FAMILIES = tuple(FRAMEWORK_CONFIG["validation_families"])


def _problem_context(family: str) -> tuple[list[str], dict]:
    def visit(node: dict, ancestors: list[str]) -> tuple[list[str], dict] | None:
        labels = ancestors + [node["label"]]
        if node["id"] == family:
            return labels, node
        for child in node.get("children", []):
            found = visit(child, labels)
            if found is not None:
                return found
        return None

    found = visit(FRAMEWORK_CONFIG["problem_types"]["tree"], [])
    if found is None:
        raise ValueError(f"family not found in problem tree: {family}")
    return found


def generate_batch(output_dir: Path, count: int, seed: int) -> list[dict]:
    """Create deterministic tasks; refuse to mix a batch into a nonempty folder."""
    output_dir = Path(output_dir)
    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"generation destination is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    queries: list[dict] = []
    for index in range(count):
        family = FAMILIES[index % len(FAMILIES)]
        level = (index // len(FAMILIES)) % len(DIFFICULTIES)
        difficulty = DIFFICULTIES[level]
        digest = hashlib.sha256(f"{GENERATOR_VERSION}:{seed}:{index}".encode()).hexdigest()
        task_seed = int(digest[:16], 16)
        task_id = f"{family}-{digest[:16]}"
        public_dir = Path("tasks") / task_id / "public"
        data = output_dir / public_dir / "data"
        data.mkdir(parents=True)
        parameters = FRAMEWORK_CONFIG["generation"]["difficulty_parameters"][difficulty]
        prompt, expected = _BUILDERS[family](data, random.Random(task_seed), parameters["rows"], parameters["changes"])
        metadata = {
            "task_id": task_id,
            "family": family,
            "difficulty": difficulty,
            "seed": seed,
            "task_seed": task_seed,
            "generator_version": GENERATOR_VERSION,
            "prompt": prompt,
        }
        _json(output_dir / public_dir / "task.json", metadata)
        reference_path = Path("references") / f"{task_id}.json"
        _json(output_dir / reference_path, {"task_id": task_id, "family": family, "difficulty": difficulty, "expected": expected})
        manifest.append({
            "task_id": task_id,
            "family": family,
            "difficulty": difficulty,
            "seed": seed,
            "public_dir": public_dir.as_posix(),
            "reference_path": reference_path.as_posix(),
        })
        labels, problem = _problem_context(family)
        queries.append({
            "query": prompt,
            "difficulty": difficulty,
            "trace_context": labels,
            "problem_type": problem["label"],
            "framework": FRAMEWORK_CONFIG["framework"]["name"],
            "requires_local_files": True,
            "used_web_search": False,
            "fuzzified": False,
            "metadata": {
                "task_id": task_id,
                "family": family,
                "seed": seed,
                "task_seed": task_seed,
                "generator_version": GENERATOR_VERSION,
                "persona": problem["persona"],
            },
        })
    (output_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n" for entry in manifest), encoding="utf-8"
    )
    (output_dir / "queries.jsonl").write_text(
        "".join(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n" for entry in queries), encoding="utf-8"
    )
    return manifest
