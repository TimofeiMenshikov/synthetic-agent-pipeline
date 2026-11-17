"""A deterministic tool-using solver for offline pipeline checks only.

It receives the same tool observations as a remote teacher. It never opens a
file, database or reference itself. Its outputs must never enter real SFT data.
"""

import json
from collections import defaultdict


class MockTeacher:
    def __init__(self, family: str, inject_failure: bool = False):
        self.family = family
        self.inject_failure = inject_failure
        self.turn = 0
        self.wrote_result = False
        self.plan = [("list_files", {})]
        if family == "table_reconciliation":
            self.plan += [
                ("read_csv", {"path": "data/ledger.csv"}),
                ("read_csv", {"path": "data/payments.csv"}),
            ]
        elif family == "inventory_discrepancies":
            self.plan += [
                ("read_csv", {"path": "data/snapshot.csv"}),
                ("inspect_database", {"path": "data/inventory.sqlite"}),
                ("query_sql", {"path": "data/inventory.sqlite", "query": "SELECT sku, opening_qty FROM opening ORDER BY sku"}),
                ("query_sql", {"path": "data/inventory.sqlite", "query": "SELECT sku, SUM(delta_qty) AS delta_qty FROM movements GROUP BY sku ORDER BY sku"}),
            ]
        elif family == "sales_summary":
            self.plan += [
                ("read_csv", {"path": "data/orders.csv"}),
                ("inspect_database", {"path": "data/customers.sqlite"}),
                ("query_sql", {"path": "data/customers.sqlite", "query": "SELECT customer_id, region FROM customers ORDER BY customer_id"}),
            ]
        else:
            raise ValueError(f"Unknown family: {family}")

    def complete(self, messages: list, tools: list, timeout: float) -> dict:
        if self.turn < len(self.plan):
            name, arguments = self.plan[self.turn]
        elif not self.wrote_result:
            result = self._solve(messages)
            if self.inject_failure:
                # A plausible, well-formed but wrong answer tests the verifier.
                first_list = next(iter(result.values()))
                if first_list:
                    first_row = first_list[0]
                    numeric_key = next(key for key, value in first_row.items() if type(value) is int)
                    first_row[numeric_key] += 1
                else:
                    result["unexpected_field"] = True
            name, arguments = "write_result", {"result": result}
            self.wrote_result = True
        else:
            return {"role": "assistant", "content": "Результат сохранён в output/result.json."}
        self.turn += 1
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": f"mock_call_{self.turn:03d}", "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments, ensure_ascii=False, allow_nan=False),
            }}],
        }

    def _solve(self, messages: list) -> dict:
        calls = {}
        observations = []
        for message in messages:
            for call in message.get("tool_calls", []):
                calls[call["id"]] = (call["function"]["name"], json.loads(call["function"]["arguments"]))
            if message["role"] == "tool":
                name, arguments = calls[message["tool_call_id"]]
                value = json.loads(message["content"])
                if "error" in value:
                    raise ValueError("Mock observed a tool failure")
                observations.append((name, arguments, value))
        csvs = {args["path"]: value["rows"] for name, args, value in observations if name == "read_csv"}
        queries = [value["rows"] for name, _, value in observations if name == "query_sql"]
        if self.family == "table_reconciliation":
            expected = {r["invoice_id"]: int(r["amount_cents"]) for r in csvs["data/ledger.csv"]}
            actual = {r["invoice_id"]: int(r["paid_cents"]) for r in csvs["data/payments.csv"]}
            result = []
            for invoice_id in sorted(expected.keys() | actual.keys()):
                exp, act = expected.get(invoice_id, 0), actual.get(invoice_id, 0)
                if invoice_id not in actual:
                    issue = "missing_payment"
                elif invoice_id not in expected:
                    issue = "unexpected_payment"
                elif exp != act:
                    issue = "amount_mismatch"
                else:
                    continue
                result.append(dict(invoice_id=invoice_id, expected_cents=exp, actual_cents=act,
                                   delta_cents=act - exp, issue=issue))
            return {"discrepancies": result}
        if self.family == "inventory_discrepancies":
            opening = {r["sku"]: r["opening_qty"] for r in queries[0]}
            changes = {r["sku"]: r["delta_qty"] for r in queries[1]}
            result = []
            for row in sorted(csvs["data/snapshot.csv"], key=lambda row: row["sku"]):
                sku, recorded = row["sku"], int(row["recorded_qty"])
                expected = opening[sku] + changes.get(sku, 0)
                if recorded != expected:
                    result.append(dict(sku=sku, expected_qty=expected, recorded_qty=recorded,
                                       delta_qty=recorded - expected))
            return {"discrepancies": result}
        customers = {row["customer_id"]: row["region"] for row in queries[0]}
        totals = defaultdict(lambda: [0, 0])
        for region in customers.values():
            totals[region]
        for row in csvs["data/orders.csv"]:
            if row["status"] == "paid":
                agg = totals[customers[row["customer_id"]]]
                agg[0] += 1
                agg[1] += int(row["amount_cents"])
        return {"summary": [dict(region=region, paid_order_count=count, revenue_cents=cents)
                            for region, (count, cents) in sorted(totals.items())]}
