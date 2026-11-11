"""A tiny read-only business dataset and grounded tool contracts.

This module is used by the agent runtime, but never imports evaluation cases.
"""

from __future__ import annotations

import json
from typing import Any

from contracts import ToolCall

CUSTOMERS = {
    1: {"id": 1, "name": "Lan"},
    2: {"id": 2, "name": "Minh"},
    3: {"id": 3, "name": "Nhi"},
}
ORDERS = (
    {"id": 101, "customer": "Lan", "status": "paid", "amount": 120000},
    {"id": 102, "customer": "Lan", "status": "paid", "amount": 80000},
    {"id": 103, "customer": "Lan", "status": "pending", "amount": 50000},
    {"id": 104, "customer": "Minh", "status": "paid", "amount": 45000},
    {"id": 105, "customer": "Nhi", "status": "pending", "amount": 90000},
)


def _schema(name: str, description: str, props: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": list(props),
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS = [
    _schema(
        "get_customer",
        "Resolve a customer by customer_id",
        {
            "customer_id": {"type": "integer", "minimum": 1},
        },
    ),
    _schema(
        "get_orders",
        "Get paid orders for a customer name returned by get_customer",
        {
            "customer": {"type": "string"},
            "status": {"type": "string", "enum": ["paid"]},
        },
    ),
    _schema(
        "sum_amounts",
        "Sum paid order amounts from get_orders",
        {
            "amounts": {"type": "array", "items": {"type": "integer", "minimum": 0}},
        },
    ),
    _schema(
        "count_orders",
        "Count paid order IDs from get_orders",
        {
            "order_ids": {"type": "array", "items": {"type": "integer", "minimum": 1}},
        },
    ),
]


class ToolRegistry:
    """Reject invented tools, mismatched IDs, ungrounded args and duplicates."""

    @staticmethod
    def schemas() -> list[dict[str, Any]]:
        return TOOL_SCHEMAS

    def execute(
        self, call: ToolCall, evidence: dict[str, dict[str, Any]], target_id: int | None
    ) -> dict[str, Any]:
        if call.name not in {
            "get_customer",
            "get_orders",
            "sum_amounts",
            "count_orders",
        }:
            return {"ok": False, "code": "UNKNOWN_TOOL"}
        if call.name in evidence:
            return {"ok": False, "code": "REPEATED_TOOL"}
        try:
            arguments = json.loads(call.arguments)
        except (ValueError, TypeError):
            return {"ok": False, "code": "INVALID_ARGUMENTS"}
        if not isinstance(arguments, dict):
            return {"ok": False, "code": "INVALID_ARGUMENTS"}

        if call.name == "get_customer":
            if (
                set(arguments) != {"customer_id"}
                or type(arguments["customer_id"]) is not int
                or arguments["customer_id"] < 1
            ):
                return {"ok": False, "code": "INVALID_ARGUMENTS"}
            if arguments["customer_id"] != target_id:
                return {"ok": False, "code": "DATA_MISMATCH"}
            customer = CUSTOMERS.get(target_id)
            return {
                "ok": True,
                "data": {"found": customer is not None, "customer": customer},
            }

        previous = evidence.get("get_customer", {}).get("customer")
        if previous is None:
            return {"ok": False, "code": "PREREQUISITE_MISSING"}
        if call.name == "get_orders":
            if (
                set(arguments) != {"customer", "status"}
                or arguments["customer"] != previous["name"]
                or arguments["status"] != "paid"
            ):
                return {"ok": False, "code": "INVALID_ARGUMENTS"}
            paid = [
                dict(order)
                for order in ORDERS
                if order["customer"] == previous["name"] and order["status"] == "paid"
            ]
            return {"ok": True, "data": {"orders": paid}}

        paid = evidence.get("get_orders", {}).get("orders")
        if paid is None:
            return {"ok": False, "code": "PREREQUISITE_MISSING"}

        if call.name == "sum_amounts":
            actual = [order["amount"] for order in paid]
            if (
                set(arguments) != {"amounts"}
                or not isinstance(arguments["amounts"], list)
                or any(type(v) is not int or v < 0 for v in arguments["amounts"])
                or arguments["amounts"] != actual
            ):
                return {"ok": False, "code": "INVALID_ARGUMENTS"}
            return {"ok": True, "data": {"total": sum(actual)}}

        ids = [order["id"] for order in paid]
        if (
            set(arguments) != {"order_ids"}
            or not isinstance(arguments["order_ids"], list)
            or any(type(v) is not int or v < 1 for v in arguments["order_ids"])
            or arguments["order_ids"] != ids
        ):
            return {"ok": False, "code": "INVALID_ARGUMENTS"}
        return {"ok": True, "data": {"count": len(ids)}}
