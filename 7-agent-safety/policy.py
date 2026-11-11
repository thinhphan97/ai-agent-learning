"""Deterministic authorization and exact-argument approval.

The model has no authority to create ApprovalGrant or choose principal scope.
This is a toy policy, not authentication.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from contracts import ApprovalGrant, Intent

READ_PATTERN = re.compile(
    r"\s*(?:show|get|read)\s+customer_id=(\d+)\s*\.?\s*\Z", re.IGNORECASE
)
WRITE_PATTERN = re.compile(
    r"\s*set\s+note\s+for\s+customer_id=(\d+)\s+to\s+'([^'\n]{1,80})'\s*\.?\s*\Z",
    re.IGNORECASE,
)
DEMO_OWNERSHIP = {"alice": frozenset({1}), "bob": frozenset({2})}


@dataclass(frozen=True)
class Decision:
    allowed: bool
    code: str


def parse_intent(request: str) -> Intent:
    if not isinstance(request, str) or len(request) > 200:
        raise ValueError("UNSUPPORTED_REQUEST")
    found = READ_PATTERN.fullmatch(request)
    if found and int(found.group(1)) > 0:
        return Intent("read", int(found.group(1)))
    found = WRITE_PATTERN.fullmatch(request)
    if found and int(found.group(1)) > 0 and found.group(2).strip():
        return Intent("write", int(found.group(1)), found.group(2))
    raise ValueError("UNSUPPORTED_REQUEST")


class SafetyPolicy:
    def __init__(
        self,
        principal: str,
        intent: Intent,
        approval: ApprovalGrant | None = None,
    ):
        self.principal = principal
        self.intent = intent
        self.approval = approval

    def check_scope(self) -> Decision:
        if self.intent.customer_id not in DEMO_OWNERSHIP.get(
            self.principal, frozenset()
        ):
            return Decision(False, "SCOPE_DENIED")
        return Decision(True, "ALLOWED")

    def check_write(self, customer_id: int, note: str) -> Decision:
        if self.intent.kind != "write":
            return Decision(False, "READ_ONLY_REQUEST")
        if customer_id != self.intent.customer_id or note != self.intent.note:
            return Decision(False, "ACTION_MISMATCH")
        if self.approval != ApprovalGrant(
            self.principal, self.intent.customer_id, self.intent.note
        ):
            return Decision(False, "APPROVAL_REQUIRED")
        return Decision(True, "ALLOWED")
