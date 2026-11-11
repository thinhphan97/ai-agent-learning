"""Minimal structured audit trail: only decisions and codes, not customer notes.

Use JSON audit export for debugging. This is not an append-only durable log.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class AuditTrail:
    def __init__(self):
        self.events: list[dict[str, Any]] = []

    def record(
        self,
        event: str,
        *,
        code: str = "OK",
        tool: str | None = None,
    ) -> None:
        # Unknown names are model-controlled strings: never log them raw.
        safe_tool = (
            tool
            if tool in {"get_customer", "set_customer_note"}
            else ("unregistered" if tool is not None else None)
        )
        self.events.append(
            {
                "seq": len(self.events) + 1,
                "event": event,
                "code": code,
                "tool": safe_tool,
            }
        )

    def export(self, path: Path, summary: dict[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summary, "events": self.events}
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=path.parent,
                prefix=".audit-",
                delete=False,
            ) as handle:
                tmp = Path(handle.name)
                json.dump(payload, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            if tmp is not None and tmp.exists():
                tmp.unlink()
