"""Trusted allowlist routing; router is NOT an authorization authority."""
from __future__ import annotations

from contracts import Request
from corpus import SOURCE_TYPES


class SourceRouter:
    def route(self, intent: str, aspect: str, request: Request) -> tuple[str, ...]:
        if aspect == "calculation":
            return ()
        if intent == "customer_revenue":
            choices = ("sql",)
        elif intent == "authorization":
            choices = ("code",)
        elif intent == "technical_serving":
            choices = ("docs", "code") if "config" in request.question.lower() else ("docs",)
        else:
            choices = ("docs",)
        if not request.allowed_sources or any(
            s not in SOURCE_TYPES for s in request.allowed_sources
        ):
            raise ValueError("Trusted source allowlist invalid")
        if any(source not in request.allowed_sources for source in choices):
            raise PermissionError(
                "The intended source is not authorized; never widen access on fallback"
            )
        return choices
