"""Minimal agent runtime under test.

Important: a syntactically valid final JSON can mark an episode 'completed'
without being correct or grounded. The INDEPENDENT evaluator catches that.
"""

from __future__ import annotations

import json
import time
from typing import Any

from contracts import Answer, Episode, Limits
from model import Model, SYSTEM_PROMPT, request_kind
from tools import ToolRegistry


class AgentRunner:
    def __init__(
        self,
        model: Model,
        registry: ToolRegistry | None = None,
        *,
        limits: Limits | None = None,
    ):
        self.model = model
        self.registry = registry or ToolRegistry()
        self.limits = limits or Limits()

    def run(self, question: str) -> Episode:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be nonempty")
        if len(question) > 1000:
            raise ValueError("Question exceeds 1000 characters")

        episode = Episode(question=question, status="running")
        started = time.perf_counter()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        used_call_ids: set[str] = set()
        _kind, target_id = request_kind(question)

        def stop(status: str, error: str | None = None) -> Episode:
            episode.status = status
            episode.error_code = error
            episode.elapsed_ms = (time.perf_counter() - started) * 1000
            episode.log("run_stopped", status=status, code=error)
            return episode

        for _ in range(self.limits.max_model_calls):
            episode.model_calls += 1
            episode.log("model_requested", attempt=episode.model_calls)
            try:
                turn = self.model.complete(messages, self.registry.schemas())
            except Exception:
                # Do not include provider exception messages in reports.
                return stop("provider_error", "MODEL_REQUEST_FAILED")

            if turn.tool_calls:
                if len(turn.tool_calls) != 1:
                    return stop("protocol_error", "MULTIPLE_TOOL_CALLS")
                call = turn.tool_calls[0]
                if not call.id or call.id in used_call_ids:
                    return stop("protocol_error", "INVALID_TOOL_CALL_ID")
                if episode.tool_attempts >= self.limits.max_tool_attempts:
                    return stop("budget_exceeded", "TOOL_ATTEMPT_LIMIT")
                used_call_ids.add(call.id)
                episode.tool_attempts += 1
                observed = self.registry.execute(call, episode.evidence, target_id)
                episode.log(
                    "tool_observation",
                    tool=call.name,
                    call_id=call.id,
                    observation=observed,
                )
                if not observed["ok"]:
                    return stop("tool_error", observed["code"])

                episode.evidence[call.name] = observed["data"]
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": turn.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments,
                            },
                        }
                    ],
                }
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": json.dumps(observed, ensure_ascii=False),
                }
                messages.extend((assistant_message, tool_message))
                continue

            try:
                episode.answer = Answer.parse(turn.content)
            except ValueError:
                return stop("protocol_error", "INVALID_FINAL_ANSWER")
            episode.log("final_answer", answer=episode.answer.as_dict())
            return stop("completed")

        return stop("budget_exceeded", "MODEL_CALL_LIMIT")
