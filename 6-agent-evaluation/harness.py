"""Evaluation harness and aggregate metrics.

The evaluation labels are held by this module and the scorer, not the agent.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from cases import DATASET_VERSION, EvalCase
from contracts import Episode, Limits
from evaluator import CaseScore, Evaluator
from model import Model
from runtime import AgentRunner


@dataclass
class CaseRun:
    score: CaseScore
    episode: Episode


class SuiteRunner:
    def __init__(
        self, model_factory: Callable[[], Model], *, limits: Limits | None = None
    ):
        self.model_factory = model_factory
        self.limits = limits or Limits()

    def run(self, cases: Sequence[EvalCase]) -> tuple[CaseRun, ...]:
        if not cases:
            raise ValueError("Evaluation suite cannot be empty")
        if len({case.id for case in cases}) != len(cases):
            raise ValueError("Evaluation case IDs must be unique")
        evaluator = Evaluator()
        outputs = []
        for case in cases:
            # No oracle fields are passed to AgentRunner, Model, or ToolRegistry.
            episode = AgentRunner(self.model_factory(), limits=self.limits).run(
                case.question
            )
            outputs.append(
                CaseRun(
                    score=evaluator.score(case, episode, self.limits),
                    episode=episode,
                )
            )
        return tuple(outputs)


def aggregate(runs: Sequence[CaseRun], *, mode: str, scenario: str) -> dict[str, Any]:
    if not runs:
        raise ValueError("Cannot aggregate an empty suite")
    scores = [run.score for run in runs]
    n = len(scores)

    def rate(key: str) -> float:
        return round(sum(bool(getattr(score, key)) for score in scores) / n, 4)

    per_category: dict[str, dict[str, Any]] = {}
    for category in sorted({score.category for score in scores}):
        group = [score for score in scores if score.category == category]
        passed = sum(score.task_success for score in group)
        per_category[category] = {
            "cases": len(group),
            "passed": passed,
            "success_rate": round(passed / len(group), 4),
        }

    errors = Counter(reason for score in scores for reason in score.failure_reasons)
    unsupported = [s for s in scores if s.category == "unsupported"]
    missing = [s for s in scores if s.category == "no_data"]

    return {
        "dataset_version": DATASET_VERSION,
        "mode": mode,
        "scenario": scenario,
        "case_ids": [score.id for score in scores],
        "cases": n,
        "passed": sum(s.task_success for s in scores),
        "task_success_rate": rate("task_success"),
        "runtime_completion_rate": rate("completed"),
        "answer_exact_rate": rate("answer_exact"),
        "evidence_grounded_rate": rate("evidence_grounded"),
        "tool_policy_ok_rate": rate("tool_policy_ok"),
        "budget_ok_rate": rate("budget_ok"),
        "abstention_accuracy": (
            round(sum(s.answer_exact for s in unsupported) / len(unsupported), 4)
            if unsupported
            else None
        ),
        "no_data_accuracy": (
            round(sum(s.answer_exact for s in missing) / len(missing), 4)
            if missing
            else None
        ),
        "avg_model_calls": round(sum(s.model_calls for s in scores) / n, 2),
        "avg_tool_attempts": round(sum(s.tool_attempts for s in scores) / n, 2),
        "avg_elapsed_ms": round(sum(s.elapsed_ms for s in scores) / n, 2),
        "by_category": per_category,
        "failure_reasons": dict(sorted(errors.items())),
    }
