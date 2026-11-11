"""Lab 7 CLI: deterministic attack demonstrations or real LLM tool-calling.

From workspace root:
  python 7-agent-safety/agent.py --scenario prompt-injection
  python 7-agent-safety/agent.py --question "Set note for customer_id=1 to 'Call Friday'."
  python 7-agent-safety/agent.py --question "Set note for customer_id=1 to 'Call Friday'." --approve
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from contracts import ApprovalGrant, Limits
from model import MockModel, openai_compatible_model
from policy import SafetyPolicy, parse_intent
from runtime import SafeAgentRunner
from tools import CustomerStore, ToolRegistry

DEFAULT_REQUEST = "Show customer_id=1."


def main() -> int:
    parser = argparse.ArgumentParser(description="Lab 7: Agent Safety & Guardrails")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--mock", action="store_true", help="Force offline model")
    mode.add_argument("--live", action="store_true", help="Use real root .env LLM")
    parser.add_argument(
        "--scenario", choices=MockModel.SCENARIOS, help="Force mock scenario"
    )
    parser.add_argument("--question", default=DEFAULT_REQUEST)
    parser.add_argument(
        "--principal",
        default="alice",
        choices=["alice", "bob"],
        help="DEMO namespace, not authentication",
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Demo approval for exact requested note write, not general tool access",
    )
    parser.add_argument(
        "--inject-untrusted-note",
        action="store_true",
        help="Include malicious text in customer note (no external effects)",
    )
    parser.add_argument(
        "--audit-file",
        type=Path,
        help="Write redacted JSON decision log into chosen path",
    )
    parser.add_argument("--max-model-calls", type=int, default=5)
    parser.add_argument("--max-tool-attempts", type=int, default=3)
    args = parser.parse_args()
    if args.live and args.scenario:
        parser.error("--live and --scenario cannot be combined")
    try:
        intent = parse_intent(args.question)
        limits = Limits(args.max_model_calls, args.max_tool_attempts)
    except ValueError as exc:
        parser.error(str(exc))
    if args.approve and intent.kind != "write":
        parser.error("--approve is valid only for explicit note-write requests")

    live = args.live or (
        not args.mock and args.scenario is None and os.getenv("LIVE_MODE") == "1"
    )
    model = (
        openai_compatible_model()
        if live
        else MockModel(intent, args.scenario or "normal")
    )

    approval = (
        ApprovalGrant(args.principal, intent.customer_id, intent.note)
        if args.approve
        else None
    )
    policy = SafetyPolicy(args.principal, intent, approval)
    store = CustomerStore(
        malicious_note=args.inject_untrusted_note or args.scenario == "prompt-injection"
    )
    registry = ToolRegistry(store, policy)
    outcome = SafeAgentRunner(model, registry, policy, limits=limits).run(args.question)
    print(
        f"Mode: {'live' if live else 'mock'}; "
        f"scenario: {args.scenario or 'normal'}; "
        f"principal: {args.principal}; intent: {intent.kind}"
    )
    for event in outcome.events:
        if event["event"] in {
            "tool_approved",
            "tool_denied",
            "scope_denied",
            "final_verified",
        }:
            print(f"{event['event']}: {event['tool'] or '-'} -> {event['code']}")
    print(f"Status: {outcome.status}; code: {outcome.code}")
    print(
        f"Model calls: {outcome.model_calls}; "
        f"tool attempts: {outcome.tool_attempts}; "
        f"simulated writes: {store.write_count}"
    )
    if outcome.answer is not None:
        print("Answer:", outcome.answer)

    if args.audit_file:
        from audit import AuditTrail

        audit = AuditTrail()
        audit.events = outcome.events
        audit.export(
            args.audit_file,
            {
                "status": outcome.status,
                "code": outcome.code,
                "model_calls": outcome.model_calls,
                "tool_attempts": outcome.tool_attempts,
                "simulated_writes": store.write_count,
            },
        )
        print("Audit:", args.audit_file)

    return 0 if outcome.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
