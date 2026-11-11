# Level 1 — Multi-Step Tool Calling

**Prerequisites:** Complete [Level 0](../0-agent-loop-react/README.md).
**Outcome:** Pass observed data across three dependent tools with precise ordering checks.

## Scenario and required trajectory

```text
User: Calculate paid order revenue for customer_id=1.
Model 1 → get_customer(customer_id=1)
Tool    → {found: true, customer: {id: 1, name: "Lan"}}
Model 2 → get_orders(customer="Lan", status="paid")
Tool    → orders with amounts [120000, 80000]
Model 3 → sum_amounts(amounts=[120000,80000])
Tool    → {total: 200000}
Model 4 → "Paid revenue for Lan: 200,000 VND"
```

The customer name and order amounts are read from observations, not copied into downstream decisions. The mock records 3 tools and 4 model calls.

## Architecture

Every lesson is independent, so `agent.py` includes its own runtime, registry and provider adapter; it does not import level 0.

- `get_customer`: resolves a numeric ID to a customer; checks input.
- `get_orders`: reads only paid orders for resolved customer.
- `sum_amounts`: sums nonnegative integers; validates input.
- `ToolRegistry`: single execution boundary; unknown tools return structured errors.
- `AgentRunner`: records trace and matches tool results to call IDs.
- `required_order`: optional teaching policy rejects skipping, reordering and batching.
- `MockModel`: turns observations into subsequent calls; independent of live API.

## Run and inspect

Before running live mode, install shared dependencies once from the root: `python -m pip install -r requirements.txt`. Mock mode uses the standard library.

```bash
python agent.py
python -m unittest discover -s tests -v
LIVE_MODE=1 python agent.py  # requires root .env + model tool calling
```

Expected trace: `get_customer → get_orders → sum_amounts`, then `completed`.

## Exercise checklist

- [x] Add `get_customer(customer_id)`.
- [x] Resolve customer before reading orders.
- [x] Verify exact order with tests.
- [x] Verify observation-driven argument passing.
- [x] Verify final answer and 3-tool/4-model accounting.
- [x] Reject out-of-order call before tool execution.

## Challenges

1. Try customer ID 2 and confirm the answer is based on Minh's orders.
2. Create a model that tries `get_orders` first; ensure it does not execute.
3. Try a model that returns a final answer early; why should the policy reject it?
4. Extend the evaluator to verify the numeric result rather than merely trust the model's final message.

## Deliberate limits

This level enforces a specific three-step workflow for teaching. General ReAct agents should not always enforce one fixed order. Unit tests exercise the mock runtime, not the intelligence or actual quality of any remote model. Runtime tool schemas are advertised to the model but are not automatically validated by a JSON Schema library.
