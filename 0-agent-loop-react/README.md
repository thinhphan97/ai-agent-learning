# Level 0 — Agent Loop & ReAct

**Prerequisites:** Python fundamentals, functions, JSON, HTTP APIs (optional).
**Outcome:** Understand the feedback loop: model decision → tool execution → observation → next model decision → final answer.

## Key ideas

- The LLM suggests structured tool calls. The runtime owns execution, validation, limits and stopping.
- ReAct alternates action selection with observations. A real model need not expose private chain-of-thought.
- One tool per turn in this teaching mock; general models can request several tools.
- Tool results are passed back using the original `tool_call_id`.
- Completing a loop is **not** proof of factual correctness.

## Scenario

User asks: “What is Lan's paid order revenue?”

1. `get_orders(customer="Lan", status="paid")` returns 120000 and 80000.
2. `sum_amounts(amounts=[120000,80000])` returns 200000.
3. The model sends the final answer.

Orders still marked pending must be excluded.

## Code walk-through

Read `agent.py` in this order:

1. `ToolCall` and `ModelTurn`: model output contract.
2. `Tool` and `ToolRegistry`: capabilities and execution boundary.
3. `AgentRunner.run`: budgeted loop, message history, tool call correlation and terminal states.
4. `MockModel`: deterministic policy for learning, **not actual LLM reasoning**.
5. `openai_compatible_model`: real provider adapter using root `.env`.

## Run

```bash
python agent.py
python -m unittest discover -s tests -v
LIVE_MODE=1 python agent.py   # optional: needs configured .env and tool-capable LLM
```

Install shared dependencies once from the workspace root with `python -m pip install -r requirements.txt` before live execution. Mock mode uses the standard library.

## Questions to answer

1. Why do two tool calls require three model turns in this mock?
2. Why is tool output inserted into message history?
3. How do `budget_exceeded`, `protocol_error` and `completed` differ?
4. What would need to change to make external tool execution secure?

## Limitations

This is a teaching runtime. No JSON Schema enforcement by registry, async execution, cancellation, durable checkpoints, production authorization, token accounting, or final-result verifier. Handlers independently validate argument types. No claim of reliability for arbitrary tools or LLMs.

**Reference:** https://arxiv.org/abs/2210.03629
