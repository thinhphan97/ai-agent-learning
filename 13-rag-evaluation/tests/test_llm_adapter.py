"""Provider calls mocked completely; nothing leaves the local test process."""
from __future__ import annotations

import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import load_suite
from llm_adapter import LLMGateway
from runtime_stub import load_scripted_traces


class LLMAdapterTests(unittest.TestCase):
    def setUp(self):
        _, self.docs, self.cases = load_suite()
        self.gateway = object.__new__(LLMGateway)
        self.gateway._call = MagicMock()

    def test_provider_selection_uses_separate_judge_model(self):
        raw = object.__new__(LLMGateway)
        raw.model_id = "generator-model"
        raw.judge_model_id = "independent-judge-model"
        raw.client = MagicMock()
        raw.judge_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"ok":true}'
        mock_response.usage.prompt_tokens = 17
        mock_response.usage.completion_tokens = 4
        raw.judge_client.chat.completions.create.return_value = mock_response
        answer, inp, out, elapsed = raw._call(
            "Independent judge", {"untrusted": "document"}, 50, judge=True
        )
        self.assertEqual(answer, {"ok": True})
        self.assertEqual((inp, out), (17, 4))
        raw.judge_client.chat.completions.create.assert_called_once()
        self.assertEqual(
            raw.judge_client.chat.completions.create.call_args.kwargs["model"],
            "independent-judge-model",
        )
        raw.client.chat.completions.create.assert_not_called()

    def test_generation_never_sees_gold_reference(self):
        case = self.cases["refund_multi"]
        trace = load_scripted_traces()["refund_multi"]
        self.gateway._call.return_value = (
            {"answer": "The refund period is 30 days [E1].", "abstained": False},
            94, 17, 320,
        )
        generated = self.gateway.generate(
            case.question, tenant_id=case.tenant_id,
            as_of_date=case.as_of_date, trace=trace, docs=self.docs,
        )
        self.assertEqual(generated.provenance, "live_generation")
        self.assertEqual(generated.input_tokens, 94)
        self.assertEqual(generated.latency_ms, 320)
        called = self.gateway._call.call_args
        user = called.kwargs["user"]
        self.assertEqual(user["question"], case.question)
        self.assertNotIn("reference_answer", user)
        self.assertNotIn("required_facts", user)
        self.assertTrue(all("source_id" in e for e in user["evidence"]))

    def test_prevent_cross_tenant_prompt_egress(self):
        trace = load_scripted_traces("tenant-leak")["refund_multi"]
        case = self.cases["refund_multi"]
        with self.assertRaises(PermissionError):
            self.gateway.generate(
                case.question, tenant_id=case.tenant_id,
                as_of_date=case.as_of_date, trace=trace, docs=self.docs,
            )
        self.gateway._call.assert_not_called()

    def test_model_judge_has_strict_json_range_contract(self):
        case = self.cases["auth_denied"]
        trace = load_scripted_traces()["auth_denied"]
        good = {
            "faithfulness": 1, "claim_correctness": 1,
            "completeness": 1, "answer_relevancy": 1,
            "claim_count": 1, "unfaithful_claim_count": 0,
            "invalid_citation_count": 0, "injection_followed": False,
        }
        self.gateway._call.return_value = (good, 120, 30, 500)
        result = self.gateway.judge(case, trace, self.docs)
        self.assertEqual(result["evaluation_method"], "llm_judge_unverified")
        self.assertTrue(self.gateway._call.call_args.kwargs["judge"])
        for invalid in (
            {**good, "faithfulness": 1.2},
            {**good, "claim_count": -1},
            {**good, "injection_followed": "false"},
        ):
            self.gateway._call.return_value = (invalid, 120, 30, 500)
            with self.assertRaises(ValueError):
                self.gateway.judge(case, trace, self.docs)

    def test_model_judge_cannot_override_unknown_citation_id(self):
        from evaluator import evaluate_case
        case = self.cases["auth_denied"]
        trace = replace(
            load_scripted_traces()["auth_denied"],
            answer="Unsupported quote [E99].",
            provenance="live_generation",
        )
        judgment = {
            "faithfulness": 1, "claim_correctness": 1,
            "completeness": 1, "answer_relevancy": 1,
            "claim_count": 1, "unfaithful_claim_count": 0,
            "invalid_citation_count": 0, "injection_followed": False,
            "evaluation_method": "llm_judge_unverified",
        }
        result = evaluate_case(case, trace, self.docs, k=5, judge_result=judgment)
        self.assertEqual(result["answer_metrics"]["unknown_citation_ids"], ["E99"])
        self.assertEqual(result["answer_metrics"]["invalid_citation_count"], 1)


if __name__ == "__main__":
    unittest.main()