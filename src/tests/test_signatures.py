#!/usr/bin/env python3
"""Unit tests for the unified EvalSignature and EvaluationModule."""

import unittest
import dspy
from bbeval.signatures import EvalSignature, EvaluationModule


class TestEvalSignature(unittest.TestCase):
    def test_eval_signature_fields(self):
        self.assertTrue(issubclass(EvalSignature, dspy.Signature))
        for field in ['request', 'guidelines', 'outcome']:
            self.assertIn(field, EvalSignature.__annotations__)
        self.assertIn('answer', EvalSignature.__annotations__)


class TestEvaluationModule(unittest.TestCase):
    def test_evaluation_module_creation_with_eval_signature(self):
        module = EvaluationModule(EvalSignature)
        self.assertIsInstance(module, dspy.Module)
        self.assertIsInstance(module.predictor, dspy.Predict)

    def test_evaluation_module_forward_signature(self):
        module = EvaluationModule(EvalSignature)
        self.assertTrue(hasattr(module, 'forward'))
        import inspect
        sig = inspect.signature(module.forward)
        self.assertIn('test_case_id', sig.parameters)
        self.assertIn('kwargs', sig.parameters)


if __name__ == '__main__':
    unittest.main()