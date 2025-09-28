#!/usr/bin/env python3
"""Unit tests for the unified QuerySignature and EvaluationModule."""

import unittest
import dspy
from bbeval.signatures import QuerySignature, EvaluationModule


class TestQuerySignature(unittest.TestCase):
    def test_query_signature_fields(self):
        self.assertTrue(issubclass(QuerySignature, dspy.Signature))
        for field in ['request', 'guidelines', 'outcome']:
            self.assertIn(field, QuerySignature.__annotations__)
        self.assertIn('answer', QuerySignature.__annotations__)


class TestEvaluationModule(unittest.TestCase):
    def test_evaluation_module_creation_with_query_signature(self):
        module = EvaluationModule(QuerySignature)
        self.assertIsInstance(module, dspy.Module)
        self.assertIsInstance(module.predictor, dspy.Predict)

    def test_evaluation_module_forward_signature(self):
        module = EvaluationModule(QuerySignature)
        self.assertTrue(hasattr(module, 'forward'))
        import inspect
        sig = inspect.signature(module.forward)
        self.assertIn('test_case_id', sig.parameters)
        self.assertIn('kwargs', sig.parameters)


if __name__ == '__main__':
    unittest.main()