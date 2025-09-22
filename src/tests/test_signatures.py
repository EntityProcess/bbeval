#!/usr/bin/env python3
"""
Unit tests for speceval.signatures module.

Tests the signature selection logic and DSPy signature classes.
"""

import unittest

from speceval.signatures import (
    CodeReview, KnowledgeQuery, CodeGeneration,
    determine_signature_from_test_case, EvaluationModule
)
import dspy


class MockTestCase:
    """Mock test case for testing signature selection."""
    
    def __init__(self, outcome: str, task: str):
        self.outcome = outcome
        self.task = task


class TestSignatureClasses(unittest.TestCase):
    """Test cases for the DSPy signature classes."""
    
    def test_code_review_signature_exists(self):
        """Test that CodeReview signature is properly defined."""
        self.assertTrue(issubclass(CodeReview, dspy.Signature))
        
        # Check input fields
        self.assertIn('task', CodeReview.__annotations__)
        self.assertIn('guidelines', CodeReview.__annotations__)
        self.assertIn('code', CodeReview.__annotations__)
        self.assertIn('context', CodeReview.__annotations__)
        
        # Check output field
        self.assertIn('answer', CodeReview.__annotations__)
    
    def test_knowledge_query_signature_exists(self):
        """Test that KnowledgeQuery signature is properly defined."""
        self.assertTrue(issubclass(KnowledgeQuery, dspy.Signature))
        
        # Check input fields
        self.assertIn('task', KnowledgeQuery.__annotations__)
        self.assertIn('guidelines', KnowledgeQuery.__annotations__)
        self.assertIn('code', KnowledgeQuery.__annotations__)
        self.assertIn('context', KnowledgeQuery.__annotations__)
        
        # Check output field
        self.assertIn('answer', KnowledgeQuery.__annotations__)
    
    def test_code_generation_signature_exists(self):
        """Test that CodeGeneration signature is properly defined."""
        self.assertTrue(issubclass(CodeGeneration, dspy.Signature))
        
        # Check input fields
        self.assertIn('task', CodeGeneration.__annotations__)
        self.assertIn('guidelines', CodeGeneration.__annotations__)
        self.assertIn('code', CodeGeneration.__annotations__)
        self.assertIn('context', CodeGeneration.__annotations__)
        
        # Check output field
        self.assertIn('answer', CodeGeneration.__annotations__)


class TestSignatureSelection(unittest.TestCase):
    """Test cases for the signature selection logic."""
    
    def test_knowledge_query_selection(self):
        """Test that knowledge query patterns are correctly identified."""
        test_cases = [
            MockTestCase(
                "Resolves FSA acronym as 'SeaAir'",
                "What does FSA stand for in the context of shipments?"
            ),
            MockTestCase(
                "Explains the definition according to Constants.cs",
                "What is the meaning of FOB in shipping terms?"
            ),
            MockTestCase(
                "Defines the acronym based on codebase",
                "Find the definition of CIF in the context of imports"
            ),
        ]
        
        for test_case in test_cases:
            with self.subTest(task=test_case.task):
                signature = determine_signature_from_test_case(test_case)
                self.assertEqual(signature, KnowledgeQuery)
    
    def test_code_generation_selection(self):
        """Test that code generation patterns are correctly identified."""
        test_cases = [
            MockTestCase(
                "Creates SE UCMP Processor with factory",
                "Please create a SE UCMP Processor and corresponding functionalities"
            ),
            MockTestCase(
                "Implements new service class",
                "Implement a new service for processing messages"
            ),
            MockTestCase(
                "Converts YAML to TypeScript",
                "Convert this YAML form flow to TypeScript:"
            ),
            MockTestCase(
                "Generates code structure",
                "I need a prompt for code review assistance"
            ),
            MockTestCase(
                "Builds new component",
                "Build a new component that handles user authentication"
            ),
        ]
        
        for test_case in test_cases:
            with self.subTest(task=test_case.task):
                signature = determine_signature_from_test_case(test_case)
                self.assertEqual(signature, CodeGeneration)
    
    def test_code_review_selection(self):
        """Test that code review patterns are correctly identified."""
        test_cases = [
            MockTestCase(
                "Flags use of 'exit 1' and recommends 'throw'",
                "Review this snippet for proper termination behavior:"
            ),
            MockTestCase(
                "Identifies absence of Set-StrictMode",
                "Assess initialization safety in this script:"
            ),
            MockTestCase(
                "Notes regex path normalization issues",
                "Evaluate path + file write practices:"
            ),
            MockTestCase(
                "Finds multiple violations",
                "Analyze this code for PowerShell best practices:"
            ),
            MockTestCase(
                "Checks for compliance",
                "Check this function for security vulnerabilities:"
            ),
        ]
        
        for test_case in test_cases:
            with self.subTest(task=test_case.task):
                signature = determine_signature_from_test_case(test_case)
                self.assertEqual(signature, CodeReview)
    
    def test_default_fallback_to_code_review(self):
        """Test that unknown patterns default to CodeReview."""
        test_case = MockTestCase(
            "Some unknown outcome pattern",
            "This is an unknown task that doesn't match any pattern"
        )
        
        signature = determine_signature_from_test_case(test_case)
        self.assertEqual(signature, CodeReview)
    
    def test_multiline_task_uses_first_line(self):
        """Test that multiline tasks use only the first line for analysis."""
        test_case = MockTestCase(
            "Creates new processor implementation",
            "Please create a new message processor.\nThis should handle multiple message types.\nInclude error handling and logging."
        )
        
        signature = determine_signature_from_test_case(test_case)
        self.assertEqual(signature, CodeGeneration)
    
    def test_task_with_code_blocks(self):
        """Test that tasks with code blocks work correctly."""
        test_case = MockTestCase(
            "Converts YAML configuration to TypeScript",
            "Convert this YAML form flow to TypeScript:\n\n```yaml\nPK: 123\nFormFlowId: TestFlow\n```"
        )
        
        signature = determine_signature_from_test_case(test_case)
        self.assertEqual(signature, CodeGeneration)
    
    def test_case_insensitive_matching(self):
        """Test that pattern matching is case insensitive."""
        test_cases = [
            ("CREATE a new service", CodeGeneration),
            ("REVIEW this code", CodeReview), 
            ("WHAT DOES FSA STAND FOR", KnowledgeQuery),
        ]
        
        for task, expected_signature in test_cases:
            with self.subTest(task=task):
                test_case = MockTestCase("Test outcome", task)
                signature = determine_signature_from_test_case(test_case)
                self.assertEqual(signature, expected_signature)


class TestEvaluationModule(unittest.TestCase):
    """Test cases for the EvaluationModule class."""
    
    def test_evaluation_module_creation(self):
        """Test that EvaluationModule can be created with different signatures."""
        signatures = [CodeReview, KnowledgeQuery, CodeGeneration]
        
        for signature_class in signatures:
            with self.subTest(signature=signature_class.__name__):
                module = EvaluationModule(signature_class)
                self.assertIsInstance(module, dspy.Module)
                self.assertIsInstance(module.predictor, dspy.Predict)
    
    def test_evaluation_module_forward_method(self):
        """Test that the forward method exists and accepts parameters."""
        module = EvaluationModule(CodeReview)
        
        # Test that forward method exists
        self.assertTrue(hasattr(module, 'forward'))
        self.assertTrue(callable(module.forward))
        
        # Test method signature accepts test_case_id and kwargs
        import inspect
        sig = inspect.signature(module.forward)
        self.assertIn('test_case_id', sig.parameters)
        self.assertIn('kwargs', sig.parameters)


if __name__ == '__main__':
    unittest.main()