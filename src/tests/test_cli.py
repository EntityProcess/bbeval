#!/usr/bin/env python3
"""
Unit tests for bbeval.cli module.

Tests the CLI helper functions for improved modularity.
"""

import unittest
from unittest.mock import Mock, patch

from bbeval import EvaluationResult
from bbeval.models import AgentTimeoutError
from bbeval.cli import _run_test_case_with_retries


class TestRunTestCaseWithRetries(unittest.TestCase):
    """Test cases for the _run_test_case_with_retries helper function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_case = Mock()
        self.test_case.id = "test_case_123"
        self.test_case.guideline_paths = []
        # Add missing attributes that the new code path expects
        self.test_case.outcome = "Test outcome for code review"
        self.test_case.task = "Review the following code for best practices"
        self.test_case.expected_assistant_raw = "Expected code output"
        
        self.evaluation_module = Mock()
        self.evaluation_module.return_value = Mock(answer="Mock review response")
        
        # Mock target and targets
        self.target = {
            'name': 'test_target',
            'provider': 'test',
            'settings': {}
        }
        self.targets = [self.target]
        
    @patch('bbeval.cli.determine_signature_from_test_case')
    @patch('bbeval.cli.build_prompt_inputs')
    @patch('bbeval.cli.evaluate_test_case')
    @patch('bbeval.cli.write_result_line')
    def test_successful_execution(self, mock_write, mock_evaluate, mock_build_prompt, mock_determine_sig):
        """Test successful test case execution without retries."""
        # Setup mocks
        from bbeval.signatures import CodeReview  # Import a non-CodeGeneration signature
        mock_determine_sig.return_value = CodeReview  # Ensure we use heuristic scorer
        mock_build_prompt.return_value = {"prompt": "test prompt"}
        mock_result = EvaluationResult(
            test_id="test_case_123",
            score=0.8,
            hits=["hit1"],
            misses=["miss1"],
            model_answer="test answer",
            expected_aspect_count=2,
            provider="test",
            model="test-model",
            timestamp="",
            raw_aspects=[]
        )
        mock_evaluate.return_value = mock_result
        
        result = _run_test_case_with_retries(
            test_case=self.test_case,
            evaluation_module=self.evaluation_module,
            repo_root="/test/repo",
            provider="test",
            settings={},
            model="test-model",
            output_file=None,
            dry_run=False,
            verbose=False,
            max_retries=2,
            target=self.target,
            targets=self.targets
        )
        
        self.assertEqual(result, mock_result)
        mock_build_prompt.assert_called_once()
        mock_evaluate.assert_called_once()
        mock_write.assert_not_called()  # No output file specified
    
    @patch('bbeval.cli.determine_signature_from_test_case')
    @patch('bbeval.cli.build_prompt_inputs')
    @patch('bbeval.cli.evaluate_test_case')
    def test_agent_timeout_with_retry(self, mock_evaluate, mock_build_prompt, mock_determine_sig):
        """Test agent timeout handling with successful retry."""
        # Setup mocks - first call raises timeout, second succeeds
        from bbeval.signatures import CodeReview  # Import a non-CodeGeneration signature
        mock_determine_sig.return_value = CodeReview  # Ensure we use heuristic scorer
        mock_build_prompt.return_value = {"prompt": "test prompt"}
        mock_result = EvaluationResult(
            test_id="test_case_123",
            score=0.8,
            hits=["hit1"],
            misses=[],
            model_answer="test answer",
            expected_aspect_count=1,
            provider="test",
            model="test-model",
            timestamp="",
            raw_aspects=[]
        )
        
        self.evaluation_module.side_effect = [AgentTimeoutError("Timeout"), Mock(answer="Success")]
        mock_evaluate.return_value = mock_result
        
        result = _run_test_case_with_retries(
            test_case=self.test_case,
            evaluation_module=self.evaluation_module,
            repo_root="/test/repo",
            provider="test",
            settings={},
            model="test-model",
            output_file=None,
            dry_run=False,
            verbose=False,
            max_retries=2,
            target=self.target,
            targets=self.targets
        )
        
        self.assertEqual(result, mock_result)
        self.assertEqual(self.evaluation_module.call_count, 2)
    
    @patch('bbeval.cli.build_prompt_inputs')
    def test_max_retries_exceeded(self, mock_build_prompt):
        """Test handling when max retries are exceeded."""
        mock_build_prompt.return_value = {"prompt": "test prompt"}
        self.evaluation_module.side_effect = AgentTimeoutError("Persistent timeout")
        
        result = _run_test_case_with_retries(
            test_case=self.test_case,
            evaluation_module=self.evaluation_module,
            repo_root="/test/repo",
            provider="test",
            settings={},
            model="test-model",
            output_file=None,
            dry_run=False,
            verbose=False,
            max_retries=1,
            target=self.target,
            targets=self.targets
        )
        
        self.assertEqual(result.test_id, "test_case_123")
        self.assertEqual(result.score, 0.0)
        self.assertIn("Agent timeout after 1 retries", result.misses[0])
        self.assertEqual(self.evaluation_module.call_count, 2)  # Initial + 1 retry
    
    @patch('bbeval.cli.dspy.Predict')
    @patch('bbeval.cli.determine_signature_from_test_case')
    @patch('bbeval.cli.build_prompt_inputs')
    @patch('bbeval.cli.write_result_line')
    def test_code_generation_llm_judge(self, mock_write, mock_build_prompt, mock_determine_sig, mock_dspy_predict):
        """Test LLM judge path for CodeGeneration test cases."""
        from bbeval.signatures import CodeGeneration
        
        # Setup mocks
        mock_determine_sig.return_value = CodeGeneration
        mock_build_prompt.return_value = {"prompt": "test prompt"}
        
        # Mock the LLM judge
        mock_judge_instance = Mock()
        mock_judge_instance.return_value = Mock(score="0.85", reasoning="Well implemented code")
        mock_dspy_predict.return_value = mock_judge_instance
        
        result = _run_test_case_with_retries(
            test_case=self.test_case,
            evaluation_module=self.evaluation_module,
            repo_root="/test/repo",
            provider="test",
            settings={},
            model="test-model",
            output_file=None,
            dry_run=False,
            verbose=False,
            max_retries=2,
            target=self.target,
            targets=self.targets
        )
        
        # Verify LLM judge was used
        mock_dspy_predict.assert_called_once()
        mock_judge_instance.assert_called_once_with(
            outcome=self.test_case.outcome,
            task_requirements=self.test_case.task,
            reference_code=self.test_case.expected_assistant_raw,
            generated_code="Mock review response"
        )
        
        # Verify result structure
        self.assertEqual(result.test_id, "test_case_123")
        self.assertEqual(result.score, 0.85)
        self.assertEqual(result.hits, ["Well implemented code"])
        self.assertEqual(result.misses, [])
        self.assertEqual(result.expected_aspect_count, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)