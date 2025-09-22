#!/usr/bin/env python3
"""
Unit tests for speceval.cli module.

Tests the CLI helper functions for improved modularity.
"""

import unittest
from unittest.mock import Mock, patch

from speceval import EvaluationResult
from speceval.models import AgentTimeoutError
from speceval.cli import _run_test_case_with_retries


class TestRunTestCaseWithRetries(unittest.TestCase):
    """Test cases for the _run_test_case_with_retries helper function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_case = Mock()
        self.test_case.id = "test_case_123"
        self.test_case.guideline_paths = []
        
        self.evaluation_module = Mock()
        self.evaluation_module.return_value = Mock(review="Mock review response")
        
    @patch('speceval.cli.focus_vscode_workspace')
    @patch('speceval.cli.build_prompt_inputs')
    @patch('speceval.cli.evaluate_test_case')
    @patch('speceval.cli.write_result_line')
    def test_successful_execution(self, mock_write, mock_evaluate, mock_build_prompt, mock_focus):
        """Test successful test case execution without retries."""
        # Setup mocks
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
            max_retries=2
        )
        
        self.assertEqual(result, mock_result)
        mock_focus.assert_called_once()
        mock_build_prompt.assert_called_once()
        mock_evaluate.assert_called_once()
        mock_write.assert_not_called()  # No output file specified
    
    @patch('speceval.cli.focus_vscode_workspace')
    @patch('speceval.cli.build_prompt_inputs')
    @patch('speceval.cli.evaluate_test_case')
    def test_agent_timeout_with_retry(self, mock_evaluate, mock_build_prompt, mock_focus):
        """Test agent timeout handling with successful retry."""
        # Setup mocks - first call raises timeout, second succeeds
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
        
        self.evaluation_module.side_effect = [AgentTimeoutError("Timeout"), Mock(review="Success")]
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
            max_retries=2
        )
        
        self.assertEqual(result, mock_result)
        self.assertEqual(self.evaluation_module.call_count, 2)
    
    @patch('speceval.cli.focus_vscode_workspace')
    @patch('speceval.cli.build_prompt_inputs')
    def test_max_retries_exceeded(self, mock_build_prompt, mock_focus):
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
            max_retries=1
        )
        
        self.assertEqual(result.test_id, "test_case_123")
        self.assertEqual(result.score, 0.0)
        self.assertIn("Agent timeout after 1 retries", result.misses[0])
        self.assertEqual(self.evaluation_module.call_count, 2)  # Initial + 1 retry


if __name__ == '__main__':
    unittest.main(verbosity=2)