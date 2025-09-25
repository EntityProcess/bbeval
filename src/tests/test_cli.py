#!/usr/bin/env python3
"""
Unit tests for bbeval.cli module.

Tests the CLI helper functions for improved modularity.
"""

import unittest
from unittest.mock import Mock, patch

from bbeval import EvaluationResult
from bbeval.models import AgentTimeoutError
from bbeval.cli import _run_test_case_grading


class TestRunTestCaseWithRetries(unittest.TestCase):
    """Test cases for the _run_test_case_grading helper function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_case = Mock()
        self.test_case.id = "test_case_123"
        self.test_case.guideline_paths = []
        # Add missing attributes that the new code path expects
        self.test_case.outcome = "Test outcome for code review"
        self.test_case.task = "Review the following code for best practices"
        self.test_case.expected_assistant_raw = "Expected code output"
        self.test_case.grader = "heuristic"  # Default grader type
        
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
    @patch('bbeval.cli.grade_test_case_heuristic')
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
        
        result = _run_test_case_grading(
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
    @patch('bbeval.cli.grade_test_case_heuristic')
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
        
        result = _run_test_case_grading(
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
        
        result = _run_test_case_grading(
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
    @patch('bbeval.cli.build_prompt_inputs')
    @patch('bbeval.cli.write_result_line')
    def test_code_generation_llm_judge(self, mock_write, mock_build_prompt, mock_dspy_predict):
        """Test LLM judge path for llm_judge grader configuration."""
        
        # Setup test case with llm_judge grader
        test_case = Mock()
        test_case.id = "test_case_123"
        test_case.guideline_paths = []
        test_case.outcome = "Test outcome for code review"
        test_case.task = "Review the following code for best practices"
        test_case.expected_assistant_raw = "Expected code output"
        test_case.grader = "llm_judge"  # Use LLM judge
        
        # Setup mocks
        mock_build_prompt.return_value = {"prompt": "test prompt"}
        
        # Mock the LLM judge
        mock_judge_instance = Mock()
        mock_judge_instance.return_value = Mock(score="0.85", reasoning="Well implemented code")
        mock_dspy_predict.return_value = mock_judge_instance
        
        result = _run_test_case_grading(
            test_case=test_case,
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
            key_principle=test_case.outcome,
            task_requirements=test_case.task,
            reference_answer=test_case.expected_assistant_raw,
            generated_answer="Mock review response"
        )
        
        # Verify result structure
        self.assertEqual(result.test_id, "test_case_123")
        self.assertEqual(result.score, 0.85)
        self.assertEqual(result.hits, ["Well implemented code"])
        self.assertEqual(result.misses, [])
        self.assertEqual(result.expected_aspect_count, 1)


class TestTargetSelection(unittest.TestCase):
    """Test cases for automatic target selection logic."""
    
    @patch('bbeval.cli.load_targets')
    @patch('bbeval.cli.find_target')
    @patch('bbeval.cli.Path')
    @patch('bbeval.cli.yaml.safe_load')
    @patch('builtins.open', create=True)
    def test_cli_target_overrides_yaml(self, mock_open, mock_yaml_load, mock_path, mock_find_target, mock_load_targets):
        """Test that CLI --target flag overrides YAML target specification."""
        # Mock file operations
        mock_path.return_value.exists.return_value = True
        mock_yaml_load.return_value = {'target': 'yaml_target'}
        mock_load_targets.return_value = [{'name': 'cli_target', 'provider': 'mock'}]
        mock_find_target.return_value = {'name': 'cli_target', 'provider': 'mock'}
        
        # Mock sys.argv for argparse
        test_args = ['bbeval', 'test.yaml', '--target', 'cli_target']
        with patch('sys.argv', test_args):
            from bbeval.cli import main
            with patch('bbeval.cli.run_evaluation') as mock_run_eval:
                with patch('bbeval.cli.print_summary'):
                    with patch('bbeval.cli.get_default_output_path'):
                        try:
                            main()
                        except SystemExit:
                            pass  # Expected due to mocked environment
        
        # Verify CLI target was used (not YAML target)
        mock_find_target.assert_called_with('cli_target', unittest.mock.ANY)
    
    @patch('bbeval.cli.load_targets')
    @patch('bbeval.cli.find_target')
    @patch('bbeval.cli.Path')
    @patch('bbeval.cli.yaml.safe_load')
    @patch('builtins.open', create=True)
    def test_yaml_target_used_when_no_cli_override(self, mock_open, mock_yaml_load, mock_path, mock_find_target, mock_load_targets):
        """Test that YAML target is used when CLI target is default."""
        # Mock file operations
        mock_path.return_value.exists.return_value = True
        mock_yaml_load.return_value = {'target': 'yaml_target'}
        mock_load_targets.return_value = [{'name': 'yaml_target', 'provider': 'mock'}]
        mock_find_target.return_value = {'name': 'yaml_target', 'provider': 'mock'}
        
        # Mock sys.argv with default target (should use YAML)
        test_args = ['bbeval', 'test.yaml']
        with patch('sys.argv', test_args):
            from bbeval.cli import main
            with patch('bbeval.cli.run_evaluation') as mock_run_eval:
                with patch('bbeval.cli.print_summary'):
                    with patch('bbeval.cli.get_default_output_path'):
                        try:
                            main()
                        except SystemExit:
                            pass  # Expected due to mocked environment
        
        # Verify YAML target was used
        mock_find_target.assert_called_with('yaml_target', unittest.mock.ANY)
    
    @patch('bbeval.cli.load_targets')
    @patch('bbeval.cli.find_target')
    @patch('bbeval.cli.Path')
    @patch('bbeval.cli.yaml.safe_load')
    @patch('builtins.open', create=True)
    def test_default_target_when_no_specification(self, mock_open, mock_yaml_load, mock_path, mock_find_target, mock_load_targets):
        """Test that 'default' target is used when no target is specified anywhere."""
        # Mock file operations - no target in YAML
        mock_path.return_value.exists.return_value = True
        mock_yaml_load.return_value = {}  # No target key
        mock_load_targets.return_value = [{'name': 'default', 'provider': 'mock'}]
        mock_find_target.return_value = {'name': 'default', 'provider': 'mock'}
        
        # Mock sys.argv with default target and no YAML target
        test_args = ['bbeval', 'test.yaml']
        with patch('sys.argv', test_args):
            from bbeval.cli import main
            with patch('bbeval.cli.run_evaluation') as mock_run_eval:
                with patch('bbeval.cli.print_summary'):
                    with patch('bbeval.cli.get_default_output_path'):
                        try:
                            main()
                        except SystemExit:
                            pass  # Expected due to mocked environment
        
        # Verify default target was used
        mock_find_target.assert_called_with('default', unittest.mock.ANY)


class TestVersionFlag(unittest.TestCase):
    """Test the --version flag for the CLI."""

    @patch('bbeval.cli.metadata.version')
    def test_version_flag_outputs_version(self, mock_meta_version):
        mock_meta_version.return_value = '9.9.9'
        test_args = ['bbeval', '--version']
        with patch('sys.argv', test_args):
            # Capture SystemExit raised by argparse after printing version
            with self.assertRaises(SystemExit) as cm:
                from bbeval.cli import main
                main()
            self.assertEqual(cm.exception.code, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
