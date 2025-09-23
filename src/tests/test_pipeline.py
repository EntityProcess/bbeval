#!/usr/bin/env python3
"""
Unit tests for bbeval.pipeline module.

Tests the EvaluationPipeline class and its methods, including
LLM judge fallback functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import tempfile
import shutil
import os
import json
from pathlib import Path

from bbeval.pipeline import EvaluationPipeline
from bbeval import EvaluationResult
from bbeval.models import AgentTimeoutError

# Import TestCase with a different name to avoid pytest collection warning
from bbeval import TestCase as BbevalTestCase

class TestEvaluationPipeline(unittest.TestCase):
    """Test cases for the EvaluationPipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_root = Path(self.temp_dir)
        
        # Create a mock git repository
        git_dir = self.repo_root / '.git'
        git_dir.mkdir()
        
        # Mock DSPy configuration to avoid actual model setup
        with patch('bbeval.pipeline.configure_dspy_model'):
            self.pipeline = EvaluationPipeline(
                provider="mock",
                model="test-model",
                repo_root=self.repo_root
            )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_init_with_defaults(self):
        """Test pipeline initialization with default parameters."""
        with patch('bbeval.pipeline.configure_dspy_model'):
            with patch.object(EvaluationPipeline, '_find_repo_root', return_value=Path('/mock/root')):
                pipeline = EvaluationPipeline()
                self.assertEqual(pipeline.provider, "azure")
                self.assertEqual(pipeline.model, "gpt-4")
                self.assertEqual(pipeline.repo_root, Path('/mock/root'))
                self.assertEqual(pipeline.results, [])
    
    def test_init_with_custom_params(self):
        """Test pipeline initialization with custom parameters."""
        with patch('bbeval.pipeline.configure_dspy_model'):
            pipeline = EvaluationPipeline(
                provider="anthropic",
                model="claude-3",
                repo_root=Path("/custom/path")
            )
            self.assertEqual(pipeline.provider, "anthropic")
            self.assertEqual(pipeline.model, "claude-3")
            self.assertEqual(pipeline.repo_root, Path("/custom/path"))
    
    def test_find_repo_root_with_git(self):
        """Test finding repository root when .git directory exists."""
        pipeline = EvaluationPipeline.__new__(EvaluationPipeline)
        
        # Create nested directory structure with .git at root
        nested_dir = self.repo_root / 'src' / 'subdir'
        nested_dir.mkdir(parents=True)
        
        with patch('pathlib.Path.cwd', return_value=nested_dir):
            root = pipeline._find_repo_root()
            self.assertEqual(root, self.repo_root)
    
    def test_find_repo_root_no_git(self):
        """Test finding repository root when no .git directory exists."""
        pipeline = EvaluationPipeline.__new__(EvaluationPipeline)
        
        # Remove .git directory
        shutil.rmtree(self.repo_root / '.git')
        
        with patch('pathlib.Path.cwd', return_value=self.repo_root):
            root = pipeline._find_repo_root()
            self.assertEqual(root, self.repo_root)
    
    @patch('bbeval.pipeline.load_testcases')
    def test_load_test_file(self, mock_load):
        """Test loading test cases from a file."""
        mock_test_cases = [
            BbevalTestCase(
                id="test1",
                task="Test task",
                user_segments=[],
                expected_assistant_raw="Expected response",
                guideline_paths=[],
                code_snippets=[],
                outcome="Should work correctly",
                grader="llm_judge"
            )
        ]
        mock_load.return_value = mock_test_cases
        
        result = self.pipeline.load_test_file("test.yaml")
        
        mock_load.assert_called_once_with("test.yaml", self.repo_root)
        self.assertEqual(result, mock_test_cases)
    
    def test_run_evaluation_empty_list(self):
        """Test running evaluation with empty test case list."""
        result = self.pipeline.run_evaluation([])
        self.assertEqual(result, [])
    
    @patch('bbeval.pipeline.determine_signature_from_test_case')
    @patch('bbeval.pipeline.EvaluationModule')
    @patch('bbeval.pipeline.build_prompt_inputs')
    @patch('bbeval.pipeline.grade_test_case_heuristic')
    def test_run_evaluation_heuristic_grader(self, mock_grade_heuristic, mock_build_inputs, 
                                           mock_eval_module_class, mock_determine_sig):
        """Test running evaluation with heuristic grader."""
        # Setup mocks
        mock_signature = Mock()
        mock_determine_sig.return_value = mock_signature
        
        mock_eval_module = Mock()
        mock_prediction = Mock()
        mock_prediction.review = "Test response"
        mock_eval_module.return_value = mock_prediction
        mock_eval_module_class.return_value = mock_eval_module
        
        mock_build_inputs.return_value = {"input": "test"}
        
        mock_result = EvaluationResult(
            test_id="test1",
            score=0.8,
            hits=["good point"],
            misses=["missed point"],
            model_answer="Test response",
            expected_aspect_count=2,
            provider="mock",
            model="test-model",
            timestamp="2023-01-01T00:00:00Z",
            raw_aspects=[]
        )
        mock_grade_heuristic.return_value = mock_result
        
        test_case = BbevalTestCase(
            id="test1",
            task="Test task",
            user_segments=[],
            expected_assistant_raw="Expected response",
            guideline_paths=[],
            code_snippets=[],
            outcome="Should work correctly",
            grader="heuristic"  # Explicitly set to heuristic
        )
        
        results = self.pipeline.run_evaluation([test_case])
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], mock_result)
        mock_grade_heuristic.assert_called_once()
    
    @patch('bbeval.pipeline.determine_signature_from_test_case')
    @patch('bbeval.pipeline.EvaluationModule')
    @patch('bbeval.pipeline.build_prompt_inputs')
    @patch('bbeval.pipeline.grade_test_case_llm_judge')
    def test_run_evaluation_llm_judge_success(self, mock_grade_llm, mock_build_inputs, 
                                            mock_eval_module_class, mock_determine_sig):
        """Test running evaluation with LLM judge (successful)."""
        # Setup mocks
        mock_signature = Mock()
        mock_determine_sig.return_value = mock_signature
        
        mock_eval_module = Mock()
        mock_prediction = Mock()
        mock_prediction.review = "Test response"
        mock_eval_module.return_value = mock_prediction
        mock_eval_module_class.return_value = mock_eval_module
        
        mock_build_inputs.return_value = {"input": "test"}
        
        mock_result = EvaluationResult(
            test_id="test1",
            score=0.9,
            hits=["excellent point"],
            misses=[],
            model_answer="Test response",
            expected_aspect_count=1,
            provider="mock",
            model="test-model",
            timestamp="2023-01-01T00:00:00Z",
            raw_aspects=[]
        )
        mock_grade_llm.return_value = mock_result
        
        test_case = BbevalTestCase(
            id="test1",
            task="Test task",
            user_segments=[],
            expected_assistant_raw="Expected response",
            guideline_paths=[],
            code_snippets=[],
            outcome="Should work correctly",
            grader="llm_judge"  # Default value
        )
        
        results = self.pipeline.run_evaluation([test_case])
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], mock_result)
        mock_grade_llm.assert_called_once()
    
    @patch('bbeval.pipeline.determine_signature_from_test_case')
    @patch('bbeval.pipeline.EvaluationModule')
    @patch('bbeval.pipeline.build_prompt_inputs')
    @patch('bbeval.pipeline.grade_test_case_llm_judge')
    @patch('bbeval.pipeline.grade_test_case_heuristic')
    def test_run_evaluation_llm_judge_fallback_on_error_result(self, mock_grade_heuristic, 
                                                              mock_grade_llm, mock_build_inputs, 
                                                              mock_eval_module_class, mock_determine_sig):
        """Test LLM judge fallback when LLM judge returns error result."""
        # Setup mocks
        mock_signature = Mock()
        mock_determine_sig.return_value = mock_signature
        
        mock_eval_module = Mock()
        mock_prediction = Mock()
        mock_prediction.review = "Test response"
        mock_eval_module.return_value = mock_prediction
        mock_eval_module_class.return_value = mock_eval_module
        
        mock_build_inputs.return_value = {"input": "test"}
        
        # LLM judge returns error result
        llm_error_result = EvaluationResult(
            test_id="test1",
            score=0.0,
            hits=[],
            misses=["LLM judge failed: API error"],
            model_answer="Test response",
            expected_aspect_count=0,
            provider="mock",
            model="test-model",
            timestamp="2023-01-01T00:00:00Z",
            raw_aspects=[]
        )
        mock_grade_llm.return_value = llm_error_result
        
        # Heuristic fallback result
        heuristic_result = EvaluationResult(
            test_id="test1",
            score=0.7,
            hits=["fallback point"],
            misses=[],
            model_answer="Test response",
            expected_aspect_count=1,
            provider="mock",
            model="test-model",
            timestamp="2023-01-01T00:00:00Z",
            raw_aspects=[]
        )
        mock_grade_heuristic.return_value = heuristic_result
        
        test_case = BbevalTestCase(
            id="test1",
            task="Test task",
            user_segments=[],
            expected_assistant_raw="Expected response",
            guideline_paths=[],
            code_snippets=[],
            outcome="Should work correctly",
            grader="llm_judge"
        )
        
        results = self.pipeline.run_evaluation([test_case])
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], heuristic_result)  # Should use heuristic result
        mock_grade_llm.assert_called_once()
        mock_grade_heuristic.assert_called_once()
    
    @patch('bbeval.pipeline.determine_signature_from_test_case')
    @patch('bbeval.pipeline.EvaluationModule')
    @patch('bbeval.pipeline.build_prompt_inputs')
    @patch('bbeval.pipeline.grade_test_case_llm_judge')
    @patch('bbeval.pipeline.grade_test_case_heuristic')
    def test_run_evaluation_llm_judge_fallback_on_exception(self, mock_grade_heuristic, 
                                                           mock_grade_llm, mock_build_inputs, 
                                                           mock_eval_module_class, mock_determine_sig):
        """Test LLM judge fallback when LLM judge raises exception."""
        # Setup mocks
        mock_signature = Mock()
        mock_determine_sig.return_value = mock_signature
        
        mock_eval_module = Mock()
        mock_prediction = Mock()
        mock_prediction.review = "Test response"
        mock_eval_module.return_value = mock_prediction
        mock_eval_module_class.return_value = mock_eval_module
        
        mock_build_inputs.return_value = {"input": "test"}
        
        # LLM judge raises exception
        mock_grade_llm.side_effect = Exception("Network error")
        
        # Heuristic fallback result
        heuristic_result = EvaluationResult(
            test_id="test1",
            score=0.6,
            hits=["fallback point"],
            misses=[],
            model_answer="Test response",
            expected_aspect_count=1,
            provider="mock",
            model="test-model",
            timestamp="2023-01-01T00:00:00Z",
            raw_aspects=[]
        )
        mock_grade_heuristic.return_value = heuristic_result
        
        test_case = BbevalTestCase(
            id="test1",
            task="Test task",
            user_segments=[],
            expected_assistant_raw="Expected response",
            guideline_paths=[],
            code_snippets=[],
            outcome="Should work correctly",
            grader="llm_judge"
        )
        
        results = self.pipeline.run_evaluation([test_case])
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], heuristic_result)  # Should use heuristic result
        mock_grade_llm.assert_called_once()
        mock_grade_heuristic.assert_called_once()
    
    @patch('bbeval.pipeline.determine_signature_from_test_case')
    @patch('bbeval.pipeline.EvaluationModule')
    @patch('bbeval.pipeline.build_prompt_inputs')
    def test_run_evaluation_timeout_retry(self, mock_build_inputs, mock_eval_module_class, mock_determine_sig):
        """Test retry logic on agent timeout."""
        # Setup mocks
        mock_signature = Mock()
        mock_determine_sig.return_value = mock_signature
        
        mock_eval_module = Mock()
        # First two calls timeout, third succeeds
        mock_eval_module.side_effect = [
            AgentTimeoutError("Timeout 1"),
            AgentTimeoutError("Timeout 2"),
            Mock(review="Success response")
        ]
        mock_eval_module_class.return_value = mock_eval_module
        
        mock_build_inputs.return_value = {"input": "test"}
        
        test_case = BbevalTestCase(
            id="test1",
            task="Test task",
            user_segments=[],
            expected_assistant_raw="Expected response",
            guideline_paths=[],
            code_snippets=[],
            outcome="Should work correctly",
            grader="heuristic"
        )
        
        with patch('bbeval.pipeline.grade_test_case_heuristic') as mock_grade:
            mock_result = EvaluationResult(
                test_id="test1",
                score=0.8,
                hits=["success"],
                misses=[],
                model_answer="Success response",
                expected_aspect_count=1,
                provider="mock",
                model="test-model",
                timestamp="2023-01-01T00:00:00Z",
                raw_aspects=[]
            )
            mock_grade.return_value = mock_result
            
            results = self.pipeline.run_evaluation([test_case], max_retries=2)
            
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0], mock_result)
            self.assertEqual(mock_eval_module.call_count, 3)  # Two retries + success
    
    @patch('bbeval.pipeline.determine_signature_from_test_case')
    @patch('bbeval.pipeline.EvaluationModule')
    @patch('bbeval.pipeline.build_prompt_inputs')
    def test_run_evaluation_timeout_max_retries_exceeded(self, mock_build_inputs, mock_eval_module_class, mock_determine_sig):
        """Test max retries exceeded for agent timeout."""
        # Setup mocks
        mock_signature = Mock()
        mock_determine_sig.return_value = mock_signature
        
        mock_eval_module = Mock()
        mock_eval_module.side_effect = AgentTimeoutError("Persistent timeout")
        mock_eval_module_class.return_value = mock_eval_module
        
        mock_build_inputs.return_value = {"input": "test"}
        
        test_case = BbevalTestCase(
            id="test1",
            task="Test task",
            user_segments=[],
            expected_assistant_raw="Expected response",
            guideline_paths=[],
            code_snippets=[],
            outcome="Should work correctly",
            grader="heuristic"
        )
        
        results = self.pipeline.run_evaluation([test_case], max_retries=1)
        
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.test_id, "test1")
        self.assertEqual(result.score, 0.0)
        self.assertTrue(any("Agent timeout after 1 retries" in miss for miss in result.misses))
        self.assertEqual(mock_eval_module.call_count, 2)  # Original + 1 retry
    
    @patch('bbeval.pipeline.load_testcases')
    def test_run_from_file(self, mock_load):
        """Test running evaluation from file."""
        mock_test_cases = [
            BbevalTestCase(
                id="test1",
                task="Test task",
                user_segments=[],
                expected_assistant_raw="Expected response",
                guideline_paths=[],
                code_snippets=[],
                outcome="Should work correctly",
                grader="heuristic"
            )
        ]
        mock_load.return_value = mock_test_cases
        
        with patch.object(self.pipeline, 'run_evaluation') as mock_run_eval:
            mock_results = [Mock()]
            mock_run_eval.return_value = mock_results
            
            result = self.pipeline.run_from_file("test.yaml")
            
            mock_load.assert_called_once_with("test.yaml", self.repo_root)
            mock_run_eval.assert_called_once_with(mock_test_cases)
            self.assertEqual(result, mock_results)
    
    def test_save_results_to_file(self):
        """Test saving results to JSONL file."""
        # Create test results
        test_results = [
            EvaluationResult(
                test_id="test1",
                score=0.8,
                hits=["good point"],
                misses=["missed point"],
                model_answer="Test response",
                expected_aspect_count=2,
                provider="mock",
                model="test-model",
                timestamp="2023-01-01T00:00:00Z",
                raw_aspects=[]
            ),
            EvaluationResult(
                test_id="test2",
                score=0.9,
                hits=["excellent point", "another good point"],
                misses=[],
                model_answer="Another response",
                expected_aspect_count=2,
                provider="mock",
                model="test-model",
                timestamp="2023-01-01T00:01:00Z",
                raw_aspects=[]
            )
        ]
        
        output_file = self.temp_dir + "/results.jsonl"
        self.pipeline.save_results(output_file, test_results)
        
        # Verify file was created and contains correct data
        self.assertTrue(os.path.exists(output_file))
        
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        self.assertEqual(len(lines), 2)
        
        # Parse and verify first result
        result1 = json.loads(lines[0])
        self.assertEqual(result1['test_id'], 'test1')
        self.assertEqual(result1['score'], 0.8)
        self.assertEqual(result1['hits'], ['good point'])
        self.assertEqual(result1['misses'], ['missed point'])
        
        # Parse and verify second result
        result2 = json.loads(lines[1])
        self.assertEqual(result2['test_id'], 'test2')
        self.assertEqual(result2['score'], 0.9)
        self.assertEqual(result2['hits'], ['excellent point', 'another good point'])
        self.assertEqual(result2['misses'], [])
    
    def test_save_results_uses_instance_results(self):
        """Test saving results uses instance results when no results parameter provided."""
        # Add results to pipeline instance
        self.pipeline.results = [
            EvaluationResult(
                test_id="test1",
                score=0.7,
                hits=["instance point"],
                misses=[],
                model_answer="Instance response",
                expected_aspect_count=1,
                provider="mock",
                model="test-model",
                timestamp="2023-01-01T00:00:00Z",
                raw_aspects=[]
            )
        ]
        
        output_file = self.temp_dir + "/instance_results.jsonl"
        self.pipeline.save_results(output_file)  # No results parameter
        
        # Verify file contains instance results
        with open(output_file, 'r', encoding='utf-8') as f:
            line = f.readline()
        
        result = json.loads(line)
        self.assertEqual(result['test_id'], 'test1')
        self.assertEqual(result['hits'], ['instance point'])
    
    def test_get_summary_stats_empty_results(self):
        """Test getting summary stats with empty results."""
        stats = self.pipeline.get_summary_stats([])
        self.assertEqual(stats, {})
    
    def test_get_summary_stats_single_result(self):
        """Test getting summary stats with single result."""
        test_results = [
            EvaluationResult(
                test_id="test1",
                score=0.8,
                hits=["good point"],
                misses=[],
                model_answer="Test response",
                expected_aspect_count=1,
                provider="mock",
                model="test-model",
                timestamp="2023-01-01T00:00:00Z",
                raw_aspects=[]
            )
        ]
        
        stats = self.pipeline.get_summary_stats(test_results)
        
        expected_stats = {
            'total_cases': 1,
            'mean_score': 0.8,
            'median_score': 0.8,
            'min_score': 0.8,
            'max_score': 0.8,
            'provider': 'mock',
            'model': 'test-model'
        }
        
        self.assertEqual(stats, expected_stats)
    
    def test_get_summary_stats_multiple_results(self):
        """Test getting summary stats with multiple results."""
        test_results = [
            EvaluationResult(
                test_id="test1",
                score=0.6,
                hits=[],
                misses=[],
                model_answer="Response 1",
                expected_aspect_count=0,
                provider="mock",
                model="test-model",
                timestamp="2023-01-01T00:00:00Z",
                raw_aspects=[]
            ),
            EvaluationResult(
                test_id="test2",
                score=0.8,
                hits=[],
                misses=[],
                model_answer="Response 2",
                expected_aspect_count=0,
                provider="mock",
                model="test-model",
                timestamp="2023-01-01T00:01:00Z",
                raw_aspects=[]
            ),
            EvaluationResult(
                test_id="test3",
                score=1.0,
                hits=[],
                misses=[],
                model_answer="Response 3",
                expected_aspect_count=0,
                provider="mock",
                model="test-model",
                timestamp="2023-01-01T00:02:00Z",
                raw_aspects=[]
            )
        ]
        
        stats = self.pipeline.get_summary_stats(test_results)
        
        self.assertEqual(stats['total_cases'], 3)
        self.assertEqual(stats['mean_score'], 0.8)
        self.assertEqual(stats['median_score'], 0.8)
        self.assertEqual(stats['min_score'], 0.6)
        self.assertEqual(stats['max_score'], 1.0)
        self.assertEqual(stats['provider'], 'mock')
        self.assertEqual(stats['model'], 'test-model')
        self.assertIn('std_deviation', stats)
        self.assertGreater(stats['std_deviation'], 0)
    
    def test_get_summary_stats_uses_instance_results(self):
        """Test getting summary stats uses instance results when no results parameter provided."""
        # Add results to pipeline instance
        self.pipeline.results = [
            EvaluationResult(
                test_id="test1",
                score=0.9,
                hits=[],
                misses=[],
                model_answer="Instance response",
                expected_aspect_count=0,
                provider="mock",
                model="test-model",
                timestamp="2023-01-01T00:00:00Z",
                raw_aspects=[]
            )
        ]
        
        stats = self.pipeline.get_summary_stats()  # No results parameter
        
        self.assertEqual(stats['total_cases'], 1)
        self.assertEqual(stats['mean_score'], 0.9)

if __name__ == '__main__':
    unittest.main()

