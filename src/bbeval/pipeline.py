"""
Pipeline Module for Bbeval

Provides high-level pipeline functions for running evaluations
and orchestrating the different components.
"""

from typing import List, Dict, Optional
from pathlib import Path
import json

from . import TestCase, EvaluationResult
from .yaml_parser import load_testcases, build_prompt_inputs
from .models import configure_dspy_model, AgentTimeoutError
from .signatures import EvaluationModule, determine_signature_from_test_case
from .grading import grade_test_case_heuristic, grade_test_case_llm_judge

class EvaluationPipeline:
    """
    Main pipeline for running evaluations.
    """
    
    def __init__(self, 
                 provider: str = "azure",
                 model: str = "gpt-4",
                 repo_root: Optional[Path] = None):
        """
        Initialize the evaluation pipeline.
        
        Args:
            provider: Model provider ('anthropic', 'azure', 'mock')
            model: Model name/deployment
            repo_root: Repository root (auto-detected if None)
        """
        self.provider = provider
        self.model = model
        self.repo_root = repo_root or self._find_repo_root()
        self.results: List[EvaluationResult] = []
        
        # Configure DSPy model
        configure_dspy_model(provider, model)
        
    def _find_repo_root(self) -> Path:
        """Find the repository root directory."""
        current = Path.cwd()
        while current != current.parent:
            if (current / '.git').exists():
                return current
            current = current.parent
        return Path.cwd()
    
    def load_test_file(self, test_file_path: str) -> List[TestCase]:
        """
        Load test cases from a YAML file.
        
        Args:
            test_file_path: Path to the test file
            
        Returns:
            List of test cases
        """
        return load_testcases(test_file_path, self.repo_root)
    
    def run_evaluation(self, 
                      test_cases: List[TestCase],
                      max_retries: int = 2) -> List[EvaluationResult]:
        """
        Run evaluation on a list of test cases.
        
        Args:
            test_cases: List of test cases to evaluate
            max_retries: Maximum number of retries for timeout cases
            
        Returns:
            List of evaluation results
        """
        if not test_cases:
            return []

        results = []

        for test_case in test_cases:
            # Determine appropriate signature for this test case
            signature_class = determine_signature_from_test_case(test_case)
            evaluation_module = EvaluationModule(signature_class=signature_class)
            
            retry_count = 0
            max_attempts = max_retries + 1
            test_completed = False
            
            while retry_count < max_attempts and not test_completed:
                try:
                    # Build prompt inputs
                    prompt_inputs = build_prompt_inputs(test_case, self.repo_root)
                    
                    # Run prediction
                    prediction = evaluation_module(**prompt_inputs)
                    candidate_response = prediction.review
                    
                    # Evaluate response based on configured grader
                    if test_case.grader == 'heuristic':
                        print("  Using heuristic grader...")
                        result = grade_test_case_heuristic(
                            test_case, 
                            candidate_response, 
                            self.provider, 
                            self.model
                        )
                    else:  # Default to LLM judge with fallback to heuristic
                        print("  Using LLM Judge for grading...")
                        try:
                            result = grade_test_case_llm_judge(
                                test_case, 
                                candidate_response, 
                                self.provider, 
                                self.model
                            )
                            # Check if LLM judge actually failed (indicated by specific error message)
                            if result.misses and any("LLM judge failed" in miss for miss in result.misses):
                                print("  LLM Judge failed, falling back to heuristic grader...")
                                result = grade_test_case_heuristic(
                                    test_case, 
                                    candidate_response, 
                                    self.provider, 
                                    self.model
                                )
                        except Exception as e:
                            print(f"  LLM Judge failed ({str(e)}), falling back to heuristic grader...")
                            result = grade_test_case_heuristic(
                                test_case, 
                                candidate_response, 
                                self.provider, 
                                self.model
                            )
                    
                    results.append(result)
                    test_completed = True
                    
                except AgentTimeoutError as e:
                    if retry_count < max_retries:
                        retry_count += 1
                        continue
                    
                    # Max retries exceeded, treat as error
                    error_result = EvaluationResult(
                        test_id=test_case.id,
                        score=0.0,
                        hits=[],
                        misses=[f"Agent timeout after {max_retries} retries: {str(e)}"],
                        model_answer=f"Agent timeout occurred: {str(e)}",
                        expected_aspect_count=0,
                        provider=self.provider,
                        model=self.model,
                        timestamp="",
                        raw_aspects=[]
                    )
                    results.append(error_result)
                    test_completed = True
                    
                except Exception as e:
                    # Create error result
                    error_result = EvaluationResult(
                        test_id=test_case.id,
                        score=0.0,
                        hits=[],
                        misses=[f"Error: {str(e)}"],
                        model_answer=f"Error occurred: {str(e)}",
                        expected_aspect_count=0,
                        provider=self.provider,
                        model=self.model,
                        timestamp="",
                        raw_aspects=[]
                    )
                    results.append(error_result)
                    test_completed = True
        
        self.results.extend(results)
        return results
    
    def run_from_file(self, test_file_path: str) -> List[EvaluationResult]:
        """
        Run evaluation from a test file.
        
        Args:
            test_file_path: Path to the test YAML file
            
        Returns:
            List of evaluation results
        """
        # Load test cases
        test_cases = self.load_test_file(test_file_path)
        
        # Run evaluation
        return self.run_evaluation(test_cases)
    
    def save_results(self, output_file: str, results: Optional[List[EvaluationResult]] = None):
        """
        Save results to a JSONL file.
        
        Args:
            output_file: Output file path
            results: Results to save (uses instance results if None)
        """
        results_to_save = results or self.results
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results_to_save:
                result_dict = {
                    'test_id': result.test_id,
                    'score': result.score,
                    'hits': result.hits,
                    'misses': result.misses,
                    'model_answer': result.model_answer,
                    'expected_aspect_count': result.expected_aspect_count,
                    'provider': result.provider,
                    'model': result.model,
                    'timestamp': result.timestamp
                }
                f.write(json.dumps(result_dict) + '\n')
    
    def get_summary_stats(self, results: Optional[List[EvaluationResult]] = None) -> Dict:
        """
        Get summary statistics for results.
        
        Args:
            results: Results to summarize (uses instance results if None)
            
        Returns:
            Dictionary with summary statistics
        """
        import statistics
        
        results_to_analyze = results or self.results
        
        if not results_to_analyze:
            return {}
        
        scores = [r.score for r in results_to_analyze]
        
        stats = {
            'total_cases': len(results_to_analyze),
            'mean_score': statistics.mean(scores),
            'median_score': statistics.median(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'provider': self.provider,
            'model': self.model,
        }
        
        if len(scores) > 1:
            stats['std_deviation'] = statistics.stdev(scores)
        
        return stats
