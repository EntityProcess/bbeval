"""
Command Line Interface for Bbeval

Provides CLI for running evaluations against test YAML files with
support for multiple model providers and configuration via execution targets.
"""

import argparse
import json
import os
import sys
import yaml
from pathlib import Path
from typing import List, Dict
import statistics
from datetime import datetime, timezone
import dspy

# Import dotenv for later use
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

from . import EvaluationResult
from .yaml_parser import load_testcases, build_prompt_inputs
from .models import configure_dspy_model, AgentTimeoutError
from .signatures import EvaluationModule, determine_signature_from_test_case, CodeGeneration, CodeComparisonJudge, QualityGrader
from .grading import grade_test_case_heuristic

def load_targets(targets_file_path: str = None) -> List[Dict]:
    """Load execution targets from a YAML file."""
    if targets_file_path:
        targets_file = Path(targets_file_path)
    else:
        # Default to looking for .bbeval/targets.yaml in the current working directory.
        cwd = Path.cwd()
        targets_file = cwd / ".bbeval" / "targets.yaml"
    
    if not targets_file.exists():
        raise FileNotFoundError(
            "Could not find '.bbeval/targets.yaml' in the current directory. "
            "Please specify the path using the --targets flag."
        )
    
    with open(targets_file, 'r', encoding='utf-8') as f:
        targets = yaml.safe_load(f)
    
    if not isinstance(targets, list):
        raise ValueError("targets.yaml must contain a list of target configurations")
    
    return targets


def find_target(target_name: str, targets: List[Dict]) -> Dict:
    """Find a target configuration by name."""
    for target in targets:
        if target.get('name') == target_name:
            return target
    
    available_targets = [t.get('name', 'unnamed') for t in targets]
    raise ValueError(f"Target '{target_name}' not found. Available targets: {', '.join(available_targets)}")

def create_judge_model(target: Dict, targets: List[Dict], model: str, verbose: bool = False):
    """
    Create a judge model based on target configuration.
    
    Args:
        target: The current execution target
        targets: List of all available targets
        model: The model name to use
        verbose: Whether to print verbose output
        
    Returns:
        Configured judge model instance
    """
    judge_target_name = target.get('judge_target')
    
    if judge_target_name:
        # Use the specified judge target
        if verbose:
            print(f"  Using judge target: {judge_target_name}")
        judge_target = find_target(judge_target_name, targets)
        judge_provider = judge_target['provider']
        judge_settings = judge_target.get('settings')
        
        # Get model from judge target settings or fall back to provided model
        judge_model = judge_settings.get('model', model) if judge_settings else model
        if isinstance(judge_model, str) and judge_model in os.environ:
            judge_model = os.getenv(judge_model, model)
    else:
        # Fallback to hardcoded Azure for backward compatibility (when no judge_target specified)
        if verbose:
            print(f"  No judge_target specified, falling back to Azure")
        judge_provider = "azure"
        judge_settings = {
            'endpoint': 'AZURE_OPEN_AI_ENDPOINT',
            'api_key': 'AZURE_OPEN_AI_API_KEY'
        }
        judge_model = model
        
        # Check if Azure credentials are available for fallback
        if not os.getenv('AZURE_OPEN_AI_ENDPOINT') or not os.getenv('AZURE_OPEN_AI_API_KEY'):
            if verbose:
                print(f"  Azure credentials not found, using mock judge")
            judge_provider = "mock"
            judge_settings = {}
            judge_model = "mock-model"
    
    from .models import create_model
    return create_model(judge_provider, judge_model, judge_settings)

def get_repo_root() -> Path:
    """Find the repository root directory."""
    current = Path.cwd()
    while current != current.parent:
        if (current / '.git').exists():
            return current
        current = current.parent
    return Path.cwd()

def get_default_output_path(test_file: str) -> str:
    """
    Generate default output path in .bbeval/results folder based on test file name.
    
    Args:
        test_file: Path to the test YAML file
        
    Returns:
        Default output file path in .bbeval/results folder
    """
    # Get the base name of the test file without extension
    test_path = Path(test_file)
    base_name = test_path.stem.replace('.test', '')
    
    # Add timestamp to make it unique
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output filename
    output_filename = f"{base_name}_{timestamp}.jsonl"
    
    # Return path relative to current working directory's .bbeval/results folder
    results_dir = Path.cwd() / ".bbeval" / "results"
    return str(results_dir / output_filename)


def _run_test_case_grading(
    test_case,
    evaluation_module,
    repo_root: str,
    provider: str,
    settings: Dict,
    model: str,
    output_file: str,
    dry_run: bool,
    verbose: bool,
    max_retries: int,
    target: Dict,
    targets: List[Dict],
    use_cache: bool = False
) -> EvaluationResult:
    """
    Execute a single test case with retry logic and conditional judging.
    
    Args:
        test_case: The test case to execute
        evaluation_module: The evaluation module to use
        repo_root: Repository root path
        provider: Model provider name
        settings: Provider settings
        model: Model identifier
        output_file: Optional output file path
        dry_run: Whether running in dry-run mode
        verbose: Whether to print verbose output
        max_retries: Maximum number of retries for timeout cases
        target: Current execution target configuration
        targets: List of all available targets
        use_cache: Whether to enable DSPy caching for LLM responses
    
    Returns:
        EvaluationResult for the test case
    """
    retry_count = 0
    max_attempts = max_retries + 1

    while retry_count < max_attempts:
        if retry_count > 0:
            print(f"  Retry attempt {retry_count}/{max_retries} for test case: {test_case.id}")

        try:
            # Build prompt inputs (without leaking expected response)
            prompt_inputs = build_prompt_inputs(test_case, repo_root)
            
            # Add guideline paths for VS Code provider
            prompt_inputs['guideline_paths'] = test_case.guideline_paths
            
            # Run the model prediction with conditional caching
            print(f"  Running prediction...")
            if not use_cache:
                print("    (Cache is disabled)")
            else:
                print("    (Cache is enabled)")
            prediction = evaluation_module(test_case_id=test_case.id, **prompt_inputs)
            candidate_response = prediction.answer
            
            # Use grader configuration from test case
            if test_case.grader == 'llm_judge':
                # Use LLM grader
                print("  Using LLM Judge for grading...")
                
                # For VSCode provider, we need to temporarily switch to a standard model for judging
                # to avoid double JSON wrapping and incorrect file naming
                if provider.lower() == "vscode":
                    print(f"  Detected VSCode provider, switching to judge model...")
                    # Save current model
                    original_lm = dspy.settings.lm
                    
                    # Create judge model based on target configuration
                    try:
                        judge_model = create_judge_model(target, targets, model, verbose)
                        dspy.settings.configure(lm=judge_model)
                        print(f"  Successfully switched to judge model for LLM judging...")
                        
                        llm_judge = dspy.Predict(QualityGrader)
                        judgement = llm_judge(
                            key_principle=test_case.outcome,
                            task_requirements=test_case.task,
                            reference_answer=test_case.expected_assistant_raw,
                            generated_answer=candidate_response
                        )
                    except Exception as e:
                        print(f"  Warning: Failed to create judge model, falling back to mock: {e}")
                        # Fallback to mock model if judge creation fails
                        from .models import create_model
                        judge_model = create_model("mock", "mock-model")
                        dspy.settings.configure(lm=judge_model)
                        
                        llm_judge = dspy.Predict(QualityGrader)
                        judgement = llm_judge(
                            key_principle=test_case.outcome,
                            task_requirements=test_case.task,
                            reference_answer=test_case.expected_assistant_raw,
                            generated_answer=candidate_response
                        )
                    finally:
                        # Restore original model
                        print(f"  Restoring original VSCode model...")
                        dspy.settings.configure(lm=original_lm)
                else:
                    llm_judge = dspy.Predict(QualityGrader)
                    judgement = llm_judge(
                        key_principle=test_case.outcome,
                        task_requirements=test_case.task,
                        reference_answer=test_case.expected_assistant_raw,
                        generated_answer=candidate_response
                    )
                
                result = EvaluationResult(
                    test_id=test_case.id,
                    score=float(judgement.score),
                    hits=[judgement.reasoning],
                    misses=[],
                    model_answer=candidate_response,
                    expected_aspect_count=1,
                    provider=provider,
                    model=model,
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
            else:
                # Use heuristic grader (default)
                print(f"  Evaluating response with heuristic grader...")
                result = grade_test_case_heuristic(test_case, candidate_response, provider, model)

            print(f"  Score: {result.score:.2f} ({result.hit_count}/{result.expected_aspect_count} aspects)")
            
            # Write result immediately if output file specified
            if output_file:
                write_result_line(result, output_file)
            
            return result
            
        except AgentTimeoutError as e:
            if retry_count < max_retries:
                print(f"  Agent timeout detected, will retry...")
                if verbose:
                    print(f"    Timeout details: {str(e)}")
                retry_count += 1
                continue
            
            # Max retries exceeded, treat as error
            print(f"  Agent timeout after {max_retries} retries: {e}")
            error_result = EvaluationResult(
                test_id=test_case.id,
                score=0.0,
                hits=[],
                misses=[f"Agent timeout after {max_retries} retries: {str(e)}"],
                model_answer=f"Agent timeout occurred: {str(e)}",
                expected_aspect_count=0,
                provider=provider,
                model=model,
                timestamp="",
                raw_aspects=[]
            )
            
            # Write error result immediately if output file specified
            if output_file:
                write_result_line(error_result, output_file)
            
            return error_result
        
        except Exception as e:
            # For non-AgentTimeoutError exceptions, check if it's a timeout-related error
            # as a fallback (e.g., subprocess.TimeoutExpired wrapped in other exceptions)
            error_message = str(e)
            is_subprocess_timeout = "TimeoutExpired" in str(type(e)) or "timed out" in error_message.lower()
            
            if is_subprocess_timeout and retry_count < max_retries:
                print(f"  Subprocess timeout detected, will retry...")
                if verbose:
                    print(f"    Error details: {error_message}")
                retry_count += 1
                continue
            
            print(f"  Error processing test case {test_case.id}: {e}")
            # Print full traceback in verbose mode
            if verbose:
                import traceback
                traceback.print_exc()
            
            # Create error result
            error_result = EvaluationResult(
                test_id=test_case.id,
                score=0.0,
                hits=[],
                misses=[f"Error: {str(e)}"],
                model_answer=f"Error occurred: {str(e)}",
                expected_aspect_count=0,
                provider=provider,
                model=model,
                timestamp="",
                raw_aspects=[]
            )
            
            # Write error result immediately if output file specified
            if output_file:
                write_result_line(error_result, output_file)
            
            return error_result


def run_evaluation(test_file: str, 
                  target: Dict,
                  targets: List[Dict],
                  output_file: str = None,
                  dry_run: bool = False,
                  verbose: bool = False,
                  test_id: str = None,
                  agent_timeout: int = 120,
                  max_retries: int = 2,
                  use_cache: bool = False) -> List[EvaluationResult]:
    """
    Run evaluation on a test file using the specified target.
    
    Args:
        test_file: Path to the test YAML file
        target: Target configuration from targets.yaml
        targets: List of all available targets
        output_file: Optional output file for results
        dry_run: If True, use mock model
        test_id: Optional test ID to run only a specific test case
        agent_timeout: Timeout in seconds for agent response polling
        max_retries: Maximum number of retries for timeout cases
        use_cache: Whether to enable DSPy caching for LLM responses
    
    Returns:
        List of evaluation results
    """
    repo_root = get_repo_root()
    
    print(f"Loading test cases from: {test_file}")
    test_cases = load_testcases(test_file, repo_root)
    print(f"Loaded {len(test_cases)} test cases")
    
    # Filter to specific test ID if provided
    if test_id:
        original_count = len(test_cases)
        test_cases = [tc for tc in test_cases if tc.id == test_id]
        if not test_cases:
            print(f"Error: Test case with ID '{test_id}' not found")
            print(f"Available test IDs: {[tc.id for tc in load_testcases(test_file, repo_root)]}")
            return []
        print(f"Filtered to test case: {test_id} (1 of {original_count} total)")
    
    if not test_cases:
        print("No valid test cases found")
        return []
    
    # Use a generic evaluation module (no domain inference required)
    
    # Extract target configuration
    provider = target['provider']
    settings = target.get('settings')
    # VS Code Copilot has no base model parameter; model value is ignored in that case
    model = os.getenv('LLM_MODEL', 'gpt-4')
    
    # Configure model
    if dry_run:
        print("Running in dry-run mode with mock model")
        configure_dspy_model("mock", "mock-model")
        provider = "mock"
        model = "mock-model"
    else:
        print(f"Configuring {provider} target: {target['name']}")
        
        try:
            # Set DSPy global cache policy first
            if not use_cache:
                # Disable both disk and memory caches per docs
                try:
                    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
                except Exception:
                    pass
            else:
                # Use default cache (enabled) unless changed by user env
                try:
                    dspy.configure_cache(enable_disk_cache=True, enable_memory_cache=True)
                except Exception:
                    pass

            # Configure the LM, passing disable_cache to ensure LM-level cache is off
            configure_dspy_model(provider, model, settings, polling_timeout=agent_timeout, disable_cache=(not use_cache))
        except ValueError as e:
            print(f"Error configuring target: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    if verbose:
        print(f"DSPy caching is {'ENABLED' if use_cache else 'DISABLED'}")
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nProcessing test case {i}/{len(test_cases)}: {test_case.id}")
        
        # Determine appropriate signature for this test case
        signature_class = determine_signature_from_test_case(test_case)
        evaluation_module = EvaluationModule(signature_class=signature_class)
        
        print(f"  Using signature: {signature_class.__name__}")
        
        result = _run_test_case_grading(
            test_case=test_case,
            evaluation_module=evaluation_module,
            repo_root=repo_root,
            provider=provider,
            settings=settings,
            model=model,
            output_file=output_file,
            dry_run=dry_run,
            verbose=verbose,
            max_retries=max_retries,
            target=target,
            targets=targets,
            use_cache=use_cache
        )
        results.append(result)
    
    return results

def write_result_line(result: EvaluationResult, output_file: str):
    """Write a single result line to JSONL output file."""
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
    
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result_dict) + '\n')

def print_summary(results: List[EvaluationResult]):
    """Print evaluation summary statistics."""
    if not results:
        print("\nNo results to summarize")
        return
    
    scores = [r.score for r in results]
    
    print(f"\n{'='*50}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total test cases: {len(results)}")
    print(f"Mean score: {statistics.mean(scores):.3f}")
    print(f"Median score: {statistics.median(scores):.3f}")
    print(f"Min score: {min(scores):.3f}")
    print(f"Max score: {max(scores):.3f}")
    
    if len(scores) > 1:
        print(f"Std deviation: {statistics.stdev(scores):.3f}")
    
    # Score distribution
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    distribution = {f"{bins[i]:.1f}-{bins[i+1]:.1f}": 0 for i in range(len(bins)-1)}
    
    for score in scores:
        for i in range(len(bins)-1):
            if bins[i] <= score <= bins[i+1]:
                distribution[f"{bins[i]:.1f}-{bins[i+1]:.1f}"] += 1
                break
    
    print(f"\nScore distribution:")
    for range_str, count in distribution.items():
        print(f"  {range_str}: {count}")
    
    # Top performing test cases
    sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
    print(f"\nTop 3 performing test cases:")
    for i, result in enumerate(sorted_results[:3], 1):
        print(f"  {i}. {result.test_id}: {result.score:.3f}")
    
    # Lowest performing test cases
    print(f"\nLowest 3 performing test cases:")
    for i, result in enumerate(sorted_results[-3:], 1):
        print(f"  {i}. {result.test_id}: {result.score:.3f}")

def main():
    """Main CLI entry point."""
    # Load environment variables from .env file in current working directory
    if DOTENV_AVAILABLE:
        # Explicitly load .env from current working directory
        env_file = Path.cwd() / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            print(f"Loaded .env file from: {env_file}")
        else:
            print(f"No .env file found at: {env_file}")
    
    parser = argparse.ArgumentParser(description="Bbeval")
    
    parser.add_argument('test_file',
                       help='Path to the .test.yaml file to run.')
    parser.add_argument('--target', default='default',
                       help='Execution target name from targets.yaml (default: default)')
    parser.add_argument('--targets', 
                       help='Path to targets.yaml file (default: ./.bbeval/targets.yaml)')
    parser.add_argument('--test-id',
                       help='Run only the test case with this specific ID')
    parser.add_argument('--out', dest='output_file',
                       help='Output JSONL file path (default: results/{testname}_{timestamp}.jsonl)')
    # Domain is auto-inferred from the test file path; no override flag is provided
    parser.add_argument('--dry-run', action='store_true',
                       help='Run with mock model for testing')
    parser.add_argument('--agent-timeout', type=int, default=120,
                       help='Timeout in seconds for agent response polling (default: 120)')
    parser.add_argument('--max-retries', type=int, default=2,
                       help='Maximum number of retries for timeout cases (default: 2)')
    parser.add_argument('--cache', action='store_true',
                       help='Enable DSPy caching for LLM responses')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Validate test file exists
    if not Path(args.test_file).exists():
        print(f"Error: Test file not found: {args.test_file}")
        sys.exit(1)
    
    # Determine target name with precedence: CLI flag (if not 'default') > YAML file key > 'default'
    target_name_from_cli = args.target
    target_name_from_file = None
    
    # Try to read target from test file
    try:
        with open(args.test_file, 'r', encoding='utf-8') as f:
            try:
                test_suite_config = yaml.safe_load(f)
                if isinstance(test_suite_config, dict):
                    target_name_from_file = test_suite_config.get('target')
            except yaml.YAMLError as ye:
                print(f"Warning: Failed to parse test file YAML while looking for 'target': {ye}")
    except Exception as e:
        print(f"Warning: Unable to read test file for target detection: {e}")
    
    # Apply precedence logic
    if target_name_from_cli != 'default':
        # CLI override provided (not the default sentinel)
        final_target_name = target_name_from_cli
        print(f"Using target from --target flag: {final_target_name}")
    elif target_name_from_file:
        # Use target from YAML file
        final_target_name = target_name_from_file
        print(f"Using target from test file: {final_target_name}")
    else:
        # Fall back to 'default' target (original behavior)
        final_target_name = 'default'
        print(f"Using default target: {final_target_name}")

    # Load targets and locate the chosen target configuration
    try:
        targets = load_targets(args.targets)
        target = find_target(final_target_name, targets)
        print(f"Using target: {target['name']} (provider: {target['provider']})")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Set default output file if not specified
    if not args.output_file:
        args.output_file = get_default_output_path(args.test_file)
        print(f"No output file specified, defaulting to: {args.output_file}")
    
    # Create output directory if specified
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Clear the output file
        if output_path.exists():
            try:
                output_path.unlink()
                if args.verbose:
                    print(f"Cleared existing output file: {output_path}")
            except Exception as e:
                print(f"Warning: Unable to clear output file {output_path}: {e}")
    
    try:
        # Run evaluation
        results = run_evaluation(
            test_file=args.test_file,
            target=target,
            targets=targets,
            output_file=args.output_file,
            dry_run=args.dry_run,
            verbose=args.verbose,
            test_id=args.test_id,
            agent_timeout=args.agent_timeout,
            max_retries=args.max_retries,
            use_cache=args.cache
        )
        
        # Print summary
        print_summary(results)
        
        if args.output_file:
            print(f"\nResults written to: {args.output_file}")
    
    except Exception as e:
        print(f"Error running evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
