"""
Command Line Interface for BbEval

Provides CLI for running evaluations against test YAML files with
support for multiple model providers and configuration via execution targets.
"""

import argparse
import json
import os
import re
import sys
import yaml
from pathlib import Path
from typing import List, Dict, Optional
import statistics
from datetime import datetime, timezone
import dspy
from importlib import metadata

# Import dotenv for later use
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

from . import EvaluationResult
from .yaml_parser import load_testcases, build_prompt_inputs
from .models import configure_dspy_model, AgentTimeoutError
from .signatures import EvaluationModule, QuerySignature, QualityGrader
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
            'endpoint': 'AZURE_OPENAI_ENDPOINT',
            'api_key': 'AZURE_OPENAI_API_KEY'
        }
        judge_model = model
        
        # Check if Azure credentials are available for fallback
        if not os.getenv('AZURE_OPENAI_ENDPOINT') or not os.getenv('AZURE_OPENAI_API_KEY'):
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


def _sanitize_for_filename(value: str) -> str:
    """Sanitize arbitrary strings for safe filename usage."""
    if not value:
        return "prompt"
    sanitized = re.sub(r'[^A-Za-z0-9._-]+', '_', value)
    return sanitized or "prompt"


def _dump_prompt_inputs(dump_dir: Path, test_case, prompt_inputs: Dict[str, str], verbose: bool = False) -> None:
    """Persist the prompt inputs sent to the model for debugging."""
    try:
        dump_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{_sanitize_for_filename(getattr(test_case, 'id', 'test'))}.json"
        dump_path = dump_dir / filename
        payload = {
            "test_id": getattr(test_case, 'id', None),
            "request": prompt_inputs.get('request'),
            "guidelines": prompt_inputs.get('guidelines'),
            "guideline_paths": getattr(test_case, 'guideline_paths', []),
        }
        with open(dump_path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        if verbose:
            print(f"  Prompt dump written to: {dump_path}")
    except Exception as exc:
        print(f"\033[33mWarning: Failed to dump prompt inputs for {getattr(test_case, 'id', 'unknown')}: {exc}\033[0m")


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
    prompt_dump_dir: Optional[Path] = None
) -> EvaluationResult:
    """
    Execute a single test case with retry logic and conditional judging.
    
    Args:
        test_case: The test case to execute
        evaluation_module: The evaluation module to use
        repo_root: Repository root path
        provider: Model provider name
        settings: Provider settings
        model: Model identifier (used for judge model configuration)
        output_file: Optional output file path
        dry_run: Whether running in dry-run mode
        verbose: Whether to print verbose output
        max_retries: Maximum number of retries for timeout cases
        target: Current execution target configuration
        targets: List of all available targets
    
    Returns:
        EvaluationResult for the test case
    """
    retry_count = 0
    max_attempts = max_retries + 1

    while retry_count < max_attempts:
        if retry_count > 0:
            print(f"  Retry attempt {retry_count}/{max_retries} for test case: {test_case.id}")

        try:
            # Build prompt inputs (request + guidelines)
            prompt_inputs = build_prompt_inputs(test_case, repo_root)

            if prompt_dump_dir is not None:
                _dump_prompt_inputs(prompt_dump_dir, test_case, prompt_inputs, verbose)

            # Run the model prediction with conditional caching
            if verbose:
                print(f"  Running prediction...")
            prediction = evaluation_module(
                test_case_id=test_case.id,
                request=prompt_inputs.get('request', ''),
                guidelines=prompt_inputs.get('guidelines', ''),
                guideline_paths=test_case.guideline_paths  # Pass guideline paths for VSCodeCopilot mandatory pre-read
            )
            candidate_response = prediction.answer
            
            # Use grader configuration from test case
            if test_case.grader == 'llm_judge':
                # Use LLM grader
                print("  Using LLM Judge for grading...")
                grader_prompt_content = None  # Initialize for both VSCode and non-VSCode cases
                
                # For VSCode provider, we need to temporarily switch to a standard model for judging
                # to avoid double JSON wrapping and incorrect file naming
                if provider.lower() == "vscode":
                    if verbose:
                        print(f"  Detected VSCode provider, switching to judge model...")
                    # Save current model
                    original_lm = dspy.settings.lm
                    
                    # Create judge model based on target configuration
                    try:
                        judge_model = create_judge_model(target, targets, model, verbose)
                        dspy.settings.configure(lm=judge_model)
                        if verbose:
                            print(f"  Successfully switched to judge model for LLM judging...")
                        
                        llm_judge = dspy.Predict(QualityGrader)
                        judgement = llm_judge(
                            expected_outcome=test_case.outcome,
                            request=test_case.task,
                            reference_answer=test_case.expected_assistant_raw,
                            generated_answer=candidate_response
                        )
                        
                        # Capture grader raw request from judge model
                        last_interaction = dspy.settings.lm.history[-1]
                        grader_prompt_content = last_interaction.get('prompt') or last_interaction.get('messages')
                    except Exception as e:
                        print(f"  Warning: Failed to create judge model, falling back to mock: {e}")
                        # Fallback to mock model if judge creation fails
                        from .models import create_model
                        judge_model = create_model("mock", "mock-model")
                        dspy.settings.configure(lm=judge_model)
                        
                        llm_judge = dspy.Predict(QualityGrader)
                        judgement = llm_judge(
                            expected_outcome=test_case.outcome,
                            request=test_case.task,
                            reference_answer=test_case.expected_assistant_raw,
                            generated_answer=candidate_response
                        )
                        
                        # Capture grader raw request from mock model
                        last_interaction = dspy.settings.lm.history[-1]
                        grader_prompt_content = last_interaction.get('prompt') or last_interaction.get('messages')
                    finally:
                        # Restore original model
                        if verbose:
                            print(f"  Restoring original VSCode model...")
                        dspy.settings.configure(lm=original_lm)
                else:
                    llm_judge = dspy.Predict(QualityGrader)
                    judgement = llm_judge(
                        expected_outcome=test_case.outcome,
                        request=test_case.task,
                        reference_answer=test_case.expected_assistant_raw,
                        generated_answer=candidate_response
                    )

                    # Get the last history entry for grader_raw_request
                    last_interaction = dspy.settings.lm.history[-1]
                    # The prompt will be in 'prompt' for completion models or 'messages' for chat models
                    grader_prompt_content = last_interaction.get('prompt') or last_interaction.get('messages')
                
                # Parse hits and misses from judgement
                hits = []
                misses = []
                if hasattr(judgement, 'hits') and judgement.hits:
                    if isinstance(judgement.hits, list):
                        hits = judgement.hits
                    else:
                        hits = [judgement.hits]  # Convert single string to list
                if hasattr(judgement, 'misses') and judgement.misses:
                    if isinstance(judgement.misses, list):
                        raw_misses = judgement.misses
                    else:
                        raw_misses = [judgement.misses]  # Convert single string to list
                    
                    # Filter out empty responses and "None" responses as a safety net
                    misses = []
                    for miss in raw_misses:
                        if miss and miss.strip() and not (miss.strip().startswith('- None') or miss.strip().lower() == 'none'):
                            misses.append(miss)
                
                # Get reasoning if available
                reasoning = getattr(judgement, 'reasoning', None)
                
                # Extract aspects from expected response to calculate correct aspect count (like heuristic grader)
                from .grading import extract_aspects
                expected_aspects = extract_aspects(test_case.expected_assistant_raw)
                
                result = EvaluationResult(
                    test_id=test_case.id,
                    score=float(judgement.score),
                    hits=hits,
                    misses=misses,
                    model_answer=candidate_response,
                    expected_aspect_count=len(expected_aspects),
                    target=target['name'],
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    reasoning=reasoning,
                    raw_request=prompt_inputs  # capture structured prompt inputs
                )
                # Attach judge raw request details for auditing
                result.grader_raw_request = grader_prompt_content
            else:
                # Use heuristic grader (default)
                print(f"  Evaluating response with heuristic grader...")
                result = grade_test_case_heuristic(test_case, candidate_response, provider, target['name'])
                # Attach raw request prompt inputs for heuristic path as well
                try:
                    result.raw_request = prompt_inputs
                except Exception:
                    pass

            # Display score with appropriate context based on grader type
            if test_case.grader == 'llm_judge':
                print(f"  Score: {result.score:.2f} (LLM Judge)")
            else:
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
                target=target['name'],
                timestamp="",
                raw_aspects=[],
                raw_request=prompt_inputs
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
                target=target['name'],
                timestamp="",
                raw_aspects=[],
                raw_request=prompt_inputs if 'prompt_inputs' in locals() else None
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
                  use_cache: bool = False,
                  prompt_dump_dir: Optional[Path] = None) -> List[EvaluationResult]:
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
        prompt_dump_dir: Optional directory to persist prompt payloads for debugging
    
    Returns:
        List of evaluation results
    """
    repo_root = get_repo_root()
    
    if verbose:
        print(f"Loading test cases from: {test_file}")
    test_cases = load_testcases(test_file, repo_root, verbose=verbose)
    if verbose:
        print(f"Loaded {len(test_cases)} test cases")
    
    # Filter to specific test ID if provided
    if test_id:
        original_count = len(test_cases)
        test_cases = [tc for tc in test_cases if tc.id == test_id]
        if not test_cases:
            print(f"Error: Test case with ID '{test_id}' not found")
            print(f"Available test IDs: {[tc.id for tc in load_testcases(test_file, repo_root, verbose=False)]}")
            return []
        print(f"Filtered to test case: {test_id} (1 of {original_count} total)")
    
    if not test_cases:
        print("No valid test cases found")
        return []
    
    # Use a generic evaluation module (no domain inference required)
    
    # Extract target configuration
    provider = target['provider']
    settings = target.get('settings')
    # For DSPy configuration, we still need a model parameter (but not for results)
    model = os.getenv('LLM_MODEL', 'gpt-4')
    
    # Configure model
    if dry_run:
        print("Running in dry-run mode with mock model")
        configure_dspy_model("mock", "mock-model", verbose=verbose)
        provider = "mock"
        model = "mock-model"
        # Create a mock target for dry runs
        target = {'name': f"{target['name']}-mock", 'provider': 'mock'}
    else:
        if verbose:
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
            configure_dspy_model(provider, model, settings, verbose=verbose, polling_timeout=agent_timeout, disable_cache=(not use_cache))
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

        # Always use unified QuerySignature
        evaluation_module = EvaluationModule(signature_class=QuerySignature)
        if verbose:
            print(f"  Using signature: QuerySignature")

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
            prompt_dump_dir=prompt_dump_dir
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
        'target': result.target,
        'timestamp': result.timestamp
    }
    if getattr(result, 'reasoning', None) is not None:
        result_dict['reasoning'] = result.reasoning
    if getattr(result, 'raw_request', None) is not None:
        result_dict['raw_request'] = result.raw_request
    if getattr(result, 'grader_raw_request', None) is not None:
        result_dict['grader_raw_request'] = result.grader_raw_request
    
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
    # Determine version dynamically from package metadata; fallback for dev
    try:
        __version__ = metadata.version("bbeval")
    except metadata.PackageNotFoundError:
        __version__ = "0.0.0-dev"

    parser = argparse.ArgumentParser(description="BbEval")
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )
    
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
    parser.add_argument('--dump-prompts', dest='dump_prompts',
                       help='Directory to save the exact request and guidelines sent to the model for each test case')
    
    args = parser.parse_args()

    prompt_dump_dir = Path(args.dump_prompts).resolve() if args.dump_prompts else None

    # Load environment variables from .env file only after parsing so we can respect --verbose
    if DOTENV_AVAILABLE:
        env_file = Path.cwd() / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            if args.verbose:
                print(f"Loaded .env file from: {env_file}")
        else:
            if args.verbose:
                print(f"No .env file found at: {env_file}")
    
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
        target_source = "CLI flag"
    elif target_name_from_file:
        # Use target from YAML file
        final_target_name = target_name_from_file
        target_source = "test file"
    else:
        # Fall back to 'default' target (original behavior)
        final_target_name = 'default'
        target_source = "default"

    # Load targets and locate the chosen target configuration
    try:
        targets = load_targets(args.targets)
        target = find_target(final_target_name, targets)
        
        if args.verbose:
            print(f"Using target from {target_source}: {target['name']} (provider: {target['provider']})")
        else:
            print(f"Using target: {target['name']} (provider: {target['provider']})")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Set default output file if not specified
    if not args.output_file:
        args.output_file = get_default_output_path(args.test_file)
        print(f"Output: {args.output_file}")
    
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
            use_cache=args.cache,
            prompt_dump_dir=prompt_dump_dir
        )
        
        # Print summary
        print_summary(results)
        
        if args.output_file:
            print(f"\nResults written to: {args.output_file}")
        if prompt_dump_dir and args.verbose:
            print(f"Prompt payloads saved under: {prompt_dump_dir}")
    
    except Exception as e:
        print(f"Error running evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
