"""
YAML Parser for Test Cases

Parses .test.yaml files and extracts TestCase objects with user segments
and expected assistant responses, ensuring no leakage of ground truth
into model prompts.
"""

import re
import yaml
from pathlib import Path
from typing import List
from . import TestCase, TestMessage


def is_guideline_file(file_path: str) -> bool:
    """Determine if a file is a guideline file (instructions or prompts)."""
    return (file_path.endswith('.instructions.md') or '/instructions/' in file_path or
        file_path.endswith('.prompt.md') or '/prompts/' in file_path)


def extract_code_blocks(segments: List[dict]) -> List[str]:
    """Extract fenced code blocks from text segments."""
    code_blocks = []
    
    for segment in segments:
        if segment.get('type') == 'text':
            text = segment.get('value', '')
            # Find fenced code blocks (```...```)
            pattern = r'```[\s\S]*?```'
            matches = re.findall(pattern, text, re.MULTILINE)
            code_blocks.extend(matches)
    
    return code_blocks


def load_testcases(test_file_path: str, repo_root: Path) -> List[TestCase]:
    """
    Load test cases from a YAML file.
    
    Args:
        test_file_path: Path to the .test.yaml file
        repo_root: Root directory of the repository for resolving file paths
    
    Returns:
        List of TestCase objects
    """
    test_path = Path(test_file_path)
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_file_path}")
    
    with open(test_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    if not data or 'testcases' not in data:
        raise ValueError(f"Invalid test file format: {test_file_path}")
    
    # Get the global grader setting (default to 'heuristic' if not specified)
    global_grader = data.get('grader', 'heuristic')
    
    test_cases = []
    
    for raw_test in data.get('testcases', []):
        if 'id' not in raw_test or 'outcome' not in raw_test or 'messages' not in raw_test:
            print(f"Warning: Skipping incomplete test case: {raw_test.get('id', 'unknown')}")
            continue
        
        # Separate user and assistant messages
        user_msgs = [msg for msg in raw_test['messages'] if msg.get('role') == 'user']
        assistant_msgs = [msg for msg in raw_test['messages'] if msg.get('role') == 'assistant']
        
        if not assistant_msgs:
            print(f"Warning: No assistant message found for test case: {raw_test['id']}")
            continue
        
        if len(assistant_msgs) > 1:
            print(f"Warning: Multiple assistant messages found for test case: {raw_test['id']}, using first")
        # Process user segments
        user_segments = []
        guideline_paths = []
        user_text_parts = []

        for msg in user_msgs:
            content = msg.get('content', [])
            if isinstance(content, str):
                # Handle simple string content
                user_segments.append({'type': 'text', 'value': content})
                user_text_parts.append(content)
            elif isinstance(content, list):
                for segment in content:
                    if segment.get('type') == 'file':
                        # Read file content
                        file_path = segment['value'].lstrip('/')
                        full_path = repo_root / file_path
                        
                        if full_path.exists():
                            try:
                                with open(full_path, 'r', encoding='utf-8') as f:
                                    file_content = f.read()
                                
                                # Check if this is an instruction or prompt file
                                if is_guideline_file(file_path):
                                    # This is a guideline file - add to guideline paths but not to user segments
                                    guideline_paths.append(file_path)
                                else:
                                    # This is a regular file - add to user segments
                                    user_segments.append({
                                        'type': 'file',
                                        'path': file_path,
                                        'text': file_content
                                    })
                            except Exception as e:
                                print(f"Warning: Could not read file {full_path}: {e}")
                        else:
                            print(f"Warning: File not found: {full_path}")
                    else:
                        # Handle text or other segment types
                        user_segments.append(segment)
                        # Capture any inline text value if present
                        if 'value' in segment and isinstance(segment['value'], str):
                            user_text_parts.append(segment['value'])
        
        # Extract code snippets from segments
        code_snippets = extract_code_blocks(user_segments)
        
        # Get expected assistant response
        expected_assistant = assistant_msgs[0]['content']
        if isinstance(expected_assistant, list):
            # If content is structured, join text parts
            expected_assistant = ' '.join([
                item.get('text', '') if isinstance(item, dict) else str(item)
                for item in expected_assistant
            ])
        
        # Build a minimal user prompt (one sentence) from user text parts, without leaking expected answer
        user_text_prompt = ' '.join([p.strip() for p in user_text_parts if p and isinstance(p, str)]).strip()

        test_case = TestCase(
            id=raw_test['id'],
            # Use the user's text as the task to avoid leaking 'outcome' details
            task=user_text_prompt or '',
            user_segments=user_segments,
            expected_assistant_raw=expected_assistant,
            guideline_paths=guideline_paths,
            code_snippets=code_snippets,
            outcome=raw_test['outcome'],
            grader=raw_test.get('grader', global_grader)  # Use test-specific grader or global default
        )
        
        test_cases.append(test_case)
    
    return test_cases


def build_prompt_inputs(test_case: TestCase, repo_root: Path) -> dict:
    """
    Build prompt inputs from a test case without leaking expected assistant response.
    
    Args:
        test_case: The test case to build inputs for
        repo_root: Repository root for resolving paths
    
    Returns:
        Dictionary with task, guidelines, and code for the model
    """
    # Collect guideline content
    guidelines = []
    for path in test_case.guideline_paths:
        full_path = repo_root / path
        if full_path.exists():
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    guidelines.append(f"=== {path} ===\n{content}")
            except Exception as e:
                print(f"Warning: Could not read guideline file {full_path}: {e}")
    
    # Collect code snippets and text content
    code_parts = []
    text_parts = []
    
    for segment in test_case.user_segments:
        if segment.get('type') == 'file':
            code_parts.append(f"=== {segment.get('path', 'file')} ===\n{segment.get('text', '')}")
        elif segment.get('type') == 'text':
            text_parts.append(segment.get('value', ''))
    
    # Add extracted code snippets
    code_parts.extend(test_case.code_snippets)
    
    return {
        'task': test_case.task,
        'guidelines': '\n\n'.join(guidelines),
        'code': '\n\n'.join(code_parts),
        'context': '\n\n'.join(text_parts)
    }
