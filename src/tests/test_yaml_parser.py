#!/usr/bin/env python3
"""
Unit tests for bbeval.yaml_parser module.

Tests YAML parsing and path resolution for instruction files.
"""

import unittest
import tempfile
import shutil
import os
from pathlib import Path
import yaml

from bbeval.yaml_parser import load_testcases, is_guideline_file


class TestYamlParser(unittest.TestCase):
    """Test cases for the yaml_parser module."""
    
    def setUp(self):
        """Set up test fixtures with a temporary directory structure."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_root = Path(self.temp_dir)
        
        # Create directory structure
        self.prompts_dir = self.repo_root / "prompts"
        self.prompts_dir.mkdir()
        
        self.files_dir = self.repo_root / "files"
        self.files_dir.mkdir()
        
        # Create instruction file
        self.instruction_file = self.prompts_dir / "test.instructions.md"
        self.instruction_file.write_text("# Test Instructions\nFollow these rules.")
        
        # Create regular file
        self.regular_file = self.files_dir / "code.py"
        self.regular_file.write_text("print('hello')")
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_is_guideline_file(self):
        """Test guideline file detection."""
        self.assertTrue(is_guideline_file("prompts/test.instructions.md"))
        self.assertTrue(is_guideline_file("foo/instructions/test.md"))
        self.assertTrue(is_guideline_file("prompts/test.prompt.md"))
        self.assertFalse(is_guideline_file("files/code.py"))
        self.assertFalse(is_guideline_file("README.md"))
        # Edge case: file ending with .instructions.md but not in instructions folder
        self.assertTrue(is_guideline_file("anywhere/file.instructions.md"))
    
    def test_guideline_path_is_absolute(self):
        """Test that guideline paths are stored as absolute paths."""
        # Create a test YAML file
        test_yaml_content = {
            'testcases': [
                {
                    'id': 'test-case-1',
                    'outcome': 'Follow the instructions.',
                    'messages': [
                        {
                            'role': 'user',
                            'content': [
                                {'type': 'file', 'value': 'prompts/test.instructions.md'},
                                {'type': 'text', 'value': 'What should I do?'}
                            ]
                        },
                        {
                            'role': 'assistant',
                            'content': 'Follow the instructions.'
                        }
                    ]
                }
            ]
        }
        
        test_yaml_path = self.repo_root / "test.yaml"
        with open(test_yaml_path, 'w') as f:
            yaml.dump(test_yaml_content, f)
        
        # Load test cases
        test_cases = load_testcases(str(test_yaml_path), self.repo_root)
        
        # Verify we got one test case
        self.assertEqual(len(test_cases), 1)
        test_case = test_cases[0]
        
        # Verify guideline_paths contains absolute path
        self.assertEqual(len(test_case.guideline_paths), 1)
        guideline_path = test_case.guideline_paths[0]
        
        # Should be an absolute path
        self.assertTrue(Path(guideline_path).is_absolute(), 
                       f"Expected absolute path but got: {guideline_path}")
        
        # Should point to the correct file
        self.assertEqual(Path(guideline_path), self.instruction_file)
        
        # Verify the file exists
        self.assertTrue(Path(guideline_path).exists())
    
    def test_guideline_path_from_different_cwd(self):
        """Test that guideline paths are correct regardless of current working directory."""
        # Create a test YAML file
        test_yaml_content = {
            'testcases': [
                {
                    'id': 'test-case-1',
                    'outcome': 'Response.',
                    'messages': [
                        {
                            'role': 'user',
                            'content': [
                                {'type': 'file', 'value': 'prompts/test.instructions.md'},
                            ]
                        },
                        {
                            'role': 'assistant',
                            'content': 'Response.'
                        }
                    ]
                }
            ]
        }
        
        test_yaml_path = self.repo_root / "test.yaml"
        with open(test_yaml_path, 'w') as f:
            yaml.dump(test_yaml_content, f)
        
        # Save current working directory
        original_cwd = os.getcwd()
        
        try:
            # Change to a different directory
            os.chdir(self.prompts_dir)
            
            # Load test cases from the different working directory
            test_cases = load_testcases(str(test_yaml_path), self.repo_root)
            
            # Verify guideline path is still correct
            self.assertEqual(len(test_cases), 1)
            guideline_path = test_cases[0].guideline_paths[0]
            
            # Should still be an absolute path pointing to the right file
            self.assertTrue(Path(guideline_path).is_absolute())
            self.assertEqual(Path(guideline_path), self.instruction_file)
            self.assertTrue(Path(guideline_path).exists())
            
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
    
    def test_multiple_guideline_files(self):
        """Test handling of multiple guideline files."""
        # Create additional instruction files
        instruction_file2 = self.prompts_dir / "python.instructions.md"
        instruction_file2.write_text("# Python Rules")
        
        instruction_file3 = self.repo_root / "instructions" / "general.prompt.md"
        instruction_file3.parent.mkdir(exist_ok=True)
        instruction_file3.write_text("# General Prompt")
        
        # Create test YAML
        test_yaml_content = {
            'testcases': [
                {
                    'id': 'test-case-1',
                    'outcome': 'Code generated.',
                    'messages': [
                        {
                            'role': 'user',
                            'content': [
                                {'type': 'file', 'value': 'prompts/test.instructions.md'},
                                {'type': 'file', 'value': 'prompts/python.instructions.md'},
                                {'type': 'file', 'value': 'instructions/general.prompt.md'},
                                {'type': 'text', 'value': 'Generate code.'}
                            ]
                        },
                        {
                            'role': 'assistant',
                            'content': 'Code generated.'
                        }
                    ]
                }
            ]
        }
        
        test_yaml_path = self.repo_root / "test.yaml"
        with open(test_yaml_path, 'w') as f:
            yaml.dump(test_yaml_content, f)
        
        # Load test cases
        test_cases = load_testcases(str(test_yaml_path), self.repo_root)
        
        # Verify all three guideline files are present as absolute paths
        self.assertEqual(len(test_cases), 1)
        guideline_paths = test_cases[0].guideline_paths
        
        self.assertEqual(len(guideline_paths), 3)
        
        # All should be absolute paths
        for path in guideline_paths:
            self.assertTrue(Path(path).is_absolute(), f"Expected absolute path but got: {path}")
            self.assertTrue(Path(path).exists(), f"Path does not exist: {path}")
        
        # Verify they point to the correct files
        self.assertIn(str(self.instruction_file), guideline_paths)
        self.assertIn(str(instruction_file2), guideline_paths)
        self.assertIn(str(instruction_file3), guideline_paths)
    
    def test_regular_files_not_in_guideline_paths(self):
        """Test that regular files are not included in guideline_paths."""
        test_yaml_content = {
            'testcases': [
                {
                    'id': 'test-case-1',
                    'outcome': 'Response.',
                    'messages': [
                        {
                            'role': 'user',
                            'content': [
                                {'type': 'file', 'value': 'files/code.py'},
                                {'type': 'file', 'value': 'prompts/test.instructions.md'},
                            ]
                        },
                        {
                            'role': 'assistant',
                            'content': 'Response.'
                        }
                    ]
                }
            ]
        }
        
        test_yaml_path = self.repo_root / "test.yaml"
        with open(test_yaml_path, 'w') as f:
            yaml.dump(test_yaml_content, f)
        
        # Load test cases
        test_cases = load_testcases(str(test_yaml_path), self.repo_root)
        
        # Verify only instruction file is in guideline_paths
        self.assertEqual(len(test_cases), 1)
        guideline_paths = test_cases[0].guideline_paths
        
        self.assertEqual(len(guideline_paths), 1)
        self.assertEqual(Path(guideline_paths[0]), self.instruction_file)
        
        # Regular file should be in user_segments instead
        user_segments = test_cases[0].user_segments
        file_segments = [s for s in user_segments if s.get('type') == 'file']
        self.assertEqual(len(file_segments), 1)
        self.assertEqual(file_segments[0]['path'], 'files/code.py')

    def test_guideline_resolution_from_nested_eval_directory(self):
        """Guidelines referenced with leading slash should resolve from test directory ancestors."""
        workspace_root = self.repo_root / "nested" / "simple"
        evals_dir = workspace_root / "evals"
        prompts_dir = workspace_root / "prompts"
        evals_dir.mkdir(parents=True)
        prompts_dir.mkdir(parents=True)

        nested_instruction = prompts_dir / "nested.instructions.md"
        nested_instruction.write_text("# Nested Instructions")

        test_yaml_content = {
            'testcases': [
                {
                    'id': 'nested-case',
                    'outcome': 'Follow nested instructions.',
                    'messages': [
                        {
                            'role': 'user',
                            'content': [
                                {'type': 'file', 'value': '/prompts/nested.instructions.md'},
                            ]
                        },
                        {
                            'role': 'assistant',
                            'content': 'Done.'
                        }
                    ]
                }
            ]
        }

        test_yaml_path = evals_dir / "nested.test.yaml"
        with open(test_yaml_path, 'w') as f:
            yaml.dump(test_yaml_content, f)

        original_cwd = os.getcwd()
        try:
            # Simulate running from repo root rather than nested directory
            os.chdir(self.repo_root)
            test_cases = load_testcases(str(test_yaml_path), self.repo_root)
        finally:
            os.chdir(original_cwd)

        self.assertEqual(len(test_cases), 1)
        guideline_path = Path(test_cases[0].guideline_paths[0])
        self.assertEqual(guideline_path, nested_instruction.resolve())
        self.assertTrue(guideline_path.exists())

    def test_guideline_resolution_with_repo_relative_path(self):
        """Guidelines with repo-relative prefixes should resolve correctly."""
        example_root = self.repo_root / "docs" / "examples" / "simple"
        evals_dir = example_root / "evals"
        prompts_dir = example_root / "prompts"
        evals_dir.mkdir(parents=True)
        prompts_dir.mkdir(parents=True)

        example_instruction = prompts_dir / "example.instructions.md"
        example_instruction.write_text("# Example Instructions")

        test_yaml_content = {
            'testcases': [
                {
                    'id': 'repo-relative-case',
                    'outcome': 'Follow repo relative instructions.',
                    'messages': [
                        {
                            'role': 'user',
                            'content': [
                                {'type': 'file', 'value': '/docs/examples/simple/prompts/example.instructions.md'},
                            ]
                        },
                        {
                            'role': 'assistant',
                            'content': 'Done.'
                        }
                    ]
                }
            ]
        }

        test_yaml_path = evals_dir / "repo-relative.test.yaml"
        with open(test_yaml_path, 'w') as f:
            yaml.dump(test_yaml_content, f)

        original_cwd = os.getcwd()
        try:
            os.chdir(evals_dir)
            test_cases = load_testcases(str(test_yaml_path), self.repo_root)
        finally:
            os.chdir(original_cwd)

        self.assertEqual(len(test_cases), 1)
        guideline_path = Path(test_cases[0].guideline_paths[0])
        self.assertEqual(guideline_path, example_instruction.resolve())
        self.assertTrue(guideline_path.exists())


if __name__ == '__main__':
    unittest.main()
