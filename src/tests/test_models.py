#!/usr/bin/env python3
"""
Unit tests for bbeval.models module.

Tests the VSCodeCopilot helper methods.
"""

import unittest
from unittest.mock import Mock, patch
import tempfile
import shutil
import os
from pathlib import Path

from bbeval.models import VSCodeCopilot, AgentTimeoutError


class TestVSCodeCopilotHelpers(unittest.TestCase):
    """Test cases for the VSCodeCopilot helper methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_path = os.path.join(self.temp_dir, "test.code-workspace")
        # Create a dummy workspace file
        with open(self.workspace_path, 'w') as f:
            f.write('{"folders": []}')
        
        self.vscode_copilot = VSCodeCopilot(
            workspace_path=self.workspace_path,
            polling_timeout=10
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_build_mandatory_preread_block_empty_files(self):
        """Test building preread block with no instruction files."""
        result = self.vscode_copilot._build_mandatory_preread_block([])
        self.assertEqual(result, "")
    
    def test_build_mandatory_preread_block_single_file(self):
        """Test building preread block with a single instruction file."""
        instruction_files = ["/path/to/file1.instructions.md"]
        result = self.vscode_copilot._build_mandatory_preread_block(instruction_files)
        
        self.assertIn("## 1. Mandatory Pre-Read", result)
        self.assertIn("`#file:/path/to/file1.instructions.md`", result)
        self.assertIn("INSTRUCTIONS_READ: [file1.instructions.md]", result)
        self.assertIn("SHA256=<hex>", result)
    
    def test_build_mandatory_preread_block_multiple_files(self):
        """Test building preread block with multiple instruction files."""
        instruction_files = [
            "/path/to/file1.instructions.md",
            "/path/to/file2.instructions.md"
        ]
        result = self.vscode_copilot._build_mandatory_preread_block(instruction_files)
        
        self.assertIn("## 1. Mandatory Pre-Read", result)
        self.assertIn("`#file:/path/to/file1.instructions.md`", result)
        self.assertIn("`#file:/path/to/file2.instructions.md`", result)
        self.assertIn("INSTRUCTIONS_READ: [file1.instructions.md]", result)
        self.assertIn("INSTRUCTIONS_READ: [file2.instructions.md]", result)
    
    def test_prepare_session_files(self):
        """Test session file preparation."""
        test_case_id = "test_case_123"
        session_dir, request_file, reply_tmp, reply_final = self.vscode_copilot._prepare_session_files(test_case_id)
        
        # Check that paths are correctly constructed
        self.assertTrue(session_dir.exists())
        self.assertEqual(request_file.name, f"{test_case_id}.req.md")
        self.assertEqual(reply_tmp.name, f"{test_case_id}.res.tmp.md")
        self.assertEqual(reply_final.name, f"{test_case_id}.res.md")
        
        # Check that files are created in the repo root's .bbeval/vscode-copilot directory
        expected_base_dir = Path.cwd() / '.bbeval' / 'vscode-copilot'
        self.assertTrue(str(session_dir).startswith(str(expected_base_dir)))
    
    def test_prepare_session_files_default_id(self):
        """Test session file preparation with default test case ID."""
        session_dir, request_file, reply_tmp, reply_final = self.vscode_copilot._prepare_session_files(None)
        
        self.assertEqual(request_file.name, "default.req.md")
        self.assertEqual(reply_tmp.name, "default.res.tmp.md")
        self.assertEqual(reply_final.name, "default.res.md")
    
    @patch('bbeval.models.subprocess.run')
    def test_execute_vscode_command_success(self, mock_subprocess):
        """Test successful VS Code command execution."""
        # Setup mock subprocess result
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        # Create test files
        session_dir, request_file, reply_tmp, reply_final = self.vscode_copilot._prepare_session_files("test")
        request_file.write_text("test request")
        reply_final.write_text("test response")
        
        result = self.vscode_copilot._execute_vscode_command_and_poll(
            request_file, reply_final, reply_tmp, session_dir, "test"
        )
        
        self.assertEqual(result, "test response")
        mock_subprocess.assert_called_once()
    
    @patch('bbeval.models.subprocess.run')
    def test_execute_vscode_command_timeout(self, mock_subprocess):
        """Test VS Code command execution with timeout."""
        # Setup mock subprocess result
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        # Create test files but don't create the final response file
        session_dir, request_file, reply_tmp, reply_final = self.vscode_copilot._prepare_session_files("test")
        request_file.write_text("test request")
        
        # Should raise AgentTimeoutError since no response file is created
        with self.assertRaises(AgentTimeoutError):
            self.vscode_copilot._execute_vscode_command_and_poll(
                request_file, reply_final, reply_tmp, session_dir, "test"
            )


if __name__ == '__main__':
    unittest.main(verbosity=2)