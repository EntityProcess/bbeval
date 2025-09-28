"""
Model Provider Abstractions

Provides a clean interface for different LLM providers (Anthropic, Azure OpenAI, VS Code)
through DSPy model abstractions.
"""

import os
import subprocess
import json
import uuid
from typing import Optional, Dict, Any, Tuple
import dspy
from pathlib import Path

def focus_vscode_workspace(workspace_env_var: str, verbose: bool = True) -> bool:
    """Focus the VS Code workspace."""
    if not workspace_env_var:
        return False
    workspace_path = os.getenv(workspace_env_var)
    if not workspace_path:
        if verbose:
            print(f"  Warning: Environment variable '{workspace_env_var}' is not set for focusing.")
        return False
    try:
        from .open_vscode_workspace import open_and_focus_workspace
        success = open_and_focus_workspace(workspace_path, focus=True)
        if success and verbose:
            print("  VS Code workspace focused successfully.")
        return success
    except Exception as e:
        if verbose:
            print(f"  Warning: Failed to focus VS Code workspace: {e}")
    return False

class StandardLM(dspy.LM):
    """Wrapper for standard DSPy models that implements the polymorphic prediction interface."""
    
    def __init__(self, lm_instance):
        # Don't call super().__init__() to avoid recreating the model with validation
        # Instead, copy all attributes from the wrapped instance
        self._wrapped_lm = lm_instance
        
        # Copy essential attributes that dspy.LM would normally set
        for attr in dir(lm_instance):
            if not attr.startswith('_') and not callable(getattr(lm_instance, attr)):
                setattr(self, attr, getattr(lm_instance, attr))
        
        # Ensure we have the essential dspy.LM methods
        self.__dict__.update(lm_instance.__dict__)
    
    def execute_prediction(self, predictor_module, **kwargs) -> dspy.Prediction:
        """Executes a prediction using the standard dspy.Predict module."""
        return predictor_module.predictor(**kwargs)
    
    def forward(self, *args, **kwargs):
        """Delegate forward calls to the wrapped model."""
        return self._wrapped_lm.forward(*args, **kwargs)

class AgentTimeoutError(Exception):
    """Custom exception raised when the agent response times out."""
    pass

class MockModel(dspy.BaseLM):
    """Mock model for testing and dry runs."""
    
    def __init__(self, response: str = None, **kwargs):
        super().__init__(model="mock", **kwargs)
        # Default mock response in JSON format that matches the expected output
        if response is None:
            self.response = '{"review": "Mock PowerShell review: This is a test response showing that the evaluator is working correctly. Issues identified: 1. Use Write-Error instead of throwing exceptions. 2. Add proper error handling. 3. Consider using approved verbs."}'
        else:
            self.response = response
    
    def forward(self, prompt: str = None, messages=None, **kwargs):
        """Return a mock response in OpenAI-compatible format."""
        # Simple mock response that mimics OpenAI response structure
        from types import SimpleNamespace
        
        # Check if this is being used for judging based on the prompt content
        response_content = self.response
        if prompt and any(judge_keyword in str(prompt).lower() for judge_keyword in ['score', 'reasoning', 'judge', 'comparison']):
            # This looks like a judge request, provide appropriate mock judge response
            response_content = '{"score": "0.85", "reasoning": "Mock judge evaluation: The code demonstrates good practices and follows most conventions. This is a test response from the mock judge model."}'
        elif messages and any(judge_keyword in str(messages).lower() for judge_keyword in ['score', 'reasoning', 'judge', 'comparison']):
            # This looks like a judge request via messages, provide appropriate mock judge response
            response_content = '{"score": "0.85", "reasoning": "Mock judge evaluation: The code demonstrates good practices and follows most conventions. This is a test response from the mock judge model."}'
        else:
            # For regular signatures (CodeGeneration, CodeReview, KnowledgeQuery), use "answer" field
            response_content = '{"answer": "Mock response: This is a test answer showing that the evaluator is working correctly. The implementation follows best practices and includes proper error handling."}'
        
        # Usage object that can be converted to dict
        class MockUsage:
            def __init__(self):
                self.prompt_tokens = 100
                self.completion_tokens = 50
                self.total_tokens = 150
            
            def __iter__(self):
                """Make the usage object iterable for dict() conversion."""
                yield ('prompt_tokens', self.prompt_tokens)
                yield ('completion_tokens', self.completion_tokens) 
                yield ('total_tokens', self.total_tokens)
        
        response = SimpleNamespace()
        response.choices = [SimpleNamespace()]
        response.choices[0].message = SimpleNamespace()
        response.choices[0].message.content = response_content
        response.choices[0].message.role = "assistant"
        response.choices[0].finish_reason = "stop"
        response.choices[0].index = 0
        
        # Usage object that DSPy can convert to dict
        response.usage = MockUsage()
        
        response.id = "mock-response"
        response.object = "chat.completion"
        response.created = 1234567890
        response.model = "mock-model"
        
        return response
    
    def execute_prediction(self, predictor_module, **kwargs) -> dspy.Prediction:
        """Executes a prediction using the standard dspy.Predict module."""
        return predictor_module.predictor(**kwargs)

class VSCodeCopilot(dspy.BaseLM):
    """VS Code Copilot model that shells out to VS Code with a prompt.

    Copilot in VS Code does not expose a base model selector and has no stable
    chat CLI. We pass the prompt to VS Code and instruct Copilot to write the
    final answer to a file we can read back.
    
    Uses session-based temporary files to avoid race conditions between tests.
    """
    
    def __init__(self, workspace_path: str, workspace_env_var: str, polling_timeout: int = 120, **kwargs):
        super().__init__(model="vscode-copilot", **kwargs)
        self.workspace_path = workspace_path
        self.workspace_env_var = workspace_env_var
        self.polling_timeout = polling_timeout
        # Generate a unique session ID for this model instance
        self.session_id = str(uuid.uuid4())[:8]
        
        # Validate that workspace file exists
        if not os.path.exists(workspace_path):
            raise ValueError(f"Workspace file not found: {workspace_path}")
    
    def _build_mandatory_preread_block(self, instruction_files: list) -> str:
        """
        Build the mandatory pre-read instruction block for instruction files.
        
        Args:
            instruction_files: List of instruction file paths
            
        Returns:
            Formatted pre-read instruction block
        """
        if not instruction_files:
            return ""
            
        # Create file list for the consolidated instruction
        file_list = []
        token_list = []
        for i, instruction_file in enumerate(instruction_files, 1):
            file_name = instruction_file.split('/')[-1]  # Get just the filename
            file_list.append(f"`#file:{instruction_file}`")
            token_list.append(f"INSTRUCTIONS_READ: [{file_name}](http://_vscodecontentref_/{i}) SHA256=<hex>")
        
        # Create single consolidated pre-read instruction
        files_text = ", ".join(file_list)
        tokens_text = "\n".join(token_list)
        
        consolidated_instruction = (
            f"**MANDATORY STEP — PRE-READ ALL INSTRUCTIONS**: Before any analysis, open and read all instruction files: {files_text}. "
            f"Locate each file using its filename or an explicit filesystem path (do not rely on a workspace text search). "
            f"After reading each file, compute its SHA256 hash using this PowerShell command: "
            f"`Get-FileHash -Algorithm SHA256 -LiteralPath '<file-path>' | Select-Object -ExpandProperty Hash`. "
            f"Then include, at the top of your reply, these exact tokens on separate lines:\n\n"
            f"{tokens_text}\n\n"
            f"Replace `<hex>` with the actual SHA256 hash value computed from the PowerShell command. "
            f"Follow each token with a one-paragraph summary of the specific rules applied from that file. "
            f"If any file is missing, fail with ERROR: missing-file <filename> and stop."
        )
        
        return f"## 1. Mandatory Pre-Read\n\n{consolidated_instruction}\n\n"
    
    def _prepare_session_files(self, test_case_id: str) -> Tuple[Path, Path, Path, Path]:
        """
        Prepare session-specific directories and file paths.
        
        Args:
            test_case_id: Identifier for the test case
            
        Returns:
            Tuple of (session_dir, request_file_path, reply_tmp_path, reply_final_path)
        """
        # Create session-specific directory rooted at the repository root (cwd)
        # so artifacts are generated alongside the running repo rather than the
        # VS Code workspace file's directory.
        output_dir = Path.cwd() / '.bbeval' / 'vscode-copilot'
        session_dir = output_dir / self.session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Use provided test case ID or default
        if test_case_id is None:
            test_case_id = 'default'
        request_file_path = session_dir / f'{test_case_id}.req.md'
        reply_tmp_path = session_dir / f'{test_case_id}.res.tmp.md'
        reply_final_path = session_dir / f'{test_case_id}.res.md'

        # Clear any existing files
        for file_path in [reply_tmp_path, reply_final_path]:
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception:
                pass
                
        return session_dir, request_file_path, reply_tmp_path, reply_final_path
    
    def _execute_vscode_command_and_poll(self, request_file_path: Path, reply_final_path: Path, 
                                       reply_tmp_path: Path, session_dir: Path, 
                                       test_case_id: str) -> str:
        """
        Execute the VS Code command and poll for the response file.
        
        Args:
            request_file_path: Path to the request file
            reply_final_path: Path to the final response file
            reply_tmp_path: Path to the temporary response file
            session_dir: Session directory path
            test_case_id: Test case identifier
            
        Returns:
            Response content from VS Code
            
        Raises:
            AgentTimeoutError: If polling times out
        """
        import time
        
        try:
            # Build the VS Code CLI chat command directly (avoid dependency on pwsh)
            # We still embed a PowerShell snippet inside the Copilot chat instruction so Copilot
            # can read the request file contents within VS Code context.
            inner_powershell_command = f'Get-Content -Raw -LiteralPath "{request_file_path.resolve()}"'
            chat_instruction = f"run command <powershell>{inner_powershell_command}</powershell> and follow its instructions."

            # Prefer calling the VS Code CLI directly. This removes requirement for 'pwsh'.
            # The 'code' executable must be on PATH (VS Code's 'Shell Command: Install 'code'' setting on macOS,
            # or automatically available on Windows after install).
            import shutil
            code_cli = shutil.which('code')
            if not code_cli:
                return ("Error: VS Code CLI 'code' was not found on PATH. "
                        "Ensure VS Code is installed and the command line launcher is enabled. "
                        "On Windows this is usually automatic; on macOS use the Command Palette: 'Shell Command: Install code command'.")

            cmd = [code_cli, 'chat', '-r', chat_instruction]
            
            # Also print a concise summary to stdout
            try:
                print(f"PowerShell + VS Code: {test_case_id}.req.md → {reply_final_path.name} (session: {self.session_id})")
            except Exception:
                pass

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180
            )

            if result is None:
                raise FileNotFoundError("PowerShell not found or failed to execute")
            if result.returncode != 0:
                # Build a richer error including command details for troubleshooting
                try:
                    (session_dir / 'last_cli_stderr.log').write_text(result.stderr or '', encoding='utf-8')
                except Exception:
                    pass
                error_msg = (
                    f"VS Code CLI chat command failed (exit code {result.returncode}).\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Stderr: {result.stderr}\n"
                    f"Request file: {request_file_path.resolve()}\n"
                    f"Note: Full command details saved under {session_dir}"
                )
                return f"Error: {error_msg}"
            else:
                # Poll for final response file based on configured timeout
                max_attempts = self.polling_timeout // 2  # 2 second intervals
                for attempt in range(max_attempts):
                    if reply_final_path.exists():
                        response_content = reply_final_path.read_text(encoding='utf-8').strip()
                        if response_content:
                            return response_content
                    time.sleep(2)
                else:
                    # If loop completes without break - polling timed out
                    print(f"Agent response polling timed out after {self.polling_timeout} seconds")
                    timeout_msg = "No response file was generated by VS Code Copilot - timeout occurred"
                    if reply_final_path.exists():
                        timeout_msg = "Empty response file generated by VS Code Copilot"
                    elif reply_tmp_path.exists():
                        timeout_msg = f"Temporary file exists but was not renamed. Content: {reply_tmp_path.read_text(encoding='utf-8').strip()}"
                    raise AgentTimeoutError(f"Agent polling timeout after {self.polling_timeout} seconds: {timeout_msg}")

        except subprocess.TimeoutExpired:
            print("PowerShell + VS Code command timed out")
            raise AgentTimeoutError("PowerShell + VS Code command timed out")
        except AgentTimeoutError:
            # Re-raise timeout errors to be caught by retry logic
            raise
        except FileNotFoundError:
            return "Error: Failed to execute VS Code CLI. Confirm that 'code' is available on PATH."
        except Exception as e:
            return f"Error: {str(e)}"

    def forward(self, prompt: str = None, messages=None, test_case_id: str = None, instruction_files: list = None, task: str = None, **kwargs):
        """Create a request file and execute VS Code with PowerShell command to read it."""
        focus_vscode_workspace(self.workspace_env_var, verbose=True)
        
        from types import SimpleNamespace

        # Extract the actual prompt content
        actual_prompt = self._extract_prompt_content(prompt, messages)
        
        # If task is provided, build a clean prompt with conditional mandatory preread
        if task:
            final_prompt = ""
            
            # Section 1: Add mandatory preread instructions if instruction files are present
            if instruction_files:
                final_prompt += self._build_mandatory_preread_block(instruction_files)
            
            # Section 2: Add the task with clear header
            final_prompt += "## 2. Task\n\n"
            final_prompt += task + "\n\n"
            
            # Use the clean final prompt instead of the complex structure
            actual_prompt = final_prompt

        # Prepare session-specific files and directories
        session_dir, request_file_path, reply_tmp_path, reply_final_path = self._prepare_session_files(test_case_id)

        # Create the request file content with file handling instructions
        enhanced_prompt = (
            f"{(actual_prompt or '').strip()}\n\n"
            f"## 3. Output Instructions\n\n"
            f"**IMPORTANT**: Follow these exact steps:\n"
            f"1. Write your complete response to: {reply_tmp_path.resolve()}\n"
            f"2. When completely finished, run this PowerShell command to signal completion:\n"
            f"   Move-Item -LiteralPath '{reply_tmp_path.resolve()}' -Destination '{reply_final_path.resolve()}'\n\n"
            f"Do not proceed to step 2 until your response is completely written to the temporary file."
        )

        # Write the request file
        try:
            request_file_path.write_text(enhanced_prompt, encoding='utf-8')
        except Exception as e:
            raise RuntimeError(f"Failed to write request file {request_file_path}: {e}")

        # Record raw request metadata for downstream capture in evaluation results
        try:
            self._last_raw_request = {
                'provider': 'vscode',
                'test_case_id': test_case_id,
                'task': task,
                'instruction_files': instruction_files,
                'request_file': str(request_file_path.resolve()),
                'reply_tmp_file': str(reply_tmp_path.resolve()),
                'reply_final_file': str(reply_final_path.resolve()),
                'enhanced_prompt': enhanced_prompt
            }
        except Exception:
            pass

        # Execute the VS Code command and poll for response
        response_content = self._execute_vscode_command_and_poll(
            request_file_path, reply_final_path, reply_tmp_path, session_dir, test_case_id
        )
        
        # Create mock usage object
        class MockUsage:
            def __init__(self):
                self.prompt_tokens = len(actual_prompt.split()) if actual_prompt else 0
                self.completion_tokens = len(response_content.split())
                self.total_tokens = self.prompt_tokens + self.completion_tokens
            
            def __iter__(self):
                yield ('prompt_tokens', self.prompt_tokens)
                yield ('completion_tokens', self.completion_tokens) 
                yield ('total_tokens', self.total_tokens)
        
        # Wrap content into a JSON payload so DSPy can extract the 'review' field
        final_content = json.dumps({"review": response_content})

        # Create OpenAI-compatible response structure
        response = SimpleNamespace()
        response.choices = [SimpleNamespace()]
        response.choices[0].message = SimpleNamespace()
        response.choices[0].message.content = final_content
        response.choices[0].message.role = "assistant"
        response.choices[0].finish_reason = "stop"
        response.choices[0].index = 0
        
        response.usage = MockUsage()
        response.id = "vscode-copilot-response"
        response.object = "chat.completion"
        response.created = 1234567890
        response.model = "vscode-copilot"
        
        return response
    
    def _extract_prompt_content(self, prompt: str = None, messages=None) -> str:
        """Extract the actual prompt content from DSPy's input format."""
        if prompt:
            return prompt
        
        if messages:
            # Attempt to extract task content from DSPy's structured messages
            # Handle multiple possible field names for robustness
            try:
                combined = ''
                if isinstance(messages, list):
                    for msg in messages:
                        part = getattr(msg, 'content', None)
                        if part is None:
                            part = str(msg)
                        combined += (str(part) + '\n')
                else:
                    combined = str(messages)
                import re as _re
                
                # Try multiple patterns for different field names
                # Note: We don't include 'expected_outcome' or 'outcome' to avoid leaking the answer
                field_patterns = [
                    r"\[\[\s*##\s*request\s*##\s*\]\]\s*(.*?)\s*(?=\[\[|$)",        # EvalSignature field
                    r"\[\[\s*##\s*task_requirements\s*##\s*\]\]\s*(.*?)\s*(?=\[\[|$)", # QualityGrader field
                    r"\[\[\s*##\s*task\s*##\s*\]\]\s*(.*?)\s*(?=\[\[|$)"           # Legacy field
                ]
                
                # Try each pattern and return the first valid match
                for pattern in field_patterns:
                    matches = list(_re.finditer(pattern, combined, flags=_re.IGNORECASE | _re.DOTALL))
                    # Prefer the last filled block that does not contain placeholders
                    for m in reversed(matches):
                        candidate = m.group(1).strip()
                        if candidate and not any(placeholder in candidate for placeholder in ['{task}', '{task_requirements}', '{request}']):
                            return candidate
                
                # Heuristic fallback: pick the first question-like line
                for line in combined.splitlines():
                    line_s = line.strip()
                    if not line_s:
                        continue
                    if '{' in line_s or '}}' in line_s or '[[' in line_s:
                        continue
                    if line_s.endswith('?'):
                        return line_s
            except Exception:
                pass
            # Fallback to previous behavior
            if isinstance(messages, list):
                content_parts = []
                for msg in messages:
                    if hasattr(msg, 'content'):
                        content_parts.append(str(msg.content))
                    else:
                        content_parts.append(str(msg))
                return '\n'.join(content_parts)
            else:
                return str(messages)
        
        return "No prompt provided"
    
    def execute_prediction(self, predictor_module, test_case_id: str = None, **kwargs) -> dspy.Prediction:
        """Executes a prediction using VSCodeCopilot-specific logic."""
        # For VSCodeCopilot, we need to call the model directly with test_case_id
        # For VSCodeCopilot, use a simple prompt structure since mandatory preread handles instructions
        # Just pass the task directly, don't build complex prompt sections
        
        # Try multiple field names to find the task content (robust handling of field renaming)
        # Note: We don't use 'expected_outcome' or 'outcome' to avoid leaking the answer
        task_content = (kwargs.get('request', '') or           # EvalSignature field
                       kwargs.get('task_requirements', '') or  # QualityGrader field  
                       kwargs.get('task', '') or               # Legacy field
                       '')
        
        # Get instruction files from guideline_paths
        instruction_files = kwargs.get('guideline_paths', [])
        
        # Call the VSCodeCopilot model directly with test_case_id, instruction files, and task
        response = self.forward(prompt=task_content, test_case_id=test_case_id, instruction_files=instruction_files, task=task_content)
        
        # Extract the answer from the response
        import json
        try:
            response_data = json.loads(response.choices[0].message.content)
            # Try multiple field names that might contain the actual response
            answer_content = (response_data.get('answer') or 
                            response_data.get('review') or 
                            response.choices[0].message.content)
        except (json.JSONDecodeError, KeyError, AttributeError):
            answer_content = response.choices[0].message.content
        
        # Return a result object that matches the expected signature
        from types import SimpleNamespace
        result = SimpleNamespace()
        result.answer = answer_content
        return result

def create_model(provider: str, model: str, settings: Dict[str, Any] = None, **kwargs):
    """
    Factory function to create model instances based on provider.
    
    Args:
        provider: 'anthropic', 'azure', 'vscode', or 'mock'
        model: Model name/deployment
        settings: Provider-specific settings from targets.yaml
        **kwargs: Additional model configuration
    
    Returns:
        Configured model instance
    """
    provider = provider.lower()
    settings = settings or {}
    
    # Extract optional cache control
    disable_cache: bool = bool(kwargs.pop('disable_cache', False))

    if provider == "vscode":
        # Get environment variable name from target settings
        workspace_env_var = settings.get('workspace_env_var')
        if not workspace_env_var:
            raise ValueError("VS Code 'settings' in targets.yaml must define 'workspace_env_var'")
        
        # Fetch value from the environment using the name specified in the YAML
        workspace_path = os.getenv(workspace_env_var)
        if not workspace_path:
            raise ValueError(f"Environment variable '{workspace_env_var}' is not set.")
        
        # Extract polling timeout from kwargs, default to 120 seconds
        polling_timeout = kwargs.pop('polling_timeout', 120)
        return VSCodeCopilot(workspace_path=workspace_path, workspace_env_var=workspace_env_var, polling_timeout=polling_timeout, **kwargs)
    elif provider == "anthropic":
        # Get environment variable names from target settings
        api_key_var = settings.get('api_key')
        
        if not api_key_var:
            raise ValueError("Anthropic 'settings' in targets.yaml must define 'api_key'")
        
        # Fetch values from the environment using the names specified in the YAML
        api_key = kwargs.get('api_key') or os.getenv(api_key_var)
        
        if not api_key:
            raise ValueError(f"Environment variable '{api_key_var}' is not set.")
        
        if disable_cache:
            lm = dspy.LM(f"anthropic/{model}", api_key=api_key, cache=False)
        else:
            lm = dspy.LM(f"anthropic/{model}", api_key=api_key)
        return StandardLM(lm)
    elif provider == "azure":
        # Get environment variable names from target settings
        endpoint_var = settings.get('endpoint')
        api_key_var = settings.get('api_key')

        if not endpoint_var or not api_key_var:
            raise ValueError("Azure 'settings' in targets.yaml must define 'endpoint' and 'api_key'")

        # Fetch values from the environment using the names specified in the YAML
        endpoint = kwargs.get('endpoint') or os.getenv(endpoint_var)
        api_key = kwargs.get('api_key') or os.getenv(api_key_var)

        if not endpoint:
            raise ValueError(f"Environment variable '{endpoint_var}' is not set.")
        if not api_key:
            raise ValueError(f"Environment variable '{api_key_var}' is not set.")
        # Handle reasoning models special requirements (GPT-5, o1, o3, etc.)
        reasoning_models = ["gpt-5", "o1", "o3"]
        is_reasoning_model = any(reasoning_model in model.lower() for reasoning_model in reasoning_models)
        
        if is_reasoning_model:
            if disable_cache:
                lm = dspy.LM(f"azure/{model}", api_key=api_key, api_base=endpoint, temperature=1.0, max_tokens=16000, cache=False)
            else:
                lm = dspy.LM(f"azure/{model}", api_key=api_key, api_base=endpoint, temperature=1.0, max_tokens=16000)
        else:
            if disable_cache:
                lm = dspy.LM(f"azure/{model}", api_key=api_key, api_base=endpoint, cache=False)
            else:
                lm = dspy.LM(f"azure/{model}", api_key=api_key, api_base=endpoint)
        return StandardLM(lm)
    elif provider == "mock":
        return MockModel(**kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}. Supported: anthropic, azure, vscode, mock")

def configure_dspy_model(provider: str, model: str, settings: Dict[str, Any] = None, **kwargs):
    """
    Configure DSPy with the specified model.
    
    Args:
        provider: Model provider
        model: Model name
        settings: Provider-specific settings
        **kwargs: Additional configuration
    """
    model_instance = create_model(provider, model, settings, **kwargs)
    dspy.settings.configure(lm=model_instance)
