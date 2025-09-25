# BbEval

A lightweight black-box agent evaluator using YAML specifications to score task completion.

## Installation and Setup

### Installation for End Users

This is the recommended method for users who want to use `bbeval` as a command-line tool.

1.  **Ensure you have `uv` installed.** If you don't, you can install them via pip:
    ```bash
    pip install uv
    ```

2.  **Install `bbeval`:**
    ```bash
    uv tool install bbeval
    ```

    Alternatively, if you want the latest (unstable) version:
    ```bash
    uv tool install "git+https://github.com/EntityProcess/bbeval.git"
    ```

3.  **Verify the installation:**
    After installation, the `bbeval` command will be available in your terminal. You can verify it by running:
    ```bash
    bbeval --help
    ```

### Local Development Setup

Follow these steps if you want to contribute to the `bbeval` project itself. This workflow uses a virtual environment and an **editable install**, which means changes you make to the source code are immediately available without reinstalling.

1.  **Clone the repository and navigate into it:**

    ```bash
    git clone https://github.com/entityprocess/bbeval.git
    cd bbeval
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    # Create the virtual environment
    uv venv

    # Activate it (macOS/Linux)
    source .venv/bin/activate

    # Activate it (Windows PowerShell)
    .venv\Scripts\Activate.ps1
    ```

3.  **Perform an editable install with development dependencies:**
    
	This command installs `bbeval` in editable (`-e`) mode and includes the extra tools needed for development and testing (`[dev]`).

    ```bash
    # For non-Windows or if you don't need VS Code focus functionality
    uv pip install -e ".[dev]"
    
    # For Windows users who want the VS Code focus functionality
    uv pip install -e ".[dev,windows]"
    ```

    **Note:** The `windows` optional dependency includes `pywin32` and `psutil`, which are needed for the `--focus` flag with the `open_vscode_workspace.py` script. Without them, the script will work but skip the window focusing feature.

You are now ready to start development. You can run the tool with `bbeval`, edit the code in `src/`, and run tests with `pytest`.

### Environment Setup

1. **Configure environment variables:**
   - Copy [.env.template](/docs/examples/simple/.env.template) to `.env` in your project root
   - Fill in your API keys, endpoints, and other configuration values

2. **Set up targets:**
   - Copy [targets.yaml](/docs/examples/simple/.bbeval/targets.yaml) to `.bbeval/targets.yaml`
   - Update the environment variable names in targets.yaml to match those defined in your `.env` file

## Quick start

**Run eval (target auto-selected from test file or CLI override):**
```powershell
# If your test.yaml contains "target: azure_base", it will be used automatically
bbeval "c:/path/to/test.yaml"

# Override the test file's target with CLI flag
bbeval --target vscode_projectx "c:/path/to/test.yaml"
```

**Run a specific test case with custom targets path:**
```powershell
bbeval --target vscode_projectx --targets "c:/path/to/targets.yaml" --test-id "my-test-case" "c:/path/to/test.yaml"
```

### Command Line Options

- `test_file`: Path to test YAML file (required, positional argument)
- `--target TARGET`: Execution target name from targets.yaml (overrides target specified in test file)
- `--targets TARGETS`: Path to targets.yaml file (default: ./.bbeval/targets.yaml)
- `--test-id TEST_ID`: Run only the test case with this specific ID
- `--out OUTPUT_FILE`: Output JSONL file path (default: results/{testname}_{timestamp}.jsonl)
- `--dry-run`: Run with mock model for testing
- `--agent-timeout SECONDS`: Timeout in seconds for agent response polling (default: 120)
- `--max-retries COUNT`: Maximum number of retries for timeout cases (default: 2)
- `--verbose`: Verbose output

### Target Selection Priority

The CLI determines which execution target to use with the following precedence:

1. **CLI flag override**: `--target my_target` (when provided and not 'default')
2. **Test file specification**: `target: my_target` key in the .test.yaml file
3. **Default fallback**: Uses the 'default' target (original behavior)

This allows test files to specify their preferred target while still allowing command-line overrides for flexibility, and maintains backward compatibility with existing workflows.

Output goes to `.bbeval/results/{testname}_{timestamp}.jsonl` unless `--out` is provided.

### Tips for VS Code Copilot Evals

**Workspace Switching:** The runner automatically switches to the target workspace when running evals. Make sure you're not actively using another VS Code instance, as this could cause prompts to be injected into the wrong workspace.

**Recommended Models:** Use *Claude Sonnet 4* or *Grok Code Fast 1* for best results, as these models are more consistent in following instruction chains.

## Requirements

- Python 3.10+ on PATH
- Evaluator location: `scripts/agent-eval/`
- `.env` for credentials/targets (recommended)

Environment keys (configured via targets.yaml):
- Azure: Set environment variables specified in your target's `settings.endpoint`, `settings.api_key`, and `settings.model`
- Anthropic: Set environment variables specified in your target's `settings.api_key` and `settings.model`
- VS Code: Set environment variable specified in your target's `settings.workspace_env_var` → `.code-workspace` path

## Targets and Environment Variables

Execution targets in `.bbeval/targets.yaml` decouple tests from providers/settings and provide flexible environment variable mapping.

### Target Configuration Structure

Each target specifies:
- `name`: Unique identifier for the target
- `provider`: The model provider (`azure`, `anthropic`, `vscode`, or `mock`)
- `settings`: Environment variable names to use for this target

### Examples

**Azure targets:**
```yaml
- name: azure_base
  provider: azure
  settings:
    endpoint: "AZURE_OPEN_AI_ENDPOINT"
    api_key: "AZURE_OPEN_AI_API_KEY"
    model: "LLM_MODEL"
```

**Anthropic targets:**
```yaml
- name: anthropic_base
  provider: anthropic
  settings:
    api_key: "ANTHROPIC_API_KEY"
    model: "LLM_MODEL"
```

**VS Code targets:**
```yaml
- name: vscode_projectx
  provider: vscode
  settings:
    workspace_env_var: "EVAL_PROJECTX_WORKSPACE_PATH"
```

## Timeout handling and retries

When using VS Code or other AI agents that may experience timeouts, the evaluator includes automatic retry functionality:

- **Timeout detection**: Automatically detects when agents timeout (based on file creation status rather than response parsing)
- **Automatic retries**: When a timeout occurs, the same test case is retried up to `--max-retries` times (default: 2)
- **Retry behavior**: Only timeouts trigger retries; other errors proceed to the next test case
- **Timeout configuration**: Use `--agent-timeout` to adjust how long to wait for agent responses

Example with custom timeout settings:
```
bbeval evals/projectx/example.test.yaml --target vscode_projectx --agent-timeout 180 --max-retries 3
```

## How the evals work

For each testcase in a `.test.yaml` file:
1) Parse YAML; collect only user messages (inline text and referenced files)
2) Extract code blocks from text for structured prompting
3) Select a domain-specific DSPy Signature; generate a candidate answer via provider/model
4) Score against the hidden expected answer (the expected answer is never included in prompts)
5) Append a JSONL line and print a summary

### VS Code Copilot target

- Opens your configured workspace (`PROJECTX_WORKSPACE_PATH`) then runs: `code chat -r "{prompt}"`.
- The prompt is built from the `.test.yaml` user content (task, files, code blocks); the expected assistant answer is never included.
- Copilot is instructed to write its final answer to `.bbeval/vscode-copilot/{test-case-id}.res.md`.

### Prompt file creation

When using VS Code targets (or dry-run mode), the evaluator creates individual prompt files for each test case:

- **Location**: `.bbeval/vscode-copilot/`
- **Naming**: `{test-case-id}.req.md`
- **Format**: Contains instruction file references, reply path, and the question/task

## Scoring and outputs

Run with `--verbose` to print stack traces on errors.

Scoring:
- Aspects = bullet/numbered lines extracted from expected assistant answer (normalized)
- Match by token overlap (case-insensitive)
- Score = hits / total aspects; report `hits`, `misses`, `expected_aspect_count`

Output file:
- Default: `.bbeval/results/{testname}_{YYYYMMDD_HHMMSS}.jsonl` (or use `--out`)
- Fields: `test_id`, `score`, `hits`, `misses`, `model_answer`, `expected_aspect_count`, `provider`, `model`, `timestamp`

## Troubleshooting

### Installation Issues

**Problem**: `uv tool install bbeval` installs an older version despite a newer version being available on PyPI.

**Solution**: Clear the uv cache and reinstall:
```bash
uv cache clean
uv tool uninstall bbeval
uv tool install bbeval
```

This forces uv to fetch fresh package metadata from PyPI instead of using potentially stale cached information.

### Troubleshooting Local Development

**Windows: "Focus requested but win32 modules not available" error:**

If you encounter this error when using the `--focus` flag with VS Code workspace opening:

1. Ensure you're in the activated virtual environment:
   ```bash
   # Check if you're in the virtual environment
   python -c "import sys; print(sys.executable)"
   # Should show a path containing .venv
   ```

2. Install the required Windows modules in your virtual environment:
   ```bash
   # Option 1: Reinstall with Windows dependencies
   uv pip install -e ".[dev,windows]"
   
   # Option 2: Install Windows dependencies separately
   uv pip install pywin32 psutil
   ```

3. If installation fails with permission errors, try:
   ```bash
   uv pip install --target .venv\Lib\site-packages pywin32 psutil
   ```

**Virtual environment not activating properly:**

- On Windows PowerShell, you may need to enable script execution:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
