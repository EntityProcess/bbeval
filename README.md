# SpecEval

Evaluates AI model answers against `.test.yaml` specs. Results are saved as JSONL with deterministic scoring and a run summary.

## Installation and Setup

### Development Installation

```powershell
# Install in development mode (creates proper package structure)
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Environment Setup

1. **Configure environment variables:**
   - Copy [.env.example](/docs/examples/simple/.env.example) to `.env` in your project root
   - Fill in your API keys, endpoints, and other configuration values

2. **Set up targets:**
   - Copy [targets.yaml](/docs/examples/simple/.speceval/targets.yaml) to `.speceval/targets.yaml`
   - Update the environment variable names in targets.yaml to match those defined in your `.env` file

## Quick start

**Run eval with default target (Azure):**
```powershell
# Using the CLI command
speceval --tests "c:/path/to/test.yaml"

# Or using the Python module
python -m speceval.cli --tests "c:/path/to/test.yaml"
```

**Run a specific test case:**
```powershell
# Using the CLI command
speceval --target vscode_projectx --targets "c:/path/to/targets.yaml" --tests "c:/path/to/test.yaml" --test-id "my-test-case"

# Or using the Python module
python -m speceval.cli --target vscode_projectx --targets "c:/path/to/targets.yaml" --tests "c:/path/to/test.yaml" --test-id "my-test-case"
```

**Specify a target explicitly:**
```powershell
# Using the CLI command
speceval --target azure_base --tests "c:/path/to/test.yaml"

# Or using the Python module
python -m speceval.cli --target azure_base --tests "c:/path/to/test.yaml"
```

**Run eval with custom targets file and test file:**
```powershell
# Using the CLI command
speceval --target vscode_projectx --targets "c:/path/to/targets.yaml" --tests "c:/path/to/test.yaml"

# Or using the Python module
python -m speceval.cli --target vscode_projectx --targets "c:/path/to/targets.yaml" --tests "c:/path/to/test.yaml"
```

### Command Line Options

- `--target TARGET`: Execution target name from targets.yaml (default: default)
- `--targets TARGETS`: Path to targets.yaml file (default: ./.speceval/targets.yaml)
- `--tests TESTS`: Path to test YAML file (required)
- `--test-id TEST_ID`: Run only the test case with this specific ID
- `--out OUTPUT_FILE`: Output JSONL file path (default: results/{testname}_{timestamp}.jsonl)
- `--dry-run`: Run with mock model for testing
- `--agent-timeout SECONDS`: Timeout in seconds for agent response polling (default: 120)
- `--max-retries COUNT`: Maximum number of retries for timeout cases (default: 2)
- `--verbose`: Verbose output

Output goes to `.speceval/results/{testname}_{timestamp}.jsonl` unless `--out` is provided.

## Requirements

- Python 3.10+ on PATH
- Evaluator location: `scripts/agent-eval/`
- `.env` for credentials/targets (recommended)

Environment keys (configured via targets.yaml):
- Azure: Set environment variables specified in your target's `settings.endpoint`, `settings.api_key`, and `settings.model`
- Anthropic: Set environment variables specified in your target's `settings.api_key` and `settings.model`
- VS Code: Set environment variable specified in your target's `settings.workspace_env_var` → `.code-workspace` path

## Targets and Environment Variables

Execution targets in `.speceval/targets.yaml` decouple tests from providers/settings and provide flexible environment variable mapping.

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
speceval --target vscode_projectx --tests evals/projectx/example.test.yaml --agent-timeout 180 --max-retries 3
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
- Copilot is instructed to write its final answer to `.speceval/vscode-copilot/{test-case-id}.res.md`.

### Prompt file creation

When using VS Code targets (or dry-run mode), the evaluator creates individual prompt files for each test case:

- **Location**: `.speceval/vscode-copilot/`
- **Naming**: `{test-case-id}.req.md`
- **Format**: Contains instruction file references, reply path, and the question/task

## Scoring and outputs

Run with `--verbose` to print stack traces on errors.

Scoring:
- Aspects = bullet/numbered lines extracted from expected assistant answer (normalized)
- Match by token overlap (case-insensitive)
- Score = hits / total aspects; report `hits`, `misses`, `expected_aspect_count`

Output file:
- Default: `.speceval/results/{testname}_{YYYYMMDD_HHMMSS}.jsonl` (or use `--out`)
- Fields: `test_id`, `score`, `hits`, `misses`, `model_answer`, `expected_aspect_count`, `provider`, `model`, `timestamp`

