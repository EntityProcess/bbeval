# DSPy Agent Evaluator

Action-based guide to run evals and understand how they work.

Evaluate LLM/Copilot answers against `.test.yaml` specs in `evals/**` with zero leakage of expected answers. Results are saved as JSONL with deterministic scoring and a run summary.

## Installation and Setup

### Development Installation (Recommended)

Open terminal in `scripts/agent-eval` and install in editable mode:

```powershell
# Install in development mode (creates proper package structure)
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Environment Setup

Copy env: `Copy-Item .env.example .env` and set keys

### Running Tests

```powershell
# Run unit tests
pytest

# Run unit tests with coverage
pytest --cov=eval_runner
```

## Quick start (Windows PowerShell)

Run eval with default target (Azure):
```
python -m eval_runner.cli --tests ../../evals/development/powershell.test.yaml
```

Or specify a target explicitly:
```
python -m eval_runner.cli --target azure_base --tests ../../evals/development/powershell.test.yaml
```

Or run via VS Code Copilot:
```
python -m eval_runner.cli --target vscode_cargowise --tests ../../evals/cargowise/base.test.yaml
```
Run a specific test case by ID:
```
python -m eval_runner.cli --tests ../../evals/development/powershell.test.yaml --test-id exit-vs-throw
```

Output goes to `results/{testname}_{timestamp}.jsonl` unless `--out` is provided.

## Requirements

- Python 3.10+ on PATH
- Evaluator location: `scripts/agent-eval/`
- `.env` for credentials/targets (recommended)

Environment keys:
- Provider: `PROVIDER=azure | anthropic` (CLI `--provider` overrides)
- Azure: `AZURE_OPEN_AI_ENDPOINT`, `AZURE_OPEN_AI_API_KEY`, `LLM_MODEL`
- Anthropic: `ANTHROPIC_API_KEY`, `LLM_MODEL` (or pass `--model`)
- VS Code: `EVAL_CARGOWISE_WORKSPACE_PATH` → `.code-workspace` path

## Targets

Execution targets in `evals/targets.yaml` decouple tests from providers/settings.
- `azure_base` → Azure OpenAI / Azure AI Foundry
- `vscode_cargowise` → VS Code Copilot via a workspace

Provider precedence: CLI `--provider` > `PROVIDER` env > default `azure`.

## Run examples (Windows PowerShell)

- Default target (uses azure provider):
  python -m eval_runner.cli `
    --tests ../../evals/development/powershell.test.yaml

- Azure (base LLM):
  python -m eval_runner.cli `
    --target azure_base `
    --tests ../../evals/development/powershell.test.yaml

- VS Code (Copilot):
  python -m eval_runner.cli `
    --target vscode_cargowise `
    --tests ../../evals/cargowise/base.test.yaml

- Dry run (no external calls):
  python -m eval_runner.cli `
    --target azure_base `
    --tests ../../evals/development/powershell.test.yaml `
    --dry-run

- Run specific test case by ID:
  python -m eval_runner.cli `
    --tests ../../evals/development/powershell.test.yaml `
    --test-id exit-vs-throw

Common flags:
- `--target <name>` execution target (default: default)
- `--test-id <id>` run only the test case with this specific ID
- `--out results/custom.jsonl` custom output file
- `--model <id>` override model per run
- `--agent-timeout <seconds>` timeout for agent response polling (default: 120)
- `--max-retries <count>` maximum retries for timeout cases (default: 2)

## Timeout handling and retries

When using VS Code Copilot or other agents that may experience timeouts, the evaluator includes automatic retry functionality:

- **Timeout detection**: Automatically detects when agents timeout (based on file creation status rather than response parsing)
- **Automatic retries**: When a timeout occurs, the same test case is retried up to `--max-retries` times (default: 2)
- **Retry behavior**: Only timeouts trigger retries; other errors proceed to the next test case
- **Timeout configuration**: Use `--agent-timeout` to adjust how long to wait for agent responses

Example with custom timeout settings:
```
python -m eval_runner.cli --target vscode_cargowise --tests ../../evals/cargowise/base.test.yaml --agent-timeout 180 --max-retries 3
```

## How the evals work

For each testcase in a `.test.yaml` file:
1) Parse YAML; collect only user messages (inline text and referenced files)
2) Extract code blocks from text for structured prompting
3) Select a domain-specific DSPy Signature; generate a candidate answer via provider/model
4) Score deterministically against the hidden expected answer by aspect coverage (the expected answer is never included in prompts)
5) Append a JSONL line and print a summary

### VS Code Copilot target

- Opens your configured workspace (`EVAL_CARGOWISE_WORKSPACE_PATH`) then runs: `code chat -r "{prompt}"`.
- The prompt is built from the `.test.yaml` user content (task, files, code blocks); the expected assistant answer is never included.
- Copilot is instructed to write its final answer to `scripts/agent-eval/.vscode-copilot/{test-case-id}.res.md`.
- If VS Code takes a moment, the runner polls up to ~30s for the reply file.

### Prompt file creation

When using VS Code targets (or dry-run mode), the evaluator creates individual prompt files for each test case:

- **Location**: `scripts/agent-eval/.vscode-copilot/`
- **Naming**: `{test-case-id}.req.md`
- **Format**: Contains instruction file references, reply path, and the question/task

**Manual execution example:**
```powershell
# After running the evaluator, you can manually execute prompts:
code chat -r "Run command Get-Content -Raw -LiteralPath C:\git\GitHub\WiseTechGlobal\WTG.AI.Prompts_1\scripts\agent-eval\.vscode-copilot\comprehensive-review-issues.req.md and follow its instructions."

# Or use the helper script:
.\run-prompt-with-vscode.ps1 -PromptFileName "comprehensive-review-issues.req.md"
```

**Workflow:**
1. Run evaluator: `python -m eval_runner.cli --target vscode_cargowise --tests ../../evals/development/powershell.test.yaml`
2. Find prompt files in `.vscode-copilot/` directory (e.g., `comprehensive-review-issues.req.md`)
3. Manually run specific prompts as needed using PowerShell + VS Code chat

## Scoring and outputs

Run with `--verbose` to print stack traces on errors.

Scoring:
- Aspects = bullet/numbered lines extracted from expected assistant answer (normalized)
- Match by token overlap (case-insensitive)
- Score = hits / total aspects; report `hits`, `misses`, `expected_aspect_count`

Output file:
- Default: `results/{testname}_{YYYYMMDD_HHMMSS}.jsonl` (or use `--out`)
- Fields: `test_id`, `score`, `hits`, `misses`, `model_answer`, `expected_aspect_count`, `provider`, `model`, `timestamp`

