---
mode: 'agent'
description: 'Apply when writing evals in YAML format'
---

## Schema Reference
- Schema: #file:../contexts/eval-schema.json (JSON Schema for validation and tooling)
- Format: YAML with structured content arrays

## Structure Requirements
- Root level: `description` (optional), `grader` (optional), `target` (optional), `testcases` (required)
- Test case fields: `id` (required), `outcome` (required), `messages` (required), `note` (optional)
- Message fields: `role` (required), `content` (required)
- Message roles: `system`, `user`, `assistant`
- Content types: `text`, `file`
- File paths must start with "/" for absolute paths (e.g., "/domains/cargowise/instructions/file.md")

## Example
```yaml
description: Evals for analyzing code patterns
grader: llm_judge
target: azure_base

testcases:
- id: external-file-reference-system-context
  outcome: Identifies bitwise operations correctly
  messages:
    - role: system
      content:
        - type: text
          value: You are an expert code analyzer
        - type: file
          value: /domains/base/instructions/analysis.md
    - role: user
      content:
        - type: text
          value: Analyze this code
        - type: file
          value: /evals/snippets/bitwise.cs
    - role: assistant
      content:
        - type: file
          value: /evals/snippets/analysis.md

- id: inline-code-block-scalars
  outcome: Handles special characters without escaping
  validator: /scripts/validators/csharp-build.ps1
  messages:
    - role: user
      content: |-
        Check this C# code:
        
        ```csharp
        var s = "value: 100 # not a comment";
        return s.Contains("<tag>") ? "xml" : "plain";
        ```
    - role: assistant
      content: |-
        Code analysis:
        - String contains # and : characters
        - Ternary operator with < > comparison
        - No XML escaping needed in YAML block scalars
```
