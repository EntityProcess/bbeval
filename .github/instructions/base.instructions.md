---
description: 'Navigate bbeval codebase structure and find specific components, properties, and evaluation flow elements'
applyTo: '**/bbeval/**/{*.py,*.md,*.yaml,*.toml}'
---

# BB-Eval Codebase Navigation Guide

## Core Architecture

**Evaluation Flow**: Test YAML → Parser → Signature Selection → Model Execution → Grading → Results

## Key Concepts

### DSPy Integration
- Uses DSPy signatures for structured LLM interactions
- Signatures define input/output fields for different evaluation types
- Generic `EvaluationModule` wraps any signature for execution

### Grading Approach
- **Heuristic scoring**: Extracts aspects from expected responses, scores without leaking answers
- **LLM judge**: Uses quality grader signature for subjective evaluation
- Results include hit/miss breakdown and full audit trail

### Test Structure
- YAML files define test cases with user/assistant message flows
- Supports file references and context injection
- Signature selection based on test case characteristics

Use semantic search and code exploration to find specific implementations and configurations.