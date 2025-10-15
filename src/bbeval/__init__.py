"""
BbEval Package

A lightweight, extensible evaluator for testing model-generated responses 
against test specifications without leaking expected answers into prompts.
"""

from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class TestMessage:
    """Represents a single message in a test conversation."""
    role: Literal['user', 'assistant']
    content: list[dict] | str

@dataclass 
class TestCase:
    """Represents a single test case with user input and expected output."""
    id: str
    task: str
    user_segments: list[dict]          # resolved segments (files/text)
    expected_assistant_raw: str        # ground truth (never shown to model)
    guideline_paths: list[str]         # paths to guideline files
    code_snippets: list[str]          # extracted code blocks from segments
    outcome: str                       # expected outcome description for signature selection
    grader: str = 'llm_judge'          # grading method: 'heuristic' or 'llm_judge'

@dataclass
class EvaluationResult:
    """Results from evaluating a single test case."""
    test_id: str
    score: float
    hits: list[str]
    misses: list[str]
    model_answer: str
    expected_aspect_count: int
    target: str
    timestamp: str
    reasoning: Optional[str] = None  # LLM judge reasoning for the score
    raw_aspects: Optional[list[str]] = None
    # Raw request metadata capturing what was actually sent to the model.
    # For standard providers this includes the structured fields (request, guidelines).
    # For the VS Code provider it additionally captures the enhanced prompt written to the .req.md file
    # and the path to that file so downstream graders / audits can reconstruct the exact conversation context.
    raw_request: Optional[dict] = None
    # Raw request information for the LLM judge (QualityGrader). Contains both the
    # structured inputs we passed and, when available, low-level prompt/messages
    # captured from the underlying LM forward() call for full auditability.
    grader_raw_request: Optional[dict] = None
    
    @property
    def hit_count(self) -> int:
        """Number of successfully matched aspects."""
        return len(self.hits)
