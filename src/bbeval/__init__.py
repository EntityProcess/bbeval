"""
Bbeval Package

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
    provider: str
    model: str
    timestamp: str
    raw_aspects: Optional[list[str]] = None
    
    @property
    def hit_count(self) -> int:
        """Number of successfully matched aspects."""
        return len(self.hits)
