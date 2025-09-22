"""
DSPy signatures and modules used by the evaluator.
"""

import dspy
import re
from typing import Type

class CodeReview(dspy.Signature):
    """Code review and analysis for any programming language."""

    task = dspy.InputField(desc="The code review or analysis task")
    guidelines = dspy.InputField(desc="Coding guidelines and best practices")
    code = dspy.InputField(desc="Code to review and analyze")
    context = dspy.InputField(desc="Additional context", prefix="Context:")

    answer = dspy.OutputField(desc="Code review with issues, recommendations, and best practices")

class KnowledgeQuery(dspy.Signature):
    """Knowledge retrieval and explanation from codebase."""

    task = dspy.InputField(desc="The knowledge query or question")
    guidelines = dspy.InputField(desc="Domain knowledge and documentation")
    code = dspy.InputField(desc="Relevant code examples and references")
    context = dspy.InputField(desc="Additional context", prefix="Context:")

    answer = dspy.OutputField(desc="Clear explanation with code references and examples")

class CodeGeneration(dspy.Signature):
    """Code generation and implementation."""

    task = dspy.InputField(desc="The code generation task or requirement")
    guidelines = dspy.InputField(desc="Architecture guidelines and patterns")
    code = dspy.InputField(desc="Reference code and examples")
    context = dspy.InputField(desc="Additional context", prefix="Context:")

    answer = dspy.OutputField(desc="Complete code implementation with proper structure and patterns")

def determine_signature_from_test_case(test_case) -> Type[dspy.Signature]:
    """
    Determine the appropriate signature class based on test case outcome and user content.
    
    Args:
        test_case: TestCase object with outcome and task information
        
    Returns:
        Appropriate DSPy signature class
    """
    outcome = test_case.outcome.lower()
    task = test_case.task.lower()
    
    # Extract first line from task for analysis (most consistent approach)
    first_line = task.split('\n')[0].strip()
    
    # Code generation patterns
    generation_patterns = [
        'create', 'creates', 'implement', 'build', 'generate', 'write.*new',
        'write.*code', 'please create', 'should contain', 'processor.*creation',
        'corresponding functionalities', 'convert.*to', 'i need.*prompt',
        'i need.*for', 'convert.*yaml.*to'
    ]
    
    # Knowledge query patterns  
    knowledge_patterns = [
        'what does.*stand for', 'resolves.*acronym', 'explains', 'defines',
        'what is', 'find.*definition', 'according to', 'stands for',
        'in the context of'
    ]
    
    # Code review patterns (everything else - reviews, analysis, evaluation)
    review_patterns = [
        'review', 'analyze', 'assessment', 'evaluate', 'assess', 'check',
        'flags', 'identifies', 'notes', 'finds', 'issues', 'violations',
        'proper.*behavior', 'best practices', 'compliance'
    ]
    
    # Check outcome and first line for patterns
    combined_text = f"{outcome} {first_line}"
    
    if any(re.search(pattern, combined_text, re.IGNORECASE) for pattern in generation_patterns):
        return CodeGeneration
    elif any(re.search(pattern, combined_text, re.IGNORECASE) for pattern in knowledge_patterns):
        return KnowledgeQuery
    elif any(re.search(pattern, combined_text, re.IGNORECASE) for pattern in review_patterns):
        return CodeReview
    else:
        # Default to code review for unknown patterns
        return CodeReview

class EvaluationModule(dspy.Module):
    """A generic evaluation module that can use any provided signature."""

    def __init__(self, signature_class: dspy.Signature):
        super().__init__()
        self.predictor = dspy.Predict(signature_class)

    def forward(self, test_case_id: str = None, **kwargs):
        """Forwards the request to the configured model's execution method."""
        return dspy.settings.lm.execute_prediction(self, test_case_id=test_case_id, **kwargs)
