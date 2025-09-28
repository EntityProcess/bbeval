"""
Scoring System for Evaluating Model Responses

Implements deterministic heuristic aspect coverage scoring without 
leaking expected answers during generation.
"""

import re
from typing import List, Dict, Set
from . import EvaluationResult

# Constants for scoring configuration
KEY_TERM_MATCH_THRESHOLD = 0.5
"""Minimum ratio of key terms that must match for a positive aspect hit."""

ACTION_WORDS = {
    'use', 'avoid', 'prefer', 'replace', 'consider', 'ensure', 'remove', 'add'
}
"""Set of action words that indicate actionable recommendations in text."""

def extract_aspects(expected_response: str) -> List[str]:
    """
    Extract individual aspects/points from the expected assistant response.
    
    Looks for bullet points, numbered lists, and other structured content
    that represents distinct evaluation criteria.
    
    Args:
        expected_response: The ground truth assistant response
    
    Returns:
        List of normalized aspects
    """
    aspects = []
    lines = expected_response.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Match bullet points (-, *, •) or numbered lists (1., 2., etc.)
        bullet_match = re.match(r'^(\s*[-*•]|\s*[0-9]+\.)\s*(.+)', line)
        if bullet_match:
            aspect = bullet_match.group(2).strip()
            if aspect:
                # Normalize the aspect text
                normalized = normalize_aspect(aspect)
                if normalized:
                    aspects.append(normalized)
        else:
            # Also consider lines that start with action words or seem like recommendations
            if any(line.lower().startswith(word) for word in ACTION_WORDS):
                normalized = normalize_aspect(line)
                if normalized:
                    aspects.append(normalized)
    
    return aspects

def normalize_aspect(aspect: str) -> str:
    """
    Normalize an aspect for matching by removing punctuation and converting to lowercase.
    
    Args:
        aspect: Raw aspect text
    
    Returns:
        Normalized aspect text
    """
    # Remove extra whitespace and punctuation
    normalized = re.sub(r'[^\w\s]', ' ', aspect.lower())
    normalized = ' '.join(normalized.split())  # Remove extra whitespace
    return normalized


def extract_key_terms(aspect: str, max_terms: int = 5) -> List[str]:
    """
    Extract key terms from an aspect for matching.
    
    Args:
        aspect: Normalized aspect text
        max_terms: Maximum number of terms to extract
    
    Returns:
        List of key terms
    """
    # Common stop words to filter out
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
    }
    
    terms = []
    words = aspect.split()
    
    for word in words:
        if len(word) > 2 and word not in stop_words:
            terms.append(word)
            if len(terms) >= max_terms:
                break
    
    return terms

def score_candidate_response(candidate: str, expected_aspects: List[str]) -> Dict:
    """
    Score a candidate response against expected aspects.
    
    Args:
        candidate: The model-generated response
        expected_aspects: List of normalized expected aspects
    
    Returns:
        Dictionary with score, hits, misses, and other metrics
    """
    if not expected_aspects:
        # If no explicit aspects exist, treat clear error outputs as failure (score 0)
        if is_error_like(candidate):
            return {
                'score': 0.0,
                'hits': [],
                'misses': ['Model produced an error instead of an answer.'],
                'hit_count': 0,
                'total_aspects': 0
            }
        # Otherwise default to neutral success to avoid punishing non-aspect cases
        return {
            'score': 1.0,
            'hits': [],
            'misses': [],
            'hit_count': 0,
            'total_aspects': 0
        }
    
    candidate_lower = candidate.lower()
    # Remove punctuation for better matching
    candidate_normalized = re.sub(r'[^\w\s]', ' ', candidate_lower)
    candidate_words = set(candidate_normalized.split())
    
    hits = []
    misses = []
    
    for aspect in expected_aspects:
        key_terms = extract_key_terms(aspect)
        
        if not key_terms:
            continue
        
        # Check if any key terms are present in the candidate
        matches = sum(1 for term in key_terms if term in candidate_words)
        match_ratio = matches / len(key_terms) if key_terms else 0
        
        # Consider it a hit if at least the threshold of key terms match
        if match_ratio >= KEY_TERM_MATCH_THRESHOLD:
            hits.append(aspect)
        else:
            # Also check for partial phrase matching
            aspect_words = aspect.split()
            if len(aspect_words) >= 2:
                # Look for consecutive word sequences
                for i in range(len(aspect_words) - 1):
                    phrase = ' '.join(aspect_words[i:i+2])
                    if phrase in candidate_normalized:
                        hits.append(aspect)
                        break
                else:
                    misses.append(aspect)
            else:
                misses.append(aspect)
    
    score = len(hits) / len(expected_aspects) if expected_aspects else 0.0
    
    return {
        'score': score,
        'hits': hits,
        'misses': misses,
        'hit_count': len(hits),
        'total_aspects': len(expected_aspects)
    }

def detect_potential_hallucinations(candidate: str, 
                                  guidelines_text: str, 
                                  code_text: str,
                                  allowlist: Set[str] = None) -> List[str]:
    """
    Detect potential hallucinations in the candidate response.
    
    Identifies content in the response that doesn't appear in the provided
    guidelines or code context.
    
    Args:
        candidate: Model response
        guidelines_text: Guideline content provided to model
        code_text: Code content provided to model
        allowlist: Set of allowed terms that shouldn't be flagged
    
    Returns:
        List of potentially hallucinated phrases
    """
    if allowlist is None:
        allowlist = {
            'recommend', 'suggest', 'consider', 'should', 'could', 'would',
            'best', 'practice', 'better', 'improve', 'optimize', 'ensure',
            'avoid', 'prefer', 'use', 'replace', 'remove', 'add', 'update'
        }
    
    # Combine all reference text
    reference_text = (guidelines_text + ' ' + code_text).lower()
    reference_words = set(re.findall(r'\b\w+\b', reference_text))
    
    candidate_words = re.findall(r'\b\w+\b', candidate.lower())
    
    hallucinations = []
    for word in candidate_words:
        if (len(word) > 3 and 
            word not in reference_words and 
            word not in allowlist and
            not word.isdigit()):
            hallucinations.append(word)
    
    # Remove duplicates while preserving order
    unique_hallucinations = []
    seen = set()
    for item in hallucinations:
        if item not in seen:
            unique_hallucinations.append(item)
            seen.add(item)
    
    return unique_hallucinations

def grade_test_case_heuristic(test_case, candidate_response: str, provider: str, target_name: str) -> EvaluationResult:
    """
    Grade a single test case against the candidate response using heuristic scoring.
    
    Args:
        test_case: TestCase object
        candidate_response: Model-generated response
        provider: Model provider name
        target_name: Target name used for evaluation
    
    Returns:
        EvaluationResult object
    """
    from datetime import datetime
    
    # Extract aspects from expected response
    expected_aspects = extract_aspects(test_case.expected_assistant_raw)
    
    # Score the candidate
    scoring_result = score_candidate_response(candidate_response, expected_aspects)
    # If no aspects but response looks like an error, surface the error message as a miss detail
    if not expected_aspects and is_error_like(candidate_response):
        # Prepend the first line of the error to misses for visibility
        first_line = candidate_response.splitlines()[0].strip() if candidate_response else ""
        if first_line and first_line not in scoring_result['misses']:
            scoring_result['misses'] = [first_line] + scoring_result['misses']
    
    # Create evaluation result
    result = EvaluationResult(
        test_id=test_case.id,
        score=scoring_result['score'],
        hits=scoring_result['hits'],
        misses=scoring_result['misses'],
        model_answer=candidate_response,
        expected_aspect_count=len(expected_aspects),
        target=target_name,
        timestamp=datetime.utcnow().isoformat() + 'Z',
        raw_aspects=expected_aspects
    )
    
    return result

def is_error_like(text: str) -> bool:
    """
    Heuristic to detect if a response is an error instead of an actual answer.

    Considers common prefixes and patterns that indicate command/tooling failures.
    """
    if not text:
        return False
    lowered = text.strip().lower()
    error_prefixes = [
        'error:', 'err:', 'vs code command failed', 'exception', 'traceback',
        'no response file was generated', 'timed out', 'cli not found'
    ]
    return any(lowered.startswith(p) for p in error_prefixes)


def grade_test_case_llm_judge(test_case, candidate_response: str, provider: str, target_name: str) -> EvaluationResult:
    """
    Grade a single test case against the candidate response using QualityGrader.
    
    Args:
        test_case: TestCase object
        candidate_response: Model-generated response
        provider: Model provider name
        target_name: Target name used for evaluation
    
    Returns:
        EvaluationResult object
    """
    import dspy
    from datetime import datetime
    from .signatures import QualityGrader
    
    # Create the judge module using QualityGrader
    judge = dspy.Predict(QualityGrader)
    
    try:
        # Run the QualityGrader
        result = judge(
            expected_outcome=test_case.outcome,
            request=getattr(test_case, 'request', '') or '',
            reference_answer=test_case.expected_assistant_raw,
            generated_answer=candidate_response
        )
        
        # Parse the score
        try:
            score = float(result.score)
            # Clamp score between 0.0 and 1.0
            score = max(0.0, min(1.0, score))
        except (ValueError, TypeError):
            score = 0.0
        
        # Extract hits and misses from QualityGrader output fields
        hits = []
        misses = []
        if hasattr(result, 'hits') and result.hits:
            if isinstance(result.hits, list):
                hits = result.hits
            else:
                hits = [result.hits]  # Convert single string to list
        if hasattr(result, 'misses') and result.misses:
            if isinstance(result.misses, list):
                misses = result.misses
            else:
                misses = [result.misses]  # Convert single string to list
        
        # Create evaluation result
        evaluation_result = EvaluationResult(
            test_id=test_case.id,
            score=score,
            hits=hits,
            misses=misses,
            model_answer=candidate_response,
            expected_aspect_count=len(hits) + len(misses) if (hits or misses) else 1,
            target=target_name,
            timestamp=datetime.utcnow().isoformat() + 'Z',
            reasoning=getattr(result, 'reasoning', None),
            raw_aspects=[result.reasoning] if hasattr(result, 'reasoning') and result.reasoning else []
        )
        
        return evaluation_result
        
    except Exception as e:
        # Fallback to error result if QualityGrader fails
        error_result = EvaluationResult(
            test_id=test_case.id,
            score=0.0,
            hits=[],
            misses=[f"QualityGrader failed: {str(e)}"],
            model_answer=candidate_response,
            expected_aspect_count=0,
            target=target_name,
            timestamp=datetime.utcnow().isoformat() + 'Z',
            raw_aspects=[]
        )
        
        return error_result
