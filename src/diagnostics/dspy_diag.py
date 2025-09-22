#!/usr/bin/env python3
"""
Quick diagnostic script to verify the Bbeval works
without external dependencies.
"""

import sys
import os
from pathlib import Path

# Add the bbeval package to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Test basic imports
    from bbeval import TestCase, EvaluationResult
    from bbeval.scoring import extract_aspects, score_candidate_response
    
    print("✓ Basic imports successful")
    
    # Test aspect extraction
    test_response = """
    Here are the issues:
    - Use Write-Output instead of echo
    - Replace exit with throw for better error handling
    - Add proper error checking
    """
    
    aspects = extract_aspects(test_response)
    print(f"✓ Extracted {len(aspects)} aspects: {aspects}")
    
    # Test scoring
    candidate = "You should use Write-Output and throw exceptions instead of exit"
    score_result = score_candidate_response(candidate, aspects)
    print(f"✓ Scoring works: {score_result['score']:.2f} score")
    
    print("\n🎉 Basic functionality check passed!")
    print("\nTo run full evaluations:")
    print("1. Install dependencies: pip install -e .")
    print("2. Configure .env with API keys")
    print("3. Run: python -m eval_runner.cli --tests path/to/test.yaml")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure all files are in place")
except Exception as e:
    print(f"✗ Diagnostic failed: {e}")
    import traceback
    traceback.print_exc()
