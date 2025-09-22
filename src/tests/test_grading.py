#!/usr/bin/env python3
"""
Unit tests for bbeval.scoring module.

Tests the scoring configuration constants and functionality.
"""

import unittest

from bbeval.grading import KEY_TERM_MATCH_THRESHOLD, ACTION_WORDS


class TestScoringConstants(unittest.TestCase):
    """Test cases for the scoring configuration constants."""
    
    def test_key_term_match_threshold_exists(self):
        """Test that the key term match threshold constant is defined."""
        self.assertIsInstance(KEY_TERM_MATCH_THRESHOLD, float)
        self.assertGreater(KEY_TERM_MATCH_THRESHOLD, 0.0)
        self.assertLessEqual(KEY_TERM_MATCH_THRESHOLD, 1.0)
    
    def test_key_term_match_threshold_value(self):
        """Test that the threshold has the expected default value."""
        self.assertEqual(KEY_TERM_MATCH_THRESHOLD, 0.5)
    
    def test_action_words_set_exists(self):
        """Test that the action words set is defined and contains expected words."""
        self.assertIsInstance(ACTION_WORDS, set)
        self.assertIn('use', ACTION_WORDS)
        self.assertIn('avoid', ACTION_WORDS)
        self.assertIn('prefer', ACTION_WORDS)
        self.assertGreater(len(ACTION_WORDS), 5)  # Should have multiple action words
    
    def test_action_words_content(self):
        """Test that action words set contains all expected action words."""
        expected_words = {
            'use', 'avoid', 'prefer', 'replace', 'consider', 'ensure', 'remove', 'add'
        }
        self.assertTrue(expected_words.issubset(ACTION_WORDS))
    
    def test_action_words_immutable(self):
        """Test that action words is a set (immutable for our purposes)."""
        # Verify it's a set and not a list that could be accidentally modified
        self.assertIsInstance(ACTION_WORDS, set)
        
        # Test that we can't accidentally modify the original
        original_size = len(ACTION_WORDS)
        test_set = ACTION_WORDS.copy()
        test_set.add('test_word')
        self.assertEqual(len(ACTION_WORDS), original_size)


if __name__ == '__main__':
    unittest.main(verbosity=2)