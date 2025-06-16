"""
Tests for the Evaluation Engine component of the Chinese Whisper game.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation_engine import EvaluationEngine, EvaluationMetrics


class TestEvaluationEngine(unittest.TestCase):
    """Test cases for EvaluationEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = EvaluationEngine()
        self.original_text = "The quick brown fox jumps over the lazy dog. This is a test sentence with numbers like 42 and dates like 2023."
        self.modified_text = "The quick brown fox jumped over the lazy dog. This is a test sentence with numbers like 43 and dates like 2024."
        self.key_elements = ["fox", "dog", "42", "2023", "test sentence"]
    
    def test_initialization(self):
        """Test EvaluationEngine initialization."""
        self.assertIsInstance(self.engine, EvaluationEngine)
    
    def test_calculate_retention_score_identical(self):
        """Test retention score calculation for identical texts."""
        score = self.engine.calculate_retention_score(self.original_text, self.original_text)
        self.assertEqual(score, 1.0)
    
    def test_calculate_retention_score_different(self):
        """Test retention score calculation for different texts."""
        score = self.engine.calculate_retention_score(self.original_text, "Completely different text")
        self.assertLess(score, 0.5)
        self.assertGreaterEqual(score, 0.0)
    
    def test_calculate_retention_score_similar(self):
        """Test retention score calculation for similar texts."""
        score = self.engine.calculate_retention_score(self.original_text, self.modified_text)
        self.assertGreater(score, 0.5)
        self.assertLess(score, 1.0)
    
    def test_calculate_semantic_similarity_identical(self):
        """Test semantic similarity for identical texts."""
        similarity = self.engine.calculate_semantic_similarity(self.original_text, self.original_text)
        self.assertEqual(similarity, 1.0)
    
    def test_calculate_semantic_similarity_different(self):
        """Test semantic similarity for different texts."""
        similarity = self.engine.calculate_semantic_similarity(self.original_text, "Completely different content")
        self.assertLess(similarity, 0.5)
        self.assertGreaterEqual(similarity, 0.0)
    
    def test_calculate_factual_accuracy_perfect(self):
        """Test factual accuracy calculation with perfect preservation."""
        accuracy = self.engine.calculate_factual_accuracy(
            self.original_text, 
            self.original_text, 
            self.key_elements
        )
        self.assertEqual(accuracy, 1.0)
    
    def test_calculate_factual_accuracy_partial(self):
        """Test factual accuracy calculation with partial preservation."""
        modified = "The quick brown fox jumps over the lazy cat. This is a different sentence."
        accuracy = self.engine.calculate_factual_accuracy(
            self.original_text, 
            modified, 
            self.key_elements
        )
        self.assertGreater(accuracy, 0.0)
        self.assertLess(accuracy, 1.0)
    
    def test_calculate_factual_accuracy_no_elements(self):
        """Test factual accuracy with no preserved elements."""
        accuracy = self.engine.calculate_factual_accuracy(
            self.original_text, 
            "Completely unrelated content", 
            self.key_elements
        )
        self.assertEqual(accuracy, 0.0)
    
    def test_calculate_structural_integrity_identical(self):
        """Test structural integrity for identical texts."""
        integrity = self.engine.calculate_structural_integrity(
            self.original_text, 
            self.original_text, 
            "general"
        )
        self.assertEqual(integrity, 1.0)
    
    def test_calculate_structural_integrity_different_types(self):
        """Test structural integrity for different information types."""
        technical_original = "Step 1: Do this. Step 2: Do that. Step 3: Finish."
        technical_modified = "Step 1: Do this. Step 2: Do something else."
        
        integrity = self.engine.calculate_structural_integrity(
            technical_original, 
            technical_modified, 
            "technical"
        )
        self.assertGreater(integrity, 0.0)
        self.assertLess(integrity, 1.0)
    
    def test_calculate_information_completeness(self):
        """Test information completeness calculation."""
        completeness = self.engine.calculate_information_completeness(
            self.original_text, 
            self.modified_text
        )
        self.assertGreaterEqual(completeness, 0.0)
        self.assertLessEqual(completeness, 1.0)
    
    def test_calculate_degradation_rate_single_step(self):
        """Test degradation rate calculation for single step."""
        chain_data = [self.original_text, self.modified_text]
        rate = self.engine.calculate_degradation_rate(chain_data)
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)
    
    def test_calculate_degradation_rate_multiple_steps(self):
        """Test degradation rate calculation for multiple steps."""
        chain_data = [
            self.original_text,
            self.modified_text,
            "Further modified text",
            "Even more modified"
        ]
        rate = self.engine.calculate_degradation_rate(chain_data)
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)
    
    def test_calculate_degradation_rate_empty_chain(self):
        """Test degradation rate calculation for empty chain."""
        rate = self.engine.calculate_degradation_rate([])
        self.assertEqual(rate, 0.0)
    
    def test_evaluate_chain_complete(self):
        """Test complete chain evaluation."""
        chain_data = [self.modified_text, "Further modified", "Final version"]
        
        metrics = self.engine.evaluate_chain(
            original_info=self.original_text,
            chain_data=chain_data,
            key_elements=self.key_elements,
            info_type="general"
        )
        
        self.assertIsInstance(metrics, EvaluationMetrics)
        self.assertGreaterEqual(metrics.retention_score, 0.0)
        self.assertLessEqual(metrics.retention_score, 1.0)
        self.assertGreaterEqual(metrics.semantic_similarity, 0.0)
        self.assertLessEqual(metrics.semantic_similarity, 1.0)
        self.assertGreaterEqual(metrics.factual_accuracy, 0.0)
        self.assertLessEqual(metrics.factual_accuracy, 1.0)
        self.assertGreaterEqual(metrics.structural_integrity, 0.0)
        self.assertLessEqual(metrics.structural_integrity, 1.0)
        self.assertGreaterEqual(metrics.information_completeness, 0.0)
        self.assertLessEqual(metrics.information_completeness, 1.0)
        self.assertGreaterEqual(metrics.degradation_rate, 0.0)
        self.assertLessEqual(metrics.degradation_rate, 1.0)
        self.assertIsInstance(metrics.key_elements_preserved, int)
        self.assertIsInstance(metrics.total_key_elements, int)
    
    def test_analyze_critical_thresholds(self):
        """Test critical threshold analysis."""
        chain_data = [
            "Original text with good quality",
            "Slightly modified text",
            "More modified text",
            "Heavily degraded text",
            "Barely recognizable"
        ]
        
        analysis = self.engine.analyze_critical_thresholds(
            chain_data=chain_data,
            original_info="Original text with good quality"
        )
        
        self.assertIsInstance(analysis, dict)
        self.assertIn('critical_point', analysis)
        self.assertIn('degradation_pattern', analysis)
        self.assertIn('threshold_metrics', analysis)
    
    def test_extract_meaningful_words(self):
        """Test meaningful word extraction."""
        words = self.engine._extract_meaningful_words(self.original_text)
        self.assertIsInstance(words, list)
        self.assertGreater(len(words), 0)
        self.assertNotIn('the', words)
        self.assertNotIn('is', words)
        self.assertIn('quick', words)
        self.assertIn('brown', words)
    
    def test_normalize_text(self):
        """Test text normalization."""
        text = "  Hello, World!  123  "
        normalized = self.engine._normalize_text(text)
        self.assertEqual(normalized, "hello world 123")
    
    def test_is_numeric(self):
        """Test numeric detection."""
        self.assertTrue(self.engine._is_numeric("123"))
        self.assertTrue(self.engine._is_numeric("123.45"))
        self.assertTrue(self.engine._is_numeric("-123"))
        self.assertFalse(self.engine._is_numeric("abc"))
        self.assertFalse(self.engine._is_numeric("12abc"))
    
    def test_find_similar_number(self):
        """Test similar number finding."""
        numbers = ["123", "456", "789"]
        
        match = self.engine._find_similar_number("123", numbers)
        self.assertEqual(match, "123")
        
        match = self.engine._find_similar_number("124", numbers)
        self.assertEqual(match, "123")
        
        match = self.engine._find_similar_number("999", numbers)
        self.assertIsNone(match)
    
    def test_find_partial_match(self):
        """Test partial string matching."""
        strings = ["hello world", "goodbye world", "hello universe"]
        
        match = self.engine._find_partial_match("hello world", strings)
        self.assertEqual(match, "hello world")
        
        match = self.engine._find_partial_match("hello", strings)
        self.assertIn(match, ["hello world", "hello universe"])
        
        match = self.engine._find_partial_match("xyz", strings)
        self.assertIsNone(match)


class TestEvaluationMetrics(unittest.TestCase):
    """Test cases for EvaluationMetrics dataclass."""
    
    def test_evaluation_metrics_creation(self):
        """Test EvaluationMetrics creation."""
        metrics = EvaluationMetrics(
            retention_score=0.8,
            semantic_similarity=0.7,
            factual_accuracy=0.9,
            structural_integrity=0.6,
            information_completeness=0.75,
            degradation_rate=0.2,
            key_elements_preserved=4,
            total_key_elements=5
        )
        
        self.assertEqual(metrics.retention_score, 0.8)
        self.assertEqual(metrics.semantic_similarity, 0.7)
        self.assertEqual(metrics.factual_accuracy, 0.9)
        self.assertEqual(metrics.structural_integrity, 0.6)
        self.assertEqual(metrics.information_completeness, 0.75)
        self.assertEqual(metrics.degradation_rate, 0.2)
        self.assertEqual(metrics.key_elements_preserved, 4)
        self.assertEqual(metrics.total_key_elements, 5)


if __name__ == '__main__':
    unittest.main()
