"""
Tests for the Information Generator component of the Chinese Whisper game.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from information_generator import InformationGenerator, InformationSeed


class TestInformationGenerator(unittest.TestCase):
    """Test cases for InformationGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = InformationGenerator(seed=42)
    
    def test_initialization(self):
        """Test InformationGenerator initialization."""
        self.assertIsInstance(self.generator, InformationGenerator)
    
    def test_generate_factual_information_simple(self):
        """Test generation of simple factual information."""
        info = self.generator.generate_information("factual", "simple")
        
        self.assertIsInstance(info, InformationSeed)
        self.assertEqual(info.information_type, "factual")
        self.assertEqual(info.complexity_level, "simple")
        self.assertIsInstance(info.content, str)
        self.assertGreater(len(info.content), 0)
        self.assertIsInstance(info.expected_key_elements, list)
        self.assertGreater(len(info.expected_key_elements), 0)
    
    def test_generate_factual_information_medium(self):
        """Test generation of medium complexity factual information."""
        info = self.generator.generate_information("factual", "medium")
        
        self.assertEqual(info.information_type, "factual")
        self.assertEqual(info.complexity_level, "medium")
        self.assertGreater(len(info.content), 100)  # Medium should be longer
        self.assertGreater(len(info.expected_key_elements), 3)  # More key elements
    
    def test_generate_factual_information_complex(self):
        """Test generation of complex factual information."""
        info = self.generator.generate_information("factual", "complex")
        
        self.assertEqual(info.information_type, "factual")
        self.assertEqual(info.complexity_level, "complex")
        self.assertGreater(len(info.content), 200)  # Complex should be longest
        self.assertGreater(len(info.expected_key_elements), 5)  # Most key elements
    
    def test_generate_narrative_information(self):
        """Test generation of narrative information."""
        info = self.generator.generate_information("narrative", "simple")
        
        self.assertEqual(info.information_type, "narrative")
        self.assertIn("story", info.content.lower())  # Should contain story elements
        self.assertIsInstance(info.expected_key_elements, list)
    
    def test_generate_technical_information(self):
        """Test generation of technical information."""
        info = self.generator.generate_information("technical", "simple")
        
        self.assertEqual(info.information_type, "technical")
        self.assertIn("step", info.content.lower())  # Should contain steps
        self.assertIsInstance(info.expected_key_elements, list)
    
    def test_generate_structured_information(self):
        """Test generation of structured information."""
        info = self.generator.generate_information("structured", "simple")
        
        self.assertEqual(info.information_type, "structured")
        self.assertTrue(any(char in info.content for char in ['-', '•', '1.', '2.']))
        self.assertIsInstance(info.expected_key_elements, list)
    
    def test_invalid_information_type(self):
        """Test handling of invalid information type."""
        info = self.generator.generate_information("invalid_type", "simple")
        
        self.assertEqual(info.information_type, "invalid_type")
        self.assertIsInstance(info.content, str)
        self.assertGreater(len(info.content), 0)
    
    def test_invalid_complexity_level(self):
        """Test handling of invalid complexity level."""
        info = self.generator.generate_information("factual", "invalid_complexity")
        
        self.assertEqual(info.complexity_level, "invalid_complexity")
        self.assertIsInstance(info.content, str)
        self.assertGreater(len(info.content), 0)
    
    def test_seed_consistency(self):
        """Test that same seed produces consistent results."""
        generator1 = InformationGenerator(seed=123)
        generator2 = InformationGenerator(seed=123)
        
        info1 = generator1.generate_information("factual", "simple")
        info2 = generator2.generate_information("factual", "simple")
        
        self.assertEqual(info1.content, info2.content)
        self.assertEqual(info1.expected_key_elements, info2.expected_key_elements)
    
    def test_different_seeds_produce_different_content(self):
        """Test that different seeds produce different content."""
        generator1 = InformationGenerator(seed=123)
        generator2 = InformationGenerator(seed=456)
        
        info1 = generator1.generate_information("factual", "simple")
        info2 = generator2.generate_information("factual", "simple")
        
        self.assertNotEqual(info1.content, info2.content)
    
    def test_information_seed_attributes(self):
        """Test InformationSeed attributes."""
        info = self.generator.generate_information("narrative", "medium")
        
        self.assertTrue(hasattr(info, 'content'))
        self.assertTrue(hasattr(info, 'information_type'))
        self.assertTrue(hasattr(info, 'complexity_level'))
        self.assertTrue(hasattr(info, 'expected_key_elements'))
        self.assertTrue(hasattr(info, 'target_length'))
        
        self.assertIsInstance(info.content, str)
        self.assertIsInstance(info.information_type, str)
        self.assertIsInstance(info.complexity_level, str)
        self.assertIsInstance(info.expected_key_elements, list)
        self.assertIsInstance(info.target_length, int)
    
    def test_target_length_scaling(self):
        """Test that target length scales with complexity."""
        simple = self.generator.generate_information("factual", "simple")
        medium = self.generator.generate_information("factual", "medium")
        complex_info = self.generator.generate_information("factual", "complex")
        
        self.assertLess(simple.target_length, medium.target_length)
        self.assertLess(medium.target_length, complex_info.target_length)


class TestInformationSeed(unittest.TestCase):
    """Test cases for InformationSeed class."""
    
    def test_information_seed_creation(self):
        """Test InformationSeed creation."""
        seed = InformationSeed(
            content="Test content",
            information_type="test",
            complexity_level="simple",
            expected_key_elements=["element1", "element2"],
            target_length=100
        )
        
        self.assertEqual(seed.content, "Test content")
        self.assertEqual(seed.information_type, "test")
        self.assertEqual(seed.complexity_level, "simple")
        self.assertEqual(seed.expected_key_elements, ["element1", "element2"])
        self.assertEqual(seed.target_length, 100)


if __name__ == '__main__':
    unittest.main()
