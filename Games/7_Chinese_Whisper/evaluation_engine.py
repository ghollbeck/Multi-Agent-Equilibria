import json
import re
import math
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from difflib import SequenceMatcher
import numpy as np

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    retention_score: float
    semantic_similarity: float
    factual_accuracy: float
    structural_integrity: float
    information_completeness: float
    degradation_rate: float
    key_elements_preserved: int
    total_key_elements: int

class EvaluationEngine:
    """
    Comprehensive evaluation engine for analyzing information degradation
    in the Chinese Whisper game.
    """
    
    def __init__(self):
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
    
    def calculate_retention_score(self, original: str, final: str) -> float:
        """Calculate percentage of original information retained."""
        if not original.strip():
            return 0.0
        
        original_words = set(self._extract_meaningful_words(original))
        final_words = set(self._extract_meaningful_words(final))
        
        if not original_words:
            return 0.0
        
        preserved_words = original_words.intersection(final_words)
        return len(preserved_words) / len(original_words)
    
    def calculate_semantic_similarity(self, original: str, final: str) -> float:
        """Calculate semantic similarity using sequence matching."""
        if not original.strip() or not final.strip():
            return 0.0
        
        original_norm = self._normalize_text(original)
        final_norm = self._normalize_text(final)
        
        matcher = SequenceMatcher(None, original_norm, final_norm)
        return matcher.ratio()
    
    def calculate_factual_accuracy(self, original: str, final: str, key_elements: List[str]) -> float:
        """Calculate how many factual elements are preserved."""
        if not key_elements:
            return 1.0  # No facts to preserve
        
        final_lower = final.lower()
        preserved_count = 0
        
        for element in key_elements:
            element_lower = element.lower()
            if element_lower in final_lower:
                preserved_count += 1
            elif self._is_numeric(element) and self._find_similar_number(element, final):
                preserved_count += 0.5  # Partial credit for similar numbers
            elif len(element) > 3 and self._find_partial_match(element_lower, final_lower):
                preserved_count += 0.5  # Partial credit for partial matches
        
        return preserved_count / len(key_elements)
    
    def calculate_structural_integrity(self, original: str, final: str, info_type: str) -> float:
        """Calculate how well the original structure is maintained."""
        if info_type == "technical":
            return self._evaluate_technical_structure(original, final)
        elif info_type == "narrative":
            return self._evaluate_narrative_structure(original, final)
        elif info_type == "structured":
            return self._evaluate_structured_format(original, final)
        else:
            return self._evaluate_general_structure(original, final)
    
    def calculate_information_completeness(self, original: str, final: str) -> float:
        """Calculate completeness based on length and content coverage."""
        if not original.strip():
            return 0.0
        
        original_sentences = self._split_sentences(original)
        final_sentences = self._split_sentences(final)
        
        length_ratio = min(len(final) / len(original), 1.0) if original else 0.0
        
        sentence_coverage = min(len(final_sentences) / len(original_sentences), 1.0) if original_sentences else 0.0
        
        return 0.6 * length_ratio + 0.4 * sentence_coverage
    
    def calculate_degradation_rate(self, chain_data: List[str]) -> float:
        """Calculate average degradation per agent in the chain."""
        if len(chain_data) < 2:
            return 0.0
        
        degradations = []
        for i in range(1, len(chain_data)):
            similarity = self.calculate_semantic_similarity(chain_data[i-1], chain_data[i])
            degradation = 1.0 - similarity
            degradations.append(degradation)
        
        return float(np.mean(degradations)) if degradations else 0.0
    
    def evaluate_chain(self, original_info: str, chain_data: List[str], 
                      key_elements: List[str], info_type: str) -> EvaluationMetrics:
        """Comprehensive evaluation of an entire whisper chain."""
        if not chain_data:
            return EvaluationMetrics(0, 0, 0, 0, 0, 0, 0, len(key_elements))
        
        final_info = chain_data[-1]
        
        retention = self.calculate_retention_score(original_info, final_info)
        similarity = self.calculate_semantic_similarity(original_info, final_info)
        factual = self.calculate_factual_accuracy(original_info, final_info, key_elements)
        structural = self.calculate_structural_integrity(original_info, final_info, info_type)
        completeness = self.calculate_information_completeness(original_info, final_info)
        degradation = self.calculate_degradation_rate([original_info] + chain_data)
        
        preserved_elements = self._count_preserved_elements(final_info, key_elements)
        
        return EvaluationMetrics(
            retention_score=retention,
            semantic_similarity=similarity,
            factual_accuracy=factual,
            structural_integrity=structural,
            information_completeness=completeness,
            degradation_rate=degradation,
            key_elements_preserved=preserved_elements,
            total_key_elements=len(key_elements)
        )
    
    def analyze_critical_thresholds(self, chain_data: List[str], original_info: str) -> Dict[str, Any]:
        """Analyze at what point in the chain information degrades significantly."""
        if len(chain_data) < 2:
            return {"critical_point": None, "threshold_analysis": []}
        
        similarities = []
        cumulative_degradation = []
        
        current_reference = original_info
        for i, info in enumerate(chain_data):
            similarity = self.calculate_semantic_similarity(current_reference, info)
            similarities.append(similarity)
            
            original_similarity = self.calculate_semantic_similarity(original_info, info)
            cumulative_degradation.append(1.0 - original_similarity)
        
        critical_point = self._find_critical_point(cumulative_degradation)
        
        return {
            "critical_point": critical_point,
            "similarities": similarities,
            "cumulative_degradation": cumulative_degradation,
            "threshold_analysis": self._analyze_thresholds(cumulative_degradation)
        }
    
    def _extract_meaningful_words(self, text: str) -> List[str]:
        """Extract meaningful words, excluding stop words."""
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if word not in self.stop_words and len(word) > 2]
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()
    
    def _is_numeric(self, text: str) -> bool:
        """Check if text represents a number."""
        try:
            float(text.replace(',', '').replace('%', ''))
            return True
        except ValueError:
            return False
    
    def _find_similar_number(self, original_num: str, text: str) -> bool:
        """Find if a similar number exists in the text."""
        numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', text)
        try:
            original_val = float(original_num.replace(',', '').replace('%', ''))
            for num in numbers:
                num_val = float(num.replace(',', ''))
                if abs(original_val - num_val) / max(original_val, 1) < 0.1:  # 10% tolerance
                    return True
        except ValueError:
            pass
        return False
    
    def _find_partial_match(self, element: str, text: str) -> bool:
        """Find partial matches for names and terms."""
        if len(element) > 6:
            for i in range(len(text) - len(element) + 1):
                substring = text[i:i+len(element)]
                similarity = SequenceMatcher(None, element, substring).ratio()
                if similarity > 0.7:
                    return True
        return False
    
    def _evaluate_technical_structure(self, original: str, final: str) -> float:
        """Evaluate preservation of technical instruction structure."""
        original_steps = len(re.findall(r'\d+\)', original))
        final_steps = len(re.findall(r'\d+\)', final))
        
        step_preservation = min(final_steps / original_steps, 1.0) if original_steps > 0 else 1.0
        
        tech_patterns = [r'[a-zA-Z-]+\.[a-zA-Z]+', r'--[a-zA-Z-]+', r'sudo\s+\w+', r'http[s]?://']
        original_tech = sum(len(re.findall(pattern, original)) for pattern in tech_patterns)
        final_tech = sum(len(re.findall(pattern, final)) for pattern in tech_patterns)
        
        tech_preservation = min(final_tech / original_tech, 1.0) if original_tech > 0 else 1.0
        
        return 0.6 * step_preservation + 0.4 * tech_preservation
    
    def _evaluate_narrative_structure(self, original: str, final: str) -> float:
        """Evaluate preservation of narrative structure."""
        original_names = set(re.findall(r'\b[A-Z][a-z]+\b', original))
        final_names = set(re.findall(r'\b[A-Z][a-z]+\b', final))
        
        name_preservation = len(original_names.intersection(final_names)) / len(original_names) if original_names else 1.0
        
        time_words = ['then', 'after', 'before', 'when', 'while', 'during', 'finally', 'first', 'next', 'last']
        original_time = sum(1 for word in time_words if word in original.lower())
        final_time = sum(1 for word in time_words if word in final.lower())
        
        time_preservation = min(final_time / original_time, 1.0) if original_time > 0 else 1.0
        
        return 0.7 * name_preservation + 0.3 * time_preservation
    
    def _evaluate_structured_format(self, original: str, final: str) -> float:
        """Evaluate preservation of structured data format."""
        original_colons = original.count(':')
        final_colons = final.count(':')
        
        colon_preservation = min(final_colons / original_colons, 1.0) if original_colons > 0 else 1.0
        
        original_parens = original.count('(') + original.count(')')
        final_parens = final.count('(') + final.count(')')
        
        paren_preservation = min(final_parens / original_parens, 1.0) if original_parens > 0 else 1.0
        
        return 0.6 * colon_preservation + 0.4 * paren_preservation
    
    def _evaluate_general_structure(self, original: str, final: str) -> float:
        """Evaluate general structural preservation."""
        original_sentences = len(self._split_sentences(original))
        final_sentences = len(self._split_sentences(final))
        
        sentence_preservation = min(final_sentences / original_sentences, 1.0) if original_sentences > 0 else 1.0
        
        original_paragraphs = len(original.split('\n\n'))
        final_paragraphs = len(final.split('\n\n'))
        
        paragraph_preservation = min(final_paragraphs / original_paragraphs, 1.0) if original_paragraphs > 0 else 1.0
        
        return 0.7 * sentence_preservation + 0.3 * paragraph_preservation
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _count_preserved_elements(self, final_text: str, key_elements: List[str]) -> int:
        """Count how many key elements are preserved in final text."""
        final_lower = final_text.lower()
        count = 0
        for element in key_elements:
            if element.lower() in final_lower:
                count += 1
        return count
    
    def _find_critical_point(self, degradation_values: List[float]) -> Optional[int]:
        """Find the point where degradation accelerates significantly."""
        if len(degradation_values) < 3:
            return None
        
        for i in range(1, len(degradation_values) - 1):
            current_rate = degradation_values[i] - degradation_values[i-1]
            next_rate = degradation_values[i+1] - degradation_values[i]
            
            if next_rate > current_rate * 1.5:  # 50% increase in degradation rate
                return i
        
        return None
    
    def _analyze_thresholds(self, degradation_values: List[float]) -> List[Dict[str, Any]]:
        """Analyze degradation thresholds."""
        thresholds = [0.2, 0.5, 0.7, 0.9]
        analysis = []
        
        for threshold in thresholds:
            for i, degradation in enumerate(degradation_values):
                if degradation >= threshold:
                    analysis.append({
                        "threshold": threshold,
                        "agent_position": i + 1,
                        "degradation_value": degradation
                    })
                    break
        
        return analysis
