"""
Tests for the Analytics component of the Chinese Whisper game.
"""

import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics import ChineseWhisperAnalytics


class TestChineseWhisperAnalytics(unittest.TestCase):
    """Test cases for ChineseWhisperAnalytics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_results = {
            'individual_results': [
                {
                    'generation_id': 1,
                    'num_agents': 3,
                    'model_name': 'test-model',
                    'temperature': 0.7,
                    'information_seed': {
                        'information_type': 'factual',
                        'complexity_level': 'simple'
                    },
                    'evaluation_metrics': {
                        'retention_score': 0.8,
                        'semantic_similarity': 0.75,
                        'factual_accuracy': 0.9,
                        'structural_integrity': 0.7,
                        'information_completeness': 0.85,
                        'degradation_rate': 0.2,
                        'key_elements_preserved': 4,
                        'total_key_elements': 5
                    }
                },
                {
                    'generation_id': 2,
                    'num_agents': 5,
                    'model_name': 'test-model',
                    'temperature': 0.5,
                    'information_seed': {
                        'information_type': 'narrative',
                        'complexity_level': 'medium'
                    },
                    'evaluation_metrics': {
                        'retention_score': 0.6,
                        'semantic_similarity': 0.65,
                        'factual_accuracy': 0.7,
                        'structural_integrity': 0.5,
                        'information_completeness': 0.6,
                        'degradation_rate': 0.4,
                        'key_elements_preserved': 2,
                        'total_key_elements': 4
                    }
                }
            ]
        }
        
        self.analytics = ChineseWhisperAnalytics(self.sample_results)
    
    def test_initialization_with_results(self):
        """Test ChineseWhisperAnalytics initialization with results."""
        self.assertIsInstance(self.analytics, ChineseWhisperAnalytics)
        self.assertIsNotNone(self.analytics.df)
        self.assertEqual(len(self.analytics.df), 2)
    
    def test_initialization_without_results(self):
        """Test initialization without results."""
        analytics = ChineseWhisperAnalytics()
        self.assertIsInstance(analytics, ChineseWhisperAnalytics)
        self.assertIsNone(analytics.df)
    
    @patch('builtins.open', new_callable=mock_open, read_data='{"individual_results": []}')
    @patch('json.load')
    def test_load_results_from_file(self, mock_json_load, mock_file):
        """Test loading results from file."""
        mock_json_load.return_value = self.sample_results
        
        analytics = ChineseWhisperAnalytics()
        analytics.load_results('test_file.json')
        
        mock_file.assert_called_once_with('test_file.json', 'r')
        mock_json_load.assert_called_once()
        self.assertIsNotNone(analytics.df)
    
    def test_create_dataframe(self):
        """Test DataFrame creation from results."""
        df = self.analytics._create_dataframe(self.sample_results)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        
        required_columns = [
            'generation_id', 'num_agents', 'model_name', 'temperature',
            'information_type', 'complexity_level', 'retention_score',
            'semantic_similarity', 'factual_accuracy', 'structural_integrity'
        ]
        for col in required_columns:
            self.assertIn(col, df.columns)
    
    def test_generate_summary_report(self):
        """Test summary report generation."""
        report = self.analytics.generate_summary_report()
        
        self.assertIsInstance(report, dict)
        self.assertIn('dataset_overview', report)
        self.assertIn('overall_performance', report)
        self.assertIn('chain_length_analysis', report)
        self.assertIn('information_type_analysis', report)
        self.assertIn('best_configurations', report)
        self.assertIn('worst_configurations', report)
    
    def test_analyze_chain_length_impact(self):
        """Test chain length impact analysis."""
        analysis = self.analytics._analyze_chain_length_impact()
        
        self.assertIsInstance(analysis, dict)
        self.assertIn('3 agents', analysis)
        self.assertIn('5 agents', analysis)
        
        for key, stats in analysis.items():
            self.assertIn('count', stats)
            self.assertIn('avg_retention', stats)
            self.assertIn('avg_semantic_similarity', stats)
            self.assertIn('avg_factual_accuracy', stats)
    
    def test_analyze_information_type_impact(self):
        """Test information type impact analysis."""
        analysis = self.analytics._analyze_information_type_impact()
        
        self.assertIsInstance(analysis, dict)
        self.assertIn('factual', analysis)
        self.assertIn('narrative', analysis)
        
        for info_type, stats in analysis.items():
            self.assertIn('count', stats)
            self.assertIn('avg_retention', stats)
            self.assertIn('most_vulnerable_metric', stats)
    
    def test_identify_best_worst_configurations(self):
        """Test identification of best and worst configurations."""
        best, worst = self.analytics._identify_best_worst_configurations()
        
        self.assertIsInstance(best, list)
        self.assertIsInstance(worst, list)
        self.assertEqual(len(best), 2)  # Should have 2 configurations
        self.assertEqual(len(worst), 2)
        
        if best and worst:
            self.assertGreaterEqual(best[0]['composite_score'], worst[-1]['composite_score'])
    
    def test_calculate_composite_score(self):
        """Test composite score calculation."""
        metrics = {
            'retention_score': 0.8,
            'semantic_similarity': 0.7,
            'factual_accuracy': 0.9,
            'structural_integrity': 0.6
        }
        
        score = self.analytics._calculate_composite_score(metrics)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        expected = (0.8 * 0.3 + 0.7 * 0.3 + 0.9 * 0.25 + 0.6 * 0.15)
        self.assertAlmostEqual(score, expected, places=3)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_plot_degradation_by_chain_length(self, mock_show, mock_savefig):
        """Test degradation plot generation."""
        fig = self.analytics.plot_degradation_by_chain_length()
        
        self.assertIsNotNone(fig)
        mock_show.assert_not_called()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_plot_information_type_comparison(self, mock_show, mock_savefig):
        """Test information type comparison plot."""
        fig = self.analytics.plot_information_type_comparison()
        
        self.assertIsNotNone(fig)
        mock_show.assert_not_called()
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_save_report(self, mock_json_dump, mock_file):
        """Test report saving."""
        test_report = {'test': 'data'}
        test_filepath = 'test_report.json'
        
        self.analytics.save_report(test_report, test_filepath)
        
        mock_file.assert_called_once_with(test_filepath, 'w')
        mock_json_dump.assert_called_once_with(test_report, mock_file.return_value, indent=2)
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty results."""
        empty_results = {'individual_results': []}
        analytics = ChineseWhisperAnalytics(empty_results)
        
        report = analytics.generate_summary_report()
        self.assertIsInstance(report, dict)
        
        fig = analytics.plot_degradation_by_chain_length()
        self.assertIsNone(fig)


if __name__ == '__main__':
    unittest.main()
