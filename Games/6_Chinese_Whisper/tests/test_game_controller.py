"""
Tests for the Game Controller component of the Chinese Whisper game.
"""

import unittest
from unittest.mock import patch, MagicMock, AsyncMock, mock_open
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_controller import ChineseWhisperGameController
from models_chinese_whisper import ChineseWhisperLogger
from information_generator import InformationSeed


class TestChineseWhisperGameController(unittest.TestCase):
    """Test cases for ChineseWhisperGameController class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = ChineseWhisperLogger()
        self.controller = ChineseWhisperGameController(logger=self.logger)
        
        self.test_seed = InformationSeed(
            content="Test information about a quick brown fox jumping over a lazy dog.",
            information_type="factual",
            complexity_level="simple",
            expected_key_elements=["fox", "dog", "jumping", "brown", "lazy"],
            target_length=100
        )
    
    def test_initialization(self):
        """Test ChineseWhisperGameController initialization."""
        self.assertIsInstance(self.controller, ChineseWhisperGameController)
        self.assertIsNotNone(self.controller.logger)
        self.assertIsNotNone(self.controller.info_generator)
        self.assertIsNotNone(self.controller.evaluation_engine)
    
    def test_initialization_without_logger(self):
        """Test initialization without explicit logger."""
        controller = ChineseWhisperGameController()
        self.assertIsNotNone(controller.logger)
    
    @patch('game_controller.ChineseWhisperAgent')
    async def test_run_single_chain_basic(self, mock_agent_class):
        """Test basic single chain execution."""
        mock_agent = MagicMock()
        mock_agent.process_information = AsyncMock(return_value={
            'confidence': 0.8,
            'changes_made': 'Minor reformulation'
        })
        mock_agent.get_processed_information.return_value = "Processed test information"
        mock_agent.get_processing_summary.return_value = {
            'agent_position': 1,
            'confidence': 0.8
        }
        mock_agent_class.return_value = mock_agent
        
        result = await self.controller.run_single_chain(
            num_agents=2,
            information_seed=self.test_seed,
            model_name="test-model",
            temperature=0.7,
            generation_id=1
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('generation_id', result)
        self.assertIn('num_agents', result)
        self.assertIn('model_name', result)
        self.assertIn('temperature', result)
        self.assertIn('information_seed', result)
        self.assertIn('chain_data', result)
        self.assertIn('transmission_log', result)
        self.assertIn('evaluation_metrics', result)
        self.assertIn('agent_summaries', result)
        
        self.assertEqual(result['generation_id'], 1)
        self.assertEqual(result['num_agents'], 2)
        self.assertEqual(result['model_name'], "test-model")
        self.assertEqual(result['temperature'], 0.7)
        
        self.assertEqual(mock_agent_class.call_count, 2)
        self.assertEqual(mock_agent.process_information.call_count, 2)
    
    @patch('game_controller.ChineseWhisperAgent')
    async def test_run_single_chain_with_error_handling(self, mock_agent_class):
        """Test single chain execution with error handling."""
        mock_agent = MagicMock()
        mock_agent.process_information = AsyncMock(side_effect=Exception("Test error"))
        mock_agent.get_processed_information.return_value = "Fallback information"
        mock_agent.get_processing_summary.return_value = {'agent_position': 1}
        mock_agent_class.return_value = mock_agent
        
        result = await self.controller.run_single_chain(
            num_agents=1,
            information_seed=self.test_seed,
            generation_id=1
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('evaluation_metrics', result)
    
    @patch('game_controller.ChineseWhisperGameController.run_single_chain')
    async def test_run_batch_experiment_basic(self, mock_run_single):
        """Test basic batch experiment execution."""
        mock_result = {
            'generation_id': 1,
            'num_agents': 2,
            'model_name': 'test-model',
            'temperature': 0.7,
            'information_seed': {'information_type': 'factual', 'complexity_level': 'simple'},
            'evaluation_metrics': {
                'retention_score': 0.8,
                'semantic_similarity': 0.7,
                'factual_accuracy': 0.9,
                'structural_integrity': 0.6,
                'degradation_rate': 0.2
            }
        }
        mock_run_single.return_value = mock_result
        
        results = await self.controller.run_batch_experiment(
            agent_counts=[2],
            information_types=["factual"],
            complexity_levels=["simple"],
            model_names=["test-model"],
            temperatures=[0.7],
            runs_per_config=1
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('experiment_config', results)
        self.assertIn('total_runs', results)
        self.assertIn('individual_results', results)
        self.assertIn('batch_summary', results)
        
        self.assertEqual(results['total_runs'], 1)
        self.assertEqual(len(results['individual_results']), 1)
        mock_run_single.assert_called_once()
    
    @patch('game_controller.ChineseWhisperGameController.run_single_chain')
    async def test_run_batch_experiment_multiple_configs(self, mock_run_single):
        """Test batch experiment with multiple configurations."""
        mock_result = {
            'generation_id': 1,
            'evaluation_metrics': {
                'retention_score': 0.8,
                'semantic_similarity': 0.7,
                'factual_accuracy': 0.9,
                'structural_integrity': 0.6,
                'degradation_rate': 0.2
            }
        }
        mock_run_single.return_value = mock_result
        
        results = await self.controller.run_batch_experiment(
            agent_counts=[2, 3],
            information_types=["factual", "narrative"],
            complexity_levels=["simple"],
            model_names=["test-model"],
            temperatures=[0.7],
            runs_per_config=1
        )
        
        expected_runs = 2 * 2 * 1 * 1 * 1 * 1
        self.assertEqual(results['total_runs'], expected_runs)
        self.assertEqual(mock_run_single.call_count, expected_runs)
    
    @patch('game_controller.ChineseWhisperGameController.run_single_chain')
    async def test_run_batch_experiment_with_errors(self, mock_run_single):
        """Test batch experiment handling of errors."""
        def side_effect(*args, **kwargs):
            if kwargs.get('generation_id', 0) % 2 == 0:
                raise Exception("Test error")
            return {
                'generation_id': kwargs.get('generation_id', 1),
                'evaluation_metrics': {'retention_score': 0.8}
            }
        
        mock_run_single.side_effect = side_effect
        
        results = await self.controller.run_batch_experiment(
            agent_counts=[2],
            information_types=["factual", "narrative"],
            complexity_levels=["simple"],
            model_names=["test-model"],
            temperatures=[0.7],
            runs_per_config=1
        )
        
        self.assertLess(results['total_runs'], 2)
        self.assertGreaterEqual(results['total_runs'], 0)
    
    def test_compile_batch_summary_empty(self):
        """Test batch summary compilation with empty results."""
        summary = self.controller._compile_batch_summary([])
        self.assertEqual(summary, {})
    
    def test_compile_batch_summary_with_results(self):
        """Test batch summary compilation with results."""
        mock_results = [
            {
                'num_agents': 2,
                'information_seed': {'information_type': 'factual', 'complexity_level': 'simple'},
                'model_name': 'test-model',
                'temperature': 0.7,
                'evaluation_metrics': {
                    'retention_score': 0.8,
                    'semantic_similarity': 0.7,
                    'factual_accuracy': 0.9,
                    'structural_integrity': 0.6,
                    'degradation_rate': 0.2,
                    'key_elements_preserved': 4,
                    'total_key_elements': 5
                }
            },
            {
                'num_agents': 2,
                'information_seed': {'information_type': 'factual', 'complexity_level': 'simple'},
                'model_name': 'test-model',
                'temperature': 0.7,
                'evaluation_metrics': {
                    'retention_score': 0.6,
                    'semantic_similarity': 0.5,
                    'factual_accuracy': 0.7,
                    'structural_integrity': 0.4,
                    'degradation_rate': 0.4,
                    'key_elements_preserved': 3,
                    'total_key_elements': 5
                }
            }
        ]
        
        summary = self.controller._compile_batch_summary(mock_results)
        
        self.assertIn('total_configurations', summary)
        self.assertIn('configuration_summaries', summary)
        self.assertEqual(summary['total_configurations'], 1)  # Same config for both results
        
        config_summary = summary['configuration_summaries'][0]
        self.assertEqual(config_summary['num_runs'], 2)
        self.assertEqual(config_summary['average_metrics']['retention_score'], 0.7)  # (0.8 + 0.6) / 2
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_save_results(self, mock_json_dump, mock_open):
        """Test results saving functionality."""
        test_results = {'test': 'data'}
        test_filepath = '/test/path.json'
        
        self.controller.save_results(test_results, test_filepath)
        
        mock_open.assert_called_once_with(test_filepath, 'w')
        mock_json_dump.assert_called_once()
    
    @patch('builtins.open', side_effect=Exception("File error"))
    def test_save_results_error_handling(self, mock_open):
        """Test results saving error handling."""
        test_results = {'test': 'data'}
        test_filepath = '/invalid/path.json'
        
        self.controller.save_results(test_results, test_filepath)
        
        logs = self.controller.logger.get_logs()
        error_logged = any("Error saving results" in log for log in logs)
        self.assertTrue(error_logged)


if __name__ == '__main__':
    def run_async_test(coro):
        """Helper to run async test methods."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    
    original_test_methods = []
    for name in dir(TestChineseWhisperGameController):
        if name.startswith('test_') and asyncio.iscoroutinefunction(getattr(TestChineseWhisperGameController, name)):
            method = getattr(TestChineseWhisperGameController, name)
            original_test_methods.append((name, method))
            
            def make_sync_wrapper(async_method):
                def sync_wrapper(self):
                    return run_async_test(async_method(self))
                return sync_wrapper
            
            setattr(TestChineseWhisperGameController, name, make_sync_wrapper(method))
    
    unittest.main()
