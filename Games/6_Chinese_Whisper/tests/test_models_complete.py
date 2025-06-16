"""
Tests for the Models component of the Chinese Whisper game.
"""

import unittest
from unittest.mock import patch, MagicMock, AsyncMock, mock_open
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models_chinese_whisper import ChineseWhisperAgent, ChineseWhisperLogger, TransmissionData, SimulationData


class TestChineseWhisperAgent(unittest.TestCase):
    """Test cases for ChineseWhisperAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = ChineseWhisperLogger()
        self.agent = ChineseWhisperAgent(
            agent_position=1,
            total_agents=3,
            logger=self.logger
        )
    
    def test_initialization(self):
        """Test ChineseWhisperAgent initialization."""
        self.assertEqual(self.agent.agent_position, 1)
        self.assertEqual(self.agent.total_agents, 3)
        self.assertEqual(self.agent.information_received, "")
        self.assertEqual(self.agent.information_processed, "")
        self.assertEqual(self.agent.confidence, 0.0)
        self.assertEqual(self.agent.changes_made, "")
        self.assertEqual(self.agent.key_elements_preserved, [])
        self.assertEqual(self.agent.processing_history, [])
        self.assertIsNotNone(self.agent.prompts)
        self.assertEqual(self.agent.logger, self.logger)
    
    def test_get_processed_information(self):
        """Test getting processed information."""
        self.agent.information_processed = "Test processed info"
        result = self.agent.get_processed_information()
        self.assertEqual(result, "Test processed info")
    
    def test_get_processing_summary(self):
        """Test getting processing summary."""
        self.agent.confidence = 0.8
        self.agent.changes_made = "Test changes"
        self.agent.key_elements_preserved = ["element1", "element2"]
        self.agent.processing_history = [{"test": "history"}]
        
        summary = self.agent.get_processing_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary['agent_position'], 1)
        self.assertEqual(summary['total_agents'], 3)
        self.assertEqual(summary['confidence'], 0.8)
        self.assertEqual(summary['changes_made'], "Test changes")
        self.assertEqual(summary['key_elements_preserved'], ["element1", "element2"])
        self.assertEqual(summary['processing_count'], 1)


class TestChineseWhisperLogger(unittest.TestCase):
    """Test cases for ChineseWhisperLogger class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = ChineseWhisperLogger()
    
    def test_initialization(self):
        """Test ChineseWhisperLogger initialization."""
        self.assertEqual(self.logger.logs, [])
        self.assertIsNone(self.logger.log_to_file)
        self.assertIsNone(self.logger.file_handle)
    
    @patch('builtins.open', new_callable=mock_open)
    def test_initialization_with_file(self, mock_file):
        """Test initialization with file logging."""
        logger = ChineseWhisperLogger(log_to_file="test.log")
        self.assertEqual(logger.log_to_file, "test.log")
        mock_file.assert_called_once_with("test.log", 'a')
    
    @patch('builtins.print')
    def test_log_message(self, mock_print):
        """Test logging a message."""
        self.logger.log("Test message")
        
        self.assertEqual(len(self.logger.logs), 1)
        self.assertEqual(self.logger.logs[0], "Test message")
        mock_print.assert_called_once_with("Test message")
    
    def test_get_logs(self):
        """Test getting all logs."""
        self.logger.logs = ["Log 1", "Log 2", "Log 3"]
        logs = self.logger.get_logs()
        self.assertEqual(logs, ["Log 1", "Log 2", "Log 3"])
    
    def test_close_without_file(self):
        """Test closing logger without file handle."""
        self.logger.close()


class TestTransmissionData(unittest.TestCase):
    """Test cases for TransmissionData dataclass."""
    
    def test_transmission_data_creation(self):
        """Test TransmissionData creation."""
        data = TransmissionData(
            generation=1,
            agent_position=2,
            information_type="factual",
            original_length=100,
            processed_length=95,
            confidence=0.8,
            changes_made="Minor edits",
            processing_time=1.5
        )
        
        self.assertEqual(data.generation, 1)
        self.assertEqual(data.agent_position, 2)
        self.assertEqual(data.information_type, "factual")
        self.assertEqual(data.original_length, 100)
        self.assertEqual(data.processed_length, 95)
        self.assertEqual(data.confidence, 0.8)
        self.assertEqual(data.changes_made, "Minor edits")
        self.assertEqual(data.processing_time, 1.5)


class TestSimulationData(unittest.TestCase):
    """Test cases for SimulationData dataclass."""
    
    def test_simulation_data_creation(self):
        """Test SimulationData creation."""
        hyperparams = {"num_agents": 3, "temperature": 0.7}
        data = SimulationData(hyperparameters=hyperparams)
        
        self.assertEqual(data.hyperparameters, hyperparams)
        self.assertEqual(data.transmissions_log, [])
        self.assertEqual(data.original_information, "")
        self.assertEqual(data.final_information, "")
    
    def test_add_transmission_entry(self):
        """Test adding transmission entry."""
        data = SimulationData(hyperparameters={})
        
        transmission = TransmissionData(
            generation=1,
            agent_position=1,
            information_type="test",
            original_length=50,
            processed_length=45,
            confidence=0.7,
            changes_made="Test changes",
            processing_time=1.0
        )
        
        data.add_transmission_entry(transmission)
        
        self.assertEqual(len(data.transmissions_log), 1)
        self.assertEqual(data.transmissions_log[0], transmission)
    
    def test_to_dict(self):
        """Test converting SimulationData to dictionary."""
        hyperparams = {"num_agents": 2}
        data = SimulationData(
            hyperparameters=hyperparams,
            original_information="Original",
            final_information="Final"
        )
        
        transmission = TransmissionData(
            generation=1,
            agent_position=1,
            information_type="test",
            original_length=50,
            processed_length=45,
            confidence=0.7,
            changes_made="Test changes",
            processing_time=1.0
        )
        data.add_transmission_entry(transmission)
        
        result_dict = data.to_dict()
        
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict['hyperparameters'], hyperparams)
        self.assertEqual(result_dict['original_information'], "Original")
        self.assertEqual(result_dict['final_information'], "Final")
        self.assertEqual(len(result_dict['transmissions_log']), 1)
        self.assertIsInstance(result_dict['transmissions_log'][0], dict)


if __name__ == '__main__':
    unittest.main()
