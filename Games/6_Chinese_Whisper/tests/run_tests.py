"""
Test runner for Chinese Whisper Game tests.
"""

import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_information_generator import TestInformationGenerator, TestInformationSeed
from test_evaluation_engine import TestEvaluationEngine, TestEvaluationMetrics
from test_game_controller import TestChineseWhisperGameController
from test_models_complete import TestChineseWhisperAgent, TestChineseWhisperLogger, TestTransmissionData, TestSimulationData
from test_analytics import TestChineseWhisperAnalytics


def create_test_suite():
    """Create a comprehensive test suite for all components."""
    suite = unittest.TestSuite()
    
    suite.addTest(unittest.makeSuite(TestInformationGenerator))
    suite.addTest(unittest.makeSuite(TestInformationSeed))
    
    suite.addTest(unittest.makeSuite(TestEvaluationEngine))
    suite.addTest(unittest.makeSuite(TestEvaluationMetrics))
    
    suite.addTest(unittest.makeSuite(TestChineseWhisperGameController))
    
    suite.addTest(unittest.makeSuite(TestChineseWhisperAgent))
    suite.addTest(unittest.makeSuite(TestChineseWhisperLogger))
    suite.addTest(unittest.makeSuite(TestTransmissionData))
    suite.addTest(unittest.makeSuite(TestSimulationData))
    
    suite.addTest(unittest.makeSuite(TestChineseWhisperAnalytics))
    
    return suite


def run_all_tests():
    """Run all tests and return results."""
    print("🧪 Running Chinese Whisper Game Test Suite")
    print("=" * 50)
    
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        return True
    else:
        print(f"\n❌ {len(result.failures + result.errors)} test(s) failed")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
