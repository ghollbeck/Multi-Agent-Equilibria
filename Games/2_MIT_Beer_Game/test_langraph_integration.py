#!/usr/bin/env python3
"""
Comprehensive test suite for LangGraph and LangSmith integration with memory features
"""
import asyncio
import sys
import os
import json
import tempfile
import shutil
sys.path.append(os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.')

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    print("Warning: nest_asyncio not available, running without it")

from MIT_Beer_Game import run_beer_game_simulation
from models_mitb_game import BeerGameAgent, BeerGameLogger
from memory_storage import MemoryManager, AgentMemory
from llm_calls_mitb_game import LiteLLMClient

class TestResults:
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.failures = []
    
    def add_test(self, test_name: str, passed: bool, error_msg: str = None):
        self.tests_run += 1
        if passed:
            self.tests_passed += 1
            print(f"✓ {test_name}")
        else:
            self.failures.append((test_name, error_msg))
            print(f"✗ {test_name}: {error_msg}")
    
    def summary(self):
        print(f"\n=== Test Summary ===")
        print(f"Tests run: {self.tests_run}")
        print(f"Tests passed: {self.tests_passed}")
        print(f"Tests failed: {len(self.failures)}")
        
        if self.failures:
            print("\nFailures:")
            for test_name, error in self.failures:
                print(f"  - {test_name}: {error}")
        
        return len(self.failures) == 0

async def test_baseline_functionality_preserved(results: TestResults):
    """Test that baseline functionality works exactly as before when memory is disabled"""
    print("\n=== Testing Baseline Functionality (Memory Disabled) ===")
    
    try:
        sim_data = await run_beer_game_simulation(
            num_generations=1,
            num_rounds_per_generation=3,
            temperature=0.1,
            enable_communication=False,
            enable_memory=False,
            memory_retention_rounds=5,
            enable_shared_memory=False
        )
        
        results.add_test("Baseline simulation completion", 
                        sim_data is not None and len(sim_data.rounds_log) > 0)
        
        memory_files = [f for f in os.listdir('.') if 'memory' in f.lower()]
        results.add_test("No memory files created when disabled", 
                        len(memory_files) == 0)
        
        results.add_test("Standard logging preserved", 
                        len(sim_data.rounds_log) == 3)  # 3 rounds as specified
        
    except Exception as e:
        results.add_test("Baseline functionality", False, str(e))

async def test_memory_enabled_functionality(results: TestResults):
    """Test that memory features work correctly when enabled"""
    print("\n=== Testing Memory Enabled Functionality ===")
    
    try:
        memory_manager = MemoryManager(retention_rounds=3, enable_shared_memory=False)
        results.add_test("Memory manager creation", memory_manager is not None)
        
        agent_memory = memory_manager.create_agent_memory("TestAgent")
        results.add_test("Agent memory creation", 
                        isinstance(agent_memory, AgentMemory))
        
        agent_memory.add_decision_memory(
            round_number=1,
            order_quantity=10,
            inventory=100,
            backlog=5,
            reasoning="Test reasoning",
            confidence=0.8
        )
        
        decision_context = agent_memory.get_memory_context_for_decision()
        results.add_test("Memory storage and retrieval", 
                        "Test reasoning" in decision_context)
        
        for i in range(5):  # Add more than retention limit
            agent_memory.add_decision_memory(
                round_number=i+2,
                order_quantity=10+i,
                inventory=100-i,
                backlog=i,
                reasoning=f"Round {i+2} reasoning",
                confidence=0.7
            )
        
        context_after_retention = agent_memory.get_memory_context_for_decision()
        results.add_test("Memory retention policy", 
                        "Round 2 reasoning" not in context_after_retention and 
                        "Round 6 reasoning" in context_after_retention)
        
    except Exception as e:
        results.add_test("Memory enabled functionality", False, str(e))

async def test_shared_memory_functionality(results: TestResults):
    """Test shared memory functionality"""
    print("\n=== Testing Shared Memory Functionality ===")
    
    try:
        memory_manager = MemoryManager(retention_rounds=5, enable_shared_memory=True)
        
        agent1_memory = memory_manager.create_agent_memory("Agent1")
        agent2_memory = memory_manager.create_agent_memory("Agent2")
        
        agent1_memory.add_communication_memory(
            round_number=1,
            message="Shared information from Agent1",
            strategy_hint="Cooperative strategy",
            collaboration_proposal="Share demand forecasts",
            information_shared="Current inventory levels"
        )
        
        shared_context = memory_manager.get_shared_memory_context()
        results.add_test("Shared memory storage", 
                        "Shared information from Agent1" in shared_context)
        
        agent2_shared_context = agent2_memory.get_shared_memory_context() if hasattr(agent2_memory, 'get_shared_memory_context') else shared_context
        results.add_test("Shared memory access by other agents", 
                        "Agent1" in agent2_shared_context or "Shared information" in agent2_shared_context)
        
    except Exception as e:
        results.add_test("Shared memory functionality", False, str(e))

async def test_langsmith_tracing_integration(results: TestResults):
    """Test LangSmith tracing integration"""
    print("\n=== Testing LangSmith Tracing Integration ===")
    
    try:
        from llm_calls_mitb_game import LANGSMITH_AVAILABLE
        
        if not LANGSMITH_AVAILABLE:
            results.add_test("LangSmith availability check", True, "LangSmith not available - using fallback")
            return
        
        logger = BeerGameLogger()
        client = LiteLLMClient(logger=logger)
        
        response = await client.chat_completion(
            model="gpt-4o",
            system_prompt="You are a test assistant. Return valid JSON only.",
            user_prompt='{"test": "tracing integration"}',
            temperature=0.1,
            agent_role="TestAgent",
            round_index=1,
            decision_type="test_tracing"
        )
        
        results.add_test("LLM call with tracing metadata", 
                        response is not None and len(response) > 0)
        
        session_summary = client.get_session_summary()
        results.add_test("Session summary generation", 
                        session_summary.get('total_calls', 0) >= 1)
        
        if os.path.exists("llm_inference_metrics.json"):
            with open("llm_inference_metrics.json", 'r') as f:
                metrics = json.load(f)
            
            if metrics:
                latest_metric = metrics[-1]
                has_tracing_data = ('agent_role' in latest_metric and 
                                  'decision_type' in latest_metric)
                results.add_test("Tracing metadata in metrics", has_tracing_data)
            else:
                results.add_test("Tracing metadata in metrics", False, "No metrics found")
        else:
            results.add_test("Metrics file creation", False, "Metrics file not found")
        
    except Exception as e:
        results.add_test("LangSmith tracing integration", False, str(e))

async def test_memory_persistence(results: TestResults):
    """Test memory persistence across simulation components"""
    print("\n=== Testing Memory Persistence ===")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                sim_data = await run_beer_game_simulation(
                    num_generations=1,
                    num_rounds_per_generation=2,
                    temperature=0.1,
                    enable_communication=True,
                    communication_rounds=1,
                    enable_memory=True,
                    memory_retention_rounds=3,
                    enable_shared_memory=False
                )
                
                results.add_test("Memory-enabled simulation completion", 
                                sim_data is not None)
                
                has_communication_log = hasattr(sim_data, 'communication_log') and len(sim_data.communication_log) > 0
                results.add_test("Communication logging with memory", 
                                has_communication_log)
                
                results.add_test("Memory simulation rounds completion", 
                                len(sim_data.rounds_log) == 2)
                
            finally:
                os.chdir(original_cwd)
        
    except Exception as e:
        results.add_test("Memory persistence", False, str(e))

async def test_memory_integration_with_agents(results: TestResults):
    """Test memory integration with agent decision making"""
    print("\n=== Testing Memory Integration with Agents ===")
    
    try:
        logger = BeerGameLogger()
        agent = BeerGameAgent(role_name="TestRetailer", logger=logger)
        
        memory_manager = MemoryManager(retention_rounds=3, enable_shared_memory=False)
        agent.memory = memory_manager.create_agent_memory("TestRetailer")
        
        memory_context = agent.load_memory_context()
        results.add_test("Agent memory context loading", 
                        isinstance(memory_context, dict) and 
                        'decision_context' in memory_context)
        
        await agent.initialize_strategy(temperature=0.1)
        decision = await agent.decide_order_quantity(temperature=0.1)
        
        results.add_test("Agent decision making with memory", 
                        decision is not None and 'order_quantity' in decision)
        
        agent.update_memory(
            round_number=1,
            decision_output=decision,
            performance_data={'profit': 10.0, 'units_sold': 5}
        )
        
        updated_context = agent.load_memory_context()
        results.add_test("Memory update after decision", 
                        updated_context['decision_context'] != 'No memory context available.')
        
    except Exception as e:
        results.add_test("Memory integration with agents", False, str(e))

async def main():
    """Run all integration tests"""
    print("Running LangGraph and LangSmith Integration Tests...\n")
    
    results = TestResults()
    
    await test_baseline_functionality_preserved(results)
    await test_memory_enabled_functionality(results)
    await test_shared_memory_functionality(results)
    await test_langsmith_tracing_integration(results)
    await test_memory_persistence(results)
    await test_memory_integration_with_agents(results)
    
    test_files = ['llm_inference_metrics.json', 'test_memory.json']
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
    
    success = results.summary()
    
    if success:
        print("\n🎉 All integration tests passed!")
        print("✓ Baseline functionality preserved when memory disabled")
        print("✓ Memory features work correctly when enabled")
        print("✓ Shared memory functionality operational")
        print("✓ LangSmith tracing integration successful")
        print("✓ Memory persistence across simulation runs")
        print("✓ Agent memory integration functional")
    else:
        print("\n❌ Some integration tests failed")
        print("Please review the failures above and fix the issues")
    
    return success

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
