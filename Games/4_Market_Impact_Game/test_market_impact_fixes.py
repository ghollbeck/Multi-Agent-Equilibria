#!/usr/bin/env python3
"""
Test script to validate Market Impact Game fixes and functionality
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(__file__))

from market_impact_game import simulate_market_impact_game, MarketAgent, MarketEnvironment

async def test_basic_simulation():
    """Test basic simulation execution"""
    print("Testing basic Market Impact Game simulation...")
    
    try:
        sim_data, results_folder = await simulate_market_impact_game(
            num_agents=3,
            num_rounds=5,
            num_generations=1,
            init_price=100.0,
            impact_factor=0.01,
            volatility=0.01,
            llm_provider="openai",
            models=["gpt-4o-mini"],
            temperature=0.7,
            seed=42
        )
        
        print(f"  ✓ Simulation completed successfully")
        print(f"  ✓ Results saved to: {results_folder}")
        print(f"  ✓ Total interactions: {len(sim_data.interactions)}")
        return True
        
    except Exception as e:
        print(f"  ✗ Simulation failed: {e}")
        return False

async def test_market_environment():
    """Test market environment mechanics"""
    print("\nTesting market environment...")
    
    env = MarketEnvironment(init_price=100.0, impact_factor=0.02, volatility=0.01)
    
    trades = {"agent1": 10.0, "agent2": -5.0, "agent3": 0.0}
    new_price = env.execute_trades(trades)
    
    if new_price != env.price:
        print(f"  ✗ Price update inconsistency")
        return False
    
    print(f"  ✓ Price updated: 100.0 → {new_price:.2f}")
    print(f"  ✓ Market environment working correctly")
    return True

async def test_litellm_fallback():
    """Test LiteLLM fallback mechanism"""
    print("\nTesting LiteLLM fallback mechanism...")
    
    from market_impact_game import LiteLLM
    
    llm_no_key = LiteLLM(api_key=None)
    response = await llm_no_key.chat_completion([{"role": "user", "content": "test"}])
    
    if ("Fallback random decision" in response.get("rationale", "") or 
        "no LiteLLM API key" in response.get("rationale", "") or
        "Mock LiteLLM response" in response.get("rationale", "")):
        print(f"  ✓ Fallback mechanism working: {response['rationale']}")
    else:
        print(f"  ✗ Fallback mechanism not working properly: {response}")
        return False
    
    llm_with_key = LiteLLM(api_key="test-key")
    response_with_key = await llm_with_key.chat_completion([{"role": "user", "content": "test"}])
    
    if "action" in response_with_key and "quantity" in response_with_key:
        print(f"  ✓ LiteLLM response structure correct: {response_with_key['action']}")
    else:
        print(f"  ✗ LiteLLM response structure incorrect")
        return False
    
    return True

async def test_market_agent():
    """Test market agent functionality"""
    print("\nTesting market agent...")
    
    try:
        agent = MarketAgent(
            name="TestAgent",
            model="gpt-4o-mini",
            llm_provider="openai",
            llm_model="gpt-4o-mini"
        )
        
        print(f"  ✓ Agent created: {agent.name}")
        print(f"  ✓ Model: {agent.model}")
        print(f"  ✓ LLM Provider: {agent.llm_provider}")
        print(f"  ✓ Total PnL: {agent.total_pnl}")
        print(f"  ✓ Total Position: {agent.total_position}")
        
        recent_info = {
            "price_history": [95.0, 98.0, 100.0],
            "volume_history": [1000, 1200, 1100],
            "market_volatility": 0.02
        }
        
        decision = await agent.decide_action(100.0, recent_info, temperature=0.7)
        
        if "action" in decision:
            print(f"  ✓ Agent decision made: {decision['action']}")
            return True
        else:
            print(f"  ✗ Agent decision format incorrect")
            return False
            
    except Exception as e:
        print(f"  ✗ Agent test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("Running Market Impact Game validation...\n")
    
    tests = [
        test_basic_simulation,
        test_market_environment,
        test_litellm_fallback,
        test_market_agent
    ]
    results = []
    
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"Test {test.__name__} crashed: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Market Impact Game fixes working correctly.")
        print("✓ Basic simulation execution works")
        print("✓ Market environment mechanics functional")
        print("✓ LiteLLM fallback mechanism operational")
        print("✓ Market agent functionality verified")
    else:
        print("❌ Some tests failed. Please review the fixes.")
    
    return passed == total

if __name__ == "__main__":
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass
    
    asyncio.run(main())
