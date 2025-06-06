#!/usr/bin/env python3
"""
Test script to validate Fishery Game fixes and enhancements
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(__file__))

from Fishery_game import (
    FisheryConfig, FisheryAgent, logistic_growth, 
    simulate_fishery_generation, FisherySimulationData,
    run_fishery_game_simulation
)

async def test_normal_operation():
    """Test 1: Normal operation with sustainable harvesting"""
    print("Testing normal operation...")
    
    config = FisheryConfig(
        num_agents=3,
        num_generations=5,
        initial_resource=50.0,
        growth_rate=0.3,
        carrying_capacity=100.0,
        max_fishable=5.0,
        temperature=0.5,
        llm_provider="openai",
        llm_model="gpt-4o-mini"
    )
    
    try:
        sim_data, agents = await run_fishery_game_simulation(config)
        final_resource = sim_data.resource_over_time[-1]
        print(f"  ✓ Simulation completed successfully")
        print(f"  ✓ Final resource level: {final_resource:.2f}")
        print(f"  ✓ Total interactions logged: {len(sim_data.interactions)}")
        return True
    except Exception as e:
        print(f"  ✗ Normal operation test failed: {e}")
        return False

async def test_recovery_from_near_collapse():
    """Test 2: Recovery mechanism from near-zero population"""
    print("\nTesting recovery from near-collapse...")
    
    near_zero_stock = 0.005
    growth_rate = 0.3
    carrying_capacity = 100.0
    
    new_stock = logistic_growth(near_zero_stock, growth_rate, carrying_capacity)
    expected_recovery = 0.1 * growth_rate * carrying_capacity  # 3.0
    
    if new_stock > near_zero_stock and abs(new_stock - expected_recovery) < 0.1:
        print(f"  ✓ Recovery mechanism works: {near_zero_stock:.3f} → {new_stock:.2f}")
    else:
        print(f"  ✗ Recovery mechanism failed: {near_zero_stock:.3f} → {new_stock:.2f}")
        return False
    
    zero_stock = 0.0
    new_stock_from_zero = logistic_growth(zero_stock, growth_rate, carrying_capacity)
    if new_stock_from_zero > 0:
        print(f"  ✓ Zero population recovery works: {zero_stock} → {new_stock_from_zero:.2f}")
        return True
    else:
        print(f"  ✗ Zero population recovery failed: {zero_stock} → {new_stock_from_zero}")
        return False

async def test_overharvesting_scenarios():
    """Test 3: Minimum viable population protection"""
    print("\nTesting overharvesting protection...")
    
    config = FisheryConfig(
        num_agents=5,
        num_generations=1,
        initial_resource=20.0,  # Low initial resource
        growth_rate=0.3,
        carrying_capacity=100.0,
        max_fishable=15.0,  # High max fishable to encourage overharvesting
        temperature=0.8,
        llm_provider="openai",
        llm_model="gpt-4o-mini"
    )
    
    sim_data = FisherySimulationData(config=config)
    agents = []
    for i in range(config.num_agents):
        agent = FisheryAgent(
            name=f"TestAgent_{i+1}",
            model=config.llm_model,
            llm_provider=config.llm_provider,
            llm_model=config.llm_model
        )
        agents.append(agent)
    
    try:
        final_resource = await simulate_fishery_generation(
            agents=agents,
            resource_stock=config.initial_resource,
            config=config,
            generation_index=1,
            sim_data=sim_data
        )
        
        min_viable_pop = 0.05 * config.carrying_capacity  # 5.0
        if final_resource >= min_viable_pop * 0.8:  # Allow some tolerance
            print(f"  ✓ Minimum viable population protected: {final_resource:.2f} >= {min_viable_pop:.2f}")
            return True
        else:
            print(f"  ✗ Minimum viable population not protected: {final_resource:.2f} < {min_viable_pop:.2f}")
            return False
    except Exception as e:
        print(f"  ✗ Overharvesting test failed: {e}")
        return False

async def test_edge_cases():
    """Test 4: Edge cases (zero population, extreme demand)"""
    print("\nTesting edge cases...")
    
    zero_growth = logistic_growth(0.0, 0.3, 100.0)
    if zero_growth > 0:
        print(f"  ✓ Zero population recovery: 0.0 → {zero_growth:.2f}")
    else:
        print(f"  ✗ Zero population recovery failed: 0.0 → {zero_growth}")
        return False
    
    high_capacity_growth = logistic_growth(50.0, 0.3, 1000.0)
    expected_high = 50.0 + (0.3 * 50.0 * (1 - 50.0/1000.0))
    if abs(high_capacity_growth - expected_high) < 0.1:
        print(f"  ✓ High carrying capacity works: {high_capacity_growth:.2f}")
    else:
        print(f"  ✗ High carrying capacity failed: {high_capacity_growth:.2f} vs {expected_high:.2f}")
        return False
    
    at_capacity_growth = logistic_growth(100.0, 0.3, 100.0)
    if abs(at_capacity_growth - 100.0) < 0.1:
        print(f"  ✓ At carrying capacity stable: {at_capacity_growth:.2f}")
        return True
    else:
        print(f"  ✗ At carrying capacity unstable: {at_capacity_growth:.2f}")
        return False

async def test_sustainability_metrics():
    """Test 5: Sustainability metrics calculation"""
    print("\nTesting sustainability metrics...")
    
    config = FisheryConfig(
        num_agents=2,
        num_generations=2,
        initial_resource=75.0,
        growth_rate=0.3,
        carrying_capacity=100.0,
        max_fishable=5.0,
        temperature=0.5,
        llm_provider="openai",
        llm_model="gpt-4o-mini"
    )
    
    try:
        sim_data, agents = await run_fishery_game_simulation(config)
        
        if len(sim_data.per_generation_metrics) > 0:
            metrics = sim_data.per_generation_metrics[0]
            if hasattr(metrics, 'sustainability_index') and hasattr(metrics, 'min_viable_population'):
                expected_sustainability = 75.0 / 100.0  # 0.75
                if abs(metrics.sustainability_index - expected_sustainability) < 0.1:
                    print(f"  ✓ Sustainability index calculated: {metrics.sustainability_index:.2f}")
                    print(f"  ✓ Min viable population tracked: {metrics.min_viable_population:.2f}")
                    return True
                else:
                    print(f"  ✗ Sustainability index incorrect: {metrics.sustainability_index:.2f}")
                    return False
            else:
                print(f"  ✗ Sustainability metrics missing from data structure")
                return False
        else:
            print(f"  ✗ No metrics data generated")
            return False
    except Exception as e:
        print(f"  ✗ Sustainability metrics test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("Running Fishery Game fixes validation...\n")
    
    tests = [
        test_normal_operation,
        test_recovery_from_near_collapse,
        test_overharvesting_scenarios,
        test_edge_cases,
        test_sustainability_metrics
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
        print("🎉 All tests passed! Fishery Game fixes are working correctly.")
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
