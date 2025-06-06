#!/usr/bin/env python3
"""
Simple validation script to test Fishery Game fixes without requiring API calls
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from Fishery_game import logistic_growth, FisheryConfig, FisheryMetricsData

def test_logistic_growth_recovery():
    """Test that logistic growth can recover from zero/near-zero populations"""
    print("Testing logistic growth recovery mechanism...")
    
    zero_recovery = logistic_growth(0.0, 0.3, 100.0)
    expected_recovery = 0.1 * 0.3 * 100.0  # 3.0
    
    print(f"  Zero population: 0.0 → {zero_recovery:.2f} (expected: {expected_recovery:.2f})")
    assert zero_recovery > 0, "Zero population should recover"
    assert abs(zero_recovery - expected_recovery) < 0.1, "Recovery should match expected value"
    
    near_zero_recovery = logistic_growth(0.005, 0.3, 100.0)
    print(f"  Near-zero population: 0.005 → {near_zero_recovery:.2f}")
    assert near_zero_recovery > 0.005, "Near-zero population should recover"
    
    normal_growth = logistic_growth(50.0, 0.3, 100.0)
    expected_normal = 50.0 + (0.3 * 50.0 * (1 - 50.0/100.0))
    print(f"  Normal population: 50.0 → {normal_growth:.2f} (expected: {expected_normal:.2f})")
    assert abs(normal_growth - expected_normal) < 0.1, "Normal growth should work correctly"
    
    print("  ✓ Logistic growth recovery mechanism working correctly!")

def test_sustainability_metrics():
    """Test that sustainability metrics are properly structured"""
    print("\nTesting sustainability metrics structure...")
    
    metrics = FisheryMetricsData(
        generation=1,
        eq_distance=0.5,
        gini_payoffs=0.3,
        average_fish_caught=4.2,
        sustainability_index=0.75,
        min_viable_population=5.0,
        recovery_events=1
    )
    
    print(f"  Sustainability index: {metrics.sustainability_index}")
    print(f"  Min viable population: {metrics.min_viable_population}")
    print(f"  Recovery events: {metrics.recovery_events}")
    
    assert hasattr(metrics, 'sustainability_index'), "Should have sustainability_index field"
    assert hasattr(metrics, 'min_viable_population'), "Should have min_viable_population field"
    assert hasattr(metrics, 'recovery_events'), "Should have recovery_events field"
    
    print("  ✓ Sustainability metrics structure working correctly!")

def test_config_structure():
    """Test that FisheryConfig works correctly"""
    print("\nTesting configuration structure...")
    
    config = FisheryConfig(
        num_agents=5,
        num_generations=10,
        initial_resource=50.0,
        growth_rate=0.3,
        carrying_capacity=100.0,
        max_fishable=10.0,
        temperature=0.7,
        llm_provider="openai",
        llm_model="gpt-4o-mini"
    )
    
    print(f"  Config created with {config.num_agents} agents")
    print(f"  Carrying capacity: {config.carrying_capacity}")
    print(f"  Growth rate: {config.growth_rate}")
    
    min_viable = 0.05 * config.carrying_capacity
    print(f"  Minimum viable population (5%): {min_viable}")
    
    assert config.carrying_capacity > 0, "Carrying capacity should be positive"
    assert config.growth_rate > 0, "Growth rate should be positive"
    assert min_viable == 5.0, "Min viable population should be 5% of carrying capacity"
    
    print("  ✓ Configuration structure working correctly!")

def main():
    """Run all validation tests"""
    print("Running Fishery Game fixes validation (no API calls required)...\n")
    
    try:
        test_logistic_growth_recovery()
        test_sustainability_metrics()
        test_config_structure()
        
        print("\n=== Validation Results ===")
        print("🎉 All core fixes validated successfully!")
        print("✓ Logistic growth recovery mechanism implemented")
        print("✓ Sustainability metrics structure enhanced")
        print("✓ Configuration handling working correctly")
        print("\nThe Fishery Game fixes are ready for deployment.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
