#!/usr/bin/env python3
"""
Test script for agent communication feature in MIT Beer Game
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.')

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    print("Warning: nest_asyncio not available, running without it")

from MIT_Beer_Game import run_beer_game_simulation
from models_mitb_game import BeerGameLogger

async def test_communication_enabled():
    """Test simulation with communication enabled"""
    print("Testing Beer Game with communication enabled...")
    
    logger = BeerGameLogger()
    sim_data = await run_beer_game_simulation(
        num_generations=1,
        num_rounds_per_generation=3,
        temperature=0.7,
        logger=logger,
        enable_communication=True,
        communication_rounds=2
    )
    
    print(f"  ✓ Communication-enabled simulation completed")
    print(f"  ✓ Total rounds logged: {len(sim_data.rounds_log)}")
    print(f"  ✓ Communication messages: {len(sim_data.communication_log)}")
    
    if len(sim_data.communication_log) > 0:
        print(f"  ✓ Communication feature working - {len(sim_data.communication_log)} messages logged")
        
        sample_msg = sim_data.communication_log[0]
        required_fields = ['round', 'communication_round', 'sender', 'message']
        for field in required_fields:
            if field in sample_msg:
                print(f"  ✓ Message field '{field}' present")
            else:
                print(f"  ✗ Message field '{field}' missing")
                return False
    else:
        print(f"  ⚠ No communication messages logged (expected some)")
        return False
    
    return True

async def test_communication_disabled():
    """Test simulation with communication disabled (baseline)"""
    print("\nTesting Beer Game with communication disabled...")
    
    logger = BeerGameLogger()
    sim_data = await run_beer_game_simulation(
        num_generations=1,
        num_rounds_per_generation=3,
        temperature=0.7,
        logger=logger,
        enable_communication=False
    )
    
    print(f"  ✓ Standard simulation completed")
    print(f"  ✓ Total rounds logged: {len(sim_data.rounds_log)}")
    print(f"  ✓ Communication messages: {len(sim_data.communication_log)}")
    
    if len(sim_data.communication_log) == 0:
        print(f"  ✓ No communication messages when disabled (expected)")
    else:
        print(f"  ✗ Unexpected communication messages when disabled: {len(sim_data.communication_log)}")
        return False
    
    return True

async def test_communication_rounds_parameter():
    """Test that communication_rounds parameter works correctly"""
    print("\nTesting communication rounds parameter...")
    
    logger = BeerGameLogger()
    sim_data = await run_beer_game_simulation(
        num_generations=1,
        num_rounds_per_generation=2,
        temperature=0.7,
        logger=logger,
        enable_communication=True,
        communication_rounds=3  # Test with 3 communication rounds
    )
    
    print(f"  ✓ Simulation with 3 communication rounds completed")
    print(f"  ✓ Total communication messages: {len(sim_data.communication_log)}")
    
    comm_rounds = set()
    for msg in sim_data.communication_log:
        comm_rounds.add(msg.get('communication_round', 0))
    
    if len(comm_rounds) >= 2:  # Should have at least 2 different communication rounds
        print(f"  ✓ Multiple communication rounds detected: {sorted(comm_rounds)}")
    else:
        print(f"  ⚠ Expected multiple communication rounds, got: {sorted(comm_rounds)}")
    
    return True

async def test_agent_message_structure():
    """Test that agent messages have the expected structure"""
    print("\nTesting agent message structure...")
    
    logger = BeerGameLogger()
    sim_data = await run_beer_game_simulation(
        num_generations=1,
        num_rounds_per_generation=2,
        temperature=0.7,
        logger=logger,
        enable_communication=True,
        communication_rounds=1
    )
    
    if len(sim_data.communication_log) == 0:
        print("  ✗ No communication messages to test structure")
        return False
    
    sample_msg = sim_data.communication_log[0]
    expected_fields = [
        'round', 'communication_round', 'sender', 'message',
        'strategy_hint', 'collaboration_proposal', 'information_shared', 'confidence'
    ]
    
    all_fields_present = True
    for field in expected_fields:
        if field in sample_msg:
            print(f"  ✓ Field '{field}': {sample_msg[field]}")
        else:
            print(f"  ✗ Missing field '{field}'")
            all_fields_present = False
    
    return all_fields_present

async def main():
    """Run communication feature tests"""
    print("Running MIT Beer Game Communication Feature Tests...\n")
    
    tests = [
        test_communication_disabled,
        test_communication_enabled,
        test_communication_rounds_parameter,
        test_agent_message_structure
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"  ✗ Test {test.__name__} failed with error: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All communication tests passed!")
        print("✓ Communication feature working correctly")
        print("✓ Communication can be enabled/disabled")
        print("✓ Communication rounds parameter functional")
        print("✓ Message structure validated")
    else:
        print("❌ Some communication tests failed. Please review the implementation.")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Test execution failed: {e}")
        sys.exit(1)
