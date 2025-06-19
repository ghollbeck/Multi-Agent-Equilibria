#!/usr/bin/env python3
"""
Debug script to test communication logging functionality
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

async def debug_communication_logging():
    """Debug communication logging with minimal simulation"""
    print("=== Debug Communication Logging ===")
    
    logger = BeerGameLogger()
    
    print("Running 1 generation, 1 round with communication enabled...")
    sim_data = await run_beer_game_simulation(
        num_generations=1,
        num_rounds_per_generation=1,
        temperature=0.7,
        logger=logger,
        enable_communication=True,
        communication_rounds=1
    )
    
    print(f"\n=== Results ===")
    print(f"Rounds logged: {len(sim_data.rounds_log)}")
    print(f"Communication messages: {len(sim_data.communication_log)}")
    
    if len(sim_data.communication_log) > 0:
        print(f"\n=== Communication Messages ===")
        for i, msg in enumerate(sim_data.communication_log):
            print(f"Message {i+1}:")
            print(f"  Round: {msg.get('round', 'N/A')}")
            print(f"  Comm Round: {msg.get('communication_round', 'N/A')}")
            print(f"  Sender: {msg.get('sender', 'N/A')}")
            print(f"  Message: {msg.get('message', 'N/A')[:100]}...")
            print()
    else:
        print("❌ No communication messages found!")
    
    return sim_data

if __name__ == "__main__":
    try:
        sim_data = asyncio.run(debug_communication_logging())
        print("✅ Debug script completed successfully")
    except Exception as e:
        print(f"❌ Debug script failed: {e}")
        import traceback
        traceback.print_exc()
