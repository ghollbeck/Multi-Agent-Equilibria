#!/usr/bin/env python3
"""
Test script to validate MIT Beer Game fixes
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(__file__))

from models_mitb_game import BeerGameAgent, BeerGameLogger
from MIT_Beer_Game import run_beer_game_generation

async def test_profit_calculation():
    """Test that profit calculations are correct"""
    print("Testing profit calculation logic...")
    
    logger = BeerGameLogger()
    agents = [BeerGameAgent(role_name=role, logger=logger) for role in ["Retailer", "Wholesaler", "Distributor", "Factory"]]
    
    for agent in agents:
        await agent.initialize_strategy(temperature=0.7, profit_per_unit_sold=5)
    
    external_demand = [16]  # Same as run_39
    
    print(f"Initial states:")
    for agent in agents:
        print(f"  {agent.role_name}: inventory={agent.inventory}, backlog={agent.backlog}, profit={agent.profit_accumulated}")
    
    await run_beer_game_generation(
        agents=agents,
        external_demand=external_demand,
        num_rounds=1,
        holding_cost_per_unit=0.5,
        backlog_cost_per_unit=1.5,
        profit_per_unit_sold=5,
        temperature=0.7,
        generation_index=1,
        logger=logger
    )
    
    print(f"\nAfter round 1:")
    for agent in agents:
        print(f"  {agent.role_name}: inventory={agent.inventory}, backlog={agent.backlog}, profit={agent.profit_accumulated}")
    
    retailer = agents[0]
    expected_holding_cost = retailer.inventory * 0.5
    expected_backlog_cost = retailer.backlog * 1.5
    print(f"\nRetailer profit validation:")
    print(f"  Holding cost: {expected_holding_cost}")
    print(f"  Backlog cost: {expected_backlog_cost}")
    print(f"  Actual profit: {retailer.profit_accumulated}")
    
    logger.close()
    return True

async def test_backlog_handling():
    """Test that backlog is handled correctly"""
    print("\nTesting backlog handling...")
    
    logger = BeerGameLogger()
    agents = [BeerGameAgent(role_name=role, logger=logger) for role in ["Retailer", "Wholesaler"]]
    
    agents[0].inventory = 10
    agents[0].backlog = 5
    agents[1].inventory = 20
    agents[1].backlog = 0
    
    print(f"Before order placement:")
    print(f"  Retailer: inventory={agents[0].inventory}, backlog={agents[0].backlog}")
    print(f"  Wholesaler: inventory={agents[1].inventory}, backlog={agents[1].backlog}")
    
    order_qty = 15
    agents[1].backlog += order_qty  # This is how orders become backlog
    
    print(f"After Retailer orders {order_qty} from Wholesaler:")
    print(f"  Retailer: inventory={agents[0].inventory}, backlog={agents[0].backlog}")
    print(f"  Wholesaler: inventory={agents[1].inventory}, backlog={agents[1].backlog}")
    
    logger.close()
    return True

if __name__ == "__main__":
    print("Running MIT Beer Game fixes validation...")
    
    async def main():
        await test_profit_calculation()
        await test_backlog_handling()
        print("\nAll tests completed!")
    
    asyncio.run(main())
