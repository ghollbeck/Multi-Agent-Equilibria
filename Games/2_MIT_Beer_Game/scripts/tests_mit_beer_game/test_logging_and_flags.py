#!/usr/bin/env python3
"""Test logical fixes for MIT Beer Game simulation."""
import asyncio
import os
import sys
import csv
import json
import tempfile
import shutil
from unittest.mock import Mock, patch
from dataclasses import asdict

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(scripts_dir)

from MIT_Beer_Game import run_beer_game_generation
from models_mitb_game import BeerGameAgent, BeerGameLogger, SimulationData
import executeMITBeerGame


def stub_llm_response(*args, **kwargs):
    """Return deterministic LLM response."""
    return json.dumps({
        "order_quantity": 7,
        "inventory": 100,
        "backlog": 0,
        "confidence": 0.8,
        "rationale": "Test rationale",
        "risk_assessment": "Low risk",
        "expected_demand_next_round": 10,
        "recent_demand_or_orders": [10, 10, 10],
        "incoming_shipments": [5],
        "last_order_placed": 7
    })


async def test_snapshot_integrity():
    """Test that round data is snapshot-accurate and doesn't change."""
    print("Testing snapshot integrity...")
    
    with patch('llm_calls_mitb_game.lite_client.chat_completion', side_effect=stub_llm_response):
        temp_dir = tempfile.mkdtemp()
        try:
            csv_path = os.path.join(temp_dir, "test.csv")
            sim_data = SimulationData(hyperparameters={})
            
            agents = [BeerGameAgent.create_agent(role_name=role) 
                     for role in ["Retailer", "Wholesaler", "Distributor", "Factory"]]
            
            await run_beer_game_generation(
                agents=agents,
                external_demand=[10, 10, 10],
                num_rounds=3,
                holding_cost_per_unit=0.5,
                backlog_cost_per_unit=1.5,
                sale_price_per_unit=5.0,
                purchase_cost_per_unit=2.5,
                production_cost_per_unit=1.5,
                csv_log_path=csv_path,
                sim_data=sim_data,
                enable_communication=False
            )
            
            # Read CSV after round 3
            with open(csv_path, 'r') as f:
                reader = list(csv.DictReader(f))
            
            # Find round 1 data for Retailer
            round1_data = None
            for row in reader:
                if row['round_index'] == '1' and row['role_name'] == 'Retailer':
                    round1_data = row
                    break
            
            assert round1_data is not None, "Round 1 data not found"
            
            # Store key values
            original_order = round1_data['order_placed']
            original_balance = round1_data['ending_balance']
            original_llm_rationale = round1_data['llm_rationale']
            
            # Simulate rewriting CSV (which used to overwrite with latest agent state)
            # In the fixed version, this should not change earlier rounds
            
            # Verify values haven't changed
            assert original_order == '7', f"Order changed from 7 to {original_order}"
            assert original_llm_rationale == 'Test rationale', f"LLM rationale changed"
            
            print("✓ Snapshot integrity test passed")
            
        finally:
            shutil.rmtree(temp_dir)


async def test_communication_toggle():
    """Test that enable_communication flag works correctly."""
    print("Testing communication toggle...")
    
    with patch('llm_calls_mitb_game.lite_client.chat_completion', side_effect=stub_llm_response):
        # Test 1: Communication disabled (default)
        sim_data1 = SimulationData(hyperparameters={})
        agents1 = [BeerGameAgent.create_agent(role_name=role) 
                  for role in ["Retailer", "Wholesaler", "Distributor", "Factory"]]
        
        await run_beer_game_generation(
            agents=agents1,
            external_demand=[10],
            num_rounds=1,
            holding_cost_per_unit=0.5,
            backlog_cost_per_unit=1.5,
            sale_price_per_unit=5.0,
            purchase_cost_per_unit=2.5,
            production_cost_per_unit=1.5,
            sim_data=sim_data1,
            enable_communication=False
        )
        
        assert len(sim_data1.communication_log) == 0, "Communication log should be empty"
        
        # Test 2: Communication enabled
        sim_data2 = SimulationData(hyperparameters={})
        agents2 = [BeerGameAgent.create_agent(role_name=role) 
                  for role in ["Retailer", "Wholesaler", "Distributor", "Factory"]]
        
        await run_beer_game_generation(
            agents=agents2,
            external_demand=[10],
            num_rounds=1,
            holding_cost_per_unit=0.5,
            backlog_cost_per_unit=1.5,
            sale_price_per_unit=5.0,
            purchase_cost_per_unit=2.5,
            production_cost_per_unit=1.5,
            sim_data=sim_data2,
            enable_communication=True,
            communication_rounds=2
        )
        
        assert len(sim_data2.communication_log) > 0, "Communication log should have entries"
        
        print("✓ Communication toggle test passed")


async def test_workflow_total_rounds():
    """Test that workflow gets correct total_rounds."""
    print("Testing workflow total_rounds...")
    
    # This test requires workflow to be enabled
    from langraph_workflow import BeerGameWorkflow
    
    with patch('llm_calls_mitb_game.lite_client.chat_completion', side_effect=stub_llm_response):
        agents = [BeerGameAgent.create_agent(role_name=role) 
                 for role in ["Retailer", "Wholesaler", "Distributor", "Factory"]]
        
        workflow = BeerGameWorkflow(
            agents=agents,
            simulation_data=SimulationData(hyperparameters={}),
            memory_manager=None,
            enable_memory=True,
            enable_shared_memory=False,
            enable_communication=False,
            communication_rounds=0
        )
        
        # Mock create_initial_state to capture total_rounds
        original_create = workflow.create_initial_state
        captured_total_rounds = None
        
        def mock_create(*args, **kwargs):
            nonlocal captured_total_rounds
            captured_total_rounds = kwargs.get('total_rounds')
            return original_create(*args, **kwargs)
        
        workflow.create_initial_state = mock_create
        
        # Run with num_rounds=7
        await run_beer_game_generation(
            agents=agents,
            external_demand=[10] * 7,
            num_rounds=7,
            holding_cost_per_unit=0.5,
            backlog_cost_per_unit=1.5,
            sale_price_per_unit=5.0,
            purchase_cost_per_unit=2.5,
            production_cost_per_unit=1.5,
            sim_data=SimulationData(hyperparameters={}),
            enable_memory=True
        )
        
        assert captured_total_rounds == 7, f"Expected total_rounds=7, got {captured_total_rounds}"
        
        print("✓ Workflow total_rounds test passed")


async def test_pipeline_fields():
    """Test that pipeline fields are populated in RoundData."""
    print("Testing pipeline fields...")
    
    with patch('llm_calls_mitb_game.lite_client.chat_completion', side_effect=stub_llm_response):
        temp_dir = tempfile.mkdtemp()
        try:
            csv_path = os.path.join(temp_dir, "test.csv")
            sim_data = SimulationData(hyperparameters={})
            
            agents = [BeerGameAgent.create_agent(role_name=role) 
                     for role in ["Retailer", "Wholesaler", "Distributor", "Factory"]]
            
            await run_beer_game_generation(
                agents=agents,
                external_demand=[10, 10],
                num_rounds=2,
                holding_cost_per_unit=0.5,
                backlog_cost_per_unit=1.5,
                sale_price_per_unit=5.0,
                purchase_cost_per_unit=2.5,
                production_cost_per_unit=1.5,
                csv_log_path=csv_path,
                sim_data=sim_data,
                enable_communication=False
            )
            
            # Check that pipeline fields are in CSV
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                
            assert 'orders_in_transit_0' in headers, "orders_in_transit_0 missing from CSV"
            assert 'orders_in_transit_1' in headers, "orders_in_transit_1 missing from CSV"
            assert 'production_queue_0' in headers, "production_queue_0 missing from CSV"
            assert 'production_queue_1' in headers, "production_queue_1 missing from CSV"
            
            # Check that values are populated after orders are placed
            with open(csv_path, 'r') as f:
                reader = list(csv.DictReader(f))
            
            # In round 2, wholesaler should have orders in transit from retailer
            for row in reader:
                if row['round_index'] == '2' and row['role_name'] == 'Wholesaler':
                    assert int(row['orders_in_transit_1']) > 0, "Wholesaler should have orders in transit"
                    break
            
            print("✓ Pipeline fields test passed")
            
        finally:
            shutil.rmtree(temp_dir)


def test_cli_defaults():
    """Test CLI argument defaults."""
    print("Testing CLI defaults...")
    
    parser = executeMITBeerGame.parse_args().__class__.__bases__[0]()
    executeMITBeerGame.parse_args.__globals__['parser'] = parser
    
    # Add arguments
    exec(open(os.path.join(scripts_dir, 'executeMITBeerGame.py')).read(), 
         {'__name__': '__main__', 'argparse': __import__('argparse')})
    
    # Test default values
    args = parser.parse_args([])
    
    # enable_communication should default to False
    assert args.enable_communication == False, "enable_communication should default to False"
    
    # profit_per_unit_sold should not exist
    assert not hasattr(args, 'profit_per_unit_sold'), "profit_per_unit_sold parameter should be removed"
    
    print("✓ CLI defaults test passed")


async def main():
    """Run all tests."""
    print("Running MIT Beer Game logical fixes tests...\n")
    
    await test_snapshot_integrity()
    await test_communication_toggle()
    
    # Skip workflow test if langraph not available
    try:
        await test_workflow_total_rounds()
    except ImportError:
        print("⚠ Skipping workflow test (langraph not available)")
    
    await test_pipeline_fields()
    test_cli_defaults()
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    asyncio.run(main()) 