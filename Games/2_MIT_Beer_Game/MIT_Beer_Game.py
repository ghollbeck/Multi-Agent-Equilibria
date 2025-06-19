"""
MIT Beer Game Simulation with LLM-Based Strategy
================================================

This script demonstrates how to run an MIT Beer Game simulation using large 
language models (LLMs) for adaptive ordering strategies. It transforms the 
architecture of an Iterated Prisoner's Dilemma simulation into a 4-role supply 
chain game:

Roles:
  1. Retailer
  2. Wholesaler
  3. Distributor
  4. Factory

Agents:
  - Each role is represented by an agent. 
  - Each agent tracks inventory, backlog, and costs. 
  - Each agent decides an "order quantity" to place upstream, 
    guided by an LLM-generated strategy. 
  - After a multi-round generation, the agent can update its strategy based 
    on performance logs.

Game Flow (per round):
  1. The Retailer observes external customer demand. 
  2. Each role receives incoming shipments sent 1 round ago from upstream. 
  3. Each role attempts to fill downstream orders from current inventory 
     (unfilled orders become backlog). 
  4. Each role incurs cost based on holding and backlog. 
  5. Each role decides how many units to order from its upstream agent. 
  6. Orders are placed and queued, to be delivered after a 1-round lead time.

LLM Prompt/Response Requirements:
  - Agents ask: 
        "Given my current inventory, backlog, demand (or order history), 
         and cost performance, how many units should I order next round?"
  - The LLM must respond with valid JSON:
        {
          "role_name": "{role_name}",
          "inventory": {inventory},
          "backlog": {backlog},
          "confidence": <float between 0 and 1>,
          "rationale": "<brief explanation of your reasoning>",
          "risk_assessment": "<describe any risks you anticipate>",
          "expected_demand_next_round": <integer>,
          "order_quantity": <integer>
        }

        IMPORTANT: Output ONLY the JSON object, with no markdown, no triple backticks, no code fences, and do NOT write the word 'json' anywhere. Your reply must be a single valid JSON object, nothing else. If you include anything else, your answer will be rejected. KEEP IT RATHER SHORT

Logging & Visualization:
  - Logs each round's orders, inventory, backlog, and costs. 
  - At the end of each generation, saves CSV/JSON logs and produces 
    matplotlib plots showing the supply chain dynamics.

Dependencies:
  - openai, asyncio (for asynchronous LLM calls)
  - pandas, matplotlib, numpy
  - tqdm (for progress bars)
  - pydantic (for data model)
  - dotenv (to load your OpenAI key from .env)
  - requests (if using an alternate endpoint or provider)
  - nest_asyncio (required to apply asyncio loops in scripts)

"""

import os
import json
import time
import datetime
import random
import asyncio
import requests
try:
    import nest_asyncio  # required to apply asyncio loops in scripts
    nest_asyncio.apply()
except ImportError:
    print("Warning: nest_asyncio not available, running without it")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import re  # needed for parse_json_with_default
import subprocess

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, ClassVar
from tqdm import tqdm
from prompts_mitb_game import BeerGamePrompts
from llm_calls_mitb_game import MODEL_NAME, lite_client
from models_mitb_game import BeerGameAgent, RoundData, SimulationData, BeerGameLogger
from analysis_mitb_game import plot_beer_game_results, calculate_nash_deviation
from memory_storage import MemoryManager
from langraph_workflow import BeerGameWorkflow

# --------------------------------------------------------------------
# 4. Agent Class: BeerGameAgent
#    Each Agent has: role, inventory, backlog, cost, strategy, etc.
# --------------------------------------------------------------------

# class BeerGameAgent(BaseModel):
#     role_name: str  # "Retailer", "Wholesaler", "Distributor", "Factory"
#     inventory: int = 100
#     backlog: int = 0
#     profit_accumulated: float = 0.0
#     # Store last round profit and last order placed for context in prompts
#     last_profit: Optional[float] = None
#     last_order_placed: Optional[int] = None
    
#     # Start with 10 units already in transit so agents receive shipments in the first rounds
#     # 1-round lead time: position 0 = what's arriving this round, position 1 = what arrives next round
#     shipments_in_transit: Dict[int,int] = Field(default_factory=lambda: {0:10, 1:10})
    
#     # Orders from the downstream agent (or external demand if Retailer)
#     # We keep a short log: the most recent N rounds (for context).
#     downstream_orders_history: List[int] = Field(default_factory=list)
    
#     # The LLM-based strategy
#     strategy: dict = Field(default_factory=dict)
    
#     # For demonstration, each agent can have its own prompts class, 
#     # or we can store a reference to a shared prompts. We'll assume a single set here.
#     prompts: ClassVar[BeerGamePrompts] = BeerGamePrompts
#     logger: BeerGameLogger = None
#     # For logging LLM I/O
#     last_decision_prompt: str = ""
#     last_decision_output: dict = Field(default_factory=dict)
#     last_update_prompt: str = ""
#     last_update_output: dict = Field(default_factory=dict)
#     last_init_prompt: str = ""
#     last_init_output: dict = Field(default_factory=dict)

#     class Config:
#         arbitrary_types_allowed = True

#     async def initialize_strategy(self, temperature=0.7, profit_per_unit_sold=5):
#         """
#         Generate initial strategy JSON from the LLM.
#         """
#         if self.logger:
#             self.logger.log(f"[Agent {self.role_name}] Initializing strategy...")
#         prompt = self.prompts.get_strategy_generation_prompt(self.role_name, self.inventory, self.backlog, profit_per_unit_sold)
#         self.last_init_prompt = prompt
#         system_prompt = "You are an expert supply chain manager. Return valid JSON only."
#         try:
#             response_str = await lite_client.chat_completion(model=MODEL_NAME,
#                                                       system_prompt=system_prompt,
#                                                       user_prompt=prompt,
#                                                       temperature=temperature)
#         except Exception as e:
#             print(f"❌ [Agent {self.role_name}] initialize_strategy: LLM call failed. Error: {e}")
#             response_str = ''
#         print(f"[Agent {self.role_name}] Strategy response: {response_str}")
#         if self.logger:
#             self.logger.log(f"[Agent {self.role_name}] Strategy response: {response_str}")
#         default_strategy = {
#             "order_quantity": 10,
#             "confidence": 1.0,
#             "rationale": "Default initial strategy",
#             "risk_assessment": "No risk",
#             "expected_demand_next_round": 10
#         }
#         try:
#             response = safe_parse_json(response_str)
#             print(f"✅ [Agent {self.role_name}] initialize_strategy: Valid JSON received.")
#         except Exception as e:
#             print(f"❌ [Agent {self.role_name}] initialize_strategy: Invalid JSON. Using default. Error: {e}")
#             response = default_strategy
#         self.strategy = response
#         self.last_init_output = response

#     async def update_strategy(self, performance_log: str, temperature=0.7, profit_per_unit_sold=5):
#         """
#         Update an existing strategy after a generation completes.
#         """
#         if self.logger:
#             self.logger.log(f"[Agent {self.role_name}] Updating strategy with performance log: {performance_log}")
#         prompt = self.prompts.get_strategy_update_prompt(self.role_name, performance_log, self.strategy, self.inventory, self.backlog, profit_per_unit_sold)
#         self.last_update_prompt = prompt
#         system_prompt = "You are an expert supply chain manager. Return valid JSON only."
#         try:
#             response_str = await lite_client.chat_completion(model=MODEL_NAME,
#                                                       system_prompt=system_prompt,
#                                                       user_prompt=prompt,
#                                                       temperature=temperature)
#         except Exception as e:
#             print(f"❌ [Agent {self.role_name}] update_strategy: LLM call failed. Error: {e}")
#             response_str = ''
#         print(f"[Agent {self.role_name}] Update response: {response_str}")
#         if self.logger:
#             self.logger.log(f"[Agent {self.role_name}] Update response: {response_str}")
#         default_update = {
#             "order_quantity": 10,
#             "confidence": 1.0,
#             "rationale": "Default update strategy",
#             "risk_assessment": "No risk",
#             "expected_demand_next_round": 10
#         }
#         try:
#             response = safe_parse_json(response_str)
#             print(f"✅ [Agent {self.role_name}] update_strategy: Valid JSON received.")
#         except Exception as e:
#             print(f"❌ [Agent {self.role_name}] update_strategy: Invalid JSON. Using default. Error: {e}")
#             response = default_update
#         self.strategy = response
#         self.last_update_output = response

#     async def decide_order_quantity(self, temperature=0.7, profit_per_unit_sold=5) -> dict:
#         """
#         Ask the LLM how many units to order from upstream in this round, 
#         given our current state. Must return the JSON dict with 'order_quantity', etc.
#         """
#         if self.logger:
#             self.logger.log(f"[Agent {self.role_name}] Deciding order quantity. Inventory: {self.inventory}, Backlog: {self.backlog}, Downstream: {self.downstream_orders_history[-3:]}, Shipments: {[self.shipments_in_transit[1]]}")
#         # Get last order placed and last profit for context
#         last_order_placed = self.last_order_placed
#         last_profit = self.last_profit
#         prompt = self.prompts.get_decision_prompt(
#             role_name=self.role_name,
#             inventory=self.inventory,
#             backlog=self.backlog,
#             recent_demand_or_orders=self.downstream_orders_history[-3:],
#             incoming_shipments=[self.shipments_in_transit[1]],
#             current_strategy=self.strategy,
#             profit_per_unit_sold=profit_per_unit_sold,
#             last_order_placed=last_order_placed,
#             last_profit=last_profit
#         )
#         self.last_decision_prompt = prompt
#         system_prompt = "You are an expert supply chain manager. Return valid JSON only."
#         try:
#             response_str = await lite_client.chat_completion(model=MODEL_NAME,
#                                                       system_prompt=system_prompt,
#                                                       user_prompt=prompt,
#                                                       temperature=temperature)
#         except Exception as e:
#             print(f"❌ [Agent {self.role_name}] decide_order_quantity: LLM call failed. Error: {e}")
#             response_str = ''
#         print(f"[Agent {self.role_name}] Decision response: {response_str}")
#         if self.logger:
#             self.logger.log(f"[Agent {self.role_name}] Decision response: {response_str}")
#         default_decision = {
#             "order_quantity": 10,
#             "confidence": 1.0,
#             "rationale": "Default decision",
#             "risk_assessment": "No risk",
#             "expected_demand_next_round": 10
#         }
#         try:
#             response = safe_parse_json(response_str)
#             print(f"✅ [Agent {self.role_name}] decide_order_quantity: Valid JSON received.")
#         except Exception as e:
#             print(f"❌ [Agent {self.role_name}] decide_order_quantity: Invalid JSON. Using default. Error: {e}")
#             response = default_decision
#         self.last_decision_output = response
#         # Store last profit for next round context
#         self.last_profit = response.get('profit', None)
#         return response

# --------------------------------------------------------------------
# 5. Simulation Logic
#    Each generation = multiple rounds. Then agents update strategy.
# --------------------------------------------------------------------

async def run_beer_game_generation(
    agents: List[BeerGameAgent],
    external_demand: List[int],
    num_rounds: int = 20,
    holding_cost_per_unit: float = 0.5,
    backlog_cost_per_unit: float = 1.5,
    profit_per_unit_sold: float = 5,
    temperature: float = 0.7,
    generation_index: int = 1,
    sim_data: SimulationData = None,
    human_log_file = None,
    logger: BeerGameLogger = None,
    csv_log_path: str = None,
    json_log_path: str = None,
    enable_communication: bool = False,
    communication_rounds: int = 2,
    enable_memory: bool = False,
    memory_retention_rounds: int = 5,
    enable_shared_memory: bool = False
):
    """
    Runs one generation of the Beer Game with the provided agents.
    We assume:
      agents[0] = Retailer
      agents[1] = Wholesaler
      agents[2] = Distributor
      agents[3] = Factory

    external_demand is a list of length num_rounds specifying 
    the customer demand in each round for the Retailer.

    Returns: None (logs are recorded in sim_data).
    """
    # Initialize memory system if enabled
    memory_manager = None
    if enable_memory:
        memory_manager = MemoryManager(
            retention_rounds=memory_retention_rounds,
            enable_shared_memory=enable_shared_memory
        )
        # Initialize agent memories
        for agent in agents:
            agent.memory = memory_manager.create_agent_memory(agent.role_name)
        
        if logger:
            logger.log(f"Memory system initialized with retention_rounds={memory_retention_rounds}, shared_memory={enable_shared_memory}")

    # Initialize LangGraph workflow if memory is enabled
    workflow = None
    if enable_memory and memory_manager:
        workflow = BeerGameWorkflow(
            agents=agents,
            simulation_data=sim_data,
            memory_manager=memory_manager,
            enable_memory=enable_memory,
            enable_shared_memory=enable_shared_memory,
            enable_communication=enable_communication,
            communication_rounds=communication_rounds
        )
        if logger:
            logger.log("LangGraph workflow initialized for memory-enabled simulation")

    # For convenience, create references:
    retailer = agents[0]
    wholesaler = agents[1] if len(agents) > 1 else None
    distributor = agents[2] if len(agents) > 2 else None
    factory = agents[3] if len(agents) > 3 else None

    # Each round, do the Beer Game steps:
    for r in range(num_rounds):
        round_index = r + 1
        if logger:
            logger.log(f"\n--------------------- Round {round_index} ---------------------")
            logger.log(f"External demand (Retailer): {external_demand[r]}")

        # 1. Retailer sees external demand
        retailer_demand = external_demand[r]

        # 2. Each role receives shipments that arrive this round
        shipments_received_list = []
        for agent in agents:
            # ---- SHIFT FIRST ----
            received = agent.shipments_in_transit.get(0, 0)        # what just arrived
            agent.shipments_in_transit[0] = agent.shipments_in_transit.get(1, 0)
            agent.shipments_in_transit[1] = 0  # Reset position 1 for new orders
            shipments_received_list.append(received)

        shipments_sent_downstream = [0 for _ in agents]

        # ✅ FIX – wire the 'received' units in for every tier, no fulfill()
        # Retailer fulfills both backlog and new demand
        # Use incoming shipment and demand to update inventory/backlog and compute amt_filled
        # 1. Retailer
        incoming_retailer = shipments_received_list[0]
        # Add incoming shipment to inventory
        retailer.inventory += incoming_retailer
        # Fulfill backlog first
        amt_to_backlog = min(retailer.inventory, retailer.backlog)
        retailer.inventory -= amt_to_backlog
        retailer.backlog -= amt_to_backlog
        # Fulfill new demand
        amt_to_demand = min(retailer.inventory, retailer_demand)
        retailer.inventory -= amt_to_demand
        leftover_demand = retailer_demand - amt_to_demand
        retailer.backlog += leftover_demand
        ship_down = amt_to_backlog + amt_to_demand  # everything that left the door
        shipments_sent_downstream[0] = ship_down
        retailer_order = ship_down  # order equals what you shipped
        retailer.downstream_orders_history.append(retailer_demand)

        # 2. Wholesaler
        if wholesaler:
            incoming_wholesaler = shipments_received_list[1]
            wholesaler.inventory += incoming_wholesaler
            # Clear backlog first
            amt_to_backlog = min(wholesaler.inventory, wholesaler.backlog)
            wholesaler.inventory -= amt_to_backlog
            wholesaler.backlog -= amt_to_backlog
            # Satisfy today's demand
            amt_to_demand = min(wholesaler.inventory, retailer_order)
            wholesaler.inventory -= amt_to_demand
            leftover_demand = retailer_order - amt_to_demand
            wholesaler.backlog += leftover_demand
            ship_down = amt_to_backlog + amt_to_demand  # everything that left the door
            shipments_sent_downstream[1] = ship_down
            wh_order = ship_down  # order equals what you shipped
            wholesaler.downstream_orders_history.append(retailer_order)
            retailer.shipments_in_transit[1] += ship_down

        # 3. Distributor
        if distributor and wholesaler:
            incoming_distributor = shipments_received_list[2]
            distributor.inventory += incoming_distributor
            # Clear backlog first
            amt_to_backlog = min(distributor.inventory, distributor.backlog)
            distributor.inventory -= amt_to_backlog
            distributor.backlog -= amt_to_backlog
            # Satisfy today's demand
            amt_to_demand = min(distributor.inventory, wh_order)
            distributor.inventory -= amt_to_demand
            leftover_demand = wh_order - amt_to_demand
            distributor.backlog += leftover_demand
            ship_down = amt_to_backlog + amt_to_demand  # everything that left the door
            shipments_sent_downstream[2] = ship_down
            dist_order = ship_down  # order equals what you shipped
            distributor.downstream_orders_history.append(wh_order)
            wholesaler.shipments_in_transit[1] += ship_down

        # 4. Factory
        if factory and distributor:
            incoming_factory = shipments_received_list[3]
            factory.inventory += incoming_factory
            # Clear backlog first
            amt_to_backlog = min(factory.inventory, factory.backlog)
            factory.inventory -= amt_to_backlog
            factory.backlog -= amt_to_backlog
            # Satisfy today's demand
            amt_to_demand = min(factory.inventory, dist_order)
            factory.inventory -= amt_to_demand
            leftover_demand = dist_order - amt_to_demand
            factory.backlog += leftover_demand
            ship_down = amt_to_backlog + amt_to_demand  # everything that left the door
            shipments_sent_downstream[3] = ship_down
            fact_order = ship_down  # order equals what you shipped
            factory.downstream_orders_history.append(dist_order)
            distributor.shipments_in_transit[1] += ship_down

        # 4. Each role pays holding + backlog cost
        for idx, agent in enumerate(agents):
            holding_cost = agent.inventory * holding_cost_per_unit
            backlog_cost = agent.backlog * backlog_cost_per_unit
            # Calculate profit for units sold (1.5 per unit)
            units_sold = shipments_sent_downstream[idx]
            profit = units_sold * profit_per_unit_sold
            # Calculate net profit (profit minus costs)
            round_profit = profit - (holding_cost + backlog_cost)
            agent.profit_accumulated += round_profit
            
            if logger:
                logger.log(f"Agent {agent.role_name}: Holding cost: {holding_cost}, Backlog cost: {backlog_cost}, Revenue: {profit}, Net profit: {round_profit}")

        communication_messages = []
        if enable_communication:
            if logger:
                logger.log(f"Starting communication phase with {communication_rounds} rounds")
            
            for comm_round in range(communication_rounds):
                if logger:
                    logger.log(f"Communication round {comm_round + 1}/{communication_rounds}")
                
                round_messages = []
                # Each agent sends a message sequentially
                for agent in agents:
                    message_response = await agent.generate_communication_message(
                        round_index=round_index,
                        other_agents=[a for a in agents if a != agent],
                        message_history=communication_messages,
                        temperature=temperature
                    )
                    
                    message_entry = {
                        "round": round_index,
                        "communication_round": comm_round + 1,
                        "sender": agent.role_name,
                        "message": message_response.get("message", ""),
                        "strategy_hint": message_response.get("strategy_hint", ""),
                        "collaboration_proposal": message_response.get("collaboration_proposal", ""),
                        "information_shared": message_response.get("information_shared", ""),
                        "confidence": message_response.get("confidence", 0.5)
                    }
                    
                    round_messages.append(message_entry)
                    communication_messages.append(message_entry)
                    
                    if logger:
                        logger.log(f"[{agent.role_name}] Message: {message_response.get('message', '')}")
                
                for agent in agents:
                    agent.message_history.extend(round_messages)

        # 5. Each role decides on new order quantity from upstream
        if workflow:
            initial_state = workflow.create_initial_state(
                round_index=round_index,
                generation=1,
                total_rounds=3,
                external_demand=retailer_demand,
                temperature=temperature,
                profit_per_unit_sold=profit_per_unit_sold
            )
            final_state = await workflow.run_round(initial_state)
            decisions = [final_state["agent_states"][agent.role_name]["decision_output"] 
                        for agent in agents if agent.role_name in final_state["agent_states"]]
        else:
            order_decision_tasks = []
            for agent in agents:
                recent_messages = communication_messages[-len(agents)*2:] if enable_communication else []
                if enable_communication and recent_messages:
                    order_decision_tasks.append(
                        agent.decide_order_quantity_with_communication(
                            temperature=temperature, 
                            profit_per_unit_sold=profit_per_unit_sold,
                            recent_communications=recent_messages
                        )
                    )
                else:
                    order_decision_tasks.append(agent.decide_order_quantity(temperature=temperature, profit_per_unit_sold=profit_per_unit_sold))
            decisions = await asyncio.gather(*order_decision_tasks)

        # 6. Place orders upstream => those orders become supplier's backlog
        orders_placed = []
        for idx, (agent, dec) in enumerate(zip(agents, decisions)):
            order_qty = dec.get("order_quantity", 10)
            orders_placed.append(order_qty)
            # store last order placed for context
            agent.last_order_placed = order_qty
            if idx < len(agents) - 1:
                # normal agent places order to its supplier
                upstream_agent = agents[idx + 1]
                upstream_agent.backlog += order_qty
            else:
                # Factory "produces" on a 1-round lead time by scheduling its own production
                agent.shipments_in_transit[1] += order_qty

        # Store round logs
        if sim_data:
            for idx, agent in enumerate(agents):
                entry = RoundData(
                    generation = generation_index,
                    round_index = round_index,
                    role_name = agent.role_name,
                    inventory = agent.inventory,
                    backlog = agent.backlog,
                    order_placed = orders_placed[idx],
                    shipment_received = shipments_received_list[idx],
                    shipment_sent_downstream = shipments_sent_downstream[idx],
                    profit = agent.profit_accumulated
                )
                sim_data.add_round_entry(entry)
                
                if enable_communication and communication_messages:
                    for msg in communication_messages:
                        if msg["round"] == round_index:
                            sim_data.add_communication_entry(msg)
                # Write to CSV after each round
                if csv_log_path:
                    write_header = not os.path.exists(csv_log_path) or os.path.getsize(csv_log_path) == 0
                    # Define the new fieldnames based on LLM decision output
                    llm_output_keys = [
                        'llm_reported_inventory', 'llm_reported_backlog', 'llm_recent_demand_or_orders',
                        'llm_incoming_shipments', 'llm_last_order_placed', 'llm_confidence',
                        'llm_rationale', 'llm_risk_assessment', 'llm_expected_demand_next_round'
                    ]
                    fieldnames = list(asdict(entry).keys()) + llm_output_keys
                    with open(csv_log_path, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        if write_header:
                            writer.writeheader()
                        # Prepare the row data including new LLM fields
                        row_data = asdict(entry)
                        llm_decision = getattr(agent, 'last_decision_output', {})
                        row_data.update({
                            # Add new LLM fields with prefixes
                            'llm_reported_inventory': llm_decision.get('inventory', None),
                            'llm_reported_backlog': llm_decision.get('backlog', None),
                            'llm_recent_demand_or_orders': json.dumps(llm_decision.get('recent_demand_or_orders', None)), # Stringify list
                            'llm_incoming_shipments': json.dumps(llm_decision.get('incoming_shipments', None)), # Stringify list
                            'llm_last_order_placed': llm_decision.get('last_order_placed', None),
                            'llm_confidence': llm_decision.get('confidence', None),
                            'llm_rationale': llm_decision.get('rationale', ''),
                            'llm_risk_assessment': llm_decision.get('risk_assessment', ''),
                            'llm_expected_demand_next_round': llm_decision.get('expected_demand_next_round', None)
                        })
                        writer.writerow(row_data)
                # Write to JSON after each round (reconstruct the logic to append new fields)
                if json_log_path:
                    # Read existing data
                    all_entries = []
                    if os.path.exists(json_log_path) and os.path.getsize(json_log_path) > 0:
                        try:
                            with open(json_log_path, 'r') as jf:
                                all_entries = json.load(jf)
                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode existing JSON file {json_log_path}. Starting fresh.")
                            all_entries = []

                    # Prepare the new entry with LLM data
                    agent_obj = agent # Use current agent in the loop
                    entry_dict = asdict(entry) # Use the current entry
                    llm_decision = getattr(agent_obj, 'last_decision_output', {})
                    entry_dict.update({
                        'last_decision_output': llm_decision, # Store as dict
                        'last_update_output': getattr(agent_obj, 'last_update_output', {}),
                        'last_init_output': getattr(agent_obj, 'last_init_output', {}),
                        # Add new LLM fields with prefixes
                        'llm_reported_inventory': llm_decision.get('inventory', None),
                        'llm_reported_backlog': llm_decision.get('backlog', None),
                        'llm_recent_demand_or_orders': llm_decision.get('recent_demand_or_orders', None), # Keep as list
                        'llm_incoming_shipments': llm_decision.get('incoming_shipments', None), # Keep as list
                        'llm_last_order_placed': llm_decision.get('last_order_placed', None),
                        'llm_confidence': llm_decision.get('confidence', None),
                        'llm_rationale': llm_decision.get('rationale', ''),
                        'llm_risk_assessment': llm_decision.get('risk_assessment', ''),
                        'llm_expected_demand_next_round': llm_decision.get('expected_demand_next_round', None)
                    })
                    all_entries.append(entry_dict)

                    # Write updated list back to JSON
                    with open(json_log_path, 'w') as jsonfile:
                        json.dump(all_entries, jsonfile, indent=2)

        # Write human-readable log for the round
        if human_log_file:
            human_log_file.write("\n--------------------- Round {} ---------------------\n".format(round_index))
            human_log_file.write("External demand (Retailer): {}\n".format(retailer_demand))
            human_log_file.write("Shipments received per agent: {}\n".format(shipments_received_list))
            
            # Log communication messages if they exist for this round
            if enable_communication and communication_messages:
                round_comm_messages = [msg for msg in communication_messages if msg["round"] == round_index]
                if round_comm_messages:
                    human_log_file.write("\n--- Communication Messages ---\n")
                    for msg in round_comm_messages:
                        human_log_file.write("Communication Round {}: {} says: {}\n".format(
                            msg["communication_round"], msg["sender"], msg["message"]
                        ))
                        human_log_file.write("  Strategy Hint: {}\n".format(msg["strategy_hint"]))
                        human_log_file.write("  Collaboration Proposal: {}\n".format(msg["collaboration_proposal"]))
                        human_log_file.write("  Information Shared: {}\n".format(msg["information_shared"]))
                        human_log_file.write("  Confidence: {}\n".format(msg["confidence"]))
                    human_log_file.write("\n")
            for idx, agent in enumerate(agents):
                human_log_file.write("Agent: {}: Inventory: {}, Backlog: {}, Order placed: {}, Units sold: {}, Profit: {:.2f}, Total Profit: {}\n".format(
                    agent.role_name, agent.inventory, agent.backlog, orders_placed[idx], shipments_sent_downstream[idx], 
                    shipments_sent_downstream[idx] * profit_per_unit_sold, agent.profit_accumulated
                ))
                # Log the LLM decision output for this agent
                decision = decisions[idx] if idx < len(decisions) else {}
                if decision:
                    # Updated format string to include all LLM output fields
                    human_log_file.write(
                        "    LLM Output: order_quantity={}, confidence={}, rationale={}, risk_assessment={}, expected_demand_next_round={}, "
                        "llm_inventory={}, llm_backlog={}, llm_recent_demand={}, llm_incoming={}, llm_last_order={}\n".format(
                            decision.get('order_quantity', 'N/A'),
                            decision.get('confidence', 'N/A'),
                            decision.get('rationale', 'N/A'),
                            decision.get('risk_assessment', 'N/A'),
                            decision.get('expected_demand_next_round', 'N/A'),
                            decision.get('inventory', 'N/A'), # llm reported inventory
                            decision.get('backlog', 'N/A'), # llm reported backlog
                            decision.get('recent_demand_or_orders', 'N/A'),
                            decision.get('incoming_shipments', 'N/A'),
                            decision.get('last_order_placed', 'N/A')
                    ))
                # Log the LLM prompt and output (Decision Output is now more detailed)
                human_log_file.write("    LLM Decision Prompt: {}\n".format(getattr(agent, 'last_decision_prompt', '')))
                human_log_file.write("    LLM Decision Output: {}\n".format(json.dumps(getattr(agent, 'last_decision_output', {}))))
                if enable_communication:
                    human_log_file.write("    LLM Communication Prompt: {}\n".format(getattr(agent, 'last_communication_prompt', '')))
                    human_log_file.write("    LLM Communication Output: {}\n".format(json.dumps(getattr(agent, 'last_communication_output', {}))))
                human_log_file.write("    LLM Update Prompt: {}\n".format(getattr(agent, 'last_update_prompt', '')))
                human_log_file.write("    LLM Update Output: {}\n".format(json.dumps(getattr(agent, 'last_update_output', {}))))
                human_log_file.write("    LLM Init Prompt: {}\n".format(getattr(agent, 'last_init_prompt', '')))
                human_log_file.write("    LLM Init Output: {}\n".format(json.dumps(getattr(agent, 'last_init_output', {}))))
            human_log_file.write("\n")
            human_log_file.flush()

        if logger:
            logger.log(f"Shipments received per agent: {shipments_received_list}")
            for idx, agent in enumerate(agents):
                if logger:
                    logger.log(f"Agent: {agent.role_name}: Inventory: {agent.inventory}, Backlog: {agent.backlog}, Order placed: {orders_placed[idx]}, Total Profit: {agent.profit_accumulated}")

# --------------------------------------------------------------------
# 6. Putting It All Together: run multiple generations
# --------------------------------------------------------------------

async def run_beer_game_simulation(
    num_generations: int = 1,
    num_rounds_per_generation: int = 20,
    temperature: float = 0.7,
    logger: BeerGameLogger = None,
    enable_communication: bool = False,
    communication_rounds: int = 2,
    enable_memory: bool = False,
    memory_retention_rounds: int = 5,
    enable_shared_memory: bool = False
):
    """
    Orchestrates multiple generations of the Beer Game. 
    Each generation:
      - Resets agent state or creates new agents
      - Runs num_rounds_per_generation
      - Agents update their strategy
    """
    # Prepare folder for logs
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_results_folder = os.path.join(os.path.dirname(__file__), "simulation_results")

    # --- New logic for run_N_date folder naming ---
    existing_folders = [f for f in os.listdir(base_results_folder) if os.path.isdir(os.path.join(base_results_folder, f))]
    run_numbers = []
    for folder in existing_folders:
        m = re.match(r"run_(\d+)_", folder)
        if m:
            run_numbers.append(int(m.group(1)))
    next_run_number = max(run_numbers) + 1 if run_numbers else 1
    results_folder = os.path.join(base_results_folder, f"run_{next_run_number}_{current_time}")
    os.makedirs(results_folder, exist_ok=True)
    # --- End new logic ---

    # Open human-readable log file
    human_log_path = os.path.join(results_folder, "human_readable_log.txt")
    human_log_file = open(human_log_path, "w")
    human_log_file.write("\n============================================================\n")
    human_log_file.write("            Starting Beer Game Simulation\n")
    human_log_file.write("============================================================\n\n")

    if logger is None:
        logger = BeerGameLogger()

    # Initialize the roles (4 default roles)
    roles = ["Retailer", "Wholesaler", "Distributor", "Factory"]
    agents = [BeerGameAgent(role_name=role, logger=logger) for role in roles]

    # Each agent obtains an initial strategy from the LLM
    await asyncio.gather(*(agent.initialize_strategy(temperature=temperature, profit_per_unit_sold=5) for agent in agents))

    # Example external demand across rounds:
    # In real usage, you might load or generate random demands.
    # Generate external demand using a normal distribution centered at 10, clipped to [0, 20] and rounded to int
    external_demand_pattern = [
        int(min(20, max(0, round(random.gauss(10, 3)))))
        for _ in range(num_rounds_per_generation)
    ]

    sim_data = SimulationData(hyperparameters={
        "num_generations": num_generations,
        "num_rounds_per_generation": num_rounds_per_generation,
        "holding_cost_per_unit": 0.5,
        "backlog_cost_per_unit": 1.5,
        "profit_per_unit_sold": 5,
        "roles": roles,
        "timestamp": current_time
    })

    csv_log_path = os.path.join(results_folder, "beer_game_detailed_log.csv")
    json_log_path = os.path.join(results_folder, "beer_game_detailed_log.json")
    # Create empty files at the start
    with open(csv_log_path, 'w', newline='') as csvfile:
        pass
    with open(json_log_path, 'w') as jsonfile:
        json.dump([], jsonfile)

    for gen_idx in range(num_generations):
        logger.log(f"\n--- Starting Generation {gen_idx+1} ---")

        # Reset each agent's state if you want them to start "fresh" each generation 
        # (except for the strategy).
        for agent in agents:
            agent.inventory = 100
            agent.backlog = 0
            agent.profit_accumulated = 0.0
            # Also reset shipments in transit:
            agent.shipments_in_transit = {0:10,1:10}
            agent.downstream_orders_history = []

        logger.log(f"\n============================================================\nGeneration {gen_idx+1}\n============================================================")

        await run_beer_game_generation(
            agents=agents,
            external_demand=external_demand_pattern,
            num_rounds=num_rounds_per_generation,
            holding_cost_per_unit=0.5,
            backlog_cost_per_unit=1.5,
            profit_per_unit_sold=5,
            temperature=temperature,
            generation_index=gen_idx+1,
            sim_data=sim_data,
            human_log_file=human_log_file,
            logger=logger,
            csv_log_path=csv_log_path,
            json_log_path=json_log_path,
            enable_communication=enable_communication,
            communication_rounds=communication_rounds,
            enable_memory=enable_memory,
            memory_retention_rounds=memory_retention_rounds,
            enable_shared_memory=enable_shared_memory
        )

        # After the generation, collect performance logs for each agent,
        # then ask them to update their strategy if desired.
        for agent in agents:
            # Summarize performance
            performance_log = (
                f"Final Inventory: {agent.inventory}, "
                f"Final Backlog: {agent.backlog}, "
                f"Total Profit: {agent.profit_accumulated:.2f}"
            )
            await agent.update_strategy(performance_log, temperature=temperature, profit_per_unit_sold=5)

    # Simple visualizations (inventory/backlog over time, cost, etc.)
    # Ensure df_rounds is defined for saving and plotting
    df_rounds = pd.DataFrame([asdict(r) for r in sim_data.rounds_log])
    
    complete_sim_data = sim_data.to_dict()
    with open(json_log_path, 'w') as jsonfile:
        json.dump(complete_sim_data, jsonfile, indent=2)
    
    # Generate visualizations
    plot_beer_game_results(df_rounds, results_folder)

    # --- Convert CSV to Markdown table automatically ---
    md_log_path = os.path.join(results_folder, "beer_game_detailed_log.md")
    try:
        with open(md_log_path, "w") as mdfile:
            subprocess.run(["csvtomd", csv_log_path], check=True, stdout=mdfile)
    except FileNotFoundError:
        print("[Warning] csvtomd is not installed. Run 'pip install csvtomd' to enable CSV to Markdown conversion.")
    except Exception as e:
        print(f"[Warning] Could not convert CSV to Markdown: {e}")
    # --- End CSV to Markdown ---

    # Calculate Nash equilibrium deviations (assumed equilibrium order quantity = 10)
    deviations = calculate_nash_deviation(df_rounds, equilibrium_order=10)
    human_log_file.write("\n----- Nash Equilibrium Analysis -----\n")
    for role, dev in deviations.items():
        human_log_file.write("Role: {} - Average Absolute Deviation: {:.2f}\n".format(role, dev))

    # Log LLM session summary
    from llm_calls_mitb_game import lite_client
    session_summary = lite_client.get_session_summary()
    
    print(f"\n🎯 [LLM SESSION SUMMARY]")
    print(f"   📞 Total LLM Calls: {session_summary['total_calls']}")
    print(f"   💰 Total Cost: ${session_summary['total_cost_usd']}")
    print(f"   📝 Total Tokens: {session_summary['total_tokens']} ({session_summary['total_input_tokens']} in + {session_summary['total_output_tokens']} out)")
    print(f"   ⏱️  Total Inference Time: {session_summary['total_inference_time_seconds']}s")
    print(f"   📊 Average per Call: {session_summary['average_inference_time_seconds']:.3f}s, ${session_summary['average_cost_per_call_usd']:.6f}")
    
    human_log_file.write("\n----- LLM Session Summary -----\n")
    human_log_file.write("Total LLM Calls: {}\n".format(session_summary['total_calls']))
    human_log_file.write("Total Cost: ${}\n".format(session_summary['total_cost_usd']))
    human_log_file.write("Total Tokens: {} ({} input + {} output)\n".format(
        session_summary['total_tokens'], 
        session_summary['total_input_tokens'], 
        session_summary['total_output_tokens']
    ))
    human_log_file.write("Total Inference Time: {}s\n".format(session_summary['total_inference_time_seconds']))
    human_log_file.write("Average per Call: {:.3f}s, ${:.6f}\n".format(
        session_summary['average_inference_time_seconds'], 
        session_summary['average_cost_per_call_usd']
    ))
    
    # Save session summary to JSON
    session_summary_path = os.path.join(results_folder, "llm_session_summary.json")
    with open(session_summary_path, 'w') as f:
        json.dump(session_summary, f, indent=2)
    
    # Move LLM metrics file to results folder if it exists
    metrics_file = "llm_inference_metrics.json"
    if os.path.exists(metrics_file):
        metrics_dest = os.path.join(results_folder, "llm_inference_metrics.json")
        os.rename(metrics_file, metrics_dest)
        print(f"📋 LLM metrics saved to: {metrics_dest}")

    human_log_file.close()
    logger.log("\nSimulation complete. Results saved to: {}".format(results_folder))
    logger.close()
    return sim_data
