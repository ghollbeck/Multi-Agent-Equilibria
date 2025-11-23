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

CRITICAL SHIPMENT CONSTRAINT:
  - Agents can only ship up to (downstream_order + current_backlog) units
  - This prevents oversupplying even when excess inventory is available
  - Maintains realistic supply chain constraints and forces strategic planning

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
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass  # print("Warning: nest_asyncio not available, running without it")  # Commented out
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
    num_rounds: int = 2,
    holding_cost_per_unit: float = None,
    backlog_cost_per_unit: float = None,
    sale_price_per_unit: float = None,
    purchase_cost_per_unit: float = None,
    production_cost_per_unit: float = None,
    initial_inventory: int = 100,
    initial_backlog: int = 0,
    initial_balance: float = 1000.0,
    temperature: float = 0,
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
    enable_shared_memory: bool = False,
    enable_orchestrator: bool = False,
    orchestrator_history: int = 3,
    orchestrator_override: bool = False,
    longtermplanning_boolean: bool = False,
    safety_stock_target: float = 10.0,
    backlog_clearance_rate: float = 0.5,
    demand_smoothing_factor: float = 0.3
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
    if enable_memory or enable_shared_memory:
        workflow = BeerGameWorkflow(
            agents=agents,
            simulation_data=sim_data,
            memory_manager=memory_manager,
            enable_memory=enable_memory,
            enable_shared_memory=enable_shared_memory,
            enable_communication=enable_communication,
            communication_rounds=communication_rounds
        )

    # Initialize orchestrator if enabled
    orchestrator = None
    if enable_orchestrator:
        from orchestrator_mitb_game import BeerGameOrchestrator
        orchestrator = BeerGameOrchestrator(history_window=orchestrator_history, logger=logger)

    # For convenience, create references:
    retailer = agents[0]
    wholesaler = agents[1] if len(agents) > 1 else None
    distributor = agents[2] if len(agents) > 2 else None
    factory = agents[3] if len(agents) > 3 else None

    # Log initial state (before any rounds are processed) as Round 0
    if sim_data:
        logger.log("📊 Logging initial state (Round 0 - before any processing)")
        for idx, agent in enumerate(agents):
            initial_entry = RoundData(
                generation = generation_index,
                round_index = 0,  # Round 0 represents initial state
                role_name = agent.role_name,
                inventory = agent.inventory,  # This is the true initial inventory
                backlog = agent.backlog,      # This is the true initial backlog
                order_placed = 0,             # No orders placed yet
                order_received = 0,           # No orders received yet
                shipment_received = 0,        # No shipments received yet
                shipment_sent_downstream = 0, # No shipments sent yet
                starting_balance = agent.balance,
                revenue = 0,                  # No revenue yet
                purchase_cost = 0,            # No purchase costs yet
                holding_cost = 0,             # No holding costs yet (will be calculated at end of round 1)
                backlog_cost = 0,             # No backlog costs yet
                ending_balance = agent.balance,
                orchestrator_order = 0,       # No orchestrator recommendations yet
                orchestrator_rationale = "Initial state"
            )
            sim_data.add_round_entry(initial_entry)

    # Each round, do the Beer Game steps (now starting from round 1):
    for round_index in range(num_rounds):
        logger.log(f"\n--- Starting Round {round_index+1} ---")
        print(f"   🎲 Round {round_index+1}/{num_rounds} - External demand: {external_demand[round_index]}")
        retailer_demand = external_demand[round_index]
        
        # Log round start to human-readable file
        if human_log_file:
            human_log_file.write("\n" + "🎲" + "="*76 + "🎲\n")
            human_log_file.write(f"🎮 ROUND {round_index+1}/{num_rounds} - External Demand: {retailer_demand} 🎮\n")
            human_log_file.write("🎲" + "="*76 + "🎲\n\n")
            human_log_file.flush()

        # 1. Retailer sees external demand
        retailer_demand = external_demand[round_index]

        # 2. Each role receives shipments that arrive this round
        # -----------------------------
        # 2a. Shift shipment pipeline  (physical goods)
        # -----------------------------
        shipments_received_list = []
        for agent in agents:
            # Read what's arriving NOW (before shifting)
            received = agent.shipments_in_transit.get(0, 0)
            shipments_received_list.append(received)
            # Then shift pipeline for next round
            agent.shipments_in_transit[0] = agent.shipments_in_transit.get(1, 0)
            agent.shipments_in_transit[1] = 0

        # -----------------------------
        # 2b. Shift order-information pipeline (purchase orders)
        #     Orders placed last round arrive as information *now*
        # -----------------------------
        orders_received_list = []
        for agent in agents:
            # Read what's arriving NOW (before shifting)
            incoming_orders = agent.orders_in_transit.get(0, 0)
            orders_received_list.append(incoming_orders)
            # Then shift pipeline for next round
            agent.orders_in_transit[0] = agent.orders_in_transit.get(1, 0)
            agent.orders_in_transit[1] = 0

        # Track downstream orders for logging (Retailer gets external demand)
        orders_received_from_downstream = [0 for _ in agents]
        orders_received_from_downstream[0] = retailer_demand  # customer demand
        for idx in range(1, len(agents)):
            orders_received_from_downstream[idx] = orders_received_list[idx]

        # Add newly arrived orders to each supplier's backlog *before* shipping
        for idx, agent in enumerate(agents):
            if idx == 0:
                # Retailer backlog handled with external demand below
                continue
            agent.backlog += orders_received_list[idx]

        shipments_sent_downstream = [0 for _ in agents]
        # orders_received_from_downstream = [0 for _ in agents]  # Track new orders received this round

        # ✅ FIX – wire the 'received' units in for every tier, no fulfill()
        # Retailer fulfills both backlog and new demand
        # Use incoming shipment and demand to update inventory/backlog and compute amt_filled
        # IMPORTANT: Agents can only ship up to (new_orders + current_backlog)
        
        # 1. Retailer
        incoming_retailer = shipments_received_list[0]
        # Add incoming shipment to inventory
        retailer.inventory += incoming_retailer
        
        # Calculate maximum allowed shipment: external demand + current backlog
        max_allowed_shipment = retailer_demand + retailer.backlog
        available_to_ship = min(retailer.inventory, max_allowed_shipment)
        
        if logger and retailer.inventory > max_allowed_shipment:
            logger.log(f"📦 [Retailer] Shipment constraint applied: inventory={retailer.inventory}, max_allowed={max_allowed_shipment}")
        
        # Fulfill backlog first
        amt_to_backlog = min(available_to_ship, retailer.backlog)
        retailer.inventory -= amt_to_backlog
        retailer.backlog -= amt_to_backlog
        available_to_ship -= amt_to_backlog
        
        # Fulfill new demand with remaining allowed shipment
        amt_to_demand = min(available_to_ship, retailer_demand)
        retailer.inventory -= amt_to_demand
        leftover_demand = retailer_demand - amt_to_demand
        retailer.backlog += leftover_demand
        
        ship_down = amt_to_backlog + amt_to_demand  # everything that left the door
        shipments_sent_downstream[0] = ship_down
        # Note: Retailer's actual order quantity is decided later in the decision phase
        retailer.downstream_orders_history.append(retailer_demand)
        orders_received_from_downstream[0] = retailer_demand  # Retailer receives external customer demand

        # 2. Wholesaler
        if wholesaler:
            incoming_wholesaler = shipments_received_list[1]
            wholesaler.inventory += incoming_wholesaler
            
            # Ship from backlog (which now includes arrived orders)
            ship_down = min(wholesaler.inventory, wholesaler.backlog)
            wholesaler.inventory -= ship_down
            wholesaler.backlog -= ship_down

            shipments_sent_downstream[1] = ship_down
            wholesaler.downstream_orders_history.append(orders_received_list[1])
            orders_received_from_downstream[1] = orders_received_list[1]
            retailer.shipments_in_transit[1] += ship_down

        # 3. Distributor
        if distributor and wholesaler:
            incoming_distributor = shipments_received_list[2]
            distributor.inventory += incoming_distributor
            
            # Ship from backlog (which now includes arrived orders)
            ship_down = min(distributor.inventory, distributor.backlog)
            distributor.inventory -= ship_down
            distributor.backlog -= ship_down

            shipments_sent_downstream[2] = ship_down
            distributor.downstream_orders_history.append(orders_received_list[2])
            orders_received_from_downstream[2] = orders_received_list[2]
            wholesaler.shipments_in_transit[1] += ship_down

        # 4. Factory
        if factory and distributor:
            incoming_factory = shipments_received_list[3]
            factory.inventory += incoming_factory
            
            # backlog already incremented earlier with orders_received_list[3]
            
            # Ship from backlog (which now includes the new order)
            ship_down = min(factory.inventory, factory.backlog)
            factory.inventory -= ship_down
            factory.backlog -= ship_down

            shipments_sent_downstream[3] = ship_down
            factory.downstream_orders_history.append(orders_received_list[3])
            orders_received_from_downstream[3] = orders_received_list[3]
            distributor.shipments_in_transit[1] += ship_down

            # --- Infinite reservoir production *with* 1-round delay ---
            # Whatever quantity factory *orders* for next round is guaranteed to arrive (no upstream cap).
            # We ensure that the factory schedules enough production to cover its backlog so it never grows unbounded.
            scheduled_production = max(factory.backlog, 0)  # produce enough to clear backlog only
            factory.shipments_in_transit[1] += scheduled_production
            if logger:
                logger.log(f"🔥 [Factory] Scheduled {scheduled_production} units for production (delivered next round)")

        # 4. Each role updates cash balance: revenues minus holding/backlog costs (purchase cost deducted later)
        starting_balances = [agent.balance for agent in agents]
        revenues = []
        holding_costs = []
        backlog_costs = []
        for idx, agent in enumerate(agents):
            holding_cost = agent.inventory * holding_cost_per_unit
            backlog_cost = agent.backlog * backlog_cost_per_unit
            units_sold = shipments_sent_downstream[idx]
            revenue = units_sold * sale_price_per_unit
            # Calculate round profit before updating balance
            round_profit = revenue - holding_cost - backlog_cost
            # Apply to balance
            agent.balance += revenue
            agent.balance -= (holding_cost + backlog_cost)
            # Update profit tracking
            agent.last_profit = round_profit
            agent.update_profit_history(round_profit, agent.balance)

            revenues.append(revenue)
            holding_costs.append(holding_cost)
            backlog_costs.append(backlog_cost)

            if logger:
                logger.log(f"Agent {agent.role_name}: Revenue: {revenue}, Holding cost: {holding_cost}, Backlog cost: {backlog_cost}, Round profit: {round_profit}, New balance: {agent.balance}")

        communication_messages = []
        if enable_communication:
            if logger:
                logger.log(f"Starting communication phase with {communication_rounds} rounds")
            print(f"   💬 Communication phase ({communication_rounds} rounds)")
            
            # Log communication phase start
            if human_log_file:
                human_log_file.write("💬 COMMUNICATION PHASE\n")
                human_log_file.write("─" * 60 + "\n\n")
                human_log_file.flush()
            
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
                        temperature=temperature,
                        longtermplanning_boolean=longtermplanning_boolean,
                        selling_price_per_unit=sale_price_per_unit,
                        unit_cost_per_unit=purchase_cost_per_unit if agent.role_name != "Factory" else production_cost_per_unit,
                        holding_cost_per_unit=holding_cost_per_unit,
                        backlog_cost_per_unit=backlog_cost_per_unit
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
                        logger.log(f"💬 [{agent.role_name}] message_sent (round {round_index+1}, comm {comm_round+1})")
                
                for agent in agents:
                    agent.message_history.extend(round_messages)

        # --------------------------------------------------
        # 🧑‍💼 Orchestrator Phase – provide chain-level advice
        # --------------------------------------------------
        orchestrator_recs = {}
        if enable_orchestrator and orchestrator:
            try:
                orchestrator_recs = await orchestrator.get_recommendations(
                    agents=agents,
                    external_demand=retailer_demand,
                    round_index=round_index,
                    history=sim_data.aggregated_rounds if sim_data else []
                )
                if logger:
                    logger.log(f"📊 [Orchestrator] Recommendations: {orchestrator_recs}")
                # Also log to human-readable log file
                if human_log_file:
                    human_log_file.write("\n📊 ORCHESTRATOR RECOMMENDATIONS (Round {0})\n".format(round_index))
                    human_log_file.write(json.dumps(orchestrator_recs, indent=2) + "\n")
            except Exception as e:
                if logger:
                    logger.log(f"[Orchestrator] Failed to generate recommendations: {e}")
                orchestrator_recs = {}

            # Share orchestrator advice as a broadcast message so agents can factor it in
            for agent in agents:
                if agent.role_name in orchestrator_recs:
                    rec = orchestrator_recs[agent.role_name]
                    orch_msg = {
                        "round": round_index,
                        "communication_round": 0,
                        "sender": "Orchestrator",
                        "message": f"Recommended order qty: {rec['order_quantity']}. {rec.get('rationale','')}",
                        "strategy_hint": "Orchestrator advice",
                        "collaboration_proposal": "",
                        "information_shared": "",
                        "confidence": 1.0
                    }
                    communication_messages.append(orch_msg)
                    agent.message_history.append(orch_msg)

        # 5. Each role decides on new order quantity from upstream
        print(f"   🤔 Agents making ordering decisions...")
        
        # Log decision phase start
        if human_log_file:
            human_log_file.write("🎯 DECISION PHASE\n")
            human_log_file.write("─" * 60 + "\n\n")
            human_log_file.flush()
        if workflow:
            initial_state = workflow.create_initial_state(
                round_index=round_index,
                generation=1,
                total_rounds=num_rounds,
                external_demand=retailer_demand,
                temperature=temperature,
                profit_per_unit_sold=sale_price_per_unit
            )
            final_state = await workflow.run_round(initial_state)
            decisions = [final_state["agent_states"][agent.role_name]["decision_output"] 
                        for agent in agents if agent.role_name in final_state["agent_states"]]
        else:
            use_comm = enable_communication or enable_orchestrator
            recent_messages = communication_messages[-len(agents)*2:] if use_comm else []
            
            # Always use llm_decision for consistent logging, regardless of communication
            order_decision_tasks = []
            for agent in agents:
                order_decision_tasks.append(
                    agent.llm_decision(
                        "decision",
                        comm_history=recent_messages if recent_messages else None,
                        orchestrator_advice=orchestrator_recs.get(agent.role_name, {}).get("rationale") if enable_orchestrator and orchestrator_recs else None,
                        history_limit=20,
                        temperature=temperature,
                        selling_price_per_unit=sale_price_per_unit,
                        unit_cost_per_unit=purchase_cost_per_unit if agent.role_name != "Factory" else production_cost_per_unit,
                        holding_cost_per_unit=holding_cost_per_unit,
                        backlog_cost_per_unit=backlog_cost_per_unit,
                        total_chain_inventory=sum(a.inventory for a in agents),
                        total_chain_backlog=sum(a.backlog for a in agents),
                        longtermplanning_boolean=longtermplanning_boolean,
                        safety_stock_target=safety_stock_target,
                        backlog_clearance_rate=backlog_clearance_rate,
                        demand_smoothing_factor=demand_smoothing_factor,
                        round_index=round_index
                    )
                )
            decisions = await asyncio.gather(*order_decision_tasks)

        # 6. LLM decisions for strategic planning (logged but not used for immediate orders)
        orders_placed = []
        for idx, (agent, dec) in enumerate(zip(agents, decisions)):
            if orchestrator_override and agent.role_name in orchestrator_recs:
                order_qty = orchestrator_recs[agent.role_name]["order_quantity"]
            else:
                order_qty = dec.get("order_quantity", 10)
            orders_placed.append(order_qty)
            # store last order placed for context
            agent.last_order_placed = order_qty
            # Queue this order so it arrives at the upstream supplier next round
            if idx < len(agents) - 1:
                upstream_agent = agents[idx + 1]
                upstream_agent.orders_in_transit[1] += order_qty

        # 6b. Deduct purchase / production cost immediately from balance
        purchase_costs = []
        for idx, (agent, qty) in enumerate(zip(agents, orders_placed)):
            if agent.role_name == "Factory":
                cost_per_unit = production_cost_per_unit
            else:
                cost_per_unit = purchase_cost_per_unit
            purchase_cost = qty * cost_per_unit
            agent.balance -= purchase_cost
            purchase_costs.append(purchase_cost)
            if logger:
                logger.log(f"💸 [Cost] {agent.role_name} prepaid {purchase_cost} for {qty} units (cost/unit={cost_per_unit}). New balance: {agent.balance}")

        # Bankruptcy check – stop simulation if any agent is insolvent
        bankrupt = any(a.balance <= 0 for a in agents)
        if bankrupt:
            logger.log("❌ Bankruptcy! Simulation terminated early due to zero or negative balance.")
            break

        # Store round logs
        if sim_data:
            # Collect all entries for this round
            round_entries = []
            for idx, agent in enumerate(agents):
                # Get the LLM decision for this specific round
                llm_decision = decisions[idx] if idx < len(decisions) else {}
                entry = RoundData(
                    generation = generation_index,
                    round_index = round_index + 1,  # Adjust since Round 0 is now initial state
                    role_name = agent.role_name,
                    inventory = agent.inventory,
                    backlog = agent.backlog,
                    order_placed = orders_placed[idx],
                    order_received = orders_received_from_downstream[idx],
                    shipment_received = shipments_received_list[idx],
                    shipment_sent_downstream = shipments_sent_downstream[idx],
                    starting_balance = starting_balances[idx],
                    revenue = revenues[idx],
                    purchase_cost = purchase_costs[idx],
                    holding_cost = holding_costs[idx],
                    backlog_cost = backlog_costs[idx],
                    ending_balance = agent.balance,
                    orchestrator_order = orchestrator_recs.get(agent.role_name, {}).get("order_quantity", 0),
                    orchestrator_rationale = orchestrator_recs.get(agent.role_name, {}).get("rationale", ""),
                    # Snapshot LLM decision for this round
                    llm_reported_inventory = llm_decision.get('inventory', None),
                    llm_reported_backlog = llm_decision.get('backlog', None),
                    llm_recent_demand_or_orders = llm_decision.get('recent_demand_or_orders', None),
                    llm_incoming_shipments = llm_decision.get('incoming_shipments', None),
                    llm_last_order_placed = llm_decision.get('last_order_placed', None),
                    llm_confidence = llm_decision.get('confidence', None),
                    llm_rationale = llm_decision.get('rationale', ''),
                    llm_risk_assessment = llm_decision.get('risk_assessment', ''),
                    llm_expected_demand_next_round = llm_decision.get('expected_demand_next_round', None),
                    # Pipeline state after shifts and before decisions
                    orders_in_transit_0 = agent.orders_in_transit.get(0, 0),
                    orders_in_transit_1 = agent.orders_in_transit.get(1, 0),
                    production_queue_0 = agent.production_queue.get(0, 0) if agent.role_name == "Factory" else 0,
                    production_queue_1 = agent.production_queue.get(1, 0) if agent.role_name == "Factory" else 0
                )
                sim_data.add_round_entry(entry)
                
                # Collect entry for CSV/JSON writing
                round_entries.append((entry, agent))
                
                if enable_communication and communication_messages:
                    for msg in communication_messages:
                        if msg["round"] == round_index:
                            sim_data.add_communication_entry(msg)
            
            # Write all entries to CSV after collecting them (outside the agent loop)
            if csv_log_path and round_entries:
                base_fields = list(asdict(round_entries[0][0]).keys())
                if 'external_demand' not in base_fields:
                    base_fields.append('external_demand')
                if 'profit_accumulated' not in base_fields:
                    base_fields.append('profit_accumulated')
                fieldnames = base_fields
                
                # Write complete CSV file (overwrite each round)
                # Ensure directory exists before writing
                os.makedirs(os.path.dirname(csv_log_path), exist_ok=True)
                with open(csv_log_path, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
                    writer.writeheader()
                    
                    # Write all rounds up to current round
                    for r in range(round_index + 1):
                        for idx, agent in enumerate(agents):
                            # Get the entry for this round and agent
                            entry_index = r * len(agents) + idx
                            if entry_index < len(sim_data.rounds_log):
                                entry = sim_data.rounds_log[entry_index]
                                row_data = asdict(entry)
                                # Add external demand for this round (same for all agents)
                                row_data['external_demand'] = external_demand[r] if r < len(external_demand) else 0
                                # profit_accumulated should be the ending_balance from that round
                                row_data['profit_accumulated'] = entry.ending_balance
                                # Stringify list fields for CSV
                                if entry.llm_recent_demand_or_orders is not None:
                                    row_data['llm_recent_demand_or_orders'] = json.dumps(entry.llm_recent_demand_or_orders)
                                if entry.llm_incoming_shipments is not None:
                                    row_data['llm_incoming_shipments'] = json.dumps(entry.llm_incoming_shipments)
                                writer.writerow(row_data)
            
            # Write complete JSON file (overwrite each round)
            if json_log_path:
                # ---- Aggregated round structure for nested logging ----
                # Collect communication messages for this specific round
                round_communication_messages = []
                if enable_communication and communication_messages:
                    round_communication_messages = [msg for msg in communication_messages if msg["round"] == round_index]
                
                # Create aggregated round structure with all agent data
                agent_data_dict = {}
                for entry, agent in round_entries:
                    agent_dict = asdict(entry)
                    llm_decision = getattr(agent, 'last_decision_output', {})
                    
                    # Add LLM output fields to agent data
                    agent_dict.update({
                        'last_decision_output': llm_decision,
                        'last_update_output': getattr(agent, 'last_update_output', {}),
                        'last_init_output': getattr(agent, 'last_init_output', {}),
                        'llm_reported_inventory': llm_decision.get('inventory', None),
                        'llm_reported_backlog': llm_decision.get('backlog', None),
                        'llm_recent_demand_or_orders': llm_decision.get('recent_demand_or_orders', None),
                        'llm_incoming_shipments': llm_decision.get('incoming_shipments', None),
                        'llm_last_order_placed': llm_decision.get('last_order_placed', None),
                        'llm_confidence': llm_decision.get('confidence', None),
                        'llm_rationale': llm_decision.get('rationale', ''),
                        'llm_risk_assessment': llm_decision.get('risk_assessment', ''),
                        'llm_expected_demand_next_round': llm_decision.get('expected_demand_next_round', None),
                        'profit_accumulated': agent.balance,
                        'orchestrator_order': orchestrator_recs.get(agent.role_name, {}).get('order_quantity', 0),
                        'orchestrator_rationale': orchestrator_recs.get(agent.role_name, {}).get('rationale', '')
                    })
                    
                    agent_data_dict[agent.role_name] = agent_dict
                
                aggregated_round = {
                    "round_index": round_index,
                    "generation": generation_index,
                    "external_demand": retailer_demand,
                    "communication": round_communication_messages,
                    "agents": agent_data_dict,
                    "orchestrator_recommendations": orchestrator_recs
                }
                
                if sim_data:
                    sim_data.add_aggregated_round(aggregated_round)
                
                # Write complete JSON structure (overwrite each round)
                complete_sim_data = sim_data.to_dict()
                nested_json_structure = {
                    "hyperparameters": complete_sim_data['hyperparameters'],
                    "rounds_log": complete_sim_data['aggregated_rounds'],
                    "flat_rounds_log": complete_sim_data['rounds_log'],
                    "communication_log": complete_sim_data['communication_log']
                }
                
                # Ensure directory exists before writing
                os.makedirs(os.path.dirname(json_log_path), exist_ok=True)
                with open(json_log_path, 'w') as jsonfile:
                    json.dump(nested_json_structure, jsonfile, indent=2)
            
            # Generate plots after each round (overwrite previous plots)
            try:
                df_rounds = pd.DataFrame([asdict(r) for r in sim_data.rounds_log])
                # Create run settings dictionary
                run_settings = {
                    'num_rounds': num_rounds,
                    'temperature': temperature,
                    'sale_price_per_unit': sale_price_per_unit,
                    'purchase_cost_per_unit': purchase_cost_per_unit,
                    'production_cost_per_unit': production_cost_per_unit,
                    'holding_cost_per_unit': holding_cost_per_unit,
                    'backlog_cost_per_unit': backlog_cost_per_unit,
                    'enable_communication': enable_communication,
                    'communication_rounds': communication_rounds,
                    'enable_memory': enable_memory,
                    'memory_retention_rounds': memory_retention_rounds,
                    'enable_orchestrator': enable_orchestrator,
                    'orchestrator_history': orchestrator_history,
                    'orchestrator_override': orchestrator_override,
                    'initial_inventory': initial_inventory,
                    'initial_backlog': initial_backlog,
                    'initial_balance': initial_balance
                }
                plot_beer_game_results(df_rounds, os.path.dirname(csv_log_path), external_demand[:round_index+1], run_settings)
                if logger:
                    logger.log(f"📊 Plots updated after round {round_index+1}")
            except Exception as e:
                if logger:
                    logger.log(f"❌ Error generating plots after round {round_index+1}: {e}")

        # Write simple round summary to human-readable log (detailed LLM calls are logged immediately)
        if human_log_file:
            human_log_file.write("\n" + "="*80 + "\n")
            human_log_file.write(f"🎲 ROUND {round_index+1}/{num_rounds} SUMMARY\n")
            human_log_file.write("="*80 + "\n\n")
            
            human_log_file.write(f"📊 External demand (Retailer): {retailer_demand}\n")
            human_log_file.write(f"📦 Orders received per agent: {orders_received_from_downstream}\n")
            human_log_file.write(f"🚚 Shipments received per agent: {shipments_received_list}\n\n")
            
            # Log agent state summary
            human_log_file.write("🏢 AGENT STATE SUMMARY\n")
            human_log_file.write("-" * 50 + "\n")
            for idx, agent in enumerate(agents):
                human_log_file.write(f"• {agent.role_name}: Inventory: {agent.inventory}, Backlog: {agent.backlog}, ")
                human_log_file.write(f"Order placed: {orders_placed[idx]}, Units sold: {shipments_sent_downstream[idx]}, ")
                human_log_file.write(f"Balance: ${agent.balance:.2f}\n")
            
            human_log_file.write(f"\n💬 Communication messages: {len([msg for msg in communication_messages if msg['round'] == round_index])}\n")
            human_log_file.write(f"🤖 LLM calls this round: {len(agents) * (communication_rounds + 1) if enable_communication else len(agents)}\n\n")
            
            human_log_file.flush()

        if logger:
            logger.log(f"Orders received per agent: {orders_received_from_downstream}")
            logger.log(f"Shipments received per agent: {shipments_received_list}")
            for idx, agent in enumerate(agents):
                if logger:
                    logger.log(f"Agent: {agent.role_name}: Inventory: {agent.inventory}, Backlog: {agent.backlog}, Order placed: {orders_placed[idx]}, Total Profit: {agent.profit_accumulated}")

# --------------------------------------------------------------------
# 6. Putting It All Together: run multiple generations
# --------------------------------------------------------------------

async def run_beer_game_simulation(
    num_rounds: int = 20,
    temperature: float = 0,
    logger: BeerGameLogger = None,
    enable_communication: bool = False,
    communication_rounds: int = 2,
    enable_memory: bool = False,
    memory_retention_rounds: int = 5,
    enable_shared_memory: bool = False,
    initial_inventory: int = 100,
    initial_backlog: int = 0,
    sale_price_per_unit: float = None,
    purchase_cost_per_unit: float = None,
    production_cost_per_unit: float = None,
    initial_balance: float = 1000.0,
    holding_cost_per_unit: float = None,
    backlog_cost_per_unit: float = None,
    enable_orchestrator: bool = False,
    orchestrator_history: int = 3,
    orchestrator_override: bool = False,
    longtermplanning_boolean: bool = False,
    safety_stock_target: float = 10.0,
    backlog_clearance_rate: float = 0.5,
    demand_smoothing_factor: float = 0.3
):
    """
    Orchestrates the Beer Game simulation with multiple rounds.
    Runs the specified number of rounds with agents adapting their strategies.
    """
    # Prepare folder for logs
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Updated path to point to simulation_results in parent directory
    base_results_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "simulation_results")

    # --- New logic for run_N_date folder naming ---
    # Create the base results folder if it doesn't exist
    os.makedirs(base_results_folder, exist_ok=True)
    
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
    human_log_file.write("🍺" + "="*78 + "🍺\n")
    human_log_file.write("🎮        MIT BEER GAME SIMULATION - DETAILED LLM LOG       🎮\n")
    human_log_file.write("🍺" + "="*78 + "🍺\n\n")
    
    human_log_file.write("📋 SIMULATION PARAMETERS\n")
    human_log_file.write("-" * 50 + "\n")
    human_log_file.write(f"🎯 Number of rounds: {num_rounds}\n")
    human_log_file.write(f"🌡️  Temperature: {temperature}\n")
    human_log_file.write(f"💰 Initial balance: ${initial_balance:.2f}\n")
    human_log_file.write(f"📦 Initial inventory: {initial_inventory} units\n")
    human_log_file.write(f"📋 Initial backlog: {initial_backlog} units\n")
    human_log_file.write(f"💵 Sale price per unit: ${sale_price_per_unit:.2f}\n")
    human_log_file.write(f"💸 Purchase cost per unit: ${purchase_cost_per_unit:.2f}\n")
    human_log_file.write(f"🏭 Production cost per unit: ${production_cost_per_unit:.2f}\n")
    human_log_file.write(f"🏪 Holding cost per unit: ${holding_cost_per_unit:.2f}\n")
    human_log_file.write(f"⏰ Backlog cost per unit: ${backlog_cost_per_unit:.2f}\n")
    human_log_file.write(f"💬 Communication enabled: {enable_communication}\n")
    if enable_communication:
        human_log_file.write(f"🔄 Communication rounds: {communication_rounds}\n")
    human_log_file.write(f"🧠 Memory enabled: {enable_memory}\n")
    if enable_memory:
        human_log_file.write(f"📚 Memory retention rounds: {memory_retention_rounds}\n")
    human_log_file.write(f"🎭 Shared memory enabled: {enable_shared_memory}\n")
    human_log_file.write(f"🎼 Orchestrator enabled: {enable_orchestrator}\n")
    if enable_orchestrator:
        human_log_file.write(f"📊 Orchestrator history: {orchestrator_history}\n")
        human_log_file.write(f"⚡ Orchestrator override: {orchestrator_override}\n")
    human_log_file.write("\n")

    if logger is None:
        logger = BeerGameLogger()

    # Initialize the roles (4 default roles)
    roles = ["Retailer", "Wholesaler", "Distributor", "Factory"]
    agents = [BeerGameAgent.create_agent(role_name=role, initial_inventory=initial_inventory, 
                                        initial_backlog=initial_backlog, initial_balance=initial_balance,
                                        logger=logger, human_log_file=human_log_file) for role in roles]

    # Each agent obtains an initial strategy from the LLM
    if human_log_file:
        human_log_file.write("\n🚀 AGENT INITIALIZATION PHASE\n")
        human_log_file.write("="*80 + "\n\n")
        human_log_file.flush()
    
    await asyncio.gather(*(agent.initialize_strategy(temperature=temperature, selling_price_per_unit=sale_price_per_unit, unit_cost_per_unit=purchase_cost_per_unit if agent.role_name != "Factory" else production_cost_per_unit, holding_cost_per_unit=holding_cost_per_unit, backlog_cost_per_unit=backlog_cost_per_unit, safety_stock_target=safety_stock_target, backlog_clearance_rate=backlog_clearance_rate, demand_smoothing_factor=demand_smoothing_factor) for agent in agents))
    
    if human_log_file:
        human_log_file.write("\n✅ All agents initialized successfully\n\n")
        human_log_file.flush()

    # Example external demand across rounds:
    # In real usage, you might load or generate random demands.
    # Generate external demand using a normal distribution centered at 10, clipped to [0, 20] and rounded to int
    external_demand_pattern = [
        int(min(20, max(0, round(random.gauss(10, 3)))))
        for _ in range(num_rounds)
    ]

    sim_data = SimulationData(hyperparameters={
        "num_rounds": num_rounds,
        "holding_cost_per_unit": 0.5,
        "backlog_cost_per_unit": 1.5,
        "sale_price_per_unit": sale_price_per_unit,
        "purchase_cost_per_unit": purchase_cost_per_unit,
        "production_cost_per_unit": production_cost_per_unit,
        "roles": roles,
        "timestamp": current_time,
        "external_demand_pattern": external_demand_pattern,
        "enable_communication": enable_communication,
        "communication_rounds": communication_rounds,
        "enable_memory": enable_memory,
        "memory_retention_rounds": memory_retention_rounds,
        "enable_shared_memory": enable_shared_memory,
        "initial_inventory": initial_inventory,
        "initial_backlog": initial_backlog,
        "initial_balance": initial_balance
    })

    csv_log_path = os.path.join(results_folder, "beer_game_detailed_log.csv")
    json_log_path = os.path.join(results_folder, "beer_game_detailed_log.json")
    # Files will be created and overwritten after each round

    logger.log(f"\n--- Starting Simulation ---")
    print(f"\n🎮 Starting Beer Game Simulation with {num_rounds} rounds")

    logger.log(f"\n============================================================\nBeer Game Simulation - {num_rounds} Rounds\n============================================================")

    await run_beer_game_generation(
        agents=agents,
        external_demand=external_demand_pattern,
        num_rounds=num_rounds,
        holding_cost_per_unit=holding_cost_per_unit,
        backlog_cost_per_unit=backlog_cost_per_unit,
        sale_price_per_unit=sale_price_per_unit,
        purchase_cost_per_unit=purchase_cost_per_unit,
        production_cost_per_unit=production_cost_per_unit,
        temperature=temperature,
        generation_index=1,
        sim_data=sim_data,
        human_log_file=human_log_file,
        logger=logger,
        csv_log_path=csv_log_path,
        json_log_path=json_log_path,
        enable_communication=enable_communication,
        communication_rounds=communication_rounds,
        enable_memory=enable_memory,
        memory_retention_rounds=memory_retention_rounds,
        enable_shared_memory=enable_shared_memory,
        enable_orchestrator=enable_orchestrator,
        orchestrator_history=orchestrator_history,
        orchestrator_override=orchestrator_override,
        initial_inventory=initial_inventory,
        initial_backlog=initial_backlog,
        initial_balance=initial_balance,
        longtermplanning_boolean=longtermplanning_boolean,
        safety_stock_target=safety_stock_target,
        backlog_clearance_rate=backlog_clearance_rate,
        demand_smoothing_factor=demand_smoothing_factor
    )

    print(f"✅ Simulation complete")

    # Simple visualizations (inventory/backlog over time, cost, etc.)
    # Ensure df_rounds is defined for saving and plotting
    df_rounds = pd.DataFrame([asdict(r) for r in sim_data.rounds_log])
    
    # JSON file is already written after each round, so no need to write again here

    # Generate visualizations
    # Create run settings dictionary for final plot
    from llm_calls_mitb_game import MODEL_NAME as _RUN_MODEL_NAME
    run_settings = {
        'num_rounds': num_rounds,
        'temperature': temperature,
        'model_name': _RUN_MODEL_NAME,
        'sale_price_per_unit': sale_price_per_unit,
        'purchase_cost_per_unit': purchase_cost_per_unit,
        'production_cost_per_unit': production_cost_per_unit,
        'holding_cost_per_unit': holding_cost_per_unit,
        'backlog_cost_per_unit': backlog_cost_per_unit,
        'enable_communication': enable_communication,
        'communication_rounds': communication_rounds,
        'enable_memory': enable_memory,
        'memory_retention_rounds': memory_retention_rounds,
        'enable_orchestrator': enable_orchestrator,
        'orchestrator_override': orchestrator_override,
        'initial_inventory': initial_inventory,
        'initial_backlog': initial_backlog,
        'initial_balance': initial_balance
    }
    plot_beer_game_results(df_rounds, results_folder, external_demand_pattern, run_settings)

    # Calculate Nash equilibrium deviations (assumed equilibrium order quantity = 10)
    deviations = calculate_nash_deviation(df_rounds, equilibrium_order=10)
    human_log_file.write("\n" + "="*80 + "\n")
    human_log_file.write("📊 SIMULATION RESULTS & ANALYSIS\n")
    human_log_file.write("="*80 + "\n\n")
    
    human_log_file.write("🎯 NASH EQUILIBRIUM ANALYSIS\n")
    human_log_file.write("-" * 50 + "\n")
    for role, dev in deviations.items():
        human_log_file.write("🏢 {}: Average Absolute Deviation: {:.2f}\n".format(role, dev))

    # Log LLM session summary
    from llm_calls_mitb_game import lite_client, get_default_client
    client = lite_client or get_default_client()
    session_summary = client.get_session_summary()
    
    print(f"\n📈 Simulation Summary:")
    print(f"   Total Rounds: {num_rounds}")
    print(f"   Total LLM Calls: {session_summary['total_calls']}")
    print(f"   Total Cost: ${session_summary['total_cost_usd']:.2f}")
    print(f"   Results saved to: {results_folder}")
    
    # Keep detailed logging only in files
    # print(f"\n🎯 [LLM SESSION SUMMARY]")  # Commented out
    # print(f"   📞 Total LLM Calls: {session_summary['total_calls']}")  # Commented out
    # print(f"   💰 Total Cost: ${session_summary['total_cost_usd']}")  # Commented out
    # print(f"   📝 Total Tokens: {session_summary['total_tokens']} ({session_summary['total_input_tokens']} in + {session_summary['total_output_tokens']} out)")  # Commented out
    # print(f"   ⏱️  Total Inference Time: {session_summary['total_inference_time_seconds']}s")  # Commented out
    # print(f"   📊 Average per Call: {session_summary['average_inference_time_seconds']:.3f}s, ${session_summary['average_cost_per_call_usd']:.6f}")  # Commented out
    
    human_log_file.write("\n🤖 LLM SESSION SUMMARY\n")
    human_log_file.write("-" * 50 + "\n")
    human_log_file.write("📞 Total LLM Calls: {}\n".format(session_summary['total_calls']))
    human_log_file.write("💰 Total Cost: ${}\n".format(session_summary['total_cost_usd']))
    human_log_file.write("📝 Total Tokens: {} ({} input + {} output)\n".format(
        session_summary['total_tokens'], 
        session_summary['total_input_tokens'], 
        session_summary['total_output_tokens']
    ))
    human_log_file.write("⏱️  Total Inference Time: {}s\n".format(session_summary['total_inference_time_seconds']))
    human_log_file.write("📊 Average per Call: {:.3f}s, ${:.6f}\n".format(
        session_summary['average_inference_time_seconds'], 
        session_summary['average_cost_per_call_usd']
    ))
    
    human_log_file.write("\n🍺" + "="*78 + "🍺\n")
    human_log_file.write("🎮           SIMULATION COMPLETED SUCCESSFULLY!           🎮\n")
    human_log_file.write("🍺" + "="*78 + "🍺\n")

    # Save session summary to JSON
    session_summary_path = os.path.join(results_folder, "llm_session_summary.json")
    with open(session_summary_path, 'w') as f:
        json.dump(session_summary, f, indent=2)
    
    # Move LLM metrics file to results folder if it exists
    metrics_file = "llm_inference_metrics.json"
    if os.path.exists(metrics_file):
        metrics_dest = os.path.join(results_folder, "llm_inference_metrics.json")
        os.rename(metrics_file, metrics_dest)
        # print(f"📋 LLM metrics saved to: {metrics_dest}")  # Commented out

    human_log_file.close()
    logger.log("\nSimulation complete. Results saved to: {}".format(results_folder))
    logger.close()
    return sim_data

# --------------------------------------------------------------------
# 7. The default entry point for convenience
# --------------------------------------------------------------------

if __name__ == "__main__":
    # Run the simulation with default settings
    asyncio.run(run_beer_game_simulation(
        num_rounds=30,
        temperature=0.7,
        logger=None
    ))
