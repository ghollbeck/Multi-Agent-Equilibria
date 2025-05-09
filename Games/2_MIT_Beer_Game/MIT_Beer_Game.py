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
import nest_asyncio  # required to apply asyncio loops in scripts
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

# Default LLM model for all chat completions
MODEL_NAME: str = "gpt-4o-mini"

# --------------------------------------------------------------------
# Utility for robust JSON extraction from LLM responses
# --------------------------------------------------------------------
def safe_parse_json(response_str: str) -> dict:
    """Extract the first JSON object in the LLM response string and parse it, handling markdown fences and incomplete JSON."""
    import re
    s = response_str.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines)
    start = s.find('{')
    if start == -1:
        try:
            return json.loads(s)
        except Exception as e:
            print(f"❌ [safe_parse_json] Could not find '{{' in response. Error: {e}. Response: {s}")
            raise
    substring = s[start:]
    brace_count = 0
    end_index = None
    for idx, ch in enumerate(substring):
        if ch == '{':
            brace_count += 1
        elif ch == '}':
            brace_count -= 1
            if brace_count == 0:
                end_index = idx + 1
                break
    if end_index is not None:
        json_text = substring[:end_index]
    else:
        json_text = substring.rstrip(', \n\r\t')
        json_text += '}' * brace_count
    try:
        return json.loads(json_text)
    except Exception as e:
        print(f"❌ [safe_parse_json] Failed to parse JSON. Error: {e}. Extracted: {json_text}")
        raise

def parse_json_with_default(response_str: str, default: dict, context: str) -> dict:
    """Parse JSON and return default value on failure, logging the error context."""
    try:
        return safe_parse_json(response_str)
    except Exception as e:
        print(f"❌ [parse_json_with_default] Error parsing JSON in {context}: {e}. Response was: {response_str!r}")
        # Attempt to salvage partial data (e.g., order_quantity)
        m = re.search(r'"order_quantity"\s*:\s*(\d+)', response_str)
        if m:
            salvaged = default.copy()
            salvaged['order_quantity'] = int(m.group(1))
            print(f"❌ [parse_json_with_default] Salvaged order_quantity={m.group(1)} in {context}.")
            return salvaged
        return default

# Helper to fill backlog and new demand for an agent and return units served
# AFTER  (put this once, e.g. utilities.py)
def fulfill(agent, new_demand: int, incoming: int = 0, logger=None) -> int:
    """
    •  Add today's inbound shipment to inventory.
    •  Satisfy backlog first, then today's demand.
    •  Update inventory & backlog in‑place; return units shipped downstream.
    """
    # 1. receive today's shipment
    agent.inventory += incoming

    # 2. figure out how many units are needed in total
    need   = agent.backlog + new_demand
    served = min(agent.inventory, need)

    # 3. update state
    agent.inventory -= served
    agent.backlog    = need - served   # whatever we could not serve

    if logger:
        logger.log(
            f"{agent.role_name}: inbound={incoming}, demand={new_demand}, "
            f"served={served}, new_inv={agent.inventory}, new_backlog={agent.backlog}"
        )
    return served


# --------------------------------------------------------------------
# 1. LLM Prompt Classes
#    Replace any references to "cooperate/defect" with Beer Game logic.
# --------------------------------------------------------------------

class BeerGamePrompts:
    """
    Provides system and user prompts for the MIT Beer Game.
    Each agent will use these to:
      1. Generate an initial ordering strategy,
      2. Update the strategy after each generation,
      3. Decide on a new order quantity each round.
    """

    @staticmethod
    def get_strategy_generation_prompt(role_name: str, inventory: int = 100, backlog: int = 0, profit_per_unit_sold: float = 5) -> str:
        """
        Prompt to generate an initial ordering strategy for a given role.
        The LLM must return valid JSON with the required keys.

        role_name can be one of: ["Retailer", "Wholesaler", "Distributor", "Factory"].
        """
        return f"""
        You are the {role_name} in the MIT Beer Game. 
        Your task is to develop an ordering strategy that will minimize total costs 
        (holding costs + backlog costs - profits) over multiple rounds.

        Current State:
          • Initial Inventory: {inventory} units
          • Initial Backlog: {backlog} units
          • Profit per unit sold: ${profit_per_unit_sold}

        Consider:
          • Your current role's position in the supply chain
          • You have a 1-round lead time for the orders you place
          • You observe demand (if Retailer) or incoming orders (for other roles)
          • You want to avoid large swings (the Bullwhip effect)
          • You have a holding cost of 0.5 per unit per round
          • You have a backlog cost of 1.5 per unit per round of unmet demand (3x higher than holding cost)
          • You earn ${profit_per_unit_sold} profit for each unit sold
         • IMPORTANT: When determining order quantities, you must account for BOTH your current backlog AND expected new demand - backlog represents unfilled orders that must be fulfilled in addition to meeting new demand
         • Never let the inventory go to zero.

        Please return only valid JSON with the following fields in order:

        {{
          "role_name": "{role_name}",
          "inventory": {inventory},
          "backlog": {backlog},
          "confidence": <float between 0 and 1>,
          "rationale": "<brief explanation of your reasoning>",
          "risk_assessment": "<describe any risks you anticipate>",
          "expected_demand_next_round": <integer>,
          "order_quantity": <integer>
        }}

        IMPORTANT: Output ONLY the JSON object, with no markdown, no triple backticks, no code fences, and do NOT write the word 'json' anywhere. Your reply must be a single valid JSON object, nothing else. If you include anything else, your answer will be rejected. KEEP IT RATHER SHORT
        """
        # You can add role-specific instructions if needed
        # for advanced prompts. For now, it's mostly generic.

    @staticmethod
    def get_strategy_update_prompt(role_name: str, performance_log: str, current_strategy: dict, 
                                 inventory: int = 100, backlog: int = 0, profit_per_unit_sold: float = 5) -> str:
        """
        Prompt to update an existing strategy after completing a generation.
        performance_log is a text summary of the agent's costs, backlog, bullwhip, etc.
        current_strategy is the JSON dict from the agent's prior strategy.

        The LLM must again return valid JSON with the required keys.
        """
        return f"""
        You are the {role_name} in the MIT Beer Game. 
        
        Current State:
          • Current Inventory: {inventory} units
          • Current Backlog: {backlog} units
          • Profit per unit sold: ${profit_per_unit_sold}
        
        Here is your recent performance log:
        {performance_log}

        Your current strategy is:
        {json.dumps(current_strategy, indent=2)}

        Based on your performance and the desire to minimize holding & backlog costs while maximizing profits, 
        please propose any improvements to your ordering policy. 
        
        Remember:
          • You have a holding cost of 0.5 per unit per round
          • You have a backlog cost of 1.5 per unit per round of unmet demand (3x higher than holding cost)
          • You earn ${profit_per_unit_sold} profit for each unit sold
         • IMPORTANT: When determining order quantities, you must account for BOTH your current backlog AND expected new demand - backlog represents unfilled orders that must be fulfilled in addition to meeting new demand
          • Never let the inventory go to zero.
          
        Return only valid JSON with the following fields in order:

        {{
          "role_name": "{role_name}",
          "inventory": {inventory},
          "backlog": {backlog},
          "confidence": <float between 0 and 1>,
          "rationale": "<brief explanation>",
          "risk_assessment": "<describe any risks>",
          "expected_demand_next_round": <integer>,
          "order_quantity": <integer>
        }}

        IMPORTANT: Output ONLY the JSON object, with no markdown, no triple backticks, no code fences, and do NOT write the word 'json' anywhere. Your reply must be a single valid JSON object, nothing else. If you include anything else, your answer will be rejected. KEEP IT RATHER SHORT
        """

    @staticmethod
    def get_decision_prompt(role_name: str, 
                            inventory: int, 
                            backlog: int, 
                            recent_demand_or_orders: List[int], 
                            incoming_shipments: List[int],
                            current_strategy: dict,
                            profit_per_unit_sold: float = 5,
                            last_order_placed: int = None,
                            last_profit: float = None) -> str:
        """
        Prompt to decide this round's order quantity, given the latest state.
        The agent must return valid JSON with the required keys.

        - role_name: "Retailer", "Wholesaler", "Distributor", or "Factory"
        - inventory: current on-hand inventory
        - backlog: current unmet demand
        - recent_demand_or_orders: a short history of demands/orders from downstream
        - incoming_shipments: the shipments that are arriving this round (from lead times)
        - current_strategy: the agent's strategy as a JSON dict
        - profit_per_unit_sold: profit earned per unit sold
        - last_order_placed: the order placed in the previous round
        - last_profit: the profit made in the previous round
        """
        return f"""
        You are the {role_name} in the MIT Beer Game. 
        Current State:
          - Inventory: {inventory} units
          - Backlog: {backlog} units
          - Recent downstream demand or orders: {recent_demand_or_orders}
          - Incoming shipments this round: {incoming_shipments}
          - Profit per unit sold: ${profit_per_unit_sold}
          - Last order placed: {last_order_placed}
          - Last round profit: {last_profit}

        Your known lead time is 1 round for any order you place.

        Economics:
          - Holding cost: $0.5 per unit per round
          - Backlog cost: $1.5 per unfilled unit per round (3x higher than holding cost)
          - Profit: ${profit_per_unit_sold} per unit sold
          - Never let the inventory go to zero.

        **Important:**
        - You should avoid letting your inventory reach zero, as this causes stockouts and lost sales.
        - When deciding how much to order, consider your expected demand and spending over the next round (the lead time before your order arrives).
        - CRITICAL: You must account for BOTH your current backlog ({backlog} units) AND expected new demand. The backlog represents unfilled orders that must be fulfilled - your order quantity should cover both clearing backlog and meeting new demand.
        - Review how much you have ordered and earned in the last round(s) to inform your decision.
        - Try to maintain a buffer of inventory to cover expected demand during the lead time.

        Current Strategy:
        {json.dumps(current_strategy, indent=2)}

        Given this state, return valid JSON with the following fields in order:

        {{
          "role_name": "{role_name}",
          "inventory": {inventory},
          "backlog": {backlog},
          "recent_demand_or_orders": {recent_demand_or_orders},
          "incoming_shipments": {incoming_shipments},
          "last_order_placed": {last_order_placed},
          "expected_demand_next_round": <integer>,
          "confidence": <float between 0 and 1>,
          "rationale": "<brief explanation>",
          "risk_assessment": "<describe any risks>",
          "order_quantity": <integer>
        }}

        IMPORTANT: Output ONLY the JSON object, with no markdown, no triple backticks, no code fences, and do NOT write the word 'json' anywhere. Your reply must be a single valid JSON object, nothing else. If you include anything else, your answer will be rejected. KEEP IT RATHER SHORT
        """

# --------------------------------------------------------------------
# 2. Data Classes for Logging & Simulation State
# --------------------------------------------------------------------

@dataclass
class RoundData:
    """
    Records data from each round of the Beer Game for a single agent:
    - round_index: which round in the generation
    - role_name: Retailer / Wholesaler / Distributor / Factory
    - inventory: units on hand at the end of the round
    - backlog: unmet demand
    - order_placed: how many units were ordered from upstream
    - shipment_received: how many units arrived from upstream
    - shipment_sent_downstream: how many units were shipped to downstream
    - cost: total cost incurred this round
    """
    generation: int
    round_index: int
    role_name: str
    inventory: int
    backlog: int
    order_placed: int
    shipment_received: int
    shipment_sent_downstream: int
    profit: float

@dataclass
class SimulationData:
    hyperparameters: dict
    rounds_log: List[RoundData] = field(default_factory=list)

    def add_round_entry(self, entry: RoundData):
        self.rounds_log.append(entry)

    def to_dict(self):
        return {
            'hyperparameters': self.hyperparameters,
            'rounds_log': [asdict(r) for r in self.rounds_log]
        }

# --------------------------------------------------------------------
# Logger for capturing all simulation events, prompts, and responses
# --------------------------------------------------------------------
class BeerGameLogger:
    def __init__(self, log_to_file=None):
        self.logs = []
        self.log_to_file = log_to_file
        self.file_handle = open(log_to_file, 'a') if log_to_file else None
    def log(self, msg):
        self.logs.append(msg)
        print(msg)
        if self.file_handle:
            self.file_handle.write(msg + '\n')
    def get_logs(self):
        return self.logs
    def close(self):
        if self.file_handle:
            self.file_handle.close()

# --------------------------------------------------------------------
# 3. LLM Client Setup
#    (Replace with your own concurrency or provider if needed)
# --------------------------------------------------------------------

load_dotenv()

class LiteLLMClient:
    def __init__(self, logger=None):
        self.api_key = os.getenv("LITELLM_API_KEY")
        self.endpoint = "https://litellm.sph-prod.ethz.ch/chat/completions"
        self.semaphore = asyncio.Semaphore(2)
        self.logger = logger

    async def chat_completion(self, model: str, system_prompt: str, user_prompt: str, temperature: float = 0.7, max_tokens: int = 450):
        print(f"[LLM SYSTEM PROMPT]: {system_prompt}")
        print(f"[LLM USER PROMPT]: {user_prompt}")
        if self.logger:
            self.logger.log(f"[LLM SYSTEM PROMPT]: {system_prompt}")
            self.logger.log(f"[LLM USER PROMPT]: {user_prompt}")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        max_retries = 5
        delay = 2
        attempt = 0
        while True:
            async with self.semaphore:
                response = await asyncio.to_thread(requests.post, self.endpoint, json=payload, headers=headers)
            if response.ok:
                break
            elif response.status_code == 429:
                attempt += 1
                if attempt >= max_retries:
                    raise Exception(f"LiteLLM API error: {response.status_code} {response.text}")
                await asyncio.sleep(delay)
                delay *= 2
            else:
                raise Exception(f"LiteLLM API error: {response.status_code} {response.text}")
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "No response from LiteLLM.")
        print(f"[LLM RAW RESPONSE]: {content}")
        if self.logger:
            self.logger.log(f"[LLM RAW RESPONSE]: {content}")
        return content

lite_client = LiteLLMClient()

# --------------------------------------------------------------------
# 4. Agent Class: BeerGameAgent
#    Each Agent has: role, inventory, backlog, cost, strategy, etc.
# --------------------------------------------------------------------

class BeerGameAgent(BaseModel):
    role_name: str  # "Retailer", "Wholesaler", "Distributor", "Factory"
    inventory: int = 100
    backlog: int = 0
    profit_accumulated: float = 0.0
    # Store last round profit and last order placed for context in prompts
    last_profit: Optional[float] = None
    last_order_placed: Optional[int] = None
    
    # Start with 10 units already in transit so agents receive shipments in the first rounds
    # 1-round lead time: position 0 = what's arriving this round, position 1 = what arrives next round
    shipments_in_transit: Dict[int,int] = Field(default_factory=lambda: {0:10, 1:10})
    
    # Orders from the downstream agent (or external demand if Retailer)
    # We keep a short log: the most recent N rounds (for context).
    downstream_orders_history: List[int] = Field(default_factory=list)
    
    # The LLM-based strategy
    strategy: dict = Field(default_factory=dict)
    
    # For demonstration, each agent can have its own prompts class, 
    # or we can store a reference to a shared prompts. We'll assume a single set here.
    prompts: ClassVar[BeerGamePrompts] = BeerGamePrompts
    logger: BeerGameLogger = None
    # For logging LLM I/O
    last_decision_prompt: str = ""
    last_decision_output: dict = Field(default_factory=dict)
    last_update_prompt: str = ""
    last_update_output: dict = Field(default_factory=dict)
    last_init_prompt: str = ""
    last_init_output: dict = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    async def initialize_strategy(self, temperature=0.7, profit_per_unit_sold=5):
        """
        Generate initial strategy JSON from the LLM.
        """
        if self.logger:
            self.logger.log(f"[Agent {self.role_name}] Initializing strategy...")
        prompt = self.prompts.get_strategy_generation_prompt(self.role_name, self.inventory, self.backlog, profit_per_unit_sold)
        self.last_init_prompt = prompt
        system_prompt = "You are an expert supply chain manager. Return valid JSON only."
        try:
            response_str = await lite_client.chat_completion(model=MODEL_NAME,
                                                      system_prompt=system_prompt,
                                                      user_prompt=prompt,
                                                      temperature=temperature)
        except Exception as e:
            print(f"❌ [Agent {self.role_name}] initialize_strategy: LLM call failed. Error: {e}")
            response_str = ''
        print(f"[Agent {self.role_name}] Strategy response: {response_str}")
        if self.logger:
            self.logger.log(f"[Agent {self.role_name}] Strategy response: {response_str}")
        default_strategy = {
            "order_quantity": 10,
            "confidence": 1.0,
            "rationale": "Default initial strategy",
            "risk_assessment": "No risk",
            "expected_demand_next_round": 10
        }
        try:
            response = safe_parse_json(response_str)
            print(f"✅ [Agent {self.role_name}] initialize_strategy: Valid JSON received.")
        except Exception as e:
            print(f"❌ [Agent {self.role_name}] initialize_strategy: Invalid JSON. Using default. Error: {e}")
            response = default_strategy
        self.strategy = response
        self.last_init_output = response

    async def update_strategy(self, performance_log: str, temperature=0.7, profit_per_unit_sold=5):
        """
        Update an existing strategy after a generation completes.
        """
        if self.logger:
            self.logger.log(f"[Agent {self.role_name}] Updating strategy with performance log: {performance_log}")
        prompt = self.prompts.get_strategy_update_prompt(self.role_name, performance_log, self.strategy, self.inventory, self.backlog, profit_per_unit_sold)
        self.last_update_prompt = prompt
        system_prompt = "You are an expert supply chain manager. Return valid JSON only."
        try:
            response_str = await lite_client.chat_completion(model=MODEL_NAME,
                                                      system_prompt=system_prompt,
                                                      user_prompt=prompt,
                                                      temperature=temperature)
        except Exception as e:
            print(f"❌ [Agent {self.role_name}] update_strategy: LLM call failed. Error: {e}")
            response_str = ''
        print(f"[Agent {self.role_name}] Update response: {response_str}")
        if self.logger:
            self.logger.log(f"[Agent {self.role_name}] Update response: {response_str}")
        default_update = {
            "order_quantity": 10,
            "confidence": 1.0,
            "rationale": "Default update strategy",
            "risk_assessment": "No risk",
            "expected_demand_next_round": 10
        }
        try:
            response = safe_parse_json(response_str)
            print(f"✅ [Agent {self.role_name}] update_strategy: Valid JSON received.")
        except Exception as e:
            print(f"❌ [Agent {self.role_name}] update_strategy: Invalid JSON. Using default. Error: {e}")
            response = default_update
        self.strategy = response
        self.last_update_output = response

    async def decide_order_quantity(self, temperature=0.7, profit_per_unit_sold=5) -> dict:
        """
        Ask the LLM how many units to order from upstream in this round, 
        given our current state. Must return the JSON dict with 'order_quantity', etc.
        """
        if self.logger:
            self.logger.log(f"[Agent {self.role_name}] Deciding order quantity. Inventory: {self.inventory}, Backlog: {self.backlog}, Downstream: {self.downstream_orders_history[-3:]}, Shipments: {[self.shipments_in_transit[1]]}")
        # Get last order placed and last profit for context
        last_order_placed = self.last_order_placed
        last_profit = self.last_profit
        prompt = self.prompts.get_decision_prompt(
            role_name=self.role_name,
            inventory=self.inventory,
            backlog=self.backlog,
            recent_demand_or_orders=self.downstream_orders_history[-3:],
            incoming_shipments=[self.shipments_in_transit[1]],
            current_strategy=self.strategy,
            profit_per_unit_sold=profit_per_unit_sold,
            last_order_placed=last_order_placed,
            last_profit=last_profit
        )
        self.last_decision_prompt = prompt
        system_prompt = "You are an expert supply chain manager. Return valid JSON only."
        try:
            response_str = await lite_client.chat_completion(model=MODEL_NAME,
                                                      system_prompt=system_prompt,
                                                      user_prompt=prompt,
                                                      temperature=temperature)
        except Exception as e:
            print(f"❌ [Agent {self.role_name}] decide_order_quantity: LLM call failed. Error: {e}")
            response_str = ''
        print(f"[Agent {self.role_name}] Decision response: {response_str}")
        if self.logger:
            self.logger.log(f"[Agent {self.role_name}] Decision response: {response_str}")
        default_decision = {
            "order_quantity": 10,
            "confidence": 1.0,
            "rationale": "Default decision",
            "risk_assessment": "No risk",
            "expected_demand_next_round": 10
        }
        try:
            response = safe_parse_json(response_str)
            print(f"✅ [Agent {self.role_name}] decide_order_quantity: Valid JSON received.")
        except Exception as e:
            print(f"❌ [Agent {self.role_name}] decide_order_quantity: Invalid JSON. Using default. Error: {e}")
            response = default_decision
        self.last_decision_output = response
        # Store last profit for next round context
        self.last_profit = response.get('profit', None)
        return response

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
    json_log_path: str = None
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

        # 5. Each role decides on new order quantity from upstream
        order_decision_tasks = []
        for agent in agents:
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
                # Write to CSV after each round
                if csv_log_path:
                    write_header = not os.path.exists(csv_log_path) or os.path.getsize(csv_log_path) == 0
                    # Define the new fieldnames based on LLM decision output
                    llm_output_keys = [
                        'llm_reported_inventory', 'llm_reported_backlog', 'llm_recent_demand_or_orders',
                        'llm_incoming_shipments', 'llm_last_order_placed', 'llm_confidence',
                        'llm_rationale', 'llm_risk_assessment', 'llm_expected_demand_next_round'
                    ]
                    fieldnames = list(asdict(entry).keys()) + [
                        'last_decision_output', 'last_update_output', 'last_init_output'
                    ] + llm_output_keys
                    with open(csv_log_path, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        if write_header:
                            writer.writeheader()
                        # Prepare the row data including new LLM fields
                        row_data = asdict(entry)
                        llm_decision = getattr(agent, 'last_decision_output', {})
                        row_data.update({
                            'last_decision_output': json.dumps(llm_decision),
                            'last_update_output': json.dumps(getattr(agent, 'last_update_output', {})),
                            'last_init_output': json.dumps(getattr(agent, 'last_init_output', {})),
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
    num_generations: int = 3,
    num_rounds_per_generation: int = 20,
    temperature: float = 0.7,
    logger: BeerGameLogger = None
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
            json_log_path=json_log_path
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
    # Persist aggregated logs
    # df_rounds.to_csv(csv_log_path, index=False)
    # df_rounds.to_json(json_log_path, orient="records", indent=2)
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

    human_log_file.close()
    logger.log("\nSimulation complete. Results saved to: {}".format(results_folder))
    logger.close()
    return sim_data

# --------------------------------------------------------------------
# 7. Visualization
# --------------------------------------------------------------------

def plot_beer_game_results(rounds_df: pd.DataFrame, results_folder: str):
    """
    Basic plots: inventory over time, backlog over time, cost, etc.
    """
    os.makedirs(results_folder, exist_ok=True)

    # Inventory by role
    plt.figure(figsize=(10,6))
    for role in rounds_df["role_name"].unique():
        subset = rounds_df[rounds_df["role_name"] == role]
        plt.plot(subset["round_index"], subset["inventory"], label=role)
    plt.title("Inventory Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Units in Inventory")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_folder, "inventory_over_time.png"))
    plt.close()

    # Backlog by role
    plt.figure(figsize=(10,6))
    for role in rounds_df["role_name"].unique():
        subset = rounds_df[rounds_df["role_name"] == role]
        plt.plot(subset["round_index"], subset["backlog"], label=role)
    plt.title("Backlog Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Unmet Demand (Backlog)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_folder, "backlog_over_time.png"))
    plt.close()

    # Cost by role (accumulated)
    plt.figure(figsize=(10,6))
    for role in rounds_df["role_name"].unique():
        subset = rounds_df[rounds_df["role_name"] == role]
        plt.plot(subset["round_index"], subset["profit"], label=role)
    plt.title("Accumulated Profit Over Time")
    plt.xlabel("Round")
    plt.ylabel("Accumulated Profit")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_folder, "cost_over_time.png"))
    plt.close()


    # Combined Plot with subplots for Inventory, Backlog, and Cost
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))

    # Subplot 1: Inventory by role
    for role in rounds_df["role_name"].unique():
        subset = rounds_df[rounds_df["role_name"] == role]
        axes[0].plot(subset["round_index"], subset["inventory"], label=role)
    axes[0].set_title("Inventory Over Rounds")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Units in Inventory")
    axes[0].legend()
    axes[0].grid(True)

    # Subplot 2: Backlog by role
    for role in rounds_df["role_name"].unique():
        subset = rounds_df["role_name"] == role
        subset = rounds_df[rounds_df["role_name"] == role]
        axes[1].plot(subset["round_index"], subset["backlog"], label=role)
    axes[1].set_title("Backlog Over Rounds")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Unmet Demand (Backlog)")
    axes[1].legend()
    axes[1].grid(True)

    # Subplot 3: Accumulated Cost Over Time
    for role in rounds_df["role_name"].unique():
        subset = rounds_df[rounds_df["role_name"] == role]
        axes[2].plot(subset["round_index"], subset["profit"], label=role)
    axes[2].set_title("Accumulated Profit Over Time")
    axes[2].set_xlabel("Round")
    axes[2].set_ylabel("Accumulated Profit")
    axes[2].legend()
    axes[2].grid(True)

    fig.tight_layout()
    combined_plot_path = os.path.join(results_folder, "combined_plots.png")
    plt.savefig(combined_plot_path)
    plt.close(fig)

def calculate_nash_deviation(rounds_df: pd.DataFrame, equilibrium_order: int = 10) -> Dict[str, float]:
    """
    Computes the average absolute deviation of the agent orders from the assumed Nash equilibrium order quantity.
    Returns a dictionary mapping each role to its average absolute deviation.
    """
    deviations = {}
    roles = rounds_df["role_name"].unique()
    for role in roles:
        role_df = rounds_df[rounds_df["role_name"] == role]
        avg_deviation = (role_df["order_placed"] - equilibrium_order).abs().mean()
        deviations[role] = avg_deviation
    print("\nNash Equilibrium Analysis (Assumed equilibrium order = {}):".format(equilibrium_order))
    for role, dev in deviations.items():
        print("Role: {} - Average Absolute Deviation: {:.2f}".format(role, dev))
    return deviations

# --------------------------------------------------------------------
# 8. Main Entry (if running as a script)
# --------------------------------------------------------------------

def main():
    nest_asyncio.apply()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_beer_game_simulation(
        num_generations=1,
        num_rounds_per_generation=7,
        temperature=0.7
    ))

if __name__ == "__main__":
    main()  # Run the simulation once