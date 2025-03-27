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
  - Each round, the agent decides an "order quantity" to place upstream, 
    guided by an LLM-generated strategy. 
  - After a multi-round generation, the agent can update its strategy based 
    on performance logs.

Game Flow (per round):
  1. The Retailer observes external customer demand. 
  2. Each role receives incoming shipments sent 2 rounds ago from upstream. 
  3. Each role attempts to fill downstream orders from current inventory 
     (unfilled orders become backlog). 
  4. Each role incurs cost based on holding and backlog. 
  5. Each role decides how many units to order from its upstream agent. 
  6. Orders are placed and queued, to be delivered after a 2-round lead time.

LLM Prompt/Response Requirements:
  - Agents ask: 
        "Given my current inventory, backlog, demand (or order history), 
         and cost performance, how many units should I order next round?"
  - The LLM must respond with valid JSON:
        {
          "order_quantity": <integer>,
          "confidence": <float>,
          "rationale": "explanation",
          "risk_assessment": "text",
          "expected_demand_next_round": <integer>
        }

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

"""

import os
import json
import time
import datetime
import random
import asyncio
import aiohttp
import requests
import nest_asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, ClassVar
from tqdm import tqdm

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
    def get_strategy_generation_prompt(role_name: str) -> str:
        """
        Prompt to generate an initial ordering strategy for a given role.
        The LLM must return valid JSON with the required keys.

        role_name can be one of: ["Retailer", "Wholesaler", "Distributor", "Factory"].
        """
        return f"""
        You are the {role_name} in the MIT Beer Game. 
        Your task is to develop an ordering strategy that will minimize total costs 
        (holding costs + backlog costs) over multiple rounds.

        Consider:
          • Your current role's position in the supply chain
          • You have a 2-round lead time for the orders you place
          • You observe demand (if Retailer) or incoming orders (for other roles)
          • You want to avoid large swings (the Bullwhip effect)
          • You have a holding cost of 0.5 per unit per round
          • You have a backlog cost of 1.0 per unit per round of unmet demand

        Please return only valid JSON with the following fields:

        {{
          "order_quantity": 10,
          "confidence": 0.9,
          "rationale": "Explain your reasoning briefly",
          "risk_assessment": "Describe any risks you anticipate",
          "expected_demand_next_round": 12
        }}

        The "order_quantity" is your suggested initial order policy 
        (e.g., a static guess or formula). 
        The "confidence" is a number between 0 and 1.
        """
        # You can add role-specific instructions if needed
        # for advanced prompts. For now, it's mostly generic.

    @staticmethod
    def get_strategy_update_prompt(role_name: str, performance_log: str, current_strategy: dict) -> str:
        """
        Prompt to update an existing strategy after completing a generation.
        performance_log is a text summary of the agent's costs, backlog, bullwhip, etc.
        current_strategy is the JSON dict from the agent's prior strategy.

        The LLM must again return valid JSON with the required keys.
        """
        return f"""
        You are the {role_name} in the MIT Beer Game. 
        Here is your recent performance log:
        {performance_log}

        Your current strategy is:
        {json.dumps(current_strategy, indent=2)}

        Based on your performance and the desire to minimize holding & backlog costs, 
        please propose any improvements to your ordering policy. 
        Return only valid JSON with the same structure as before:
        {{
          "order_quantity": <integer>,
          "confidence": <float>,
          "rationale": "...",
          "risk_assessment": "...",
          "expected_demand_next_round": <integer>
        }}
        """

    @staticmethod
    def get_decision_prompt(role_name: str, 
                            inventory: int, 
                            backlog: int, 
                            recent_demand_or_orders: List[int], 
                            incoming_shipments: List[int],
                            current_strategy: dict) -> str:
        """
        Prompt to decide this round's order quantity, given the latest state.
        The agent must return valid JSON with the required keys.

        - role_name: "Retailer", "Wholesaler", "Distributor", or "Factory"
        - inventory: current on-hand inventory
        - backlog: current unmet demand
        - recent_demand_or_orders: a short history of demands/orders from downstream
        - incoming_shipments: the shipments that are arriving this round (from lead times)
        - current_strategy: the agent's strategy as a JSON dict
        """
        return f"""
        You are the {role_name} in the MIT Beer Game. 
        Current State:
          - Inventory: {inventory}
          - Backlog: {backlog}
          - Recent downstream demand or orders: {recent_demand_or_orders}
          - Incoming shipments this round: {incoming_shipments}

        Your known lead time is 2 rounds for any order you place.

        Current Strategy:
        {json.dumps(current_strategy, indent=2)}

        Given this state, how many units do you order from your upstream supplier 
        **this round**? Return valid JSON with:

        {{
          "order_quantity": <integer>,
          "confidence": <float>,
          "rationale": "...",
          "risk_assessment": "...",
          "expected_demand_next_round": <integer>
        }}
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
    cost: float

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
# 3. LLM Client Setup
#    (Replace with your own concurrency or provider if needed)
# --------------------------------------------------------------------

load_dotenv()

class LiteLLMClient:
    def __init__(self):
        self.api_key = os.getenv("LITELLM_API_KEY")
        self.endpoint = "https://litellm.sph-prod.ethz.ch/chat/completions"
        self.semaphore = asyncio.Semaphore(2)

    async def chat_completion(self, model: str, system_prompt: str, user_prompt: str, temperature: float = 0.7, max_tokens: int = 150):
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
        return content

lite_client = LiteLLMClient()

# --------------------------------------------------------------------
# 4. Agent Class: BeerGameAgent
#    Each Agent has: role, inventory, backlog, cost, strategy, etc.
# --------------------------------------------------------------------

class BeerGameAgent(BaseModel):
    role_name: str  # "Retailer", "Wholesaler", "Distributor", "Factory"
    inventory: int = 12
    backlog: int = 0
    cost_accumulated: float = 0.0
    
    # We'll store a short queue for shipments in transit (2-round lead time).
    # shipments_in_transit[r] = how many units will arrive after r more rounds.
    shipments_in_transit: Dict[int,int] = Field(default_factory=lambda: {1:0, 2:0})
    
    # Orders from the downstream agent (or external demand if Retailer)
    # We keep a short log: the most recent N rounds (for context).
    downstream_orders_history: List[int] = Field(default_factory=list)
    
    # The LLM-based strategy
    strategy: dict = Field(default_factory=dict)
    
    # For demonstration, each agent can have its own prompts class, 
    # or we can store a reference to a shared prompts. We'll assume a single set here.
    prompts: ClassVar[BeerGamePrompts] = BeerGamePrompts

    class Config:
        arbitrary_types_allowed = True

    async def initialize_strategy(self, temperature=0.7):
        """
        Generate initial strategy JSON from the LLM.
        """
        prompt = self.prompts.get_strategy_generation_prompt(self.role_name)
        system_prompt = "You are an expert supply chain manager. Return valid JSON only."
        response_str = await lite_client.chat_completion(model="gpt-4o",
                                                      system_prompt=system_prompt,
                                                      user_prompt=prompt,
                                                      temperature=temperature)
        try:
            response = json.loads(response_str)
        except Exception as e:
            print(f"Error parsing JSON in initialize_strategy for {self.role_name}: {e}")
            response = {
                "order_quantity": 10,
                "confidence": 1.0,
                "rationale": "Default initial strategy",
                "risk_assessment": "No risk",
                "expected_demand_next_round": 10
            }
        self.strategy = response

    async def update_strategy(self, performance_log: str, temperature=0.7):
        """
        Update an existing strategy after a generation completes.
        """
        prompt = self.prompts.get_strategy_update_prompt(self.role_name, performance_log, self.strategy)
        system_prompt = "You are an expert supply chain manager. Return valid JSON only."
        response_str = await lite_client.chat_completion(model="gpt-4o",
                                                      system_prompt=system_prompt,
                                                      user_prompt=prompt,
                                                      temperature=temperature)
        try:
            response = json.loads(response_str)
        except Exception as e:
            print(f"Error parsing JSON in update_strategy for {self.role_name}: {e}")
            response = {
                "order_quantity": 10,
                "confidence": 1.0,
                "rationale": "Default update strategy",
                "risk_assessment": "No risk",
                "expected_demand_next_round": 10
            }
        self.strategy = response

    async def decide_order_quantity(self, temperature=0.7) -> dict:
        """
        Ask the LLM how many units to order from upstream in this round, 
        given our current state. Must return the JSON dict with 'order_quantity', etc.
        """
        prompt = self.prompts.get_decision_prompt(
            role_name=self.role_name,
            inventory=self.inventory,
            backlog=self.backlog,
            recent_demand_or_orders=self.downstream_orders_history[-3:],
            incoming_shipments=[self.shipments_in_transit[1]],
            current_strategy=self.strategy
        )
        system_prompt = "You are an expert supply chain manager. Return valid JSON only."
        response_str = await lite_client.chat_completion(model="gpt-4o",
                                                      system_prompt=system_prompt,
                                                      user_prompt=prompt,
                                                      temperature=temperature)
        try:
            response = json.loads(response_str)
        except Exception as e:
            print(f"Error parsing JSON in decide_order_quantity for {self.role_name}: {e}")
            response = {
                "order_quantity": 10,
                "confidence": 1.0,
                "rationale": "Default decision",
                "risk_assessment": "No risk",
                "expected_demand_next_round": 10
            }
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
    backlog_cost_per_unit: float = 1.0,
    temperature: float = 0.7,
    generation_index: int = 1,
    sim_data: SimulationData = None,
    human_log_file = None
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

        # 1. Retailer sees external demand
        retailer_demand = external_demand[r]

        # 2. Each role receives shipments from upstream that were placed 2 rounds ago
        shipments_received_list = []
        for agent in agents:
            received = agent.shipments_in_transit.get(0, 0)
            shipments_received_list.append(received)
            agent.inventory += received
            # shift transit queue: key0 <- key1, key1 <- key2, key2 reset to 0
            agent.shipments_in_transit[0] = agent.shipments_in_transit.get(1, 0)
            agent.shipments_in_transit[1] = agent.shipments_in_transit.get(2, 0)
            agent.shipments_in_transit[2] = 0

        # 3. Each role attempts to fill the *downstream* agent's order from inventory
        #    or the external demand in the case of the retailer.
        #    Then pass the unfilled portion to backlog if not enough inventory.

        # Retailer tries to fill external demand:
        amt_filled_retailer = min(retailer.inventory, retailer_demand)
        retailer.inventory -= amt_filled_retailer
        leftover_demand = retailer_demand - amt_filled_retailer
        retailer.backlog += leftover_demand

        # Record the retailer's "downstream order" as external demand for logging
        retailer.downstream_orders_history.append(retailer_demand)

        # Wholesaler tries to fill retailer's "order" => which is how much the retailer actually requested
        if wholesaler:
            # The "order" from retailer is (retailer_demand + retailer.backlog) 
            # but let's consider the portion not yet filled. For simplicity, we assume 
            # the retailer is effectively "requesting" retailer_demand each round.
            # Another approach is to incorporate backlog. This is flexible.
            retailer_order = retailer_demand  
            amt_filled_wh = min(wholesaler.inventory, retailer_order)
            wholesaler.inventory -= amt_filled_wh
            leftover_retailer_order = retailer_order - amt_filled_wh
            # The retailer's backlog is already accounted for above. 
            # But for the wholesaler, let's track how many the retailer "ordered."
            wholesaler.backlog += leftover_retailer_order
            # Log the wholesaler's perspective of retailer orders:
            wholesaler.downstream_orders_history.append(retailer_order)

        # Distributor tries to fill wholesaler's order
        if distributor and wholesaler:
            wh_order = wholesaler.downstream_orders_history[-1] if wholesaler.downstream_orders_history else 0
            amt_filled_dist = min(distributor.inventory, wh_order)
            distributor.inventory -= amt_filled_dist
            leftover_wh_order = wh_order - amt_filled_dist
            distributor.backlog += leftover_wh_order
            distributor.downstream_orders_history.append(wh_order)

        # Factory tries to fill distributor's order
        if factory and distributor:
            dist_order = distributor.downstream_orders_history[-1] if distributor.downstream_orders_history else 0
            amt_filled_fac = min(factory.inventory, dist_order)
            factory.inventory -= amt_filled_fac
            leftover_dist_order = dist_order - amt_filled_fac
            factory.backlog += leftover_dist_order
            factory.downstream_orders_history.append(dist_order)

        # 4. Each role pays holding + backlog cost
        for agent in agents:
            holding_cost = agent.inventory * holding_cost_per_unit
            backlog_cost = agent.backlog * backlog_cost_per_unit
            round_cost = holding_cost + backlog_cost
            agent.cost_accumulated += round_cost

        # 5. Each role decides on new order quantity from upstream
        order_decision_tasks = []
        for agent in agents:
            order_decision_tasks.append(agent.decide_order_quantity(temperature=temperature))
        decisions = await asyncio.gather(*order_decision_tasks)

        # 6. Place orders upstream => those orders become shipments_in_transit after 2 rounds and record order quantity
        orders_placed = []
        for agent, dec in zip(agents, decisions):
            if "order_quantity" not in dec:
                print(f"Warning: agent {agent.role_name} decision missing 'order_quantity', using default of 10.")
            order_qty = dec.get("order_quantity", 10)
            orders_placed.append(order_qty)
            agent.shipments_in_transit[2] = agent.shipments_in_transit.get(2, 0) + order_qty

        # Store round logs
        if sim_data:
            for idx, agent in enumerate(agents):
                sim_data.add_round_entry(RoundData(
                    generation = generation_index,
                    round_index = round_index,
                    role_name = agent.role_name,
                    inventory = agent.inventory,
                    backlog = agent.backlog,
                    order_placed = orders_placed[idx],
                    shipment_received = shipments_received_list[idx],
                    shipment_sent_downstream = 0,
                    cost = agent.cost_accumulated
                ))

        # Write human-readable log for the round
        if human_log_file:
            human_log_file.write("\n--------------------- Round {} ---------------------\n".format(round_index))
            human_log_file.write("External demand (Retailer): {}\n".format(retailer_demand))
            human_log_file.write("Shipments received per agent: {}\n".format(shipments_received_list))
            for idx, agent in enumerate(agents):
                human_log_file.write("Agent: {}: Inventory: {}, Backlog: {}, Order placed: {}, Total Cost: {}\n".format(
                    agent.role_name, agent.inventory, agent.backlog, orders_placed[idx], agent.cost_accumulated
                ))
            human_log_file.write("\n")

# --------------------------------------------------------------------
# 6. Putting It All Together: run multiple generations
# --------------------------------------------------------------------

async def run_beer_game_simulation(
    num_generations: int = 3,
    num_rounds_per_generation: int = 20,
    temperature: float = 0.7
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
    results_folder = f"beer_game_results_{current_time}"
    os.makedirs(results_folder, exist_ok=True)

    # Open human-readable log file
    human_log_path = os.path.join(results_folder, "human_readable_log.txt")
    human_log_file = open(human_log_path, "w")
    human_log_file.write("\n============================================================\n")
    human_log_file.write("            Starting Beer Game Simulation\n")
    human_log_file.write("============================================================\n\n")

    # Initialize the roles (4 default roles)
    roles = ["Retailer", "Wholesaler", "Distributor", "Factory"]
    agents = [BeerGameAgent(role_name=role) for role in roles]

    # Each agent obtains an initial strategy from the LLM
    await asyncio.gather(*(agent.initialize_strategy(temperature=temperature) for agent in agents))

    # Example external demand across rounds:
    # In real usage, you might load or generate random demands.
    external_demand_pattern = [random.randint(8,12) for _ in range(num_rounds_per_generation)]

    sim_data = SimulationData(hyperparameters={
        "num_generations": num_generations,
        "num_rounds_per_generation": num_rounds_per_generation,
        "holding_cost_per_unit": 0.5,
        "backlog_cost_per_unit": 1.0,
        "roles": roles,
        "timestamp": current_time
    })

    for gen_idx in range(num_generations):
        print(f"\n--- Starting Generation {gen_idx+1} ---")

        # Reset each agent's state if you want them to start "fresh" each generation 
        # (except for the strategy).
        for agent in agents:
            agent.inventory = 12
            agent.backlog = 0
            agent.cost_accumulated = 0.0
            # Also reset shipments in transit:
            agent.shipments_in_transit = {0:0,1:0,2:0}
            agent.downstream_orders_history = []

        human_log_file.write("\n============================================================\n")
        human_log_file.write("                  Generation {}\n".format(gen_idx+1))
        human_log_file.write("============================================================\n")

        await run_beer_game_generation(
            agents=agents,
            external_demand=external_demand_pattern,
            num_rounds=num_rounds_per_generation,
            holding_cost_per_unit=0.5,
            backlog_cost_per_unit=1.0,
            temperature=temperature,
            generation_index=gen_idx+1,
            sim_data=sim_data,
            human_log_file=human_log_file
        )

        # After the generation, collect performance logs for each agent,
        # then ask them to update their strategy if desired.
        for agent in agents:
            # Summarize performance
            performance_log = (
                f"Final Inventory: {agent.inventory}, "
                f"Final Backlog: {agent.backlog}, "
                f"Total Cost: {agent.cost_accumulated:.2f}"
            )
            await agent.update_strategy(performance_log, temperature=temperature)

    # Save logs to disk
    df_rounds = pd.DataFrame([asdict(r) for r in sim_data.rounds_log])
    df_rounds.to_csv(os.path.join(results_folder, "beer_game_detailed_log.csv"), index=False)
    df_rounds.to_json(os.path.join(results_folder, "beer_game_detailed_log.json"), orient="records", indent=2)

    # Simple visualizations (inventory/backlog over time, cost, etc.)
    plot_beer_game_results(df_rounds, results_folder)

    # Calculate Nash equilibrium deviations (assumed equilibrium order quantity = 10)
    deviations = calculate_nash_deviation(df_rounds, equilibrium_order=10)
    human_log_file.write("\n----- Nash Equilibrium Analysis -----\n")
    for role, dev in deviations.items():
        human_log_file.write("Role: {} - Average Absolute Deviation: {:.2f}\n".format(role, dev))

    human_log_file.close()
    print(f"\nSimulation complete. Results saved to: {results_folder}")
    return sim_data

# --------------------------------------------------------------------
# 7. Visualization
# --------------------------------------------------------------------

def plot_beer_game_results(rounds_df: pd.DataFrame, results_folder: str):
    """
    Basic plots: inventory over time, backlog over time, cost accumulation, etc.
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
        plt.plot(subset["round_index"], subset["cost"], label=role)
    plt.title("Accumulated Cost Over Time")
    plt.xlabel("Round")
    plt.ylabel("Accumulated Cost")
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
        axes[2].plot(subset["round_index"], subset["cost"], label=role)
    axes[2].set_title("Accumulated Cost Over Time")
    axes[2].set_xlabel("Round")
    axes[2].set_ylabel("Accumulated Cost")
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
        num_rounds_per_generation=20,
        temperature=0.7
    ))

if __name__ == "__main__":
    main()