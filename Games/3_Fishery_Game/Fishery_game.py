# fishery_game_extended.py
# -------------------------------------------------------------------------------------------
# Extended Fishery Game with Additional Scientific Metrics & Theoretical Benchmarks
#
# Copyright (c) 2025
# All rights reserved.
#
# Author: You (with suggestions from Ms. Hanassoni)
# -------------------------------------------------------------------------------------------

import os
import sys
import asyncio
import datetime
import random
import json
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Union
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import concurrent.futures
from functools import partial
from tqdm import tqdm
import subprocess
import math

# Hypothetical LLM usage
from openai import OpenAI, AsyncOpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)
async_client = AsyncOpenAI(api_key=api_key)

# -------------------------------------------------------------------------------------------
# 1. Configuration and Data Classes
# -------------------------------------------------------------------------------------------

@dataclass
class FisheryConfig:
    """
    A configuration object capturing all key parameters for the fishery game.
    This is useful for running systematic experiments or 'operational testing.'
    """
    num_agents: int = 5
    num_generations: int = 10
    initial_resource: float = 50.0
    growth_rate: float = 0.3
    carrying_capacity: float = 100.0
    max_fishable: float = 10.0
    temperature: float = 0.8
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"

@dataclass
class FisheryInteractionData:
    """
    Data about a single interaction (generation) in the Fishery Game simulation.
    """
    generation: int
    agent_name: str
    fish_caught: float
    resource_before: float
    resource_after: float
    payoff: float
    reasoning: str

@dataclass
class FisheryMetricsData:
    """
    Additional metrics for each generation:
      - eq_distance: a measure of how far from a chosen 'theoretical' extraction the group is
      - gini_payoffs: distribution of payoffs among agents in the current generation
      - average_fish_caught: to see the mean extraction among agents
    """
    generation: int
    eq_distance: float
    gini_payoffs: float
    average_fish_caught: float

@dataclass
class FisherySimulationData:
    """
    Container for all simulation data, including hyperparameters, interactions,
    resource evolution, and advanced metrics.
    """
    config: FisheryConfig
    interactions: List[FisheryInteractionData] = field(default_factory=list)
    resource_over_time: List[float] = field(default_factory=list)
    per_generation_metrics: List[FisheryMetricsData] = field(default_factory=list)

    def add_interaction(self, interaction: FisheryInteractionData):
        self.interactions.append(interaction)

    def add_generation_metrics(self, metrics: FisheryMetricsData):
        self.per_generation_metrics.append(metrics)

    def to_dict(self):
        return {
            'config': asdict(self.config),
            'interactions': [asdict(inter) for inter in self.interactions],
            'resource_over_time': self.resource_over_time,
            'per_generation_metrics': [asdict(m) for m in self.per_generation_metrics]
        }

# -------------------------------------------------------------------------------------------
# 2. Agent Class
# -------------------------------------------------------------------------------------------

class FisheryAgent(BaseModel):
    """
    Represents an agent in the Fishery Game that queries an LLM to decide how many fish to catch.
    """

    name: str
    model: str = Field(default="gpt-4o-mini")
    llm_provider: str = Field(default="openai")  
    llm_model: str = Field(default="gpt-4o-mini")

    total_payoff: float = 0.0
    fish_history: List[float] = Field(default_factory=list)
    reasoning_history: List[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    async def decide_fish_amount(
        self,
        resource_stock: float,
        temperature: float,
        generation_index: int,
        max_fishable: float,
        prompt_func
    ) -> dict:
        prompt = prompt_func(
            agent_name=self.name,
            generation=generation_index,
            resource_level=resource_stock,
            max_fishable=max_fishable
        )
        if self.llm_provider == "openai":
            decision = await self._decide_with_openai(prompt, temperature)
        elif self.llm_provider == "litellm":
            decision = self._decide_with_litellm(prompt, temperature)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

        fish_amount = decision.get("fish_amount", 0.0)
        fish_amount = max(0.0, min(float(fish_amount), max_fishable))
        rationale = decision.get("rationale", "No rationale provided.")

        return {"fish_amount": fish_amount, "rationale": rationale}

    async def _decide_with_openai(self, prompt: str, temperature: float) -> dict:
        # Increased retry and clearer error logs, as recommended
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = await async_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are playing a dynamic fishery game. Respond ONLY with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    max_tokens=300
                )
                json_str = response.choices[0].message.content.strip()
                decision = json.loads(json_str)
                if "fish_amount" not in decision:
                    decision["fish_amount"] = 0.0
                return decision
            except json.JSONDecodeError as e:
                print(f"[{self.name}] JSON decode error on attempt {attempt+1}: {e}")
            except Exception as e:
                print(f"[{self.name}] Unexpected error on attempt {attempt+1}: {e}")
        # Fallback
        return {"fish_amount": 0.0, "rationale": "Fallback after repeated JSON parse failures."}

    def _decide_with_litellm(self, prompt: str, temperature: float) -> dict:
        messages = [
            {"role": "system", "content": "You are playing a dynamic fishery game. Respond ONLY with valid JSON."},
            {"role": "user", "content": prompt}
        ]
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self._call_litellm(messages, model=self.llm_model, temperature=temperature)
                decision = json.loads(response.strip())
                if "fish_amount" not in decision:
                    decision["fish_amount"] = 0.0
                return decision
            except json.JSONDecodeError as e:
                print(f"[{self.name}] JSON decode error on attempt {attempt+1}: {e}")
            except Exception as e:
                print(f"[{self.name}] Unexpected error on attempt {attempt+1}: {e}")
        # Fallback
        return {"fish_amount": 0.0, "rationale": "Fallback after repeated JSON parse failures."}

    def _call_litellm(self, messages, model="gpt-4o-mini", temperature=0.8) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('LITELLM_API_KEY')}",
        }
        url = "https://litellm.sph-prod.ethz.ch/chat/completions"
        response = requests.post(url, json=payload, headers=headers)
        if not response.ok:
            raise Exception(f"LiteLLM API error: {response.status_code} {response.reason}")
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "No response from LiteLLM.")
        return content

    def log_decision(self, fish_amount: float, rationale: str):
        self.fish_history.append(fish_amount)
        self.reasoning_history.append(rationale)

    def add_payoff(self, payoff: float):
        self.total_payoff += payoff

# -------------------------------------------------------------------------------------------
# 3. Prompt Function for Fishery
# -------------------------------------------------------------------------------------------

class FisheryPrompts:
    @staticmethod
    def fishery_decision_prompt(
        agent_name: str,
        generation: int,
        resource_level: float,
        max_fishable: float
    ) -> str:
        return f"""
You are {agent_name} in a repeated Fishery Game.

- Generation: {generation}
- Current fish population: {resource_level:.2f}
- You may catch between 0 and {max_fishable:.2f} fish (not exceeding resource).

Your payoff = # of fish you catch.
Overfishing may deplete future stocks.

Respond ONLY in valid JSON:
{{
    "fish_amount": <float between 0 and {max_fishable}>,
    "rationale": "explain briefly"
}}
"""

# -------------------------------------------------------------------------------------------
# 4. Theoretical Equilibria and Metrics
# -------------------------------------------------------------------------------------------

def compute_theoretical_equilibria(
    num_agents: int,
    growth_rate: float,
    carrying_capacity: float,
    max_fishable: float
) -> Tuple[float, float]:
    """
    Attempt to provide two rough reference points:
      - 'Tragedy' catch: If each agent acts non-cooperatively to maximize immediate payoff, 
        they might collectively request an unsustainable total near num_agents * max_fishable.
      - 'Social Optimum': A lower, stable extraction level per agent that (hypothetically) 
        maintains resource at equilibrium for indefinite time.

    In practice, deriving a closed-form multi-agent Nash equilibrium for logistic resource 
    growth is complex. This function just returns rough guidelines for demonstration.

    Returns:
        (noncoop_per_agent, social_opt_per_agent)
    """
    # Very rough approximation: 
    # *Tragedy approach:* Each tries to catch max_fishable
    # *Social approach:* If stable, we want resource to remain near K/2 for max logistic growth.
    #   Then each agent might limit catch to growth_rate*(K/2)*(1 - (K/2)/K) / num_agents 
    #   = (growth_rate * K/2 * 1/2) / num_agents = growth_rate * K / (4 * num_agents)

    tragedy_per_agent = max_fishable  
    # Social optimum attempt:
    social_opt_per_agent = (growth_rate * carrying_capacity) / (4.0 * num_agents)
    return (tragedy_per_agent, social_opt_per_agent)

def gini_coefficient(values: List[float]) -> float:
    """
    Calculate the Gini coefficient of a list of values.
    Gini = (1 / (2n^2)) * sum_i sum_j |x_i - x_j| / mean
    Lower = more equal, higher = more unequal.
    """
    if len(values) == 0:
        return 0.0
    mean_val = np.mean(values)
    if mean_val == 0:
        return 0.0
    abs_diffs_sum = 0
    n = len(values)
    for i in range(n):
        for j in range(n):
            abs_diffs_sum += abs(values[i] - values[j])
    return abs_diffs_sum / (2 * n * n * mean_val)

def measure_equilibrium_distance(
    actual_catches: List[float],
    reference_catch_per_agent: float
) -> float:
    """
    Given the actual fish_caught by each agent in a generation, compute how far 
    the group is from the reference strategy (like 'tragedy' or 'social optimum').
    For simplicity, we measure the L2-norm difference from the reference.

    eq_distance = sqrt( sum( (actual_i - reference_catch)^2 ) / n )
    """
    if len(actual_catches) == 0:
        return 0.0
    n = len(actual_catches)
    diffs_sq = [(x - reference_catch_per_agent)**2 for x in actual_catches]
    mean_diffs_sq = sum(diffs_sq) / n
    return math.sqrt(mean_diffs_sq)

# -------------------------------------------------------------------------------------------
# 5. Fishery Dynamics
# -------------------------------------------------------------------------------------------

def logistic_growth(current_stock: float, growth_rate: float, carrying_capacity: float) -> float:
    growth = growth_rate * current_stock * (1 - (current_stock / carrying_capacity))
    new_stock = current_stock + growth
    return max(0.0, new_stock)

async def simulate_fishery_generation(
    agents: List[FisheryAgent],
    resource_stock: float,
    config: FisheryConfig,
    generation_index: int,
    sim_data: FisherySimulationData,
    prompt_func=None
):
    if prompt_func is None:
        prompt_func = FisheryPrompts.fishery_decision_prompt

    # Asynchronous decisions
    tasks = []
    for agent in agents:
        t = agent.decide_fish_amount(
            resource_stock=resource_stock,
            temperature=config.temperature,
            generation_index=generation_index,
            max_fishable=config.max_fishable,
            prompt_func=prompt_func
        )
        tasks.append(t)

    decisions_results = await asyncio.gather(*tasks)
    fish_demands = [res["fish_amount"] for res in decisions_results]
    total_demand = sum(fish_demands)

    # Distribute fish if total_demand > resource_stock
    if total_demand <= resource_stock:
        catches = fish_demands
        resource_after_catch = resource_stock - total_demand
    else:
        ratio = resource_stock / total_demand if total_demand > 0 else 0
        catches = [d * ratio for d in fish_demands]
        resource_after_catch = 0.0

    # Log interactions
    for i, agent in enumerate(agents):
        fish_caught = catches[i]
        payoff = fish_caught
        reasoning = decisions_results[i]["rationale"]
        interaction_record = FisheryInteractionData(
            generation=generation_index,
            agent_name=agent.name,
            fish_caught=fish_caught,
            resource_before=resource_stock,
            resource_after=resource_after_catch,
            payoff=payoff,
            reasoning=reasoning
        )
        sim_data.add_interaction(interaction_record)
        agent.log_decision(fish_caught, reasoning)
        agent.add_payoff(payoff)

    # Resource regrowth
    new_resource_stock = logistic_growth(resource_after_catch, config.growth_rate, config.carrying_capacity)
    return new_resource_stock

# -------------------------------------------------------------------------------------------
# 6. Main Simulation & Metrics
# -------------------------------------------------------------------------------------------

async def run_fishery_game_simulation(
    config: FisheryConfig
) -> Tuple[FisherySimulationData, List[FisheryAgent]]:
    """
    Main function to run the fishery simulation with additional metrics and
    theoretical reference points for equilibrium comparisons.
    """
    start_time = datetime.datetime.now()
    timestamp_str = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    base_path = os.path.dirname(os.path.abspath(__file__))
    results_folder = os.path.join(base_path, "fishery_simulation_results")
    os.makedirs(results_folder, exist_ok=True)
    run_folder = os.path.join(results_folder, f"run_{timestamp_str}")
    os.makedirs(run_folder, exist_ok=True)

    sim_data = FisherySimulationData(config=config)

    # Initialize Agents
    agents = []
    for i in range(config.num_agents):
        agent = FisheryAgent(
            name=f"FishAgent_{i+1}",
            model=config.llm_model,
            llm_provider=config.llm_provider,
            llm_model=config.llm_model
        )
        agents.append(agent)

    current_resource = config.initial_resource

    # Theoretical reference points for entire simulation
    tragedy_per_agent, social_opt_per_agent = compute_theoretical_equilibria(
        config.num_agents,
        config.growth_rate,
        config.carrying_capacity,
        config.max_fishable
    )

    # Main simulation loop
    for g in range(1, config.num_generations + 1):
        sim_data.resource_over_time.append(current_resource)
        current_resource = await simulate_fishery_generation(
            agents=agents,
            resource_stock=current_resource,
            config=config,
            generation_index=g,
            sim_data=sim_data
        )

        # Compute additional metrics for generation g
        generation_interactions = [i for i in sim_data.interactions if i.generation == g]
        actual_catches = [inter.fish_caught for inter in generation_interactions]
        payoffs = [inter.payoff for inter in generation_interactions]

        # Compare average actual to tragedy & social optimum
        # We'll pick the 'social optimum' as the main reference
        eq_distance = measure_equilibrium_distance(actual_catches, social_opt_per_agent)
        gini_payoffs = gini_coefficient(payoffs)
        avg_fish_caught = np.mean(actual_catches) if actual_catches else 0.0

        metrics_data = FisheryMetricsData(
            generation=g,
            eq_distance=eq_distance,
            gini_payoffs=gini_payoffs,
            average_fish_caught=avg_fish_caught
        )
        sim_data.add_generation_metrics(metrics_data)

    # Final resource log
    sim_data.resource_over_time.append(current_resource)

    # Save data
    interactions_df = pd.DataFrame([asdict(i) for i in sim_data.interactions])
    interactions_df.to_csv(os.path.join(run_folder, "fishery_detailed_logs.csv"), index=False)
    interactions_df.to_json(os.path.join(run_folder, "fishery_detailed_logs.json"), orient="records", indent=4)

    metrics_df = pd.DataFrame([asdict(m) for m in sim_data.per_generation_metrics])
    metrics_df.to_csv(os.path.join(run_folder, "fishery_metrics.csv"), index=False)
    metrics_df.to_json(os.path.join(run_folder, "fishery_metrics.json"), orient="records", indent=4)

    with open(os.path.join(run_folder, "parameters.json"), "w") as f:
        json.dump(asdict(config), f, indent=4)

    # Generate plots
    create_fishery_plots(sim_data, agents, run_folder)
    format_logs_with_prettier(run_folder)

    return sim_data, agents

# -------------------------------------------------------------------------------------------
# 7. Enhanced Plotting & Comparison
# -------------------------------------------------------------------------------------------

def create_fishery_plots(
    sim_data: FisherySimulationData,
    agents: List[FisheryAgent],
    run_folder: str
):
    generations_range = range(len(sim_data.resource_over_time))

    # Plot resource over time
    plt.figure(figsize=(10, 6))
    plt.plot(generations_range, sim_data.resource_over_time, '-o', color='blue')
    plt.title("Fishery Resource Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Resource Stock")
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(run_folder, "resource_over_time.png"))
    plt.close()

    # Plot eq_distance over time
    df_metrics = pd.DataFrame([asdict(m) for m in sim_data.per_generation_metrics])
    plt.figure(figsize=(10, 6))
    plt.plot(df_metrics["generation"], df_metrics["eq_distance"], '-s', color='green')
    plt.title("Distance from Social Optimum over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Equilibrium Distance (lower = closer)")
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(run_folder, "eq_distance_over_time.png"))
    plt.close()

    # Plot Gini coefficient over time
    plt.figure(figsize=(10, 6))
    plt.plot(df_metrics["generation"], df_metrics["gini_payoffs"], '-^', color='red')
    plt.title("Gini Coefficient of Payoffs (Equity) per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Gini Coefficient")
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(run_folder, "gini_payoffs_over_time.png"))
    plt.close()

    # Comprehensive subplot
    plt.figure(figsize=(14, 8))
    plt.suptitle("Fishery Game: Resource & Metrics", fontsize=16)

    plt.subplot(2, 2, 1)
    plt.plot(generations_range, sim_data.resource_over_time, '-o', color='blue')
    plt.title("Resource Over Time")
    plt.xlabel("Gen")
    plt.ylabel("Stock")
    plt.grid(alpha=0.5)

    plt.subplot(2, 2, 2)
    plt.plot(df_metrics["generation"], df_metrics["eq_distance"], '-s', color='green')
    plt.title("Distance from Social Optimum")
    plt.xlabel("Gen")
    plt.ylabel("Distance")
    plt.grid(alpha=0.5)

    plt.subplot(2, 2, 3)
    plt.plot(df_metrics["generation"], df_metrics["gini_payoffs"], '-^', color='red')
    plt.title("Gini Payoffs")
    plt.xlabel("Gen")
    plt.ylabel("Gini")
    plt.grid(alpha=0.5)

    plt.subplot(2, 2, 4)
    plt.plot(df_metrics["generation"], df_metrics["average_fish_caught"], '-o', color='orange')
    plt.title("Avg Fish Caught per Agent")
    plt.xlabel("Gen")
    plt.ylabel("Avg Fish")
    plt.grid(alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(run_folder, "comprehensive_metrics.png"))
    plt.close()

def format_logs_with_prettier(run_folder: str):
    for file_name in os.listdir(run_folder):
        if file_name.endswith('.json'):
            file_path = os.path.join(run_folder, file_name)
            try:
                subprocess.run(['prettier', '--write', file_path], check=True)
                print(f"Formatted {file_name} with Prettier.")
            except FileNotFoundError:
                print("Prettier not installed, skipping JSON formatting.")
                break
        elif file_name.endswith('.csv'):
            print(f"Skipping CSV file {file_name} for Prettier formatting.")

# -------------------------------------------------------------------------------------------
# 8. Comparison to Human Data (Placeholder)
# -------------------------------------------------------------------------------------------

def compare_with_human_data(sim_data: FisherySimulationData, human_data_path: str):
    """
    Illustrative placeholder function to show how you might compare LLM-driven agent 
    outcomes with existing human experimental data on a fishery or common-pool resource game.
    """
    # Suppose human_data.csv has columns: generation, avg_catch, ...
    if not os.path.exists(human_data_path):
        print("Human data file not found. Skipping comparison.")
        return

    human_df = pd.read_csv(human_data_path)
    # For demonstration, compare average fish caught
    sim_metrics_df = pd.DataFrame([asdict(m) for m in sim_data.per_generation_metrics])
    merged_df = pd.merge(
        sim_metrics_df, 
        human_df, 
        on="generation", 
        suffixes=("_sim", "_human"), 
        how="inner"
    )

    # Example: compute difference in average fish caught
    merged_df["catch_diff"] = merged_df["average_fish_caught"] - merged_df["avg_catch"]
    print(merged_df[["generation", "average_fish_caught", "avg_catch", "catch_diff"]])

# -------------------------------------------------------------------------------------------
# 9. CLI Entry Point
# -------------------------------------------------------------------------------------------

def main():
    print("Running Extended Fishery Game with Additional Metrics & Theoretical References...\n")

    # Example: read from environment or pass arguments
    config = FisheryConfig(
        num_agents=5,
        num_generations=10,
        initial_resource=50.0,
        growth_rate=0.3,
        carrying_capacity=100.0,
        max_fishable=10.0,
        temperature=0.8,
        llm_provider="openai",
        llm_model="gpt-4o-mini"
    )

    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass

    loop = asyncio.get_event_loop()
    sim_data, agents = loop.run_until_complete(run_fishery_game_simulation(config))

    # Example of how we might do a quick comparison to hypothetical human data
    # compare_with_human_data(sim_data, "path_to_human_data.csv")

    print("\n=== Simulation Complete ===")
    print(f"Final Resource Level: {sim_data.resource_over_time[-1]:.2f}")
    for agent in agents:
        print(f"{agent.name} final payoff: {agent.total_payoff:.2f}")

    print("Extended logs and plots (including advanced metrics) saved.")
    print("End of simulation.")

if __name__ == "__main__":
    main()