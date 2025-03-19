#!/usr/bin/env python3

# %% [markdown]
# # Optimized LLM-driven Decision Making for Iterated Prisoner's Dilemma
# 
# This script implements an LLM-driven simulation of the Iterated Prisoner's Dilemma using OpenAI's API. 
# 
# The simulation features:
# - Dynamic strategy generation using GPT-4
# - Evolutionary agent selection
# - Asynchronous execution for improved performance
# - Detailed logging and visualization

# %% [markdown]
# ## Setup and Imports

#%% 
from openai import OpenAI, AsyncOpenAI
import os
import random
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from dotenv import load_dotenv
import asyncio
from typing import List, Tuple, Dict
import aiohttp
import sys
from dataclasses import dataclass, field, asdict
import numpy as np

# Load environment variables and setup OpenAI client
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
async_client = AsyncOpenAI(api_key=api_key)

# =============================================================================
# Comprehensive List of Iterated Prisoner's Dilemma (IPD) Strategies
# (Arranged from Most Social/Collaborative to Most Antisocial)
# =============================================================================
# Each tuple contains (strategy_key, description)
PD_STRATEGIES_SORTED = [
    ("generous_tit_for_tat", "Generous Tit-for-Tat: Highly cooperative and forgiving, it promotes long-term mutual cooperation by occasionally overlooking defections.", 0.85),
    ("tit_for_tat", "Tit-for-Tat: Begins with cooperation and reciprocates the opponent's previous move, fostering reciprocal behavior.", 0.70),
    ("win_stay_lose_shift", "Win-Stay, Lose-Shift (Pavlov): Repeats a move if it yielded a favorable payoff, adapting quickly to both cooperation and defection.", 0.65),
    ("contrite_tit_for_tat", "Contrite Tit-for-Tat: Similar to Tit-for-Tat but offers forgiveness for accidental defections, thereby restoring cooperation.", 0.75),
    ("always_cooperate", "Always Cooperate: Consistently cooperates regardless of the opponent's actions—very social but vulnerable to exploitation.", 1.0),
    ("grim_trigger", "Grim Trigger: Cooperates until the first defection, then defects forever, enforcing strict punishment against betrayal.", 0.40),
    ("suspicious_tit_for_tat", "Suspicious Tit-for-Tat: Starts with defection to test the opponent before potentially cooperating, less immediately cooperative.", 0.35),
    ("always_defect", "Always Defect: Consistently defects, maximizing short-term gain at the expense of long-term cooperation.", 0.0),
    ("random", "Random: Chooses actions unpredictably, lacking a consistent social or antisocial pattern.", 0.50)
]

# =============================================================================
# Data Classes for Structured Logging
# =============================================================================
@dataclass
class InteractionData:
    generation: int
    pair: str
    round_actions: str
    payoffs: str
    reasoning_A: str
    reasoning_B: str
    score_A: int
    score_B: int

@dataclass
class SimulationData:
    hyperparameters: dict
    interactions: List[InteractionData] = field(default_factory=list)
    equilibrium_metrics: dict = field(default_factory=dict)

    def add_interaction(self, interaction: InteractionData):
        self.interactions.append(interaction)

    def add_equilibrium_data(self, generation, nash_deviation, best_response_diff):
        self.equilibrium_metrics[generation] = {
            'nash_deviation': nash_deviation,
            'best_response_diff': best_response_diff
        }

    def to_dict(self):
        return {
            'hyperparameters': self.hyperparameters,
            'interactions': [asdict(inter) for inter in self.interactions],
            'equilibrium_metrics': self.equilibrium_metrics
        }

# =============================================================================
# Enhanced Visualization Functions
# =============================================================================
def create_comprehensive_plots(generation_summary, run_folder, strategy_distribution):
    """Create a comprehensive set of plots showing all important metrics."""
    
    # Set up the plotting style
    plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # 1. Main Metrics Over Time (4x4 grid)
    fig1 = plt.figure(figsize=(20, 20))
    
    # Plot 1: Average Score
    plt.subplot(4, 4, 1)
    plt.plot([m["Average_Score"] for m in generation_summary], marker='o')
    plt.title("Average Score Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.grid(True)
    
    # Plot 2: Strategy Diversity
    plt.subplot(4, 4, 2)
    plt.plot([m["Strategy_Diversity"] for m in generation_summary], marker='o', color='green')
    plt.title("Strategy Diversity")
    plt.xlabel("Generation")
    plt.ylabel("Number of Unique Strategies")
    plt.grid(True)
    
    # Plot 3: Pareto Efficiency
    plt.subplot(4, 4, 3)
    plt.plot([m["Pareto_Efficiency"] for m in generation_summary], marker='o', color='red')
    plt.title("Pareto Efficiency")
    plt.xlabel("Generation")
    plt.ylabel("Efficiency")
    plt.grid(True)
    
    # Plot 4: Nash Deviation
    plt.subplot(4, 4, 4)
    plt.plot([m["Nash_Deviation"] for m in generation_summary], marker='o', color='purple')
    plt.title("Nash Equilibrium Deviation")
    plt.xlabel("Generation")
    plt.ylabel("Deviation")
    plt.grid(True)
    
    # Plot 5: Mutual Cooperation
    plt.subplot(4, 4, 5)
    plt.plot([m["mutual_cooperation"] for m in generation_summary], marker='o', color='blue')
    plt.title("Mutual Cooperation Instances")
    plt.xlabel("Generation")
    plt.ylabel("Count")
    plt.grid(True)
    
    # Plot 6: Mutual Defection
    plt.subplot(4, 4, 6)
    plt.plot([m["mutual_defection"] for m in generation_summary], marker='o', color='red')
    plt.title("Mutual Defection Instances")
    plt.xlabel("Generation")
    plt.ylabel("Count")
    plt.grid(True)
    
    # Plot 7: Temptation Payoffs
    plt.subplot(4, 4, 7)
    plt.plot([m["temptation_payoffs"] for m in generation_summary], marker='o', color='orange')
    plt.title("Temptation Payoff Instances")
    plt.xlabel("Generation")
    plt.ylabel("Count")
    plt.grid(True)
    
    # Plot 8: Sucker Payoffs
    plt.subplot(4, 4, 8)
    plt.plot([m["sucker_payoffs"] for m in generation_summary], marker='o', color='brown')
    plt.title("Sucker Payoff Instances")
    plt.xlabel("Generation")
    plt.ylabel("Count")
    plt.grid(True)
    
    # Plot 9: Total Payoffs
    plt.subplot(4, 4, 9)
    plt.plot([m["total_payoffs"] for m in generation_summary], marker='o', color='green')
    plt.title("Total Payoffs")
    plt.xlabel("Generation")
    plt.ylabel("Sum")
    plt.grid(True)
    
    # Plot 10: Cooperation vs Defection Ratio
    coop_def_ratio = []
    for m in generation_summary:
        coop = m["mutual_cooperation"] + m["sucker_payoffs"]
        def_ = m["mutual_defection"] + m["temptation_payoffs"]
        ratio = coop / (def_ + 1e-10)  # Avoid division by zero
        coop_def_ratio.append(ratio)
    
    plt.subplot(4, 4, 10)
    plt.plot(coop_def_ratio, marker='o', color='purple')
    plt.title("Cooperation to Defection Ratio")
    plt.xlabel("Generation")
    plt.ylabel("Ratio")
    plt.grid(True)
    
    # Plot 11: Strategy Success Rate
    if "strategy_distribution" in generation_summary[0]:
        plt.subplot(4, 4, 11)
        strategies = list(generation_summary[0]["strategy_distribution"].keys())
        for strategy in strategies:
            counts = [gen["strategy_distribution"][strategy] for gen in generation_summary]
            plt.plot(counts, label=strategy, marker='o', alpha=0.7)
        plt.title("Strategy Success Over Time")
        plt.xlabel("Generation")
        plt.ylabel("Count")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_folder, "comprehensive_metrics.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Strategy Distribution Heatmap
    if "strategy_distribution" in generation_summary[0]:
        plt.figure(figsize=(15, 8))
        strategies = list(generation_summary[0]["strategy_distribution"].keys())
        generations = range(1, len(generation_summary) + 1)
        data = np.array([[gen["strategy_distribution"][strat] for strat in strategies] for gen in generation_summary])
        
        sns.heatmap(data.T, xticklabels=generations, yticklabels=strategies, 
                    cmap='YlOrRd', annot=True, fmt='d')
        plt.title("Strategy Distribution Heatmap")
        plt.xlabel("Generation")
        plt.ylabel("Strategy")
        plt.tight_layout()
        plt.savefig(os.path.join(run_folder, "strategy_heatmap.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Final Generation Analysis
    plt.figure(figsize=(15, 5))
    
    # Final Generation Outcome Distribution
    plt.subplot(1, 3, 1)
    outcomes = ['Mutual C', 'Mutual D', 'Temptation', 'Sucker']
    values = [generation_summary[-1]["mutual_cooperation"],
              generation_summary[-1]["mutual_defection"],
              generation_summary[-1]["temptation_payoffs"],
              generation_summary[-1]["sucker_payoffs"]]
    plt.bar(outcomes, values, color=['green', 'red', 'orange', 'blue'])
    plt.title("Final Generation Outcome Distribution")
    plt.xticks(rotation=45)
    
    # Final Strategy Distribution
    if "strategy_distribution" in generation_summary[-1]:
        plt.subplot(1, 3, 2)
        strategies = list(generation_summary[-1]["strategy_distribution"].keys())
        counts = [generation_summary[-1]["strategy_distribution"][s] for s in strategies]
        plt.bar(strategies, counts)
        plt.title("Final Generation Strategy Distribution")
        plt.xticks(rotation=45)
    
    # Final Performance Metrics
    plt.subplot(1, 3, 3)
    metrics = ['Avg Score', 'Pareto Eff', 'Nash Dev']
    values = [generation_summary[-1]["Average_Score"],
              generation_summary[-1]["Pareto_Efficiency"],
              generation_summary[-1]["Nash_Deviation"]]
    plt.bar(metrics, values, color=['blue', 'green', 'red'])
    plt.title("Final Generation Performance Metrics")
    plt.ylim(0, 1.2)  # Assuming metrics are normalized between 0 and 1
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_folder, "final_generation_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Print summary statistics
    print("\nSimulation Summary Statistics:")
    print("==============================")
    print(f"Number of Generations: {len(generation_summary)}")
    print(f"Final Average Score: {generation_summary[-1]['Average_Score']:.2f}")
    print(f"Final Pareto Efficiency: {generation_summary[-1]['Pareto_Efficiency']:.2f}")
    print(f"Final Nash Deviation: {generation_summary[-1]['Nash_Deviation']:.2f}")
    print(f"Final Strategy Diversity: {generation_summary[-1]['Strategy_Diversity']}")
    print("\nOutcome Distribution in Final Generation:")
    print(f"Mutual Cooperation: {generation_summary[-1]['mutual_cooperation']}")
    print(f"Mutual Defection: {generation_summary[-1]['mutual_defection']}")
    print(f"Temptation Payoffs: {generation_summary[-1]['temptation_payoffs']}")
    print(f"Sucker Payoffs: {generation_summary[-1]['sucker_payoffs']}")
    print(f"Total Payoffs: {generation_summary[-1]['total_payoffs']}")

# %% [markdown]
# ## Agent Implementation
# 
# The EnhancedAgent class represents a player in the Prisoner's Dilemma game.
# 
# Each agent:
# 
# - Has a unique strategy matrix generated by GPT-4
# 
# - Maintains a history of interactions
# 
# - Makes decisions based on past interactions and current game state

# %%
#%% 
class EnhancedAgent:
    def __init__(self, name, model="gpt-4-turbo", 
                 strategy_tactic="tit_for_tat", 
                 cooperation_bias=0.5,      # Bias toward cooperation (0 to 1)
                 risk_aversion=0.5,         # Tendency to avoid risky moves (0 to 1)
                 game_theoretic_prior=None  # Additional prior parameters as a dict
                ):
        self.name = name
        self.model = model  # Track model architecture
        self.strategy_tactic = strategy_tactic  # Must be one of the keys in PD_STRATEGIES
        self.cooperation_bias = cooperation_bias
        self.risk_aversion = risk_aversion
        self.game_theoretic_prior = game_theoretic_prior if game_theoretic_prior is not None else {}
        
        self.total_score = 0
        self.history = []  # Each entry: (opponent_name, own_action, opp_action, payoff)
        self.strategy_matrix = None
        self.strategy_evolution = []  # Track strategy changes over generations
        self.cooperation_rate = 0.0
        self.reciprocity_index = 0.0  # Measure tit-for-tat behavior

    async def initialize(self):
        """Asynchronously initialize the agent's strategy matrix."""
        self.strategy_matrix = await self.generate_strategy_matrix()
        return self

    async def generate_strategy_matrix(self):
        prompt = """System: You are developing a novel strategy for the Iterated Prisoner's Dilemma. 
Create a unique approach that considers:
- Long-term relationship building
- Error correction mechanisms
- Adaptive response patterns
- Potential for both cooperation and defection

Format: JSON structure with:
{
    "strategy_rules": [list of conditional statements],
    "forgiveness_factor": 0-1,
    "retaliation_threshold": 0-1,
    "adaptability": 0-1,
    "rationale": "str"
}"""
        
        for _ in range(3):  # Retry up to 3 times
            try:
                response = await async_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": "You are a game theory expert creating novel IPD strategies. Respond ONLY with valid JSON."},
                              {"role": "user", "content": prompt}],
                    temperature=0.8,
                    response_format={"type": "json_object"},
                    max_tokens=300
                )
                json_str = response.choices[0].message.content.strip()
                if not json_str.startswith('{') or not json_str.endswith('}'):
                    raise json.JSONDecodeError("Missing braces", json_str, 0)
                strategy = json.loads(json_str)
                if all(k in strategy for k in ["strategy_rules", "forgiveness_factor", "retaliation_threshold", "adaptability"]):
                    return strategy
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Retrying strategy generation due to error: {str(e)}")
                continue
        
        return {
            "strategy_rules": ["CC: C", "CD: D", "DC: D", "DD: C"],
            "forgiveness_factor": 0.5,
            "retaliation_threshold": 0.5,
            "adaptability": 0.5,
            "rationale": "Default fallback strategy"
        }
    
    def decide_action_explicit(self, opponent) -> Dict:
        """
        Implements an explicit decision based on the chosen strategy tactic.
        Returns a dict with keys: action, confidence, rationale, expected_opponent_action, risk_assessment.
        """
        if self.strategy_tactic == "always_cooperate":
            return {"action": "C", "confidence": 1.0, "rationale": "Always Cooperate strategy", "expected_opponent_action": "C", "risk_assessment": "Low"}
        elif self.strategy_tactic == "always_defect":
            return {"action": "D", "confidence": 1.0, "rationale": "Always Defect strategy", "expected_opponent_action": "D", "risk_assessment": "Low"}
        elif self.strategy_tactic == "tit_for_tat":
            if self.history:
                last_opponent_action = self.history[-1][2]
                return {"action": last_opponent_action, "confidence": 1.0, "rationale": "Tit-for-Tat: mirroring opponent's last move", "expected_opponent_action": last_opponent_action, "risk_assessment": "Medium"}
            else:
                return {"action": "C", "confidence": 1.0, "rationale": "Tit-for-Tat: starting with cooperation", "expected_opponent_action": "C", "risk_assessment": "Low"}
        elif self.strategy_tactic == "grim_trigger":
            if any(interaction[2] == "D" for interaction in self.history):
                return {"action": "D", "confidence": 1.0, "rationale": "Grim Trigger: defecting after observed defection", "expected_opponent_action": "D", "risk_assessment": "High"}
            else:
                return {"action": "C", "confidence": 1.0, "rationale": "Grim Trigger: continuing cooperation", "expected_opponent_action": "C", "risk_assessment": "Low"}
        elif self.strategy_tactic == "win_stay_lose_shift":
            if self.history:
                last_payoff = self.history[-1][3]
                last_move = self.history[-1][1]
                if last_payoff >= 3:
                    return {"action": last_move, "confidence": 1.0, "rationale": "Win-Stay, Lose-Shift: repeating successful move", "expected_opponent_action": "C", "risk_assessment": "Low"}
                else:
                    new_move = "D" if last_move == "C" else "C"
                    return {"action": new_move, "confidence": 1.0, "rationale": "Win-Stay, Lose-Shift: switching due to low payoff", "expected_opponent_action": "C", "risk_assessment": "Medium"}
            else:
                return {"action": "C", "confidence": 1.0, "rationale": "Win-Stay, Lose-Shift: default cooperation", "expected_opponent_action": "C", "risk_assessment": "Low"}
        elif self.strategy_tactic == "generous_tit_for_tat":
            if self.history:
                last_opponent_action = self.history[-1][2]
                if last_opponent_action == "D" and random.random() < self.cooperation_bias:
                    return {"action": "C", "confidence": 1.0, "rationale": "Generous Tit-for-Tat: forgiving defection", "expected_opponent_action": "C", "risk_assessment": "Medium"}
                else:
                    return {"action": last_opponent_action, "confidence": 1.0, "rationale": "Generous Tit-for-Tat: mirroring last move", "expected_opponent_action": last_opponent_action, "risk_assessment": "Medium"}
            else:
                return {"action": "C", "confidence": 1.0, "rationale": "Generous Tit-for-Tat: default cooperation", "expected_opponent_action": "C", "risk_assessment": "Low"}
        elif self.strategy_tactic == "suspicious_tit_for_tat":
            if self.history:
                last_opponent_action = self.history[-1][2]
                return {"action": last_opponent_action, "confidence": 1.0, "rationale": "Suspicious Tit-for-Tat: mirroring opponent's move", "expected_opponent_action": last_opponent_action, "risk_assessment": "High"}
            else:
                return {"action": "D", "confidence": 1.0, "rationale": "Suspicious Tit-for-Tat: starting with defection", "expected_opponent_action": "D", "risk_assessment": "High"}
        elif self.strategy_tactic == "contrite_tit_for_tat":
            if self.history:
                last_self_move = self.history[-1][1]
                last_opponent_move = self.history[-1][2]
                if last_opponent_move == "D" and last_self_move == "D":
                    return {"action": "C", "confidence": 1.0, "rationale": "Contrite Tit-for-Tat: apologizing for unintended defection", "expected_opponent_action": "C", "risk_assessment": "Medium"}
                else:
                    return {"action": last_opponent_move, "confidence": 1.0, "rationale": "Contrite Tit-for-Tat: mirroring opponent's last move", "expected_opponent_action": last_opponent_move, "risk_assessment": "Medium"}
            else:
                return {"action": "C", "confidence": 1.0, "rationale": "Contrite Tit-for-Tat: default cooperation", "expected_opponent_action": "C", "risk_assessment": "Low"}
        elif self.strategy_tactic == "always_defect":
            return {"action": "D", "confidence": 1.0, "rationale": "Always Defect strategy", "expected_opponent_action": "D", "risk_assessment": "Low"}
        elif self.strategy_tactic == "always_cooperate":
            return {"action": "C", "confidence": 1.0, "rationale": "Always Cooperate strategy", "expected_opponent_action": "C", "risk_assessment": "Low"}
        elif self.strategy_tactic == "random":
            action = random.choice(["C", "D"])
            return {"action": action, "confidence": 1.0, "rationale": "Random strategy: unpredictable decision", "expected_opponent_action": "C", "risk_assessment": "Variable"}
        else:
            return None

    async def decide_action(self, opponent):
        """
        Determine an action using an explicit tactic if available.
        Otherwise, fallback to the LLM-based decision approach.
        Only the last three rounds are provided as context (partial visibility).
        """
        explicit_decision = self.decide_action_explicit(opponent)
        if explicit_decision is not None:
            return explicit_decision

        analysis_prompt = f"""Analyze this Prisoner's Dilemma interaction history with {opponent.name}:
Previous Rounds (last 3): {str(self.history[-3:]) if len(self.history) > 0 else 'None'}

Your Strategy: {json.dumps(self.strategy_matrix)}
Opponent's Model: {opponent.model}
Opponent's Cooperation Rate: {opponent.cooperation_rate:.2f}

Output MUST be valid JSON with:
{{
    "action": "C/D",
    "confidence": 0-1,
    "rationale": "str",
    "expected_opponent_action": "C/D",
    "risk_assessment": "str"
}}"""
        
        for _ in range(3):
            try:
                response = await async_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": "You are an AI game theorist. Respond ONLY with valid JSON."},
                              {"role": "user", "content": analysis_prompt}],
                    temperature=0.4,
                    response_format={"type": "json_object"},
                    max_tokens=150
                )
                json_str = response.choices[0].message.content.strip()
                decision = json.loads(json_str)
                action = decision.get("action", "C").upper()
                if action not in ["C", "D"]:
                    action = random.choice(["C", "D"])
                return decision
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Retrying decision due to error: {str(e)}")
                continue
        
        return {
            "action": random.choice(["C", "D"]),
            "confidence": 0.5,
            "rationale": "Fallback decision",
            "expected_opponent_action": "C",
            "risk_assessment": "Unknown"
        }

    def log_interaction(self, opponent, own_action, opp_action, payoff):
        self.history.append((opponent, own_action, opp_action, payoff))

# %% [markdown]
# ## Game Configuration
# 
# Define the payoff matrix for the Prisoner's Dilemma and helper functions for agent creation and interaction.

# %%
#%% 
# Define the standard Prisoner's Dilemma payoff matrix.
payoff_matrix = {
    ('C', 'C'): (3, 3),  # Both cooperate
    ('C', 'D'): (0, 5),  # Player 1 cooperates, Player 2 defects
    ('D', 'C'): (5, 0),  # Player 1 defects, Player 2 cooperates
    ('D', 'D'): (1, 1),  # Both defect
}

async def create_enhanced_agents(n=4) -> List[EnhancedAgent]:
    """Create and initialize multiple agents concurrently."""
    # Choose from our sorted strategies
    sorted_keys = [key for key, desc in PD_STRATEGIES_SORTED]
    agents = [EnhancedAgent(f"Agent_{i}",
                            strategy_tactic=random.choice(sorted_keys),
                            cooperation_bias=random.uniform(0.3, 0.7),
                            risk_aversion=random.uniform(0.3, 0.7))
              for i in range(n)]
    agents = await asyncio.gather(*(agent.initialize() for agent in agents))
    return agents

async def simulate_interaction(agent_a: EnhancedAgent, agent_b: EnhancedAgent) -> Dict:
    """Simulate an interaction between two agents asynchronously."""
    decision_a, decision_b = await asyncio.gather(
        agent_a.decide_action(agent_b),
        agent_b.decide_action(agent_a)
    )
    
    def normalize_action(decision):
        action = str(decision.get("action", "C")).upper()
        return "C" if action == "C" else "D"
    
    action_a = normalize_action(decision_a)
    action_b = normalize_action(decision_b)
    
    payoff_a, payoff_b = payoff_matrix[(action_a, action_b)]
    agent_a.total_score += payoff_a
    agent_b.total_score += payoff_b
    agent_a.log_interaction(agent_b.name, action_a, action_b, payoff_a)
    agent_b.log_interaction(agent_a.name, action_b, action_a, payoff_b)
    
    interaction = InteractionData(
        generation=None,  # To be filled in by simulation runner
        pair=f"{agent_a.name}-{agent_b.name}",
        round_actions=f"{action_a}-{action_b}",
        payoffs=f"{payoff_a}-{payoff_b}",
        reasoning_A=decision_a.get("rationale", ""),
        reasoning_B=decision_b.get("rationale", ""),
        score_A=agent_a.total_score,
        score_B=agent_b.total_score
    )
    
    return {
        "interaction": interaction,
        "pair": f"{agent_a.name}-{agent_b.name}",
        "Actions": f"{action_a}-{action_b}",
        "Payoffs": f"{payoff_a}-{payoff_b}",
        "Strategy_A": agent_a.strategy_matrix,
        "Strategy_B": agent_b.strategy_matrix,
        "Reasoning_A": decision_a.get("rationale", ""),
        "Reasoning_B": decision_b.get("rationale", ""),
        "Score_A": agent_a.total_score,
        "Score_B": agent_b.total_score
    }

# %% [markdown]
# ## Main Simulation
# 
# The main simulation function runs multiple generations of agents, with each generation involving:
# 
# 1. Concurrent agent interactions
# 
# 2. Logging of results
# 
# 3. Evolution (selection of top performers)
# 
# 4. Creation of new agents

# %%
#%% 
async def run_llm_driven_simulation(num_agents=4, num_generations=5, models=["gpt-4-turbo"]):
    try:
        # Set the results folder to the correct path
        results_folder = "Multi-Agent-Equilibria/Games/1_Prisoners_Dilemma/simulation_results"
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_folder = os.path.join(results_folder, f"run_{current_time}")
        os.makedirs(run_folder, exist_ok=True)

        sim_data = SimulationData(hyperparameters={
            "num_agents": num_agents,
            "num_generations": num_generations,
            "payoff_matrix": {
                "CC": payoff_matrix[('C', 'C')],
                "CD": payoff_matrix[('C', 'D')],
                "DC": payoff_matrix[('D', 'C')],
                "DD": payoff_matrix[('D', 'D')]
            },
            "timestamp": current_time,
            "models": models
        })

        # Create the dictionary for cooperation biases
        PD_COOPERATION_BIASES = {key: bias for key, _, bias in PD_STRATEGIES_SORTED}

        # Update the PD_STRATEGIES dictionary for proper lookups
        PD_STRATEGIES = {key: desc for key, desc, _ in PD_STRATEGIES_SORTED}

        # Update agent creation to use strategy-specific cooperation biases
        async def create_enhanced_agents(n=4) -> List[EnhancedAgent]:
            """Create and initialize multiple agents concurrently with strategy-specific cooperation biases."""
            # Choose from our sorted strategies
            sorted_keys = [key for key, _, _ in PD_STRATEGIES_SORTED]
            agents = []

            for i in range(n):
                # Select a random strategy
                strategy = random.choice(sorted_keys)
                # Get the corresponding cooperation bias
                cooperation_bias = PD_COOPERATION_BIASES[strategy]
                # Create agent with strategy-appropriate bias
                agent = EnhancedAgent(
                    f"Agent_{i}",
                    strategy_tactic=strategy,
                    cooperation_bias=cooperation_bias,
                    risk_aversion=random.uniform(0.3, 0.7)
                )
                agents.append(agent)

            agents = await asyncio.gather(*(agent.initialize() for agent in agents))
            return agents

        agents = await create_enhanced_agents(num_agents)
        all_detailed_logs = []
        generation_summary = []

        # Track strategy distribution across generations
        strategy_distribution = {gen: {strategy: 0 for strategy, _, _ in PD_STRATEGIES_SORTED} 
                                for gen in range(1, num_generations + 1)}

        for gen in range(num_generations):
            print(f"\n=== Generation {gen+1} ===")
            detailed_logs = []
            random.shuffle(agents)

            # Record strategy distribution for this generation
            for agent in agents:
                strategy_distribution[gen+1][agent.strategy_tactic] += 1

            interaction_tasks = []
            for i in range(0, len(agents), 2):
                if i + 1 < len(agents):
                    interaction_tasks.append(simulate_interaction(agents[i], agents[i+1]))

            interaction_results = await asyncio.gather(*interaction_tasks)

            gen_metrics = {
                "mutual_cooperation": 0,
                "mutual_defection": 0,
                "temptation_payoffs": 0,
                "sucker_payoffs": 0,
                "total_payoffs": 0
            }

            for result in interaction_results:
                result["interaction"].generation = gen + 1
                sim_data.add_interaction(result["interaction"])

                detailed_logs.append({
                    "Generation": gen+1,
                    **result
                })
                print(f"{result['pair']}: {result['Actions']}, Payoffs: {result['Payoffs']}")

                actions = result['Actions'].split('-')
                payoffs = [int(p) for p in result['Payoffs'].split('-')]
                gen_metrics["total_payoffs"] += sum(payoffs)

                if actions == ['C', 'C']:
                    gen_metrics["mutual_cooperation"] += 1
                elif actions == ['D', 'D']:
                    gen_metrics["mutual_defection"] += 1
                elif 'D' in actions and 'C' in actions:
                    if actions[0] == 'D': 
                        gen_metrics["temptation_payoffs"] += 1
                    else: 
                        gen_metrics["sucker_payoffs"] += 1

            all_detailed_logs.extend(detailed_logs)

            # Calculate Nash equilibrium metrics
            br_diffs = []
            nash_regrets = []

            for result in interaction_results:
                # Calculate best response differences
                action_a, action_b = result['Actions'].split('-')
                payoff_a, payoff_b = result['Payoffs'].split('-')
                payoff_a, payoff_b = int(payoff_a), int(payoff_b)

                # Calculate theoretical best responses
                br_a = 'D' if action_b == 'C' else 'C'  # Simplified best response logic
                br_b = 'D' if action_a == 'C' else 'C'

                # Calculate payoff differences
                br_diff_a = payoff_matrix[(br_a, action_b)][0] - payoff_a
                br_diff_b = payoff_matrix[(action_a, br_b)][1] - payoff_b

                br_diffs.extend([br_diff_a, br_diff_b])
                nash_regrets.extend([br_diff_a > 0, br_diff_b > 0])

            # Calculate equilibrium metrics
            avg_br_diff = np.mean(br_diffs) if br_diffs else 0
            nash_dev = np.mean(nash_regrets) if nash_regrets else 0

            # Store equilibrium data
            sim_data.add_equilibrium_data(gen+1, nash_dev, avg_br_diff)

            total_possible = 3 * len(interaction_results) * 2
            pareto_eff = gen_metrics["total_payoffs"] / total_possible if total_possible > 0 else 0
            strat_diversity = len(set(hash(json.dumps(a.strategy_matrix)) for a in agents))

            avg_score = sum(a.total_score for a in agents) / len(agents) if agents else 0
            generation_summary.append({
                "Generation": gen+1,
                "Average_Score": avg_score,
                "Pareto_Efficiency": pareto_eff,
                "Nash_Deviation": nash_dev,
                "Strategy_Diversity": strat_diversity,
                "best_response_diff": avg_br_diff,
                **gen_metrics
            })

            # After simulation, add strategy distribution to summary
            generation_summary[-1]["strategy_distribution"] = strategy_distribution[gen+1].copy()

            agents.sort(key=lambda a: a.total_score, reverse=True)
            top_agents = agents[:num_agents // 2]
            new_agents = await create_enhanced_agents(num_agents // 2)
            agents = top_agents + new_agents

            for agent in agents:
                agent.total_score = 0

        with open(os.path.join(run_folder, "parameters.json"), 'w') as f:
            json.dump(sim_data.hyperparameters, f, indent=4)

        detailed_df = pd.DataFrame([asdict(inter) for inter in sim_data.interactions])
        detailed_df.to_csv(os.path.join(run_folder, "detailed_logs.csv"), index=False)
        detailed_df.to_json(os.path.join(run_folder, "detailed_logs.json"), orient="records", indent=4)

        summary_df = pd.DataFrame(generation_summary)
        summary_df.to_csv(os.path.join(run_folder, "generation_summary.csv"), index=False)
        summary_df.to_json(os.path.join(run_folder, "generation_summary.json"), orient="records", indent=4)

        plt.figure(figsize=(10, 6))
        generations = range(1, num_generations + 1)
        avg_scores = [entry["Average_Score"] for entry in generation_summary]
        plt.plot(generations, avg_scores, marker='o', linestyle='-', linewidth=2)
        plt.title("Average Cooperation Score over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Average Score")
        plt.grid(True)
        plt.savefig(os.path.join(run_folder, "cooperation_over_generations.png"))
        plt.close()

        def create_strategy_distribution_plot(strategy_distribution, run_folder, num_generations):
            """
            Create a line plot showing the distribution of strategies across generations.
            Each line represents a different strategy.
            """
            plt.figure(figsize=(12, 8))

            # Get all unique strategies
            strategies = [strategy for strategy, _, _ in PD_STRATEGIES_SORTED]
            generations = range(1, num_generations + 1)

            # Create line for each strategy
            for strategy in strategies:
                counts = [strategy_distribution[gen][strategy] for gen in generations]
                plt.plot(generations, counts, marker='o', linewidth=2, label=strategy)

            plt.title("Strategy Distribution Across Generations", fontsize=16)
            plt.xlabel("Generation", fontsize=14)
            plt.ylabel("Number of Agents", fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(title="Strategies", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            plt.savefig(os.path.join(run_folder, "strategy_distribution.png"))
            plt.close()

        # Create visualization for strategy distribution over generations
        create_strategy_distribution_plot(strategy_distribution, run_folder, num_generations)

        # Add this function to count strategy occurrences over generations
        def count_strategy_occurrences(agents, num_generations):
            strategy_counts = {strategy: [0] * num_generations for strategy, _, _ in PD_STRATEGIES_SORTED}
            for gen in range(num_generations):
                for agent in agents:
                    strategy_counts[agent.strategy_tactic][gen] += 1
            return strategy_counts

        # Create research visualizations function with proper indentation
        def create_research_visualizations():
            plt.figure(figsize=(12, 8))

            # Existing plots
            plt.subplot(2, 2, 1)
            plt.plot([m["Strategy_Diversity"] for m in generation_summary], marker='o')
            plt.title("Strategy Diversity Over Generations")

            plt.subplot(2, 2, 2)
            plt.bar(["Mutual C", "Mutual D", "Temptation", "Sucker"], 
                    [generation_summary[-1]["mutual_cooperation"],
                     generation_summary[-1]["mutual_defection"],
                     generation_summary[-1]["temptation_payoffs"],
                     generation_summary[-1]["sucker_payoffs"]])
            plt.title("Final Generation Outcome Distribution")

            plt.subplot(2, 2, 3)
            plt.plot([m["Pareto_Efficiency"] for m in generation_summary], color='green')
            plt.title("Pareto Efficiency Progress")

            # Get strategy distribution data
            strategies = [strategy for strategy, _, _ in PD_STRATEGIES_SORTED]

            # Plot strategy distribution over generations
            plt.subplot(2, 2, 4)
            for strategy in strategies:
                counts = [gen_summary.get("strategy_distribution", {}).get(strategy, 0) 
                         for gen_summary in generation_summary]
                plt.plot(generations, counts, marker='o', linewidth=2, label=strategy)

            plt.title("Strategy Distribution")
            plt.xlabel("Generation")
            plt.ylabel("Number of Agents")
            plt.legend(loc='upper right', fontsize='x-small')

            plt.tight_layout()
            plt.savefig(os.path.join(run_folder, "research_metrics.png"))

        # Add new visualization function
        def plot_equilibrium_metrics(sim_data, run_folder):
            """Plot Nash equilibrium deviation metrics across generations"""
            generations = sorted(sim_data.equilibrium_metrics.keys())
            nash_devs = [sim_data.equilibrium_metrics[g]['nash_deviation'] for g in generations]
            br_diffs = [sim_data.equilibrium_metrics[g]['best_response_diff'] for g in generations]

            plt.figure(figsize=(12, 6))

            # Nash Deviation plot
            plt.subplot(1, 2, 1)
            plt.plot(generations, nash_devs, marker='o', color='darkred')
            plt.title('Nash Equilibrium Deviation')
            plt.xlabel('Generation')
            plt.ylabel('Average Regret (ε)')
            plt.grid(True, alpha=0.3)

            # Best Response Difference plot
            plt.subplot(1, 2, 2)
            plt.plot(generations, br_diffs, marker='o', color='darkblue')
            plt.title('Best Response Potential')
            plt.xlabel('Generation')
            plt.ylabel('Average BR Payoff Difference')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(run_folder, "equilibrium_metrics.png"))
            plt.close()

        create_research_visualizations()
        plot_equilibrium_metrics(sim_data, run_folder)

        # Replace the old visualization code with the new comprehensive plotting
        create_comprehensive_plots(generation_summary, run_folder, strategy_distribution)
        
        print(f"\nSimulation completed. Results saved in: {run_folder}")
        return generation_summary, sim_data.interactions

    except Exception as e:
        print(f"An error occurred during the simulation: {e}")
        return None, None

# %%


# %%


# %% [markdown]
# ## Run the Simulation
# 
# Execute the simulation with specified parameters.
# 
# Note: This will make multiple API calls to OpenAI's GPT-4, so ensure your API key is set up correctly.
# 
# 
# 
# To run in a Jupyter notebook, first install nest_asyncio:
# 
# ```bash
# 
# pip install nest_asyncio
# 
# ```
# 
# 
# 
# Then run the following cells:

# %% [markdown]
# 

# %%
#%%
# Helper function to run async code in Jupyter
async def run_simulation():
    return await run_llm_driven_simulation(num_agents=5, num_generations=3)


# %%
#%%
# Setup for Jupyter notebook execution
if not os.getenv('JUPYTER_RUNNING_IN_SCRIPT'):
    try:
        import nest_asyncio
        nest_asyncio.apply()
        import asyncio
        # Create event loop and run simulation
        loop = asyncio.get_event_loop()
        summary, logs = loop.run_until_complete(run_simulation())
    except ImportError:
        print("Please install nest_asyncio: pip install nest_asyncio")
else:
    # For running as a script
    summary, logs = asyncio.run(run_llm_driven_simulation(num_agents=4, num_generations=3))



