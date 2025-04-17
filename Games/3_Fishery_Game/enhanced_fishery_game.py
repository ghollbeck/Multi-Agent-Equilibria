#!/usr/bin/env python3
"""
Enhanced Dynamic Fishery Game Simulation

Simulates a dynamic common pool resource (fishery) game where multiple agents decide how much to harvest
from a shared fish stock each round. The fish stock grows logistically and is reduced by agents' harvests.
Agents are language models using either OpenAI's API or LiteLLM.

This implementation includes:
- Asynchronous execution for improved performance
- Multiple agent types with different strategic focuses
- Detailed logging and visualization
- Comprehensive metrics and analysis
"""
import os
import time
import logging
import argparse
import random
import json
import asyncio
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import concurrent.futures
from functools import partial
from tqdm import tqdm
from dataclasses import dataclass, field, asdict
from pydantic import BaseModel, Field
import requests
import sys
from dotenv import load_dotenv
import subprocess

# Try importing OpenAI package - handle both old and new versions
try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_NEW_API = True
except ImportError:
    # Fall back to old API
    import openai
    OPENAI_NEW_API = False

# Set up script directory and default paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_DIR = os.path.join(SCRIPT_DIR, 'logs')
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, 'simulation_results')

# Load environment variables
load_dotenv()

# Configure logging
def setup_logging():
    """Configure logging with both file and console handlers."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("fishery_simulation.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('fishery')

logger = setup_logging()

# Set up API clients
if OPENAI_NEW_API:
    api_key = os.getenv("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=api_key)
    async_openai_client = AsyncOpenAI(api_key=api_key)

# =============================================================================
# Data Classes for Structured Logging
# =============================================================================
@dataclass
class HarvestData:
    """Data class for individual harvest decisions and outcomes."""
    generation: int
    round_num: int
    agent_name: str
    stock_before: float
    harvest_amount: float
    reasoning: str
    profit: float  # Score gained from this harvest
    total_score: float  # Accumulated score

@dataclass
class RoundData:
    """Data class for information about each round."""
    generation: int
    round_num: int
    stock_before: float
    total_harvest: float
    growth: float
    stock_after: float
    harvests: Dict[str, float]

@dataclass
class SimulationData:
    """Container for all simulation data and metrics."""
    hyperparameters: dict
    rounds: List[RoundData] = field(default_factory=list)
    harvests: List[HarvestData] = field(default_factory=list)
    sustainability_metrics: dict = field(default_factory=dict)

    def add_round(self, round_data: RoundData):
        """Add round data to the simulation log."""
        self.rounds.append(round_data)

    def add_harvest(self, harvest_data: HarvestData):
        """Add harvest data to the simulation log."""
        self.harvests.append(harvest_data)

    def add_sustainability_metrics(self, generation, metrics):
        """Add sustainability metrics for a generation."""
        self.sustainability_metrics[generation] = metrics

    def to_dict(self):
        """Convert the simulation data to a dictionary."""
        return {
            'hyperparameters': self.hyperparameters,
            'rounds': [asdict(r) for r in self.rounds],
            'harvests': [asdict(h) for h in self.harvests],
            'sustainability_metrics': self.sustainability_metrics
        }

# =============================================================================
# Prompt Classes for Different Agent Strategies
# =============================================================================
class Prompts1:
    """
    Prompts focused on long-term sustainability and resource management.
    These prompts encourage agents to consider the future health of the fishery.
    """
    
    @staticmethod
    def get_harvest_decision_prompt(agent_name, round_num, current_stock, history=None, max_harvest=None):
        """
        Generate a prompt for deciding how much to harvest with a sustainability focus.
        
        Parameters:
        -----------
        agent_name : str
            Name of the agent making the decision
        round_num : int
            Current round number
        current_stock : float
            Current fish stock level
        history : list, optional
            History of previous rounds
        max_harvest : float, optional
            Maximum allowed harvest (defaults to current_stock)
        
        Returns:
        --------
        str
            The prompt for harvest decision-making
        """
        max_harvest = current_stock if max_harvest is None else min(max_harvest, current_stock)
        history_str = ""
        
        if history:
            history_str = "Previous rounds:\n"
            for h in history[-5:]:  # Only show last 5 rounds to avoid token limits
                history_str += f"- Round {h['round']}: Stock was {h['stock']:.2f}, you harvested {h['harvested']:.2f}\n"
            
            if len(history) > 5:
                history_str += f"(Showing the last 5 of {len(history)} rounds)\n"
        
        return f"""
You are {agent_name} in a dynamic fishery game focused on sustainable resource management.

Game information:
- The fish stock grows according to a logistic growth model (grows fastest at intermediate levels)
- Each agent harvests a portion of the stock each round
- The game continues for multiple rounds
- You aim to maximize your total harvest across all rounds
- Harvesting too much now depletes the stock and reduces future growth potential
- The fishery can collapse if overharvested

Current state:
- Round {round_num}
- Current fish stock: {current_stock:.2f} units
- Maximum possible harvest this round: {max_harvest:.2f} units

{history_str}

Your task is to decide how many fish to harvest this round, keeping in mind:
1. Sustainability of the resource for future rounds
2. Your long-term total harvest
3. The potential growth rate of the fish population
4. The risk of stock depletion

Return your decision as a JSON object with the format:
{{
  "harvest": number between 0 and {max_harvest:.2f},
  "rationale": "Brief explanation of your decision"
}}
"""

class Prompts2:
    """
    Prompts focused on short-term profit maximization.
    These prompts encourage agents to prioritize immediate gains.
    """
    
    @staticmethod
    def get_harvest_decision_prompt(agent_name, round_num, current_stock, history=None, max_harvest=None):
        """
        Generate a prompt for deciding how much to harvest with a profit-maximizing focus.
        
        Parameters:
        -----------
        agent_name : str
            Name of the agent making the decision
        round_num : int
            Current round number
        current_stock : float
            Current fish stock level
        history : list, optional
            History of previous rounds
        max_harvest : float, optional
            Maximum allowed harvest (defaults to current_stock)
        
        Returns:
        --------
        str
            The prompt for harvest decision-making
        """
        max_harvest = current_stock if max_harvest is None else min(max_harvest, current_stock)
        history_str = ""
        
        if history:
            history_str = "Previous rounds:\n"
            for h in history[-3:]:  # Show fewer rounds to emphasize short-term thinking
                history_str += f"- Round {h['round']}: Stock was {h['stock']:.2f}, you harvested {h['harvested']:.2f}\n"
            
            if len(history) > 3:
                history_str += f"(Showing only the last 3 rounds)\n"
        
        return f"""
You are {agent_name} in a competitive fishery game focused on maximizing your immediate profits.

Game information:
- You compete with other fishers to harvest from a shared fish stock
- Your profit directly corresponds to how much you harvest each round
- You aim to maximize your immediate harvest in each round
- Other fishers will likely take what you don't harvest

Current state:
- Round {round_num}
- Current fish stock: {current_stock:.2f} units
- Maximum possible harvest this round: {max_harvest:.2f} units

{history_str}

Your task is to decide how many fish to harvest this round, keeping in mind:
1. Immediate profit is your top priority
2. Competitors may take resources you don't claim
3. There's uncertainty about future availability
4. You want to maximize your share of the available resources

Return your decision as a JSON object with the format:
{{
  "harvest": number between 0 and {max_harvest:.2f},
  "rationale": "Brief explanation of your decision"
}}
"""

# =============================================================================
# Agent Implementation
# =============================================================================
class FisheryAgent(BaseModel):
    """
    An agent in the fishery game that makes decisions about how much to harvest.
    Uses language models to determine harvest amounts.
    """
    name: str
    model: str = Field(default="gpt-4")
    llm_provider: str = Field(default="openai")  # 'openai' or 'litellm'
    llm_model: str = Field(default="gpt-4")  # Model name to use with provider
    strategy_type: str = Field(default="sustainable")  # 'sustainable' or 'profit-maximizing'
    
    total_score: float = 0.0
    history: list = Field(default_factory=list)  # Each entry: {'round': int, 'stock': float, 'harvested': float}
    sustainability_index: float = 0.0  # Measure of how sustainable the agent's harvests are
    prompt_class: Any = Prompts1  # Default to sustainable prompts
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        super().__init__(**data)
        # Set the appropriate prompt class based on strategy type
        if self.strategy_type == "sustainable":
            self.prompt_class = Prompts1
        elif self.strategy_type == "profit-maximizing":
            self.prompt_class = Prompts2
    
    async def decide_harvest(self, round_num: int, current_stock: float, temperature: float = 0.7) -> Dict:
        """
        Asynchronously determine how much to harvest based on the current state.
        
        Parameters:
        -----------
        round_num : int
            Current round number
        current_stock : float
            Current fish stock level
        temperature : float
            Temperature parameter for the LLM
            
        Returns:
        --------
        Dict
            Dictionary containing the harvest amount and rationale
        """
        prompt = self.prompt_class.get_harvest_decision_prompt(
            agent_name=self.name,
            round_num=round_num,
            current_stock=current_stock,
            history=self.history,
            max_harvest=current_stock
        )
        
        if self.llm_provider == "openai":
            return await self._decide_with_openai(prompt, temperature)
        elif self.llm_provider == "litellm":
            return await self._decide_with_litellm(prompt, temperature)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    async def _decide_with_openai(self, prompt: str, temperature: float) -> Dict:
        """
        Use OpenAI to make a harvest decision.
        
        Parameters:
        -----------
        prompt : str
            The prompt to send to the LLM
        temperature : float
            Temperature parameter for the LLM
            
        Returns:
        --------
        Dict
            Dictionary containing the harvest amount and rationale
        """
        for _ in range(3):  # Retry up to 3 times
            try:
                if OPENAI_NEW_API:
                    response = await async_openai_client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are playing a fishery resource management game. Respond ONLY with valid JSON with the format: {\"harvest\": number, \"rationale\": \"string\"}"},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature,
                        max_tokens=300
                    )
                    json_str = response.choices[0].message.content.strip()
                else:
                    response = await openai.ChatCompletion.acreate(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are playing a fishery resource management game. Respond ONLY with valid JSON with the format: {\"harvest\": number, \"rationale\": \"string\"}"},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature,
                        max_tokens=300
                    )
                    json_str = response.choices[0].message['content'].strip()
                
                decision = json.loads(json_str)
                # Validate and normalize the harvest amount
                harvest = float(decision.get("harvest", 0))
                
                if "rationale" not in decision:
                    decision["rationale"] = "No rationale provided."
                    
                return decision
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Retrying decision for {self.name} due to error: {str(e)}")
                await asyncio.sleep(1)  # Brief pause before retry
                continue
        
        # Fallback if all retries fail
        logger.error(f"Using fallback decision for {self.name} after multiple failures")
        return {
            "harvest": 0.0,
            "rationale": "Fallback decision after multiple LLM failures"
        }
    
    async def _decide_with_litellm(self, prompt: str, temperature: float) -> Dict:
        """
        Use LiteLLM to make a harvest decision.
        
        Parameters:
        -----------
        prompt : str
            The prompt to send to the LLM
        temperature : float
            Temperature parameter for the LLM
            
        Returns:
        --------
        Dict
            Dictionary containing the harvest amount and rationale
        """
        messages = [
            {"role": "system", "content": "You are playing a fishery resource management game. Respond ONLY with valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        for _ in range(3):  # Retry up to 3 times
            try:
                response = await self._call_litellm_async(messages, model=self.llm_model, temperature=temperature)
                json_str = response.strip()
                decision = json.loads(json_str)
                
                # Validate and normalize the harvest amount
                harvest = float(decision.get("harvest", 0))
                
                if "rationale" not in decision:
                    decision["rationale"] = "No rationale provided."
                    
                return decision
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Retrying decision for {self.name} due to error: {str(e)}")
                await asyncio.sleep(1)
                continue
        
        # Fallback if all retries fail
        logger.error(f"Using fallback decision for {self.name} after multiple failures")
        return {
            "harvest": 0.0,
            "rationale": "Fallback decision after multiple LLM failures"
        }
    
    async def _call_litellm_async(self, messages, model="gpt-4", temperature=0.7):
        """
        Call LiteLLM API asynchronously.
        
        Parameters:
        -----------
        messages : list
            List of message dictionaries to send to the API
        model : str
            Model to use
        temperature : float
            Temperature parameter for the LLM
            
        Returns:
        --------
        str
            Response content from the API
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('LITELLM_API_KEY')}",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://litellm.sph-prod.ethz.ch/chat/completions", 
                json=payload, 
                headers=headers
            ) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise ValueError(f"LiteLLM API error: {response.status} {response.reason} - {error_text}")
                
                data = await response.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "No response from LiteLLM.")
    
    def log_harvest(self, round_num: int, stock: float, harvest: float, profit: float):
        """
        Log a harvest decision to the agent's history.
        
        Parameters:
        -----------
        round_num : int
            Round number
        stock : float
            Stock level before harvest
        harvest : float
            Amount harvested
        profit : float
            Profit gained from the harvest
        """
        self.history.append({
            'round': round_num,
            'stock': stock,
            'harvested': harvest,
            'profit': profit
        })
        self.total_score += profit
        
        # Update sustainability index (average ratio of harvest to stock)
        total_harvests = sum(h['harvested'] for h in self.history)
        avg_stock = sum(h['stock'] for h in self.history) / len(self.history) if self.history else 1
        
        # Lower values indicate more sustainable harvesting
        if avg_stock > 0:
            self.sustainability_index = total_harvests / (avg_stock * len(self.history))

# =============================================================================
# Simulation Functions
# =============================================================================
async def create_agents(
    num_agents: int, 
    strategy_distribution: Dict[str, float] = None,
    models: List[str] = None,
    llm_provider: str = "openai",
    temperature: float = 0.7
) -> List[FisheryAgent]:
    """
    Create agents for the simulation.
    
    Parameters:
    -----------
    num_agents : int
        Number of agents to create
    strategy_distribution : Dict[str, float], optional
        Distribution of strategy types (e.g., {'sustainable': 0.6, 'profit-maximizing': 0.4})
    models : List[str], optional
        List of models to use
    llm_provider : str
        LLM provider to use ('openai' or 'litellm')
    temperature : float
        Temperature parameter for initialization
        
    Returns:
    --------
    List[FisheryAgent]
        List of initialized agents
    """
    if strategy_distribution is None:
        strategy_distribution = {'sustainable': 0.5, 'profit-maximizing': 0.5}
    
    if models is None:
        models = ["gpt-4"]
    
    # Normalize the strategy distribution
    total = sum(strategy_distribution.values())
    strategy_distribution = {k: v/total for k, v in strategy_distribution.items()}
    
    # Calculate the number of agents per strategy
    strategy_counts = {}
    remaining = num_agents
    
    for strategy, proportion in strategy_distribution.items():
        if strategy == list(strategy_distribution.keys())[-1]:
            # Last strategy gets all remaining agents
            strategy_counts[strategy] = remaining
        else:
            count = int(num_agents * proportion)
            strategy_counts[strategy] = count
            remaining -= count
    
    # Create agents
    agents = []
    agent_progress = tqdm(total=num_agents, desc="Creating agents", unit="agent")
    
    for strategy, count in strategy_counts.items():
        for i in range(count):
            agent_id = len(agents) + 1
            agent = FisheryAgent(
                name=f"Agent_{agent_id}",
                model=random.choice(models),
                llm_provider=llm_provider,
                llm_model=random.choice(models),
                strategy_type=strategy
            )
            agents.append(agent)
            agent_progress.update(1)
    
    agent_progress.close()
    return agents

async def simulate_round(
    round_num: int,
    agents: List[FisheryAgent],
    current_stock: float,
    r: float,
    K: float,
    temperature: float,
    generation: int,
    sim_data: SimulationData
) -> float:
    """
    Simulate a single round of the fishery game.
    
    Parameters:
    -----------
    round_num : int
        Current round number
    agents : List[FisheryAgent]
        List of agents participating in the simulation
    current_stock : float
        Current fish stock level
    r : float
        Intrinsic growth rate
    K : float
        Carrying capacity
    temperature : float
        Temperature parameter for LLM decisions
    generation : int
        Current generation number
    sim_data : SimulationData
        Simulation data object for logging
        
    Returns:
    --------
    float
        Next stock level after growth and harvests
    """
    logger.info(f"=== Round {round_num} ===")
    logger.info(f"Current stock: {current_stock:.2f}")
    
    # Skip this round if stock is depleted
    if current_stock <= 0.01:
        logger.warning("Stock depleted - skipping round")
        # Record round data with zero harvests
        round_data = RoundData(
            generation=generation,
            round_num=round_num,
            stock_before=current_stock,
            total_harvest=0.0,
            growth=0.0,
            stock_after=current_stock,
            harvests={agent.name: 0.0 for agent in agents}
        )
        sim_data.add_round(round_data)
        return current_stock
    
    # Get decisions from all agents concurrently
    decision_tasks = []
    for agent in agents:
        task = agent.decide_harvest(round_num, current_stock, temperature)
        decision_tasks.append(task)
    
    decisions = await asyncio.gather(*decision_tasks)
    
    # Process harvests
    total_harvest = 0.0
    harvests = {}
    
    for agent, decision in zip(agents, decisions):
        harvest = float(decision.get("harvest", 0.0))
        harvest = max(0.0, min(harvest, current_stock - total_harvest))  # Ensure harvest is valid
        
        agent.log_harvest(round_num, current_stock, harvest, harvest)  # Profit = harvest for simplicity
        harvests[agent.name] = harvest
        total_harvest += harvest
        
        # Log the harvest decision
        harvest_data = HarvestData(
            generation=generation,
            round_num=round_num,
            agent_name=agent.name,
            stock_before=current_stock,
            harvest_amount=harvest,
            reasoning=decision.get("rationale", "No rationale provided"),
            profit=harvest,
            total_score=agent.total_score
        )
        sim_data.add_harvest(harvest_data)
        
        logger.info(f"[{agent.name}] harvested {harvest:.2f} - Rationale: {decision.get('rationale', 'No rationale')[:100]}...")
    
    # Calculate growth using logistic growth model
    growth = r * current_stock * (1 - current_stock / K)
    next_stock = current_stock + growth - total_harvest
    next_stock = max(0.0, next_stock)
    
    logger.info(f"Total harvest: {total_harvest:.2f}, Growth: {growth:.2f}, Next stock: {next_stock:.2f}")
    
    # Record round data
    round_data = RoundData(
        generation=generation,
        round_num=round_num,
        stock_before=current_stock,
        total_harvest=total_harvest,
        growth=growth,
        stock_after=next_stock,
        harvests=harvests
    )
    sim_data.add_round(round_data)
    
    return next_stock

async def run_fishery_simulation(
    num_agents: int = 4,
    num_generations: int = 1,
    num_rounds: int = 20,
    r: float = 0.2,
    K: float = 100.0,
    init_stock: float = 50.0,
    strategy_distribution: Dict[str, float] = None,
    models: List[str] = None,
    llm_provider: str = "openai",
    temperature: float = 0.7,
    seed: int = None
) -> Tuple[SimulationData, str]:
    """
    Run the fishery game simulation.
    
    Parameters:
    -----------
    num_agents : int
        Number of agents to participate in the simulation
    num_generations : int
        Number of generations to run
    num_rounds : int
        Number of rounds per generation
    r : float
        Intrinsic growth rate
    K : float
        Carrying capacity
    init_stock : float
        Initial fish stock level
    strategy_distribution : Dict[str, float], optional
        Distribution of strategy types
    models : List[str], optional
        List of models to use
    llm_provider : str
        LLM provider to use
    temperature : float
        Temperature parameter for LLM
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    Tuple[SimulationData, str]
        Simulation data and path to results folder
    """
    start_time = time.time()  # Start timer
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        logger.info(f"Random seed set to {seed}")
    
    # Create results directory
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_folder = os.path.join(DEFAULT_RESULTS_DIR, f"fishery_run_{current_time}")
    os.makedirs(results_folder, exist_ok=True)
    logger.info(f"Created results folder: {results_folder}")
    
    # Set default distributions if not provided
    if strategy_distribution is None:
        strategy_distribution = {
            "sustainable": 0.5,
            "profit-maximizing": 0.5
        }
    
    if models is None:
        models = ["gpt-4"]
    
    # Initialize simulation data
    sim_data = SimulationData(hyperparameters={
        "num_agents": num_agents,
        "num_generations": num_generations,
        "num_rounds": num_rounds,
        "r": r,
        "K": K,
        "init_stock": init_stock,
        "strategy_distribution": strategy_distribution,
        "models": models,
        "llm_provider": llm_provider,
        "temperature": temperature,
        "seed": seed,
        "timestamp": current_time
    })
    
    logger.info("🚀 Starting fishery game simulation")
    logger.info(f"Parameters: {num_agents} agents, {num_generations} generations, {num_rounds} rounds")
    logger.info(f"Fishery parameters: r={r}, K={K}, initial stock={init_stock}")
    logger.info(f"Strategy distribution: {strategy_distribution}")
    logger.info(f"Models: {models}")
    logger.info(f"LLM provider: {llm_provider}")
    
    # Create agents
    logger.info("Creating agents...")
    agents = await create_agents(
        num_agents=num_agents,
        strategy_distribution=strategy_distribution,
        models=models,
        llm_provider=llm_provider,
        temperature=temperature
    )
    
    # Log agent information
    for agent in agents:
        logger.info(f"Created {agent.name} with strategy: {agent.strategy_type}, model: {agent.model}")
    
    generation_summary = []
    
    # Run through generations
    for gen in tqdm(range(num_generations), desc="Simulating Generations", unit="generation"):
        logger.info(f"\n🔄 === Generation {gen+1} ===")
        
        # Reset stock for this generation
        stock = init_stock
        
        # Run through rounds
        for round_num in tqdm(range(1, num_rounds+1), desc=f"Gen {gen+1} Rounds", unit="round", leave=False):
            stock = await simulate_round(
                round_num=round_num,
                agents=agents,
                current_stock=stock,
                r=r,
                K=K,
                temperature=temperature,
                generation=gen+1,
                sim_data=sim_data
            )
        
        # Calculate sustainability metrics for this generation
        metrics = calculate_sustainability_metrics(sim_data, gen+1, K)
        sim_data.add_sustainability_metrics(gen+1, metrics)
        
        # Add to generation summary
        generation_summary.append({
            "Generation": gen+1,
            "Final_Stock": stock,
            "Sustainability_Index": metrics["avg_sustainability_index"],
            "Total_Harvest": metrics["total_harvest"],
            "Resource_Utilization": metrics["resource_utilization"],
            "Stock_Variance": metrics["stock_variance"]
        })
        
        logger.info(f"Generation {gen+1} complete. Final stock: {stock:.2f}")
        logger.info(f"Sustainability metrics: {metrics}")
        
        # Reset agent scores for next generation but keep history
        for agent in agents:
            agent.total_score = 0.0
    
    logger.info(f"Simulation completed in {time.time() - start_time:.2f} seconds")
    
    # Save results
    save_simulation_results(sim_data, generation_summary, results_folder)
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    generate_visualizations(sim_data, results_folder)
    
    return sim_data, results_folder

def calculate_sustainability_metrics(sim_data: SimulationData, generation: int, carrying_capacity: float) -> Dict:
    """
    Calculate sustainability metrics for a generation.
    
    Parameters:
    -----------
    sim_data : SimulationData
        Simulation data object
    generation : int
        Generation number to calculate metrics for
    carrying_capacity : float
        Carrying capacity of the fishery
        
    Returns:
    --------
    Dict
        Dictionary of sustainability metrics
    """
    # Filter data for this generation
    gen_rounds = [r for r in sim_data.rounds if r.generation == generation]
    gen_harvests = [h for h in sim_data.harvests if h.generation == generation]
    
    if not gen_rounds or not gen_harvests:
        return {
            "avg_sustainability_index": 0.0,
            "total_harvest": 0.0,
            "resource_utilization": 0.0,
            "stock_variance": 0.0,
            "agent_scores": {}
        }
    
    # Calculate metrics
    stocks = [r.stock_before for r in gen_rounds]
    final_stock = gen_rounds[-1].stock_after
    
    # Total harvest
    total_harvest = sum(r.total_harvest for r in gen_rounds)
    
    # Resource utilization (how efficiently used vs. potential growth)
    theoretical_max = carrying_capacity * len(gen_rounds)
    resource_utilization = total_harvest / theoretical_max if theoretical_max > 0 else 0
    
    # Stock variance (lower is better, indicates stability)
    stock_variance = np.var(stocks) if len(stocks) > 1 else 0
    
    # Agent scores
    agent_names = set(h.agent_name for h in gen_harvests)
    agent_scores = {}
    agent_sustainability = {}
    
    for agent_name in agent_names:
        agent_harvests = [h for h in gen_harvests if h.agent_name == agent_name]
        total_score = agent_harvests[-1].total_score if agent_harvests else 0
        agent_scores[agent_name] = total_score
        
        # Calculate sustainability index for each agent
        if agent_harvests:
            harvest_ratio_sum = sum(h.harvest_amount / h.stock_before if h.stock_before > 0 else 0 for h in agent_harvests)
            sustainability_index = harvest_ratio_sum / len(agent_harvests) if len(agent_harvests) > 0 else 0
            agent_sustainability[agent_name] = sustainability_index
    
    avg_sustainability_index = np.mean(list(agent_sustainability.values())) if agent_sustainability else 0
    
    return {
        "avg_sustainability_index": float(avg_sustainability_index),
        "total_harvest": float(total_harvest),
        "resource_utilization": float(resource_utilization),
        "stock_variance": float(stock_variance),
        "final_stock": float(final_stock),
        "agent_scores": agent_scores,
        "agent_sustainability": agent_sustainability
    }

def save_simulation_results(sim_data: SimulationData, generation_summary: List[Dict], results_folder: str):
    """
    Save simulation results to files.
    
    Parameters:
    -----------
    sim_data : SimulationData
        Simulation data object
    generation_summary : List[Dict]
        List of generation summaries
    results_folder : str
        Path to the results folder
    """
    # Save simulation data
    sim_data_dict = sim_data.to_dict()
    with open(os.path.join(results_folder, "simulation_data.json"), 'w') as f:
        json.dump(sim_data_dict, f, indent=4)
    
    # Save round data as CSV
    rounds_df = pd.DataFrame([asdict(r) for r in sim_data.rounds])
    rounds_df.to_csv(os.path.join(results_folder, "rounds.csv"), index=False)
    
    # Save harvest data as CSV
    harvests_df = pd.DataFrame([asdict(h) for h in sim_data.harvests])
    harvests_df.to_csv(os.path.join(results_folder, "harvests.csv"), index=False)
    
    # Save generation summary
    summary_df = pd.DataFrame(generation_summary)
    summary_df.to_csv(os.path.join(results_folder, "generation_summary.csv"), index=False)
    with open(os.path.join(results_folder, "generation_summary.json"), 'w') as f:
        json.dump(generation_summary, f, indent=4)
    
    # Save parameters
    with open(os.path.join(results_folder, "parameters.json"), 'w') as f:
        json.dump(sim_data.hyperparameters, f, indent=4)
    
    logger.info(f"Results saved to {results_folder}")

def generate_visualizations(sim_data: SimulationData, results_folder: str):
    """
    Generate visualizations from the simulation data.
    
    Parameters:
    -----------
    sim_data : SimulationData
        Simulation data object
    results_folder : str
        Path to the results folder
    """
    # Extract data
    rounds_df = pd.DataFrame([asdict(r) for r in sim_data.rounds])
    harvests_df = pd.DataFrame([asdict(h) for h in sim_data.harvests])
    
    if rounds_df.empty or harvests_df.empty:
        logger.warning("No data to visualize")
        return
    
    # Get hyperparameters
    K = sim_data.hyperparameters.get("K", 100.0)
    r = sim_data.hyperparameters.get("r", 0.2)
    num_generations = sim_data.hyperparameters.get("num_generations", 1)
    
    # 1. Plot stock levels over time with carrying capacity
    plt.figure(figsize=(12, 6))
    
    for gen in range(1, num_generations + 1):
        gen_data = rounds_df[rounds_df['generation'] == gen]
        if not gen_data.empty:
            plt.plot(gen_data['round_num'], gen_data['stock_before'], 
                     marker='o', label=f'Generation {gen}')
    
    plt.axhline(y=K, color='r', linestyle='--', label=f'Carrying Capacity (K={K})')
    plt.axhline(y=K/2, color='g', linestyle='--', label=f'Optimal Sustainable Stock (K/2={K/2})')
    
    plt.title('Fish Stock Levels Over Time')
    plt.xlabel('Round')
    plt.ylabel('Stock Level')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, "stock_levels.png"), dpi=300)
    plt.close()
    
    # 2. Plot total harvest per round for each generation
    plt.figure(figsize=(12, 6))
    
    for gen in range(1, num_generations + 1):
        gen_data = rounds_df[rounds_df['generation'] == gen]
        if not gen_data.empty:
            plt.plot(gen_data['round_num'], gen_data['total_harvest'], 
                     marker='x', label=f'Generation {gen}')
    
    plt.title('Total Harvest Per Round')
    plt.xlabel('Round')
    plt.ylabel('Total Harvest')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, "total_harvest.png"), dpi=300)
    plt.close()
    
    # 3. Plot individual agent harvests (last generation only)
    if num_generations > 0:
        last_gen = num_generations
        last_gen_rounds = rounds_df[rounds_df['generation'] == last_gen]
        
        if not last_gen_rounds.empty:
            plt.figure(figsize=(14, 7))
            
            # Get each agent's harvests
            agent_harvests = {}
            for _, round_data in last_gen_rounds.iterrows():
                harvests = round_data['harvests']
                if isinstance(harvests, str):
                    # If stored as string, convert to dict
                    harvests = json.loads(harvests.replace("'", '"'))

                for agent, harvest in harvests.items():
                    if agent not in agent_harvests:
                        agent_harvests[agent] = []
                    agent_harvests[agent].append(harvest)
            
            # Plot each agent's harvests
            rounds = last_gen_rounds['round_num'].tolist()
            for agent, harvests in agent_harvests.items():
                plt.plot(rounds, harvests, marker='o', label=agent)
            
            plt.title(f'Individual Agent Harvests (Generation {last_gen})')
            plt.xlabel('Round')
            plt.ylabel('Harvest Amount')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(results_folder, "individual_harvests.png"), dpi=300)
            plt.close()
    
    # 4. Create a comprehensive dashboard visualization
    plt.figure(figsize=(16, 12))
    plt.suptitle('Fishery Game Simulation Dashboard', fontsize=16)
    
    # 4.1 Stock levels over time
    plt.subplot(2, 2, 1)
    for gen in range(1, num_generations + 1):
        gen_data = rounds_df[rounds_df['generation'] == gen]
        if not gen_data.empty:
            plt.plot(gen_data['round_num'], gen_data['stock_before'], 
                     marker='o', label=f'Gen {gen}')
    
    plt.axhline(y=K, color='r', linestyle='--', label=f'K={K}')
    plt.axhline(y=K/2, color='g', linestyle='--', label=f'K/2={K/2}')
    plt.title('Stock Levels')
    plt.xlabel('Round')
    plt.ylabel('Stock')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4.2 Total harvest per round
    plt.subplot(2, 2, 2)
    for gen in range(1, num_generations + 1):
        gen_data = rounds_df[rounds_df['generation'] == gen]
        if not gen_data.empty:
            plt.plot(gen_data['round_num'], gen_data['total_harvest'], 
                     marker='x', label=f'Gen {gen}')
    
    plt.title('Total Harvest Per Round')
    plt.xlabel('Round')
    plt.ylabel('Harvest')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4.3 Sustainability metrics by generation
    plt.subplot(2, 2, 3)
    
    if num_generations > 0:
        metrics = []
        gen_nums = []
        
        for gen in range(1, num_generations + 1):
            if gen in sim_data.sustainability_metrics:
                gen_metrics = sim_data.sustainability_metrics[gen]
                metrics.append([
                    gen_metrics.get('avg_sustainability_index', 0),
                    gen_metrics.get('resource_utilization', 0),
                    gen_metrics.get('stock_variance', 0) / K  # Normalize variance by K for scale
                ])
                gen_nums.append(gen)
        
        if metrics:
            metrics = np.array(metrics)
            x = np.array(gen_nums)
            width = 0.25
            
            plt.bar(x - width, metrics[:, 0], width, label='Sustainability')
            plt.bar(x, metrics[:, 1], width, label='Resource Util.')
            plt.bar(x + width, metrics[:, 2], width, label='Stock Var. (norm)')
            
            plt.title('Sustainability Metrics by Generation')
            plt.xlabel('Generation')
            plt.ylabel('Metric Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    # 4.4 Final stock by generation
    plt.subplot(2, 2, 4)
    
    final_stocks = []
    gen_nums = []
    
    for gen in range(1, num_generations + 1):
        gen_data = rounds_df[rounds_df['generation'] == gen]
        if not gen_data.empty:
            final_stocks.append(gen_data.iloc[-1]['stock_after'])
            gen_nums.append(gen)
    
    if final_stocks:
        bars = plt.bar(gen_nums, final_stocks, color='skyblue')
        plt.title('Final Stock by Generation')
        plt.xlabel('Generation')
        plt.ylabel('Final Stock')
        plt.axhline(y=K, color='r', linestyle='--', label=f'K={K}')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
    plt.savefig(os.path.join(results_folder, "dashboard.png"), dpi=300)
    plt.close()
    
    # 5. Strategy comparison (if multiple strategy types exist)
    agent_strategies = {}
    for h in sim_data.harvests:
        if h.agent_name not in agent_strategies:
            # Determine strategy by name (assumes strategy is in agent name)
            if 'sustainable' in h.agent_name.lower():
                agent_strategies[h.agent_name] = 'sustainable'
            elif 'profit' in h.agent_name.lower():
                agent_strategies[h.agent_name] = 'profit-maximizing'
            else:
                # Check through sustainability metrics
                for gen in sim_data.sustainability_metrics:
                    if 'agent_sustainability' in sim_data.sustainability_metrics[gen]:
                        sus_data = sim_data.sustainability_metrics[gen]['agent_sustainability']
                        if h.agent_name in sus_data:
                            # Higher index = less sustainable
                            agent_strategies[h.agent_name] = 'profit-maximizing' if sus_data[h.agent_name] > 0.5 else 'sustainable'
                            break
                
                # Default if not found
                if h.agent_name not in agent_strategies:
                    agent_strategies[h.agent_name] = 'unknown'
    
    # Check if we have multiple strategies
    strategies = set(agent_strategies.values())
    if len(strategies) > 1 and 'unknown' not in strategies:
        plt.figure(figsize=(12, 6))
        
        strategy_harvests = {strategy: [] for strategy in strategies}
        strategy_counts = {strategy: 0 for strategy in strategies}
        
        # Aggregate harvests by strategy
        for h in sim_data.harvests:
            if h.agent_name in agent_strategies:
                strategy = agent_strategies[h.agent_name]
                strategy_harvests[strategy].append(h.harvest_amount)
                strategy_counts[strategy] += 1
        
        # Calculate average harvest per strategy
        avg_harvests = {s: np.mean(h) if h else 0 for s, h in strategy_harvests.items()}
        
        # Plot average harvest by strategy
        plt.bar(avg_harvests.keys(), avg_harvests.values(), color=['green', 'red'])
        plt.title('Average Harvest by Strategy')
        plt.xlabel('Strategy')
        plt.ylabel('Average Harvest')
        plt.grid(True, alpha=0.3)
        
        # Add count labels
        for i, (strategy, avg) in enumerate(avg_harvests.items()):
            plt.text(i, avg + 0.1, f'n={strategy_counts[strategy]}', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, "strategy_comparison.png"), dpi=300)
        plt.close()
    
    logger.info(f"Visualizations saved to {results_folder}")

def format_logs_with_prettier(results_folder):
    """
    Format JSON files in the results folder using Prettier.
    Skip CSV files as Prettier doesn't support them.
    """
    for file_name in os.listdir(results_folder):
        if file_name.endswith('.json'):
            file_path = os.path.join(results_folder, file_name)
            try:
                subprocess.run(['prettier', '--write', file_path], check=True)
                logger.info(f"✨ Formatted {file_name} with Prettier.")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to format {file_name} with Prettier: {e}")
            except FileNotFoundError:
                logger.error(f"❌ Prettier not found. Install with 'npm install -g prettier' to enable formatting.")
                return
        elif file_name.endswith('.csv'):
            # Skip CSV files and inform user
            logger.info(f"ℹ️ Skipping {file_name} - Prettier doesn't support CSV format.")

def main():
    global DEFAULT_RESULTS_DIR  # <-- Must be at the top before any use
    parser = argparse.ArgumentParser(description="Enhanced Dynamic Fishery Game Simulation")
    
    # Basic simulation parameters
    parser.add_argument('--agents', type=int, default=4, help="Number of agents (default: 4)")
    parser.add_argument('--generations', type=int, default=1, help="Number of generations (default: 1)")
    parser.add_argument('--rounds', type=int, default=20, help="Number of rounds per generation (default: 20)")
    
    # Fishery parameters
    parser.add_argument('--r', type=float, default=0.2, help="Intrinsic growth rate (default: 0.2)")
    parser.add_argument('--K', type=float, default=100.0, help="Carrying capacity (default: 100.0)")
    parser.add_argument('--init_stock', type=float, default=50.0, help="Initial fish stock (default: 50.0)")
    
    # Agent strategy distribution
    parser.add_argument('--sustainable_ratio', type=float, default=0.5,
                      help="Ratio of agents with sustainable strategy (default: 0.5)")
    
    # LLM parameters
    parser.add_argument('--provider', choices=['openai', 'litellm'], default='openai',
                      help="LLM provider to use (default: openai)")
    parser.add_argument('--models', nargs='+', default=['gpt-4'],
                      help="LLM models to use (default: gpt-4)")
    parser.add_argument('--temperature', type=float, default=0.7,
                      help="Temperature for LLM sampling (default: 0.7)")
    parser.add_argument('--api_key', type=str, default=None,
                      help="API key for the LLM provider (default: from environment)")
    
    # Other parameters
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_DIR,
                      help=f"Directory to save results (default: {DEFAULT_RESULTS_DIR})")
    parser.add_argument('--seed', type=int, default=None,
                      help="Random seed for reproducibility")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Update global variables
    DEFAULT_RESULTS_DIR = args.results_dir
    
    # Set API key if provided
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
        if args.provider == 'litellm':
            os.environ["LITELLM_API_KEY"] = args.api_key
    
    # Calculate strategy distribution
    strategy_distribution = {
        "sustainable": args.sustainable_ratio,
        "profit-maximizing": 1.0 - args.sustainable_ratio
    }
    
    # Print header
    logger.info("\n" + "="*80)
    logger.info(" "*30 + "FISHERY GAME SIMULATION")
    logger.info("="*80 + "\n")
    
    # Print parameters
    logger.info("⚙️ SIMULATION PARAMETERS:")
    logger.info("-"*50)
    logger.info(f"Number of agents:          {args.agents}")
    logger.info(f"Number of generations:     {args.generations}")
    logger.info(f"Number of rounds:          {args.rounds}")
    logger.info(f"Intrinsic growth rate:     {args.r}")
    logger.info(f"Carrying capacity:         {args.K}")
    logger.info(f"Initial stock:             {args.init_stock}")
    logger.info(f"Strategy distribution:     {strategy_distribution}")
    logger.info(f"LLM provider:              {args.provider}")
    logger.info(f"Models:                    {args.models}")
    logger.info(f"Temperature:               {args.temperature}")
    logger.info(f"Random seed:               {args.seed}")
    logger.info("-"*50 + "\n")
    
    # Run simulation
    try:
        # Set up asyncio
        if 'ipykernel' in sys.modules:
            # We're in a Jupyter notebook, import and apply nest_asyncio
            import nest_asyncio
            nest_asyncio.apply()
            logger.info("Running in Jupyter, applied nest_asyncio")
        
        # Create and run event loop
        loop = asyncio.get_event_loop()
        sim_data, results_folder = loop.run_until_complete(run_fishery_simulation(
            num_agents=args.agents,
            num_generations=args.generations,
            num_rounds=args.rounds,
            r=args.r,
            K=args.K,
            init_stock=args.init_stock,
            strategy_distribution=strategy_distribution,
            models=args.models,
            llm_provider=args.provider,
            temperature=args.temperature,
            seed=args.seed
        ))
        
        # Format logs with Prettier if available
        format_logs_with_prettier(results_folder)
        
        # Print results location
        logger.info("\n📊 RESULTS AND VISUALIZATIONS:")
        logger.info("-"*50)
        logger.info(f"Results saved to: {results_folder}")
        for file in os.listdir(results_folder):
            if file.endswith('.png'):
                logger.info(f"Generated visualization: {os.path.join(results_folder, file)}")
        logger.info("-"*50 + "\n")
        
        logger.info("\n" + "="*80)
        logger.info(" "*30 + "SIMULATION COMPLETED")
        logger.info("="*80 + "\n")
        
        return sim_data, results_folder
        
    except Exception as e:
        logger.error(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    main()