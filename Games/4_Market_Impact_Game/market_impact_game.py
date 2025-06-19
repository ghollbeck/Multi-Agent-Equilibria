# market_impact_game.py
import os
import json
import random
import asyncio
import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import aiohttp
import time
import sys
from tqdm import tqdm
import subprocess

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ========== SIMULATION CONFIGURATION ==========
# You can edit these parameters to customize the simulation
SIMULATION_CONFIG = {
    # Simulation structure
    "num_agents": 5,           # Number of agents in the simulation
    "num_rounds": 10,          # Number of rounds per generation
    "num_generations": 1,      # Number of generations to run

    # Market parameters
    "init_price": 100.0,       # Initial asset price
    "impact_factor": 0.02,     # How strongly trades affect the market price
    "volatility": 0.01,        # Random price noise component

    # LLM parameters
    "llm_provider": "openai",  # LLM provider ('openai' or 'litellm')
    "llm_model": "gpt-4o",     # Default model to use
    "temperature": 0.7,        # Temperature for LLM sampling (0.0-1.0)
    
    # Random seed (set to None for random behavior)
    "seed": None,              # Random seed for reproducibility
}
# ==========================================

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("market_simulation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('market')

# ========== LLM Client Setup ==========
# Try importing OpenAI package - handle both old and new versions
try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_NEW_API = True
except ImportError:
    # Fall back to old API
    import openai
    OPENAI_NEW_API = False

# Set up API clients
if OPENAI_NEW_API:
    api_key = os.getenv("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=api_key)
    async_openai_client = AsyncOpenAI(api_key=api_key)

class LiteLLM:
    """Wrapper for LiteLLM calls with fallback mechanism."""
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("LITELLM_API_KEY")
    
    async def chat_completion(self, messages, temperature=0.7):
        if not self.api_key:
            return {
                "action": random.choice(["BUY", "SELL", "HOLD"]),
                "quantity": random.uniform(0.1, 5.0),
                "rationale": "Fallback random decision (no LiteLLM API key)"
            }
        return {
            "action": random.choice(["BUY", "SELL", "HOLD"]),
            "quantity": random.uniform(0.1, 5.0),
            "rationale": "Mock LiteLLM response"
        }

################################################################################
# DATA STRUCTURES
################################################################################
@dataclass
class MarketInteraction:
    round_number: int
    agent_name: str
    action: str
    quantity: float
    rationale: str
    old_price: float
    new_price: float
    profit: float
    total_position: float
    total_pnl: float
    generation: int = 1  # For multi-generation

@dataclass
class SimulationData:
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    interactions: List[MarketInteraction] = field(default_factory=list)
    equilibrium_metrics: dict = field(default_factory=dict)
    
    def add_interaction(self, interaction: MarketInteraction):
        self.interactions.append(interaction)
    
    def add_equilibrium_data(self, generation, metrics):
        self.equilibrium_metrics[generation] = metrics
    
    def to_dict(self):
        return {
            "hyperparams": self.hyperparams,
            "interactions": [asdict(i) for i in self.interactions],
            "equilibrium_metrics": self.equilibrium_metrics
        }

################################################################################
# MARKET AGENT (LLM-DRIVEN)
################################################################################
class MarketAgent(BaseModel):
    name: str
    model: str = Field(default="gpt-4o")
    llm_provider: str = Field(default="openai")
    llm_model: str = Field(default="gpt-4o")
    
    total_pnl: float = 0.0
    total_position: float = 0.0
    history: list = Field(default_factory=list)  # track agent's own trades
    # Agents will now see a short memory of recent price changes in the environment

    class Config:
        arbitrary_types_allowed = True

    async def decide_action(self, current_price: float, recent_info: Dict[str, Any], temperature=0.7) -> Dict:
        """
        Query an LLM to decide a trading action based on market conditions
        """
        prompt = self._build_decision_prompt(current_price, recent_info)
        
        if self.llm_provider == "openai":
            return await self._decide_with_openai(prompt, temperature)
        elif self.llm_provider == "litellm":
            return await self._decide_with_litellm(prompt, temperature)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _build_decision_prompt(self, current_price: float, recent_info: Dict[str, Any]) -> str:
        """
        Build a more sophisticated prompt including:
        - The agent's last 3 trades
        - The last 3 price changes from the environment
        """
        # 1) Own trade history
        history_text = ""
        if self.history:
            history_text = "Your recent trade history:\n"
            for trade in self.history[-3:]:
                history_text += (f"- Round {trade['round']}: {trade['action']} "
                                 f"{trade['quantity']} @ ${trade['price']:.2f}, "
                                 f"Profit: ${trade['profit']:.2f}\n")
        
        # 2) Environment price history
        env_price_changes = recent_info.get("price_history", [])
        price_history_text = ""
        if env_price_changes:
            price_history_text = "Recent market price changes:\n"
            for i, p in enumerate(env_price_changes, start=1):
                price_history_text += f"- Price {i} rounds ago: ${p:.2f}\n"
        
        round_num = recent_info.get('round', 0)
        volatility = recent_info.get('volatility', 0.01)
        
        prompt = f"""
You are a strategic algorithmic trader in a market impact game.

Current Market Information:
- Current price: ${current_price:.2f}
- Round: {round_num} of {recent_info.get('total_rounds', 10)}
- Recent volatility: {volatility:.2%}
- Your current position: {self.total_position} units
- Your total P&L: ${self.total_pnl:.2f}

{history_text}
{price_history_text}

Important: Your trading decisions (BUY/SELL/HOLD) affect the market price. 
Larger net orders from all agents raise or lower the price more.

Return one JSON object:
{{
  "action": "BUY"/"SELL"/"HOLD",
  "quantity": float (positive),
  "rationale": "Short explanation"
}}
"""
        return prompt.strip()

    async def _decide_with_openai(self, prompt: str, temperature: float) -> Dict:
        """Use OpenAI to make a decision"""
        for _ in range(3):  # Retry up to 3 times
            try:
                if OPENAI_NEW_API:
                    response = await async_openai_client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are an algorithmic trader. Respond with valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature,
                        response_format={"type": "json_object"},
                        max_tokens=250
                    )
                    json_str = response.choices[0].message.content.strip()
                else:
                    # old openai usage
                    response = await openai.ChatCompletion.acreate(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are an algorithmic trader. Respond ONLY with valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature,
                        max_tokens=250
                    )
                    json_str = response.choices[0].message['content'].strip()
                
                decision = json.loads(json_str)
                
                # Validate
                action = decision.get("action", "HOLD").upper()
                if action not in ["BUY", "SELL", "HOLD"]:
                    logger.warning(f"Invalid action '{action}' received, defaulting to HOLD")
                    action = "HOLD"
                
                quantity = float(decision.get("quantity", 0.0))
                if quantity < 0:
                    logger.warning(f"Negative quantity, converting to positive: {quantity}")
                    quantity = abs(quantity)
                
                rationale = decision.get("rationale", "No rationale provided.")
                
                return {
                    "action": action,
                    "quantity": quantity,
                    "rationale": rationale
                }
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Retrying decision for {self.name} due to parse error: {e}")
                await asyncio.sleep(1)
                continue
        
        logger.error(f"All retries failed for {self.name}; defaulting to HOLD.")
        return {
            "action": "HOLD",
            "quantity": 0.0,
            "rationale": "Fallback after multiple LLM failures"
        }

    async def _decide_with_litellm(self, prompt: str, temperature: float) -> Dict:
        """Use LiteLLM to make a decision"""
        messages = [
            {"role": "system", "content": "You are an algorithmic trader. Respond ONLY with valid JSON."},
            {"role": "user", "content": prompt}
        ]
        for _ in range(3):
            try:
                response = await self._call_litellm_async(messages, model=self.llm_model, temperature=temperature)
                json_str = response.strip()
                decision = json.loads(json_str)

                action = decision.get("action", "HOLD").upper()
                if action not in ["BUY", "SELL", "HOLD"]:
                    logger.warning(f"Invalid action '{action}' received, defaulting to HOLD")
                    action = "HOLD"
                
                quantity = float(decision.get("quantity", 0.0))
                if quantity < 0:
                    quantity = abs(quantity)
                
                rationale = decision.get("rationale", "No rationale provided.")
                
                return {
                    "action": action,
                    "quantity": quantity,
                    "rationale": rationale
                }
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Retrying decision for {self.name}: {e}")
                await asyncio.sleep(1)
                continue
        
        logger.error(f"All retries failed for {self.name}; defaulting to HOLD.")
        return {
            "action": "HOLD",
            "quantity": 0.0,
            "rationale": "Fallback after multiple LLM failures"
        }

    async def _call_litellm_async(self, messages, model="gpt-4o", temperature=0.7):
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
                    err_txt = await response.text()
                    raise ValueError(f"LiteLLM error: {response.status} {response.reason} - {err_txt}")
                data = await response.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "No response.")
    
    def log_trade(self, round_num: int, action: str, quantity: float, price: float, profit: float):
        """Record the agent's trade info in self.history."""
        self.history.append({
            'round': round_num,
            'action': action,
            'quantity': quantity,
            'price': price,
            'profit': profit
        })

################################################################################
# MARKET ENVIRONMENT
################################################################################
class MarketEnvironment:
    def __init__(self, init_price=100.0, impact_factor=0.01, volatility=0.01, flash_crash_mode=False):
        self.price = init_price
        self.impact_factor = impact_factor
        self.volatility = volatility
        self.flash_crash_mode = flash_crash_mode
        self.price_history = [init_price]
        
    def execute_trades(self, trades: Dict[str, float]) -> float:
        """
        trades: e.g. {agent_name: net_order_size}
        Return new price after applying net flow + random noise.
        """
        total_flow = sum(trades.values())
        
        # If flash_crash_mode is on, check if all trades < 0 => amplified negative impact
        all_sell = all(q < 0 for q in trades.values()) and len(trades) > 0
        
        # Basic deterministic price impact
        impact = self.impact_factor * (total_flow / 100.0)
        
        # Possibly amplify if flash_crash_mode and all SELL
        if self.flash_crash_mode and all_sell:
            logger.warning("All agents SELLing => flash crash amplification!")
            impact *= 2.0  # e.g. double the negative shift
        
        # Market noise
        noise = np.random.normal(0, self.volatility)
        
        # Price update
        new_price = self.price * (1.0 + impact + noise)
        # Floor at 1.0 to avoid going negative
        self.price = max(new_price, 1.0)
        
        self.price_history.append(self.price)
        return self.price
    
    def reset(self, init_price=None):
        if init_price is not None:
            self.price = init_price
        self.price_history = [self.price]


def calculate_equilibrium_metrics(interactions, round_num, agent_names):
    """Gather summary stats for the final round of each generation."""
    round_data = [i for i in interactions if i.round_number == round_num]
    if not round_data:
        return {}
    
    avg_price = np.mean([i.new_price for i in round_data])
    
    # Position concentration
    positions = {agent: next((x.total_position for x in round_data if x.agent_name == agent), 0)
                 for agent in agent_names}
    total_abs = sum(abs(p) for p in positions.values())
    position_concentration = 0
    if total_abs > 0:
        normalized = [abs(p)/total_abs for p in positions.values()]
        # "variance from uniform" approach
        position_concentration = sum((x - 1/len(agent_names))**2 for x in normalized)
    
    # Profit disparity
    profits = [x.profit for x in round_data]
    profit_std = np.std(profits) if len(profits) > 1 else 0.0
    
    # Trading activity
    active_trades = sum(1 for x in round_data if x.action != "HOLD")
    active_ratio = active_trades / len(agent_names) if agent_names else 0.0
    
    # Price volatility (instant round's change)
    first_item = round_data[0]
    price_change = abs((first_item.new_price / first_item.old_price) - 1) if first_item.old_price != 0 else 0.0
    
    return {
        "avg_price": float(avg_price),
        "position_concentration": float(position_concentration),
        "profit_disparity": float(profit_std),
        "trading_activity": float(active_ratio),
        "price_volatility": float(price_change)
    }

################################################################################
# MAIN SIMULATION
################################################################################
async def simulate_market_impact_game(
    num_agents=5,
    num_rounds=10,
    num_generations=1,
    init_price=100.0,
    impact_factor=0.01,
    volatility=0.01,
    llm_provider="openai",
    models=None,
    temperature=0.7,
    seed=None
) -> Tuple[SimulationData, str]:
    """
    Run a complete market impact game simulation with multiple generations.
    
    Parameters:
    -----------
    num_agents : int
        Number of agents participating in the market
    num_rounds : int 
        Number of trading rounds per generation
    num_generations : int
        Number of separate market simulations to run
    init_price : float
        Starting price of the asset
    impact_factor : float
        How strongly trades affect the market price
    volatility : float
        Random price noise component
    llm_provider : str
        Provider for LLM calls ('openai' or 'litellm')
    models : List[str]
        List of models to use with agents
    temperature : float
        Temperature for LLM sampling
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    Tuple[SimulationData, str]
        SimulationData object and path to results directory
    """
    start_time = time.time()  # Start timing
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        logger.info(f"Random seed set to {seed}")
    
    # Set default models if None provided
    if models is None:
        models = ["gpt-4o"]
    
    # Create results directory
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create the parent market_results directory in the Market Impact Game folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    market_results_dir = os.path.join(script_dir, "market_results")
    os.makedirs(market_results_dir, exist_ok=True)
    
    # Create the specific run results folder inside the market_results directory
    results_folder = os.path.join(market_results_dir, f"market_run_{current_time}")
    os.makedirs(results_folder, exist_ok=True)
    
    logger.info(f"Created results folder: {results_folder}")
    
    # Initialize simulation data
    sim_data = SimulationData(hyperparams={
        "num_agents": num_agents,
        "num_rounds": num_rounds,
        "num_generations": num_generations,
        "init_price": init_price,
        "impact_factor": impact_factor,
        "volatility": volatility,
        "llm_provider": llm_provider,
        "models": models,
        "temperature": temperature,
        "seed": seed,
        "timestamp": current_time
    })
    
    logger.info("🚀 Starting market impact game")
    logger.info(f"Parameters: {num_agents} agents, {num_generations} generations, {num_rounds} rounds")
    logger.info(f"Market: impact={impact_factor}, volatility={volatility}, init_price=${init_price:.2f}")
    logger.info(f"LLM: provider={llm_provider}, models={models}, temperature={temperature}")
    
    # Create agents
    agent_progress = tqdm(total=num_agents, desc="Creating agents", unit="agent")
    agents = []
    for i in range(num_agents):
        a = MarketAgent(
            name=f"Agent_{i+1}",
            model=random.choice(models),
            llm_provider=llm_provider,
            llm_model=random.choice(models)
        )
        agents.append(a)
        agent_progress.update(1)
    agent_progress.close()
    
    for ag in agents:
        logger.info(f"Created {ag.name} using model={ag.model}, provider={ag.llm_provider}")
    
    # Initialize environment
    env = MarketEnvironment(
        init_price=init_price,
        impact_factor=impact_factor,
        volatility=volatility
    )
    
    generation_summary = []
    agent_names = [a.name for a in agents]
    
    # Run multiple generations
    for gen in tqdm(range(num_generations), desc="Simulating Generations", unit="gen"):
        logger.info(f"=== Generation {gen+1} ===")
        
        # Reset environment & agent states
        env.reset(init_price)
        for ag in agents:
            ag.total_position = 0.0
            ag.total_pnl = 0.0
            ag.history.clear()
        
        # For each round
        for r in tqdm(range(1, num_rounds+1), desc=f"G{gen+1}", leave=False):
            old_price = env.price
            
            # Build a short history of last 3 prices for the environment
            short_price_hist = env.price_history[-3:]  # up to 3 previous prices
            # Reverse them for "1 round ago, 2 rounds ago, etc."
            short_price_hist = short_price_hist[::-1]
            
            # Each agent decides
            tasks = []
            for ag in agents:
                tasks.append(
                    ag.decide_action(
                        current_price=old_price,
                        recent_info={
                            "volatility": volatility,
                            "round": r,
                            "total_rounds": num_rounds,
                            "price_history": short_price_hist
                        },
                        temperature=temperature
                    )
                )
            
            try:
                decisions = await asyncio.gather(*tasks)
            except Exception as e:
                logger.error(f"Error in decision gathering: {e}")
                decisions = [{"action":"HOLD","quantity":0.0,"rationale":f"Error: {e}"} for _ in agents]
            
            # Convert decisions into net orders
            trades = {}
            for ag, dec in zip(agents, decisions):
                if dec["action"] == "BUY":
                    trades[ag.name] = dec["quantity"]
                elif dec["action"] == "SELL":
                    trades[ag.name] = -dec["quantity"]
                else:
                    trades[ag.name] = 0
            
            new_price = env.execute_trades(trades)
            
            # Update each agent
            for ag, dec in zip(agents, decisions):
                net_order = trades[ag.name]
                transaction_cost = 0.001 * abs(net_order) * old_price
                
                # Unrealized from existing pos
                unreal = ag.total_position * (new_price - old_price)
                
                # Update position
                ag.total_position += net_order
                
                # Round profit
                profit = unreal - transaction_cost
                ag.total_pnl += profit
                
                # Log
                interaction = MarketInteraction(
                    round_number=r,
                    agent_name=ag.name,
                    action=dec["action"],
                    quantity=dec["quantity"] if dec["action"] != "HOLD" else 0,
                    rationale=dec["rationale"],
                    old_price=old_price,
                    new_price=new_price,
                    profit=profit,
                    total_position=ag.total_position,
                    total_pnl=ag.total_pnl,
                    generation=gen+1
                )
                sim_data.add_interaction(interaction)
                
                ag.log_trade(r, dec["action"], dec["quantity"], new_price, profit)
            
            # Log round info
            volume_traded = sum(abs(v) for v in trades.values())
            logger.info(f"Round {r}: Price {old_price:.2f} -> {new_price:.2f}, Vol {volume_traded:.2f}")
            
            # On final round, gather eq metrics
            if r == num_rounds:
                eq_metrics = calculate_equilibrium_metrics(sim_data.interactions, r, agent_names)
                sim_data.add_equilibrium_data(gen+1, eq_metrics)
                logger.info(f"End of Gen {gen+1}, metrics: {eq_metrics}")
        
        # Summarize generation
        final_price = env.price
        price_vol = np.std(env.price_history)
        total_vol = sum(abs(a.total_position) for a in agents)
        max_pnl = max(a.total_pnl for a in agents)
        min_pnl = min(a.total_pnl for a in agents)
        avg_pnl = np.mean([a.total_pnl for a in agents]) if agents else 0.0
        
        generation_summary.append({
            "Generation": gen+1,
            "Final_Price": final_price,
            "Price_Volatility": price_vol,
            "Total_Trading_Volume": total_vol,
            "Max_PnL": max_pnl,
            "Min_PnL": min_pnl,
            "Avg_PnL": avg_pnl
        })
    
    logger.info(f"Simulation done in {time.time() - start_time:.2f} s.")
    
    # Save logs & produce visualizations
    save_simulation_results(sim_data, generation_summary, results_folder)
    generate_visualizations(sim_data, results_folder)
    
    return sim_data, results_folder


################################################################################
# SAVE AND PLOT RESULTS
################################################################################
def save_simulation_results(sim_data: SimulationData, gen_summary: List[Dict], results_folder: str):
    """Saves JSON/CSV logs and the generation summary."""
    sim_data_dict = sim_data.to_dict()
    # Save JSON
    with open(os.path.join(results_folder, "simulation_data.json"), 'w') as f:
        json.dump(sim_data_dict, f, indent=4)
    
    # Interactions as CSV
    df = pd.DataFrame([asdict(x) for x in sim_data.interactions])
    df.to_csv(os.path.join(results_folder, "interactions.csv"), index=False)
    
    # Gen summary
    sdf = pd.DataFrame(gen_summary)
    sdf.to_csv(os.path.join(results_folder, "generation_summary.csv"), index=False)
    with open(os.path.join(results_folder, "generation_summary.json"), 'w') as f:
        json.dump(gen_summary, f, indent=4)
    
    # Hyperparams
    with open(os.path.join(results_folder, "parameters.json"), 'w') as f:
        json.dump(sim_data.hyperparams, f, indent=4)
    
    logger.info(f"Results saved to {results_folder}")


def generate_visualizations(sim_data: SimulationData, results_folder: str):
    """Generates price over time, PnL comparison, positions over time, plus a new histogram of final PnLs."""
    interactions_df = pd.DataFrame([asdict(x) for x in sim_data.interactions])
    
    if interactions_df.empty:
        logger.warning("No data to visualize.")
        return
    
    num_generations = sim_data.hyperparams.get("num_generations", 1)
    
    # 1) Price Over Time
    plt.figure(figsize=(12, 6))
    for g in range(1, num_generations+1):
        gd = interactions_df[interactions_df['generation'] == g]
        rounds = sorted(gd['round_number'].unique())
        prices = []
        for r in rounds:
            # first row in round
            row = gd[gd['round_number'] == r].iloc[0]
            prices.append(row.new_price)
        plt.plot(rounds, prices, marker='o', label=f'Gen {g}')
    plt.title("Asset Price Over Time")
    plt.xlabel("Round")
    plt.ylabel("Price ($)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, "price_over_time.png"), dpi=300)
    plt.close()
    
    # 2) Final PnL by Agent (last gen)
    plt.figure(figsize=(12, 6))
    last_gen = num_generations
    data_lg = interactions_df[interactions_df['generation'] == last_gen]
    if not data_lg.empty:
        last_round = data_lg['round_number'].max()
        final_data = data_lg[data_lg['round_number'] == last_round]
        # unique agent => total_pnl
        agent_final = {}
        for idx, row in final_data.iterrows():
            agent_final[row.agent_name] = row.total_pnl
        sorted_items = sorted(agent_final.items(), key=lambda x: x[1], reverse=True)
        names, pnls = zip(*sorted_items)
        
        plt.bar(names, pnls, color='skyblue')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        plt.title(f"Final P&L by Agent (Gen {last_gen})")
        plt.xlabel("Agent")
        plt.ylabel("Total P&L ($)")
        plt.grid(True, axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        for i, val in enumerate(pnls):
            plt.text(i, val + (3 if val >= 0 else -3), f"{val:.2f}",
                     ha='center', va='bottom' if val >= 0 else 'top')
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, "agent_pnl_comparison.png"), dpi=300)
        plt.close()
    
    # 3) Position Evolution (last gen)
    plt.figure(figsize=(12, 6))
    if not data_lg.empty:
        for agent_name in data_lg['agent_name'].unique():
            agent_df = data_lg[data_lg['agent_name'] == agent_name]
            rounds = sorted(agent_df['round_number'].unique())
            positions = [agent_df[agent_df['round_number'] == r]['total_position'].iloc[0] for r in rounds]
            plt.plot(rounds, positions, marker='o', label=agent_name)
        plt.title(f"Agent Positions Over Time (Gen {last_gen})")
        plt.xlabel("Round")
        plt.ylabel("Position")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, "position_evolution.png"), dpi=300)
        plt.close()
    
    # 4) Distribution of Final PnLs (new feature)
    plt.figure(figsize=(10, 6))
    if not data_lg.empty:
        # last round only
        final_data = data_lg[data_lg['round_number'] == data_lg['round_number'].max()]
        final_pnls = final_data['total_pnl'].values
        plt.hist(final_pnls, bins=8, color='purple', alpha=0.7)
        plt.title("Distribution of Final PnLs (Last Generation)")
        plt.xlabel("PnL")
        plt.ylabel("Number of Agents")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, "final_pnl_distribution.png"), dpi=300)
        plt.close()
    
    # (Optional) Combined Dashboard or additional plots can be added here.
    
    logger.info(f"Visualizations saved to {results_folder}")

def format_logs_with_prettier(folder: str):
    """Optionally format JSON files with Prettier (if installed)."""
    for f in os.listdir(folder):
        if f.endswith(".json"):
            path = os.path.join(folder, f)
            try:
                subprocess.run(["prettier", "--write", path], check=True)
                logger.info(f"Formatted {f} with Prettier.")
            except FileNotFoundError:
                logger.info("Prettier not installed or not found. Skipping formatting.")
            except subprocess.CalledProcessError as e:
                logger.error(f"Prettier failed on {f}: {e}")

################################################################################
# ENTRY POINT
################################################################################
if __name__ == "__main__":
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        print("Warning: nest_asyncio not available, running without it")
    
    import argparse
    
    # Parse command line arguments (these override the SIMULATION_CONFIG values if specified)
    parser = argparse.ArgumentParser(description="Market Impact Game Simulation")
    parser.add_argument("--agents", type=int, default=SIMULATION_CONFIG["num_agents"], help="Number of agents")
    parser.add_argument("--rounds", type=int, default=SIMULATION_CONFIG["num_rounds"], help="Number of rounds per generation")
    parser.add_argument("--generations", type=int, default=SIMULATION_CONFIG["num_generations"], help="Number of generations to run")
    parser.add_argument("--init_price", type=float, default=SIMULATION_CONFIG["init_price"], help="Initial asset price")
    parser.add_argument("--impact", type=float, default=SIMULATION_CONFIG["impact_factor"], help="Market impact factor")
    parser.add_argument("--volatility", type=float, default=SIMULATION_CONFIG["volatility"], help="Market volatility")
    parser.add_argument("--provider", choices=["openai", "litellm"], default=SIMULATION_CONFIG["llm_provider"], help="LLM provider")
    parser.add_argument("--model", type=str, default=SIMULATION_CONFIG["llm_model"], help="LLM model to use")
    parser.add_argument("--temperature", type=float, default=SIMULATION_CONFIG["temperature"], help="Temperature for LLM sampling")
    parser.add_argument("--seed", type=int, default=SIMULATION_CONFIG["seed"], help="Random seed for reproducibility")
    parser.add_argument("--load_results", type=str, default=None, help="Load and analyze results from a specific run folder")
    
    args = parser.parse_args()
    
    # Print header
    logger.info("\n" + "="*80)
    logger.info(" "*25 + "MARKET IMPACT GAME SIMULATION")
    logger.info("="*80 + "\n")
    
    # If loading existing results
    if args.load_results:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        market_results_dir = os.path.join(script_dir, "market_results")
        results_folder = os.path.join(market_results_dir, args.load_results)
        
        if os.path.exists(results_folder):
            logger.info(f"Loading results from: {results_folder}")
            
            # Load simulation data
            with open(os.path.join(results_folder, "simulation_data.json"), 'r') as f:
                sim_data_dict = json.load(f)
            
            # Create visualization from loaded data
            interactions_df = pd.DataFrame(sim_data_dict["interactions"])
            generate_visualizations(SimulationData(**sim_data_dict), results_folder)
            
            logger.info(f"Visualizations regenerated in: {results_folder}")
            sys.exit(0)
        else:
            logger.error(f"Results folder not found: {results_folder}")
            sys.exit(1)
    
    # Print parameters
    logger.info("⚙️ SIMULATION PARAMETERS:")
    logger.info("-"*50)
    logger.info(f"Number of agents:      {args.agents}")
    logger.info(f"Number of rounds:      {args.rounds}")
    logger.info(f"Number of generations: {args.generations}")
    logger.info(f"Initial price:         ${args.init_price:.2f}")
    logger.info(f"Impact factor:         {args.impact}")
    logger.info(f"Volatility:            {args.volatility}")
    logger.info(f"LLM Provider:          {args.provider}")
    logger.info(f"LLM Model:             {args.model}")
    logger.info(f"Temperature:           {args.temperature}")
    logger.info(f"Random seed:           {args.seed}")
    logger.info("-"*50 + "\n")
    
    # Run simulation
    loop = asyncio.get_event_loop()
    sim_data, results_folder = loop.run_until_complete(
        simulate_market_impact_game(
            num_agents=args.agents,
            num_rounds=args.rounds,
            num_generations=args.generations,
            init_price=args.init_price,
            impact_factor=args.impact,
            volatility=args.volatility,
            llm_provider=args.provider,
            models=[args.model],
            temperature=args.temperature,
            seed=args.seed
        )
    )
    
    # Format logs with Prettier if available
    format_logs_with_prettier(results_folder)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info(" "*30 + "SIMULATION COMPLETED")
    logger.info("="*80 + "\n")
    
    # List result files
    logger.info("\n📊 RESULTS AND VISUALIZATIONS:")
    logger.info("-"*50)
    logger.info(f"Results saved to: {results_folder}")
    for file in os.listdir(results_folder):
        if file.endswith('.png'):
            logger.info(f"Generated visualization: {os.path.join(results_folder, file)}")
    logger.info("-"*50 + "\n")
