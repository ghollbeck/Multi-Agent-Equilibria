"""Runner script to execute all treatments and store JSON‑lines logs."""
import pathlib
import random
from typing import List

import numpy as np
from tqdm import tqdm

from oligopoly import (
    BaselineAgent,
    HeuristicAgent,
    LLMAgent,
    MixedAgent,
    OligopolyGame,
)

# ---------------------------
# Parameters
# ---------------------------
N_LIST = [2, 3, 5]
COST = 10.0
DELTA = 1.0
GRID_M = 20  # price_grid spans COST .. COST + M*DELTA
PRICE_GRID = [COST + i * DELTA for i in range(GRID_M + 1)]
A = 50.0
B = 1.0
NOISE_LEVELS = [0.0, 0.5 * B, 1.0 * B]
HISTORY_K = 10
ROUNDS = 10
SEEDS = 100

LOG_DIR = pathlib.Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# ---------------------------
# Treatments
# ---------------------------
def make_costs(N: int, asymmetric: bool) -> List[float]:
    if not asymmetric:
        return [COST] * N
    low = COST - 2.0
    return [low] + [COST] * (N - 1)

def make_agents(matchup: str, costs: List[float], price_grid):
    agents = []
    for i, c in enumerate(costs):
        if matchup == "baseline":
            agents.append(BaselineAgent(c))
        elif matchup == "heuristic":
            agents.append(HeuristicAgent(c, price_grid[1] - c, HISTORY_K))
        elif matchup == "llm":
            agents.append(LLMAgent(i, c, price_grid, HISTORY_K))
        elif matchup == "mixed":
            agents.append(MixedAgent(i, c, price_grid, HISTORY_K))
        else:
            raise ValueError(matchup)
    return agents

def run_single(seed: int, N: int, noise: float, asymmetric: bool, matchup: str):
    random.seed(seed)
    np.random.seed(seed)
    costs = make_costs(N, asymmetric)
    log_file = LOG_DIR / f"log_N{N}_noise{noise}_asym{asymmetric}_{matchup}_{seed}.jsonl"
    game = OligopolyGame(
        N=N,
        cost=costs,
        price_grid=PRICE_GRID,
        a=A,
        b=B,
        noise_std=noise,
        history_length=HISTORY_K,
        rng=np.random.default_rng(seed),
        log_path=log_file,
    )
    agents = make_agents(matchup, costs, PRICE_GRID)
    for _ in range(ROUNDS):
        actions = [agent.act(game, i) for i, agent in enumerate(agents)]
        demand, profits = game.step(actions)
        # Terminal output with emojis
        print(f"\n📈 Round {game.t} Results 📈")
        print("   Prices:", [f"{p:.2f}" for p in actions])
        print("   Profits:", [f"{p:.2f}" for p in profits])
        corr = game.price_correlation(window=HISTORY_K)
        print("   Correlation:", f"{corr:.2f}" if corr is not None else "N/A")
        print()

def main():
    matchups = ["baseline", "heuristic", "llm", "mixed"]
    for N in N_LIST:
        for noise in NOISE_LEVELS:
            for asym in [False, True]:
                for matchup in matchups:
                    for seed in tqdm(range(SEEDS), desc=f"N={N}, noise={noise}, asym={asym}, {matchup}"):
                        run_single(seed, N, noise, asym, matchup)

if __name__ == "__main__":
    main()
