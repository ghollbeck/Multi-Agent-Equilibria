"""Runner script to execute all treatments and append JSON‑lines to one master log."""
import json, os, pathlib, random
import warnings
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

# Check if OpenAI is available
try:
    import openai
    if os.environ.get("OPENAI_API_KEY") and not os.environ.get("OPENAI_API_KEY").startswith("sk-your"):
        OPENAI_AVAILABLE = True
        print(f"OpenAI API key found - LLM agents enabled")
    else:
        OPENAI_AVAILABLE = False
        warnings.warn("OpenAI API key not found or invalid - LLM agents disabled")
except ImportError:
    OPENAI_AVAILABLE = False
    warnings.warn("OpenAI not available – LLM agents disabled")

# ──────────────────────────────
# Parameters
# ──────────────────────────────
N_LIST = [2, 3, 5]                       # firm counts
COST   = 10.0
DELTA  = 0.2                             # much finer grid
GRID_M = 50                              # upper bound = 10 + 50·0.2 = 20
PRICE_GRID = [COST + i * DELTA for i in range(GRID_M + 1)]

A, B           = 50.0, 1.0
NOISE_LEVELS   = [0.0, 0.5 * B, 1.0 * B]
HISTORY_K      = 10
ROUNDS         = 5                       # reduced for quick runs
# reduce seeds for faster baseline/heuristic experiments
SEEDS          = 10
VERBOSE        = False

# --- LLM usage budget -------------------------------------------------
MAX_LLM_CALLS = 100          # hard cap on total OpenAI calls
SEEDS_LLM     = 2            # run very few seeds for LLM / mixed

LOG_DIR   = pathlib.Path("logs")
LOG_DIR.mkdir(exist_ok=True)
MASTER_LOG = LOG_DIR / "all_runs.jsonl"

# ──────────────────────────────
# Helpers
# ──────────────────────────────
def make_costs(N: int, asymmetric: bool) -> List[float]:
    """Identical costs unless we introduce one low‑cost firm."""
    return ([COST - 2.0] + [COST] * (N - 1)) if asymmetric else [COST] * N

def make_agents(matchup: str, costs: List[float]):
    from oligopoly import llm_call_stats
    if matchup in ("llm", "mixed") and llm_call_stats() >= MAX_LLM_CALLS:
        raise ValueError("LLM‑budget exhausted")
    
    agents = []
    for i, c in enumerate(costs):
        if matchup == "baseline":
            agents.append(BaselineAgent(c, DELTA))            # ← new delta arg
        elif matchup == "heuristic":
            agents.append(HeuristicAgent(c, DELTA, HISTORY_K))
        elif matchup == "llm":
            if not OPENAI_AVAILABLE:
                raise ValueError("OpenAI not available for LLM matchup")
            agents.append(LLMAgent(i, c, PRICE_GRID, HISTORY_K))
        elif matchup == "mixed":
            if not OPENAI_AVAILABLE:
                raise ValueError("OpenAI not available for mixed matchup")
            agents.append(MixedAgent(i, c, PRICE_GRID, HISTORY_K))
        else:
            raise ValueError(matchup)
    return agents

def run_single(seed: int, N: int, noise: float, asymmetric: bool, matchup: str):
    random.seed(seed);  np.random.seed(seed)
    costs  = make_costs(N, asymmetric)
    game   = OligopolyGame(
        N=N, cost=costs, price_grid=PRICE_GRID,
        a=A, b=B, noise_std=noise, history_length=HISTORY_K,
        rng=np.random.default_rng(seed), log_path=None   # ← disable internal logging
    )
    
    try:
        agents = make_agents(matchup, costs)
    except ValueError as e:
        if "LLM‑budget exhausted" in str(e):
            return  # silently skip once we hit the cap
        print(f"Skipping {matchup} (reason: {e})")
        return
        
    with MASTER_LOG.open("a", encoding="utf‑8") as fh:
        for _ in range(ROUNDS):
            actions = [ag.act(game, i) for i, ag in enumerate(agents)]
            demand, profits = game.step(actions)
            # one flat record per round with full metadata
            fh.write(json.dumps({
                "seed": seed, "N": N, "noise": noise, "asym": asymmetric,
                "matchup": matchup, "round": game.t,
                "prices": actions, "demand": demand[0], "profits": profits
            }) + "\n")

    if not VERBOSE:
        print(f"✓ seed={seed:3d}  N={N}  noise={noise:.1f}  asym={asymmetric}  matchup={matchup}")
    
    # When LLMs are enabled, echo the cumulative number of calls every 10 seeds
    if OPENAI_AVAILABLE and seed % 10 == 0:
        from oligopoly import llm_call_stats
        print(f"   ↪ cumulative LLM calls: {llm_call_stats()}")
        with warnings.catch_warnings(record=True) as w:
            for warning in w:
                if issubclass(warning.category, UserWarning) and "LLMAgent fallback" in str(warning.message):
                    print(f"      ⚠️  recent LLM error: {warning.message}")

def main():
    matchups = ["baseline", "heuristic"]
    if OPENAI_AVAILABLE:
        matchups.extend(["llm", "mixed"])
        print(f"Running all 4 matchups: {matchups}")
    else:
        print(f"Running only non-LLM matchups: {matchups}")
        
    # Run a small test batch first
    test_N = 3
    test_noise = 0.0
    test_asym = False
    print(f"\n=== Running test batch: N={test_N}, noise={test_noise}, asym={test_asym} ===")
    for matchup in matchups:
        for seed in range(3):  # Just 3 seeds for the test
            run_single(seed, test_N, test_noise, test_asym, matchup)
    
    print("\n=== Running full experiment ===")
    for N in N_LIST:
        for noise in NOISE_LEVELS:
            for asym in (False, True):
                for matchup in matchups:
                    seed_cap = SEEDS_LLM if matchup in ("llm", "mixed") else SEEDS
                    for seed in tqdm(range(seed_cap),
                                     desc=f"N={N}, σ={noise}, asym={asym}, {matchup}"):
                        run_single(seed, N, noise, asym, matchup)

    from oligopoly import llm_call_stats
    print(f"\n=== Finished: total successful LLM calls = {llm_call_stats()} (cap = {MAX_LLM_CALLS}) ===")

if __name__ == "__main__":
    main()
