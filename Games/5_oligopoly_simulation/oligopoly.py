"""Oligopoly simulation environment, agent definitions, and logging utilities.

Author: ChatGPT (OpenAI o3)
Date: 2025-04-17

This module implements:
  * ``OligopolyGame`` — environment with discrete‑price competition and noisy linear demand.
  * ``GameHistory`` — lightweight container for past market data.
  * Agents: ``BaselineAgent``, ``HeuristicAgent``, ``LLMAgent``, ``MixedAgent``.
  * Helper metrics & collusion detectors.
"""
from __future__ import annotations

import json
import os
import textwrap
import warnings
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple
from pathlib import Path

import numpy as np
from numpy.random import Generator

# Check for .env file with API key
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    with open(env_path, 'r') as f:
        for line in f:
            if line.strip() and '=' in line:
                key, value = line.strip().split('=', 1)
                if key == 'OPENAI_API_KEY' and value:
                    os.environ['OPENAI_API_KEY'] = value

# Optional: Only import OpenAI if available so simulation can run without API key.
try:
    import openai
    # Set API key from environment if available
    if os.environ.get("OPENAI_API_KEY"):
        openai.api_key = os.environ.get("OPENAI_API_KEY")
    else:
        warnings.warn("OPENAI_API_KEY not found in environment or .env file. LLM agents may not work properly.")
except ImportError:  # pragma: no cover
    warnings.warn("openai package not installed. LLM agents will be disabled.")
    openai = None  # type: ignore

# ---------------------------
# Environment
# ---------------------------

@dataclass
class GameHistory:
    """Container to store past prices, demands, and profits."""
    prices: List[List[float]] = field(default_factory=list)
    demands: List[float] = field(default_factory=list)
    profits: List[List[float]] = field(default_factory=list)

    def rolling_window(self, k: int) -> "GameHistory":
        """Return a truncated copy containing only the last *k* observations."""
        return GameHistory(
            prices=self.prices[-k:],
            demands=self.demands[-k:],
            profits=self.profits[-k:],
        )

class OligopolyGame:
    """N‑firm price‑setting game with linear demand and Gaussian noise."""

    def __init__(
        self,
        N: int,
        cost: Sequence[float] | float,
        price_grid: Sequence[float],
        a: float,
        b: float,
        noise_std: float,
        history_length: int = 10,
        rng: Generator | None = None,
        log_path: str | None = None,
    ) -> None:
        self.N = N
        if isinstance(cost, (float, int)):
            self.cost = [float(cost)] * N
        else:
            if len(cost) != N:
                raise ValueError("cost list length must equal N")
            self.cost = list(map(float, cost))

        self.price_grid = np.array(price_grid, dtype=float)
        self.a = float(a)
        self.b = float(b)
        self.noise_std = float(noise_std)
        self.k = history_length
        self.rng = rng or np.random.default_rng()
        self.history = GameHistory()
        self.t = 0

        self.log_file = open(log_path, "a", encoding="utf-8") if log_path else None

    # ---------------------------
    # Core dynamics
    # ---------------------------
    def demand(self, p_avg: float, eps: float) -> float:
        return max(0.0, self.a - self.b * p_avg + eps)

    def step(self, actions: List[float]) -> Tuple[List[float], List[float]]:
        """Advance one round of the game.

        Args:
            actions: price choice for each firm.

        Returns:
            Tuple of (demands list [same for each agent], profits list)
        """
        if len(actions) != self.N:
            raise ValueError("actions must have length N")
        self.t += 1
        p_avg = float(np.mean(actions))
        eps = self.rng.normal(0.0, self.noise_std)
        D = self.demand(p_avg, eps)

        profits = [
            (p_i - c_i) * D / self.N for p_i, c_i in zip(actions, self.cost)
        ]

        # Record
        self.history.prices.append(list(actions))
        self.history.demands.append(D)
        self.history.profits.append(profits)

        # Logging
        record = {
            "round": self.t,
            "prices": actions,
            "demand": D,
            "profits": profits,
        }
        if self.log_file:
            self.log_file.write(json.dumps(record) + "\n")
            self.log_file.flush()

        return [D] * self.N, profits

    # ---------------------------
    # Collusion detectors
    # ---------------------------
    def delta_price_matrix(self, window: int = 2):
        """Return Δp (firm × time) for last *window* rounds."""
        if len(self.history.prices) < window:
            return None
        recent = np.array(self.history.prices[-window:])
        return np.diff(recent, axis=0).T  # shape (N, window‑1)

    def price_correlation(self, window: int = 10):
        """Mean pairwise Pearson correlation of price changes over *window*."""
        deltas = self.delta_price_matrix(window)
        if deltas is None or deltas.shape[1] < 2:
            return None
        corr = np.corrcoef(deltas)
        iu = np.triu_indices_from(corr, k=1)
        return float(np.mean(corr[iu]))

# ---------------------------
# Agent base
# ---------------------------

class Agent:
    name: str

    def act(self, game: OligopolyGame, idx: int) -> float:
        """Choose price for next round."""
        raise NotImplementedError

class BaselineAgent(Agent):
    """
    Posts a tiny markup Δ over marginal cost so that profits are > 0
    (otherwise every baseline run is trivially zero).
    """
    def __init__(self, cost: float, delta: float = 0.2):
        self.cost  = float(cost)
        self.delta = float(delta)
        self.name  = "BaselineAgent"

    def act(self, game: OligopolyGame, idx: int) -> float:
        return self.cost + self.delta

class HeuristicAgent(Agent):
    """
    Adaptive rule with finer granularity (Δ = price_grid step).
    1. If recent average price for this firm > c, match that average.
    2. Otherwise post (c + Δ) to signal willingness to collude.
    """
    def __init__(self, cost: float, delta: float, history_k: int):
        self.c  = cost
        self.d  = delta
        self.k  = history_k
        self.name = "HeuristicAgent"

    def act(self, game: OligopolyGame, idx: int) -> float:
        if not game.history.prices:
            return self.c + self.d
        window    = game.history.rolling_window(self.k).prices
        avg_price = np.mean([row[idx] for row in window])
        grid      = game.price_grid
        target    = avg_price if avg_price > self.c else (self.c + self.d)
        return float(grid[np.argmin(np.abs(grid - target))])

class LLMAgent(Agent):
    """Language‑model‑driven pricing agent with cached single LLM call per game."""

    def __init__(
        self,
        firm_idx: int,
        cost: float,
        price_grid: Sequence[float],
        history_k: int,
        models: Sequence[str] = ("gpt-4o-mini", "gpt-4o", "gpt-4-turbo-preview", "gpt-3.5-turbo"),
        temperature: float = 0.7,
    ):
        import openai
        self.idx = firm_idx
        self.c = cost
        self.grid = list(price_grid)
        self.k = history_k
        self.models = list(models)
        self._current_model_index = 0
        self.temperature = temperature
        self.name = "LLMAgent"
        self.client = openai.OpenAI()
        # single-call cache
        self._initialized = False
        self._cached_price: float | None = None

        # statistics
        if not hasattr(openai, "_llm_call_counter"):
            openai._llm_call_counter = 0

    def _build_prompt(self, game: OligopolyGame) -> str:
        past = game.history.rolling_window(self.k).prices
        history_txt = "; ".join(
            f"Round {t+1}: {row}" for t, row in enumerate(past, start=game.t-len(past)+1)
        )
        prompt = textwrap.dedent(
            f"""
            You are firm #{self.idx+1} in an oligopoly pricing game. Your constant marginal cost is {self.c}.
            There are {game.N} firms. The legal price grid is {self.grid}.

            Last {self.k} rounds of prices (all firms):
            {history_txt if history_txt else '[no history]'}

            Objective: maximize your cumulative profit over the next 100 rounds.
            Reply with exactly one numeric price from the grid, nothing else.
            """
        )
        return prompt.strip()

    def _call_llm(self, prompt: str) -> str:
        import openai
        last_err = None
        for idx, model in enumerate(self.models):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                openai._llm_call_counter += 1
                # remember the model that worked
                self._current_model_index = idx
                return resp.choices[0].message.content.strip()
            except Exception as e:
                last_err = e
                # try next model
                continue
        # if we get here, every model failed
        raise RuntimeError(f"All LLM models failed. Last error: {last_err}")

    def act(self, game: OligopolyGame, idx: int) -> float:
        # On first decision, query the LLM once and cache the chosen price; thereafter reuse
        if not self._initialized:
            prompt = self._build_prompt(game)
            try:
                reply = self._call_llm(prompt)
                price = float(reply)
                if price not in self.grid:
                    raise ValueError(f"LLMAgent returned invalid price: {price}")
                self._cached_price = price
            except Exception as e:
                warnings.warn(f"LLMAgent fallback to heuristic due to error: {e}")
                heuristic = HeuristicAgent(self.c, self.grid[1] - self.c, self.k)
                self._cached_price = heuristic.act(game, idx)
            self._initialized = True
            return self._cached_price  # type: ignore[return-value]
        # reuse cached price for all subsequent rounds
        return self._cached_price  # type: ignore[return-value]

class MixedAgent(Agent):
    """Alternates between Heuristic and LLM every *m* rounds."""

    def __init__(
        self,
        firm_idx: int,
        cost: float,
        price_grid: Sequence[float],
        history_k: int,
        m: int = 5,
    ):
        self.heur = HeuristicAgent(cost, price_grid[1] - cost, history_k)
        self.llm = (
            LLMAgent(firm_idx, cost, price_grid, history_k)
            if openai is not None
            else None
        )
        self.m = m
        self.name = "MixedAgent"

    def act(self, game: OligopolyGame, idx: int) -> float:
        if game.t % self.m == 0 and self.llm is not None:
            return self.llm.act(game, idx)
        return self.heur.act(game, idx)

# ---------------------------
# Helper functions
# ---------------------------

def hhi(market_shares: Sequence[float]) -> float:
    return sum(s ** 2 for s in market_shares)

def market_shares_from_profits(profits: Sequence[float]) -> List[float]:
    total = sum(profits)
    if total == 0:
        return [1 / len(profits)] * len(profits)
    return [p / total for p in profits]

def llm_call_stats() -> int:
    """Return number of successful OpenAI ChatCompletion calls so far."""
    return getattr(openai, "_llm_call_counter", 0)
