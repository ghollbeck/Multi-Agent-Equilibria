#!/usr/bin/env python3
"""Verify that orchestrator recommendations override agent decisions when orchestrator_override=True.
The test monkey-patches a dummy orchestrator that always recommends order_quantity = 15 for all roles.
It also patches BeerGameAgent.llm_decision so that no external LLM call is made.
The CSV log produced by run_beer_game_generation is examined to confirm that every `order_placed`
entry equals the orchestrator recommendation.
"""
import asyncio
import os
import sys
import types
import tempfile
import shutil
import csv

import pandas as pd

# Add scripts directory to sys.path so imports work when test is executed via pytest from project root
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(scripts_dir)

from MIT_Beer_Game import run_beer_game_generation  # noqa: E402
from models_mitb_game import BeerGameAgent, BeerGameLogger  # noqa: E402


class DummyOrchestrator:  # pylint: disable=too-few-public-methods
    """Simple stub that returns identical recommendations for all agents."""

    def __init__(self, history_window=3, logger=None):
        self.history_window = history_window
        self.logger = logger

    async def get_recommendations(self, *, agents, **_kwargs):  # noqa: D401, N803
        return {
            agent.role_name: {
                "order_quantity": 15,
                "rationale": "dummy-recommendation",
            }
            for agent in agents
        }


async def _run_sim(temp_dir: str):  # pylint: disable=too-many-locals
    """One-round generation with orchestrator override enabled."""
    # Patch orchestrator module *before* the function imports it internally
    orchestrator_mod = types.ModuleType("orchestrator_mitb_game")
    orchestrator_mod.BeerGameOrchestrator = DummyOrchestrator
    sys.modules["orchestrator_mitb_game"] = orchestrator_mod

    # Patch BeerGameAgent.llm_decision so no external call occurs
    async def _dummy_llm(self, *_, **__):  # noqa: D401, N802
        return {"order_quantity": 3}

    BeerGameAgent.llm_decision = _dummy_llm  # type: ignore

    from models_mitb_game import SimulationData  # noqa: E402

    logger = BeerGameLogger()
    roles = ["Retailer", "Wholesaler", "Distributor", "Factory"]
    agents = [
        BeerGameAgent.create_agent(role_name=r, logger=logger, human_log_file=None)
        for r in roles
    ]
    csv_path = os.path.join(temp_dir, "log.csv")
    sim_data = SimulationData(hyperparameters={})

    external_demand = [10]  # one round
    await run_beer_game_generation(
        agents=agents,
        external_demand=external_demand,
        num_rounds=1,
        holding_cost_per_unit=0.5,
        backlog_cost_per_unit=1.5,
        sale_price_per_unit=5.0,
        purchase_cost_per_unit=2.5,
        production_cost_per_unit=1.5,
        csv_log_path=csv_path,
        json_log_path=None,
        logger=logger,
        sim_data=sim_data,
        human_log_file=None,
        enable_communication=False,
        enable_memory=False,
        enable_shared_memory=False,
        enable_orchestrator=True,
        orchestrator_history=3,
        orchestrator_override=True,
    )
    return csv_path


def test_orchestrator_override_takes_effect():  # noqa: D401
    temp_dir = tempfile.mkdtemp(prefix="orch_override_test_")
    try:
        csv_path = asyncio.run(_run_sim(temp_dir))
        assert os.path.exists(csv_path), "CSV log not created"

        with open(csv_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

        # Filter rows for round_index==1 (first real round)
        first_round = [row for row in rows if row["round_index"] == "1"]
        assert first_round, "No round 1 data in CSV"

        for row in first_round:
            assert int(row["order_placed"]) == 15, (
                "Orchestrator override did not set order_placed to 15"
            )
            assert int(row["orchestrator_order"]) == 15, (
                "orchestrator_order column mismatch"
            )
    finally:
        shutil.rmtree(temp_dir) 