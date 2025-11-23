#!/usr/bin/env python3
"""Ensure orders placed by each role arrive to their supplier in the next round.
Checks `order_placed` (round t) vs `order_received` (round t+1) in the CSV log.
"""
import asyncio
import os
import sys
import types
import tempfile
import shutil
import csv

# Path handling so imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(scripts_dir)

from MIT_Beer_Game import run_beer_game_generation  # noqa: E402
from models_mitb_game import BeerGameAgent, BeerGameLogger, SimulationData  # noqa: E402


# Pre-defined constant orders per role to make test assertions easy
ROLE_ORDERS = {
    "Retailer": 20,
    "Wholesaler": 15,
    "Distributor": 10,
    "Factory": 0,  # not used but must exist
}


async def _run_sim(temp_dir: str):
    """Two-round generation with deterministic orders."""

    # Patch LLM call
    async def _dummy_llm(self, *_, **__):  # noqa: D401, N802
        return {"order_quantity": ROLE_ORDERS[self.role_name]}

    BeerGameAgent.llm_decision = _dummy_llm  # type: ignore

    logger = BeerGameLogger()
    roles = ["Retailer", "Wholesaler", "Distributor", "Factory"]
    agents = [BeerGameAgent.create_agent(role_name=r, logger=logger) for r in roles]

    csv_path = os.path.join(temp_dir, "orders_pipeline.csv")
    sim_data = SimulationData(hyperparameters={})

    await run_beer_game_generation(
        agents=agents,
        external_demand=[10, 10],
        num_rounds=2,
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
        enable_orchestrator=False,
    )
    return csv_path


def test_orders_flow_downstream():  # noqa: D401
    temp_dir = tempfile.mkdtemp(prefix="orders_pipe_test_")
    try:
        csv_path = asyncio.run(_run_sim(temp_dir))
        assert os.path.exists(csv_path), "CSV log not created"

        with open(csv_path, newline="") as csvfile:
            reader = list(csv.DictReader(csvfile))

        # Build helper dict: {(role, round): row}
        rows = { (row["role_name"], int(row["round_index"])): row for row in reader }

        # For each supplier (except Retailer) check that order_received at round 2 == order_placed of customer at round 1
        mapping = {
            "Wholesaler": "Retailer",
            "Distributor": "Wholesaler",
            "Factory": "Distributor",
        }
        for supplier, customer in mapping.items():
            placed = int(rows[(customer, 1)]["order_placed"])
            received = int(rows[(supplier, 2)]["order_received"])
            assert received == placed, (
                f"{supplier} should receive {placed} from {customer} in round 2, got {received}"
            )
    finally:
        shutil.rmtree(temp_dir) 