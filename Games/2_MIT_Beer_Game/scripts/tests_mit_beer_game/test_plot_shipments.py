#!/usr/bin/env python3
"""Test that the new shipments plot and combined plot are generated correctly."""
import os
import sys
import tempfile
import shutil

import pandas as pd

# Ensure the parent directory (scripts) is in the path so we can import analysis_mitb_game
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from analysis_mitb_game import plot_beer_game_results  # noqa: E402


def test_shipments_plot_generated():
    """Create a minimal rounds_df and verify that the shipments plot files are saved."""
    # Synthetic simulation data for two roles across three rounds
    data = []
    roles = ["Retailer", "Wholesaler"]
    for role in roles:
        for rnd in range(3):
            data.append({
                "generation": 1,
                "round_index": rnd + 1,
                "role_name": role,
                "inventory": 100 - rnd,
                "backlog": rnd,
                "order_placed": 10 + rnd,
                "shipment_sent_downstream": 8 + rnd,
                "ending_balance": 1000 - rnd * 10,
            })

    rounds_df = pd.DataFrame(data)

    # Temporary directory to store plots
    results_dir = tempfile.mkdtemp(prefix="beer_game_test_")
    try:
        plot_beer_game_results(rounds_df, results_dir)
        shipments_path = os.path.join(results_dir, "shipments_over_time.png")
        combined_path = os.path.join(results_dir, "combined_plots.png")

        assert os.path.exists(shipments_path), "Shipments plot was not created."
        assert os.path.exists(combined_path), "Combined plot was not created."
    finally:
        # Clean up temporary directory
        shutil.rmtree(results_dir) 