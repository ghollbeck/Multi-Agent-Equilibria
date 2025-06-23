import os
from typing import Dict
import pandas as pd
import matplotlib.pyplot as plt


def plot_beer_game_results(rounds_df: pd.DataFrame, results_folder: str):
    """
    Basic plots: inventory over time, backlog over time, cost, etc.
    """
    os.makedirs(results_folder, exist_ok=True)

    # -------------------------------------------------------------
    # If multiple generations were run, the same round_index values
    # repeat (e.g., 1-20 for each generation). This produces multiple
    # overlapping lines in the plots, which can be mis-interpreted as
    # duplicate agents. To provide a single continuous trajectory per
    # role across the entire simulation, compute a **global round** that
    # uniquely identifies each step regardless of generation.
    # -------------------------------------------------------------

    # Determine number of rounds per generation (assume constant)
    rounds_per_gen = rounds_df["round_index"].max()
    # Create a global_round column if it doesn't already exist
    rounds_df = rounds_df.copy()
    rounds_df["global_round"] = (rounds_df["generation"] - 1) * rounds_per_gen + rounds_df["round_index"]

    # Inventory by role
    plt.figure(figsize=(10, 6))
    for role in rounds_df["role_name"].unique():
        subset = rounds_df[rounds_df["role_name"] == role]
        plt.plot(subset["global_round"], subset["inventory"], label=role)
    plt.title("Inventory Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Units in Inventory")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_folder, "inventory_over_time.png"))
    plt.close()

    # Backlog by role
    plt.figure(figsize=(10, 6))
    for role in rounds_df["role_name"].unique():
        subset = rounds_df[rounds_df["role_name"] == role]
        plt.plot(subset["global_round"], subset["backlog"], label=role)
    plt.title("Backlog Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Unmet Demand (Backlog)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_folder, "backlog_over_time.png"))
    plt.close()

    # Accumulated profit by role
    plt.figure(figsize=(10, 6))
    for role in rounds_df["role_name"].unique():
        subset = rounds_df[rounds_df["role_name"] == role]
        plt.plot(subset["global_round"], subset["profit"], label=role)
    plt.title("Accumulated Profit Over Time")
    plt.xlabel("Round")
    plt.ylabel("Accumulated Profit")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_folder, "cost_over_time.png"))
    plt.close()

    # Combined plot with subplots for Inventory, Backlog, and Profit
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))

    # Subplot 1: Inventory
    for role in rounds_df["role_name"].unique():
        subset = rounds_df[rounds_df["role_name"] == role]
        axes[0].plot(subset["global_round"], subset["inventory"], label=role)
    axes[0].set_title("Inventory Over Rounds")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Units in Inventory")
    axes[0].legend()
    axes[0].grid(True)

    # Subplot 2: Backlog
    for role in rounds_df["role_name"].unique():
        subset = rounds_df[rounds_df["role_name"] == role]
        axes[1].plot(subset["global_round"], subset["backlog"], label=role)
    axes[1].set_title("Backlog Over Rounds")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Unmet Demand (Backlog)")
    axes[1].legend()
    axes[1].grid(True)

    # Subplot 3: Profit
    for role in rounds_df["role_name"].unique():
        subset = rounds_df[rounds_df["role_name"] == role]
        axes[2].plot(subset["global_round"], subset["profit"], label=role)
    axes[2].set_title("Accumulated Profit Over Time")
    axes[2].set_xlabel("Round")
    axes[2].set_ylabel("Accumulated Profit")
    axes[2].legend()
    axes[2].grid(True)

    fig.tight_layout()
    combined_plot_path = os.path.join(results_folder, "combined_plots.png")
    plt.savefig(combined_plot_path)
    plt.close(fig)


def calculate_nash_deviation(rounds_df: pd.DataFrame, equilibrium_order: int = 10) -> Dict[str, float]:
    """
    Computes the average absolute deviation of the agent orders from the assumed Nash equilibrium order quantity.
    Returns a dictionary mapping each role to its average absolute deviation.
    """
    deviations: Dict[str, float] = {}
    for role in rounds_df["role_name"].unique():
        role_df = rounds_df[rounds_df["role_name"] == role]
        avg_deviation = (role_df["order_placed"] - equilibrium_order).abs().mean()
        deviations[role] = avg_deviation
    print(f"\nNash Equilibrium Analysis (Assumed equilibrium order = {equilibrium_order}):")
    for role, dev in deviations.items():
        print(f"Role: {role} - Average Absolute Deviation: {dev:.2f}")
    return deviations 