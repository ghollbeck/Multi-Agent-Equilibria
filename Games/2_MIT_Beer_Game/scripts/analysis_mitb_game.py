import os
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------------------------------------
# Use CMU Modern Serif for all plot text
# ---------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Computer Modern Serif", "DejaVu Serif"],
    # Ensure math text also uses Computer Modern
    "mathtext.fontset": "cm",
})


def plot_beer_game_results(rounds_df: pd.DataFrame, results_folder: str, external_demand: Optional[List[int]] = None, run_settings: Optional[Dict] = None):
    """
    Basic plots: inventory over time, backlog over time, cost, etc.
    If external_demand is provided, it will be plotted as well.
    Now supports real-time plotting with overwriting of previous plots.
    """
    os.makedirs(results_folder, exist_ok=True)

    # Handle empty dataframe case
    if rounds_df.empty:
        print("Warning: No data to plot")
        return

    # -------------------------------------------------------------
    # If multiple generations were run, the same round_index values
    # repeat (e.g., 1-20 for each generation). This produces multiple
    # overlapping lines in the plots, which can be mis-interpreted as
    # duplicate agents. To provide a single continuous trajectory per
    # role across the entire simulation, compute a **global round** that
    # uniquely identifies each step regardless of generation.
    # -------------------------------------------------------------

    # Determine number of rounds per generation (assume constant)
    rounds_per_gen = rounds_df["round_index"].max() if len(rounds_df) > 0 else 1
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
        plt.plot(subset["global_round"], subset["ending_balance"], label=role)
    plt.title("Ending Balance Over Time")
    plt.xlabel("Round")
    plt.ylabel("Bank Account Balance ($)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_folder, "balance_over_time.png"))
    plt.close()

    # Order quantities by role (stand-alone)
    plt.figure(figsize=(10, 6))
    for role in rounds_df["role_name"].unique():
        subset = rounds_df[rounds_df["role_name"] == role]
        plt.plot(subset["global_round"], subset["order_placed"], label=role)
    plt.title("Order Quantity Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Units Ordered Upstream")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_folder, "orders_over_time.png"))
    plt.close()

    # Combined plot with subplots for Inventory, Backlog, Profit, Orders, Orchestrator Advice, and External Demand
    num_subplots = 5 + (1 if external_demand else 0)
    fig, axes = plt.subplots(num_subplots, 1, figsize=(12, 6 * num_subplots))
    
    # Add run settings as text box if provided
    if run_settings:
        settings_text = "Run Settings:\n"
        for key, value in run_settings.items():
            if key != 'run_settings':  # Avoid recursive display
                settings_text += f"{key}: {value}\n"
        
        # Add text box in the top-right corner
        fig.text(0.98, 0.98, settings_text, transform=fig.transFigure, 
                fontsize=8, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

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
        axes[2].plot(subset["global_round"], subset["ending_balance"], label=role)
    axes[2].set_title("Ending Balance Over Time")
    axes[2].set_xlabel("Round")
    axes[2].set_ylabel("Balance ($)")
    axes[2].legend()
    axes[2].grid(True)

    # Subplot 4: Orders
    for role in rounds_df["role_name"].unique():
        subset = rounds_df[rounds_df["role_name"] == role]
        axes[3].plot(subset["global_round"], subset["order_placed"], label=role)
    axes[3].set_title("Order Quantity Over Rounds")
    axes[3].set_xlabel("Round")
    axes[3].set_ylabel("Units Ordered Upstream")
    axes[3].legend()
    axes[3].grid(True)

    # Subplot 5: Orchestrator Recommended Orders
    if "orchestrator_order" in rounds_df.columns:
        for role in rounds_df["role_name"].unique():
            subset = rounds_df[rounds_df["role_name"] == role]
            axes[4].plot(subset["global_round"], subset["orchestrator_order"], '--', label=role)
        axes[4].set_title("Orchestrator Recommended Orders")
        axes[4].set_xlabel("Round")
        axes[4].set_ylabel("Units")
        axes[4].legend()
        axes[4].grid(True)

    # Subplot 6: External Demand (if provided)
    if external_demand:
        ext_ax = axes[5] if "orchestrator_order" in rounds_df.columns else axes[4]
        rounds = list(range(len(external_demand)))
        ext_ax.plot(rounds, external_demand, 'ko-', linewidth=2, markersize=6, label="External Demand")
        ext_ax.set_title("External Customer Demand")
        ext_ax.set_xlabel("Round")
        ext_ax.set_ylabel("Demand (Units)")
        ext_ax.legend()
        ext_ax.grid(True)
        # Add average demand line
        avg_demand = sum(external_demand) / len(external_demand)
        ext_ax.axhline(y=avg_demand, color='r', linestyle='--', alpha=0.5, label=f"Average: {avg_demand:.1f}")
        ext_ax.legend()

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