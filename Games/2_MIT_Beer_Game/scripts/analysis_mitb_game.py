import os
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

# ---------------------------------------------
# Use reliable fonts for all plot text
# ---------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Helvetica", "sans-serif"],
    "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": False,  # Use ASCII minus instead of Unicode minus
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

    # Define distinct line styles and markers for each role
    role_styles = {
        'Retailer': {'linestyle': '-', 'marker': 'o', 'markersize': 4, 'alpha': 0.8, 'linewidth': 2.5},
        'Wholesaler': {'linestyle': '--', 'marker': 's', 'markersize': 4, 'alpha': 0.8, 'linewidth': 2.5},
        'Distributor': {'linestyle': '-.', 'marker': '^', 'markersize': 4, 'alpha': 0.8, 'linewidth': 2.5},
        'Factory': {'linestyle': ':', 'marker': 'D', 'markersize': 4, 'alpha': 0.8, 'linewidth': 3}
    }

    # Inventory by role
    plt.figure(figsize=(10, 6))
    for role in rounds_df["role_name"].unique():
        subset = rounds_df[rounds_df["role_name"] == role]
        style = role_styles.get(role, {'linestyle': '-', 'marker': 'o', 'markersize': 4, 'alpha': 0.8, 'linewidth': 2})
        plt.plot(subset["global_round"], subset["inventory"], label=role, **style)
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
        style = role_styles.get(role, {'linestyle': '-', 'marker': 'o', 'markersize': 4, 'alpha': 0.8, 'linewidth': 2})
        plt.plot(subset["global_round"], subset["backlog"], label=role, **style)
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
        style = role_styles.get(role, {'linestyle': '-', 'marker': 'o', 'markersize': 4, 'alpha': 0.8, 'linewidth': 2})
        plt.plot(subset["global_round"], subset["ending_balance"], label=role, **style)
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
        style = role_styles.get(role, {'linestyle': '-', 'marker': 'o', 'markersize': 4, 'alpha': 0.8, 'linewidth': 2})
        plt.plot(subset["global_round"], subset["order_placed"], label=role, **style)
    plt.title("Order Quantity Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Units Ordered Upstream")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_folder, "orders_over_time.png"))
    plt.close()

    # Shipments sent downstream by role (stand-alone)
    plt.figure(figsize=(10, 6))
    for role in rounds_df["role_name"].unique():
        subset = rounds_df[rounds_df["role_name"] == role]
        style = role_styles.get(role, {'linestyle': '-', 'marker': 'o', 'markersize': 4, 'alpha': 0.8, 'linewidth': 2})
        if "shipment_sent_downstream" in subset.columns:
            plt.plot(subset["global_round"], subset["shipment_sent_downstream"], label=role, **style)
    plt.title("Shipments Sent Downstream Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Units Shipped Downstream")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_folder, "shipments_over_time.png"))
    plt.close()

    # Combined plot with subplots for Inventory, Backlog, Profit, Orders, Shipments, Orchestrator Advice, and External Demand
    num_subplots = 6 + (1 if external_demand else 0)
    fig, axes = plt.subplots(num_subplots, 1, figsize=(12, 6 * num_subplots))
    fig.subplots_adjust(hspace=2.0)  # Increased vertical spacing between subplots for better readability

    title_fontsize = 14  # 10% larger than default (usually 12)
    label_fontsize = 12  # 10% larger than default (usually 10-11)

    # Subplot 1: Inventory
    for role in rounds_df["role_name"].unique():
        subset = rounds_df[rounds_df["role_name"] == role]
        style = role_styles.get(role, {'linestyle': '-', 'marker': 'o', 'markersize': 4, 'alpha': 0.8, 'linewidth': 2})
        axes[0].plot(subset["global_round"], subset["inventory"], label=role, **style)
    axes[0].set_title("Inventory Over Rounds", fontsize=title_fontsize, fontweight='bold')
    axes[0].set_xlabel("Round", fontsize=label_fontsize)
    axes[0].set_ylabel("Units in Inventory", fontsize=label_fontsize)
    axes[0].legend()
    axes[0].grid(True)

    # Subplot 2: Backlog
    for role in rounds_df["role_name"].unique():
        subset = rounds_df[rounds_df["role_name"] == role]
        style = role_styles.get(role, {'linestyle': '-', 'marker': 'o', 'markersize': 4, 'alpha': 0.8, 'linewidth': 2})
        axes[1].plot(subset["global_round"], subset["backlog"], label=role, **style)
    axes[1].set_title("Backlog Over Rounds", fontsize=title_fontsize, fontweight='bold')
    axes[1].set_xlabel("Round", fontsize=label_fontsize)
    axes[1].set_ylabel("Unmet Demand (Backlog)", fontsize=label_fontsize)
    axes[1].legend()
    axes[1].grid(True)

    # Subplot 3: Profit
    for role in rounds_df["role_name"].unique():
        subset = rounds_df[rounds_df["role_name"] == role]
        style = role_styles.get(role, {'linestyle': '-', 'marker': 'o', 'markersize': 4, 'alpha': 0.8, 'linewidth': 2})
        axes[2].plot(subset["global_round"], subset["ending_balance"], label=role, **style)
    axes[2].set_title("Ending Balance Over Time", fontsize=title_fontsize, fontweight='bold')
    axes[2].set_xlabel("Round", fontsize=label_fontsize)
    axes[2].set_ylabel("Balance ($)", fontsize=label_fontsize)
    axes[2].legend()
    axes[2].grid(True)

    # Subplot 4: Orders
    for role in rounds_df["role_name"].unique():
        subset = rounds_df[rounds_df["role_name"] == role]
        style = role_styles.get(role, {'linestyle': '-', 'marker': 'o', 'markersize': 4, 'alpha': 0.8, 'linewidth': 2})
        axes[3].plot(subset["global_round"], subset["order_placed"], label=role, **style)
    axes[3].set_title("Order Quantity Over Rounds", fontsize=title_fontsize, fontweight='bold')
    axes[3].set_xlabel("Round", fontsize=label_fontsize)
    axes[3].set_ylabel("Units Ordered Upstream", fontsize=label_fontsize)
    axes[3].legend()
    axes[3].grid(True)

    # Subplot 5: Shipments Sent Downstream
    for role in rounds_df["role_name"].unique():
        subset = rounds_df[rounds_df["role_name"] == role]
        style = role_styles.get(role, {'linestyle': '-', 'marker': 'o', 'markersize': 4, 'alpha': 0.8, 'linewidth': 2})
        if "shipment_sent_downstream" in subset.columns:
            axes[4].plot(subset["global_round"], subset["shipment_sent_downstream"], label=role, **style)
    axes[4].set_title("Shipments Sent Downstream Over Rounds", fontsize=title_fontsize, fontweight='bold')
    axes[4].set_xlabel("Round", fontsize=label_fontsize)
    axes[4].set_ylabel("Units Shipped Downstream", fontsize=label_fontsize)
    axes[4].legend()
    axes[4].grid(True)

    # Subplot 6: Orchestrator Recommended Orders (if present)
    if "orchestrator_order" in rounds_df.columns:
        for role in rounds_df["role_name"].unique():
            subset = rounds_df[rounds_df["role_name"] == role]
            style = role_styles.get(role, {'linestyle': '-', 'marker': 'o', 'markersize': 4, 'alpha': 0.8, 'linewidth': 2}).copy()
            style['linestyle'] = '--'  # Override to dashed for orchestrator
            style['alpha'] = 0.6  # Make slightly more transparent
            axes[5].plot(subset["global_round"], subset["orchestrator_order"], label=role, **style)
        axes[5].set_title("Orchestrator Recommended Orders", fontsize=title_fontsize, fontweight='bold')
        axes[5].set_xlabel("Round", fontsize=label_fontsize)
        axes[5].set_ylabel("Units", fontsize=label_fontsize)
        axes[5].legend()
        axes[5].grid(True)

    # Subplot 7: External Demand (if provided)
    if external_demand:
        ext_ax = axes[6] if "orchestrator_order" in rounds_df.columns else axes[5]
        rounds = list(range(len(external_demand)))
        ext_ax.plot(rounds, external_demand, 'ko-', linewidth=2, markersize=6, label="External Demand")
        ext_ax.set_title("External Customer Demand", fontsize=title_fontsize, fontweight='bold')
        ext_ax.set_xlabel("Round", fontsize=label_fontsize)
        ext_ax.set_ylabel("Demand (Units)", fontsize=label_fontsize)
        ext_ax.legend()
        ext_ax.grid(True)
        avg_demand = sum(external_demand) / len(external_demand)
        ext_ax.axhline(y=avg_demand, color='r', linestyle='--', alpha=0.5, label=f"Average: {avg_demand:.1f}")
        ext_ax.legend()

    fig.tight_layout()
    
    # Add run settings as subtitle at the bottom if provided
    if run_settings:
        # Define default values to identify which parameters were explicitly set
        defaults = {
            'num_rounds': 10,
            'temperature': 0.7,
            'sale_price_per_unit': 5.0,
            'purchase_cost_per_unit': 2.5,
            'production_cost_per_unit': 1.5,
            'holding_cost_per_unit': 0.5,
            'backlog_cost_per_unit': 1.5,
            'model_name': 'gpt-3.5-turbo-1106',  # Default model name
            'enable_communication': False,
            'communication_rounds': 2,
            'enable_memory': False,
            'memory_retention_rounds': 5,
            'enable_orchestrator': False,
            'orchestrator_history': 3,
            'orchestrator_override': False,
            'initial_inventory': 100,
            'initial_backlog': 0,
            'initial_balance': 1000.0,
            'longtermplanning_boolean': False
        }
        
        # Only bold parameters that were explicitly specified in run_settings and differ from default
        param_pairs = []
        
        # Always show model name first and prominently (bolded)
        if 'model_name' in run_settings and run_settings['model_name']:
            param_pairs.append(f"**model={run_settings['model_name']}**")
        
        for key, value in run_settings.items():
            if key != 'run_settings' and key != 'model_name':  # Avoid recursive display and skip model_name (already added)
                # Check if this parameter differs from default
                is_non_default = key in defaults and value != defaults[key]
                # Only bold if it was explicitly specified (i.e., present in run_settings and not default)
                if is_non_default:
                    param_pairs.append(f"**{key}={value}**")
                else:
                    param_pairs.append(f"{key}={value}")
        
        # Format parameters into two columns, left-aligned
        n = len(param_pairs)
        n_rows = math.ceil(n / 2)
        col1 = param_pairs[:n_rows]
        col2 = param_pairs[n_rows:]
        # Pad columns to equal length
        if len(col2) < n_rows:
            col2 += [''] * (n_rows - len(col2))
        # Compute max width for left column (without markdown)
        max_col1_width = max((len(p.replace('**','')) for p in col1), default=0)
        # Build lines with left alignment
        settings_lines = []
        for left, right in zip(col1, col2):
            left_clean = left.replace('**','')
            pad = ' ' * (max_col1_width - len(left_clean) + 2)
            line = f"{left}{pad}{right}"
            settings_lines.append(line)
        settings_text = "\n".join(settings_lines)
        # Calculate required bottom margin based on number of lines
        num_lines = len(settings_lines)
        bottom_margin = max(0.12, 0.04 * num_lines + 0.06)
        # Add as subtitle at the bottom with better positioning
        fig.suptitle(f"MIT Beer Game Simulation Results\n{settings_text}", 
                    fontsize=9, y=bottom_margin/2, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        # Adjust layout to make room for subtitle
        fig.subplots_adjust(bottom=bottom_margin)
    
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