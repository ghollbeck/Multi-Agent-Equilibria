import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_logs(run_folder):
    detailed_logs = pd.read_csv(os.path.join(run_folder, "detailed_logs.csv"))
    generation_summary = pd.read_csv(os.path.join(run_folder, "generation_summary.csv"))
    return detailed_logs, generation_summary

def sort_csv(run_folder, num_cols=8):
    # Load the CSV file and extract agent numbers for sorting
    detailed_logs = pd.read_csv(os.path.join(run_folder, "detailed_logs.csv"))
    detailed_logs['Agent_A'] = detailed_logs['pair'].apply(lambda x: int(x.split('-')[0].split('_')[1]))
    detailed_logs['Agent_B'] = detailed_logs['pair'].apply(lambda x: int(x.split('-')[1].split('_')[1]))
    sorted_logs = detailed_logs.sort_values(by=['Agent_A', 'Agent_B'])
    sorted_logs = sorted_logs.drop(columns=['Agent_A', 'Agent_B'])
    sorted_logs.to_csv(os.path.join(run_folder, "sorted_detailed_logs.csv"), index=False)
    return sorted_logs

def calculate_average_cooperation(sorted_logs, pd_strategies_sorted):
    # Use a fixed set of agent numbers for an 8x8 grid
    unique_A = list(range(1, 9))
    unique_B = list(range(1, 9))
    num_rows = 8
    num_cols = 8
    # Create mapping dictionaries for agent A and agent B.
    mapping_A = {agent: agent - 1 for agent in unique_A}
    mapping_B = {agent: agent - 1 for agent in unique_B}
    
    # Create a 3D array: one score for agent A and one for agent B per grid cell
    avg_scores = np.zeros((num_rows, num_cols, 2))
    counts = np.zeros((num_rows, num_cols))
    
    # Loop over each interaction in the log.
    for _, log in sorted_logs.iterrows():
        agent_a_strategy = log['reasoning_A'].split(':')[0].strip()
        agent_b_strategy = log['reasoning_B'].split(':')[0].strip()
        # Lookup the cooperation score for each strategy.
        strategy_scores = {key: score for key, _, score in pd_strategies_sorted}
        agent_a_score = strategy_scores.get(agent_a_strategy, 0)
        agent_b_score = strategy_scores.get(agent_b_strategy, 0)
        
        # Extract the actual agent numbers and map them to the fixed grid.
        agent_a_number = int(log['pair'].split('-')[0].split('_')[1])
        agent_b_number = int(log['pair'].split('-')[1].split('_')[1])
        agent_a_mod = ((agent_a_number - 1) % 8) + 1
        agent_b_mod = ((agent_b_number - 1) % 8) + 1
        row_idx = mapping_A[agent_a_mod]
        col_idx = mapping_B[agent_b_mod]
        
        avg_scores[row_idx, col_idx, 0] += agent_a_score
        avg_scores[row_idx, col_idx, 1] += agent_b_score
        counts[row_idx, col_idx] += 1

    # Divide accumulated scores by the number of interactions per cell.
    for i in range(num_rows):
        for j in range(num_cols):
            if counts[i, j] > 0:
                avg_scores[i, j, 0] /= counts[i, j]
                avg_scores[i, j, 1] /= counts[i, j]
    return avg_scores

def plot_strategy_line(generation_summary, run_folder):
    plt.figure(figsize=(10, 6))
    # Plot each strategy column (except the 'Generation' column)
    for strategy in generation_summary.columns[1:]:
        plt.plot(generation_summary['Generation'], generation_summary[strategy], label=strategy)
    plt.title("Strategy Distribution Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Number of Agents")
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    plt.savefig(os.path.join(run_folder, "ExternalPlot_strategy_line_graph.png"))
    plt.close()

def plot_cooperation_heatmap(avg_scores, run_folder):
    # For a conventional heatmap, take the mean of agent A and B scores for each cell.
    mean_scores = np.mean(avg_scores, axis=2)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(mean_scores, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Average Cooperation Scores Heatmap")
    ax.set_xlabel("Agent B (Column)")
    ax.set_ylabel("Agent A (Row)")
    plt.savefig(os.path.join(run_folder, "ExternalPlot_cooperation_heatmap.png"))
    plt.close()

def plot_dual_heatmap(avg_scores, run_folder):
    """
    Plot a dual heatmap where each cell (of a 6x8 grid) is split into two subcells.
    The left subcell shows the average score for agent A and is colored using one colormap,
    and the right subcell shows agent B’s average score with a different colormap.
    """
    num_rows, num_cols, _ = avg_scores.shape
    fig, ax = plt.subplots(figsize=(num_cols * 1.2, num_rows * 1.2))
    # Define two different colormaps
    cmap_A = plt.get_cmap("coolwarm")
    cmap_B = plt.get_cmap("viridis")
    norm = plt.Normalize(0, 1)  # assuming scores are between 0 and 1
    
    # Loop through each cell in the grid
    for i in range(num_rows):
        for j in range(num_cols):
            score_A = avg_scores[i, j, 0]
            score_B = avg_scores[i, j, 1]
            # Set coordinates so that row 0 appears at the top.
            x = j
            y = num_rows - 1 - i
            # Draw left subcell for agent A (width=0.5)
            rect_A = plt.Rectangle((x, y), 0.5, 1, facecolor=cmap_A(norm(score_A)), edgecolor='white')
            ax.add_patch(rect_A)
            # Draw right subcell for agent B (width=0.5)
            rect_B = plt.Rectangle((x + 0.5, y), 0.5, 1, facecolor=cmap_B(norm(score_B)), edgecolor='white')
            ax.add_patch(rect_B)
            # Place text labels in the centers of the subcells
            ax.text(x + 0.25, y + 0.5, f"{score_A:.2f}", ha="center", va="center", fontsize=8, color="black")
            ax.text(x + 0.75, y + 0.5, f"{score_B:.2f}", ha="center", va="center", fontsize=8, color="black")
    
    ax.set_xlim(0, num_cols)
    ax.set_ylim(0, num_rows)
    ax.set_xticks(np.arange(0.5, num_cols, 1))
    ax.set_yticks(np.arange(0.5, num_rows, 1))
    # Label columns and rows (the row labels are reversed so row 1 appears at the top)
    ax.set_xticklabels([f"Agent {j+1}" for j in range(num_cols)])
    ax.set_yticklabels([f"Agent {i+1}" for i in range(num_rows)][::-1])
    ax.set_title("Dual Heatmap of Average Strategy Scores\n(Left: Agent A score, Right: Agent B score)")
    ax.invert_yaxis()
    plt.grid(True, which="both", color="gray", linestyle="--", linewidth=0.5)
    plt.savefig(os.path.join(run_folder, "ExternalPlot_dual_heatmap.png"))
    plt.close()

def main():
    run_folder = "Multi-Agent-Equilibria/Games/1_Prisoners_Dilemma/simulation_results/run_2025-03-24_18-34-55"
    # PD strategies sorted: (Strategy Name, Display Name, Cooperation Score)
    pd_strategies_sorted = [
        ("Generous Tit-for-Tat", "Generous Tit-for-Tat", 0.85),
        ("Tit-for-Tat", "Tit-for-Tat", 0.70),
        ("Win-Stay, Lose-Shift", "Win-Stay, Lose-Shift", 0.65),
        ("Contrite Tit-for-Tat", "Contrite Tit-for-Tat", 0.75),
        ("Always Cooperate", "Always Cooperate", 1.0),
        ("Grim Trigger", "Grim Trigger", 0.40),
        ("Suspicious Tit-for-Tat", "Suspicious Tit-for-Tat", 0.35),
        ("Always Defect", "Always Defect", 0.0)
    ]
    num_cols = 8
    num_rows = 6  # grid of 6x8 agents
    total_cells = num_rows * num_cols  # 48 agents in total

    detailed_logs, generation_summary = load_logs(run_folder)
    sorted_logs = sort_csv(run_folder, num_cols)
    
    # Calculate average strategy scores for each agent pair (each cell will have two numbers)
    avg_scores = calculate_average_cooperation(sorted_logs, pd_strategies_sorted)    
    # Plot a line graph for strategy distribution over generations
    plot_strategy_line(generation_summary, run_folder)
    
    # Plot a conventional heatmap of the average (mean of agent A and B scores)
    plot_cooperation_heatmap(avg_scores, run_folder)
    
    # Plot a dual heatmap where each cell shows two subcells (left for Agent A and right for Agent B)
    plot_dual_heatmap(avg_scores, run_folder)

if __name__ == "__main__":
    main()