import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the log files
def load_logs(run_folder):
    detailed_logs = pd.read_csv(os.path.join(run_folder, "detailed_logs.csv"))
    generation_summary = pd.read_csv(os.path.join(run_folder, "generation_summary.csv"))
    return detailed_logs, generation_summary

# Calculate average cooperation scores
def calculate_average_cooperation(detailed_logs, pd_strategies_sorted):
    strategy_scores = {key: score for key, _, score in pd_strategies_sorted}
    cooperation_scores = np.zeros((8, 8, 2))  # 8x8 grid with two scores per cell

    for _, log in detailed_logs.iterrows():
        agent_a_strategy = log['reasoning_A'].split(':')[0].strip()
        agent_b_strategy = log['reasoning_B'].split(':')[0].strip()
        agent_a_score = strategy_scores.get(agent_a_strategy, 0)
        agent_b_score = strategy_scores.get(agent_b_strategy, 0)

        # Calculate indices for the grid
        agent_a_index = int(log['pair'].split('-')[0].split('_')[1]) - 1
        agent_b_index = int(log['pair'].split('-')[1].split('_')[1]) - 1

        # Update the scores
        cooperation_scores[agent_a_index // 8, agent_b_index % 8, 0] += agent_a_score
        cooperation_scores[agent_a_index // 8, agent_b_index % 8, 1] += agent_b_score

    # Average the scores
    cooperation_scores /= len(detailed_logs) / 64
    return cooperation_scores

# Plot the line graph of strategies
def plot_strategy_line(generation_summary, run_folder):
    plt.figure(figsize=(10, 6))
    for strategy in generation_summary.columns[1:]:
        plt.plot(generation_summary['Generation'], generation_summary[strategy], label=strategy)
    plt.title("Strategy Distribution Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Number of Agents")
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    plt.savefig(os.path.join(run_folder, "ExternalPlot_strategy_line_graph.png"))
    plt.close()

# Plot the heatmap of average cooperation scores
def plot_cooperation_heatmap(cooperation_scores, run_folder):
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(np.mean(cooperation_scores, axis=2), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Average Cooperation Scores Heatmap")
    ax.set_xlabel("Agent B")
    ax.set_ylabel("Agent A")
    plt.savefig(os.path.join(run_folder, "ExternalPlot_cooperation_heatmap.png"))
    plt.close()

def sort_csv(run_folder):
    # Load the CSV file
    detailed_logs = pd.read_csv(os.path.join(run_folder, "detailed_logs.csv"))
    
    # Extract agent numbers and sort
    detailed_logs['Agent_A'] = detailed_logs['pair'].apply(lambda x: int(x.split('-')[0].split('_')[1]))
    detailed_logs['Agent_B'] = detailed_logs['pair'].apply(lambda x: int(x.split('-')[1].split('_')[1]))
    
    # Sort by Agent_A and then by Agent_B
    sorted_logs = detailed_logs.sort_values(by=['Agent_A', 'Agent_B'])
    
    # Drop the temporary columns
    sorted_logs = sorted_logs.drop(columns=['Agent_A', 'Agent_B'])
    
    # Save the sorted CSV
    sorted_logs.to_csv(os.path.join(run_folder, "sorted_detailed_logs.csv"), index=False)

    return sorted_logs

# Main function to execute the script
def main():
    run_folder = "Multi-Agent-Equilibria/Games/1_Prisoners_Dilemma/simulation_results/run_2025-03-24_18-34-55"
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

    detailed_logs, generation_summary = load_logs(run_folder)
    # Sort the CSV and save it
    sorted_logs = sort_csv(run_folder)

    # Use the sorted logs for further processing
    cooperation_scores = calculate_average_cooperation(sorted_logs, pd_strategies_sorted)
    plot_strategy_line(generation_summary, run_folder)
    plot_cooperation_heatmap(cooperation_scores, run_folder)

if __name__ == "__main__":
    main()