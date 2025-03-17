# New Cell: Plot Flowchart of the Simulation Process

import graphviz

dot_code = """
digraph Flowchart {
    rankdir=TB;
    node [shape=box, style=rounded];
    
    A [label="Start: Initialize Environment"];
    B [label="Import Libraries & Setup OpenAI Client"];
    C [label="Define EnhancedAgent Class"];
    D [label="Generate Strategy Matrix via LLM Call"];
    E [label="Decide Next Action via LLM Call\n(Uses history & strategy)"];
    F [label="Log Interaction\n(Store opponent, actions, payoff)"];
    G [label="Define Payoff Matrix"];
    H [label="Create Enhanced Agents\n(Initial Population)"];
    I [label="Begin Simulation Loop\n(For each Generation)"];
    J [label="Shuffle Agents"];
    K [label="Pair Agents for Interaction"];
    L [label="For each Pair:\n- Agents decide actions\n- Compute payoffs\n- Update scores\n- Log interactions"];
    M [label="Collect Detailed Logs\nfor Generation"];
    N [label="Sort Agents by Total Score"];
    O [label="Select Top Agents\n(Survivors)"];
    P [label="Generate New Agents\n(Reproduction/Mutation)"];
    Q [label="Combine Survivors & New Agents"];
    R [label="Reset Scores\nfor Next Generation"];
    S [label="End Simulation Loop"];
    T [label="Save Logs (CSV & JSON)\nin Subfolder"];
    U [label="Simulation Completed"];
    
    A -> B;
    B -> C;
    C -> D;
    C -> E;
    E -> F;
    F -> G;
    G -> H;
    H -> I;
    I -> J;
    J -> K;
    K -> L;
    L -> M;
    M -> N;
    N -> O;
    O -> P;
    P -> Q;
    Q -> R;
    R -> I [label="Next Generation"];
    I -> S [label="After final generation"];
    S -> T;
    T -> U;
}
"""

# Create and render the flowchart
flowchart = graphviz.Source(dot_code)
flowchart.render("flowchart", format="png", cleanup=True)  # This saves the flowchart as flowchart.png
flowchart  # Display inline (in Jupyter Notebook)