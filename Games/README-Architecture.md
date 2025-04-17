# Architecture Flowchart Generation System

This system generates Mermaid flowcharts for all games in the Multi-Agent-Equilibria project. It uses LiteLLM to analyze Python code and create a visual representation of the code structure.

## System Architecture

The system consists of:

1. **Main External Script** (`architecture_flowchart.py`): Located in the Games directory, this is the central script that handles the LLM calls and flowchart generation.

2. **Game-Specific Scripts** (`plot_architecture.py`): Each game folder contains a script that imports the main script and applies it to the specific game file.

3. **Shell Script** (`generate_architecture.sh`): A convenient shell script to run the generation for one or all games.

## Directory Structure

```
Multi-Agent-Equilibria/
└── Games/
    ├── architecture_flowchart.py   # Main external script
    ├── generate_architecture.sh    # Shell script for running all
    ├── 1_Prisoners_Dilemma/
    │   ├── plot_architecture.py    # Internal script for this game
    │   └── architecture/           # Generated flowcharts (created automatically)
    ├── 2_MIT_Beer_Game/
    │   ├── plot_architecture.py
    │   └── architecture/
    ├── 3_Fishery_Game/
    │   ├── plot_architecture.py
    │   └── architecture/
    └── 4_Market_Impact_Game/
        ├── plot_architecture.py
        └── architecture/
```

## How to Use

### Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install requests python-dotenv
```

You need to set your LiteLLM API key in your environment variables or a `.env` file:

```
LITELLM_API_KEY=your_api_key_here
```

### Option 1: Run for a Specific Game

To generate a flowchart for a specific game (e.g., Market Impact Game):

```bash
cd Multi-Agent-Equilibria/Games/4_Market_Impact_Game
python plot_architecture.py
```

### Option 2: Run for All Games Using the Shell Script

```bash
cd Multi-Agent-Equilibria/Games
./generate_architecture.sh
```

### Option 3: Run for a Specific Game Using the Shell Script

```bash
cd Multi-Agent-Equilibria/Games
./generate_architecture.sh 4  # For Market Impact Game (4)
```

## Generated Files

For each game, the system creates:

1. `<game_name>_llm_output.txt`: The raw output from the LLM
2. `<game_name>_flowchart.mmd`: The Mermaid flowchart code
3. `<game_name>_flowchart.html`: An HTML file that renders the flowchart in a browser

## Viewing the Flowcharts

The easiest way to view a flowchart is to open the generated HTML file in a web browser. Alternatively, you can copy the contents of the `.mmd` file to an online Mermaid editor like [Mermaid Live Editor](https://mermaid.live/).

## How It Works

1. The internal script in a game directory identifies the main game file.
2. It calls the external script with the path to the game file.
3. The external script reads the file content and sends it to LiteLLM with a prompt asking for a Mermaid flowchart.
4. The response is processed, and the Mermaid code is extracted.
5. The code is saved as a `.mmd` file and also embedded in an HTML template for easy viewing.

## Customizing

If you need to analyze a different file in a game directory, you can modify the `game_file` variable in the internal `plot_architecture.py` script for that game.