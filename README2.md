# ElectOMate/AutoCreate - Script Documentation

This README provides comprehensive documentation for all scripts in the ElectOMate/AutoCreate workspace. The project contains multiple game theory simulations and utility scripts for research and experimentation.

## Table of Contents

- [Root Level Scripts](#root-level-scripts)
- [Game Simulations](#game-simulations)
  - [1. Prisoners Dilemma](#1-prisoners-dilemma)
  - [2. MIT Beer Game](#2-mit-beer-game)
  - [3. Fishery Game](#3-fishery-game)
  - [4. Market Impact Game](#4-market-impact-game)
  - [5. Oligopoly Simulation](#5-oligopoly-simulation)
  - [6. Chinese Whisper Game](#6-chinese-whisper-game)
  - [7. Chinese Whisper Game (Alternative)](#7-chinese-whisper-game-alternative)
  - [6. Security Dilemma](#6-security-dilemma)
- [Utility Scripts](#utility-scripts)
  - [PlotArchitecture](#plotarchitecture)
- [Configuration Files](#configuration-files)

---

## Root Level Scripts

### `pushing_script.sh`
**Type:** Shell Script  
**Purpose:** Git utility script for repository maintenance  
**Description:** Removes `.DS_Store` files from git tracking while keeping them locally, commits the changes, and pushes to the main branch. Useful for macOS development environments.

**Usage:**
```bash
./pushing_script.sh
```

---

## Game Simulations

### 1. Prisoners Dilemma

Located in `Games/1_Prisoners_Dilemma/`

#### `prisoner_dilemma FinalPrompt2_Free.py`
**Type:** Python Script (Executable)  
**Purpose:** Free-form prisoner's dilemma simulation with LLM agents  
**Description:** Multi-agent simulation where LLM-powered agents play prisoner's dilemma across multiple generations. Supports various LLM providers and models.

**Key Features:**
- Configurable number of agents and generations
- Multiple LLM provider support (OpenAI, others)
- Temperature control for agent decision-making
- Generation-based evolution of strategies

#### `prisoner_dilemma FinalPrompt2.py`
**Type:** Python Script (Executable)  
**Purpose:** Standard prisoner's dilemma simulation  
**Description:** Similar to the free version but with more structured constraints and prompts.

#### `prisoner_dilemma Final64.py`
**Type:** Python Script (Executable)  
**Purpose:** Optimized prisoner's dilemma simulation  
**Description:** Performance-optimized version for larger-scale experiments.

#### `plot_architecture.py`
**Type:** Python Script (Executable)  
**Purpose:** Generate architectural flowcharts for the prisoner's dilemma game  
**Description:** Creates visual representations of the game architecture and agent interaction patterns.

#### `prompts.py`
**Type:** Python Module  
**Purpose:** Prompt definitions and utilities  
**Description:** Contains predefined prompts and prompt generation utilities for LLM agents.

---

### 2. MIT Beer Game

Located in `Games/2_MIT_Beer_Game/`

#### `executeMITBeerGame.py`
**Type:** Python Script (Executable)  
**Purpose:** Main execution script for MIT Beer Game simulation  
**Description:** Runs supply chain management simulation where agents manage inventory, orders, and costs across a beer supply chain.

**Key Features:**
- Configurable generations and rounds
- Adjustable cost parameters (holding, backlog, profit)
- Agent communication capabilities
- LLM model selection
- Temperature control

**Usage:**
```python
python executeMITBeerGame.py --num_generations 3 --num_rounds_per_generation 15 --temperature 0.8
```

#### `test_communication_feature.py`
**Type:** Python Script (Testing)  
**Purpose:** Test communication features between agents  
**Description:** Validates agent-to-agent communication mechanisms in the beer game.

#### `debug_communication_logging.py`
**Type:** Python Script (Debugging)  
**Purpose:** Debug communication logging functionality  
**Description:** Diagnostic script for troubleshooting communication logging issues.

#### `plot_architecture.py`
**Type:** Python Script (Executable)  
**Purpose:** Generate architectural diagrams for the beer game  
**Description:** Creates visual flowcharts showing the beer game's architecture and agent interactions.

---

### 3. Fishery Game

Located in `Games/3_Fishery_Game/`

#### `enhanced_fishery_game.py`
**Type:** Python Script (Executable)  
**Purpose:** Main fishery resource management simulation  
**Description:** Simulates a common pool resource (fishery) game where multiple agents decide harvest amounts from a shared fish stock. Features logistic growth models and sustainability metrics.

**Key Features:**
- Asynchronous execution for performance
- Multiple agent strategy types
- Detailed logging and visualization
- Sustainability metrics analysis
- Support for OpenAI and LiteLLM providers

#### `Fishery_game.py`
**Type:** Python Script (Executable)  
**Purpose:** Basic fishery game implementation  
**Description:** Simpler version of the fishery simulation without advanced features.

#### `test_fishery_fixes.py`
**Type:** Python Script (Testing)  
**Purpose:** Test suite for fishery game bug fixes  
**Description:** Comprehensive tests to validate fishery game functionality and recent fixes.

#### `validate_fixes.py`
**Type:** Python Script (Validation)  
**Purpose:** Validation script for fishery game improvements  
**Description:** Validates that recent fixes and improvements work correctly.

#### `plot_architecture.py`
**Type:** Python Script (Executable)  
**Purpose:** Generate fishery game architecture diagrams  
**Description:** Creates visual representations of the fishery game structure.

---

### 4. Market Impact Game

Located in `Games/4_Market_Impact_Game/`

#### `market_impact_game.py`
**Type:** Python Script (Executable)  
**Purpose:** Market trading simulation with price impact  
**Description:** Simulates algorithmic trading where agent decisions affect market prices. Features market makers, price impact calculations, and profit/loss tracking.

**Key Features:**
- Configurable market parameters (price impact, volatility)
- Multiple LLM-driven trading agents
- Real-time price impact from trades
- Comprehensive trading analytics
- Support for both OpenAI and LiteLLM

**Configuration:** Key parameters can be modified in `SIMULATION_CONFIG` dictionary at the top of the file.

#### `test_market_impact_fixes.py`
**Type:** Python Script (Testing)  
**Purpose:** Test market impact game functionality  
**Description:** Tests for market simulation components and recent bug fixes.

#### `plot_architecture.py`
**Type:** Python Script (Executable)  
**Purpose:** Generate market game architecture diagrams  
**Description:** Creates visual flowcharts for the market impact game.

---

### 5. Oligopoly Simulation

Located in `Games/5_oligopoly_simulation/`

#### `oligopoly.py`
**Type:** Python Module  
**Purpose:** Core oligopoly simulation environment and agents  
**Description:** Implements N-firm price-setting competition with linear demand and Gaussian noise. Includes multiple agent types: baseline, heuristic, LLM-driven, and mixed.

**Agent Types:**
- **BaselineAgent:** Simple markup over marginal cost
- **HeuristicAgent:** Adaptive rules based on price history
- **LLMAgent:** Language model-driven pricing decisions
- **MixedAgent:** Alternates between heuristic and LLM strategies

#### `run_experiments.py`
**Type:** Python Script (Executable)  
**Purpose:** Experiment runner for oligopoly simulations  
**Description:** Executes comprehensive experiments across different market configurations, firm counts, and noise levels.

**Key Features:**
- Multiple firm counts (2, 3, 5)
- Variable noise levels
- Asymmetric cost structures
- LLM call budget management
- JSON-line logging

#### `analyze.py`
**Type:** Python Script (Analysis)  
**Purpose:** Analysis tools for oligopoly results  
**Description:** Processes experimental results and generates analytical insights.

#### `plot_architecture.py`
**Type:** Python Script (Executable)  
**Purpose:** Generate oligopoly simulation architecture diagrams  
**Description:** Creates visual representations of the oligopoly simulation structure.

---

### 6. Chinese Whisper Game

Located in `Games/6_Chinese_Whisper/`

#### `Chinese_Whisper_Game.py`
**Type:** Python Script (Executable)  
**Purpose:** Information degradation study through LLM chains  
**Description:** Simulates the "telephone game" using LLM agents to study how information degrades as it passes through chains of language models.

**Experiments:**
- **Simple Experiment:** Basic information passing through agent chain
- **Batch Experiment:** Comprehensive multi-parameter testing
- **Degradation Analysis:** Focused study on information loss patterns

**Information Types:**
- Factual data (names, dates, numbers)
- Narrative stories
- Technical instructions
- Structured data

#### `analytics.py`
**Type:** Python Module  
**Purpose:** Analytics tools for Chinese Whisper results  
**Description:** Provides analysis capabilities for information retention and degradation metrics.

#### Test Scripts:
- `tests/run_tests.py` - Main test runner
- `tests/__init__.py` - Test package initialization

---

### 7. Chinese Whisper Game (Alternative)

Located in `Games/7_Chinese_Whisper/`

Similar structure to the version 6 but with different implementation approaches or experimental configurations.

---

### 6. Security Dilemma

Located in `Games/6_Security_Dilemma/`

#### `play_security_dilemma.py`
**Type:** Python Script (Executable)  
**Purpose:** Interactive and batch security dilemma simulation  
**Description:** Comprehensive implementation of the security dilemma game with multiple strategies, configuration support, and advanced analytics.

**Key Features:**
- Multiple strategies (random, tit-for-tat, grim-trigger, Pavlov, generous tit-for-tat)
- JSON/YAML configuration support
- Batch simulation capabilities
- Advanced analytics and plotting
- LiteLLM integration
- Menu-driven CLI interface

**Strategies Available:**
- **Random:** Random cooperation/defection
- **Tit-for-Tat:** Copy opponent's last move
- **Grim Trigger:** Cooperate until opponent defects, then always defect
- **Pavlov:** Win-stay, lose-shift strategy
- **Generous Tit-for-Tat:** Forgives defection with some probability

#### `test_security_dilemma_fixes.py`
**Type:** Python Script (Testing)  
**Purpose:** Test suite for security dilemma functionality  
**Description:** Comprehensive tests for game mechanics and recent improvements.

#### `plot_architecture.py`
**Type:** Python Script (Executable)  
**Purpose:** Generate security dilemma architecture diagrams  
**Description:** Creates visual representations of the security dilemma game structure.

---

## Utility Scripts

### PlotArchitecture

Located in `Games/PlotArchitecture/`

#### `generate_architecture.sh`
**Type:** Shell Script  
**Purpose:** Bulk flowchart generator for game architectures  
**Description:** Generates architectural flowcharts for all games or a specific game. Supports customizable parameters like chunk size and model selection.

**Usage:**
```bash
# Generate for all games
./generate_architecture.sh

# Generate for specific game (1-4)
./generate_architecture.sh 2

# With custom parameters
./generate_architecture.sh -c 200 -m gpt-4o 3
```

**Supported Games:**
1. Prisoners Dilemma
2. MIT Beer Game
3. Fishery Game
4. Market Impact Game

#### `plot_architecture.py`
**Type:** Python Script  
**Purpose:** Core architecture plotting functionality  
**Description:** Main plotting script that generates flowcharts from game code.

#### `architecture_flowchart.py`
**Type:** Python Module  
**Purpose:** Flowchart generation utilities  
**Description:** Core utilities for creating architectural flowcharts from Python code.

---

## Configuration Files

### `requirements.txt`
**Purpose:** Python package dependencies  
**Description:** Lists all required Python packages for running the simulations.

### `.gitignore`
**Purpose:** Git ignore patterns  
**Description:** Specifies files and directories to ignore in version control.

### `metrics_summary.csv`
**Purpose:** Performance metrics data  
**Description:** Contains summary metrics from simulation runs (524KB, 7202 lines).

---

## Running the Scripts

### Prerequisites

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Set up API keys in environment variables:
```bash
export OPENAI_API_KEY="your-openai-key"
export LITELLM_API_KEY="your-litellm-key"  # if using LiteLLM
```

### General Usage Patterns

Most game scripts can be run directly:
```bash
cd Games/{game_directory}/
python {main_script}.py
```

Many scripts accept command-line arguments for configuration:
```bash
python script.py --help  # See available options
```

### Output

Scripts typically generate results in:
- `results/` or `{game}_results/` directories
- CSV and JSON files for data analysis
- PNG files for visualizations
- Log files for debugging

### Architecture Diagrams

Generate visual diagrams for any game:
```bash
cd Games/PlotArchitecture/
./generate_architecture.sh {game_number}
```

---

## Development Notes

- All games support LLM integration with OpenAI and/or LiteLLM
- Most simulations include comprehensive logging and analytics
- Test scripts are provided for quality assurance
- Architecture generation helps visualize code structure
- Batch processing capabilities for scientific experiments

For detailed configuration options and advanced usage, refer to individual script documentation and `--help` output.