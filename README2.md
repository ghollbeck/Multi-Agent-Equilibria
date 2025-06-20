# README2.md - ElectOMate-Frontend Scripts Documentation

This document provides comprehensive explanations of all scripts in the ElectOMate-Frontend project, organized by category and purpose.

## Table of Contents
- [Development Scripts](#development-scripts)
- [Game Scripts](#game-scripts)
  - [1. Prisoners Dilemma](#1-prisoners-dilemma)
  - [2. MIT Beer Game](#2-mit-beer-game)
  - [3. Fishery Game](#3-fishery-game)
  - [4. Market Impact Game](#4-market-impact-game)
  - [5. Oligopoly Simulation](#5-oligopoly-simulation)
  - [6. Chinese Whisper Game](#6-chinese-whisper-game)
  - [7. Security Dilemma](#7-security-dilemma)
- [Architecture and Visualization Scripts](#architecture-and-visualization-scripts)
- [Requirements and Dependencies](#requirements-and-dependencies)

---

## Development Scripts

### `pushing_script.sh`
**Purpose**: Git repository maintenance script  
**Description**: Removes `.DS_Store` files from git tracking (macOS system files) while keeping them locally, commits the changes, and pushes to the main branch.  
**Usage**: `./pushing_script.sh`

---

## Game Scripts

### 1. Prisoners Dilemma

**Location**: `Games/1_Prisoners_Dilemma/`

#### Main Game Scripts
- **`PD_GameFinal.py`**: Final version of the Prisoners Dilemma game implementation
- **`prisoner_dilemma Final64.py`**: Extended version with 64-bit implementation features
- **`prisoner_dilemma FinalPrompt2.py`**: Version with enhanced prompt engineering
- **`prisoner_dilemma FinalPrompt2_Free.py`**: Free-play version without constraints

#### Configuration and Prompts
- **`prompts.py`**: Contains all prompt templates for the Prisoners Dilemma game
- **`prompts_free.py`**: Prompt templates for the free-play version

#### Visualization
- **`plot_architecture.py`**: Generates architecture diagrams for the Prisoners Dilemma system
- **`flowchart.png`**: Visual flowchart of the game logic

#### Notebook
- **`1_Prisoners_Dilemma.ipynb`**: Jupyter notebook with analysis and experimentation

---

### 2. MIT Beer Game

**Location**: `Games/2_MIT_Beer_Game/`

#### Main Execution Scripts
- **`executeMITBeerGame.py`**: Main execution script with command-line arguments for running MIT Beer Game simulations
- **`MIT_Beer_Game.py`**: Core game implementation

#### Core Modules
- **`llm_calls_mitb_game.py`**: Handles all LLM (Large Language Model) calls and API interactions
- **`models_mitb_game.py`**: Data models and structures for the MIT Beer Game
- **`prompts_mitb_game.py`**: Prompt templates and engineering for agent interactions
- **`analysis_mitb_game.py`**: Analysis and metrics calculation for game results

#### Testing and Debugging
- **`test_communication_feature.py`**: Tests the communication features between agents
- **`debug_communication_logging.py`**: Debugging utilities for communication logging

#### Visualization
- **`plot_architecture.py`**: Creates architecture diagrams for the MIT Beer Game

#### Documentation
- **`ReadMe_MIT_BeerGame.md`**: Specific documentation for the MIT Beer Game

---

### 3. Fishery Game

**Location**: `Games/3_Fishery_Game/`

#### Main Game Scripts
- **`enhanced_fishery_game.py`**: Enhanced version of the fishery game with additional features
- **`Fishery_game.py`**: Core fishery game implementation

#### Testing and Validation
- **`test_fishery_fixes.py`**: Test suite for fishery game bug fixes and features
- **`validate_fixes.py`**: Validation scripts for ensuring fixes work correctly

#### Visualization
- **`plot_architecture.py`**: Architecture diagram generation for the Fishery Game

#### Documentation
- **`ReadMe_3_Fishery_Game.md`**: Specific documentation for the Fishery Game

---

### 4. Market Impact Game

**Location**: `Games/4_Market_Impact_Game/`

#### Main Game Scripts
- **`market_impact_game.py`**: Main implementation of the market impact simulation game

#### Testing
- **`test_market_impact_fixes.py`**: Test suite for market impact game functionality

#### Visualization
- **`plot_architecture.py`**: Generates architecture diagrams for the Market Impact Game

#### Documentation
- **`Readme_4_Market_impact.md`**: Specific documentation for the Market Impact Game

---

### 5. Oligopoly Simulation

**Location**: `Games/5_oligopoly_simulation/`

#### Main Execution Scripts
- **`run_experiments.py`**: Main experimental runner that executes all treatments and logs results
- **`oligopoly.py`**: Core oligopoly game implementation with different agent types

#### Analysis Scripts
- **`analyze.py`**: Analysis tools for processing simulation results

#### Visualization
- **`plot_architecture.py`**: Architecture diagram generation for the Oligopoly Simulation

#### Data Files
- **`metrics_summary.csv`**: Large dataset containing experimental results and metrics

#### Documentation
- **`README.md`**: Detailed documentation for the oligopoly simulation

---

### 6. Chinese Whisper Game

**Location**: `Games/6_Chinese_Whisper/` and `Games/7_Chinese_Whisper/`

#### Main Game Scripts
- **`Chinese_Whisper_Game.py`**: Core Chinese Whisper game implementation
- **`game_controller.py`**: Game flow control and coordination
- **`information_generator.py`**: Generates information and content for the whisper chain

#### LLM Integration
- **`llm_calls_chinese_whisper.py`**: Handles LLM API calls for the Chinese Whisper game
- **`models_chinese_whisper.py`**: Data models for the Chinese Whisper game
- **`prompts_chinese_whisper.py`**: Prompt templates for agent interactions

#### Analysis and Evaluation
- **`evaluation_engine.py`**: Evaluates game performance and agent behavior
- **`analytics.py`**: Advanced analytics and metrics calculation

---

### 7. Security Dilemma

**Location**: `Games/6_Security_Dilemma/`

#### Main Game Scripts
- **`play_security_dilemma.py`**: Main implementation of the security dilemma game

#### Testing
- **`test_security_dilemma_fixes.py`**: Test suite for security dilemma functionality

#### Visualization
- **`plot_architecture.py`**: Architecture diagram generation for the Security Dilemma

#### Documentation
- **`ReadMe_6_Security_Dilemma.md`**: Specific documentation for the Security Dilemma

---

## Architecture and Visualization Scripts

**Location**: `Games/PlotArchitecture/`

### Main Scripts
- **`generate_architecture.sh`**: Bash script that generates flowcharts for all games or a specific game
- **`plot_architecture.py`**: Main Python script for creating architecture diagrams
- **`architecture_flowchart.py`**: Specialized script for generating detailed architecture flowcharts

### Features
- Supports command-line arguments for customization:
  - `-c, --chunk-lines`: Override chunk size for processing (default: 150)
  - `-m, --model`: Override the model used (default: gpt-4o)
  - Game number (1-4): Generate architecture for a specific game

### Usage Examples
```bash
# Generate architecture for all games
./generate_architecture.sh

# Generate for a specific game (e.g., Prisoners Dilemma)
./generate_architecture.sh 1

# Customize chunk size and model
./generate_architecture.sh -c 200 -m gpt-4 2
```

### Documentation
- **`README-Architecture.md`**: Detailed documentation for the architecture generation system

---

## Requirements and Dependencies

### `requirements.txt`
Contains all Python dependencies required for the project:

- **`python-dotenv`**: Environment variable management
- **`numpy`**: Numerical computing library
- **`pandas`**: Data manipulation and analysis
- **`matplotlib`**: Plotting and visualization
- **`tqdm`**: Progress bars for long-running operations
- **`pyyaml`**: YAML file parsing
- **`jsonschema`**: JSON schema validation
- **`requests`**: HTTP library for API calls

### Installation
```bash
pip install -r requirements.txt
```

---

## Project Structure Summary

The ElectOMate-Frontend project implements multiple game theory simulations with AI agents:

1. **Game Implementations**: Each game has its own directory with core logic, testing, and visualization
2. **LLM Integration**: Most games integrate with Large Language Models for intelligent agent behavior
3. **Analysis Tools**: Comprehensive analysis and metrics collection for research purposes
4. **Architecture Documentation**: Automated generation of system architecture diagrams
5. **Testing Infrastructure**: Test suites for ensuring game functionality and fixes

Each game can be run independently, and the project provides tools for batch processing, analysis, and visualization of results across multiple simulation runs.