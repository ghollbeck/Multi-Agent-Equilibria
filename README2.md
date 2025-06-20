# ElectOMate/AutoCreate - Script Documentation

This document provides a comprehensive overview of all scripts in the ElectOMate/AutoCreate project, organized by functionality and directory structure.

## 📋 Table of Contents

1. [Root Level Scripts](#root-level-scripts)
2. [Game Simulations](#game-simulations)
3. [Architecture & Visualization Tools](#architecture--visualization-tools)
4. [Testing & Validation Scripts](#testing--validation-scripts)
5. [Analysis & Metrics Scripts](#analysis--metrics-scripts)
6. [Configuration & Utility Scripts](#configuration--utility-scripts)

---

## 🔧 Root Level Scripts

### `pushing_script.sh`
**Purpose**: Git utility script for cleaning up macOS-specific files  
**Functionality**: 
- Removes `.DS_Store` files from git tracking
- Commits and pushes changes to remove these system files
- Maintains clean repository structure

**Usage**:
```bash
./pushing_script.sh
```

---

## 🎮 Game Simulations

### 1. Prisoner's Dilemma (`Games/1_Prisoners_Dilemma/`)

#### Core Game Scripts
- **`prisoner_dilemma Final64.py`**: Advanced prisoner's dilemma simulation with 64-bit precision
- **`prisoner_dilemma FinalPrompt2.py`**: Enhanced version with improved prompting strategies
- **`prisoner_dilemma FinalPrompt2_Free.py`**: Free-form variant allowing more flexible agent responses
- **`PD_GameFinal.py`**: Main prisoner's dilemma game engine

#### Supporting Scripts
- **`prompts.py`**: Prompt templates for LLM agents in prisoner's dilemma
- **`prompts_free.py`**: Free-form prompt templates for more creative responses
- **`plot_architecture.py`**: Visualization tool for game architecture

#### Interactive Resources
- **`1_Prisoners_Dilemma.ipynb`**: Jupyter notebook for interactive experimentation

### 2. MIT Beer Game (`Games/2_MIT_Beer_Game/`)

#### Main Execution
- **`executeMITBeerGame.py`**: Primary script to run MIT Beer Game simulations
  - Configurable parameters (generations, rounds, costs, communication)
  - Command-line argument support
  - Async simulation execution

#### Core Game Engine
- **`MIT_Beer_Game.py`**: Complete implementation of the MIT Beer Game supply chain simulation

#### Supporting Modules
- **`models_mitb_game.py`**: Data models and simulation structures
- **`llm_calls_mitb_game.py`**: LLM interaction handling for game agents
- **`prompts_mitb_game.py`**: Prompt templates for supply chain decision-making
- **`analysis_mitb_game.py`**: Post-simulation analysis and metrics

#### Testing & Debug
- **`test_communication_feature.py`**: Tests for agent communication features
- **`debug_communication_logging.py`**: Debugging utilities for communication logs

### 3. Fishery Game (`Games/3_Fishery_Game/`)

#### Main Game Scripts
- **`Fishery_game.py`**: Core fishery resource management simulation with logistic growth models
- **`enhanced_fishery_game.py`**: Extended version with additional scientific metrics and theoretical benchmarks

#### Testing & Validation
- **`test_fishery_fixes.py`**: Comprehensive testing suite for fishery simulation fixes
- **`validate_fixes.py`**: Validation scripts for ensuring simulation accuracy

### 4. Market Impact Game (`Games/4_Market_Impact_Game/`)

#### Core Simulation
- **`market_impact_game.py`**: Trading simulation modeling market impact effects

#### Testing
- **`test_market_impact_fixes.py`**: Testing suite for market impact simulation fixes

### 5. Oligopoly Simulation (`Games/5_oligopoly_simulation/`)

#### Experiment Execution
- **`run_experiments.py`**: Comprehensive experiment runner for oligopoly scenarios
  - Multiple firm counts (2, 3, 5)
  - Different noise levels and cost structures
  - Support for baseline, heuristic, LLM, and mixed agent types

#### Core Game Engine
- **`oligopoly.py`**: Complete oligopoly game implementation with various agent types

#### Analysis
- **`analyze.py`**: Post-experiment analysis and statistical evaluation

### 6. Security Dilemma (`Games/6_Security_Dilemma/`)

#### Main Game
- **`play_security_dilemma.py`**: Security dilemma game simulation for international relations modeling

#### Testing
- **`test_security_dilemma_fixes.py`**: Testing suite for security dilemma fixes

### 7. Chinese Whisper Games (`Games/6_Chinese_Whisper/` & `Games/7_Chinese_Whisper/`)

#### Main Game Engine
- **`Chinese_Whisper_Game.py`**: Primary script for running Chinese Whisper experiments
  - Simple experiments, batch processing, degradation analysis
  - Multiple information types (factual, narrative, technical)
  - Configurable chain lengths and model parameters

#### Core Components
- **`game_controller.py`**: Central controller for managing Chinese Whisper experiments
- **`information_generator.py`**: Generates seed information for experiments
- **`evaluation_engine.py`**: Evaluates information retention and degradation
- **`analytics.py`**: Advanced analytics for experiment results

#### LLM Integration
- **`llm_calls_chinese_whisper.py`**: LLM interaction handling
- **`models_chinese_whisper.py`**: Data models and simulation structures
- **`prompts_chinese_whisper.py`**: Prompt templates for information processing

---

## 🏗️ Architecture & Visualization Tools

### PlotArchitecture (`Games/PlotArchitecture/`)

#### Main Scripts
- **`plot_architecture.py`**: Core architecture visualization tool
- **`architecture_flowchart.py`**: Advanced flowchart generation for complex architectures
- **`generate_architecture.sh`**: Bulk flowchart generator script
  - Processes all game directories or specific games
  - Configurable chunk sizes and model parameters
  - Supports flags for customization (`-c` for chunk lines, `-m` for model)

#### Usage Examples
```bash
# Generate for all games
./generate_architecture.sh

# Generate for specific game with custom parameters
./generate_architecture.sh -c 200 -m gpt-4o 3

# Generate with different model
./generate_architecture.sh -m gpt-4o-mini
```

### Game-Specific Visualization
Each game directory contains its own `plot_architecture.py` for generating architecture diagrams specific to that simulation.

---

## 🧪 Testing & Validation Scripts

### Comprehensive Testing Suite
- **`test_communication_feature.py`** (MIT Beer Game): Tests agent communication features
- **`test_fishery_fixes.py`** (Fishery Game): Validates fishery simulation corrections
- **`test_market_impact_fixes.py`** (Market Impact): Tests market impact simulation fixes
- **`test_security_dilemma_fixes.py`** (Security Dilemma): Validates security dilemma fixes

### Validation Scripts
- **`validate_fixes.py`** (Fishery Game): Ensures simulation accuracy and bug fixes
- **`debug_communication_logging.py`** (MIT Beer Game): Debugging utilities for communication systems

---

## 📊 Analysis & Metrics Scripts

### Core Analysis Tools
- **`analyze.py`** (Oligopoly): Statistical analysis of oligopoly experiment results
- **`analysis_mitb_game.py`** (MIT Beer Game): Post-simulation metrics and analysis
- **`analytics.py`** (Chinese Whisper): Advanced analytics for information degradation studies

### Metrics and Evaluation
- **`evaluation_engine.py`** (Chinese Whisper): Evaluates information retention, semantic similarity, and factual accuracy
- Built-in metrics in game scripts:
  - Gini coefficients for inequality analysis
  - Equilibrium distance measurements
  - Sustainability indices
  - Degradation rate calculations

---

## ⚙️ Configuration & Utility Scripts

### Prompt Management
- **`prompts.py`** (Prisoner's Dilemma): Standard prompt templates
- **`prompts_free.py`** (Prisoner's Dilemma): Free-form prompt variants
- **`prompts_mitb_game.py`** (MIT Beer Game): Supply chain decision prompts
- **`prompts_chinese_whisper.py`** (Chinese Whisper): Information processing prompts

### Data Models
- **`models_mitb_game.py`**: MIT Beer Game data structures
- **`models_chinese_whisper.py`**: Chinese Whisper game data models

### LLM Integration
- **`llm_calls_mitb_game.py`**: MIT Beer Game LLM interactions
- **`llm_calls_chinese_whisper.py`**: Chinese Whisper LLM handling

---

## 🚀 Quick Start Guide

### Running Individual Games

#### Prisoner's Dilemma
```bash
cd Games/1_Prisoners_Dilemma/
python prisoner_dilemma FinalPrompt2.py
```

#### MIT Beer Game
```bash
cd Games/2_MIT_Beer_Game/
python executeMITBeerGame.py --num_generations 5 --enable_communication
```

#### Fishery Game
```bash
cd Games/3_Fishery_Game/
python Fishery_game.py
```

#### Oligopoly Simulation
```bash
cd Games/5_oligopoly_simulation/
python run_experiments.py
```

#### Chinese Whisper
```bash
cd Games/7_Chinese_Whisper/
python Chinese_Whisper_Game.py
```

### Generating Architecture Diagrams
```bash
cd Games/PlotArchitecture/
./generate_architecture.sh
```

---

## 📁 Project Structure Overview

```
ElectOMate/AutoCreate/
├── pushing_script.sh                     # Git utility
├── requirements.txt                      # Python dependencies
├── Games/
│   ├── 1_Prisoners_Dilemma/             # Game theory simulation
│   ├── 2_MIT_Beer_Game/                 # Supply chain simulation
│   ├── 3_Fishery_Game/                  # Resource management
│   ├── 4_Market_Impact_Game/            # Trading simulation
│   ├── 5_oligopoly_simulation/          # Economic competition
│   ├── 6_Security_Dilemma/              # International relations
│   ├── 6_Chinese_Whisper/               # Information degradation
│   ├── 7_Chinese_Whisper/               # Enhanced whisper game
│   └── PlotArchitecture/                # Visualization tools
└── README2.md                           # This documentation
```

---

## 🔧 Dependencies

Core requirements are listed in `requirements.txt`:
- OpenAI API for LLM interactions
- NumPy for numerical computations
- Pandas for data manipulation
- Matplotlib for visualization
- Asyncio for concurrent execution
- Pydantic for data validation

---

## 📝 Notes

- Most simulations support configurable parameters via command-line arguments
- LLM-based agents require valid OpenAI API keys
- Results are typically saved to `simulation_results/` or `results/` directories
- Architecture diagrams are generated in `architecture/` subdirectories
- All scripts include comprehensive error handling and fallback mechanisms

For specific configuration options and parameters, refer to the individual script documentation or use the `--help` flag where available.