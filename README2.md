# Scripts Documentation - Multi-Agent Equilibria Research Framework

This document provides comprehensive explanations of all scripts in the Multi-Agent Equilibria Research Framework repository, organized by location and functionality.

## Table of Contents

- [Root Level Scripts](#root-level-scripts)
- [Game-Specific Scripts](#game-specific-scripts)
  - [1. Prisoners Dilemma](#1-prisoners-dilemma)
  - [2. MIT Beer Game](#2-mit-beer-game)
  - [3. Fishery Game](#3-fishery-game)
  - [4. Market Impact Game](#4-market-impact-game)
  - [5. Oligopoly Simulation](#5-oligopoly-simulation)
  - [6. Security Dilemma](#6-security-dilemma)
  - [7. Chinese Whisper Games](#7-chinese-whisper-games)
- [Architecture & Visualization Scripts](#architecture--visualization-scripts)
- [Usage Guidelines](#usage-guidelines)

---

## Root Level Scripts

### `pushing_script.sh`
**Location**: `/pushing_script.sh`
**Purpose**: Git utility script for cleaning up repository artifacts
**Description**: 
- Removes `.DS_Store` files from git tracking while keeping them locally
- Commits changes with a descriptive message
- Pushes changes to the main branch
- Useful for maintaining clean repository state on macOS systems

**Usage**:
```bash
./pushing_script.sh
```

---

## Game-Specific Scripts

### 1. Prisoners Dilemma
**Location**: `Games/1_Prisoners_Dilemma/`

#### Core Game Scripts

**`prisoner_dilemma Final64.py`**
- **Purpose**: Advanced implementation that simulates all strategies against all other strategies
- **Features**: 
  - Plots time evolution of outcomes across generations
  - Supports 64+ different strategy combinations
  - Generates comprehensive visualization outputs
  - Implements evolutionary dynamics with strategy selection

**`prisoner_dilemma FinalPrompt2.py`**
- **Purpose**: Multi-provider LLM implementation with customizable prompts
- **Features**:
  - Selection between multiple LLM providers (OpenAI, LiteLLM)
  - Multiple prompt template options
  - Configurable temperature settings
  - Enhanced logging and analysis capabilities

**`prisoner_dilemma FinalPrompt2_Free.py`**
- **Purpose**: Free-tier optimized version of the multi-provider implementation
- **Features**:
  - Reduced API call overhead
  - Optimized for free-tier API limits
  - Streamlined functionality for basic experiments

**`PD_GameFinal.py`**
- **Purpose**: Core implementation of the Iterated Prisoner's Dilemma
- **Features**:
  - Complete game logic with LLM-driven agents
  - 8 classic strategies implementation
  - Asynchronous execution for performance
  - Comprehensive result logging

#### Supporting Scripts

**`prompts.py`**
- **Purpose**: Contains prompt templates for paid LLM tiers
- **Content**: Structured prompts for decision-making, strategy descriptions, and context setting

**`prompts_free.py`**
- **Purpose**: Optimized prompt templates for free LLM tiers
- **Content**: Shorter, more efficient prompts designed to minimize token usage

**`plot_architecture.py`**
- **Purpose**: Generates architectural flowcharts for the Prisoner's Dilemma implementation
- **Function**: Creates visual documentation of code structure and execution flow

---

### 2. MIT Beer Game
**Location**: `Games/2_MIT_Beer_Game/`

#### Core Scripts

**`MIT_Beer_Game.py`**
- **Purpose**: Main implementation of the four-role supply chain simulation
- **Features**:
  - Retailer, Wholesaler, Distributor, and Factory roles
  - Adaptive ordering policies via LLM agents
  - Tracks inventory, backlog, bullwhip effect, and cost metrics
  - Comprehensive logging and visualization

#### Supporting Modules

**`models_mitb_game.py`**
- **Purpose**: Data models and class definitions for the beer game
- **Content**: Role classes, game state management, metric tracking structures

**`llm_calls_mitb_game.py`**
- **Purpose**: LLM interaction layer for the beer game
- **Features**: API call management, response parsing, error handling

**`prompts_mitb_game.py`**
- **Purpose**: Prompt templates specific to supply chain decision-making
- **Content**: Role-specific prompts for ordering decisions, market analysis

**`analysis_mitb_game.py`**
- **Purpose**: Post-simulation analysis and visualization
- **Features**: Statistical analysis, trend identification, performance metrics

#### Testing & Debugging Scripts

**`test_communication_feature.py`**
- **Purpose**: Tests inter-role communication mechanisms
- **Function**: Validates message passing, information sharing between supply chain roles

**`debug_communication_logging.py`**
- **Purpose**: Debug logging for communication features
- **Function**: Detailed logging of communication flows for troubleshooting

**`executeMITBeerGame.py`**
- **Purpose**: Command-line interface for running beer game simulations
- **Features**: Parameter configuration, batch execution, result organization

---

### 3. Fishery Game
**Location**: `Games/3_Fishery_Game/`

#### Core Game Scripts

**`enhanced_fishery_game.py`**
- **Purpose**: Advanced implementation of the common-pool resource game
- **Features**:
  - Regenerative fish stock modeling with logistic growth
  - Multi-generation learning and adaptation
  - Gini coefficient calculations for inequality analysis
  - Equilibrium distance metrics
  - Rich visualization and logging

**`Fishery_game.py`**
- **Purpose**: Basic implementation of the fishery simulation
- **Features**:
  - Core resource extraction mechanics
  - Agent decision-making via LLM
  - Basic visualization and logging

#### Testing & Validation Scripts

**`test_fishery_fixes.py`**
- **Purpose**: Comprehensive testing suite for fishery game mechanics
- **Function**: Validates game logic, agent behavior, mathematical models

**`validate_fixes.py`**
- **Purpose**: Validation script for bug fixes and improvements
- **Function**: Ensures stability and correctness of game implementations

---

### 4. Market Impact Game
**Location**: `Games/4_Market_Impact_Game/`

#### Core Scripts

**`market_impact_game.py`**
- **Purpose**: Simulates algorithmic trading with market impact
- **Features**:
  - BUY/SELL/HOLD decisions by LLM agents
  - Price impact modeling
  - P&L tracking and market depth analysis
  - Equilibrium metrics across rounds and generations

#### Testing Scripts

**`test_market_impact_fixes.py`**
- **Purpose**: Testing suite for market impact game
- **Function**: Validates trading logic, price calculations, agent interactions

---

### 5. Oligopoly Simulation
**Location**: `Games/5_oligopoly_simulation/`

#### Core Scripts

**`oligopoly.py`**
- **Purpose**: Classical price-setting oligopoly simulation
- **Features**:
  - Multiple agent types (baseline, heuristic, LLM, mixed)
  - Markup analysis and HHI (Herfindahl-Hirschman Index) calculations
  - Collusion detection and time-to-collusion metrics
  - Varying parameters: N (number of firms), noise, cost asymmetries

**`run_experiments.py`**
- **Purpose**: Batch experiment runner for oligopoly scenarios
- **Features**:
  - Parameter sweeps across different market conditions
  - Automated experiment orchestration
  - Result aggregation and organization

**`analyze.py`**
- **Purpose**: Post-simulation analysis and visualization
- **Features**:
  - Statistical analysis of market outcomes
  - Competition metrics calculation
  - Trend analysis and visualization generation

---

### 6. Security Dilemma
**Location**: `Games/6_Security_Dilemma/`

#### Core Scripts

**`play_security_dilemma.py`**
- **Purpose**: Implementation of the security dilemma (arms race) game
- **Features**:
  - Iterated interactions with misperception noise
  - Rich strategy library (Tit-for-Tat, Grim Trigger, Pavlov, etc.)
  - CLI and batch execution modes
  - JSON schema-validated logging
  - Comprehensive strategy analysis

#### Testing Scripts

**`test_security_dilemma_fixes.py`**
- **Purpose**: Testing suite for security dilemma game mechanics
- **Function**: Validates strategy implementations, noise effects, logging accuracy

---

### 7. Chinese Whisper Games
**Locations**: `Games/6_Chinese_Whisper/` and `Games/7_Chinese_Whisper/`

Both directories contain identical script sets representing different versions or experiments of the Chinese Whisper game.

#### Core Game Components

**`Chinese_Whisper_Game.py`**
- **Purpose**: Main game controller for information transmission simulation
- **Features**: Message passing, distortion modeling, chain communication analysis

**`game_controller.py`**
- **Purpose**: Game flow management and coordination
- **Function**: Orchestrates rounds, manages player turns, controls game state

**`information_generator.py`**
- **Purpose**: Generates initial information and content for transmission
- **Function**: Creates diverse message types, complexity levels, content variations

**`evaluation_engine.py`**
- **Purpose**: Evaluates information fidelity and transmission accuracy
- **Features**: Semantic similarity analysis, distortion measurement, accuracy metrics

#### Supporting Modules

**`analytics.py`**
- **Purpose**: Comprehensive analysis of game results
- **Features**: Statistical analysis, visualization generation, trend identification

**`models_chinese_whisper.py`**
- **Purpose**: Data models and structures for the Chinese Whisper game
- **Content**: Player classes, message structures, game state definitions

**`llm_calls_chinese_whisper.py`**
- **Purpose**: LLM interaction layer for Chinese Whisper game
- **Function**: API management, response processing, error handling

**`prompts_chinese_whisper.py`**
- **Purpose**: Prompt templates for information transmission tasks
- **Content**: Context-setting prompts, instruction templates, role definitions

---

## Architecture & Visualization Scripts

### PlotArchitecture System
**Location**: `Games/PlotArchitecture/`

This system provides automated documentation and visualization of code architecture across all games.

#### Core Scripts

**`architecture_flowchart.py`**
- **Purpose**: Generic driver for generating Mermaid flowcharts from Python code
- **Features**:
  - LLM-powered code analysis
  - Automatic flowchart generation
  - Call-graph and class hierarchy visualization
  - Multiple output formats (Mermaid, HTML)

**`plot_architecture.py`**
- **Purpose**: Main interface for architecture visualization
- **Features**:
  - Configurable analysis depth
  - Batch processing capabilities
  - Multiple rendering options
  - Integration with game-specific plotters

**`generate_architecture.sh`**
- **Purpose**: Batch script for generating architecture diagrams
- **Features**:
  - Processes all games or individual selections
  - Configurable chunk sizes and models
  - Automated HTML rendering
  - Progress tracking and error handling

#### Game-Specific Architecture Scripts

Each game directory contains a `plot_architecture.py` script that:
- Calls the central architecture generation system
- Configures game-specific parameters
- Saves flowcharts to local `architecture/` directories
- Provides game-tailored visualization options

---

## Usage Guidelines

### Running Individual Scripts

Most core game scripts can be run directly:
```bash
# Prisoner's Dilemma
python "Games/1_Prisoners_Dilemma/prisoner_dilemma Final64.py"

# MIT Beer Game
python "Games/2_MIT_Beer_Game/MIT_Beer_Game.py"

# Fishery Game
python "Games/3_Fishery_Game/enhanced_fishery_game.py"

# Market Impact
python "Games/4_Market_Impact_Game/market_impact_game.py"

# Oligopoly
python "Games/5_oligopoly_simulation/run_experiments.py"

# Security Dilemma
python "Games/6_Security_Dilemma/play_security_dilemma.py"
```

### Architecture Generation

Generate flowcharts for all games:
```bash
cd Games/PlotArchitecture
./generate_architecture.sh
```

Generate for specific game:
```bash
./generate_architecture.sh 1  # For Prisoner's Dilemma
./generate_architecture.sh 2  # For MIT Beer Game
# etc.
```

### Testing Scripts

Run test suites for validation:
```bash
# Test specific game fixes
python "Games/3_Fishery_Game/test_fishery_fixes.py"
python "Games/4_Market_Impact_Game/test_market_impact_fixes.py"
python "Games/6_Security_Dilemma/test_security_dilemma_fixes.py"
```

### Analysis Scripts

Run post-simulation analysis:
```bash
# Oligopoly analysis
python "Games/5_oligopoly_simulation/analyze.py"

# MIT Beer Game analysis
python "Games/2_MIT_Beer_Game/analysis_mitb_game.py"

# Chinese Whisper analytics
python "Games/6_Chinese_Whisper/analytics.py"
```

---

## Dependencies

All scripts require:
- Python 3.8+
- OpenAI API key (for LLM functionality)
- Required packages: `openai`, `pandas`, `matplotlib`, `numpy`, `nest_asyncio`
- Additional packages per `requirements.txt`

## Output Locations

Each game saves results to dedicated directories:
- **Prisoners Dilemma**: `simulation_results/`
- **MIT Beer Game**: `simulation_results/`
- **Fishery Game**: `fishery_simulation_results/`
- **Market Impact**: `market_results/`
- **Oligopoly**: `logs/`, `plots/`
- **Security Dilemma**: `results/`
- **Chinese Whisper**: Various output directories
- **Architecture**: `<game>/architecture/`

## Contributing

When adding new scripts:
1. Follow the established naming conventions
2. Include comprehensive docstrings
3. Add corresponding architecture generation
4. Update test suites as needed
5. Document new scripts in this README2.md

---

*This documentation covers all scripts as of the repository snapshot. For the most current script functionality, refer to individual script docstrings and the main README.md.*