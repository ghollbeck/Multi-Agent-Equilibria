# Oligopoly Simulation

A simulation of price-setting oligopoly with various agent types.

## Overview

This simulation implements:
- N-firm price competition with linear demand
- Multiple agent types (baseline, heuristic, LLM, mixed)
- Logging and metrics calculation

## Updated Features

The oligopoly simulation now includes:
- Finer price grid (0.2 increments vs 1.0) for more coordination possibilities
- Baseline agents with small markup over cost for non-zero profits
- LLM agents that use OpenAI API (optional)
- Consolidated logging in a single master log file

## Running the Simulation

### Set up OpenAI API (Optional)

To use LLM-based agents:

1. Make sure the OpenAI Python package is installed:
   ```
   pip install openai
   ```

2. Add your API key in one of these ways:
   - Set the OPENAI_API_KEY environment variable
   - Edit the .env file in this directory with your key

If no valid API key is provided, the simulation will automatically skip LLM-based matchups.

### Run the experiments

```bash
python run_experiments.py
```

The script will:
1. Run a small test batch first (N=3, noise=0, symmetric costs)
2. If successful, proceed to run the full experiment with all combinations

## Logs and Analysis

All simulation results are stored in:
- `logs/all_runs.jsonl` - Contains all runs with complete metadata
- `logs/metrics_summary.csv` - Analysis results

## Parameter Settings

Current settings:
- 3 firm counts: N ∈ {2, 3, 5}
- 3 noise levels: σ ∈ {0, 0.5, 1.0}
- 2 cost structures: symmetric and asymmetric
- 4 agent matchups: baseline, heuristic, LLM, mixed
- 100 seeds per combination
- 10 rounds per simulation (for testing)

To modify these settings, edit the parameters at the top of `run_experiments.py`.

## Structure

```
oligopoly.py          # Environment, agents, logging, collusion detectors
run_experiments.py    # Executes all treatments & saves JSON‑lines logs
analyze.py            # Post‑hoc metrics & visualisation
logs/                 # Output directory (auto‑created)
```

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install numpy pandas matplotlib scipy scikit-learn tqdm openai
```

⚠️ **OpenAI key**  
Set your key via `export OPENAI_API_KEY="sk-..."` if you wish to use `LLMAgent`.
Without a key, you can still run Baseline, Heuristic, and Mixed agents (LLM parts fallback).

## Analysis & Plots

```bash
python analyze.py
```

This script:

* Computes average markup, HHI, time‑to‑collusion
* Saves `metrics_summary.csv`
* Shows a noise×N heat‑map of markup
* Performs one‑sample *t*‑tests against the competitive benchmark.

## Reproducibility

*Random seeds* are fixed per run.  
Demand noise uses `numpy.random.Generator(seed)` for deterministic draws.

## License

MIT License (c) 2025
