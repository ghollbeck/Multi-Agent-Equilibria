
# Oligopoly Pricing Simulation

This repository provides a full simulation harness to study whether advanced language‑model‑driven
pricing algorithms can tacitly coordinate on supra‑competitive prices via implicit price signalling.

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

## Running Simulations

```bash
python run_experiments.py
```

This will iterate over:

* Firm counts ∈ {2, 3, 5}
* Demand noise σ ∈ {0, 0.5 b, b}
* Cost structures: symmetric vs one low‑cost firm
* Agent match‑ups: Baseline / Heuristic / LLM / Mixed  
  (100 random seeds each)

Logs are written as JSON‑lines files in `logs/`.

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
