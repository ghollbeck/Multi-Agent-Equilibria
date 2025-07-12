# Chinese Whispers SQL - Organized Results Structure

This document explains the new organized structure for storing simulation results and SQL steps.

## Directory Structure

```
Games/8_Chinese_Whisper2.0/
├── results/                          # Main results directory
│   ├── run_2025-06-23_16-54-02/     # Individual run folder (timestamped)
│   │   ├── history.jsonl             # Story evolution history
│   │   ├── queries.jsonl             # Generated SQL queries
│   │   ├── results.json              # Evaluation results
│   │   ├── llm_inference_metrics.json # LLM usage metrics
│   │   ├── diff_vs_agent.png         # Drift visualization
│   │   ├── run_info.json             # Run metadata
│   │   ├── sql_step_0.sql            # Individual SQL files
│   │   ├── sql_step_1.sql
│   │   ├── sql_step_2.sql
│   │   ├── ...
│   │   └── sql_step_N.sql
│   ├── run_2025-06-23_17-30-15/     # Another run folder
│   └── migrated_run_2025-06-23_16-54-02/  # Migrated old results
├── simulate_chain.py                 # Modified to use organized structure
├── generate_sql.py                  # Modified to use organized structure
├── evaluate_sql.py                  # Modified to use organized structure
├── run_complete_pipeline.py         # New: Complete pipeline runner
├── migrate_old_results.py           # New: Migration utility
└── ...
```

## Key Changes

### 1. Timestamped Run Folders
- Each simulation run creates a unique folder: `run_YYYY-MM-DD_HH-MM-SS`
- All files for that run are contained within this folder
- Easy to compare results across different runs

### 2. Modified Scripts
All scripts now use the organized structure:

- **simulate_chain.py**: Creates run folder and saves all outputs there
- **generate_sql.py**: Loads from latest run folder, saves SQL files there
- **evaluate_sql.py**: Loads from latest run folder, saves results there

### 3. New Files per Run
Each run folder contains:

- `history.jsonl` - Story evolution through agents
- `queries.jsonl` - All generated SQL queries
- `results.json` - Evaluation results with drift metrics
- `llm_inference_metrics.json` - LLM usage and cost tracking
- `diff_vs_agent.png` - Visualization of query drift
- `run_info.json` - Metadata about the run
- `sql_step_*.sql` - Individual SQL files for each step
- `migration_info.json` - (Only for migrated runs)

## Usage

### Running a Complete Pipeline
```bash
# Run all steps in sequence
python run_complete_pipeline.py

# Skip specific steps
python run_complete_pipeline.py --skip-simulate
python run_complete_pipeline.py --skip-generate --skip-evaluate
```

### Running Individual Steps
```bash
# Step 1: Simulate story evolution
python simulate_chain.py

# Step 2: Generate SQL from stories (uses latest run)
python generate_sql.py

# Step 3: Evaluate SQL queries (uses latest run)
python evaluate_sql.py

# Use specific run folder
python generate_sql.py --run-folder run_2025-06-23_16-54-02
python evaluate_sql.py --run-folder run_2025-06-23_16-54-02
```

### Migrating Old Results
```bash
# Move old files from root to organized structure
python migrate_old_results.py
```

## Benefits

1. **Organization**: All files for a run are grouped together
2. **Comparison**: Easy to compare results across runs
3. **Archival**: Old runs are preserved with timestamps
4. **Tracking**: Complete history of experiments
5. **Reproducibility**: Each run is self-contained

## File Descriptions

### Core Result Files
- **history.jsonl**: Line-delimited JSON with story evolution
- **queries.jsonl**: Generated SQL queries for each step
- **results.json**: Evaluation results with row counts and drift
- **diff_vs_agent.png**: Plot showing drift over agent steps

### Metadata Files  
- **run_info.json**: Configuration and run metadata
- **llm_inference_metrics.json**: Token usage, costs, timing
- **migration_info.json**: Info about migrated runs

### SQL Files
- **sql_step_0.sql**: Original/baseline SQL query
- **sql_step_1.sql**: Query after first agent transformation
- **sql_step_N.sql**: Query after Nth agent transformation

## Notes

- Scripts automatically find the latest run folder when no specific folder is provided
- All timestamps are in local time for folder names, UTC for internal timestamps
- The migration script safely moves old files without overwriting
- Each run is completely self-contained for reproducibility 