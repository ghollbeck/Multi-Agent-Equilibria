# Chinese Whispers SQL Drift Study

This extension to the Students Database system studies how LLM-generated SQL queries drift through a Chinese Whispers chain. The system uses LangGraph for orchestration and LangSmith for tracing.

## Overview

The study consists of three tasks:

1. **Simulation** (`simulate_chain.py`): Run a chain of N agents that rewrite a story
2. **SQL Generation** (`generate_sql.py`): Convert each story to a SQL query
3. **Evaluation** (`evaluate_sql.py`): Execute queries and measure drift in result-set size

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file at the **project root** (not in this directory):

```bash
# Navigate to project root
cd ../..

# Create .env file with required variables
cat > .env << EOF
# LangSmith Configuration
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=chinese_whispers_sql
LANGCHAIN_API_KEY=your_langsmith_api_key_here

# LiteLLM Configuration (choose one option)
# Option 1: Use LiteLLM API key (for proxy/hosted service)
LITELLM_API_KEY=your_litellm_api_key_here
LITELLM_BASE_URL=https://api.litellm.ai  # Optional: for custom proxy URL

# Option 2: Use direct provider keys
# OPENAI_API_KEY=your_openai_api_key_here
# ANTHROPIC_API_KEY=your_anthropic_api_key_here
# COHERE_API_KEY=your_cohere_api_key_here
EOF

# Edit .env and add your actual API keys
```

### 3. Configure Settings

Edit `config.yaml` to customize:
- LLM model and parameters
- Number of agents in the chain
- Initial story
- Prompt templates

## Usage

**Note**: Run all commands from the `Games/8_Chinese_Whisper2.0/` directory. The scripts will automatically load the `.env` file from the project root.

### Task 1: Run the Chinese Whispers Chain

```bash
cd Games/8_Chinese_Whisper2.0/
python simulate_chain.py --config config.yaml
```

This will:
- Run N agents that sequentially rewrite a story
- Save results to `history.jsonl`
- Trace execution to LangSmith

### Task 2: Generate SQL Queries

Generate SQL for all steps:
```bash
python generate_sql.py --config config.yaml
```

Or generate SQL for specific steps:
```bash
python generate_sql.py --step 1 3 5 --config config.yaml
```

This will:
- Read stories from `history.jsonl`
- Generate SQL queries using an LLM
- Save to `queries.jsonl` and individual `sql_step_*.sql` files

### Task 3: Evaluate SQL Queries

```bash
python evaluate_sql.py --baseline 5 --config config.yaml
```

This will:
- Create an in-memory SQLite database
- Populate it with 5 sample students
- Execute each SQL query and count rows
- Calculate drift relative to the baseline step
- Save results to `results.json`
- Generate plot `diff_vs_agent.png`

## Output Files

| File | Description |
|------|-------------|
| `history.jsonl` | Story at each step of the chain |
| `queries.jsonl` | SQL queries for each step |
| `sql_step_*.sql` | Individual SQL query files |
| `results.json` | Execution results with row counts and drift |
| `diff_vs_agent.png` | Plot of drift vs agent number |

## Example Workflow

```bash
# Navigate to the Chinese Whisper 2.0 directory
cd Games/8_Chinese_Whisper2.0/

# 1. Run the chain
python simulate_chain.py

# 2. Generate SQL queries for all steps
python generate_sql.py

# 3. Evaluate with step 5 as baseline
python evaluate_sql.py --baseline 5

# View the results
cat results.json
open diff_vs_agent.png
```

## Configuration Options

### config.yaml

```yaml
# LLM settings
model: "gpt-3.5-turbo"
temperature: 0.7
max_tokens: 300

# SQL generation settings
sql_model: "gpt-3.5-turbo"
sql_temperature: 0.0  # Lower for consistent SQL
sql_max_tokens: 500

# Chain settings
num_agents: 5

# Initial story
initial_story: |
  Find all high school students who speak English and have passed all their subjects, 
  but are currently marked as inactive in the system.
```

## Database Schema

The system uses the existing `students` table with 10 columns:

1. `student_id` (PRIMARY KEY)
2. `first_name` (VARCHAR)
3. `last_name` (VARCHAR) 
4. `speaks_english` (0/1)
5. `speaks_spanish` (0/1)
6. `grade_math_pass` (0/1)
7. `grade_science_pass` (0/1)
8. `grade_english_pass` (0/1)
9. `is_highschool` (0/1)
10. `is_active` (0/1)

## Sample Data

The evaluation uses these 5 students:
- Alice Johnson: High school, speaks English, all passed, active
- Bob Smith: Primary, bilingual, failed math, active
- Charlie Brown: High school, speaks Spanish, failed science, active
- Maria Garcia: Primary, speaks Spanish, failed English, active
- John Doe: High school, speaks English, all passed, inactive

## Monitoring with LangSmith

1. Ensure your LangSmith API key is set in `.env`
2. View traces at: https://smith.langchain.com/
3. Look for project: `chinese_whispers_sql`

## Troubleshooting

### Environment Variables Not Found
- Ensure `.env` file is at the **project root** (Multi-Agent-Equilibria/), not in Games/8_Chinese_Whisper2.0/
- The scripts automatically look for `.env` two directories up from their location

### LiteLLM Configuration
- If using `LITELLM_API_KEY`, the scripts will automatically configure LiteLLM
- If using direct provider keys (like `OPENAI_API_KEY`), make sure to comment out the LiteLLM options
- For custom LiteLLM proxy endpoints, set `LITELLM_BASE_URL` to your proxy URL

### LangSmith Connection Issues
- Verify `LANGCHAIN_API_KEY` is set correctly in the root `.env` file
- Check network connectivity to LangSmith

### SQL Execution Errors
- Check `sql_step_*.sql` files for malformed queries
- Verify the SQL follows SQLite syntax
- Review error messages in console output

### Memory Issues
- The evaluation uses in-memory SQLite databases
- For large result sets, consider modifying to use file-based DBs 