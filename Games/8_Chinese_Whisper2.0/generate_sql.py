#!/usr/bin/env python3
"""
generate_sql.py - Generate SQL queries from Chinese Whispers history
"""
import json
import asyncio
import os
from typing import List, Dict, Any, Optional
import glob

import yaml
from langsmith import traceable

# Import the centralized LiteLLM client
from llm_client import lite_client

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'  # End color
    BOLD = '\033[1m'

def print_success(message: str):
    print(f"{Colors.GREEN}✅{Colors.ENDC} {message}")

def print_error(message: str):
    print(f"{Colors.RED}❌{Colors.ENDC} {message}")

def print_info(message: str):
    print(f"{Colors.BLUE}ℹ️{Colors.ENDC} {message}")

def print_warning(message: str):
    print(f"{Colors.YELLOW}⚠️{Colors.ENDC} {message}")


class SQLGenerator:
    """Generate SQL queries from story descriptions"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = config.get("sql_model", config["model"])
        self.temperature = config.get("sql_temperature", 0.0)
        self.max_tokens = config.get("sql_max_tokens", 500)
    
    @traceable(name="generate_sql")
    async def generate_sql(self, story: str) -> str:
        """Generate SQL query from a story"""
        prompt = """Given the following request, write one valid SQL query (SQLite dialect) that satisfies it. 
The database has a 'students' table with columns: student_id, first_name, last_name, 
speaks_english (0/1), speaks_spanish (0/1), grade_math_pass (0/1), grade_science_pass (0/1), 
grade_english_pass (0/1), is_highschool (0/1), is_active (0/1).

Request: {story}

Respond with only the SQL query, no explanations.""".format(story=story)
        
        try:
            response_str = await lite_client.chat_completion(
                model=self.model,
                system_prompt="You are an expert SQL developer. Generate only valid SQLite queries without explanations.",
                user_prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                task_type="sql_generation"
            )
            
            sql = response_str.strip()
            # Clean up the SQL if it has markdown code blocks
            if sql.startswith("```"):
                sql = sql.split("```")[1]
                if sql.startswith("sql"):
                    sql = sql[3:]
                sql = sql.strip()
            
            return sql
            
        except Exception as e:
            print_error(f"SQL generation failed: {str(e)}")
            if "authentication" in str(e).lower() or "api key" in str(e).lower():
                print_error("API Key issue detected. Please check your .env file")
            raise e


def find_latest_run_folder() -> str:
    """Find the most recent run folder in the results directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    
    if not os.path.exists(results_dir):
        raise FileNotFoundError("No results directory found. Please run simulate_chain.py first.")
    
    # Find all run folders
    run_pattern = os.path.join(results_dir, "run_*")
    run_folders = glob.glob(run_pattern)
    
    if not run_folders:
        raise FileNotFoundError("No run folders found in results directory. Please run simulate_chain.py first.")
    
    # Sort by creation time (most recent first)
    run_folders.sort(key=os.path.getctime, reverse=True)
    return run_folders[0]


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate SQL from stories")
    parser.add_argument("--step", nargs="*", type=int, help="Specific steps to process")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--run-folder", help="Specific run folder to process (default: latest)")
    args = parser.parse_args()
    
    print_info("=" * 50)
    print_info("SQL Query Generation")
    print_info("=" * 50)
    
    # Determine run folder to use
    try:
        if args.run_folder:
            if os.path.isabs(args.run_folder):
                run_folder = args.run_folder
            else:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                run_folder = os.path.join(script_dir, "results", args.run_folder)
        else:
            run_folder = find_latest_run_folder()
        
        print_info(f"Using run folder: {run_folder}")
        
        if not os.path.exists(run_folder):
            raise FileNotFoundError(f"Run folder not found: {run_folder}")
            
    except Exception as e:
        print_error(f"Failed to determine run folder: {e}")
        return
    
    # Load config - check if path is relative, if so make it relative to script directory
    config_path = args.config
    if not os.path.isabs(config_path) and not os.path.exists(config_path):
        # Try to find config relative to script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, args.config)
    
    try:
        print_info(f"Loading configuration from: {config_path}")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print_success("Configuration loaded successfully")
    except Exception as e:
        print_error(f"Failed to load config: {e}")
        return
    
    # Load history from run folder
    history_path = os.path.join(run_folder, "history.jsonl")
    
    try:
        print_info(f"Loading history from {history_path}...")
        history = []
        with open(history_path) as f:
            for line in f:
                history.append(json.loads(line))
        print_success(f"Loaded {len(history)} history entries")
    except Exception as e:
        print_error(f"Failed to load history: {e}")
        print_error("Make sure to run simulate_chain.py first")
        return
    
    # Determine which steps to process
    if args.step:
        steps_to_process = args.step
        print_info(f"Processing specific steps: {steps_to_process}")
    else:
        steps_to_process = [entry["step"] for entry in history]
        print_info(f"Processing all steps: {steps_to_process}")
    
    # Generate SQL for each step
    try:
        print_info("Initializing SQL generator...")
        generator = SQLGenerator(config)
        print_success("SQL generator initialized")
        
        queries = []
        print_info("-" * 30)
        
        for entry in history:
            if entry["step"] not in steps_to_process:
                continue
            
            print_info(f"Processing step {entry['step']}...")
            print_info(f"Story: {entry['story'][:100]}...")
            
            try:
                sql = await generator.generate_sql(entry["story"])
                
                query_entry = {
                    "step": entry["step"],
                    "sql": sql
                }
                queries.append(query_entry)
                
                print_success(f"Step {entry['step']} - SQL generated")
                print_info(f"SQL: {sql}")
                
            except Exception as e:
                print_error(f"Failed to generate SQL for step {entry['step']}: {e}")
                continue
        
        print_info("-" * 30)
        
        # Save all queries to JSONL in run folder
        if queries:
            queries_path = os.path.join(run_folder, "queries.jsonl")
            with open(queries_path, "w") as f:
                for query in queries:
                    f.write(json.dumps(query) + "\n")
            
            print_success(f"Generated {len(queries)} SQL queries")
            print_success("All results saved to queries.jsonl")
            
            # Print session summary
            summary = lite_client.get_session_summary()
            if isinstance(summary, dict):
                print_info("-" * 30)
                print_success("LLM Session Summary:")
                print_info(f"  Total Calls: {summary['total_calls']}")
                print_info(f"  Total Cost: ${summary['total_cost_usd']}")
                print_info(f"  Total Tokens: {summary['total_tokens']}")
                print_info(f"  Total Time: {summary['total_inference_time_seconds']}s")
                
                # Update the session summary in run folder
                summary_file = os.path.join(run_folder, "llm_inference_metrics.json")
                if os.path.exists(summary_file):
                    # Load existing summary and merge
                    with open(summary_file, "r") as f:
                        existing_summary = json.load(f)
                    # Add SQL generation metrics
                    merged_summary = existing_summary.copy()
                    merged_summary["sql_generation"] = summary
                    with open(summary_file, "w") as f:
                        json.dump(merged_summary, f, indent=2)
                else:
                    # Create new summary file
                    with open(summary_file, "w") as f:
                        json.dump({"sql_generation": summary}, f, indent=2)
                print_success(f"SQL generation metrics added to {summary_file}")
        else:
            print_error("No SQL queries were generated successfully")
            
        print_info(f"All SQL files saved to: {run_folder}")
            
    except Exception as e:
        print_error(f"SQL generation process failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 