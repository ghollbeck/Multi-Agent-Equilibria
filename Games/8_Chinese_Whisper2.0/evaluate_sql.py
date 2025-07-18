#!/usr/bin/env python3
"""
evaluate_sql.py - Execute SQL queries and measure drift in result-set size
"""
import json
import sqlite3
from typing import List, Dict, Any
import sys
import os
import glob
import logging
from datetime import datetime

import yaml
import matplotlib.pyplot as plt

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

# Add parent directory to path to import StudentsDBManager
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from students_db_manager import StudentsDBManager


class SQLEvaluator:
    """Evaluate SQL queries against the students database"""
    
    def __init__(self, log_file_path: str = None):
        print_info("Initializing SQLEvaluator...")
        # Use in-memory database
        self.db_manager = StudentsDBManager(":memory:")
        self.log_file_path = log_file_path
        self.logger = self._setup_file_logger() if log_file_path else None
        self._populate_sample_data()
        print_success("SQLEvaluator initialized with in-memory DB and sample data.")
        
    def _setup_file_logger(self):
        """Setup file logger for dual logging (terminal + file)"""
        logger = logging.getLogger('sql_evaluator')
        logger.setLevel(logging.INFO)
        
        # Remove any existing handlers to avoid duplication
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            
        # Create file handler
        file_handler = logging.FileHandler(self.log_file_path, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        return logger
    
    def log_dual(self, message: str, level: str = "info"):
        """Log to both terminal and file"""
        # Terminal logging
        if level == "info":
            print_info(message)
        elif level == "success":
            print_success(message)
        elif level == "warning":
            print_warning(message)
        elif level == "error":
            print_error(message)
        else:
            print(message)
            
        # File logging
        if self.logger:
            # Strip ANSI color codes for file logging
            clean_message = message
            for color_code in [Colors.GREEN, Colors.RED, Colors.YELLOW, Colors.BLUE, Colors.ENDC, Colors.BOLD]:
                clean_message = clean_message.replace(color_code, '')
            clean_message = clean_message.replace('✅', '[SUCCESS]').replace('❌', '[ERROR]').replace('ℹ️', '[INFO]').replace('⚠️', '[WARNING]')
            
            if level == "error":
                self.logger.error(clean_message)
            elif level == "warning":
                self.logger.warning(clean_message)
            else:
                self.logger.info(clean_message)
    
    def _populate_sample_data(self):
        """Populate database with sample data from students_db_manager.py"""
        print_info("Populating database with sample data...")
        sample_students = [
            ("Alice", "Johnson", 1, 0, 1, 1, 1, 1, 1),
            ("Bob", "Smith", 1, 1, 0, 1, 1, 0, 1),
            ("Charlie", "Brown", 0, 1, 1, 0, 1, 1, 1),
            ("Maria", "Garcia", 0, 1, 1, 1, 0, 0, 1),
            ("John", "Doe", 1, 0, 1, 1, 1, 1, 0)
        ]
        
        for i, student_data in enumerate(sample_students):
            self.db_manager.add_student(*student_data)
            # print_info(f"  Added sample student {i+1}: {student_data[0]} {student_data[1]}")
        print_success(f"Added {len(sample_students)} sample students to the database.")
    
    def execute_query(self, sql: str) -> List[Dict[str, Any]]:
        """Execute a SQL query and return rows as list of dicts (column_name -> value)"""
        try:
            # Check if comprehensive database exists
            script_dir = os.path.dirname(os.path.abspath(__file__))
            full_db_path = os.path.join(script_dir, "full_students.db")
            
            if os.path.exists(full_db_path):
                # Use the comprehensive database
                with sqlite3.connect(full_db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(sql)
                    fetched = cursor.fetchall()
                    col_names = [d[0] for d in cursor.description]
                    rows = [dict(zip(col_names, row)) for row in fetched]
                    return rows
            else:
                # Fallback to in-memory database with sample data
                print_warning("full_students.db not found, using sample data. Run generate_full_database.py first for comprehensive results.")
                
                with sqlite3.connect(":memory:") as conn:
                    cursor = conn.cursor()
                    # Create schema
                    cursor.execute('''
                        CREATE TABLE students (
                            student_id INTEGER PRIMARY KEY AUTOINCREMENT,
                            first_name VARCHAR(50) NOT NULL,
                            last_name VARCHAR(50) NOT NULL,
                            speaks_english INTEGER CHECK (speaks_english IN (0, 1)),
                            speaks_spanish INTEGER CHECK (speaks_spanish IN (0, 1)),
                            grade_math_pass INTEGER CHECK (grade_math_pass IN (0, 1)),
                            grade_science_pass INTEGER CHECK (grade_science_pass IN (0, 1)),
                            grade_english_pass INTEGER CHECK (grade_english_pass IN (0, 1)),
                            is_highschool INTEGER CHECK (is_highschool IN (0, 1)),
                            is_active INTEGER CHECK (is_active IN (0, 1))
                        )
                    ''')
                    
                    # Insert sample data
                    sample_students_data = [
                        ("Alice", "Johnson", 1, 0, 1, 1, 1, 1, 1),
                        ("Bob", "Smith", 1, 1, 0, 1, 1, 0, 1),
                        ("Charlie", "Brown", 0, 1, 1, 0, 1, 1, 1),
                        ("Maria", "Garcia", 0, 1, 1, 1, 0, 0, 1),
                        ("John", "Doe", 1, 0, 1, 1, 1, 1, 0)
                    ]
                    
                    for student in sample_students_data:
                        cursor.execute('''
                            INSERT INTO students (first_name, last_name, speaks_english, speaks_spanish, 
                                                grade_math_pass, grade_science_pass, grade_english_pass, 
                                                is_highschool, is_active)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', student)
                    
                    # Execute the query
                    cursor.execute(sql)
                    fetched = cursor.fetchall()
                    col_names = [d[0] for d in cursor.description]
                    rows = [dict(zip(col_names, row)) for row in fetched]
                    return rows
                
        except Exception as e:
            print_error(f"Error executing query: {e}")
            print_error(f"Failed Query: {sql}")
            return [] # Indicate error
    
    def log_drift_explanation(self):
        """Log the drift evaluation explanation"""
        explanation = """
==================================================
How Drift is Evaluated
==================================================

The drift is calculated as:
    Drift = |rows_returned - baseline_rows|

Where:
• baseline_rows: Number of rows returned by the baseline query (typically step 0 - the original story's SQL)
• rows_returned: Number of rows returned by each agent's SQL query  
• The result is the absolute difference in row counts

What the Values Mean:
• 0 drift: The SQL query returns exactly the same number of rows as the baseline
• Higher drift: The SQL query returns that many more/fewer rows than the baseline

This measures semantic drift in LLM chains - how meaning changes as stories pass through multiple agents
and how well the final SQL captures the original intent.
==================================================
"""
        self.log_dual(explanation, "info")
    
    def evaluate_queries(self, queries: List[Dict[str, Any]], baseline_step: int) -> List[Dict[str, Any]]:
        """Evaluate all queries and compute drift"""
        # Log the drift explanation first
        self.log_drift_explanation()
        
        results = []
        baseline_count = -1 # Initialize to error state
        
        self.log_dual(f"Attempting to find baseline for step: {baseline_step}", "info")
        # First, find the baseline count
        for query_data in queries:
            if query_data["step"] == baseline_step:
                self.log_dual(f"Executing baseline query for step {baseline_step}: {query_data['sql']}", "info")
                baseline_rows = self.execute_query(query_data["sql"])
                baseline_count = len(baseline_rows)
                self.log_dual(f"Baseline (step {baseline_step}) row count: {baseline_count}", "success")
                break
        
        if baseline_count == -1: # If baseline step not found or failed
            if queries: # Check if there are any queries to use as a fallback
                last_step_data = queries[-1]
                self.log_dual(f"Baseline step {baseline_step} not found or failed. Using last available step ({last_step_data['step']}) as baseline.", "warning")
                self.log_dual(f"Executing fallback baseline query for step {last_step_data['step']}: {last_step_data['sql']}", "info")
                baseline_rows = self.execute_query(last_step_data["sql"])
                baseline_count = len(baseline_rows)
                self.log_dual(f"Fallback baseline (step {last_step_data['step']}) row count: {baseline_count}", "success")
                results.append({
                    "step": last_step_data["step"],
                    "row_count": baseline_count,
                    "sql": last_step_data["sql"],
                    "diff": "ERROR",
                    "rows": baseline_rows
                })
            else:
                self.log_dual("No queries available to determine a baseline. Cannot compute diffs.", "error")
                return []

        self.log_dual("-" * 30, "info")
        # Evaluate all queries
        for query_data in queries:
            self.log_dual(f"Executing query for step {query_data['step']}: {query_data['sql']}", "info")
            rows = self.execute_query(query_data["sql"])
            row_count = len(rows)
            
            if rows != []:
                diff = abs(row_count - baseline_count)
                result = {
                    "step": query_data["step"],
                    "row_count": row_count,
                    "sql": query_data["sql"],
                    "diff": diff,
                    "rows": rows
                }
                results.append(result)
                # Log rows info (truncate if large)
                preview = rows if len(rows) <= 5 else rows[:5] + [f"...and {len(rows)-5} more rows"]
                self.log_dual(f"Step {query_data['step']}: {row_count} rows (diff: {diff}) Rows: {preview}", "success")
            else:
                self.log_dual(f"Skipping diff calculation for step {query_data['step']} due to query execution error.", "error")
                results.append({
                    "step": query_data["step"],
                    "row_count": "ERROR",
                    "sql": query_data["sql"],
                    "diff": "ERROR",
                    "rows": []
                })

        self.log_dual("-" * 30, "info")
        return results
    
    def plot_drift(self, results: List[Dict[str, Any]], output_file: str = "diff_vs_agent.png"):
        """Plot the drift vs agent number"""
        # Filter out steps with errors for plotting
        valid_results = [r for r in results if r["step"] > 0 and isinstance(r["diff"], (int, float)) and r["diff"] != -1 and r["diff"] != "ERROR"]

        if not valid_results:
            self.log_dual("No valid results to plot. Skipping plot generation.", "error")
            return

        steps = [r["step"] for r in valid_results]
        diffs = [r["diff"] for r in valid_results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, diffs, marker='o', linestyle='-', linewidth=2, markersize=8)
        plt.xlabel("Agent Number", fontsize=12)
        plt.ylabel("Drift (|rows - rows_baseline|)", fontsize=12)
        plt.title("SQL Query Result Drift in Chinese Whispers Chain", fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Ensure all integer steps are shown on x-axis if there are few steps
        if steps:
            min_step = min(steps)
            max_step = max(steps)
            if max_step - min_step < 10: # Heuristic for few steps
                 plt.xticks(range(min_step, max_step + 1))
            else: # Otherwise, let matplotlib decide or set a reasonable number of ticks
                 plt.xticks(steps) # Show only the steps that have data
        
        # Add value labels on points
        for i, (x, y) in enumerate(zip(steps, diffs)):
            plt.annotate(f'{y}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        self.log_dual(f"Plot saved to {output_file}", "success")


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


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate SQL queries")
    parser.add_argument("--baseline", type=int, help="Baseline step for drift calculation. Defaults to the last agent's step if not provided.")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--run-folder", help="Specific run folder to process (default: latest)")
    args = parser.parse_args()

    print_info("=" * 50)
    print_info("SQL Query Evaluation")
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

    # Resolve configuration path
    config_path = args.config
    if not os.path.isabs(config_path) and not os.path.exists(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, args.config)

    # Load config (for potential future use, e.g. num_agents for default baseline)
    try:
        print_info(f"Loading configuration from: {config_path}")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print_success("Configuration loaded successfully")
        num_agents = config.get("num_agents", 5)  # Default to 5 if not in config
    except Exception as e:
        print_error(f"Failed to load config: {e}")
        num_agents = 5  # Fallback
        print_warning(f"Using default number of agents for baseline: {num_agents}")

    baseline_step = args.baseline if args.baseline is not None else num_agents
    print_info(f"Using baseline step: {baseline_step}")

    # Load queries from run folder
    queries_path = os.path.join(run_folder, "queries.jsonl")
    try:
        print_info(f"Loading queries from {queries_path}...")
        queries = []
        with open(queries_path) as f:
            for line in f:
                queries.append(json.loads(line))
        print_success(f"Loaded {len(queries)} queries")
    except Exception as e:
        print_error(f"Failed to load queries from {queries_path}: {e}")
        print_error("Make sure to run generate_sql.py first")
        return

    if not queries:
        print_error("No queries found in queries.jsonl. Cannot proceed.")
        return

    # Initialize evaluator with log file path
    try:
        log_file_path = os.path.join(run_folder, "evaluation.log")
        evaluator = SQLEvaluator(log_file_path=log_file_path)
        print_info(f"Evaluation log will be saved to: {log_file_path}")
    except Exception as e:
        print_error(f"Failed to initialize SQLEvaluator: {e}")
        return

    # Evaluate queries
    try:
        print_info("Starting SQL query evaluation...")
        results = evaluator.evaluate_queries(queries, baseline_step)
        print_success("Query evaluation completed.")
    except Exception as e:
        print_error(f"Query evaluation process failed: {e}")
        return

    # Save results to run folder
    if results:
        try:
            results_path = os.path.join(run_folder, "results.json")
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            print_success(f"Results saved to {results_path}")
        except Exception as e:
            print_error(f"Failed to save results.json: {e}")
    else:
        print_warning("No results to save.")

    # Plot drift and save to run folder
    try:
        if results:
            print_info("Generating drift plot...")
            plot_path = os.path.join(run_folder, "diff_vs_agent.png")
            evaluator.plot_drift(results, plot_path)
        else:
            print_warning("No results to plot.")
    except Exception as e:
        print_error(f"Failed to generate plot: {e}")
    
    print_info(f"All evaluation results saved to: {run_folder}")


if __name__ == "__main__":
    main() 