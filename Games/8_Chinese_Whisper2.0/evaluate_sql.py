#!/usr/bin/env python3
"""
evaluate_sql.py - Execute SQL queries and measure drift in result-set size
"""
import json
import sqlite3
from typing import List, Dict, Any
import sys
import os

import yaml
import matplotlib.pyplot as plt

# Add parent directory to path to import StudentsDBManager
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from students_db_manager import StudentsDBManager


class SQLEvaluator:
    """Evaluate SQL queries against the students database"""
    
    def __init__(self):
        # Use in-memory database
        self.db_manager = StudentsDBManager(":memory:")
        self._populate_sample_data()
    
    def _populate_sample_data(self):
        """Populate database with sample data from students_db_manager.py"""
        sample_students = [
            ("Alice", "Johnson", 1, 0, 1, 1, 1, 1, 1),
            ("Bob", "Smith", 1, 1, 0, 1, 1, 0, 1),
            ("Charlie", "Brown", 0, 1, 1, 0, 1, 1, 1),
            ("Maria", "Garcia", 0, 1, 1, 1, 0, 0, 1),
            ("John", "Doe", 1, 0, 1, 1, 1, 1, 0)
        ]
        
        for student in sample_students:
            self.db_manager.add_student(*student)
    
    def execute_query(self, sql: str) -> int:
        """Execute a SQL query and return the row count"""
        try:
            with sqlite3.connect(":memory:") as conn:
                # Re-create and populate for each query to ensure clean state
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
                sample_students = [
                    ("Alice", "Johnson", 1, 0, 1, 1, 1, 1, 1),
                    ("Bob", "Smith", 1, 1, 0, 1, 1, 0, 1),
                    ("Charlie", "Brown", 0, 1, 1, 0, 1, 1, 1),
                    ("Maria", "Garcia", 0, 1, 1, 1, 0, 0, 1),
                    ("John", "Doe", 1, 0, 1, 1, 1, 1, 0)
                ]
                
                for student in sample_students:
                    cursor.execute('''
                        INSERT INTO students (first_name, last_name, speaks_english, speaks_spanish, 
                                            grade_math_pass, grade_science_pass, grade_english_pass, 
                                            is_highschool, is_active)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', student)
                
                # Execute the query
                cursor.execute(sql)
                rows = cursor.fetchall()
                return len(rows)
                
        except Exception as e:
            print(f"Error executing query: {e}")
            print(f"Query: {sql}")
            return 0
    
    def evaluate_queries(self, queries: List[Dict[str, Any]], baseline_step: int) -> List[Dict[str, Any]]:
        """Evaluate all queries and compute drift"""
        results = []
        baseline_count = None
        
        # First, find the baseline count
        for query in queries:
            if query["step"] == baseline_step:
                baseline_count = self.execute_query(query["sql"])
                break
        
        if baseline_count is None:
            print(f"Warning: baseline step {baseline_step} not found, using last step")
            baseline_count = self.execute_query(queries[-1]["sql"])
        
        # Evaluate all queries
        for query in queries:
            row_count = self.execute_query(query["sql"])
            diff = abs(row_count - baseline_count)
            
            result = {
                "step": query["step"],
                "row_count": row_count,
                "diff": diff
            }
            results.append(result)
            
            print(f"Step {query['step']}: {row_count} rows (diff: {diff})")
        
        return results
    
    def plot_drift(self, results: List[Dict[str, Any]], output_file: str = "diff_vs_agent.png"):
        """Plot the drift vs agent number"""
        steps = [r["step"] for r in results if r["step"] > 0]  # Exclude initial story
        diffs = [r["diff"] for r in results if r["step"] > 0]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, diffs, marker='o', linestyle='-', linewidth=2, markersize=8)
        plt.xlabel("Agent Number", fontsize=12)
        plt.ylabel("Drift (|rows - rows_baseline|)", fontsize=12)
        plt.title("SQL Query Result Drift in Chinese Whispers Chain", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(steps)
        
        # Add value labels on points
        for i, (x, y) in enumerate(zip(steps, diffs)):
            plt.annotate(f'{y}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"Plot saved to {output_file}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate SQL queries")
    parser.add_argument("--baseline", type=int, default=5, help="Baseline step for drift calculation")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()
    
    # Load config (for potential future use)
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load queries
    queries = []
    with open("queries.jsonl") as f:
        for line in f:
            queries.append(json.loads(line))
    
    if not queries:
        print("No queries found in queries.jsonl")
        return
    
    # Initialize evaluator
    evaluator = SQLEvaluator()
    
    # Evaluate queries
    print("Evaluating SQL queries...")
    results = evaluator.evaluate_queries(queries, args.baseline)
    
    # Save results
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to results.json")
    
    # Plot drift
    evaluator.plot_drift(results)


if __name__ == "__main__":
    main() 