#!/usr/bin/env python3
"""
generate_sql.py - Generate SQL queries from Chinese Whispers history
"""
import json
import asyncio
from typing import List, Dict, Any, Optional
import os

import yaml
from dotenv import load_dotenv
from litellm import acompletion
from langsmith import traceable

load_dotenv()


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
        
        response = await acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        sql = response.choices[0].message.content.strip()
        # Clean up the SQL if it has markdown code blocks
        if sql.startswith("```"):
            sql = sql.split("```")[1]
            if sql.startswith("sql"):
                sql = sql[3:]
            sql = sql.strip()
        
        return sql


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate SQL from stories")
    parser.add_argument("--step", nargs="*", type=int, help="Specific steps to process")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load history
    history = []
    with open("history.jsonl") as f:
        for line in f:
            history.append(json.loads(line))
    
    # Determine which steps to process
    if args.step:
        steps_to_process = args.step
    else:
        steps_to_process = [entry["step"] for entry in history]
    
    # Generate SQL for each step
    generator = SQLGenerator(config)
    queries = []
    
    for entry in history:
        if entry["step"] not in steps_to_process:
            continue
        
        print(f"Generating SQL for step {entry['step']}...")
        sql = await generator.generate_sql(entry["story"])
        
        query_entry = {
            "step": entry["step"],
            "sql": sql
        }
        queries.append(query_entry)
        
        # Save individual SQL file
        sql_file = f"sql_step_{entry['step']}.sql"
        with open(sql_file, "w") as f:
            f.write(sql)
        print(f"  Saved to {sql_file}")
    
    # Save all queries to JSONL
    with open("queries.jsonl", "w") as f:
        for query in queries:
            f.write(json.dumps(query) + "\n")
    
    print(f"\nGenerated {len(queries)} SQL queries")
    print("Results saved to queries.jsonl and individual sql_step_*.sql files")


if __name__ == "__main__":
    asyncio.run(main()) 