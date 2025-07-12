#!/usr/bin/env python3
"""
run_complete_pipeline.py - Run the complete Chinese Whispers SQL pipeline
"""
import asyncio
import subprocess
import sys
import os
from datetime import datetime

# Import story definitions
from story_definitions import list_stories

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

def print_header(message: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{message}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}\n")


def run_command(command: str, description: str) -> bool:
    """Run a command and return True if successful"""
    print_info(f"Running: {description}")
    print_info(f"Command: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        
        print_success(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print_error(f"❌ {description} failed with exit code {e.returncode}")
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            print(f"STDERR:\n{e.stderr}")
        return False
    except Exception as e:
        print_error(f"❌ {description} failed with error: {e}")
        return False


def list_run_results(run_folder: str):
    """List the contents of a run folder"""
    print_header("RUN RESULTS SUMMARY")
    print_info(f"Run folder: {run_folder}")
    
    if not os.path.exists(run_folder):
        print_error(f"Run folder does not exist: {run_folder}")
        return
    
    print_info("Generated files:")
    for file in sorted(os.listdir(run_folder)):
        file_path = os.path.join(run_folder, file)
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path)
            print(f"  📄 {file} ({file_size} bytes)")
    
    # Show SQL files specifically
    sql_files = [f for f in os.listdir(run_folder) if f.startswith('sql_step_') and f.endswith('.sql')]
    if sql_files:
        print_info(f"\nSQL files generated: {len(sql_files)}")
        for sql_file in sorted(sql_files):
            print(f"  🗄️  {sql_file}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run complete Chinese Whispers SQL pipeline")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--story", default=None, help="Story name from story_definitions.py")
    parser.add_argument("--list-stories", action="store_true", help="List available stories and exit")
    parser.add_argument("--num-agents", type=int, default=None, help="Number of agents (overrides config)")
    parser.add_argument("--skip-simulate", action="store_true", help="Skip simulation step")
    parser.add_argument("--skip-generate", action="store_true", help="Skip SQL generation step")
    parser.add_argument("--skip-evaluate", action="store_true", help="Skip evaluation step")
    args = parser.parse_args()
    
    if args.list_stories:
        list_stories()
        return
    
    print_header("CHINESE WHISPERS SQL PIPELINE")
    print_info(f"Starting pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.story:
        print_info(f"Using story: {args.story}")
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print_info(f"Working directory: {script_dir}")
    
    success = True
    
    # Build command arguments
    story_arg = f" --story {args.story}" if args.story else ""
    num_agents_arg = f" --num-agents {args.num_agents}" if args.num_agents else ""
    
    # Step 1: Run simulation
    if not args.skip_simulate:
        print_header("STEP 1: RUNNING SIMULATION")
        success &= run_command(
            f"python simulate_chain.py --config {args.config}{story_arg}{num_agents_arg}",
            "Chinese Whispers simulation"
        )
    else:
        print_info("Skipping simulation step")
    
    # Step 2: Generate SQL
    if success and not args.skip_generate:
        print_header("STEP 2: GENERATING SQL QUERIES")
        success &= run_command(
            f"python generate_sql.py --config {args.config}",
            "SQL query generation"
        )
    else:
        print_info("Skipping SQL generation step")
    
    # Step 3: Evaluate SQL
    if success and not args.skip_evaluate:
        print_header("STEP 3: EVALUATING SQL QUERIES")
        success &= run_command(
            f"python evaluate_sql.py --config {args.config}",
            "SQL query evaluation"
        )
    else:
        print_info("Skipping SQL evaluation step")
    
    # Show results
    if success:
        try:
            # Find the latest run folder
            results_dir = os.path.join(script_dir, "results")
            if os.path.exists(results_dir):
                import glob
                run_folders = glob.glob(os.path.join(results_dir, "run_*"))
                if run_folders:
                    latest_run = max(run_folders, key=os.path.getctime)
                    list_run_results(latest_run)
                else:
                    print_warning("No run folders found")
            else:
                print_warning("Results directory not found")
        except Exception as e:
            print_error(f"Failed to list results: {e}")
        
        print_header("PIPELINE COMPLETED SUCCESSFULLY")
        print_success("All steps completed successfully!")
        print_info("Check the results folder for organized output files")
        
        if args.story:
            print_info(f"Story used: {args.story}")
    else:
        print_header("PIPELINE FAILED")
        print_error("Pipeline failed at one or more steps")
        sys.exit(1)


if __name__ == "__main__":
    main() 