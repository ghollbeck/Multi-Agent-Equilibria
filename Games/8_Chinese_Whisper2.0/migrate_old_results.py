#!/usr/bin/env python3
"""
migrate_old_results.py - Migrate old results to new organized structure
"""
import os
import shutil
import json
from datetime import datetime

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


def migrate_old_results():
    """Migrate old results from root directory to organized structure"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Files that should be moved to a migration folder
    old_files = [
        "history.jsonl",
        "queries.jsonl", 
        "results.json",
        "llm_inference_metrics.json",
        "diff_vs_agent.png"
    ]
    
    # Find SQL files
    sql_files = []
    for file in os.listdir(script_dir):
        if file.startswith("sql_step_") and file.endswith(".sql"):
            sql_files.append(file)
    
    old_files.extend(sql_files)
    
    # Check if any old files exist
    existing_files = []
    for file in old_files:
        file_path = os.path.join(script_dir, file)
        if os.path.exists(file_path):
            existing_files.append(file)
    
    if not existing_files:
        print_info("No old result files found in root directory")
        return
    
    print_info(f"Found {len(existing_files)} old result files:")
    for file in existing_files:
        print(f"  📄 {file}")
    
    # Create migration folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(script_dir, "results")
    migration_folder = os.path.join(results_dir, f"migrated_run_{timestamp}")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(migration_folder, exist_ok=True)
    
    print_info(f"Created migration folder: {migration_folder}")
    
    # Move files
    moved_count = 0
    for file in existing_files:
        old_path = os.path.join(script_dir, file)
        new_path = os.path.join(migration_folder, file)
        
        try:
            shutil.move(old_path, new_path)
            print_success(f"Moved {file}")
            moved_count += 1
        except Exception as e:
            print_error(f"Failed to move {file}: {e}")
    
    # Create a migration info file
    migration_info = {
        "migration_timestamp": datetime.utcnow().isoformat(),
        "migrated_files": existing_files,
        "migration_reason": "Migrated from root directory to organized structure",
        "original_location": script_dir
    }
    
    info_file = os.path.join(migration_folder, "migration_info.json")
    with open(info_file, "w") as f:
        json.dump(migration_info, f, indent=2)
    
    print_success(f"Successfully migrated {moved_count} files to {migration_folder}")
    print_info("Migration complete! Old files are now organized in the results folder.")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate old results to organized structure")
    parser.add_argument("--confirm", action="store_true", help="Confirm migration without prompt")
    args = parser.parse_args()
    
    print_info("=" * 50)
    print_info("Old Results Migration Tool")
    print_info("=" * 50)
    
    if not args.confirm:
        print_warning("This will move old result files from the root directory to an organized structure.")
        print_warning("Files will be moved to: results/migrated_run_<timestamp>/")
        response = input("Do you want to continue? (y/N): ")
        if response.lower() != 'y':
            print_info("Migration cancelled")
            return
    
    try:
        migrate_old_results()
    except Exception as e:
        print_error(f"Migration failed: {e}")
        return
    
    print_success("Migration completed successfully!")


if __name__ == "__main__":
    main() 