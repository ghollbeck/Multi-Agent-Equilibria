#!/usr/bin/env python3
"""
Internal script to generate architecture flowchart for the Oligopoly Simulation
"""

import os
import sys
from pathlib import Path

# Get the current file's directory
current_dir = Path(__file__).parent.absolute()

# Add the Games directory to the Python path to find the architecture_flowchart module
plot_architecture_dir = current_dir.parent / "PlotArchitecture"
sys.path.append(str(plot_architecture_dir))

# Import the main plotting script
# Import the main plotting script from the PlotArchitecture directory
from Games.PlotArchitecture.architecture_flowchart import create_flowchart_from_script

# Paths to the main game files
run_experiments_file = current_dir / "run_experiments.py"
oligopoly_file = current_dir / "oligopoly.py"
analyze_file = current_dir / "analyze.py"

# Output directory - create an "architecture" subfolder in the current directory
output_dir = current_dir / "architecture"
os.makedirs(output_dir, exist_ok=True)

def main():
    """Generate architecture flowcharts for the Oligopoly Simulation"""
    # Check if files exist
    missing_files = []
    for file_path in [run_experiments_file, oligopoly_file, analyze_file]:
        if not file_path.exists():
            missing_files.append(file_path.name)
    
    if missing_files:
        print(f"Error: The following files were not found: {', '.join(missing_files)}")
        sys.exit(1)
    
    # Generate flowcharts for each file
    for file_path in [run_experiments_file, oligopoly_file, analyze_file]:
        print(f"\nGenerating architecture flowchart for: {file_path}")
        print(f"Output directory: {output_dir}")
        
        # Create a specific output directory for each file
        file_output_dir = output_dir / file_path.stem
        os.makedirs(file_output_dir, exist_ok=True)
        
        # Generate the flowchart
        result_files = architecture_flowchart.create_flowchart_from_script(
            script_path=str(file_path),
            output_dir=str(file_output_dir),
            render=True,
            save_llm_output=True
        )
        
        print(f"\nFlowchart generation complete for {file_path.name}!")
        print(f"Generated files:")
        for file_type, file_path in result_files.items():
            print(f"- {file_type}: {file_path}")
    
    print("\nAll flowcharts generated successfully!")

if __name__ == "__main__":
    main()