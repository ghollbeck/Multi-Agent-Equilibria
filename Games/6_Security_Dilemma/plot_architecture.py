#!/usr/bin/env python3
"""
Internal script to generate architecture flowchart for the Market Impact Game
"""

import os
import sys
from pathlib import Path

# Get the current file's directory
current_dir = Path(__file__).parent.absolute()

# Add the PlotArchitecture directory to the Python path to find the architecture_flowchart module
plot_architecture_dir = current_dir.parent / "PlotArchitecture"
sys.path.append(str(plot_architecture_dir))

# Import the main plotting script
import architecture_flowchart

# Path to the main game file
game_file = current_dir / "play_security_dilemma.py"

# Output directory - create an "architecture" subfolder in the current directory
output_dir = current_dir / "architecture"
os.makedirs(output_dir, exist_ok=True)


def main():
    """Generate architecture flowchart for the Market Impact Game"""
    print(f"Generating architecture flowchart for: {game_file}")
    print(f"Output directory: {output_dir}")
    
    # Generate the flowchart
    result_files = architecture_flowchart.create_flowchart_from_script(
        script_path=str(game_file),
        output_dir=str(output_dir),
        render=True,
        save_llm_output=True
    )
    
    print("\nFlowchart generation complete!")
    print(f"Generated files:")
    for file_type, file_path in result_files.items():
        print(f"- {file_type}: {file_path}")

if __name__ == "__main__":
    main()