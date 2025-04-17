#!/bin/bash
# Generate architecture flowcharts for game files
# Usage: ./generate_architecture.sh [game_number]
# Example: ./generate_architecture.sh 4 (generates for Market Impact Game)

# Directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Games available
GAMES=(
  "1_Prisoners_Dilemma"
  "2_MIT_Beer_Game"
  "3_Fishery_Game"
  "4_Market_Impact_Game"
)

# Function to generate architecture flowchart for a specific game
generate_for_game() {
  local game_dir="$SCRIPT_DIR/$1"
  local plot_script="$game_dir/plot_architecture.py"
  
  if [ ! -f "$plot_script" ]; then
    echo "Error: Plot script not found at $plot_script"
    echo "Make sure the plot_architecture.py script exists in the game directory"
    return 1
  fi
  
  echo "Generating architecture flowchart for $1..."
  python3 "$plot_script"
  
  if [ $? -eq 0 ]; then
    echo "✅ Architecture flowchart generated successfully for $1"
  else
    echo "❌ Failed to generate architecture flowchart for $1"
  fi
}

# Check if a game number was provided
if [ $# -eq 1 ]; then
  if [[ $1 =~ ^[1-4]$ ]]; then
    # Generate for the specified game only
    game_index=$((10#$1 - 1))
    generate_for_game "${GAMES[$game_index]}"
  else
    echo "Error: Invalid game number. Please specify a number between 1 and 4."
    exit 1
  fi
else
  # No game number provided, generate for all games
  echo "Generating architecture flowcharts for all games..."
  
  for game in "${GAMES[@]}"; do
    generate_for_game "$game"
    echo ""
  done
fi

echo "All done!"