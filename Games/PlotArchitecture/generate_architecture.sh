#!/usr/bin/env bash
# ╭──────────────────────────────────────────────────────────╮
# │ Bulk flowchart generator for all or one game directory.  │
# │ Now accepts optional flags that propagate through to     │
# │ the Python pipeline (chunk size, model, etc.).           │
# ╰──────────────────────────────────────────────────────────╯

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GAMES=(
  "1_Prisoners_Dilemma"
  "2_MIT_Beer_Game"
  "3_Fishery_Game"
  "4_Market_Impact_Game"
)

usage() {
  echo "Usage: $0 [-c CHUNK_LINES] [-m MODEL] [game_number]"
  echo "  -c, --chunk-lines  Override ARCHFC_CHUNK_LINES (default 150)"
  echo "  -m, --model        Override ARCHFC_MODEL         (default gpt-4o)"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--chunk-lines) export ARCHFC_CHUNK_LINES="$2"; shift 2;;
    -m|--model)       export ARCHFC_MODEL="$2";        shift 2;;
    -h|--help)        usage;;
    [1-4])            TARGET_GAME="$1";               shift;;
    *) echo "Unknown argument $1"; usage;;
  esac
done

generate_for_game() {
  local game_dir="$SCRIPT_DIR/$1"
  local plot_script="$game_dir/plot_architecture.py"
  [[ -f "$plot_script" ]] || { echo "No plot_architecture.py in $game_dir"; return 1; }
  echo "▶  $1  (chunks:${ARCHFC_CHUNK_LINES:-150}, model:${ARCHFC_MODEL:-gpt-4o})"
  python3 "$plot_script"
}

if [[ -n "${TARGET_GAME:-}" ]]; then
  generate_for_game "${GAMES[$((10#$TARGET_GAME-1))]}"
else
  echo "Generating for all games…"
  for g in "${GAMES[@]}"; do generate_for_game "$g"; echo; done
fi

echo "✓ Done."
