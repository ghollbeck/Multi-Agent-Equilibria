#!/usr/bin/env python3
"""
Per‑game wrapper that calls the advanced flowchart generator.

Place this file inside a game directory (`1_Prisoners_Dilemma/plot_architecture.py`,
etc.).  It automatically locates the *largest* `.py` file in the folder (assumed
to be the game entry‑point), but you can override via CLI.

usage: python plot_architecture.py            # auto‑detect main script  
       python plot_architecture.py mygame.py  # explicit
"""
from __future__ import annotations
import sys
import os
from pathlib import Path

# ─── locate shared module ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]      # …/Games
sys.path.insert(0, str(ROOT))                   # Add …/Games to sys.path
sys.path.append(str(ROOT / "PlotArchitecture"))

from PlotArchitecture.architecture_flowchart import create_flowchart_from_script


def _default_target() -> Path:
    py_files = sorted(Path(__file__).parent.glob("*.py"), key=lambda p: p.stat().st_size, reverse=True)
    for f in py_files:
        if f.name != Path(__file__).name:
            return f
    raise FileNotFoundError("No candidate .py file found.")

def main():
    script = Path(sys.argv[1]) if len(sys.argv) > 1 else _default_target()
    out_dir = Path(__file__).parent / "architecture"
    out_dir.mkdir(exist_ok=True)
    print(f"Generating flowchart for {script.name} → {out_dir}")
    files = create_flowchart_from_script(script, out_dir, render=True, save_llm_output=True)
    print("Artifacts:")
    for k, v in files.items():
        print(f"  {k:<15} {v.relative_to(Path.cwd())}")

if __name__ == "__main__":
    main()
