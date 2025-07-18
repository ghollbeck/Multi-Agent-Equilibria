#!/usr/bin/env python3
"""
generate_dataset.py - Slice a single source text into multiple context lengths for Chinese Whisper 2.1 experiments.

Usage:
    python generate_dataset.py --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import yaml

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def tokenize(text: str) -> List[str]:
    """A simple whitespace tokenizer returning a list of *tokens* (words)."""
    # For reproducibility we keep it extremely simple; punctuation retained.
    return text.split()


def slice_story(tokens: List[str], length: int, offset: int) -> str:
    """Return a slice of length tokens starting at offset (wrap-around if needed)."""
    if len(tokens) < length:
        raise ValueError("Source text shorter than requested length.")

    start = offset % len(tokens)
    end = start + length

    # Simple circular slicing to avoid out-of-range
    slice_tokens = (tokens * 2)[start:end]
    return " ".join(slice_tokens)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def build_dataset(cfg: dict, out_path: Path) -> None:
    """Create dataset.jsonl according to config and save to out_path."""
    src_path = Path(cfg["source_text_path"]).expanduser()
    if not src_path.exists():
        raise FileNotFoundError(f"Source text not found: {src_path}")

    with open(src_path, "r", encoding="utf-8") as f:
        full_text = f.read().strip()

    tokens = tokenize(full_text)

    story_id = 0
    rows = []
    for length in cfg["context_lengths"]:
        for sample_idx in range(cfg["stories_per_length"]):
            offset = sample_idx * length  # deterministic but distinct
            story = slice_story(tokens, length, offset)
            rows.append({
                "id": story_id,
                "length": length,
                "text": story
            })
            story_id += 1

    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✅ Generated {len(rows)} stories → {out_path}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset for Chinese Whisper 2.1 experiments")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--output", default="dataset.jsonl", help="Output dataset path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_path = Path(args.output)
    build_dataset(cfg, output_path) 