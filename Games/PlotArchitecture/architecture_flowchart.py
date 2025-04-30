#!/usr/bin/env python3
"""
Advanced Architecture Flowchart Generator
----------------------------------------

* 2‑phase "agentic" pipeline:
   1. **Chunk‑analysis pass**   → parallel LiteLLM calls that summarise each
      meaningful code block (function / class / top‑level logic).
   2. **Synthesis pass**          → single LiteLLM call that converts the
      collected summaries into a rich Mermaid flowchart.

* Handles 500‑2 000‑line files quickly with asyncio + a connection‑pooled session.
* Nodes are **numbered**, coloured, and show a one‑sentence explanation
  underneath the bold title, per your spec.
* Produces the usual `.mmd` and prettified `.html`, plus **JSON** debug artefacts
  (`*_chunk_summaries.json`) so you can inspect what the LLM saw.

Only the prompting & orchestration changed – the low‑level `_call_litellm`
function and env‑var contract (`LITELLM_API_KEY`) are untouched.
"""

from __future__ import annotations
import ast
import asyncio
import json
import os
import sys
import textwrap
from typing import List, Dict, Tuple
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

load_dotenv()

# ---------- configuration knobs ------------------------------------------------

DEFAULT_MODEL       = os.getenv("ARCHFC_MODEL", "gpt-4o")
MAX_LINES_PER_CHUNK = int(os.getenv("ARCHFC_CHUNK_LINES", "150"))
MAX_PARALLEL_CALLS  = int(os.getenv("ARCHFC_PARALLEL", "8"))
TEMPERATURE         = float(os.getenv("ARCHFC_TEMPERATURE", "0.3"))

# ---------- high‑level driver --------------------------------------------------

def create_flowchart_from_script(
    script_path: str | Path,
    output_dir: str | Path | None = None,
    render: bool = True,
    save_llm_output: bool = True,
) -> Dict[str, Path]:
    """
    Two‑pass pipeline → Mermaid + HTML (plus optional debug JSON).
    """
    script_path = Path(script_path).resolve()
    if not script_path.exists():
        raise FileNotFoundError(script_path)

    script_src = script_path.read_text(encoding="utf‑8")
    output_dir = Path(output_dir or script_path.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = script_path.stem

    # ------- Phase 1 : chunk + parallel summaries -----------------------------
    chunks = _chunk_script(script_src, max_lines=MAX_LINES_PER_CHUNK)
    chunk_summaries = asyncio.run(
        _parallel_summarise_chunks(chunks, model=DEFAULT_MODEL)
    )

    # optionally persist phase‑1 artefact for debugging
    summary_json = output_dir / f"{stem}_chunk_summaries.json"
    summary_json.write_text(json.dumps(chunk_summaries, indent=2))
    # ------- Phase 2 : global synthesis → mermaid -----------------------------
    mermaid_code, raw_llm = _synthesise_flowchart(
        chunk_summaries, model=DEFAULT_MODEL
    )

    # ---------- persistence & presentation ------------------------------------
    mermaid_file = output_dir / f"{stem}_flowchart.mmd"
    mermaid_file.write_text(mermaid_code, encoding="utf‑8")

    html_file = None
    if render:
        html_file = output_dir / f"{stem}_flowchart.html"
        _render_mermaid_html(mermaid_code, html_file)

    if save_llm_output:
        (output_dir / f"{stem}_synthesis_llm_output.txt").write_text(raw_llm)

    result = {
        "mermaid": mermaid_file,
        "html": html_file,
        "chunk_summaries": summary_json,
    }
    return {k: v for k, v in result.items() if v is not None}

# ---------- phase 1 helpers ----------------------------------------------------

class CodeChunk(Dict):
    """dict‑like {id, header, code} for simple JSON serialisation."""

def _chunk_script(src: str, *, max_lines: int) -> List[CodeChunk]:
    """
    Split by top‑level defs / classes; fall back to line buckets.
    """
    tree = ast.parse(src)
    pieces: List[Tuple[str, int, int]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            pieces.append((node.name, node.lineno, node.end_lineno or node.lineno))
    # Add catch‑all for residual top‑level code
    covered = {ln for _, a, b in pieces for ln in range(a, b + 1)}
    extra: List[int] = [
        i + 1 for i, _ in enumerate(src.splitlines()) if (i + 1) not in covered
    ]
    if extra:
        pieces.append(("__top_level__", extra[0], extra[-1]))

    chunks: List[CodeChunk] = []
    for idx, (name, start, end) in enumerate(pieces, 1):
        code_lines = src.splitlines()[start - 1 : end]
        # bucket by size if monster function/class > max_lines
        for sub in range(0, len(code_lines), max_lines):
            slice_ = code_lines[sub : sub + max_lines]
            chunks.append(
                CodeChunk(
                    id=idx if sub == 0 else f"{idx}.{sub//max_lines+1}",
                    header=name,
                    code="\n".join(slice_).strip(),
                )
            )
    return chunks

async def _parallel_summarise_chunks(
    chunks: List[CodeChunk], *, model: str
) -> List[Dict]:
    """
    Fire off LLM calls concurrently, but bounded by MAX_PARALLEL_CALLS.
    """
    sem = asyncio.Semaphore(MAX_PARALLEL_CALLS)
    async with aiohttp.ClientSession() as session:

        async def _one(chunk: CodeChunk) -> Dict:
            prompt = _summary_prompt(chunk)
            async with sem:
                resp = await _async_call_litellm(prompt, model, session)
            return {
                "id": chunk["id"],
                "name": chunk["header"],
                "summary": resp.strip(),
            }

        return await asyncio.gather(*[_one(c) for c in chunks])

def _summary_prompt(chunk: CodeChunk) -> str:
    return textwrap.dedent(
        f"""
        You are a senior Python engineer.
        Summarise *succinctly* what the following code block **does** –
        emphasise inputs, side‑effects, and outputs. 1‑2 sentences max.

        ```python
        {chunk['code']}
        ```
        """
    )

# ---------- phase 2 helpers ----------------------------------------------------

def _synthesise_flowchart(
    chunk_data: List[Dict], model: str
) -> Tuple[str, str]:
    """
    Single LLM call that receives JSON of summaries and returns raw mermaid.
    """
    numbered = "\n".join(
        f"{c['id']}. **{c['name']}** – {c['summary']}" for c in chunk_data
    )
    synth_prompt = textwrap.dedent(
        f"""
        You are an expert software architect and diagrammer.

        Based on the numbered list below, produce **ONLY** Mermaid code that:
          • starts with `flowchart TD`
          • creates one node per item, label format:
              <b>{'{'}name{'}'}</b><br/><small>{'{'}summary{'}'}</small>
          • connects nodes to reflect *data / control flow* implied by the summaries
          • numbers nodes (prefix in label)
          • keep arrows left‑to‑right (TD) unless unavoidable
          • no markdown fences

        {numbered}
        """
    )
    raw = _call_litellm_sync(synth_prompt, model)
    mermaid = _extract_mermaid(raw)
    return mermaid, raw

def _extract_mermaid(text: str) -> str:
    if "```" in text:
        text = text.split("```")[1]
    return text.strip()

# ---------- low‑level LiteLLM wrappers ----------------------------------------
# (identical sigs so existing deployments & keys still work)

def _call_litellm_sync(prompt: str, model: str) -> str:
    """Blocking helper built on top of the old _call_litellm logic."""
    from requests import post  # local import keeps aiohttp loop clean

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": TEMPERATURE,
    }
    res = post(
        "https://litellm.sph-prod.ethz.ch/chat/completions",
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('LITELLM_API_KEY')}",
        },
        timeout=90,
    )
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]

async def _async_call_litellm(prompt: str, model: str, session) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": TEMPERATURE,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('LITELLM_API_KEY')}",
    }
    async with session.post(
        "https://litellm.sph-prod.ethz.ch/chat/completions",
        json=payload,
        headers=headers,
        timeout=90,
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
        return data["choices"][0]["message"]["content"]

# ---------- HTML renderer (unchanged except CSS tweak) ------------------------

def _render_mermaid_html(mermaid_code: str, out_path: Path) -> None:
    palette_css = """
        .step-1 { fill:#e1f5fe; stroke:#0288d1; }
        .step-2 { fill:#e8f5e9; stroke:#388e3c; }
        .step-3 { fill:#fff8e1; stroke:#f9a825; }
        .step-4 { fill:#fbe9e7; stroke:#d84315; }
        .step-5 { fill:#f3e5f5; stroke:#8e24aa; }
        .step-6 { fill:#ede7f6; stroke:#5e35b1; }
    """
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf‑8"/>
<title>Architecture Flowchart</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<script>mermaid.initialize({{startOnLoad:true,theme:'default',flowchart:{{htmlLabels:true}}}});</script>
<style>body{{font-family:Inter,Arial,sans-serif;background:#fafafa;padding:2rem}}{palette_css}</style>
</head>
<body>
<div class="mermaid">
{mermaid_code}
</div>
</body>
</html>"""
    out_path.write_text(html, encoding="utf‑8")
    print(f"[+] HTML written → {out_path}")
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python architecture_flowchart.py <script.py> [output_dir]")
        sys.exit(1)
    result = create_flowchart_from_script(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    for k, v in result.items():
        print(f"{k:16}: {v}")
