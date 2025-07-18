from __future__ import annotations

"""metrics_utils.py - simple similarity metrics for memory-loss evaluation"""

import re
from typing import List

def whitespace_tokenize(text: str) -> List[str]:
    return text.split()

def token_overlap_ratio(source: str, target: str) -> float:
    """Return |intersection| / |source| token ratio."""
    src_tokens = set(whitespace_tokenize(source))
    tgt_tokens = set(whitespace_tokenize(target))
    if not src_tokens:
        return 0.0
    return len(src_tokens & tgt_tokens) / len(src_tokens) 