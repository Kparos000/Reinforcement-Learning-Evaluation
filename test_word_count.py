#!/usr/bin/env python3
"""Test word count for new scenario examples."""

import re

def _word_count(s: str) -> int:
    """Grader's word count function."""
    return len(re.findall(r"\b\w+\b", s))

# Test examples
examples = {
    "legal": "expires 12/31/2025, renewal $50,000, 30-day window, 90-day notice",
    "finance_v1": "open $127.50, high $134.20, close $131.85, 4.7M shares, P/E 23.4x",
    "finance_v2": "$127.50 open, $134.20 high, $131.85 close, volume 4.7M, P/E 23.4x",
    "sports_v1": "Lakers 112-98, LeBron 28pts, 9 reb, 7 ast, 47.3% FG, 14 3PT",
    "sports_v2": "W 112-98, James 28 points, 9 boards, 7 dimes, FG 47.3%, 14 threes",
}

print("Word count test (must be <= 16 words):\n")
for name, example in examples.items():
    count = _word_count(example)
    status = "[OK]" if count <= 16 else "[ERROR]"
    print(f"{status} {name:20} = {count:2} words | {example}")
