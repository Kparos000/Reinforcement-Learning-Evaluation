"""
Utilities for building prompts from the shared task specification.

This module centralizes prompt construction so both evaluation and training
entrypoints can stay aligned with the same instructions in `prompt.txt`.
"""

from __future__ import annotations

from math import floor
from pathlib import Path
from typing import Tuple

from .scenarios.base import Scenario

# Path to the canonical task prompt used across scripts
PROMPT_PATH = Path(__file__).with_name("prompt.txt")


def compute_limits(
    scenario: Scenario,
    concision_limit: float | None = None,
    word_cap: int | None = None,
) -> Tuple[int, int]:
    """
    Derive max character and word limits for a scenario.

    Args:
        scenario: Scenario with original text and optional caps.
        concision_limit: Override for concision ratio (0-1). Defaults to scenario or 0.60.
        word_cap: Override for word cap. Defaults to scenario or 16.

    Returns:
        (max_chars, max_words)
    """
    limit = concision_limit or getattr(scenario, "concision_limit", None) or 0.60
    max_chars = floor(len(scenario.original) * limit)
    max_words = word_cap or getattr(scenario, "word_cap", None) or 16
    return max_chars, max_words


def build_user_message(max_chars: int, max_words: int, scenario: Scenario) -> str:
    """Combine the task prompt with fixtures and numeric caps."""
    with PROMPT_PATH.open("r", encoding="utf-8") as f:
        task_text = f.read()

    facts_lines = "\n".join(f"- {f}" for f in scenario.facts)
    banned_lines = "\n".join(f"- {b}" for b in sorted(scenario.banned))

    hard_limit = (
        f"\nHARD LIMITS FOR THIS RUN:\n"
        f"- MAX_CHARS for rewrite: {max_chars}\n"
        f"- MAX_WORDS for rewrite: {max_words}\n"
        f"If your rewrite exceeds {max_chars} characters OR {max_words} words, the submission FAILS.\n"
    )

    return (
        f"{task_text}\n"
        f"{hard_limit}\n"
        f"ORIGINAL:\n{scenario.original}\n\n"
        f"FACTS:\n{facts_lines}\n\n"
        f"BANNED:\n{banned_lines}\n"
    )

