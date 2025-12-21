"""
Reward functions for RL algorithms.
Core bridge from deterministic grading to scalar rewards; used by evaluation,
Best-of-N sampling, and experimental training loops.

This module converts grader outputs into scalar rewards that RL algorithms
can optimize. Provides both binary (pass/fail) and dense (partial credit) rewards.
"""

from typing import Callable

from ace_task.grader import GradeResult, grade_detailed


def dense_reward_from_grade(grade: GradeResult) -> float:
    """
    Compute a dense reward in [0,1] from a GradeResult.

    - Base: coverage = facts_matched / facts_total
    - Penalty for banned terms: -0.1 if any banned terms hit
    - Penalty for length violation: -0.05 if length violation
    - Clipped to [0, 1]
    """
    coverage = grade.facts_matched / max(grade.facts_total, 1)
    reward = coverage

    if grade.banned_term_violations > 0:
        reward -= 0.10
    if grade.length_violation:
        reward -= 0.05

    return max(0.0, min(1.0, reward))


def binary_reward_from_grade(grade: GradeResult) -> float:
    """Return 1.0 if passed, else 0.0."""
    return 1.0 if grade.passed else 0.0


def compute_binary_reward(
    scenario,
    model_output: str,
    grader_config: dict | None = None,
) -> float:
    """
    Compute binary reward (1/0) by grading the model output.
    """
    grader_config = grader_config or {}
    grade = _grade_output(scenario, model_output, grader_config)
    return binary_reward_from_grade(grade)


def compute_dense_reward(
    scenario,
    model_output: str,
    grader_config: dict | None = None,
) -> float:
    """
    Compute dense reward (0..1) by grading the model output.
    """
    grader_config = grader_config or {}
    grade = _grade_output(scenario, model_output, grader_config)
    return dense_reward_from_grade(grade)


def create_reward_function(
    scenario,
    reward_scheme: str = "binary",
    grader_config: dict | None = None,
) -> Callable[[str], float]:
    """
    Create a reward function for a specific scenario.

    Args:
        scenario: Scenario object with original, facts, banned attributes
        reward_scheme: "binary" or "dense"
        grader_config: Optional overrides for grading (concision_limit, word_cap, alias_map)
    """
    grader_config = grader_config or {}

    if reward_scheme == "binary":
        return BinaryRewardFunction(scenario, grader_config=grader_config)
    if reward_scheme == "dense":
        return DenseRewardFunction(scenario, grader_config=grader_config)
    raise ValueError(f"Unknown reward scheme: {reward_scheme}")


def _grade_output(scenario, model_output: str, grader_config: dict) -> GradeResult:
    alias_map = grader_config.get("alias_map")
    if alias_map is None and hasattr(scenario, "alias_map"):
        alias_map = scenario.alias_map

    concision_limit = grader_config.get("concision_limit")
    word_cap = grader_config.get("word_cap")

    return grade_detailed(
        original=scenario.original,
        facts=scenario.facts,
        banned=scenario.banned,
        model_text=model_output,
        alias_map=alias_map,
        concision_limit=concision_limit,
        word_cap=word_cap,
    )


class BinaryRewardFunction:
    """Binary reward: 1.0 if output passes grading, 0.0 otherwise."""

    def __init__(self, scenario, grader_config: dict | None = None):
        self.scenario = scenario
        self.grader_config = grader_config or {}

    def __call__(self, model_output: str) -> float:
        try:
            grade = _grade_output(self.scenario, model_output, self.grader_config)
            return binary_reward_from_grade(grade)
        except Exception as e:
            print(f"Warning: Grading failed with error: {e}")
            return 0.0

    def __repr__(self) -> str:
        return f"BinaryRewardFunction(scenario={self.scenario.__class__.__name__})"


class DenseRewardFunction:
    """Dense reward: partial credit based on facts covered and penalties for violations."""

    def __init__(self, scenario, grader_config: dict | None = None):
        self.scenario = scenario
        self.grader_config = grader_config or {}

    def __call__(self, model_output: str) -> float:
        try:
            grade = _grade_output(self.scenario, model_output, self.grader_config)
            return dense_reward_from_grade(grade)
        except Exception as e:
            print(f"Warning: Grading failed with error: {e}")
            return 0.0

    def __repr__(self) -> str:
        return f"DenseRewardFunction(scenario={self.scenario.__class__.__name__})"
