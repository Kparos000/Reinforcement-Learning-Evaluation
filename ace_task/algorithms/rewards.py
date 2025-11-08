"""
Reward functions for RL algorithms.

This module converts grader outputs into scalar rewards that RL algorithms
can optimize. Different reward schemes enable different learning behaviors.
"""

from typing import Callable

from ace_task.grader import grade


def create_reward_function(
    scenario,
    reward_scheme: str = "binary",
) -> Callable[[str], float]:
    """
    Create a reward function for a specific scenario.

    Args:
        scenario: Scenario object with original, facts, banned attributes
        reward_scheme: Type of reward to compute:
            - "binary": 1.0 if pass, 0.0 if fail (default)
            - "partial": Future work - partial credit for some correct facts
            - "dense": Future work - fine-grained reward components

    Returns:
        Function that maps model output text to scalar reward

    Example:
        >>> from ace_task.scenarios import get_scenario
        >>> scenario = get_scenario("economics")
        >>> reward_fn = create_reward_function(scenario)
        >>> reward = reward_fn("GDP grew 3.2% in Q4 2024")
        >>> print(reward)  # 1.0 or 0.0
    """
    if reward_scheme == "binary":
        return BinaryRewardFunction(scenario)
    else:
        raise ValueError(f"Unknown reward scheme: {reward_scheme}")


class BinaryRewardFunction:
    """
    Binary reward: 1.0 if output passes grading, 0.0 otherwise.

    This is the simplest reward function and works well for Best-of-N sampling.
    For REINFORCE and PPO (Phases 2 & 3), we may want more nuanced rewards.

    Attributes:
        scenario: The scenario to evaluate against
        grader_config: Optional grader configuration overrides
    """

    def __init__(self, scenario, grader_config: dict | None = None):
        """
        Initialize binary reward function.

        Args:
            scenario: Scenario with original, facts, banned attributes
            grader_config: Optional dict with alias_map, concision_limit, word_cap
        """
        self.scenario = scenario
        self.grader_config = grader_config or {}

    def __call__(self, model_output: str) -> float:
        """
        Compute binary reward for model output.

        Args:
            model_output: Generated text to evaluate

        Returns:
            1.0 if passes all grading criteria, 0.0 otherwise
        """
        try:
            # Get alias_map from scenario if available, otherwise from grader_config
            alias_map = self.grader_config.get("alias_map")
            if alias_map is None and hasattr(self.scenario, 'alias_map'):
                alias_map = self.scenario.alias_map

            # Get concision_limit and word_cap from grader_config or use defaults
            concision_limit = self.grader_config.get("concision_limit", 0.60)
            word_cap = self.grader_config.get("word_cap", 16)

            passed, reason = grade(
                original=self.scenario.original,
                facts=self.scenario.facts,
                banned=self.scenario.banned,
                model_text=model_output,
                alias_map=alias_map,
                concision_limit=concision_limit,
                word_cap=word_cap,
            )
            return 1.0 if passed else 0.0
        except Exception as e:
            # If grading fails (e.g., malformed output), treat as failure
            print(f"Warning: Grading failed with error: {e}")
            return 0.0

    def __repr__(self) -> str:
        return f"BinaryRewardFunction(scenario={self.scenario.__class__.__name__})"


class PartialRewardFunction:
    """
    Partial credit reward: Decompose grading into components.

    FUTURE WORK (Phase 2: REINFORCE)

    This will enable the model to learn from partial success:
    - Fact accuracy: % of facts correctly included
    - Concision: Penalty for exceeding word limit
    - Banned words: Penalty for each banned word used
    - ACE insight: Bonus for novel insights

    Total reward = w1*facts + w2*concision + w3*banned + w4*insight
    """

    def __init__(self, scenario, weights: dict | None = None):
        """Initialize partial reward (placeholder for Phase 2)."""
        raise NotImplementedError("Partial rewards coming in Phase 2 (REINFORCE)")

    def __call__(self, model_output: str) -> float:
        raise NotImplementedError("Partial rewards coming in Phase 2 (REINFORCE)")
