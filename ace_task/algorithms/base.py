"""
Base classes for RL algorithms.

This module defines the abstract interface that all RL algorithms must implement,
ensuring consistency and enabling easy comparison between methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import anthropic


@dataclass
class RLResult:
    """
    Container for RL algorithm results.

    Attributes:
        best_sample: The highest-reward output selected by the algorithm
        best_reward: The reward value of the best sample
        all_samples: List of all generated samples
        all_rewards: Corresponding reward values for each sample
        metadata: Additional algorithm-specific information
    """

    best_sample: str
    best_reward: float
    all_samples: list[str]
    all_rewards: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate that lists have consistent lengths."""
        if len(self.all_samples) != len(self.all_rewards):
            raise ValueError(
                f"Sample/reward length mismatch: {len(self.all_samples)} samples, "
                f"{len(self.all_rewards)} rewards"
            )

    @property
    def num_samples(self) -> int:
        """Total number of samples generated."""
        return len(self.all_samples)

    @property
    def avg_reward(self) -> float:
        """Average reward across all samples."""
        return sum(self.all_rewards) / len(self.all_rewards) if self.all_rewards else 0.0

    @property
    def success_rate(self) -> float:
        """Proportion of samples with reward > 0 (assuming binary rewards)."""
        if not self.all_rewards:
            return 0.0
        return sum(1 for r in self.all_rewards if r > 0) / len(self.all_rewards)


class RLAlgorithm(ABC):
    """
    Abstract base class for all RL algorithms.

    All RL strategies (Best-of-N, REINFORCE, PPO) implement this interface,
    enabling consistent usage and fair comparison.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 1.0,
        max_tokens: int = 1000,
    ):
        """
        Initialize the RL algorithm.

        Args:
            client: Anthropic API client for LLM calls
            model: Claude model identifier
            temperature: Sampling temperature (0=deterministic, >0=stochastic)
            max_tokens: Maximum tokens in generated output
        """
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def generate(
        self,
        prompt: str,
        reward_fn: Callable[[str], float],
        **kwargs,
    ) -> RLResult:
        """
        Generate output(s) using the RL algorithm.

        Args:
            prompt: Input prompt for the LLM
            reward_fn: Function that maps output text to scalar reward
            **kwargs: Algorithm-specific parameters

        Returns:
            RLResult containing the best output and all samples/rewards
        """
        pass

    def _call_llm(self, prompt: str) -> str:
        """
        Make a single LLM API call.

        Args:
            prompt: Input prompt

        Returns:
            Generated text response
        """
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
