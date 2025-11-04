"""
Best-of-N Sampling Algorithm.

This module implements the simplest yet effective RL strategy: generate N outputs
and select the one with the highest reward. No training required!

Mathematical Foundation:
    If a single sample has success probability p, then the probability that at
    least one of N samples succeeds is:

        P(success) = 1 - (1-p)^N

    This grows exponentially with N, providing dramatic improvements.

Example:
    >>> from anthropic import Anthropic
    >>> client = Anthropic()
    >>> sampler = BestOfNSampler(client, n=5)
    >>>
    >>> def reward_fn(text):
    ...     return 1.0 if "correct" in text else 0.0
    >>>
    >>> result = sampler.generate("Solve 2+2", reward_fn)
    >>> print(result.best_sample)  # Highest reward output
    >>> print(result.success_rate)  # Proportion that succeeded
"""

import time
from typing import Callable

from ace_task.algorithms.base import RLAlgorithm, RLResult


class BestOfNSampler(RLAlgorithm):
    """
    Best-of-N sampling: Generate N outputs and select the best.

    This is the foundation for more advanced RL techniques. It establishes
    a performance ceiling (what's possible with the base model) without
    any training overhead.

    Attributes:
        n: Number of samples to generate (higher = better quality but more cost)
        early_stop: If True, stop sampling once we achieve perfect reward (1.0)
    """

    def __init__(
        self,
        client,
        n: int = 5,
        early_stop: bool = True,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 1.0,
        max_tokens: int = 1000,
    ):
        """
        Initialize Best-of-N sampler.

        Args:
            client: Anthropic API client
            n: Number of samples to generate
            early_stop: Stop early if perfect reward is achieved
            model: Claude model identifier
            temperature: Sampling temperature (must be > 0 for diversity)
            max_tokens: Maximum tokens per generation
        """
        super().__init__(client, model, temperature, max_tokens)
        self.n = n
        self.early_stop = early_stop

        if self.temperature == 0 and self.n > 1:
            raise ValueError(
                "Best-of-N requires temperature > 0 for sample diversity. "
                f"Got temperature={self.temperature} with n={self.n}. "
                "Set temperature >= 0.7 or use n=1 for deterministic generation."
            )

    def generate(
        self,
        prompt: str,
        reward_fn: Callable[[str], float],
        verbose: bool = False,
    ) -> RLResult:
        """
        Generate N samples and return the one with highest reward.

        Args:
            prompt: Input prompt for LLM
            reward_fn: Function mapping output text to scalar reward
            verbose: If True, print progress during sampling

        Returns:
            RLResult with best sample, all samples, and rewards
        """
        samples = []
        rewards = []
        start_time = time.time()

        best_sample = None
        best_reward = float("-inf")

        for i in range(self.n):
            if verbose:
                print(f"  Generating sample {i+1}/{self.n}...", end=" ", flush=True)

            # Generate sample
            sample = self._call_llm(prompt)
            samples.append(sample)

            # Compute reward
            reward = reward_fn(sample)
            rewards.append(reward)

            # Track best
            if reward > best_reward:
                best_reward = reward
                best_sample = sample

            if verbose:
                print(f"reward={reward:.2f}", flush=True)

            # Early stopping: if we achieve perfect score, no need to continue
            if self.early_stop and reward >= 1.0:
                if verbose:
                    print(f"  Early stop: Perfect reward achieved at sample {i+1}")
                break

        elapsed = time.time() - start_time

        # Compute diversity metrics
        unique_samples = len(set(samples))
        diversity_ratio = unique_samples / len(samples)

        metadata = {
            "n_requested": self.n,
            "n_generated": len(samples),
            "early_stopped": len(samples) < self.n,
            "temperature": self.temperature,
            "elapsed_seconds": elapsed,
            "unique_samples": unique_samples,
            "diversity_ratio": diversity_ratio,
        }

        return RLResult(
            best_sample=best_sample,
            best_reward=best_reward,
            all_samples=samples,
            all_rewards=rewards,
            metadata=metadata,
        )

    def __repr__(self) -> str:
        return (
            f"BestOfNSampler(n={self.n}, model={self.model}, "
            f"temperature={self.temperature}, early_stop={self.early_stop})"
        )
