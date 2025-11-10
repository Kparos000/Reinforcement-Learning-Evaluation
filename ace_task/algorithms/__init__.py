"""
Reinforcement Learning algorithms for ACE evaluation.

This module provides various RL strategies for improving LLM outputs:
- Best-of-N: Sample multiple outputs and select the best (Phase 1 ✓)
- REINFORCE: Policy gradient method (Phase 2 ✓)
- PPO: Proximal Policy Optimization (coming in Phase 3)
"""

from ace_task.algorithms.base import RLAlgorithm, RLResult
from ace_task.algorithms.best_of_n import BestOfNSampler
from ace_task.algorithms.rewards import BinaryRewardFunction, create_reward_function
from ace_task.algorithms.reinforce import REINFORCETrainer, REINFORCEConfig

__all__ = [
    "RLAlgorithm",
    "RLResult",
    "BestOfNSampler",
    "BinaryRewardFunction",
    "create_reward_function",
    "REINFORCETrainer",
    "REINFORCEConfig",
]
