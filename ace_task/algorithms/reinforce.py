"""
REINFORCE Algorithm Implementation (Policy Gradient)

This implements the classic REINFORCE algorithm (Williams, 1992) for training
language models to generate compressed text while preserving facts.

Key differences from Best-of-N:
- Best-of-N: Sample multiple, pick best (no learning)
- REINFORCE: Learn to generate good outputs directly via policy gradients

Algorithm:
1. Generate output from current policy (language model)
2. Compute reward (from grader)
3. Compute policy gradient: ∇log π(a|s) * (R - baseline)
4. Update model parameters to increase probability of high-reward actions
"""

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class REINFORCEConfig:
    """Configuration for REINFORCE training."""

    learning_rate: float = 1e-5
    gamma: float = 1.0  # Discount factor (1.0 for single-step tasks)
    baseline_type: str = "moving_average"  # none, moving_average, learned
    baseline_decay: float = 0.9  # For moving average baseline
    max_length: int = 512  # Max generation length
    temperature: float = 1.0  # Sampling temperature
    batch_size: int = 1  # Number of samples per iteration
    gradient_clip: float = 1.0  # Gradient clipping to prevent instability


class REINFORCETrainer:
    """
    REINFORCE trainer for policy gradient learning.

    This trainer implements the REINFORCE algorithm to fine-tune a language model
    to generate high-reward outputs (successful ACE compressions).
    """

    def __init__(
        self,
        model_name: str,
        reward_fn: Callable[[str], float],
        config: Optional[REINFORCEConfig] = None,
        device: str = "cpu",
    ):
        """
        Initialize REINFORCE trainer.

        Args:
            model_name: HuggingFace model name (e.g., "gpt2", "gpt2-medium")
            reward_fn: Function that maps generated text to reward (0.0 or 1.0)
            config: REINFORCE configuration
            device: Device to run on ("cpu" or "cuda")
        """
        self.config = config or REINFORCEConfig()
        self.device = device

        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        # Reward function
        self.reward_fn = reward_fn

        # Baseline for variance reduction
        self.baseline = 0.0

        # Training statistics
        self.step = 0
        self.total_reward = 0.0

    def generate(self, prompt: str) -> tuple[str, torch.Tensor, list[int]]:
        """
        Generate output from current policy and return log probabilities.

        Args:
            prompt: Input prompt

        Returns:
            Tuple of (generated_text, log_probs, token_ids)
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        prompt_len = input_ids.shape[1]

        # For ACE task, we need ~200-400 tokens for JSON output
        # Don't generate more than necessary (faster on CPU!)
        max_generation = 400  # Enough for JSON with facts

        if self.config.max_length < max_generation:
            max_generation = self.config.max_length

        print(f"Generating up to {max_generation} tokens (prompt: {prompt_len} tokens)...")

        # Generate with early stopping (stops at EOS or max tokens)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_generation,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                # Early stopping - stop when EOS is generated
                early_stopping=True,
            )

        # Get generated tokens (excluding prompt)
        generated_ids = outputs.sequences[0][input_ids.shape[1] :]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Compute log probabilities for each generated token
        # This is needed for policy gradient: ∇log π(a|s)
        log_probs = []
        for i, token_id in enumerate(generated_ids):
            # Get logits for this step
            logits = outputs.scores[i][0]  # Shape: (vocab_size,)
            # Convert to log probabilities
            log_prob = F.log_softmax(logits, dim=-1)
            # Get log prob of selected token
            log_probs.append(log_prob[token_id].item())

        log_probs_tensor = torch.tensor(log_probs, device=self.device)

        return generated_text, log_probs_tensor, generated_ids.tolist()

    def compute_loss(self, log_probs: torch.Tensor, reward: float) -> torch.Tensor:
        """
        Compute REINFORCE loss (negative expected reward).

        REINFORCE objective: maximize E[R(τ)]
        Policy gradient: ∇θ J(θ) = E[∇log π(a|s) * (R - b)]
        Loss: -sum(log_probs) * (reward - baseline)

        Args:
            log_probs: Log probabilities of generated tokens
            reward: Reward received for this trajectory

        Returns:
            Loss tensor
        """
        # Update baseline (moving average of rewards)
        if self.config.baseline_type == "moving_average":
            self.baseline = (
                self.config.baseline_decay * self.baseline
                + (1 - self.config.baseline_decay) * reward
            )

        # Advantage: reward - baseline (reduces variance)
        advantage = reward - self.baseline

        # REINFORCE loss: -sum(log π(a|s)) * advantage
        # Negative because we want to maximize, but optimizers minimize
        loss = -log_probs.sum() * advantage

        return loss

    def train_step(self, prompt: str, verbose: bool = True) -> dict:
        """
        Perform one REINFORCE training step.

        Args:
            prompt: Input prompt for generation
            verbose: Print training info

        Returns:
            Dict with training metrics
        """
        self.model.train()

        # 1. Generate output from current policy
        generated_text, log_probs, token_ids = self.generate(prompt)

        # 2. Compute reward
        reward = self.reward_fn(generated_text)
        self.total_reward += reward

        # 3. Compute loss (policy gradient)
        loss = self.compute_loss(log_probs, reward)

        # 4. Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (prevents instability)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

        self.optimizer.step()

        self.step += 1

        # Training metrics
        metrics = {
            "step": self.step,
            "loss": loss.item(),
            "reward": reward,
            "baseline": self.baseline,
            "advantage": reward - self.baseline,
            "avg_reward": self.total_reward / self.step,
            "generated_text": generated_text,
            "num_tokens": len(token_ids),
        }

        if verbose:
            print(f"Step {self.step}: reward={reward:.2f}, loss={loss.item():.4f}, "
                  f"baseline={self.baseline:.2f}, avg_reward={metrics['avg_reward']:.2f}")

        return metrics

    def evaluate(self, prompt: str, n_samples: int = 5) -> dict:
        """
        Evaluate current policy by generating multiple samples.

        Args:
            prompt: Input prompt
            n_samples: Number of samples to generate

        Returns:
            Dict with evaluation metrics
        """
        self.model.eval()

        rewards = []
        outputs = []

        with torch.no_grad():
            for i in range(n_samples):
                generated_text, _, _ = self.generate(prompt)
                reward = self.reward_fn(generated_text)
                rewards.append(reward)
                outputs.append(generated_text)

        return {
            "mean_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "success_rate": sum(1 for r in rewards if r > 0.5) / len(rewards),
            "samples": outputs,
            "rewards": rewards,
        }

    def save(self, path: str):
        """Save model checkpoint."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model checkpoint."""
        self.model = AutoModelForCausalLM.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        print(f"Model loaded from {path}")
