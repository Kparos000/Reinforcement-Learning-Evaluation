"""
Configuration management for ACE RL Evaluation Framework.

Uses Pydantic for validation and YAML for human-readable config files.
Implements environment variable overrides and validation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    """Model configuration settings."""

    provider: Literal["anthropic", "openai", "huggingface"] = "anthropic"
    name: str = "claude-3-5-haiku-latest"
    max_tokens: int = Field(400, gt=0, le=4096)
    temperature: float = Field(0.45, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)


class GraderConfig(BaseModel):
    """Grader configuration and validation rules."""

    concision_limit: float = Field(0.60, gt=0.0, le=1.0)
    word_cap: int = Field(16, gt=0)
    min_delta_words: int = Field(6, gt=0)
    strict_mode: bool = True


class EvaluationConfig(BaseModel):
    """Evaluation run configuration."""

    num_runs: int = Field(10, gt=0)
    random_seed: int = 42
    max_words_variation: list[int] = Field(default_factory=lambda: [12, 14, 16])
    parallel_workers: int = Field(4, ge=1)

    @field_validator("max_words_variation")
    def validate_word_variation(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("max_words_variation cannot be empty")
        if any(w <= 0 for w in v):
            raise ValueError("All word limits must be positive")
        return v


class BestOfNConfig(BaseModel):
    """Best-of-N sampling configuration."""

    n_samples: int = Field(5, gt=0)
    temperature_range: list[float] = Field(default_factory=lambda: [0.3, 0.7])

    @field_validator("temperature_range")
    def validate_temp_range(cls, v: list[float]) -> list[float]:
        if len(v) != 2:
            raise ValueError("temperature_range must have exactly 2 values [min, max]")
        if v[0] >= v[1]:
            raise ValueError("Min temperature must be less than max temperature")
        return v


class ReinforceConfig(BaseModel):
    """REINFORCE policy gradient configuration."""

    learning_rate: float = Field(0.001, gt=0.0)
    gamma: float = Field(1.0, ge=0.0, le=1.0)
    baseline: Literal["none", "moving_average", "learned"] = "moving_average"


class PPOConfig(BaseModel):
    """PPO (Proximal Policy Optimization) configuration."""

    learning_rate: float = Field(0.0003, gt=0.0)
    gamma: float = Field(1.0, ge=0.0, le=1.0)
    clip_epsilon: float = Field(0.2, gt=0.0)
    epochs: int = Field(4, gt=0)
    batch_size: int = Field(64, gt=0)


class RLConfig(BaseModel):
    """RL algorithm configuration."""

    algorithm: Literal["best_of_n", "reinforce", "ppo"] = "best_of_n"
    best_of_n: BestOfNConfig = Field(default_factory=BestOfNConfig)
    reinforce: ReinforceConfig = Field(default_factory=ReinforceConfig)
    ppo: PPOConfig = Field(default_factory=PPOConfig)


class ScenarioConfig(BaseModel):
    """Scenario-specific configuration."""

    concision_limit: Optional[float] = None


class ScenariosConfig(BaseModel):
    """Multi-scenario configuration."""

    enabled: list[str] = Field(default_factory=lambda: ["economics"])
    economics: ScenarioConfig = Field(default_factory=ScenarioConfig)
    medical: ScenarioConfig = Field(default_factory=ScenarioConfig)
    legal: ScenarioConfig = Field(default_factory=ScenarioConfig)
    scientific: ScenarioConfig = Field(default_factory=ScenarioConfig)


class VisualizationConfig(BaseModel):
    """Visualization output configuration."""

    enabled: bool = True
    output_dir: str = "results/visualizations"
    formats: list[str] = Field(default_factory=lambda: ["png", "pdf"])
    dpi: int = Field(300, gt=0)


class TrackingConfig(BaseModel):
    """Experiment tracking configuration."""

    enabled: bool = False
    backend: Literal["mlflow", "wandb"] = "mlflow"
    experiment_name: str = "ace-rl-evaluation"
    run_name: Optional[str] = None


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/ace_evaluation.log"


class ReproducibilityConfig(BaseModel):
    """Reproducibility settings."""

    seed: int = 42
    deterministic: bool = True
    benchmark: bool = False


class Config(BaseModel):
    """Root configuration object."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    grader: GraderConfig = Field(default_factory=GraderConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    scenarios: ScenariosConfig = Field(default_factory=ScenariosConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    reproducibility: ReproducibilityConfig = Field(default_factory=ReproducibilityConfig)


def load_config(config_path: str | None = None) -> Config:
    """
    Load configuration from YAML file with environment variable overrides.

    Args:
        config_path: Path to config.yaml. If None, uses default location.

    Returns:
        Validated Config object.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config validation fails.
    """
    config_file: Path
    if config_path is None:
        config_file = Path(__file__).parent.parent / "config.yaml"
    else:
        config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)  # type: ignore[assignment]

    # Environment variable overrides
    if api_key := os.getenv("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = api_key

    return Config(**config_dict)


def get_config() -> Config:
    """Get the global configuration object."""
    return load_config()


# Validate environment on import
def validate_environment() -> None:
    """
    Validate that all required environment variables are set.

    Raises:
        EnvironmentError: If required variables are missing.
    """
    required_vars = ["ANTHROPIC_API_KEY"]

    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Please create a .env file with:\n"
            + "\n".join(f"{var}=your_value_here" for var in missing)
        )
