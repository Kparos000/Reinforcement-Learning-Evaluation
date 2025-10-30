"""Tests for configuration management."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from ace_task.config import (
    Config,
    GraderConfig,
    ModelConfig,
    load_config,
    validate_environment,
)


class TestModelConfig:
    """Test ModelConfig validation."""

    def test_default_values(self):
        config = ModelConfig()
        assert config.provider == "anthropic"
        assert config.name == "claude-3-5-haiku-latest"
        assert config.temperature == 0.45

    def test_temperature_validation(self):
        with pytest.raises(ValueError):
            ModelConfig(temperature=-0.1)
        with pytest.raises(ValueError):
            ModelConfig(temperature=2.1)

    def test_max_tokens_validation(self):
        with pytest.raises(ValueError):
            ModelConfig(max_tokens=0)
        with pytest.raises(ValueError):
            ModelConfig(max_tokens=5000)


class TestGraderConfig:
    """Test GraderConfig validation."""

    def test_default_values(self):
        config = GraderConfig()
        assert config.concision_limit == 0.60
        assert config.word_cap == 16
        assert config.strict_mode is True

    def test_concision_limit_validation(self):
        with pytest.raises(ValueError):
            GraderConfig(concision_limit=0.0)
        with pytest.raises(ValueError):
            GraderConfig(concision_limit=1.5)


class TestConfigLoading:
    """Test configuration loading from YAML."""

    def test_load_default_config(self, tmp_path):
        """Test loading valid config file."""
        config_data = {
            "model": {"provider": "anthropic", "name": "test-model"},
            "grader": {"concision_limit": 0.5, "word_cap": 20},
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_file))
        assert config.model.name == "test-model"
        assert config.grader.word_cap == 20

    def test_missing_config_file(self):
        """Test error when config file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_invalid_yaml(self, tmp_path):
        """Test error on malformed YAML."""
        config_file = tmp_path / "bad_config.yaml"
        with open(config_file, "w") as f:
            f.write("invalid: yaml: content:")

        with pytest.raises(Exception):
            load_config(str(config_file))


class TestEnvironmentValidation:
    """Test environment variable validation."""

    def test_validate_with_api_key(self, monkeypatch):
        """Test validation passes when API key is set."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        validate_environment()  # Should not raise

    def test_validate_without_api_key(self, monkeypatch):
        """Test validation fails when API key is missing."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="Missing required environment variables"):
            validate_environment()


class TestConfigIntegration:
    """Integration tests for full config loading."""

    def test_full_config_structure(self):
        """Test that config has all expected sections."""
        # Create minimal valid config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({}, f)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert hasattr(config, "model")
            assert hasattr(config, "grader")
            assert hasattr(config, "evaluation")
            assert hasattr(config, "rl")
            assert hasattr(config, "scenarios")
            assert hasattr(config, "visualization")
            assert hasattr(config, "tracking")
            assert hasattr(config, "logging")
            assert hasattr(config, "reproducibility")
        finally:
            os.unlink(temp_path)
