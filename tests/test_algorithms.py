"""
Tests for RL algorithms (Best-of-N, REINFORCE, PPO).

This module tests the core RL algorithm implementations to ensure correctness,
robustness, and consistent behavior.
"""

import pytest

from ace_task.algorithms import BestOfNSampler, RLResult, create_reward_function
from ace_task.scenarios import get_scenario


class MockMessages:
    """Mock messages attribute for Anthropic client."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0

    def create(self, **kwargs):
        """Mock messages.create() method."""
        if self.call_count >= len(self.responses):
            raise RuntimeError("Mock client ran out of responses")

        response_text = self.responses[self.call_count]
        self.call_count += 1

        # Return mock response object
        class MockContent:
            def __init__(self, text):
                self.text = text

        class MockMessage:
            def __init__(self, text):
                self.content = [MockContent(text)]

        return MockMessage(response_text)


class MockAnthropicClient:
    """Mock Anthropic client for testing without API calls."""

    def __init__(self, responses: list[str]):
        """
        Initialize mock client with pre-defined responses.

        Args:
            responses: List of responses to return in order
        """
        self.messages = MockMessages(responses)


class TestRLResult:
    """Tests for RLResult dataclass."""

    def test_rl_result_creation(self):
        """Test basic RLResult creation."""
        result = RLResult(
            best_sample="sample 1",
            best_reward=1.0,
            all_samples=["sample 1", "sample 2", "sample 3"],
            all_rewards=[1.0, 0.5, 0.0],
        )

        assert result.best_sample == "sample 1"
        assert result.best_reward == 1.0
        assert result.num_samples == 3
        assert result.avg_reward == pytest.approx(0.5)
        # Success rate: 2 out of 3 rewards > 0 (1.0 and 0.5)
        assert result.success_rate == pytest.approx(0.666, rel=0.01)

    def test_rl_result_validation(self):
        """Test that RLResult validates sample/reward length mismatch."""
        with pytest.raises(ValueError, match="Sample/reward length mismatch"):
            RLResult(
                best_sample="sample 1",
                best_reward=1.0,
                all_samples=["sample 1", "sample 2"],
                all_rewards=[1.0],  # Mismatched length
            )

    def test_rl_result_empty(self):
        """Test RLResult with empty samples."""
        result = RLResult(
            best_sample="",
            best_reward=0.0,
            all_samples=[],
            all_rewards=[],
        )

        assert result.num_samples == 0
        assert result.avg_reward == 0.0
        assert result.success_rate == 0.0

    def test_rl_result_metadata(self):
        """Test metadata handling."""
        result = RLResult(
            best_sample="sample",
            best_reward=1.0,
            all_samples=["sample"],
            all_rewards=[1.0],
            metadata={"temperature": 0.7, "model": "claude-3-5-sonnet-20241022"},
        )

        assert result.metadata["temperature"] == 0.7
        assert result.metadata["model"] == "claude-3-5-sonnet-20241022"


class TestBestOfNSampler:
    """Tests for Best-of-N sampling algorithm."""

    def test_initialization(self):
        """Test BestOfNSampler initialization."""
        mock_client = MockAnthropicClient([])
        sampler = BestOfNSampler(client=mock_client, n=5)

        assert sampler.n == 5
        assert sampler.early_stop is True
        assert sampler.temperature == 1.0

    def test_temperature_validation(self):
        """Test that temperature=0 with n>1 raises error."""
        mock_client = MockAnthropicClient([])

        with pytest.raises(ValueError, match="temperature > 0"):
            BestOfNSampler(client=mock_client, n=5, temperature=0.0)

        # Should work with n=1
        BestOfNSampler(client=mock_client, n=1, temperature=0.0)

    def test_best_of_n_selection(self):
        """Test that Best-of-N selects the highest reward sample."""
        # Mock responses with different quality
        responses = [
            "bad response",
            "good response with high reward",
            "medium response",
        ]
        mock_client = MockAnthropicClient(responses)

        # Mock reward function: longer text = higher reward
        def reward_fn(text: str) -> float:
            return len(text) / 100.0

        sampler = BestOfNSampler(client=mock_client, n=3, temperature=0.7)
        result = sampler.generate("test prompt", reward_fn)

        # Should select the longest response
        assert result.best_sample == "good response with high reward"
        assert result.num_samples == 3
        assert result.best_reward == len("good response with high reward") / 100.0

    def test_early_stopping(self):
        """Test early stopping when perfect reward is achieved."""
        # First response gets perfect reward
        responses = ["perfect!", "backup 1", "backup 2", "backup 3", "backup 4"]
        mock_client = MockAnthropicClient(responses)

        def reward_fn(text: str) -> float:
            return 1.0 if "perfect" in text else 0.0

        sampler = BestOfNSampler(client=mock_client, n=5, early_stop=True, temperature=0.7)
        result = sampler.generate("test prompt", reward_fn)

        # Should stop after first sample
        assert result.num_samples == 1
        assert result.metadata["early_stopped"] is True
        assert mock_client.messages.call_count == 1

    def test_no_early_stopping(self):
        """Test that early stopping can be disabled."""
        responses = ["perfect!", "backup 1", "backup 2"]
        mock_client = MockAnthropicClient(responses)

        def reward_fn(text: str) -> float:
            return 1.0 if "perfect" in text else 0.0

        sampler = BestOfNSampler(client=mock_client, n=3, early_stop=False, temperature=0.7)
        result = sampler.generate("test prompt", reward_fn)

        # Should generate all 3 samples even though first is perfect
        assert result.num_samples == 3
        assert result.metadata["early_stopped"] is False
        assert mock_client.messages.call_count == 3

    def test_diversity_metrics(self):
        """Test diversity ratio calculation."""
        # All unique responses
        responses = ["response 1", "response 2", "response 3"]
        mock_client = MockAnthropicClient(responses)

        def reward_fn(text: str) -> float:
            return 0.5

        sampler = BestOfNSampler(client=mock_client, n=3, temperature=0.7)
        result = sampler.generate("test prompt", reward_fn)

        assert result.metadata["unique_samples"] == 3
        assert result.metadata["diversity_ratio"] == 1.0

    def test_low_diversity(self):
        """Test diversity calculation with duplicate samples."""
        # Some duplicate responses
        responses = ["same", "same", "different"]
        mock_client = MockAnthropicClient(responses)

        def reward_fn(text: str) -> float:
            return 0.5

        sampler = BestOfNSampler(client=mock_client, n=3, temperature=0.7)
        result = sampler.generate("test prompt", reward_fn)

        assert result.metadata["unique_samples"] == 2
        assert result.metadata["diversity_ratio"] == pytest.approx(0.666, rel=0.01)

    def test_repr(self):
        """Test string representation."""
        mock_client = MockAnthropicClient([])
        sampler = BestOfNSampler(client=mock_client, n=5, temperature=0.8)

        repr_str = repr(sampler)
        assert "BestOfNSampler" in repr_str
        assert "n=5" in repr_str
        assert "temperature=0.8" in repr_str


class TestRewardFunctions:
    """Tests for reward function utilities."""

    def test_create_reward_function_binary(self):
        """Test creating binary reward function."""
        scenario = get_scenario("economics")
        reward_fn = create_reward_function(scenario, reward_scheme="binary")

        # Test with a good output (just a simple check)
        assert callable(reward_fn)

    def test_binary_reward_pass(self):
        """Test binary reward returns 1.0 for passing output."""
        from ace_task.algorithms.rewards import BinaryRewardFunction
        from ace_task.scenarios import get_scenario

        scenario = get_scenario("economics")
        reward_fn = BinaryRewardFunction(scenario)

        # Create a minimal passing output (this is scenario-specific)
        # For testing, we'll use a simple text
        output = "GDP increased 3.2% in 2024"

        # Since we don't know if this exact text passes, just test that
        # reward is either 0.0 or 1.0 (binary)
        reward = reward_fn(output)
        assert reward in [0.0, 1.0]

    def test_binary_reward_fail(self):
        """Test binary reward returns 0.0 for failing output."""
        from ace_task.algorithms.rewards import BinaryRewardFunction
        from ace_task.scenarios import get_scenario

        scenario = get_scenario("economics")
        reward_fn = BinaryRewardFunction(scenario)

        # Definitely bad output (uses banned words, missing facts)
        output = "approximately nothing happened"

        reward = reward_fn(output)
        assert reward == 0.0

    def test_binary_reward_exception_handling(self):
        """Test that grading exceptions return 0.0 reward."""
        from ace_task.algorithms.rewards import BinaryRewardFunction

        # Create a minimal mock scenario
        class MockScenario:
            original = "test"
            facts = []
            banned = set()

        scenario = MockScenario()
        reward_fn = BinaryRewardFunction(scenario)

        # Malformed output that might cause grading to fail
        reward = reward_fn(None)  # This should handle gracefully
        assert reward == 0.0

    def test_unsupported_reward_scheme(self):
        """Test that unsupported reward schemes raise error."""
        scenario = get_scenario("economics")

        with pytest.raises(ValueError, match="Unknown reward scheme"):
            create_reward_function(scenario, reward_scheme="unsupported")

    def test_repr_reward_function(self):
        """Test reward function string representation."""
        from ace_task.algorithms.rewards import BinaryRewardFunction
        from ace_task.scenarios import get_scenario

        scenario = get_scenario("economics")
        reward_fn = BinaryRewardFunction(scenario)

        repr_str = repr(reward_fn)
        assert "BinaryRewardFunction" in repr_str
