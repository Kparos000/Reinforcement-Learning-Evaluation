import types

import pytest
import torch


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.config = types.SimpleNamespace(pad_token_id=0, eos_token_id=0)

    def to(self, device):
        return self

    def generate(self, input_ids, **kwargs):
        # Simulate two new tokens appended after the prompt
        sequences = torch.tensor([[1, 2, 0, 1]])
        scores = [
            torch.tensor([[0.6, 0.4]]),
            torch.tensor([[0.5, 0.5]]),
        ]
        return types.SimpleNamespace(sequences=sequences, scores=scores)

    def save_pretrained(self, path):
        pass


class DummyTokenizer:
    def __init__(self):
        self.pad_token = None
        self.eos_token = "</s>"
        self.eos_token_id = 0
        self.pad_token_id = 0

    def encode(self, prompt, return_tensors="pt"):
        return torch.tensor([[1, 2]])

    def decode(self, ids, skip_special_tokens=True):
        return "dummy"

    def save_pretrained(self, path):
        pass


@pytest.mark.parametrize("baseline_type", ["moving_average", "none"])
def test_generate_and_compute_loss(monkeypatch, baseline_type):
    reinforce = pytest.importorskip("ace_task.algorithms.reinforce")

    monkeypatch.setattr(reinforce.AutoModelForCausalLM, "from_pretrained", lambda name: DummyModel())
    monkeypatch.setattr(reinforce.AutoTokenizer, "from_pretrained", lambda name: DummyTokenizer())

    config = reinforce.REINFORCEConfig(max_length=4, baseline_decay=0.5, baseline_type=baseline_type)
    trainer = reinforce.REINFORCETrainer(
        model_name="dummy-model",
        reward_fn=lambda text: 1.0,
        config=config,
        device="cpu",
    )

    text, log_probs, token_ids = trainer.generate("prompt")

    assert text == "dummy"
    assert len(token_ids) == len(log_probs)

    # Compute loss and ensure baseline behaves as expected
    trainer.baseline = 0.0
    loss = trainer.compute_loss(log_probs, reward=1.0)
    assert loss.item() >= 0
    if baseline_type == "moving_average":
        assert trainer.baseline > 0
    else:
        assert trainer.baseline == 0.0
