import importlib
import os
import sys


def test_build_user_message_includes_limits(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # Ensure we import fresh to respect env override
    if "ace_task.evaluate" in sys.modules:
        del sys.modules["ace_task.evaluate"]

    evaluate = importlib.import_module("ace_task.evaluate")
    scenario = evaluate.get_scenario("economics")
    message = evaluate.build_user_message(max_chars=50, max_words=10, scenario=scenario)

    assert "HARD LIMITS FOR THIS RUN" in message
    assert "MAX_CHARS for rewrite: 50" in message
    assert "MAX_WORDS for rewrite: 10" in message
    assert "ORIGINAL:" in message
    assert "FACTS:" in message
    assert "BANNED:" in message
