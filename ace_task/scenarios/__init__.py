"""Scenario registry for multi-domain evaluation."""

from typing import List

from .base import Scenario
from .economics import EconomicsScenario
from .legal import LegalScenario
from .medical import MedicalScenario
from .scientific import ScientificScenario

# Registry of all available scenarios
SCENARIO_REGISTRY: dict[str, type[Scenario]] = {
    "economics": EconomicsScenario,
    "medical": MedicalScenario,
    "legal": LegalScenario,
    "scientific": ScientificScenario,
}


def get_scenario(name: str) -> Scenario:
    """
    Get a scenario instance by name.

    Args:
        name: Scenario identifier (e.g., 'economics', 'medical')

    Returns:
        Initialized scenario instance

    Raises:
        KeyError: If scenario name not found
    """
    if name not in SCENARIO_REGISTRY:
        raise KeyError(
            f"Unknown scenario '{name}'. Available: {list(SCENARIO_REGISTRY.keys())}"
        )
    return SCENARIO_REGISTRY[name]()


def list_scenarios() -> List[str]:
    """Return list of available scenario names."""
    return list(SCENARIO_REGISTRY.keys())


__all__ = [
    "Scenario",
    "EconomicsScenario",
    "MedicalScenario",
    "LegalScenario",
    "ScientificScenario",
    "get_scenario",
    "list_scenarios",
    "SCENARIO_REGISTRY",
]
