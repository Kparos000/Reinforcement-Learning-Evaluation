"""Scenario registry for multi-domain evaluation."""

from typing import List

from .base import Scenario
from .economics import EconomicsScenario
from .finance import FinanceScenario
from .legal import LegalScenario
from .medical import MedicalScenario
from .scientific import ScientificScenario
from .sports import SportsScenario

# Phase 2: Long-form scenarios
from .medical_long import MedicalLongScenario
from .business_long import BusinessLongScenario
from .legal_long import LegalLongScenario

# Registry of all available scenarios
SCENARIO_REGISTRY: dict[str, type[Scenario]] = {
    # Phase 1: Short scenarios
    "economics": EconomicsScenario,
    "medical": MedicalScenario,
    "legal": LegalScenario,
    "scientific": ScientificScenario,
    "finance": FinanceScenario,
    "sports": SportsScenario,
    # Phase 2: Long scenarios
    "medical_long": MedicalLongScenario,
    "business_long": BusinessLongScenario,
    "legal_long": LegalLongScenario,
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
        raise KeyError(f"Unknown scenario '{name}'. Available: {list(SCENARIO_REGISTRY.keys())}")
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
    "FinanceScenario",
    "SportsScenario",
    "MedicalLongScenario",
    "BusinessLongScenario",
    "LegalLongScenario",
    "get_scenario",
    "list_scenarios",
    "SCENARIO_REGISTRY",
]
