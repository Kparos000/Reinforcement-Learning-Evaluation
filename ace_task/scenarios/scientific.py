"""Scientific domain scenario - research findings."""

from typing import Dict, List, Set

from .base import Scenario


class ScientificScenario(Scenario):
    """Scientific research findings compression task."""

    @property
    def name(self) -> str:
        return "scientific"

    @property
    def domain(self) -> str:
        return "scientific"

    @property
    def original(self) -> str:
        return (
            "The catalyst achieved 94.5% conversion at 250°C under 2.5 bar pressure. "
            "Selectivity for the desired product was 89% with a turnover frequency of 125 h⁻¹."
        )

    @property
    def facts(self) -> List[str]:
        return [
            "94.5% conversion",
            "250°C",
            "2.5 bar pressure",
            "89% selectivity",
            "125 h⁻¹ turnover frequency",
        ]

    @property
    def banned(self) -> Set[str]:
        return {"failed experiment", "contamination detected", "98% yield"}

    @property
    def alias_map(self) -> Dict[str, List[str]]:
        return {
            "94.5% conversion": ["conversion 94.5%", "94.5% converted"],
            "250°C": ["250 degrees C", "250°C temp"],
            "2.5 bar pressure": ["2.5 bar", "pressure 2.5 bar"],
            "89% selectivity": ["selectivity 89%", "89% selective"],
            "125 h⁻¹ turnover frequency": ["TOF 125 h⁻¹", "125/h turnover"],
        }

    @property
    def difficulty(self) -> str:
        return "medium"
