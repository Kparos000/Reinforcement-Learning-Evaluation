"""Medical domain scenario - clinical trial results."""

from typing import Dict, List, Set

from .base import Scenario


class MedicalScenario(Scenario):
    """Clinical trial results compression task."""

    @property
    def name(self) -> str:
        return "medical"

    @property
    def domain(self) -> str:
        return "medical"

    @property
    def original(self) -> str:
        return (
            "Phase III trial showed 87% efficacy with 15mg dose administered twice daily. "
            "Common adverse events included nausea in 12% of patients and headache in 8% of patients."
        )

    @property
    def facts(self) -> List[str]:
        return [
            "87% efficacy",
            "15mg dose",
            "twice daily",
            "nausea in 12% of patients",
            "headache in 8% of patients",
        ]

    @property
    def banned(self) -> Set[str]:
        return {"death", "mortality rate 3%", "liver damage"}

    @property
    def alias_map(self) -> Dict[str, List[str]]:
        return {
            "87% efficacy": ["87% effective", "efficacy 87%"],
            "15mg dose": ["15mg", "dose of 15mg"],
            "twice daily": ["2x daily", "bid"],
            "nausea in 12% of patients": ["nausea 12%", "12% nausea"],
            "headache in 8% of patients": ["headache 8%", "8% headache"],
        }

    @property
    def difficulty(self) -> str:
        return "medium"
