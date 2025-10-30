"""Economics domain scenario - original ACE task."""

from typing import Dict, List, Set

from .base import Scenario


class EconomicsScenario(Scenario):
    """Economic indicators compression task."""

    @property
    def name(self) -> str:
        return "economics"

    @property
    def domain(self) -> str:
        return "economics"

    @property
    def original(self) -> str:
        return (
            "In Q2, GDP grew by 3.2% while inflation was 2.1%. "
            "Exports increased as supply chains normalized."
        )

    @property
    def facts(self) -> List[str]:
        return ["GDP grew by 3.2%", "inflation was 2.1%", "exports increased"]

    @property
    def banned(self) -> Set[str]:
        return {"recession", "budget deficit $9.9B"}

    @property
    def alias_map(self) -> Dict[str, List[str]]:
        return {
            "GDP grew by 3.2%": ["GDP +3.2%"],
            "inflation was 2.1%": ["inflation 2.1%"],
            "exports increased": ["exports rose", "exports grew"],
        }

    @property
    def difficulty(self) -> str:
        return "easy"
