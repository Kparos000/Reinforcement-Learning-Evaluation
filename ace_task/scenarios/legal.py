"""Legal domain scenario - contract terms."""

from typing import Dict, List, Set

from .base import Scenario


class LegalScenario(Scenario):
    """Legal contract terms compression task."""

    @property
    def name(self) -> str:
        return "legal"

    @property
    def domain(self) -> str:
        return "legal"

    @property
    def original(self) -> str:
        return (
            "The agreement terminates on December 31, 2025, with a renewal fee of $50,000 "
            "payable within 30 days of notice. Either party may terminate with 90 days written notice."
        )

    @property
    def facts(self) -> List[str]:
        return [
            "terminates on December 31, 2025",
            "renewal fee of $50,000",
            "within 30 days",
            "90 days written notice",
        ]

    @property
    def banned(self) -> Set[str]:
        return {"automatic renewal", "no termination clause", "penalty fee $100,000"}

    @property
    def alias_map(self) -> Dict[str, List[str]]:
        return {
            "terminates on December 31, 2025": ["ends Dec 31, 2025", "expires 12/31/2025"],
            "renewal fee of $50,000": ["$50K renewal", "renewal $50,000"],
            "within 30 days": ["30-day window", "30d"],
            "90 days written notice": ["90d written notice", "90-day notice"],
        }

    @property
    def difficulty(self) -> str:
        return "hard"
