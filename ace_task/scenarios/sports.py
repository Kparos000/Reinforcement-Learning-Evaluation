"""Sports domain scenario - basketball game statistics."""

from typing import Dict, List, Set

from .base import Scenario


class SportsScenario(Scenario):
    """Basketball game statistics compression task."""

    @property
    def name(self) -> str:
        return "sports"

    @property
    def domain(self) -> str:
        return "sports"

    @property
    def original(self) -> str:
        return (
            "The Lakers won 112-98 with LeBron James scoring 28 points, 9 rebounds, and 7 assists. "
            "The team shot 47.3% from the field and made 14 three-pointers."
        )

    @property
    def facts(self) -> List[str]:
        return [
            "Lakers won 112-98",
            "LeBron James 28 points",
            "9 rebounds",
            "7 assists",
            "47.3% field goal",
            "14 three-pointers",
        ]

    @property
    def banned(self) -> Set[str]:
        return {"lost", "injured", "suspended"}

    @property
    def alias_map(self) -> Dict[str, List[str]]:
        return {
            "Lakers won 112-98": ["Lakers 112-98", "W 112-98"],
            "LeBron James 28 points": ["LeBron 28pts", "James 28 points", "LeBron 28 points"],
            "9 rebounds": ["9 reb", "9 boards"],
            "7 assists": ["7 ast", "7 dimes"],
            "47.3% field goal": ["47.3% FG", "FG 47.3%"],
            "14 three-pointers": ["14 3PT", "14 threes"],
        }

    @property
    def difficulty(self) -> str:
        return "medium"
