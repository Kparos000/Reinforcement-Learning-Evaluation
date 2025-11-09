"""Finance domain scenario - stock market data."""

from typing import Dict, List, Set

from .base import Scenario


class FinanceScenario(Scenario):
    """Stock market data compression task."""

    @property
    def name(self) -> str:
        return "finance"

    @property
    def domain(self) -> str:
        return "finance"

    @property
    def original(self) -> str:
        return (
            "The stock opened at $127.50, reached a high of $134.20, and closed at $131.85 "
            "with trading volume of 4.7 million shares. The price-to-earnings ratio stands at 23.4x."
        )

    @property
    def facts(self) -> List[str]:
        return [
            "opened at $127.50",
            "high of $134.20",
            "closed at $131.85",
            "4.7 million shares",
            "P/E ratio 23.4x",
        ]

    @property
    def banned(self) -> Set[str]:
        return {"bankruptcy", "delisted", "SEC investigation"}

    @property
    def alias_map(self) -> Dict[str, List[str]]:
        return {
            "opened at $127.50": ["open $127.50", "$127.50 open"],
            "high of $134.20": ["high $134.20", "$134.20 high"],
            "closed at $131.85": ["close $131.85", "$131.85 close"],
            "4.7 million shares": ["4.7M shares", "volume 4.7M"],
            "P/E ratio 23.4x": ["P/E 23.4x", "23.4x P/E"],
        }

    @property
    def difficulty(self) -> str:
        return "medium"
