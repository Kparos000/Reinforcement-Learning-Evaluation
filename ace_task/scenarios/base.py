"""Base scenario class defining the interface for all scenarios."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass
class Scenario(ABC):
    """
    Base class for evaluation scenarios.

    Each scenario defines:
    - ORIGINAL: The text to be compressed
    - FACTS: Required facts that must be preserved
    - BANNED: Terms that must not appear
    - ALIAS_MAP: Acceptable paraphrases for facts
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Scenario identifier."""
        pass

    @property
    @abstractmethod
    def domain(self) -> str:
        """Domain category (e.g., 'medical', 'legal', 'economics')."""
        pass

    @property
    @abstractmethod
    def original(self) -> str:
        """Original text to be compressed."""
        pass

    @property
    @abstractmethod
    def facts(self) -> List[str]:
        """Required facts that must be preserved."""
        pass

    @property
    @abstractmethod
    def banned(self) -> Set[str]:
        """Terms that must not appear in the rewrite."""
        pass

    @property
    @abstractmethod
    def alias_map(self) -> Dict[str, List[str]]:
        """Acceptable paraphrases for facts."""
        pass

    @property
    def difficulty(self) -> str:
        """
        Difficulty rating: 'easy', 'medium', 'hard'.
        Override in subclasses if needed.
        """
        return "medium"

    def __str__(self) -> str:
        return f"{self.name} ({self.domain}, {self.difficulty})"
