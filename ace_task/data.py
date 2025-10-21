"""
Canonical data fixtures for the ACE RL evaluation task.
These are deterministic and self-contained.
"""

ORIGINAL = (
    "In Q2, GDP grew by 3.2% while inflation was 2.1%. "
    "Exports increased as supply chains normalized."
)

FACTS = ["GDP grew by 3.2%", "inflation was 2.1%", "exports increased"]

BANNED = {"recession", "budget deficit $9.9B"}

ALIAS_MAP = {
    "GDP grew by 3.2%": ["GDP growth was 3.2%", "GDP rose 3.2%"],
    "inflation was 2.1%": ["inflation hit 2.1%", "inflation stood at 2.1%"],
    "exports increased": ["exports rose", "exports went up"],
}
