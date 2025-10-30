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

# Minimal aliases so some paraphrases pass, others fail (yields ~10–40% with temp>0).
ALIAS_MAP = {
    "GDP grew by 3.2%": ["GDP +3.2%"],
    "inflation was 2.1%": ["inflation 2.1%"],
    "exports increased": ["exports rose", "exports grew"],
}
