"""Top-15 fact variant of the report_long scenario for learnable long-form benchmarking.

Derived from report_long: same domain (financial + ESG annual report), but narrowed
to the 15 most critical quantitative facts to make the task achievable for mid-tier models.
Use this as the primary long-form benchmark for RL experiments.
"""

from typing import Dict, List, Set

from .base import Scenario


class ReportTop15Scenario(Scenario):
    """Business sustainability report with a top-15 fact focus."""

    @property
    def name(self) -> str:
        return "report_top15"

    @property
    def domain(self) -> str:
        return "business"

    @property
    def original(self) -> str:
        # Reuse the full text to preserve difficulty; facts list is narrowed.
        return (
            "Our 2024 integrated report covers financial performance, customer metrics, platform reliability, and "
            "sustainability progress. Revenue reached $4.8 billion, up 11.2% year over year, with net income of "
            "$620 million and free cash flow of $710 million. Subscription revenue now represents 78% of the mix, "
            "driven by cloud ARR of $2.1 billion, which grew 27%. Gross retention held at 94% and net revenue "
            "retention finished at 108%. The customer base expanded to 58,000, with enterprise customers (>$100k ARR) "
            "rising to 3,900 and an average NPS of 63. Europe now contributes 38% of total revenue, up from 34% last "
            "year. Platform reliability remained strong with 99.98% uptime and no Severity 1 incidents. Security "
            "performance improved with zero material breaches, a 22% drop in phishing click-through rates, and 100% "
            "completion of annual security training. Data center efficiency improved to a PUE of 1.18, down from 1.22. "
            "Renewable energy coverage rose to 86% (target 92% by 2027) and Scope 1 and 2 emissions decreased 18% "
            "year over year; Scope 3 emissions decreased 6%. Water usage declined 9% and landfill waste fell 14%. "
            "We invested $68 million in R&D for efficiency and shipped 142 product releases, including AI-assisted "
            "workflows that cut average task time by 19%. Employee engagement scored 82/100 and voluntary attrition "
            "was 8.4%. Lost time injury frequency rate improved to 0.21. Community investment totaled $12.5 million "
            "and employees volunteered 92,000 hours. Looking ahead, we plan to expand renewable PPAs, raise cloud ARR "
            "growth to the high 20s, and maintain uptime at or above 99.98%."
        )

    @property
    def facts(self) -> List[str]:
        return [
            # Financial core
            "Revenue reached $4.8 billion",
            "Revenue grew 11.2% year over year",
            "Net income was $620 million",
            "Free cash flow was $710 million",
            # ARR and mix
            "Subscription revenue is 78% of mix",
            "Cloud ARR is $2.1 billion",
            "Cloud ARR grew 27%",
            # Customer/retention
            "Gross retention was 94%",
            "Net revenue retention was 108%",
            "Enterprise customers rose to 3,900",
            # Reliability/security
            "Uptime was 99.98%",
            "Zero material breaches",
            # ESG/efficiency
            "Data center PUE was 1.18",
            "Renewable energy coverage is 86%",
            "Scope 1 and 2 emissions decreased 18%",
        ]

    @property
    def banned(self) -> Set[str]:
        return {"layoffs", "restatement", "data breach"}

    @property
    def alias_map(self) -> Dict[str, List[str]]:
        return {
            "Revenue reached $4.8 billion": ["$4.8B revenue"],
            "Revenue grew 11.2% year over year": ["11.2% revenue growth"],
            "Net income was $620 million": ["$620M net income"],
            "Free cash flow was $710 million": ["$710M free cash flow", "FCF $710M"],
            "Subscription revenue is 78% of mix": ["78% subscription mix"],
            "Cloud ARR is $2.1 billion": ["$2.1B cloud ARR"],
            "Cloud ARR grew 27%": ["cloud ARR up 27%"],
            "Gross retention was 94%": ["94% gross retention"],
            "Net revenue retention was 108%": ["108% NRR"],
            "Enterprise customers rose to 3,900": ["3,900 enterprise customers"],
            "Uptime was 99.98%": ["99.98% availability"],
            "Zero material breaches": ["0 material breaches"],
            "Data center PUE was 1.18": ["PUE 1.18"],
            "Renewable energy coverage is 86%": ["86% renewables"],
            "Scope 1 and 2 emissions decreased 18%": ["Scope 1-2 down 18%"],
        }

    @property
    def difficulty(self) -> str:
        return "hard"

    @property
    def word_cap(self) -> int:
        # Keep parity with report_long to preserve compression pressure.
        return 140

    @property
    def concision_limit(self) -> float:
        # Slightly forgiving concision for long-form while keeping it challenging.
        return 0.70
