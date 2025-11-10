"""Business domain scenario - Quarterly earnings report (Phase 2)."""

from typing import Dict, List, Set

from .base import Scenario


class BusinessLongScenario(Scenario):
    """Quarterly earnings report compression task - realistic length."""

    @property
    def name(self) -> str:
        return "business_long"

    @property
    def domain(self) -> str:
        return "business"

    @property
    def original(self) -> str:
        return (
            "TechCorp Inc. (NASDAQ: TECH) announced third quarter fiscal year 2024 financial results "
            "on October 28, 2024 after market close. The company reported revenue of $847.3 million "
            "for the three months ended September 30, 2024, representing a 23.7% increase compared to "
            "$684.7 million in the same period last year and exceeding the analyst consensus estimate "
            "of $812 million by 4.3%. Revenue growth was driven primarily by the company's cloud "
            "infrastructure segment, which grew 38.2% year-over-year to $423.5 million, and the "
            "enterprise software division, which increased 18.4% to $312.8 million. The legacy hardware "
            "business declined 12.3% to $111.0 million as expected due to the ongoing transition to "
            "cloud-based solutions. Gross profit margin improved significantly to 68.2% compared to "
            "64.8% in Q3 2023, driven by favorable product mix shifts toward higher-margin cloud "
            "services and improved operational efficiencies in data center operations. Operating "
            "expenses for the quarter totaled $423.6 million, representing 50.0% of revenue, compared "
            "to $356.2 million or 52.0% of revenue in the prior year period. Research and development "
            "expenses increased to $152.3 million (18.0% of revenue) from $130.4 million in Q3 2023, "
            "reflecting continued investment in artificial intelligence capabilities and next-generation "
            "cloud infrastructure. Sales and marketing expenses rose to $186.4 million (22.0% of revenue) "
            "from $157.3 million as the company expanded its go-to-market teams in North America and "
            "Europe. General and administrative expenses were $84.9 million (10.0% of revenue), up from "
            "$68.5 million in the prior year. Operating income reached $153.9 million, yielding an "
            "operating margin of 18.2% compared to 13.8% in Q3 2023. Net income for the quarter was "
            "$127.5 million, or $2.14 per diluted share, compared to $89.2 million, or $1.52 per diluted "
            "share, in the prior year period. The company ended the quarter with $1.24 billion in cash, "
            "cash equivalents, and short-term investments, up from $1.08 billion at the end of Q2 2024. "
            "Total debt remained at zero, maintaining the company's debt-free balance sheet. Operating "
            "cash flow for the quarter was $218.7 million, representing a 25.8% cash conversion rate. "
            "Free cash flow, calculated as operating cash flow less capital expenditures of $47.2 million, "
            "was $171.5 million. The Board of Directors authorized a new $500 million share repurchase "
            "program to be executed over the next 24 months, replacing the previous program under which "
            "$287 million remained available. During Q3, the company repurchased 1.8 million shares at "
            "an average price of $78.50 per share for a total of $141.3 million. For the fourth quarter "
            "of fiscal year 2024, the company provided guidance of revenue between $880 million and "
            "$920 million, representing year-over-year growth of 20-25%, and earnings per diluted share "
            "between $2.25 and $2.45. For the full fiscal year 2024, the company raised its revenue "
            "guidance to $3.31-3.35 billion from the previous range of $3.20-3.28 billion, representing "
            "growth of 24-26% compared to fiscal year 2023."
        )

    @property
    def facts(self) -> List[str]:
        return [
            "TechCorp Inc. NASDAQ: TECH",
            "Q3 FY2024 results announced October 28, 2024",
            "quarter ended September 30, 2024",
            "revenue $847.3 million",
            "23.7% year-over-year growth",
            "exceeded consensus $812 million by 4.3%",
            "cloud infrastructure $423.5 million, up 38.2%",
            "enterprise software $312.8 million, up 18.4%",
            "legacy hardware $111.0 million, down 12.3%",
            "gross margin 68.2% vs 64.8% prior year",
            "operating expenses $423.6 million, 50.0% of revenue",
            "R&D $152.3 million, 18.0% of revenue",
            "sales and marketing $186.4 million, 22.0% of revenue",
            "G&A $84.9 million, 10.0% of revenue",
            "operating income $153.9 million",
            "operating margin 18.2% vs 13.8% prior year",
            "net income $127.5 million",
            "EPS $2.14 vs $1.52 prior year",
            "cash $1.24 billion",
            "zero debt",
            "operating cash flow $218.7 million",
            "capex $47.2 million",
            "free cash flow $171.5 million",
            "new $500 million buyback authorized over 24 months",
            "repurchased 1.8 million shares at $78.50 average",
            "$141.3 million spent on buybacks in Q3",
            "Q4 revenue guidance $880-920 million",
            "Q4 EPS guidance $2.25-2.45",
            "FY2024 revenue guidance raised to $3.31-3.35 billion",
            "FY2024 growth 24-26%",
        ]

    @property
    def banned(self) -> Set[str]:
        return {"bankrupt", "layoffs", "lawsuit", "fraud", "investigation"}

    @property
    def alias_map(self) -> Dict[str, List[str]]:
        return {
            "TechCorp Inc. NASDAQ: TECH": ["TECH", "TechCorp (TECH)"],
            "Q3 FY2024 results announced October 28, 2024": ["Q3 FY24 10/28/24", "Q3 results 10/28/24"],
            "quarter ended September 30, 2024": ["Q ended 9/30/24", "quarter end 9/30/24"],
            "revenue $847.3 million": ["revenue $847.3M", "$847.3M revenue"],
            "23.7% year-over-year growth": ["23.7% YoY growth", "+23.7% YoY"],
            "exceeded consensus $812 million by 4.3%": ["beat consensus $812M by 4.3%", "$812M consensus, beat 4.3%"],
            "cloud infrastructure $423.5 million, up 38.2%": ["cloud $423.5M +38.2%", "cloud infra $423.5M, up 38.2%"],
            "enterprise software $312.8 million, up 18.4%": ["enterprise SW $312.8M +18.4%", "software $312.8M, up 18.4%"],
            "legacy hardware $111.0 million, down 12.3%": ["hardware $111.0M -12.3%", "legacy $111M, down 12.3%"],
            "gross margin 68.2% vs 64.8% prior year": ["GM 68.2% vs 64.8%", "gross margin 68.2%, was 64.8%"],
            "operating expenses $423.6 million, 50.0% of revenue": ["opex $423.6M (50.0%)", "OpEx $423.6M, 50%"],
            "R&D $152.3 million, 18.0% of revenue": ["R&D $152.3M (18%)", "research $152.3M, 18%"],
            "sales and marketing $186.4 million, 22.0% of revenue": ["S&M $186.4M (22%)", "sales/marketing $186.4M, 22%"],
            "G&A $84.9 million, 10.0% of revenue": ["G&A $84.9M (10%)", "admin $84.9M, 10%"],
            "operating income $153.9 million": ["op income $153.9M", "operating income $153.9M"],
            "operating margin 18.2% vs 13.8% prior year": ["op margin 18.2% vs 13.8%", "18.2% op margin, was 13.8%"],
            "net income $127.5 million": ["NI $127.5M", "net income $127.5M"],
            "EPS $2.14 vs $1.52 prior year": ["EPS $2.14 vs $1.52", "$2.14 EPS, was $1.52"],
            "cash $1.24 billion": ["cash $1.24B", "$1.24B cash"],
            "zero debt": ["no debt", "debt-free"],
            "operating cash flow $218.7 million": ["OCF $218.7M", "op cash flow $218.7M"],
            "capex $47.2 million": ["capex $47.2M", "CapEx $47.2M"],
            "free cash flow $171.5 million": ["FCF $171.5M", "free cash flow $171.5M"],
            "new $500 million buyback authorized over 24 months": ["$500M buyback over 24mo", "$500M repurchase program, 24mo"],
            "repurchased 1.8 million shares at $78.50 average": ["bought back 1.8M shares @ $78.50", "1.8M shares repurchased, $78.50 avg"],
            "$141.3 million spent on buybacks in Q3": ["$141.3M buybacks Q3", "$141.3M repurchased Q3"],
            "Q4 revenue guidance $880-920 million": ["Q4 guide $880-920M", "Q4 revenue $880-920M"],
            "Q4 EPS guidance $2.25-2.45": ["Q4 EPS $2.25-2.45", "Q4 guide $2.25-2.45 EPS"],
            "FY2024 revenue guidance raised to $3.31-3.35 billion": ["FY24 raised to $3.31-3.35B", "FY24 guide $3.31-3.35B"],
            "FY2024 growth 24-26%": ["FY24 +24-26%", "FY24 growth 24-26%"],
        }

    @property
    def difficulty(self) -> str:
        return "hard"
