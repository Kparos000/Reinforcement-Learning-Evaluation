"""Legal domain scenario - Service Level Agreement (Phase 2)."""

from typing import Dict, List, Set

from .base import Scenario


class LegalLongScenario(Scenario):
    """Service Level Agreement compression task - realistic length."""

    @property
    def name(self) -> str:
        return "legal_long"

    @property
    def domain(self) -> str:
        return "legal"

    @property
    def original(self) -> str:
        return (
            "This Service Level Agreement (\"SLA\") is entered into effective January 1, 2025, "
            "by and between TechHost Infrastructure Inc., a Delaware corporation with principal "
            "offices at 1500 Technology Drive, San Jose, California 95110 (\"Provider\"), and "
            "Customer Corporation (\"Customer\"). This SLA governs the provision of cloud infrastructure "
            "services as defined in the Master Services Agreement dated December 1, 2024. Provider "
            "guarantees 99.95% monthly uptime for all production systems and services, calculated as "
            "the percentage of time services are available during each calendar month, measured from "
            "00:00:01 UTC on the first day to 23:59:59 UTC on the last day of the month. Uptime "
            "calculations exclude scheduled maintenance windows, Customer-caused outages, force majeure "
            "events, and issues arising from Customer's applications or configurations. Scheduled "
            "maintenance windows shall not exceed 4 hours per calendar month and must be announced with "
            "at least 72 hours advance written notice via email to Customer's designated technical "
            "contacts. Emergency maintenance may be performed with 24 hours notice in cases of critical "
            "security vulnerabilities or system stability issues. In the event Provider fails to meet "
            "the guaranteed 99.95% monthly uptime, Customer shall be entitled to receive service credits "
            "calculated as follows: for monthly uptime between 99.50% and 99.94%, Customer receives a "
            "credit equal to 10% of monthly fees paid; for uptime between 99.00% and 99.49%, credit "
            "equals 25% of monthly fees; for uptime between 95.00% and 98.99%, credit equals 50% of "
            "monthly fees; and for uptime below 95.00%, credit equals 100% of monthly fees. Service "
            "credits represent Customer's sole and exclusive remedy for Provider's failure to meet the "
            "SLA. To receive service credits, Customer must submit a detailed credit request within "
            "30 calendar days following the end of the month in which the SLA breach occurred. Requests "
            "must include specific dates, times (in UTC), duration of outages, and affected services. "
            "Provider will review credit requests and issue approved credits within 45 days of receiving "
            "a complete request. Credits will be applied to Customer's account and may be used against "
            "future invoices but are not redeemable for cash. The initial term of this agreement is "
            "24 months from the effective date, with automatic renewal for successive 12-month periods "
            "unless either party provides written notice of non-renewal at least 90 days prior to the "
            "end of the then-current term. Either party may terminate this agreement for cause upon "
            "30 days written notice if the other party materially breaches any provision and fails to "
            "cure within 30 days of receiving written notice of the breach. Customer may terminate for "
            "convenience upon 90 days written notice, subject to payment of an early termination fee "
            "equal to 50% of fees for the remaining months in the current term, with such fee waived "
            "if termination occurs after month 18 of the initial term. Provider's total liability under "
            "this agreement, whether arising from breach of contract, warranty, tort, or any other legal "
            "theory, shall not exceed an amount equal to 12 months of fees actually paid by Customer "
            "during the 12 months immediately preceding the event giving rise to liability. This "
            "limitation applies to all claims in aggregate. Provider shall maintain comprehensive general "
            "liability insurance of at least $5 million per occurrence and $10 million aggregate, and "
            "cyber liability insurance of at least $10 million per claim and $20 million aggregate, "
            "with Customer named as additional insured."
        )

    @property
    def facts(self) -> List[str]:
        return [
            "effective January 1, 2025",
            "TechHost Infrastructure Inc., Delaware corporation",
            "offices at 1500 Technology Drive, San Jose, CA 95110",
            "Master Services Agreement dated December 1, 2024",
            "99.95% monthly uptime guarantee",
            "measured 00:00:01 UTC first day to 23:59:59 UTC last day",
            "scheduled maintenance max 4 hours per month",
            "72 hours advance notice for scheduled maintenance",
            "24 hours notice for emergency maintenance",
            "uptime 99.50-99.94%: credit 10% of monthly fees",
            "uptime 99.00-99.49%: credit 25% of monthly fees",
            "uptime 95.00-98.99%: credit 50% of monthly fees",
            "uptime below 95.00%: credit 100% of monthly fees",
            "credit requests within 30 days of breach month",
            "credits issued within 45 days of complete request",
            "credits not redeemable for cash",
            "initial term 24 months",
            "auto-renewal for 12-month periods",
            "non-renewal notice 90 days before term end",
            "termination for cause upon 30 days notice",
            "30 days to cure material breach",
            "convenience termination 90 days notice",
            "early termination fee 50% of remaining fees",
            "ETF waived after month 18",
            "liability cap 12 months of fees paid",
            "general liability insurance $5M per occurrence",
            "general liability insurance $10M aggregate",
            "cyber liability insurance $10M per claim",
            "cyber liability insurance $20M aggregate",
        ]

    @property
    def banned(self) -> Set[str]:
        return {"unlimited liability", "no guarantee", "best effort", "as-is"}

    @property
    def alias_map(self) -> Dict[str, List[str]]:
        return {
            "effective January 1, 2025": ["eff. 1/1/2025", "effective 1/1/25"],
            "TechHost Infrastructure Inc., Delaware corporation": ["TechHost Inc. (DE)", "TechHost (Delaware)"],
            "offices at 1500 Technology Drive, San Jose, CA 95110": ["1500 Tech Dr, San Jose CA 95110", "San Jose office"],
            "Master Services Agreement dated December 1, 2024": ["MSA 12/1/2024", "MSA dated 12/1/24"],
            "99.95% monthly uptime guarantee": ["99.95% uptime SLA", "uptime guarantee 99.95%"],
            "measured 00:00:01 UTC first day to 23:59:59 UTC last day": ["measured full month UTC", "calendar month UTC"],
            "scheduled maintenance max 4 hours per month": ["maint max 4h/month", "4hr monthly maintenance max"],
            "72 hours advance notice for scheduled maintenance": ["72h notice scheduled maint", "3 days notice for maintenance"],
            "24 hours notice for emergency maintenance": ["24h notice emergency maint", "1 day notice emergencies"],
            "uptime 99.50-99.94%: credit 10% of monthly fees": ["99.5-99.94% = 10% credit", "10% credit for 99.5-99.94%"],
            "uptime 99.00-99.49%: credit 25% of monthly fees": ["99.0-99.49% = 25% credit", "25% credit for 99-99.49%"],
            "uptime 95.00-98.99%: credit 50% of monthly fees": ["95-98.99% = 50% credit", "50% credit for 95-98.99%"],
            "uptime below 95.00%: credit 100% of monthly fees": ["<95% = 100% credit", "100% credit below 95%"],
            "credit requests within 30 days of breach month": ["credit claim 30 days", "30 day claim window"],
            "credits issued within 45 days of complete request": ["credits issued 45 days", "45 day credit processing"],
            "credits not redeemable for cash": ["no cash redemption", "credits non-refundable"],
            "initial term 24 months": ["24mo initial term", "2-year initial term"],
            "auto-renewal for 12-month periods": ["auto-renew 12mo", "12-month renewals"],
            "non-renewal notice 90 days before term end": ["90 day non-renewal notice", "90d notice to cancel"],
            "termination for cause upon 30 days notice": ["30d termination for cause", "terminate for cause 30 days"],
            "30 days to cure material breach": ["30d cure period", "30 days to cure breach"],
            "convenience termination 90 days notice": ["90d convenience termination", "terminate convenience 90 days"],
            "early termination fee 50% of remaining fees": ["ETF 50% remaining", "50% ETF"],
            "ETF waived after month 18": ["no ETF after 18mo", "ETF waived post-18 months"],
            "liability cap 12 months of fees paid": ["liability capped 12mo fees", "12-month fee liability cap"],
            "general liability insurance $5M per occurrence": ["GL $5M per occurrence", "$5M general liability"],
            "general liability insurance $10M aggregate": ["GL $10M aggregate", "$10M GL aggregate"],
            "cyber liability insurance $10M per claim": ["cyber $10M per claim", "$10M cyber liability"],
            "cyber liability insurance $20M aggregate": ["cyber $20M aggregate", "$20M cyber aggregate"],
        }

    @property
    def difficulty(self) -> str:
        return "hard"
