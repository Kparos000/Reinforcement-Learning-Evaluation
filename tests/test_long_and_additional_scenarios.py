from ace_task.scenarios import BusinessLongScenario, FinanceScenario, LegalLongScenario, MedicalLongScenario, SportsScenario


def test_long_scenarios_have_facts_and_aliases():
    scenarios = [
        MedicalLongScenario(),
        BusinessLongScenario(),
        LegalLongScenario(),
    ]

    for scenario in scenarios:
        assert len(scenario.facts) >= 20
        assert scenario.original
        assert scenario.alias_map


def test_finance_and_sports_scenarios_minimum_coverage():
    finance = FinanceScenario()
    sports = SportsScenario()

    assert len(finance.facts) >= 5
    assert len(sports.facts) >= 5
    assert "P/E" in "".join(finance.facts)
    assert "points" in " ".join(sports.facts).lower()
