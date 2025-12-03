from ace_task.scenarios import get_scenario


def test_report_top15_facts_and_limits():
    scenario = get_scenario("report_top15")

    assert len(scenario.facts) == 15
    assert scenario.word_cap == 140
    assert scenario.concision_limit == 0.70

    # Basic sanity: aliases align with facts
    for fact in scenario.facts:
        if fact in scenario.alias_map:
            for alias in scenario.alias_map[fact]:
                assert alias != fact
