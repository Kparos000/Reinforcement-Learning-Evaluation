from ace_task.algorithms.rewards import (
    binary_reward_from_grade,
    compute_dense_reward,
    dense_reward_from_grade,
)
from ace_task.grader import GradeResult, grade_detailed
from ace_task.scenarios import get_scenario


def test_dense_reward_perfect():
    grade = GradeResult(
        passed=True,
        reason="pass",
        facts_total=5,
        facts_matched=5,
        banned_term_violations=0,
        length_violation=False,
    )
    reward = dense_reward_from_grade(grade)
    assert 0.99 <= reward <= 1.0
    assert binary_reward_from_grade(grade) == 1.0


def test_dense_reward_with_penalties():
    grade = GradeResult(
        passed=False,
        reason="missing facts",
        facts_total=5,
        facts_matched=2,
        banned_term_violations=1,
        length_violation=True,
    )
    reward = dense_reward_from_grade(grade)
    # Base coverage 0.4, minus penalties, clipped >= 0
    assert 0.0 <= reward < 0.4
    assert binary_reward_from_grade(grade) == 0.0


def test_dense_reward_not_negative():
    grade = GradeResult(
        passed=False,
        reason="bad",
        facts_total=10,
        facts_matched=0,
        banned_term_violations=3,
        length_violation=True,
    )
    reward = dense_reward_from_grade(grade)
    assert reward >= 0.0


def test_compute_dense_reward_with_real_grade():
    scenario = get_scenario("economics")
    model_output = (
        '{'
        '"rewrite": "GDP grew by 3.2%, inflation 2.1%, exports increased",'
        '"preserved_facts": ["GDP grew by 3.2%", "inflation was 2.1%", "exports increased"],'
        '"at_risk_facts": [],'
        '"key_insight": "preserving quantitative details prevents context collapse",'
        '"delta_update": "always keep numeric details to avoid losing critical information."'
        '}'
    )
    grade = grade_detailed(
        original=scenario.original,
        facts=scenario.facts,
        banned=scenario.banned,
        model_text=model_output,
        alias_map=scenario.alias_map,
    )
    reward = compute_dense_reward(scenario, model_output)
    assert grade.passed
    assert reward > 0.8
