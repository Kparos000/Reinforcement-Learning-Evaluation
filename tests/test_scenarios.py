"""Tests for scenario registry and domain scenarios."""

import pytest

from ace_task.scenarios import (
    EconomicsScenario,
    LegalScenario,
    MedicalScenario,
    ScientificScenario,
    get_scenario,
    list_scenarios,
)


class TestScenarioRegistry:
    """Test scenario registry functionality."""

    def test_list_scenarios(self):
        scenarios = list_scenarios()
        assert isinstance(scenarios, list)
        assert len(scenarios) >= 4
        assert "economics" in scenarios
        assert "medical" in scenarios
        assert "legal" in scenarios
        assert "scientific" in scenarios

    def test_get_scenario_economics(self):
        scenario = get_scenario("economics")
        assert isinstance(scenario, EconomicsScenario)
        assert scenario.name == "economics"

    def test_get_scenario_medical(self):
        scenario = get_scenario("medical")
        assert isinstance(scenario, MedicalScenario)
        assert scenario.name == "medical"

    def test_get_scenario_invalid(self):
        with pytest.raises(KeyError):
            get_scenario("nonexistent")


class TestScenarioInterface:
    """Test that all scenarios implement the required interface."""

    @pytest.fixture(params=["economics", "medical", "legal", "scientific"])
    def scenario(self, request):
        return get_scenario(request.param)

    def test_has_name(self, scenario):
        assert isinstance(scenario.name, str)
        assert len(scenario.name) > 0

    def test_has_domain(self, scenario):
        assert isinstance(scenario.domain, str)
        assert len(scenario.domain) > 0

    def test_has_original(self, scenario):
        assert isinstance(scenario.original, str)
        assert len(scenario.original) > 0

    def test_has_facts(self, scenario):
        assert isinstance(scenario.facts, list)
        assert len(scenario.facts) > 0
        assert all(isinstance(f, str) for f in scenario.facts)

    def test_has_banned(self, scenario):
        assert isinstance(scenario.banned, set)
        assert len(scenario.banned) > 0

    def test_has_alias_map(self, scenario):
        assert isinstance(scenario.alias_map, dict)

    def test_has_difficulty(self, scenario):
        assert scenario.difficulty in ["easy", "medium", "hard"]


class TestEconomicsScenario:
    """Test economics-specific scenario."""

    def setup_method(self):
        self.scenario = EconomicsScenario()

    def test_facts_in_original(self):
        original_lower = self.scenario.original.lower()
        # Check that key concepts appear
        assert "gdp" in original_lower
        assert "3.2%" in self.scenario.original
        assert "inflation" in original_lower

    def test_banned_not_in_original(self):
        original_lower = self.scenario.original.lower()
        for banned in self.scenario.banned:
            assert banned.lower() not in original_lower

    def test_difficulty(self):
        assert self.scenario.difficulty == "easy"


class TestMedicalScenario:
    """Test medical-specific scenario."""

    def setup_method(self):
        self.scenario = MedicalScenario()

    def test_medical_terminology(self):
        original = self.scenario.original
        assert "efficacy" in original or "Efficacy" in original
        assert "dose" in original or "Dose" in original
        assert "adverse" in original or "Adverse" in original

    def test_numeric_precision(self):
        assert "87%" in self.scenario.original
        assert "15mg" in self.scenario.original
        assert "12%" in self.scenario.original

    def test_difficulty(self):
        assert self.scenario.difficulty == "medium"


class TestLegalScenario:
    """Test legal-specific scenario."""

    def setup_method(self):
        self.scenario = LegalScenario()

    def test_legal_terminology(self):
        original = self.scenario.original
        assert "terminate" in original.lower()
        assert "notice" in original.lower()

    def test_dates_and_amounts(self):
        assert "2025" in self.scenario.original
        assert "$50,000" in self.scenario.original
        assert "90 days" in self.scenario.original

    def test_difficulty(self):
        assert self.scenario.difficulty == "hard"


class TestScientificScenario:
    """Test scientific-specific scenario."""

    def setup_method(self):
        self.scenario = ScientificScenario()

    def test_scientific_terminology(self):
        original = self.scenario.original
        assert "catalyst" in original.lower() or "conversion" in original.lower()
        assert "selectivity" in original.lower()

    def test_units_and_measurements(self):
        assert "°C" in self.scenario.original
        assert "bar" in self.scenario.original
        assert "%" in self.scenario.original

    def test_difficulty(self):
        assert self.scenario.difficulty == "medium"


class TestAliasMapValidity:
    """Test that alias maps are well-formed."""

    @pytest.fixture(params=["economics", "medical", "legal", "scientific"])
    def scenario(self, request):
        return get_scenario(request.param)

    def test_aliases_match_facts(self, scenario):
        """Aliases should be for facts that exist."""
        for fact in scenario.alias_map.keys():
            assert fact in scenario.facts, f"Alias for '{fact}' but fact not in facts list"

    def test_aliases_are_different(self, scenario):
        """Aliases should differ from the original fact."""
        for fact, aliases in scenario.alias_map.items():
            for alias in aliases:
                assert alias != fact, f"Alias '{alias}' identical to fact '{fact}'"

    def test_no_empty_alias_lists(self, scenario):
        """No alias list should be empty."""
        for fact, aliases in scenario.alias_map.items():
            assert len(aliases) > 0, f"Fact '{fact}' has empty alias list"
