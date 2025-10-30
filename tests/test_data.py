"""Tests for data fixtures and aliases."""

from ace_task.data import ALIAS_MAP, BANNED, FACTS, ORIGINAL


class TestDataFixtures:
    """Test that data fixtures are well-formed."""

    def test_original_exists(self):
        assert isinstance(ORIGINAL, str)
        assert len(ORIGINAL) > 0

    def test_facts_list(self):
        assert isinstance(FACTS, list)
        assert len(FACTS) > 0
        assert all(isinstance(f, str) for f in FACTS)

    def test_banned_set(self):
        assert isinstance(BANNED, set)
        assert len(BANNED) > 0
        assert all(isinstance(b, str) for b in BANNED)

    def test_alias_map_structure(self):
        assert isinstance(ALIAS_MAP, dict)
        for fact, aliases in ALIAS_MAP.items():
            assert isinstance(fact, str)
            assert isinstance(aliases, list)
            assert all(isinstance(a, str) for a in aliases)


class TestFactsInOriginal:
    """Test that all FACTS appear in ORIGINAL."""

    def test_all_facts_present(self):
        """Verify that all facts can be found in the original text."""
        original_lower = ORIGINAL.lower()
        for fact in FACTS:
            # Simple substring check - facts should be in original
            fact_words = set(fact.lower().replace("%", "").split())
            original_words = set(original_lower.replace("%", "").split())
            # At least some words from each fact should appear
            assert len(fact_words & original_words) > 0, f"Fact '{fact}' not reflected in ORIGINAL"


class TestBannedTermsNotInOriginal:
    """Test that BANNED terms don't appear in ORIGINAL."""

    def test_no_banned_in_original(self):
        """Verify original text doesn't contain banned terms."""
        original_lower = ORIGINAL.lower()
        for banned in BANNED:
            assert banned.lower() not in original_lower, f"Banned term '{banned}' in ORIGINAL"


class TestAliasMapCoverage:
    """Test alias map properties."""

    def test_aliases_for_key_facts(self):
        """Check that we have aliases for expected facts."""
        # At minimum, GDP and inflation should have aliases
        assert any("GDP" in fact for fact in ALIAS_MAP.keys())
        assert any("inflation" in fact for fact in ALIAS_MAP.keys())

    def test_aliases_are_different(self):
        """Aliases should differ from original facts."""
        for fact, aliases in ALIAS_MAP.items():
            for alias in aliases:
                assert alias != fact, f"Alias '{alias}' is identical to fact '{fact}'"

    def test_no_empty_aliases(self):
        """No alias list should be empty."""
        for fact, aliases in ALIAS_MAP.items():
            assert len(aliases) > 0, f"Fact '{fact}' has empty alias list"
