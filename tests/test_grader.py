"""Tests for the deterministic grader."""

import json

from ace_task.data import ALIAS_MAP, BANNED, FACTS, ORIGINAL
from ace_task.grader import _fact_present, _has_sentence, _norm, _nums, _word_count, grade


class TestNormalization:
    """Test text normalization utilities."""

    def test_norm_lowercase(self):
        assert _norm("GDP Grew") == "gdp grew"

    def test_norm_punctuation(self):
        assert _norm("GDP grew by 3.2%!") == "gdp grew by 3.2%"
        assert _norm("Price: $5.1B") == "price $5.1b"

    def test_norm_whitespace(self):
        assert _norm("GDP   grew    by 3.2%") == "gdp grew by 3.2%"


class TestWordCount:
    """Test word counting."""

    def test_simple_count(self):
        assert _word_count("one two three") == 3

    def test_punctuation_words(self):
        # "GDP", "grew", "by", "3", "2" - decimal creates two number tokens
        assert _word_count("GDP grew by 3.2%") == 5

    def test_empty_string(self):
        assert _word_count("") == 0


class TestFactPresence:
    """Test fact detection with aliases."""

    def test_exact_match(self):
        rewrite = "gdp grew by 3.2%"
        assert _fact_present(rewrite, "GDP grew by 3.2%", ALIAS_MAP)

    def test_alias_match(self):
        rewrite = "gdp 3.2%"  # Normalized version of "GDP +3.2%"
        assert _fact_present(rewrite, "GDP grew by 3.2%", ALIAS_MAP)

    def test_no_match(self):
        rewrite = "gdp increased"
        assert not _fact_present(rewrite, "GDP grew by 3.2%", ALIAS_MAP)


class TestNumericExtraction:
    """Test numeric value extraction."""

    def test_percentage(self):
        nums = _nums("GDP grew 3.2% and inflation 2.1%")
        assert "3.2%" in nums
        assert "2.1%" in nums

    def test_currency(self):
        nums = _nums("Budget deficit $9.9B")
        assert "$9.9B" in nums or "$9.9" in nums

    def test_plain_numbers(self):
        nums = _nums("In Q2 the value was 2 units")
        assert "2" in nums  # Standalone "2" matches \b\d+ pattern


class TestSentenceValidation:
    """Test sentence structure validation."""

    def test_has_period(self):
        assert _has_sentence("This is a sentence.")

    def test_has_question(self):
        assert _has_sentence("Is this a sentence?")

    def test_sufficient_words(self):
        assert _has_sentence("one two three four five six seven")

    def test_insufficient(self):
        assert not _has_sentence("too short")


class TestGradeFunction:
    """Test the main grading function."""

    def test_valid_output(self):
        """Test that a valid output passes all checks."""
        valid_json = json.dumps(
            {
                "rewrite": "Q2: GDP +3.2%, inflation 2.1%; exports rose.",
                "preserved_facts": ["GDP grew by 3.2%", "inflation was 2.1%", "exports increased"],
                "at_risk_facts": [],
                "key_insight": "Preserving numeric detail prevents context collapse.",
                "delta_update": "Use numeric shorthand and semicolons to reduce words.",
            }
        )

        ok, msg = grade(ORIGINAL, FACTS, BANNED, valid_json, alias_map=ALIAS_MAP)
        assert ok, f"Valid output should pass but failed with: {msg}"
        assert msg == "pass"

    def test_invalid_json(self):
        """Test that malformed JSON fails."""
        ok, msg = grade(ORIGINAL, FACTS, BANNED, "not json", alias_map=ALIAS_MAP)
        assert not ok
        assert "Bad JSON" in msg

    def test_missing_keys(self):
        """Test that missing required keys fail."""
        incomplete = json.dumps({"rewrite": "test", "preserved_facts": []})
        ok, msg = grade(ORIGINAL, FACTS, BANNED, incomplete, alias_map=ALIAS_MAP)
        assert not ok
        assert "Wrong keys" in msg

    def test_empty_rewrite(self):
        """Test that empty rewrite fails."""
        empty_rewrite = json.dumps(
            {
                "rewrite": "",
                "preserved_facts": [],
                "at_risk_facts": [],
                "key_insight": "test",
                "delta_update": "test",
            }
        )
        ok, msg = grade(ORIGINAL, FACTS, BANNED, empty_rewrite, alias_map=ALIAS_MAP)
        assert not ok
        assert "non-empty string" in msg

    def test_too_long_rewrite(self):
        """Test that overly long rewrite fails concision check."""
        long_rewrite = json.dumps(
            {
                "rewrite": ORIGINAL * 2,  # Definitely >60%
                "preserved_facts": FACTS,
                "at_risk_facts": [],
                "key_insight": "preserving quantitative detail prevents context collapse",
                "delta_update": "Use shorthand and reduce connecting words.",
            }
        )
        ok, msg = grade(ORIGINAL, FACTS, BANNED, long_rewrite, alias_map=ALIAS_MAP)
        assert not ok
        assert "not concise enough" in msg

    def test_banned_term(self):
        """Test that banned terms are detected."""
        with_banned = json.dumps(
            {
                "rewrite": "Q2: recession looms, GDP +3.2%, inflation 2.1%",
                "preserved_facts": FACTS,
                "at_risk_facts": [],
                "key_insight": "preserving quantitative metrics prevents collapse",
                "delta_update": "Keep numbers precise and use shorthand.",
            }
        )
        ok, msg = grade(ORIGINAL, FACTS, BANNED, with_banned, alias_map=ALIAS_MAP)
        assert not ok
        assert "banned term" in msg.lower()

    def test_missing_fact(self):
        """Test that missing facts are detected."""
        missing_fact = json.dumps(
            {
                "rewrite": "Q2: GDP +3.2%",  # Missing inflation and exports
                "preserved_facts": ["GDP grew by 3.2%"],
                "at_risk_facts": [],
                "key_insight": "preserving numeric detail prevents context collapse",
                "delta_update": "Use shorthand notation for efficiency.",
            }
        )
        ok, msg = grade(ORIGINAL, FACTS, BANNED, missing_fact, alias_map=ALIAS_MAP)
        assert not ok
        assert "Missing facts" in msg

    def test_numeric_fidelity(self):
        """Test that changing numbers fails."""
        wrong_numbers = json.dumps(
            {
                "rewrite": "Q2: GDP +3.5%, inflation 2.0%; exports rose.",
                "preserved_facts": ["GDP grew by 3.2%", "inflation was 2.1%", "exports increased"],
                "at_risk_facts": [],
                "key_insight": "preserving quantitative data prevents context collapse",
                "delta_update": "Maintain exact numeric precision throughout.",
            }
        )
        ok, msg = grade(ORIGINAL, FACTS, BANNED, wrong_numbers, alias_map=ALIAS_MAP)
        assert not ok
        # Changed numbers cause both fact mismatch and numeric loss - either error is valid
        assert "Missing facts" in msg or "Numeric info lost" in msg

    def test_weak_key_insight(self):
        """Test that non-ACE-aligned key_insight fails."""
        weak_insight = json.dumps(
            {
                "rewrite": "Q2: GDP +3.2%, inflation 2.1%; exports rose.",
                "preserved_facts": FACTS,
                "at_risk_facts": [],
                "key_insight": "This is a good summary.",  # Not ACE-aligned
                "delta_update": "Make summaries more concise going forward.",
            }
        )
        ok, msg = grade(ORIGINAL, FACTS, BANNED, weak_insight, alias_map=ALIAS_MAP)
        assert not ok
        assert "key_insight not ACE-aligned" in msg

    def test_weak_delta_update(self):
        """Test that short delta_update fails."""
        weak_delta = json.dumps(
            {
                "rewrite": "Q2: GDP +3.2%, inflation 2.1%; exports rose.",
                "preserved_facts": FACTS,
                "at_risk_facts": [],
                "key_insight": "Preserving quantitative detail prevents context collapse.",
                "delta_update": "Be brief.",  # Too short
            }
        )
        ok, msg = grade(ORIGINAL, FACTS, BANNED, weak_delta, alias_map=ALIAS_MAP)
        assert not ok
        assert "delta_update not a clear actionable sentence" in msg
