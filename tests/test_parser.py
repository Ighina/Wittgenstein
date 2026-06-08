"""Tests for the parser layer."""

import pytest
from src.parser.location_parser import (
    parse_error_location,
    fuzzy_match_locations,
    LocationType,
)
from src.parser.schema_analyzer import analyze_dataset_schema


class TestLocationParser:
    """Tests for error location parsing."""

    @pytest.mark.parametrize(
        "raw,expected_type,expected_normalized",
        [
            ("Equation 6", LocationType.EQUATION, "equation 6"),
            ("Eq. 1", LocationType.EQUATION, "equation 1"),
            ("Eq. (12)", LocationType.EQUATION, "equation 12"),
            ("Fig 5", LocationType.FIGURE, "figure 5"),
            ("Fig. 4", LocationType.FIGURE, "figure 4"),
            ("Figure 2d", LocationType.FIGURE, "figure 2d"),
            ("Lemma 3,4", LocationType.LEMMA, "lemma 3,4"),
            ("Lemma 1", LocationType.LEMMA, "lemma 1"),
            ("Lemma 4.2", LocationType.LEMMA, "lemma 4.2"),
            ("Theorem 1.1", LocationType.THEOREM, "theorem 1.1"),
            ("Theorems 1.2, 1.3", LocationType.THEOREM, "theorem 1.2,1.3"),
            ("Theorem 7", LocationType.THEOREM, "theorem 7"),
            ("Proposition 2", LocationType.PROPOSITION, "proposition 2"),
            ("Proposition 3.9", LocationType.PROPOSITION, "proposition 3.9"),
            ("Section 4.2.3", LocationType.SECTION, "section 4.2.3"),
            ("Sec 3", LocationType.SECTION, "section 3"),
            ("Sec 3.1", LocationType.SECTION, "section 3.1"),
            ("Table 2", LocationType.TABLE, "table 2"),
            ("Table. 1", LocationType.TABLE, "table 1"),
            ("Page 4", LocationType.PAGE, "page 4"),
            ("Claim 3", LocationType.CLAIM, "claim 3"),
            ("Claim 7", LocationType.CLAIM, "claim 7"),
            ("1.10. Corollary", LocationType.COROLLARY, "corollary 1.10"),
            ("Appendix B", LocationType.APPENDIX, "appendix B"),
        ],
    )
    def test_parse_known_formats(self, raw, expected_type, expected_normalized):
        """Test parsing of all known location formats."""
        ref = parse_error_location(raw)
        assert ref.location_type == expected_type, f"Expected {expected_type}, got {ref.location_type}"
        assert ref.normalized == expected_normalized, f"Expected '{expected_normalized}', got '{ref.normalized}'"

    def test_parse_unknown_returns_unknown_type(self):
        """Test that unrecognizable strings map to UNKNOWN."""
        ref = parse_error_location("Gobbledygook nonsense")
        assert ref.location_type == LocationType.UNKNOWN

    def test_parse_overall(self):
        """Test that 'Overall' is recognized."""
        ref = parse_error_location("Overall")
        assert ref.location_type == LocationType.OVERALL

    def test_parse_overview(self):
        """Test that 'Overview' is recognized."""
        ref = parse_error_location("Overview")
        assert ref.location_type == LocationType.OVERALL

    def test_multi_identifier_split(self):
        """Test splitting of multi-reference identifiers."""
        ref = parse_error_location("Lemma 3,4")
        assert len(ref.identifiers) == 2
        assert "3" in ref.identifiers
        assert "4" in ref.identifiers
        assert ref.is_range is True

    def test_single_identifier(self):
        """Test single identifier is not a range."""
        ref = parse_error_location("Equation 6")
        assert len(ref.identifiers) == 1
        assert ref.is_range is False


class TestFuzzyMatching:
    """Tests for fuzzy location matching."""

    @pytest.mark.parametrize(
        "a,b,min_score",
        [
            ("Equation 7", "Eq. (7)", 1.0),
            ("Fig 5", "Figure 5", 1.0),
            ("Section 3.1", "Sec 3.1", 1.0),
            ("Section 3.1", "§3.1", 0.85),
            ("Lemma 3,4", "Lemma 3", 0.7),
            ("Equation 6", "Equation 7", 0.8),
            ("Theorem 1.1", "Theorem 2.1", 0.8),
        ],
    )
    def test_fuzzy_match_expected_scores(self, a, b, min_score):
        """Test fuzzy matching produces expected scores."""
        score = fuzzy_match_locations(a, b)
        assert score >= min_score, f"Match '{a}' <-> '{b}' scored {score}, expected >= {min_score}"

    def test_different_types_no_match(self):
        """Test that different location types get low scores."""
        score = fuzzy_match_locations("Equation 6", "Figure 6")
        assert score < 0.5

    def test_same_location_exact_match(self):
        """Test that identical locations match perfectly."""
        score = fuzzy_match_locations("Lemma 3,4", "Lemma 3,4")
        assert score == 1.0


class TestSchemaAnalyzer:
    """Tests for dataset schema analysis."""

    def test_analyze_returns_report(self):
        """Test that schema analysis returns a valid report."""
        report = analyze_dataset_schema(
            "data/train-00000-of-00001.parquet",
            sample_rows=2,
        )
        assert report.total_rows == 68
        assert report.total_columns == 10
        assert "text" in report.content_types
        assert "image_url" in report.content_types
        assert len(report.column_names) > 0
        assert report.text_item_count > 0
        assert report.image_item_count > 0
