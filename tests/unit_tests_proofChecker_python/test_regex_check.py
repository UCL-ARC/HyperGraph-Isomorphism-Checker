"""Tests for regex_check module."""

from proofChecker_python_serial.regex_check import (
    matches_unit_pattern,
    matches_group_pattern,
    matches_word_pattern,
    matches_signature_pattern,
)


class TestUnitPattern:
    """Test the unit pattern: digits+ then chars+ (chars exclude digits, '-', '_', and space)."""

    def test_valid_units(self):
        """Test valid unit patterns."""
        valid_units = ["1a", "123abc", "0xyz", "42F", "999ZZZ", "1A", "23bcdefg"]
        for unit in valid_units:
            assert matches_unit_pattern(unit), f"'{unit}' should match unit pattern"

    def test_invalid_units_no_digits(self):
        """Test units that don't start with digits."""
        invalid_units = ["abc", "A123", "xyz1", "F42"]
        for unit in invalid_units:
            assert not matches_unit_pattern(
                unit
            ), f"'{unit}' should not match unit pattern"

    def test_invalid_units_no_chars(self):
        """Test units that are only digits."""
        invalid_units = ["123", "0", "999"]
        for unit in invalid_units:
            assert not matches_unit_pattern(
                unit
            ), f"'{unit}' should not match unit pattern"

    def test_invalid_units_forbidden_chars(self):
        """Test units containing forbidden characters."""
        invalid_units = [
            "1a-b",  # contains hyphen
            "123_abc",  # contains underscore
            "42 F",  # contains space
            "1a2",  # contains digit after letters
            "123a-",  # ends with hyphen
            "0xyz_",  # ends with underscore
        ]
        for unit in invalid_units:
            assert not matches_unit_pattern(
                unit
            ), f"'{unit}' should not match unit pattern"

    def test_empty_and_special_cases(self):
        """Test empty string and edge cases."""
        assert not matches_unit_pattern("")
        assert not matches_unit_pattern("1")  # only digit
        assert not matches_unit_pattern("a")  # only letter


class TestGroupPattern:
    """Test the group pattern: one or more units separated by underscores."""

    def test_valid_single_unit_groups(self):
        """Test groups with single units."""
        valid_groups = ["1a", "123abc", "42F"]
        for group in valid_groups:
            assert matches_group_pattern(group), f"'{group}' should match group pattern"

    def test_valid_multi_unit_groups(self):
        """Test groups with multiple units."""
        valid_groups = ["1a_2b", "123abc_456def", "1A_2B_3C", "42F_99G_1H_88I"]
        for group in valid_groups:
            assert matches_group_pattern(group), f"'{group}' should match group pattern"

    def test_invalid_groups_bad_separators(self):
        """Test groups with invalid separators."""
        invalid_groups = [
            "1a-2b",  # hyphen instead of underscore
            "1a 2b",  # space instead of underscore
            "1a_2b-3c",  # mixed separators
            "1a__2b",  # double underscore
        ]
        for group in invalid_groups:
            assert not matches_group_pattern(
                group
            ), f"'{group}' should not match group pattern"

    def test_invalid_groups_bad_units(self):
        """Test groups containing invalid units."""
        invalid_groups = [
            "abc_def",  # units don't start with digits
            "1a_2",  # second unit is only digits
            "123_456def",  # first unit is only digits
            "1a_2b3",  # unit contains digit after letters
        ]
        for group in invalid_groups:
            assert not matches_group_pattern(
                group
            ), f"'{group}' should not match group pattern"

    def test_empty_and_edge_cases(self):
        """Test empty string and edge cases."""
        assert not matches_group_pattern("")
        assert not matches_group_pattern("_")
        assert not matches_group_pattern("1a_")
        assert not matches_group_pattern("_1a")


class TestWordPattern:
    """Test the word pattern: unit-group-group."""

    def test_valid_words(self):
        """Test valid word patterns."""
        valid_words = [
            "1a-2b-3c",
            "123F-456G-789H",
            "1A-2B_3C-4D",
            "42x-1y_2z-99w",
            "1a-2b_3c_4d-5e_6f",
        ]
        for word in valid_words:
            assert matches_word_pattern(word), f"'{word}' should match word pattern"

    def test_invalid_words_wrong_structure(self):
        """Test words with wrong structure."""
        invalid_words = [
            "1a-2b",  # only two parts
            "1a-2b-3c-4d",  # four parts
            "1a_2b_3c",  # underscores instead of hyphens
            "1a 2b 3c",  # spaces instead of hyphens
            "1a",  # single unit
            "",  # empty string
        ]
        for word in invalid_words:
            assert not matches_word_pattern(
                word
            ), f"'{word}' should not match word pattern"

    def test_invalid_words_bad_components(self):
        """Test words with invalid unit/group components."""
        invalid_words = [
            "abc-2b-3c",  # first part not a unit
            "1a-def-3c",  # second part not a group
            "1a-2b-ghi",  # third part not a group
            "123-2b-3c",  # first part only digits
            "1a-456-3c",  # second part only digits
            "1a-2b-789",  # third part only digits
        ]
        for word in invalid_words:
            assert not matches_word_pattern(
                word
            ), f"'{word}' should not match word pattern"


class TestSignaturePattern:
    """Test the full signature pattern: word (space word)*."""

    def test_valid_single_word_signatures(self):
        """Test signatures with single words."""
        valid_signatures = ["1a-2b-3c", "123F-456G-789H", "1A-2B_3C-4D_5E"]
        for signature in valid_signatures:
            assert matches_signature_pattern(
                signature
            ), f"'{signature}' should match signature pattern"

    def test_valid_multi_word_signatures(self):
        """Test signatures with multiple words."""
        valid_signatures = [
            "1a-2b-3c 4d-5e-6f",
            "123F-456G-789H 1A-2B-3C",
            "1a-2b_3c-4d 5e-6f-7g 8h-9i_10j-11k",
            "42x-1y-2z 99a-88b_77c-66d_55e",
        ]
        for signature in valid_signatures:
            assert matches_signature_pattern(
                signature
            ), f"'{signature}' should match signature pattern"

    def test_invalid_signatures_bad_separators(self):
        """Test signatures with invalid word separators."""
        invalid_signatures = [
            "1a-2b-3c  4d-5e-6f",  # double space
            "1a-2b-3c\t4d-5e-6f",  # tab
            "1a-2b-3c_4d-5e-6f",  # underscore
            "1a-2b-3c-4d-5e-6f",  # hyphen
            "1a-2b-3c,4d-5e-6f",  # comma
        ]
        for signature in invalid_signatures:
            assert not matches_signature_pattern(
                signature
            ), f"'{signature}' should not match signature pattern"

    def test_invalid_signatures_bad_words(self):
        """Test signatures containing invalid words."""
        invalid_signatures = [
            "1a-2b 3c-4d-5e",  # first word invalid
            "1a-2b-3c 4d-5e",  # second word invalid
            "abc-def-ghi",  # word doesn't start with digits
            "123-456-789",  # word is only digits
            "1a-2b-3c 4d-5e-6f-7g",  # second word has too many parts
        ]
        for signature in invalid_signatures:
            assert not matches_signature_pattern(
                signature
            ), f"'{signature}' should not match signature pattern"

    def test_empty_and_edge_cases(self):
        """Test empty string and edge cases."""
        assert not matches_signature_pattern("")
        assert not matches_signature_pattern(" ")
        assert not matches_signature_pattern("1a-2b-3c ")  # trailing space
        assert not matches_signature_pattern(" 1a-2b-3c")  # leading space


class TestRegexPatternConsistency:
    """Test that the patterns are consistent with each other."""

    def test_unit_in_group_consistency(self):
        """Test that valid units work in groups."""
        valid_unit = "123abc"
        assert matches_unit_pattern(valid_unit)
        assert matches_group_pattern(valid_unit)  # single unit is also a valid group

    def test_group_in_word_consistency(self):
        """Test that valid groups work in words."""
        valid_group = "1a_2b"
        assert matches_group_pattern(valid_group)

        # Use the group in a word
        word = f"3c-{valid_group}-4d"
        assert matches_word_pattern(word)

    def test_word_in_signature_consistency(self):
        """Test that valid words work in signatures."""
        valid_word = "1a-2b_3c-4d_5e"
        assert matches_word_pattern(valid_word)
        assert matches_signature_pattern(
            valid_word
        )  # single word is also a valid signature

    def test_complex_consistency(self):
        """Test consistency across all pattern levels."""
        # Build up from unit to full signature
        unit1, unit2, unit3 = "123abc", "456def", "789ghi"

        # Test individual units
        for unit in [unit1, unit2, unit3]:
            assert matches_unit_pattern(unit)

        # Build groups
        group1 = f"{unit1}_{unit2}"
        group2 = unit3

        for group in [group1, group2]:
            assert matches_group_pattern(group)

        # Build word
        word = f"{unit1}-{group1}-{group2}"
        assert matches_word_pattern(word)

        # Build signature
        signature = f"{word} {word}"
        assert matches_signature_pattern(signature)
