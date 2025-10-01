"""Tests for validation functions in the proofChecker module which are not covered elsewhere."""

import pytest
from proofChecker_python_serial.validation import validate_common_fields


class LabelOnly:
    label = "A"


class IndexOnly:
    index = 0


class ValidObject:
    index = 1
    label = "A"


def test_label_only_object_validation():
    """Test that an object without index raises ValueError."""

    with pytest.raises(ValueError):
        validate_common_fields(LabelOnly())


def test_index_only_object_validation():
    """Test that an object without label raises ValueError."""

    with pytest.raises(ValueError):
        validate_common_fields(IndexOnly())


def test_valid_object_validation():
    """Test that a valid object passes validation."""
    validate_common_fields(ValidObject())


def test_object_with_digits_in_label():
    """Test that an object with digits in label raises ValueError."""

    class ObjWithDigits:
        index = 1
        label = "A1"

    with pytest.raises(ValueError):
        validate_common_fields(ObjWithDigits())


def test_object_with_invalid_label_characters():
    """Test that an object with invalid characters in label raises ValueError."""

    class ObjWithInvalidLabel:
        index = 1
        label = "A-B"

    with pytest.raises(ValueError):
        validate_common_fields(ObjWithInvalidLabel())
