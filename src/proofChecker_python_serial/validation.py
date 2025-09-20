"""Common validation utilities for dataclass post_init methods."""

from typing import Any


def validate_index(index: Any, field_name: str = "Index") -> None:
    """Validate that index is a non-negative integer."""
    if not isinstance(index, int):
        raise ValueError(f"{field_name} must be an integer.")

    if index < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")


def validate_label(
    label: Any, allow_digits: bool = False, field_name: str = "Label"
) -> None:
    """Validate that label is a non-empty string, optionally without digits."""
    if not isinstance(label, str):
        raise ValueError(f"{field_name} must be a string.")

    if label == "":
        raise ValueError(f"{field_name} must be a non-empty string.")

    if not allow_digits and any(char.isdigit() for char in label):
        raise ValueError(f"{field_name} must not contain digits.")


def validate_common_fields(obj: Any, allow_digits_in_label: bool = False) -> None:
    """Validate common fields (index and label) present in multiple classes."""

    if not hasattr(obj, "index") or not hasattr(obj, "label"):
        raise ValueError("Object must have 'index' and 'label' attributes.")

    validate_index(getattr(obj, "index"))
    validate_label(getattr(obj, "label"), allow_digits=allow_digits_in_label)
