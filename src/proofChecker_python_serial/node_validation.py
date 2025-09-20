"""Node-specific validation utilities."""

from typing import Any
from proofChecker_python_serial.node import Node


def validate_node_list(node_list: Any, field_name: str) -> None:
    """Validate that field is a non-empty list of Node objects."""
    if not isinstance(node_list, list):
        raise ValueError(f"{field_name} must be a list.")

    if not all(isinstance(node, Node) for node in node_list):
        raise ValueError(f"All elements in {field_name} must be Node objects.")

    if len(node_list) == 0:
        raise ValueError(f"{field_name} must be a non-empty list of Node objects.")
