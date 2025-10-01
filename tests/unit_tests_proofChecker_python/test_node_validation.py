"""Module for testing node validation."""

import pytest
from proofChecker_python_serial.node import Node
from proofChecker_python_serial.node_validation import validate_node_list


def test_validate_node_list_valid():
    valid_nodes = [Node(index=0, label="a"), Node(index=1, label="b")]
    validate_node_list(valid_nodes, "test_field")


def test_validate_node_list_invalid_type():
    with pytest.raises(ValueError):
        validate_node_list("not_a_list", "test_field")


def test_validate_node_list_empty():
    with pytest.raises(ValueError):
        validate_node_list([], "test_field")


def test_validate_node_list_invalid_element():
    with pytest.raises(ValueError):
        validate_node_list([Node(index=0, label="a"), "not_a_node"], "test_field")
