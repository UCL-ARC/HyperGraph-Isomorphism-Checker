"""Tests for node class in hypergraph module."""

import pytest
from proofChecker_python_serial.hypergraph import Node


def test_node_creation():
    node = Node(index=0, label="a")
    assert node.label == "a"
    assert node.prev is None
    assert node.next is None


def test_node_creation_with_user_defined_prev():
    with pytest.raises(TypeError):
        Node(label="A", prev=-1)


def test_node_creation_with_user_defined_next():
    with pytest.raises(TypeError):
        Node(label="A", next=-1)


def test_node_without_index():
    with pytest.raises(TypeError):
        Node(label="A")


def test_node_without_label():
    with pytest.raises(TypeError):
        Node(index=0)


def test_node_with_negative_index():
    with pytest.raises(ValueError):
        Node(index=-1, label="A")


def test_node_with_empty_label():
    with pytest.raises(ValueError):
        Node(index=0, label="")


def test_node_with_digits_in_label():
    with pytest.raises(ValueError):
        Node(index=0, label="A1")


def test_node_with_non_string_label():
    with pytest.raises(ValueError):
        Node(index=0, label=123)


def test_node_with_non_integer_index():
    with pytest.raises(ValueError):
        Node(index="zero", label="A")
