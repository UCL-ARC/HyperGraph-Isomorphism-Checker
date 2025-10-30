"""Tests for node class in hypergraph module."""

import pytest
from IsomorphismChecker_python_serial.hypergraph import Node


def test_node_creation():
    node = Node(index=0, label="a")
    assert node.label == "a"
    assert node.prev == []
    assert node.next == []


def test_node_creation_with_user_defined_prev():
    with pytest.raises(TypeError):
        Node(label="A", prev=-1)


def test_node_creation_with_user_defined_next():
    with pytest.raises(TypeError):
        Node(label="A", next=-1)


def test_node_without_index():
    with pytest.raises(TypeError):
        Node(label="A")
