"""Tests for node class in hypergraph module."""

from IsomorphismChecker_python_serial.hypergraph import Node


def test_node_creation():
    node = Node(index=0, label="a")
    assert node.label == "a"
    assert node.prev == []
    assert node.next == []
