"""Tests for node class in hypergraph module."""

import pytest
from proof_checker.hypergraph import Node


def test_node_creation():
    node = Node(label="A", prev=None, next=None)
    assert node.label == "A"
    assert node.prev is None
    assert node.next is None


def test_node_creation_with_invalid_prev():
    with pytest.raises(
        ValueError, match="Previous node index must be non-negative or None."
    ):
        Node(label="A", prev=-1, next=None)


def test_node_creation_with_invalid_next():
    with pytest.raises(
        ValueError, match="Next node index must be non-negative or None."
    ):
        Node(label="A", prev=None, next=-1)
