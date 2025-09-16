"""Tests for node class in hypergraph module."""

import pytest
from proofChecker_python_serial.hypergraph import Node


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


def test_is_isolated():
    node = Node(label="A", prev=None, next=None)
    assert node.is_isolated is True
    node2 = Node(label="B", prev=0, next=None)
    assert node2.is_isolated is False
    node3 = Node(label="C", prev=None, next=1)
    assert node3.is_isolated is False
    node4 = Node(label="D", prev=0, next=1)
    assert node4.is_isolated is False


def test_is_input():
    node = Node(label="A", prev=None, next=1)
    assert node.is_input is True
    node2 = Node(label="B", prev=0, next=None)
    assert node2.is_input is False
    node3 = Node(label="C", prev=None, next=None)
    assert node3.is_input is False
    node4 = Node(label="D", prev=0, next=1)
    assert node4.is_input is False


def test_is_output():
    node = Node(label="A", prev=0, next=None)
    assert node.is_output is True
    node2 = Node(label="B", prev=None, next=1)
    assert node2.is_output is False
    node3 = Node(label="C", prev=None, next=None)
    assert node3.is_output is False
    node4 = Node(label="D", prev=0, next=1)
    assert node4.is_output is False
