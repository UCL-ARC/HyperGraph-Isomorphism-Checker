"""Tests for HyperEdge class in hypergraph module."""

from proofChecker_python_serial.hypergraph import HyperEdge, Node


def test_hyperedge_creation():
    node1 = Node(label="A", prev=None, next=0)
    node2 = Node(label="B", prev=None, next=0)
    node3 = Node(label="C", prev=0, next=None)

    edge = HyperEdge(sources=[node1, node2], targets=[node3], label="f")

    assert edge.sources == [node1, node2]
    assert edge.targets == [node3]
    assert edge.label == "f"
    assert edge.signature.sources == [node1, node2]
    assert edge.signature.targets == [node3]
