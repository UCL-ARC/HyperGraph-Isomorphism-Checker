"""Tests for HyperEdge class in hypergraph module."""

from proofChecker_python_serial.hypergraph import HyperEdge, Node


def test_hyperedge_creation():
    node1 = Node(index=0, label="a")
    node2 = Node(index=1, label="b")
    node3 = Node(index=2, label="c")

    edge = HyperEdge(sources=[node1, node2], targets=[node3], label="F", index=0)

    assert edge.sources == [node1, node2]
    assert edge.targets == [node3]
    assert edge.label == "F"
    assert edge.display_label == "F, 0 (a,b)->(c)"
