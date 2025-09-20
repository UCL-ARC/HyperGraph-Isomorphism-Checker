"""Tests for HyperEdge class in hypergraph module."""

import pytest
from proofChecker_python_serial.hypergraph import HyperEdge, Node


@pytest.fixture
def mock_nodes() -> list[Node]:
    """Create a list of mock nodes for testing."""
    return [
        Node(index=0, label="a"),
        Node(index=1, label="b"),
        Node(index=2, label="c"),
    ]


@pytest.fixture
def mock_hyperedge(mock_nodes: list[Node]) -> HyperEdge:
    """Create a mock hyperedge for testing."""
    return HyperEdge(
        sources=mock_nodes[:2], targets=[mock_nodes[2]], label="F", index=0
    )


def test_hyperedge_creation(mock_hyperedge):
    edge = mock_hyperedge
    assert edge is not None


def test_hyperedge_sources(mock_hyperedge: HyperEdge, mock_nodes: list[Node]):
    assert mock_hyperedge.sources == [mock_nodes[0], mock_nodes[1]]


def test_hyperedge_targets(mock_hyperedge: HyperEdge, mock_nodes: list[Node]):
    assert mock_hyperedge.targets == [mock_nodes[2]]


def test_hyperedge_label(mock_hyperedge: HyperEdge):
    assert mock_hyperedge.label == "F"


def test_hyperedge_index(mock_hyperedge: HyperEdge):
    assert mock_hyperedge.display_label == "F, 0 (a,b)->(c)"
