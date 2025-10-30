"""Tests for OpenHypergraph class in hypergraph module."""

import pytest
from IsomorphismChecker_python_serial.hyperedge import HyperEdge
from IsomorphismChecker_python_serial.node import Node
from IsomorphismChecker_python_serial.hypergraph import OpenHypergraph


@pytest.fixture
def mock_nodes() -> dict[str, Node]:
    """Create a set of mock nodes for testing."""
    return {
        "input1": Node(index=0, label="X"),
        "input2": Node(index=1, label="Y"),
        "intermediate1": Node(index=2, label="Z"),
        "intermediate2": Node(index=3, label="W"),
        "output1": Node(index=4, label="A"),
        "output2": Node(index=5, label="B"),
        "isolated": Node(index=6, label="I"),
    }


@pytest.fixture
def mock_edges():
    """Create a set of mock edges for testing."""
    return {
        "edge1": HyperEdge(
            sources=[0, 1],
            targets=[2],
            label="f",
            index=0,
        ),
        "edge2": HyperEdge(
            sources=[2],
            targets=[3],
            label="g",
            index=1,
        ),
        "edge3": HyperEdge(
            sources=[3],
            targets=[4, 5],
            label="h",
            index=2,
        ),
        # "split_edge": HyperEdge(
        #     sources=[mock_nodes["input1"]],
        #     targets=[mock_nodes["output1"], mock_nodes["output2"]],
        #     label="s",
        # ),
        # "merge_edge": HyperEdge(
        #     sources=[mock_nodes["input1"], mock_nodes["input2"]],
        #     targets=[mock_nodes["output1"]],
        #     label="m",
        # ),
    }


def create_sample_hypergraph() -> OpenHypergraph:
    """Create a simple valid hypergraph for basic testing."""
    node1 = Node(index=0, label="A")
    node2 = Node(index=1, label="B")
    node3 = Node(index=2, label="C")

    edge = HyperEdge(sources=[0, 1], targets=[2], label="f", index=0)

    return OpenHypergraph(nodes=[node1, node2, node3], edges=[edge])


def create_complex_hypergraph(
    mock_nodes: dict[str, Node], mock_edges: dict[str, HyperEdge]
) -> OpenHypergraph:
    """Create a more complex hypergraph using mock data."""
    nodes = [
        mock_nodes["input1"],
        mock_nodes["input2"],
        mock_nodes["intermediate1"],
        mock_nodes["intermediate2"],
        mock_nodes["output1"],
        mock_nodes["output2"],
    ]
    edges = [mock_edges["edge1"], mock_edges["edge2"], mock_edges["edge3"]]
    return OpenHypergraph(nodes=nodes, edges=edges)


def create_invalid_hypergraph_with_isolated(
    mock_nodes: dict[str, Node],
) -> OpenHypergraph:
    """Create an invalid hypergraph with isolated nodes."""
    nodes = [
        mock_nodes["input1"],
        mock_nodes["output1"],
        mock_nodes["isolated"],  # This makes it invalid
    ]
    edges = [
        HyperEdge(
            sources=[0],
            targets=[1],
            label="d",
            index=0,
        )
    ]
    return OpenHypergraph(nodes=nodes, edges=edges)


def test_open_hypergraph_creation():
    """Test basic hypergraph creation and properties."""
    hypergraph = create_sample_hypergraph()
    assert len(hypergraph.nodes) == 3
    assert len(hypergraph.edges) == 1
    assert hypergraph.is_valid() is True


def test_complex_hypergraph_creation(
    mock_nodes: dict[str, Node], mock_edges: dict[str, HyperEdge]
):
    """Test creation of a more complex hypergraph with multiple edges."""
    hypergraph = create_complex_hypergraph(mock_nodes, mock_edges)

    assert len(hypergraph.nodes) == 6
    assert len(hypergraph.edges) == 3
    assert hypergraph.is_valid() is True


def test_empty_hypergraph():
    """Test that an empty hypergraph is invalid."""
    hypergraph = OpenHypergraph()
    assert hypergraph.is_valid() is False


def test_hypergraph_no_nodes(mock_edges: dict[str, HyperEdge]):
    """Test hypergraph validation with no nodes."""
    with pytest.raises(ValueError, match="Edge f has nodes not in hypergraph nodes"):
        OpenHypergraph(nodes=[], edges=[mock_edges["edge1"]])


def test_hypergraph_no_edges(mock_nodes: dict[str, Node]):
    """Test hypergraph validation with no edges."""
    hypergraph = OpenHypergraph(
        nodes=[mock_nodes["input1"], mock_nodes["output1"]], edges=[]
    )
    assert hypergraph.is_valid() is False


def test_hypergraph_with_isolated_nodes(mock_nodes):
    """Test hypergraph validation with isolated nodes."""
    hypergraph = create_invalid_hypergraph_with_isolated(mock_nodes)
    assert len(hypergraph.nodes) == 3
    assert len(hypergraph.edges) == 1


def test_hypergraph_multiple_sources_targets():
    """Test hypergraph with edges having multiple sources and targets."""
    node1 = Node(index=0, label="A")
    node2 = Node(index=1, label="B")
    node3 = Node(index=2, label="C")
    node4 = Node(index=3, label="D")

    edge1 = HyperEdge(sources=[0, 1], targets=[2, 3], label="f", index=0)
    edge2 = HyperEdge(sources=[1], targets=[3], label="g", index=1)

    with pytest.warns(UserWarning):
        hypergraph = OpenHypergraph(
            nodes=[node1, node2, node3, node4], edges=[edge1, edge2]
        )

    assert hypergraph.is_valid() is True
    assert hypergraph.edges[0].sources == [0, 1]
    assert hypergraph.edges[0].targets == [2, 3]
