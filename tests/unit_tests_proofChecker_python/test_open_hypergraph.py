"""Tests for OpenHypergraph class in hypergraph module."""

import pytest
from proofChecker_python_serial.hyperedge import HyperEdge
from proofChecker_python_serial.node import Node
from proofChecker_python_serial.hypergraph import OpenHypergraph


@pytest.fixture
def mock_nodes() -> dict[str, Node]:
    """Create a set of mock nodes for testing."""
    return {
        "input1": Node(label="X", index=0),
        "input2": Node(label="Y", index=1),
        "intermediate1": Node(label="Z", index=2),
        "intermediate2": Node(label="W", index=3),
        "output1": Node(label="A", index=4),
        "output2": Node(label="B", index=5),
        "isolated": Node(label="I", index=6),
    }


@pytest.fixture
def mock_edges(mock_nodes: dict[str, Node]):
    """Create a set of mock edges for testing."""
    return {
        "edge1": HyperEdge(
            sources=[mock_nodes["input1"], mock_nodes["input2"]],
            targets=[mock_nodes["intermediate1"]],
            label="f",
            index=0,
        ),
        "edge2": HyperEdge(
            sources=[mock_nodes["intermediate1"]],
            targets=[mock_nodes["intermediate2"]],
            label="g",
            index=1,
        ),
        "edge3": HyperEdge(
            sources=[mock_nodes["intermediate2"]],
            targets=[mock_nodes["output1"], mock_nodes["output2"]],
            label="h",
            index=2,
        ),
        "split_edge": HyperEdge(
            sources=[mock_nodes["input1"]],
            targets=[mock_nodes["output1"], mock_nodes["output2"]],
            label="s",
            index=3,
        ),
        "merge_edge": HyperEdge(
            sources=[mock_nodes["input1"], mock_nodes["input2"]],
            targets=[mock_nodes["output1"]],
            label="m",
            index=4,
        ),
    }


def create_sample_hypergraph() -> OpenHypergraph:
    """Create a simple valid hypergraph for basic testing."""
    node1 = Node(label="A", index=0)
    node2 = Node(label="B", index=1)
    node3 = Node(label="C", index=2)

    edge = HyperEdge(sources=[node1, node2], targets=[node3], label="f", index=0)

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
            sources=[mock_nodes["input1"]],
            targets=[mock_nodes["output1"]],
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
    assert hypergraph.input_nodes == [hypergraph.nodes[0], hypergraph.nodes[1]]
    assert hypergraph.output_nodes == [hypergraph.nodes[2]]
    assert hypergraph.isolated_nodes == []
    assert hypergraph.is_valid() is True


def test_complex_hypergraph_creation(
    mock_nodes: dict[str, Node], mock_edges: dict[str, HyperEdge]
):
    """Test creation of a more complex hypergraph with multiple edges."""
    hypergraph = create_complex_hypergraph(mock_nodes, mock_edges)

    assert len(hypergraph.nodes) == 6
    assert len(hypergraph.edges) == 3
    assert len(hypergraph.input_nodes) == 2
    assert len(hypergraph.output_nodes) == 2
    assert hypergraph.is_valid() is True


def test_empty_hypergraph():
    """Test that an empty hypergraph is invalid."""
    hypergraph = OpenHypergraph()
    assert hypergraph.is_valid() is False


def test_hypergraph_no_nodes(mock_edges: dict[str, HyperEdge]):
    """Test hypergraph validation with no nodes."""

    with pytest.raises(ValueError):
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

    assert len(hypergraph.isolated_nodes) == 1
    assert hypergraph.isolated_nodes[0].label == "I"
    assert hypergraph.is_valid() is False


def test_edge_signatures(mock_edges: dict[str, HyperEdge]):
    """Test that edges have correct signatures."""
    edge1 = mock_edges["edge1"]
    signature = edge1.signature
    assert isinstance(signature, int)


def test_split_edge_pattern(mock_edges: dict[str, HyperEdge]):
    """Test a split pattern (1 input -> 2 outputs)."""
    split_edge = mock_edges["split_edge"]

    assert len(split_edge.sources) == 1
    assert len(split_edge.targets) == 2
    assert split_edge.label == "s"


def test_merge_edge_pattern(mock_edges: dict[str, HyperEdge]):
    """Test a merge pattern (2 inputs -> 1 output)."""
    merge_edge = mock_edges["merge_edge"]

    assert len(merge_edge.sources) == 2
    assert len(merge_edge.targets) == 1
    assert merge_edge.label == "m"


def test_add_nodes_and_edges(
    mock_nodes: dict[str, Node], mock_edges: dict[str, HyperEdge]
):
    """Test adding nodes and edges to hypergraph."""
    hypergraph = OpenHypergraph()

    # Add individual node and edge
    hypergraph.add_node(mock_nodes["input1"])
    hypergraph.add_edge(mock_edges["split_edge"])

    assert len(hypergraph.nodes) == 1
    assert len(hypergraph.edges) == 1

    # Add multiple nodes and edges
    new_nodes = [mock_nodes["input2"], mock_nodes["output1"]]
    new_edges = [mock_edges["merge_edge"]]

    hypergraph.add_nodes(new_nodes)
    hypergraph.add_edges(new_edges)

    assert len(hypergraph.nodes) == 3
    assert len(hypergraph.edges) == 2


def test_validation_messages(mock_nodes: dict[str, Node]):
    """Test detailed validation messages."""
    # Test empty hypergraph
    empty_hypergraph = OpenHypergraph()
    errors = empty_hypergraph.validate()
    assert "at least one node" in errors[0]
    assert "at least one edge" in errors[1]

    # Test hypergraph with isolated nodes
    invalid_hypergraph = create_invalid_hypergraph_with_isolated(mock_nodes)
    errors = invalid_hypergraph.validate()
    assert any("isolated nodes" in error for error in errors)
    assert "I" in str(errors)  # Should mention the isolated node label


def test_node_reference_validation(mock_nodes: dict[str, Node]):
    """Test validation when edges reference nodes not in the hypergraph."""
    # Create hypergraph missing a node that an edge references

    with pytest.raises(ValueError):
        OpenHypergraph(
            nodes=[mock_nodes["input1"]],  # Missing output1 that edge references
            edges=[
                HyperEdge(
                    sources=[mock_nodes["input1"]],
                    targets=[mock_nodes["output1"]],
                    label="b",
                    index=0,
                )
            ],
        )


def test_open_hypergraph_invalid_no_inputs(mock_nodes: dict[str, Node]):
    """Test that hypergraph with no input nodes is invalid."""

    hypergraph = OpenHypergraph(
        nodes=[mock_nodes["output2"]],
        edges=[
            HyperEdge(
                sources=[mock_nodes["output2"]],
                targets=[mock_nodes["output2"]],
                label="d",
                index=0,
            )
        ],
    )
    assert hypergraph.is_valid() is False


def test_open_hypergraph_invalid_no_outputs(mock_nodes: dict[str, Node]):
    """Test that hypergraph with no output nodes is invalid."""
    hypergraph = OpenHypergraph(
        nodes=[mock_nodes["input1"]],
        edges=[
            HyperEdge(
                sources=[mock_nodes["input1"]],
                targets=[mock_nodes["input1"]],
                label="d",
                index=0,
            )
        ],
    )
    assert hypergraph.is_valid() is False


def test_open_hypergraph_invalid_absurd_node(mock_nodes: dict[str, Node]):
    """Test that hypergraph with connections to a non-existent node is invalid."""
    absurd_node = Node(label="Absurd", index=2)  # Not in the hypergraph node list

    with pytest.raises(ValueError):
        OpenHypergraph(
            nodes=[mock_nodes["input1"], mock_nodes["output1"]],
            edges=[
                HyperEdge(
                    sources=[mock_nodes["input1"]],
                    targets=[absurd_node],
                    label="d",
                    index=0,
                )
            ],
        )
