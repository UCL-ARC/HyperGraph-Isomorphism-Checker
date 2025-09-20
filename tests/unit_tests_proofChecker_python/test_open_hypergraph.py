"""Tests for OpenHypergraph class in hypergraph module."""

import pytest
import json
import tempfile
import os
from proofChecker_python_serial.hyperedge import HyperEdge
from proofChecker_python_serial.node import Node
from proofChecker_python_serial.hypergraph import OpenHypergraph, create_hypergraph


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
    # assert hypergraph.is_valid() is True
    assert hypergraph.validate() == []


def test_empty_hypergraph():
    """Test that an empty hypergraph is invalid."""
    hypergraph = OpenHypergraph()
    assert hypergraph.is_valid() is False


def test_hypergraph_no_nodes(mock_edges: dict[str, HyperEdge]):
    """Test hypergraph validation with no nodes."""

    hypergraph = OpenHypergraph(nodes=[], edges=[mock_edges["edge1"]])
    assert hypergraph.is_valid() is False


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

    hypergraph = OpenHypergraph(
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
    assert hypergraph.is_valid() is False


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

    hypergraph = OpenHypergraph(
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

    assert hypergraph.is_valid() is False


def test_invalid_branch_creation(mock_nodes: dict[str, Node]):
    """Test hypergraph with invalid branching (node used as both input and output)."""

    node_a = mock_nodes["input1"]
    node_b = mock_nodes["output1"]
    node_c = mock_nodes["output2"]

    edge1 = HyperEdge(sources=[node_a], targets=[node_b], label="f", index=0)
    edge2 = HyperEdge(sources=[node_a], targets=[node_c], label="g", index=1)

    with pytest.raises(ValueError):
        OpenHypergraph(nodes=[node_a, node_b, node_c], edges=[edge1, edge2])


def test_invalid_branch_merge(mock_nodes: dict[str, Node]):
    """Test hypergraph with invalid merging (node used as both input and output)."""

    node_a = mock_nodes["input1"]
    node_b = mock_nodes["input2"]
    node_c = mock_nodes["output2"]

    edge1 = HyperEdge(sources=[node_a], targets=[node_c], label="f", index=0)
    edge2 = HyperEdge(sources=[node_b], targets=[node_c], label="g", index=1)

    with pytest.raises(ValueError):
        OpenHypergraph(nodes=[node_a, node_b, node_c], edges=[edge1, edge2])


def test_invalid_graph_with_unknown_node(mock_nodes: dict[str, Node]):
    """Test hypergraph with an edge referencing a node not in the node list."""

    node_a = mock_nodes["input1"]
    node_b = mock_nodes["output1"]
    unknown_node = Node(label="Unknown", index=99)  # Not in the hypergraph node list

    edge = HyperEdge(
        sources=[node_a], targets=[node_b, unknown_node], label="f", index=0
    )

    hypergraph = OpenHypergraph(nodes=[node_a, node_b], edges=[edge])
    assert hypergraph.is_valid() is False
    assert len(hypergraph.validate()) > 0


# Tests for create_hypergraph function
@pytest.fixture
def sample_hypergraph_json():
    """Create sample JSON data for hypergraph creation testing."""
    return {
        "graph_name": "TestGraph",
        "comment": ["Test hypergraph for unit testing"],
        "nodes": [
            {"type_label": "a"},
            {"type_label": "b"},
            {"type_label": "c"},
            {"type_label": "d"},
        ],
        "hyperedges": [
            {"type_label": "F", "source_nodes": [0, 1], "target_nodes": [2]},
            {"type_label": "G", "source_nodes": [2], "target_nodes": [3]},
        ],
    }


@pytest.fixture
def temp_json_file(sample_hypergraph_json):
    """Create a temporary JSON file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_hypergraph_json, f)
        temp_file_path = f.name

    yield temp_file_path

    # Cleanup
    os.unlink(temp_file_path)


def test_create_hypergraph_basic(temp_json_file):
    """Test basic hypergraph creation from JSON file."""
    hypergraph = create_hypergraph(temp_json_file)

    # Test nodes creation
    assert len(hypergraph.nodes) == 4
    assert hypergraph.nodes[0].label == "a"
    assert hypergraph.nodes[0].index == 0
    assert hypergraph.nodes[1].label == "b"
    assert hypergraph.nodes[1].index == 1
    assert hypergraph.nodes[2].label == "c"
    assert hypergraph.nodes[2].index == 2
    assert hypergraph.nodes[3].label == "d"
    assert hypergraph.nodes[3].index == 3

    # Test edges creation
    assert len(hypergraph.edges) == 2

    edge1 = hypergraph.edges[0]
    assert edge1.label == "F"
    assert edge1.index == 0
    assert len(edge1.sources) == 2
    assert len(edge1.targets) == 1
    assert edge1.sources[0] == hypergraph.nodes[0]  # node "a"
    assert edge1.sources[1] == hypergraph.nodes[1]  # node "b"
    assert edge1.targets[0] == hypergraph.nodes[2]  # node "c"

    edge2 = hypergraph.edges[1]
    assert edge2.label == "G"
    assert edge2.index == 1
    assert len(edge2.sources) == 1
    assert len(edge2.targets) == 1
    assert edge2.sources[0] == hypergraph.nodes[2]  # node "c"
    assert edge2.targets[0] == hypergraph.nodes[3]  # node "d"


def test_create_hypergraph_complex():
    """Test hypergraph creation with more complex structure."""
    complex_json = {
        "graph_name": "ComplexGraph",
        "nodes": [
            {"type_label": "x"},
            {"type_label": "y"},
            {"type_label": "z"},
            {"type_label": "w"},
            {"type_label": "u"},
            {"type_label": "v"},
        ],
        "hyperedges": [
            {"type_label": "A", "source_nodes": [0], "target_nodes": [2, 3]},
            {"type_label": "B", "source_nodes": [1], "target_nodes": [4]},
            {"type_label": "C", "source_nodes": [2, 3, 4], "target_nodes": [5]},
        ],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(complex_json, f)
        temp_file_path = f.name

    try:
        hypergraph = create_hypergraph(temp_file_path)

        # Test basic structure
        assert len(hypergraph.nodes) == 6
        assert len(hypergraph.edges) == 3

        # Test edge A: 1 source -> 2 targets
        edge_a = hypergraph.edges[0]
        assert edge_a.label == "A"
        assert len(edge_a.sources) == 1
        assert len(edge_a.targets) == 2

        # Test edge C: 3 sources -> 1 target
        edge_c = hypergraph.edges[2]
        assert edge_c.label == "C"
        assert len(edge_c.sources) == 3
        assert len(edge_c.targets) == 1

    finally:
        os.unlink(temp_file_path)


def test_create_hypergraph_empty_file():
    """Test error handling for malformed JSON."""
    empty_json = {"graph_name": "EmptyGraph", "nodes": [], "hyperedges": []}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(empty_json, f)
        temp_file_path = f.name

    try:
        hypergraph = create_hypergraph(temp_file_path)
        assert len(hypergraph.nodes) == 0
        assert len(hypergraph.edges) == 0
        assert hypergraph.is_valid() is False  # Empty hypergraph is invalid

    finally:
        os.unlink(temp_file_path)


def test_create_hypergraph_invalid_file():
    """Test error handling for non-existent file."""
    with pytest.raises(FileNotFoundError):
        create_hypergraph("non_existent_file.json")


def test_create_hypergraph_malformed_json():
    """Test error handling for malformed JSON."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("{ invalid json }")
        temp_file_path = f.name

    try:
        with pytest.raises(json.JSONDecodeError):
            create_hypergraph(temp_file_path)
    finally:
        os.unlink(temp_file_path)


def test_create_hypergraph_missing_fields():
    """Test error handling for JSON missing required fields."""
    incomplete_json = {
        "graph_name": "IncompleteGraph",
        "nodes": [{"type_label": "a"}],
        # Missing "hyperedges" field
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(incomplete_json, f)
        temp_file_path = f.name

    try:
        with pytest.raises(KeyError):
            create_hypergraph(temp_file_path)
    finally:
        os.unlink(temp_file_path)


def test_create_hypergraph_invalid_node_indices():
    """Test error handling for invalid node indices in edges."""
    invalid_json = {
        "graph_name": "InvalidGraph",
        "nodes": [{"type_label": "a"}, {"type_label": "b"}],
        "hyperedges": [
            {
                "type_label": "F",
                "source_nodes": [0],
                "target_nodes": [5],
            }  # Index 5 doesn't exist
        ],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(invalid_json, f)
        temp_file_path = f.name

    try:
        with pytest.raises(IndexError):
            create_hypergraph(temp_file_path)
    finally:
        os.unlink(temp_file_path)


def test_create_hypergraph_signature_generation(temp_json_file):
    """Test that hypergraph signature is properly generated during creation."""
    hypergraph = create_hypergraph(temp_json_file)

    # The signature should be generated in __post_init__
    assert hypergraph.signature != ""
    assert isinstance(hypergraph.signature, str)

    # Should contain edge information
    assert "F" in hypergraph.signature  # First edge label
    assert "G" in hypergraph.signature  # Second edge label
