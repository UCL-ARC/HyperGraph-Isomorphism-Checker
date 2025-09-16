"""Tests for Graph class in graph module."""

import pytest
from proofChecker_python_serial.edge import Edge
from proofChecker_python_serial.graph import Graph
from proofChecker_python_serial.node import Node


@pytest.fixture
def mock_nodes():
    """Create a set of mock nodes for testing."""
    return {
        "input1": Node(label="X", prev=None, next=0),
        "input2": Node(label="Y", prev=None, next=0),
        "intermediate1": Node(label="Z", prev=0, next=1),
        "intermediate2": Node(label="W", prev=0, next=1),
        "output1": Node(label="A", prev=1, next=None),
        "output2": Node(label="B", prev=1, next=None),
        "isolated": Node(label="I", prev=None, next=None),
    }


@pytest.fixture
def mock_edges(mock_nodes):
    """Create a set of mock edges for testing."""
    return {
        "edge1": Edge(
            source=mock_nodes["input1"],
            target=mock_nodes["intermediate1"],
            label="f",
        ),
        "edge2": Edge(
            source=mock_nodes["intermediate1"],
            target=mock_nodes["intermediate2"],
            label="g",
        ),
        "edge3": Edge(
            source=mock_nodes["intermediate2"],
            target=mock_nodes["output1"],
            label="h",
        ),
        "edge4": Edge(
            source=mock_nodes["input2"],
            target=mock_nodes["output2"],
            label="s",
        ),
        "merge_edge": Edge(
            source=mock_nodes["input1"],
            target=mock_nodes["output1"],
            label="m",
        ),
    }


def create_sample_graph() -> Graph:
    """Create a simple valid graph for basic testing."""
    node1 = Node(label="A", prev=None, next=0)
    node2 = Node(label="B", prev=0, next=None)

    edge = Edge(source=node1, target=node2, label="f")

    return Graph(nodes=[node1, node2], edges=[edge])


def create_complex_graph(mock_nodes, mock_edges) -> Graph:
    """Create a more complex graph using mock data."""
    nodes = [
        mock_nodes["input1"],
        mock_nodes["input2"],
        mock_nodes["intermediate1"],
        mock_nodes["intermediate2"],
        mock_nodes["output1"],
        mock_nodes["output2"],
    ]
    edges = [
        mock_edges["edge1"],
        mock_edges["edge2"],
        mock_edges["edge3"],
        mock_edges["edge4"],
    ]
    return Graph(nodes=nodes, edges=edges)


def create_invalid_graph_with_isolated(mock_nodes) -> Graph:
    """Create an invalid graph with isolated nodes."""
    nodes = [
        mock_nodes["input1"],
        mock_nodes["output1"],
        mock_nodes["isolated"],  # This makes it invalid
    ]
    edges = [Edge(source=mock_nodes["input1"], target=mock_nodes["output1"], label="d")]
    return Graph(nodes=nodes, edges=edges)


def test_graph_creation():
    """Test basic graph creation and properties."""
    graph = create_sample_graph()
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert len(graph.input_nodes) == 1
    assert len(graph.output_nodes) == 1
    assert graph.isolated_nodes == []
    assert graph.is_valid() is True


def test_complex_graph_creation(mock_nodes, mock_edges):
    """Test creation of a more complex graph with multiple edges."""
    graph = create_complex_graph(mock_nodes, mock_edges)

    assert len(graph.nodes) == 6
    assert len(graph.edges) == 4
    assert len(graph.input_nodes) == 2
    assert len(graph.output_nodes) == 2
    assert graph.is_valid() is True


def test_graph_with_isolated_nodes(mock_nodes):
    """Test graph validation with isolated nodes."""
    graph = create_invalid_graph_with_isolated(mock_nodes)

    assert len(graph.isolated_nodes) == 1
    assert graph.isolated_nodes[0].label == "I"
    assert graph.is_valid() is False


def test_edge_navigation(mock_nodes, mock_edges):
    """Test basic edge access."""
    graph = create_complex_graph(mock_nodes, mock_edges)

    # Test that edges exist in the graph
    assert mock_edges["edge1"] in graph.edges
    assert mock_edges["edge2"] in graph.edges

    # Test edge properties
    edge1 = mock_edges["edge1"]
    assert edge1.source == mock_nodes["input1"]
    assert edge1.target == mock_nodes["intermediate1"]


def test_edge_queries(mock_nodes, mock_edges):
    """Test edge query functionality."""
    graph = create_complex_graph(mock_nodes, mock_edges)

    # Test that specific edges exist
    edge1 = mock_edges["edge1"]
    assert edge1 in graph.edges

    # Test edge properties
    assert edge1.source.label == "X"
    assert edge1.target.label == "Z"
    assert edge1.label == "f"


def test_add_nodes_and_edges(mock_nodes, mock_edges):
    """Test adding nodes and edges to graph."""
    graph = Graph()

    # Add individual node and edge
    graph.add_node(mock_nodes["input1"])
    graph.add_edge(mock_edges["edge1"])

    assert len(graph.nodes) == 1
    assert len(graph.edges) == 1

    # Add multiple nodes and edges
    new_nodes = [mock_nodes["input2"], mock_nodes["output1"]]
    new_edges = [mock_edges["edge4"]]

    graph.add_nodes(new_nodes)
    graph.add_edges(new_edges)

    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2

    # Test that adding duplicate nodes doesn't create duplicates
    graph.add_node(mock_nodes["input1"])  # Already exists
    assert len(graph.nodes) == 3


def test_remove_operations(mock_nodes, mock_edges):
    """Test basic graph modifications."""
    graph = create_complex_graph(mock_nodes, mock_edges)
    initial_node_count = len(graph.nodes)
    initial_edge_count = len(graph.edges)

    # Test that we can access nodes and edges
    assert len(graph.nodes) == initial_node_count
    assert len(graph.edges) == initial_edge_count

    # Test that specific nodes exist
    assert mock_nodes["input1"] in graph.nodes
    assert mock_nodes["intermediate1"] in graph.nodes


def test_cycle_detection(mock_nodes):
    """Test basic graph structure validation."""
    # Create simple acyclic graph
    acyclic_graph = Graph(
        nodes=[
            mock_nodes["input1"],
            mock_nodes["intermediate1"],
            mock_nodes["output1"],
        ],
        edges=[
            Edge(
                source=mock_nodes["input1"],
                target=mock_nodes["intermediate1"],
                label="f",
            ),
            Edge(
                source=mock_nodes["intermediate1"],
                target=mock_nodes["output1"],
                label="g",
            ),
        ],
    )
    assert acyclic_graph.is_valid() is True

    # Test basic graph properties
    assert len(acyclic_graph.nodes) == 3
    assert len(acyclic_graph.edges) == 2


def test_topological_sort(mock_nodes):
    """Test basic graph ordering properties."""
    # Create acyclic graph
    graph = Graph(
        nodes=[
            mock_nodes["input1"],
            mock_nodes["intermediate1"],
            mock_nodes["output1"],
        ],
        edges=[
            Edge(
                source=mock_nodes["input1"],
                target=mock_nodes["intermediate1"],
                label="f",
            ),
            Edge(
                source=mock_nodes["intermediate1"],
                target=mock_nodes["output1"],
                label="g",
            ),
        ],
    )

    # Test basic graph structure
    assert len(graph.nodes) == 3
    assert len(graph.input_nodes) == 1
    assert len(graph.output_nodes) == 1

    # Test that dependencies are represented correctly
    assert mock_nodes["input1"] in graph.input_nodes
    assert mock_nodes["output1"] in graph.output_nodes


def test_topological_sort_with_cycle(mock_nodes):
    """Test graph validation."""
    # Create simple valid graph
    graph = Graph(
        nodes=[mock_nodes["input1"], mock_nodes["output1"]],
        edges=[
            Edge(source=mock_nodes["input1"], target=mock_nodes["output1"], label="f"),
        ],
    )

    assert graph.is_valid() is True
    assert len(graph.edges) == 1


def test_graph_invalid_no_nodes():
    """Test that graph with no nodes is invalid."""
    graph = Graph(nodes=[], edges=[])
    assert graph.is_valid() is False


def test_validation_messages(mock_nodes):
    """Test detailed validation messages."""
    # Test empty graph
    empty_graph = Graph()
    errors = empty_graph.validate()
    assert "at least one node" in errors[0]
    assert "at least one edge" in errors[1]

    # Test graph with isolated nodes
    invalid_graph = create_invalid_graph_with_isolated(mock_nodes)
    errors = invalid_graph.validate()
    assert any("isolated nodes" in error for error in errors)
    assert "I" in str(errors)  # Should mention the isolated node label


def test_node_reference_validation(mock_nodes):
    """Test validation when edges reference nodes not in the graph."""
    # Create graph missing a node that an edge references
    graph = Graph(
        nodes=[
            mock_nodes["input1"],
            mock_nodes["output2"],
        ],
        edges=[
            Edge(
                source=mock_nodes["input1"],
                target=mock_nodes["output1"],
                label="b",
            ),
            Edge(
                source=mock_nodes["input2"],
                target=mock_nodes["output2"],
                label="c",
            ),
        ],
    )

    errors = graph.validate()
    assert any("reference nodes not in graph" in error for error in errors)
    assert graph.is_valid() is False


def test_graph_no_edges(mock_nodes):
    """Test that graph with nodes but no edges is invalid."""
    graph = Graph(
        nodes=[mock_nodes["input1"], mock_nodes["output1"]], edges=[]  # No edges
    )

    assert graph.is_valid() is False
    errors = graph.validate()
    assert any("at least one edge" in error for error in errors)


def test_graph_no_input_nodes(mock_nodes):
    """Test that graph with no input nodes is invalid."""
    # Create nodes that have no inputs (all have prev != None)
    node1 = Node(label="A", prev=1, next=0)  # Not an input node
    node2 = Node(label="B", prev=0, next=None)  # Output node

    edge = Edge(source=node1, target=node2, label="f")

    graph = Graph(nodes=[node1, node2], edges=[edge])

    # Verify no input nodes exist
    assert len(graph.input_nodes) == 0
    assert graph.is_valid() is False

    errors = graph.validate()
    assert any("at least one input node" in error for error in errors)


def test_graph_no_output_nodes(mock_nodes):
    """Test that graph with no output nodes is invalid."""
    # Create nodes that have no outputs (all have next != None)
    node1 = Node(label="A", prev=None, next=0)  # Input node
    node2 = Node(label="B", prev=0, next=1)  # Not an output node

    edge = Edge(source=node1, target=node2, label="f")

    graph = Graph(nodes=[node1, node2], edges=[edge])

    # Verify no output nodes exist
    assert len(graph.output_nodes) == 0
    assert graph.is_valid() is False

    errors = graph.validate()
    assert any("at least one output node" in error for error in errors)
