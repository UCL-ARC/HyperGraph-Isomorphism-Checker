"""Tests for Diagram class in diagram module."""

from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
from proofChecker_python_serial.hypergraph import OpenHypergraph
from proofChecker_python_serial.hyperedge import HyperEdge
from proofChecker_python_serial.node import Node
from proofChecker_python_serial.diagram import Diagram, draw_graph


@pytest.fixture
def sample_hypergraph():
    """Create a simple valid hypergraph for testing."""
    node1 = Node(label="A", index=1)
    node2 = Node(label="B", index=2)
    node3 = Node(label="C", index=3)

    print(node1, node2, node3)

    edge = HyperEdge(sources=[node1, node2], targets=[node3], label="f", index=0)

    return OpenHypergraph(nodes=[node1, node2, node3], edges=[edge])


def test_diagram_creation(sample_hypergraph: OpenHypergraph):
    """Test creating a Diagram from a valid hypergraph."""
    diagram = Diagram(openHyperGraph=sample_hypergraph)
    assert len(diagram.nodes) == 3
    assert len(diagram.hyperEdges) == 1


def test_diagram_render(sample_hypergraph: OpenHypergraph, tmp_path: Path):
    """Test rendering the diagram and getting its source."""
    diagram = Diagram(openHyperGraph=sample_hypergraph)
    output_path = tmp_path / "test_diagram"
    diagram.render(str(output_path))

    # Check if the file was created
    assert (output_path.with_suffix(".png")).exists()


def test_diagram_source(sample_hypergraph: OpenHypergraph):
    """Test getting the source of the diagram."""
    diagram = Diagram(openHyperGraph=sample_hypergraph)
    source = diagram.source()

    assert "digraph" in source
    assert "A" in source
    assert "B" in source
    assert "C" in source
    assert "f" in source


def test_diagram_label():
    """Test the label property of the Diagram."""

    with pytest.raises(
        ValueError, match="Type must be ElementType.NODE or ElementType.EDGE."
    ):
        Diagram.diagram_label("f", 0, "non-sensical")


# Tests for draw_graph function
def test_draw_graph_basic_functionality(
    sample_hypergraph: OpenHypergraph, tmp_path: Path
):
    """Test that draw_graph creates a diagram and renders it."""
    # Change to temporary directory to avoid cluttering the project
    original_cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)

        # Capture print output
        with patch("builtins.print") as mock_print:
            draw_graph(sample_hypergraph)

        # Check that print was called with expected content
        assert mock_print.call_count >= 2  # At least "Diagram source:" and the source

        # Check that the first call was the header
        first_call_args = mock_print.call_args_list[0][0]
        assert "Diagram source:" in first_call_args[0]

        # Check that subsequent calls contain diagram source
        source_calls = [call[0][0] for call in mock_print.call_args_list[1:]]
        source_content = "\n".join(source_calls)
        assert "digraph" in source_content

        # Check that the PNG file was created
        png_file = tmp_path / "hypergraph_diagram.png"
        assert png_file.exists()

    finally:
        os.chdir(original_cwd)


def test_draw_graph_with_complex_hypergraph(tmp_path: Path):
    """Test draw_graph with a more complex hypergraph."""
    # Create a more complex hypergraph
    nodes = [
        Node(label="X", index=0),
        Node(label="Y", index=1),
        Node(label="Z", index=2),
        Node(label="W", index=3),
    ]

    edges = [
        HyperEdge(sources=[nodes[0]], targets=[nodes[1]], label="F", index=0),
        HyperEdge(sources=[nodes[1], nodes[2]], targets=[nodes[3]], label="G", index=1),
    ]

    complex_hypergraph = OpenHypergraph(nodes=nodes, edges=edges)

    original_cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)

        with patch("builtins.print") as mock_print:
            draw_graph(complex_hypergraph)

        # Verify all nodes and edges appear in the printed source
        source_calls = [call[0][0] for call in mock_print.call_args_list[1:]]
        source_content = "\n".join(source_calls)

        # Check for all node labels
        for node in nodes:
            assert node.label in source_content

        # Check for all edge labels
        for edge in edges:
            assert edge.label in source_content

        # Check that the file was created
        png_file = tmp_path / "hypergraph_diagram.png"
        assert png_file.exists()

    finally:
        os.chdir(original_cwd)


def test_draw_graph_prints_correct_format(
    sample_hypergraph: OpenHypergraph, tmp_path: Path
):
    """Test that draw_graph prints the source in the correct format."""
    original_cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)

        with patch("builtins.print") as mock_print:
            draw_graph(sample_hypergraph)

        # Get all printed content
        printed_lines = [call[0][0] for call in mock_print.call_args_list]

        # Check that it starts with the header
        assert printed_lines[0] == "Diagram source:"

        # Check that the source contains graphviz syntax
        source_content = "\n".join(printed_lines[1:])
        assert source_content.strip().startswith("digraph")
        assert "shape=circle" in source_content  # Node shape
        assert "shape=box" in source_content  # Edge shape

    finally:
        os.chdir(original_cwd)


def test_draw_graph_file_creation(sample_hypergraph: OpenHypergraph, tmp_path: Path):
    """Test that draw_graph creates the expected file."""
    original_cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)

        # Mock print to suppress output during test
        with patch("builtins.print"):
            draw_graph(sample_hypergraph)

        # Check that the file exists and has content
        png_file = tmp_path / "hypergraph_diagram.png"
        assert png_file.exists()
        assert png_file.stat().st_size > 0  # File is not empty

    finally:
        os.chdir(original_cwd)


@pytest.fixture
def empty_hypergraph():
    """Create an empty hypergraph for edge case testing."""
    return OpenHypergraph(nodes=[], edges=[])


def test_draw_graph_with_empty_hypergraph(
    empty_hypergraph: OpenHypergraph, tmp_path: Path
):
    """Test draw_graph behavior with an empty hypergraph."""
    original_cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)

        with patch("builtins.print") as mock_print:
            draw_graph(empty_hypergraph)

        # Should still print header and source (even if minimal)
        assert mock_print.call_count >= 1

        # Check that the file was still created
        png_file = tmp_path / "hypergraph_diagram.png"
        assert png_file.exists()

    finally:
        os.chdir(original_cwd)


def test_draw_graph_creates_diagram_object(
    sample_hypergraph: OpenHypergraph, tmp_path: Path
):
    """Test that draw_graph properly creates and uses a Diagram object internally."""
    original_cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)

        # Mock the Diagram class to verify it's being used correctly
        with patch("proofChecker_python_serial.diagram.Diagram") as mock_diagram_class:
            mock_diagram_instance = MagicMock()
            mock_diagram_class.return_value = mock_diagram_instance
            mock_diagram_instance.source.return_value = "digraph test { }"

            with patch("builtins.print"):
                draw_graph(sample_hypergraph)

            # Verify Diagram was instantiated with the hypergraph
            mock_diagram_class.assert_called_once_with(openHyperGraph=sample_hypergraph)

            # Verify render and source methods were called
            mock_diagram_instance.render.assert_called_once_with("hypergraph_diagram")
            mock_diagram_instance.source.assert_called_once()

    finally:
        os.chdir(original_cwd)
