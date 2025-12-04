"""Tests for Diagram class in diagram module."""

from pathlib import Path
import pytest
from IsomorphismChecker_python_serial.hypergraph import OpenHypergraph
from IsomorphismChecker_python_serial.hyperedge import HyperEdge
from IsomorphismChecker_python_serial.node import Node
from IsomorphismChecker_python_serial.diagram import Diagram


@pytest.fixture
def sample_hypergraph():
    """Create a simple valid hypergraph for testing."""
    node1 = Node(index=0, label="A")
    node2 = Node(index=1, label="B")
    node3 = Node(index=2, label="C")

    edge = HyperEdge(sources=[0, 1], targets=[2], label="f", index=0)

    return OpenHypergraph(nodes=[node1, node2, node3], edges=[edge])


@pytest.fixture
def invalid_hypergraph():
    """Create an invalid hypergraph for testing."""
    node1 = Node(index=0, label="A")
    node2 = Node(index=1, label="B")
    node3 = Node(index=2, label="C")
    # node4 = Node(index=3, label="D")

    edge = HyperEdge(sources=[0, 1], targets=[2, 3], label="f", index=0)
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
        Diagram.diagram_label("f", 0, "non-sensical")  # type: ignore


def test_draw_invalid_hypergraph():
    """Test that drawing an invalid hypergraph raises an error."""

    invalid_hypergraph = OpenHypergraph()
    with pytest.raises(ValueError, match="The provided OpenHypergraph is not valid."):
        Diagram(openHyperGraph=invalid_hypergraph)

    assert False
