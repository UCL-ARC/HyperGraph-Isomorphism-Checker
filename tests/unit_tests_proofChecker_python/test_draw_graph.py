import pytest
from pathlib import Path
from IsomorphismChecker_python_serial.hypergraph import OpenHypergraph
from IsomorphismChecker_python_serial.node import Node
from IsomorphismChecker_python_serial.hyperedge import HyperEdge
from IsomorphismChecker_python_serial.draw import draw_graph


@pytest.fixture
def sample_hypergraph() -> OpenHypergraph:
    """Create a simple valid hypergraph for testing."""
    node1 = Node(index=0, label="A")
    node2 = Node(index=1, label="B")
    node3 = Node(index=2, label="C")

    edge = HyperEdge(sources=[0, 1], targets=[2], label="f", index=0)

    return OpenHypergraph(nodes=[node1, node2, node3], edges=[edge])


def test_draw_graph(sample_hypergraph: OpenHypergraph, tmp_path: Path):
    """Test drawing the graph and saving to a file."""
    fname = tmp_path / "test_graph"
    draw_graph(sample_hypergraph, str(fname))

    fpath = Path(fname.with_suffix(".png"))
    assert fpath.exists()
    assert fpath.stat().st_size > 0


def test_draw_graph_verbose(
    sample_hypergraph: OpenHypergraph, capsys: pytest.CaptureFixture
):
    """Test drawing the graph with verbose output."""
    draw_graph(sample_hypergraph, verbose=True)

    captured = capsys.readouterr()
    assert "Diagram source:" in captured.out
    assert "digraph" in captured.out


def test_print_graph(sample_hypergraph: OpenHypergraph, capsys: pytest.CaptureFixture):
    """Test the print_graph function output."""
    from IsomorphismChecker_python_serial.draw import print_graph

    print_graph(sample_hypergraph)

    captured = capsys.readouterr()
    assert "(0, 'A')" in captured.out
    assert "(1, 'B')" in captured.out
    assert "(2, 'C')" in captured.out
    assert "('f', [0, 1], [2])" in captured.out
