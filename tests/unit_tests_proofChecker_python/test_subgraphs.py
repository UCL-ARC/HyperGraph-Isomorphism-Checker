from IsomorphismChecker_python_serial.graph_utils import create_hypergraph
from IsomorphismChecker_python_serial.isomorphisms import (
    get_connected_subgraphs,
)
import pytest

test_graph_dir = "tests/inputs/"

connected_graphs = ["MA_Graph.json", "Disconnected.json", "Cyclic_Graph.json"]


@pytest.mark.parametrize("graph_file", connected_graphs)
def test_connected_graph(graph_file):
    g = create_hypergraph(test_graph_dir + graph_file)
    subgraphs, _ = get_connected_subgraphs(g)
    assert len(subgraphs) == 1

    nodes, edges = subgraphs[0]
    assert sorted(nodes) == list(range(0, len(g.nodes)))
    assert sorted(edges) == list(range(0, len(g.edges)))


connected_graphs = ["Two_Subgraphs.json"]


@pytest.mark.parametrize("graph_file", connected_graphs)
def test_disconnected_graph(graph_file):
    g = create_hypergraph(test_graph_dir + graph_file)
    subgraphs, _ = get_connected_subgraphs(g)
    assert len(subgraphs) == 2

    assert sorted(subgraphs[0][0]) == list(range(0, 6))
    assert sorted(subgraphs[0][1]) == list(range(0, 2))

    assert sorted(subgraphs[1][0]) == list(range(6, 12))
    assert sorted(subgraphs[1][1]) == list(range(2, 4))
