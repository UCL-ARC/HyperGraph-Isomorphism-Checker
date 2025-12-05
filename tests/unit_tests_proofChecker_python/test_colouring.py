from IsomorphismChecker_python_serial.graph_utils import create_hypergraph
from IsomorphismChecker_python_serial.isomorphisms import (
    Colouring,
)
from IsomorphismChecker_python_serial.diagram import Diagram
import pytest

test_graph_dir = "tests/inputs/"

graphs_to_colour: list[str] = ["Acyclic_Graph.json"]


@pytest.mark.parametrize("graph_file", graphs_to_colour)
def test_colouring(graph_file):
    g = create_hypergraph(test_graph_dir + graph_file)

    colouring = Colouring(g, g)
    colouring.node_colouring.colouring = list(range(colouring.n_nodes))
    colouring.edge_colouring.colouring = list(range(colouring.n_edges))

    d = Diagram(g)
    d.drawGraph(colouring=colouring)
    d.render("hypergraph_diagram_test")
    # colouring = Colour_Graph_Pair(g, g)
    # print(colouring.node_colouring)
    # print(colouring.edge_colouring)
