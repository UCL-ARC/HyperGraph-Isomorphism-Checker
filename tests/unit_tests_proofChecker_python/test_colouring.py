from IsomorphismChecker_python_serial.graph_utils import create_hypergraph
from IsomorphismChecker_python_serial.isomorphisms import Colouring, Colour_Graph_Pair
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

    colouring = Colour_Graph_Pair(g, g)
    d.drawGraph(colouring=colouring)
    d.render("hypergraph_colouring_test")
    # print(colouring.node_colouring)
    # print(colouring.edge_colouring)
