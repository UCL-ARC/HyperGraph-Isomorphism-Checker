from IsomorphismChecker_python_serial.graph_utils import create_hypergraph
from IsomorphismChecker_python_serial.isomorphisms import (
    Colouring,
    Get_Canonical_Graph_Colouring,
)
import pytest

test_graph_dir = "tests/inputs/"

initial_step_graphs: list[str] = [
    "Acyclic_Graph.json",
    "Acyclic_Reordered_Edge_Output.json",
    "Acyclic_Wrong_Input_Connectivity.json",
    "Cyclic_Graph.json",
]
multi_step_graphs: list[str] = ["Multi_Step_Colouring.json"]

non_monogamous_graphs: list[str] = [
    "NonMonogamous_Ambiguous_Branching.json",
    "Ring.json",
]

symmetric_graphs: list[str] = ["Anonymous_Ring.json", "Clique.json"]

graphs_to_colour = (
    initial_step_graphs + multi_step_graphs + non_monogamous_graphs + symmetric_graphs
)


@pytest.mark.parametrize("graph_file", graphs_to_colour)
def test_colouring(graph_file):
    g = create_hypergraph(test_graph_dir + graph_file)

    colouring = Colouring(g)
    colouring.node_colouring.colouring = list(range(colouring.n_nodes))
    colouring.edge_colouring.colouring = list(range(colouring.n_edges))

    file_stub = graph_file[:-5]
    colouring = Get_Canonical_Graph_Colouring(
        g, file_stub + "_colouring", draw_steps=False
    )
    print(f"Node colouring of {file_stub}: {colouring.node_colouring.colouring}")
    print(f"Edge colouring of {file_stub}: {colouring.edge_colouring.colouring}")
    (node_uniqueness, _), (edge_uniqueness, _) = colouring.check_uniqueness()
    assert node_uniqueness
    assert edge_uniqueness
