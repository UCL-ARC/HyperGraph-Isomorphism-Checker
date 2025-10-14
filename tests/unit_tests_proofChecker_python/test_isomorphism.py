from proofChecker_python_serial.graph_utils import create_hypergraph
from proofChecker_python_serial.isomorphisms import MC_isomorphism, permute_graph
import pytest


def assert_isomorphism(g1, g2, pi, p_nodes, p_edges, isomorphic):
    assert isomorphic
    assert p_nodes == pi
    n_nodes = len(g1.nodes)
    n_edges = len(g2.edges)
    assert len(p_nodes) == n_nodes
    assert len(p_edges) == n_edges
    for i in range(n_nodes):
        assert i in p_nodes
    for i in range(n_edges):
        assert i in p_edges


def Random_Permutation_Test(graph_file):
    g1 = create_hypergraph(
        "tests/unit_tests_proofChecker_python/example_graphs/" + graph_file
    )
    (pi, g2) = permute_graph(g1)  # calculates a random permutation of the graph

    isomorphic, p_nodes, p_edges = MC_isomorphism(g1, g2)
    assert_isomorphism(g1, g2, pi, p_nodes, p_edges, isomorphic)


def test_monogamous_acyclic_ismorphic():
    Random_Permutation_Test("MA_Graph.json")


def test_MA_no_inputs_isomorphic():
    Random_Permutation_Test("No_Inputs_Graph.json")


def test_MA_no_outputs_isomorphic():
    Random_Permutation_Test("No_Outputs_Graph.json")


def test_monogamous_acylic_non_isomorphic():
    pass


def test_node_back_tracking_isomorphic():
    Random_Permutation_Test("Acyclic_Graph.json")


def test_edge_back_tracking_isomorphic():
    pass


def test_node_back_tracking_non_isomorphic():
    pass


def test_edge_back_tracking_non_isomorphic():
    pass


cyclic_graphs = ["Cyclic_Graph.json", "Recursive_Function_Graph.json"]


@pytest.mark.parametrize("graph_file", cyclic_graphs)
def test_cyclic_ismorphic(graph_file):
    Random_Permutation_Test(graph_file)


def test_cyclic_non_isomorphic():
    pass


def test_all_features_isomorphic():
    pass


def test_all_features_non_isomorphic():
    pass
