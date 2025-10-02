from proofChecker_python_serial.graph_utils import create_hypergraph
from proofChecker_python_serial.isomorphisms import MC_isomorphism, permute_graph


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


def test_monogamous_acyclic_ismorphic():
    g1 = create_hypergraph(
        "tests/unit_tests_proofChecker_python/example_graphs/MA_Graph.json"
    )
    (pi, g2) = permute_graph(g1)  # calculates a random permutation of the graph

    isomorphic, p_nodes, p_edges = MC_isomorphism(g1, g2)
    assert_isomorphism(g1, g2, pi, p_nodes, p_edges, isomorphic)


def test_monogamous_acylic_non_isomorphic():
    pass


def test_node_back_tracking_isomorphic():
    g1 = create_hypergraph(
        "tests/unit_tests_proofChecker_python/example_graphs/Acyclic_Graph.json"
    )
    (pi, g2) = permute_graph(g1)  # calculates a random permutation of the graph

    isomorphic, p_nodes, p_edges = MC_isomorphism(g1, g2)
    assert_isomorphism(g1, g2, pi, p_nodes, p_edges, isomorphic)


def test_edge_back_tracking_isomorphic():
    pass


def test_node_back_tracking_non_isomorphic():
    pass


def test_edge_back_tracking_non_isomorphic():
    pass


def test_cyclic_ismorphic():
    g1 = create_hypergraph(
        "tests/unit_tests_proofChecker_python/example_graphs/Cyclic_Graph.json"
    )
    (pi, g2) = permute_graph(g1)  # calculates a random permutation of the graph

    isomorphic, p_nodes, p_edges = MC_isomorphism(g1, g2)
    assert_isomorphism(g1, g2, pi, p_nodes, p_edges, isomorphic)


def test_cyclic_non_isomorphic():
    pass


def test_all_features_isomorphic():
    pass


def test_all_features_non_isomorphic():
    pass
