from IsomorphismChecker_python_serial.graph_utils import create_hypergraph
from IsomorphismChecker_python_serial.draw import print_graph
from IsomorphismChecker_python_serial.isomorphisms import (
    MC_isomorphism,
    permute_graph,
    disconnected_subgraph_isomorphism,
)
from IsomorphismChecker_python_serial.hypergraph import OpenHypergraph
import pytest


def assert_isomorphism(g1, g2, pi_n, pi_e, p_nodes, p_edges, isomorphic):
    assert isomorphic
    assert p_nodes == pi_n
    assert p_edges == pi_e
    n_nodes = len(g1.nodes)
    n_edges = len(g2.edges)
    assert len(p_nodes) == n_nodes
    assert len(p_edges) == n_edges
    for i in range(n_nodes):
        assert i in p_nodes
        assert g1.nodes[i].label == g2.nodes[p_nodes[i]].label
    for i in range(n_edges):
        assert i in p_edges
        assert g1.edges[i].label == g2.edges[p_edges[i]].label

    for i in range(n_edges):
        e1 = g1.edges[i]
        e2 = g2.edges[p_edges[i]]

        for s in range(len(e1.sources)):
            assert p_nodes[e1.sources[s]] == e2.sources[s]
        for t in range(len(e1.targets)):
            assert p_nodes[e1.targets[t]] == e2.targets[t]


test_graph_dir = "tests/inputs/"


def Random_Permutation_Test(graph_file):
    g1 = create_hypergraph(test_graph_dir + graph_file)
    (pi_n, pi_e, g2) = permute_graph(g1)  # calculates a random permutation of the graph

    # print_graph(g1)
    # print_graph(g2)
    # print(pi_n, pi_e)

    isomorphic, p_nodes, p_edges = MC_isomorphism(g1, g2)
    assert_isomorphism(g1, g2, pi_n, pi_e, p_nodes, p_edges, isomorphic)


def check_non_isomorphic(graph_file_1, graph_file_2):
    g1 = create_hypergraph(test_graph_dir + graph_file_1)
    g2 = create_hypergraph(test_graph_dir + graph_file_2)
    (isomorphic, _, _) = MC_isomorphism(g1, g2)
    assert not isomorphic


def test_monogamous_acyclic_ismorphic():
    Random_Permutation_Test("MA_Graph.json")


def test_MA_no_inputs_isomorphic():
    Random_Permutation_Test("No_Inputs_Graph.json")


def test_MA_no_outputs_isomorphic():
    Random_Permutation_Test("No_Outputs_Graph.json")


acyclic_non_isomorphisms = [
    "Acyclic_Wrong_Edge_Label.json",
    "Acyclic_Wrong_Input_Connectivity.json",
    "Acyclic_Missing_Node.json",
    "Acyclic_Reordered_Edge_Output.json",
    "Acyclic_Wrong_Node_Label.json",
]


@pytest.mark.parametrize("graph_file", acyclic_non_isomorphisms)
def test_monogamous_acylic_non_isomorphic(graph_file):
    check_non_isomorphic("Acyclic_Graph.json", graph_file)


def test_node_back_tracking_isomorphic():
    Random_Permutation_Test("Acyclic_Graph.json")


def test_edge_back_tracking_isomorphic():
    Random_Permutation_Test("Edge_Backtrack.json")


def test_edge_back_tracking_non_isomorphic():
    check_non_isomorphic("Edge_Backtrack.json", "Edge_Backtrack_NonIsomorphism.json")


cyclic_graphs = ["Cyclic_Graph.json", "Recursive_Function_Graph.json"]


@pytest.mark.parametrize("graph_file", cyclic_graphs)
def test_cyclic_ismorphic(graph_file):
    Random_Permutation_Test(graph_file)


def test_cyclic_non_isomorphic():
    check_non_isomorphic("Cyclic_Graph.json", "Cyclic_NonIsomorphism.json")


def test_unit_graph():
    Random_Permutation_Test("Unit_Graph.json")
    check_non_isomorphic("Unit_Graph.json", "Non_Isomorphic_Unit.json")


@pytest.mark.parametrize("graph_file", acyclic_non_isomorphisms)
def test_empty_graph(graph_file):
    """Empty graph should be isomorphic to itself but not to any other graphs"""
    g1 = OpenHypergraph([], [], [], [])
    g2 = OpenHypergraph([], [], [], [])
    isomorphic, _, _ = MC_isomorphism(g1, g2)
    assert isomorphic

    g3 = create_hypergraph(test_graph_dir + graph_file)
    isomorphic, _, _ = MC_isomorphism(g1, g3)
    assert not isomorphic


def test_mono_disconnected_subgraph():
    g1 = create_hypergraph(test_graph_dir + "Disconnected.json")
    g2 = create_hypergraph(test_graph_dir + "Disconnected2.json")

    isomorphism = disconnected_subgraph_isomorphism(g1, g2)
    assert isomorphism.isomorphic


def test_mono_disconnected_subgraph_noniso():
    g1 = create_hypergraph(test_graph_dir + "Disconnected.json")
    g2 = create_hypergraph(test_graph_dir + "Disconnected3.json")

    isomorphism = disconnected_subgraph_isomorphism(g1, g2)
    assert not isomorphism.isomorphic


disconnected_graphs = ["Two_Subgraphs.json", "Three_Subgraphs.json"]


@pytest.mark.parametrize("graph_file", disconnected_graphs)
def test_disconnected_graph_isomorphism(graph_file):
    g1 = create_hypergraph(test_graph_dir + graph_file)
    pi_n, pi_e, g2 = permute_graph(g1)
    print(pi_n, pi_e)
    isomorphism = disconnected_subgraph_isomorphism(g1, g2)
    assert isomorphism.isomorphic


disconnected_non_isomorphisms = ["Three_Subgraphs_NonIso.json", "Two_Subgraphs.json"]


@pytest.mark.parametrize("graph_file", disconnected_non_isomorphisms)
def test_disconnected_graph_non_iso(graph_file):
    g1 = create_hypergraph(test_graph_dir + "Three_Subgraphs.json")
    g2 = create_hypergraph(test_graph_dir + graph_file)

    isomorphism = disconnected_subgraph_isomorphism(g1, g2)
    assert not isomorphism.isomorphic


non_monogamous_graphs = ["NonMonogamousGraph.json"]


@pytest.mark.parametrize("graph_file", non_monogamous_graphs)
def test_nonmonogamous_graphs(graph_file):
    g1 = create_hypergraph(test_graph_dir + graph_file)
    pi_n, pi_e, g2 = permute_graph(g1)
    print(pi_n, pi_e)
    print_graph(g2)
