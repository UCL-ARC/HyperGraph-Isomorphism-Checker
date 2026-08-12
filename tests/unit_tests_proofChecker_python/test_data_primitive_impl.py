from IsomorphismChecker_python_serial.graph_utils import create_hypergraph
from IsomorphismChecker_python_serial.hypergraph import constructLabelMap
from IsomorphismChecker_python_serial.draw import draw_graph
from IsomorphismChecker_python_serial.data_primitive_isomorphism import (
    ColourGlobalInterface,
    ColourData,
    ColouredGraph,
    InitialColouring,
    constructNodeColourKeys,
    constructEdgeColourKeys,
    setupColourCellKeyArrays,
    initialiseKeyArrays,
    InitialCompare,
    refineColouring,
    convergeColouring,
    selectTargetCell,
    forceRecolour,
    checkCompleteness,
    rollBackColouring,
    checkIsomorphism,
    checkBranch,
    determineIsomorphism,
)
from IsomorphismChecker_python_serial.isomorphisms import permute_graph
import numpy as np
import copy
import pytest

test_graph_dir = "tests/inputs/"


def test_flatten_graph():
    g = create_hypergraph(test_graph_dir + "Acyclic_Graph.json")
    g_flat = g.flatten()
    assert np.array_equal(g_flat.node_labels, np.array([0, 0, 1, 1, 0, 1, 2]))
    assert np.array_equal(g_flat.edge_labels, np.array([0, 1]))
    assert np.array_equal(g_flat.global_interface, np.array([0, 1, 2, 4, 5]))
    assert np.array_equal(g_flat.node_sources.initials, np.array([0, 0, 0, 0, 1, 2, 3]))
    assert np.array_equal(g_flat.node_sources.sizes, np.array([0, 0, 0, 1, 1, 1, 0]))
    assert np.array_equal(g_flat.node_sources.elements, np.array([0, 1, 1]))
    assert np.array_equal(g_flat.node_targets.initials, np.array([0, 1, 2, 3, 4, 4, 4]))
    assert np.array_equal(g_flat.node_targets.sizes, np.array([1, 1, 1, 1, 0, 0, 1]))
    assert np.array_equal(g_flat.node_targets.elements, np.array([0, 0, 1, 1, 1]))
    assert g_flat.num_inputs == 3


def test_initial_comparison():
    g1 = create_hypergraph(test_graph_dir + "Acyclic_Graph.json")
    (_, _, g2) = permute_graph(g1)
    g3 = create_hypergraph(test_graph_dir + "Acyclic_Wrong_Edge_Label.json")

    vertex_label_map = constructLabelMap(
        [v.label for v in g1.nodes + g2.nodes + g3.nodes]
    )
    edge_label_map = constructLabelMap(
        [e.label for e in g1.edges + g2.edges + g3.edges]
    )

    g1_flat = g1.flatten(vertex_label_map, edge_label_map)
    g2_flat = g2.flatten(vertex_label_map, edge_label_map)
    assert InitialCompare(g1_flat, g2_flat)

    g3_flat = g3.flatten(vertex_label_map, edge_label_map)
    assert not InitialCompare(g1_flat, g3_flat)


def test_interface_colour():
    g = create_hypergraph(test_graph_dir + "Acyclic_Graph.json")
    g_flat = g.flatten()
    node_colouring = ColourData(len(g_flat.node_labels))
    c_max = ColourGlobalInterface(g_flat, node_colouring)
    assert c_max == 4
    expected_nodes = np.array([0, 1, 2, 4, 5])
    expected_colours = np.array([0, 1, 2, 3, 4])
    expected_sizes = np.array([1, 1, 1, 1, 1])
    assert np.array_equal(node_colouring.c2v[:5], expected_nodes)
    for c, v in zip(range(5), expected_nodes):
        assert node_colouring.v2c[v] == c
    assert np.array_equal(node_colouring.c_sizes[:5], expected_sizes)
    assert np.array_equal(
        node_colouring.deltas[:5], np.column_stack((expected_colours, expected_sizes))
    )


def test_initial_colour():
    g = create_hypergraph(test_graph_dir + "Acyclic_Graph.json")
    node_colouring, edge_colouring = flatten_and_init_colour(g)
    assert np.all(node_colouring.v2c == np.array([0, 1, 2, 5, 3, 4, 6]))
    assert np.all(node_colouring.c2v == np.array([0, 1, 2, 4, 5, 3, 6]))
    assert np.all(edge_colouring.v2c == np.array([0, 1]))
    assert np.all(edge_colouring.c2v == np.array([0, 1]))

    g2 = create_hypergraph(test_graph_dir + "NonMonogamous_Ambiguous_Branching.json")
    n_colours2, e_colours2 = flatten_and_init_colour(g2)
    assert np.all(n_colours2.v2c == np.array([0, 1, 2, 7, 3, 4, 5, 6]))
    assert np.all(e_colours2.v2c == np.array([0, 1, 2, 2]))


def flatten_and_init_colour(g):
    g_flat = g.flatten()
    node_colouring = ColourData(g_flat.num_nodes)
    edge_colouring = ColourData(g_flat.num_edges)
    c_max = ColourGlobalInterface(g_flat, node_colouring)
    (_, _) = InitialColouring(g_flat, node_colouring, edge_colouring, c_max)
    # assert delta_entry == 6
    # assert delta_entry_edges == 1
    print(node_colouring)
    print(edge_colouring)
    for v in node_colouring.c2v:
        assert 0 <= v < g_flat.num_nodes
    for c in node_colouring.v2c:
        assert 0 <= c < g_flat.num_nodes
    for v in edge_colouring.c2v:
        assert 0 <= v < g_flat.num_edges
    for c in edge_colouring.v2c:
        assert 0 <= c < g_flat.num_edges
    return node_colouring, edge_colouring


def test_construct_keys():
    g = create_hypergraph(test_graph_dir + "NonMonogamous_Ambiguous_Branching.json")
    g_flat = g.flatten()
    node_colouring = ColourData(g_flat.num_nodes)
    edge_colouring = ColourData(g_flat.num_edges)
    cmax_v = ColourGlobalInterface(g_flat, node_colouring)
    (cmax_v, cmax_e) = InitialColouring(g_flat, node_colouring, edge_colouring, cmax_v)

    constructNodeColourKeys(
        g_flat.num_nodes,
        g_flat.node_keys,
        g_flat.node_sources,
        g_flat.node_s_ports,
        g_flat.node_targets,
        g_flat.node_t_ports,
        node_colouring,
        edge_colouring,
    )
    assert np.all(g_flat.node_keys.elements == 0)

    constructEdgeColourKeys(
        g_flat.num_edges,
        g_flat.edge_keys,
        g_flat.edge_sources,
        g_flat.edge_targets,
        edge_colouring,
        node_colouring,
    )
    segment_idx = g_flat.edge_keys.initials[2]  # colour of H edges
    segment = g_flat.edge_keys.elements[
        segment_idx : segment_idx + (2 * 2)
    ]  # 2 H edges share a colour and have total valency 2
    assert np.all(segment == np.array([7, 5, 7, 6]))


def test_colour_decomposition():
    g = create_hypergraph(test_graph_dir + "NonMonogamous_Ambiguous_Branching.json")
    draw_graph(g, "colour_decomp_test_graph.png")
    g_flat = g.flatten()
    node_colouring = ColourData(g_flat.num_nodes)
    edge_colouring = ColourData(g_flat.num_edges)
    c_max = ColourGlobalInterface(g_flat, node_colouring)
    (_, _) = InitialColouring(g_flat, node_colouring, edge_colouring, c_max)
    setupColourCellKeyArrays(
        g_flat.num_nodes,
        g_flat.node_sources,
        g_flat.node_targets,
        g_flat.node_cell_keys,
        node_colouring,
    )
    setupColourCellKeyArrays(
        g_flat.num_edges,
        g_flat.edge_sources,
        g_flat.edge_targets,
        g_flat.edge_cell_keys,
        edge_colouring,
    )

    refineColouring(g_flat, node_colouring, edge_colouring, 2)

    assert np.all(edge_colouring.c2v == np.array([0, 1, 2, 3]))
    assert np.all(edge_colouring.v2c == np.array([0, 1, 2, 3]))
    assert np.all(edge_colouring.Delta == np.array([1, 1, 2, 1]))
    assert np.all(edge_colouring.Delta_t == np.array([1, 1, 1, 2]))


def testcolourDecomposition2():
    g = create_hypergraph(test_graph_dir + "Fork_Join.json")
    draw_graph(g, "colour_decomp_test_graph.png")
    g_flat = g.flatten()
    node_colouring = ColourData(g_flat.num_nodes)
    edge_colouring = ColourData(g_flat.num_edges)
    c_max = ColourGlobalInterface(g_flat, node_colouring)
    assert node_colouring.v2c[0] == 0
    assert node_colouring.v2c[15] == 1
    assert np.all(node_colouring.v2c[1:15] == -1)
    (_, _) = InitialColouring(g_flat, node_colouring, edge_colouring, c_max)
    assert np.all(
        node_colouring.v2c
        == np.array([0, 14, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 15, 1])
    )
    assert np.all(
        edge_colouring.v2c
        == np.array([0, 0, 0, 14, 0, 0, 0, 0, 16, 0, 0, 0, 0, 14, 0, 0, 0])
    )
    setupColourCellKeyArrays(
        g_flat.num_nodes,
        g_flat.node_sources,
        g_flat.node_targets,
        g_flat.node_cell_keys,
        node_colouring,
    )
    setupColourCellKeyArrays(
        g_flat.num_edges,
        g_flat.edge_sources,
        g_flat.edge_targets,
        g_flat.edge_cell_keys,
        edge_colouring,
    )

    refineColouring(g_flat, node_colouring, edge_colouring, 2)
    assert np.all(
        node_colouring.v2c
        == np.array([0, 14, 2, 8, 11, 2, 2, 10, 13, 2, 2, 8, 11, 2, 15, 1])
    )
    assert np.all(
        edge_colouring.v2c
        == np.array([0, 10, 1, 14, 7, 4, 10, 3, 16, 9, 4, 10, 1, 14, 7, 4, 13])
    )


def testConvergeColouring():
    """Tests multi-step refinement until colouring is stable"""
    g = create_hypergraph(test_graph_dir + "Fork_Join.json")
    draw_graph(g, "colour_decomp_test_graph.png")
    g_flat = g.flatten()
    node_colouring = ColourData(g_flat.num_nodes)
    edge_colouring = ColourData(g_flat.num_edges)
    c_max = ColourGlobalInterface(g_flat, node_colouring)
    assert node_colouring.v2c[0] == 0
    assert node_colouring.v2c[15] == 1
    assert np.all(node_colouring.v2c[1:15] == -1)
    (_, _) = InitialColouring(g_flat, node_colouring, edge_colouring, c_max)
    assert np.all(
        node_colouring.v2c
        == np.array([0, 14, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 15, 1])
    )
    assert np.all(
        edge_colouring.v2c
        == np.array([0, 0, 0, 14, 0, 0, 0, 0, 16, 0, 0, 0, 0, 14, 0, 0, 0])
    )
    setupColourCellKeyArrays(
        g_flat.num_nodes,
        g_flat.node_sources,
        g_flat.node_targets,
        g_flat.node_cell_keys,
        node_colouring,
    )
    setupColourCellKeyArrays(
        g_flat.num_edges,
        g_flat.edge_sources,
        g_flat.edge_targets,
        g_flat.edge_cell_keys,
        edge_colouring,
    )
    convergeColouring(g_flat, node_colouring, edge_colouring, 1)
    assert np.all(
        node_colouring.v2c
        == np.array([0, 14, 5, 8, 11, 2, 7, 10, 13, 4, 5, 8, 11, 2, 15, 1])
    )
    assert np.all(
        edge_colouring.v2c
        == np.array([0, 10, 1, 14, 7, 4, 12, 3, 16, 9, 6, 10, 1, 14, 7, 4, 13])
    )


def testCompareNodeInvarient():
    pass


def testSelectTargetCell():
    """Tests multi-step refinement until colouring is stable"""
    g = create_hypergraph(test_graph_dir + "Fork_Join.json")
    draw_graph(g, "colour_decomp_test_graph.png")
    g_flat = g.flatten()
    node_colouring = ColourData(g_flat.num_nodes)
    edge_colouring = ColourData(g_flat.num_edges)
    c_max = ColourGlobalInterface(g_flat, node_colouring)
    (_, _) = InitialColouring(g_flat, node_colouring, edge_colouring, c_max)
    setupColourCellKeyArrays(
        g_flat.num_nodes,
        g_flat.node_sources,
        g_flat.node_targets,
        g_flat.node_cell_keys,
        node_colouring,
    )
    setupColourCellKeyArrays(
        g_flat.num_edges,
        g_flat.edge_sources,
        g_flat.edge_targets,
        g_flat.edge_cell_keys,
        edge_colouring,
    )
    convergeColouring(g_flat, node_colouring, edge_colouring, 1)
    c_node = selectTargetCell(node_colouring)
    assert c_node == 2


def testRecolouring():
    """Tests multi-step refinement until colouring is stable"""
    g = create_hypergraph(test_graph_dir + "Fork_Join.json")
    draw_graph(g, "colour_decomp_test_graph.png")
    cg = ColouredGraph(g.flatten())
    c_max = ColourGlobalInterface(cg.g, cg.vertexColours)
    (_, _) = InitialColouring(cg.g, cg.vertexColours, cg.edgeColours, c_max)
    setupColourCellKeyArrays(
        cg.g.num_nodes,
        cg.g.node_sources,
        cg.g.node_targets,
        cg.g.node_cell_keys,
        cg.vertexColours,
    )
    setupColourCellKeyArrays(
        cg.g.num_edges,
        cg.g.edge_sources,
        cg.g.edge_targets,
        cg.g.edge_cell_keys,
        cg.edgeColours,
    )
    t = convergeColouring(cg.g, cg.vertexColours, cg.edgeColours, 1)

    c_target = selectTargetCell(cg.vertexColours)
    (size, c_new, t) = forceRecolour(cg, c_target, t)
    assert size == 2
    assert c_new == 3
    assert checkCompleteness(cg.vertexColours, cg.edgeColours) == (True, True)


def testCheckIsomorphism():
    """Tests multi-step refinement until colouring is stable"""
    g = create_hypergraph(test_graph_dir + "Fork_Join.json")
    draw_graph(g, "colour_decomp_test_graph.png")
    g_flat = g.flatten()
    node_colouring = ColourData(g_flat.num_nodes)
    edge_colouring = ColourData(g_flat.num_edges)
    c_max = ColourGlobalInterface(g_flat, node_colouring)
    (_, _) = InitialColouring(g_flat, node_colouring, edge_colouring, c_max)
    setupColourCellKeyArrays(
        g_flat.num_nodes,
        g_flat.node_sources,
        g_flat.node_targets,
        g_flat.node_cell_keys,
        node_colouring,
    )
    setupColourCellKeyArrays(
        g_flat.num_edges,
        g_flat.edge_sources,
        g_flat.edge_targets,
        g_flat.edge_cell_keys,
        edge_colouring,
    )
    convergeColouring(g_flat, node_colouring, edge_colouring, 1)
    node_permutation, edge_permutation, permuted_graph = permute_graph(g)
    node_permutation = np.array(node_permutation)
    node_inverse = np.array(
        [np.where(node_permutation == i)[0][0] for i in range(len(node_permutation))]
    )
    edge_permutation = np.array(edge_permutation)
    edge_inverse = np.array(
        [np.where(edge_permutation == i)[0][0] for i in range(len(edge_permutation))]
    )

    cg1 = ColouredGraph(g_flat)
    cg1.vertexColours = node_colouring
    cg1.edgeColours = edge_colouring

    # Manually construct colouring for isomorphic graph
    cg2 = ColouredGraph(permuted_graph.flatten())
    cg2.vertexColours = copy.deepcopy(node_colouring)
    cg2.vertexColours.c2v = node_permutation[cg2.vertexColours.c2v]
    cg2.vertexColours.v2c = cg2.vertexColours.v2c[node_inverse]
    cg2.edgeColours = copy.deepcopy(edge_colouring)
    cg2.edgeColours.c2v = edge_permutation[cg2.edgeColours.c2v]
    cg2.edgeColours.v2c = cg2.edgeColours.v2c[edge_inverse]

    assert checkIsomorphism(cg1, cg2)


def testRollback():
    """Tests multi-step refinement until colouring is stable"""
    g = create_hypergraph(test_graph_dir + "Fork_Join.json")
    draw_graph(g, "colour_decomp_test_graph.png")
    g_flat = g.flatten()
    node_colouring = ColourData(g_flat.num_nodes)
    edge_colouring = ColourData(g_flat.num_edges)
    c_max = ColourGlobalInterface(g_flat, node_colouring)
    assert node_colouring.v2c[0] == 0
    assert node_colouring.v2c[15] == 1
    assert np.all(node_colouring.v2c[1:15] == -1)
    (_, _) = InitialColouring(g_flat, node_colouring, edge_colouring, c_max)
    assert np.all(
        node_colouring.v2c
        == np.array([0, 14, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 15, 1])
    )
    assert np.all(
        edge_colouring.v2c
        == np.array([0, 0, 0, 14, 0, 0, 0, 0, 16, 0, 0, 0, 0, 14, 0, 0, 0])
    )
    setupColourCellKeyArrays(
        g_flat.num_nodes,
        g_flat.node_sources,
        g_flat.node_targets,
        g_flat.node_cell_keys,
        node_colouring,
    )
    setupColourCellKeyArrays(
        g_flat.num_edges,
        g_flat.edge_sources,
        g_flat.edge_targets,
        g_flat.edge_cell_keys,
        edge_colouring,
    )
    refineColouring(g_flat, node_colouring, edge_colouring, 2)
    assert np.all(
        node_colouring.v2c
        == np.array([0, 14, 2, 8, 11, 2, 2, 10, 13, 2, 2, 8, 11, 2, 15, 1])
    )
    assert np.all(
        edge_colouring.v2c
        == np.array([0, 10, 1, 14, 7, 4, 10, 3, 16, 9, 4, 10, 1, 14, 7, 4, 13])
    )

    t0 = 3

    refineColouring(g_flat, node_colouring, edge_colouring, t0)

    # t = 4
    assert np.all(
        node_colouring.v2c
        == np.array([0, 14, 5, 8, 11, 2, 7, 10, 13, 4, 5, 8, 11, 2, 15, 1])
    )
    assert np.all(
        edge_colouring.v2c
        == np.array([0, 10, 1, 14, 7, 4, 12, 3, 16, 9, 6, 10, 1, 14, 7, 4, 13])
    )

    rollBackColouring(node_colouring, t0)
    rollBackColouring(edge_colouring, t0)
    assert np.all(
        node_colouring.v2c
        == np.array([0, 14, 2, 8, 11, 2, 2, 10, 13, 2, 2, 8, 11, 2, 15, 1])
    )
    assert np.all(
        edge_colouring.v2c
        == np.array([0, 10, 1, 14, 7, 4, 10, 3, 16, 9, 4, 10, 1, 14, 7, 4, 13])
    )


def testCheckBranch():
    """Tests multi-step refinement until colouring is stable"""
    g = create_hypergraph(test_graph_dir + "Fork_Join.json")
    draw_graph(g, "colour_decomp_test_graph.png")
    cg = ColouredGraph(g.flatten())
    c_max = ColourGlobalInterface(cg.g, cg.vertexColours)
    (_, _) = InitialColouring(cg.g, cg.vertexColours, cg.edgeColours, c_max)
    initialiseKeyArrays(cg)
    t1 = convergeColouring(cg.g, cg.vertexColours, cg.edgeColours, 1)

    c_target = selectTargetCell(cg.vertexColours)
    (size, c_new, t1) = forceRecolour(cg, c_target, t1)

    g2 = create_hypergraph(test_graph_dir + "Fork_Join.json")
    pn, pe, g2prime = permute_graph(g2)
    cg2 = ColouredGraph(g2prime.flatten())
    c_max2 = ColourGlobalInterface(cg2.g, cg2.vertexColours)
    (_, _) = InitialColouring(cg2.g, cg2.vertexColours, cg2.edgeColours, c_max2)
    initialiseKeyArrays(cg2)
    t2 = convergeColouring(cg2.g, cg2.vertexColours, cg2.edgeColours, 1)
    cg3 = copy.deepcopy(cg2)
    # These branches are automorphic so both should work
    assert checkBranch(cg, cg2, c_target, c_new, 1, t1, t2)
    assert checkBranch(cg, cg3, c_target, c_new, 0, t1, t2)


graphs = [
    "Fork_Join",
    "Acyclic_Graph",
    "Clique",
    "Unit_Graph",
    "Cyclic_Graph",
    "Recursive_Function_Graph",
    "NonMonogamousGraph",
    "Multi_Step_Colouring",
]


@pytest.mark.parametrize("graph_file", graphs)
def testFullIsomorphism(graph_file):
    g1 = create_hypergraph(test_graph_dir + graph_file + ".json")
    # draw_graph(g1, "colour_decomp_test_graph.png")
    g1_flat = g1.flatten()

    g2 = create_hypergraph(test_graph_dir + graph_file + ".json")
    pv, pe, g2 = permute_graph(g2)
    # draw_graph(g2, "colour_decomp_test_graph_iso.png")
    g2_flat = g2.flatten()

    assert determineIsomorphism(g1_flat, g2_flat)
