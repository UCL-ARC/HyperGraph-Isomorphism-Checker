from IsomorphismChecker_python_serial.graph_utils import create_hypergraph
from IsomorphismChecker_python_serial.hypergraph import constructLabelMap
from IsomorphismChecker_python_serial.draw import draw_graph
from IsomorphismChecker_python_serial.data_primitive_isomorphism import (
    ColourGlobalInterface,
    ColourData,
    InitialColouring,
    constructNodeColourKeys,
    constructEdgeColourKeys,
    colourSetDecomposition,
    setupColourCellKeyArrays,
    InitialCompare,
)
from IsomorphismChecker_python_serial.isomorphisms import permute_graph
import numpy as np

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
    (delta_entry, delta_entry_edges) = InitialColouring(
        g_flat, node_colouring, edge_colouring, c_max
    )
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


def test_construct_edge_keys():
    g = create_hypergraph(test_graph_dir + "NonMonogamous_Ambiguous_Branching.json")
    g_flat = g.flatten()
    node_colouring = ColourData(g_flat.num_nodes)
    edge_colouring = ColourData(g_flat.num_edges)
    cmax_v = ColourGlobalInterface(g_flat, node_colouring)
    (cmax_v, cmax_e) = InitialColouring(g_flat, node_colouring, edge_colouring, cmax_v)
    constructEdgeColourKeys(
        g_flat.num_edges,
        g_flat.edge_keys,
        g_flat.edge_sources,
        g_flat.edge_targets,
        node_colouring,
    )
    segment_idx = g_flat.edge_keys.initials[2]  # colour of H edges
    segment = g_flat.edge_keys.elements[
        segment_idx : segment_idx + (2 * 2)
    ]  # 2 H edges share a colour and have total valency 2
    assert np.all(segment == np.array([7, 5, 7, 6]))


def test_construct_node_keys():
    g = create_hypergraph(test_graph_dir + "Acyclic_Graph.json")
    g_flat = g.flatten()
    node_colouring = ColourData(g_flat.num_nodes)
    edge_colouring = ColourData(g_flat.num_edges)
    c_max = ColourGlobalInterface(g_flat, node_colouring)
    (delta_entry, delta_entry_edges) = InitialColouring(
        g_flat, node_colouring, edge_colouring, c_max
    )
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
    constructEdgeColourKeys(
        g_flat.num_edges,
        g_flat.edge_keys,
        g_flat.edge_sources,
        g_flat.edge_targets,
        node_colouring,
    )


def test_colour_decomposition():
    g = create_hypergraph(test_graph_dir + "NonMonogamous_Ambiguous_Branching.json")
    draw_graph(g, "colour_decomp_test_graph.png")
    g_flat = g.flatten()
    node_colouring = ColourData(g_flat.num_nodes)
    edge_colouring = ColourData(g_flat.num_edges)
    c_max = ColourGlobalInterface(g_flat, node_colouring)
    (delta_entry, delta_entry_edges) = InitialColouring(
        g_flat, node_colouring, edge_colouring, c_max
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
    print(node_colouring)
    print(edge_colouring)
    print(g_flat.node_sources)
    print(g_flat.node_targets)
    print(g_flat.node_keys)
    constructNodeColourKeys(
        g_flat.num_nodes,
        g_flat.node_keys,
        g_flat.node_sources,
        g_flat.node_s_ports,
        g_flat.node_targets,
        g_flat.node_t_ports,
        edge_colouring,
    )
    constructEdgeColourKeys(
        g_flat.num_edges,
        g_flat.edge_keys,
        g_flat.edge_sources,
        g_flat.edge_targets,
        node_colouring,
    )
    colourSetDecomposition(
        g_flat.num_nodes, g_flat.node_cell_keys, g_flat.node_keys, node_colouring
    )
    print("Edge set decomposition")
    colourSetDecomposition(
        g_flat.num_edges, g_flat.edge_cell_keys, g_flat.edge_keys, edge_colouring
    )
