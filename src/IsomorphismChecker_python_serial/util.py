from enum import Enum


def assert_isomorphism(g1, g2, isomorphism):
    assert isomorphism.isomorphic

    n_nodes = len(g1.nodes)
    n_edges = len(g2.edges)
    assert len(isomorphism.p_nodes) == n_nodes
    assert len(isomorphism.p_edges) == n_edges
    for i in range(n_nodes):
        assert i in isomorphism.p_nodes
        assert g1.nodes[i].label == g2.nodes[isomorphism.p_nodes[i]].label
    for i in range(n_edges):
        assert i in isomorphism.p_edges
        assert g1.edges[i].label == g2.edges[isomorphism.p_edges[i]].label

    for i in range(n_edges):
        e1 = g1.edges[i]
        e2 = g2.edges[isomorphism.p_edges[i]]

        for s in range(len(e1.sources)):
            assert isomorphism.p_nodes[e1.sources[s]] == e2.sources[s]
        for t in range(len(e1.targets)):
            assert isomorphism.p_nodes[e1.targets[t]] == e2.targets[t]


class MappingMode(Enum):
    NODE = "node"
    EDGE = "edge"
