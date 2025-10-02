from proofChecker_python_serial.hypergraph import OpenHypergraph, Node, HyperEdge
import random


def permute_graph(g):
    n = len(g.nodes)
    permutation = list(range(n))
    random.shuffle(permutation)
    nodes = []
    input_nodes = []
    output_nodes = []

    i = 0
    for i in range(n):
        nodes.append(Node(i, g.nodes[permutation[i]].label))
        i += 1

    for i in g.input_nodes:
        input_nodes.append(permutation[i])

    for o in g.output_nodes:
        output_nodes.append(permutation[o])

    i = 0
    edges = []
    for e in g.edges:
        s = []
        t = []
        for j in range(len(e.sources)):
            s.append(permutation[e.sources[j]])
        for j in range(len(e.targets)):
            t.append(permutation[e.targets[j]])
        edges.append(HyperEdge(s, t, e.label, i))
        i += 1
    print("make new grpah")
    g2 = OpenHypergraph(nodes, edges, input_nodes, output_nodes)
    return permutation, g2


def MC_isomorphism(g1, g2):
    """Check for graph isomorphism in monogamous, cartesian case"""

    def check_lengths(l1, l2):
        return len(l1) == len(l2)

    len_checks = (
        check_lengths(g1.nodes, g2.nodes)
        and check_lengths(g1.input_nodes, g2.input_nodes)
        and check_lengths(g1.output_nodes, g2.output_nodes)
        and check_lengths(g1.edges, g2.edges)
    )
    if not len_checks:
        return False

    n = len(g1.nodes)
    pi = [-1] * n  # list for the permutations
    E = len(g1.edges)
    edge_map = [-1] * E

    visited_nodes = []
    visited_edges = []

    n_inputs = len(g1.input_nodes)
    n_outputs = len(g2.output_nodes)

    def update_permutation(p, i, j):
        if p[i] == -1 or p[i] == j:
            p[i] = j
            return True
        else:
            return False

    valid = True

    # We can begin with the input and output nodes
    for i in range(n_inputs):
        valid &= update_permutation(pi, g1.input_nodes[i], g2.input_nodes[i])
    for o in range(n_outputs):
        valid &= update_permutation(pi, g1.output_nodes[o], g2.output_nodes[o])

    # depth-first graph traversal
    for i in range(n):
        print(f"Next {i}", g1.nodes[i].next, g2.nodes[i].next)
        print(f"Prev {i}", g1.nodes[i].prev, g2.nodes[i].prev)

    def check_edge_equality(e1, e2):
        print("Check equality", e1, e2)
        if (e1 is None) or (e2 is None):
            return e1 is None and e2 is None

        edge1 = g1.edges[e1]
        edge2 = g2.edges[e2]

        return (
            (edge1.label == edge2.label)
            and check_lengths(edge1.sources, edge2.sources)
            and check_lengths(edge1.targets, edge2.targets)
        )

    def explore_edges(e1, e2):
        if e1 in visited_edges:
            return edge_map[e1] == e2

        if not check_edge_equality(e1, e2):
            return False

        if e1 is None:
            return True
        else:
            visited_edges.append(e1)
            valid = update_permutation(edge_map, e1, e2)
            if valid:
                print(g1.edges[e1], g2.edges[e2])
                n_sources = len(g1.edges[e1].sources)
                n_targets = len(g1.edges[e1].targets)
                t1 = g1.edges[e1].targets
                t2 = g2.edges[e2].targets
                # check sources
                for s in range(n_sources):
                    s1 = g1.edges[e1].sources[s]
                    s2 = g2.edges[e2].sources[s]
                    v1, v2 = g1.nodes[s1], g2.nodes[s2]
                    print(f"Prev nodes = {v1, v2}")
                    valid &= update_permutation(pi, v1.index, v2.index)
                    valid &= traverse_from_nodes(v1, v2)
                # check targets
                for t in range(n_targets):
                    t1 = g1.edges[e1].targets[t]
                    t2 = g2.edges[e2].targets[t]
                    v1, v2 = g1.nodes[t1], g2.nodes[t2]
                    print(f"Prev nodes = {v1, v2}")
                    valid &= update_permutation(pi, v1.index, v2.index)
                    valid &= traverse_from_nodes(v1, v2)
            return valid

    def traverse_from_nodes(v1, v2):
        print(f"Traversing {v1, v2}")
        if v1.index in visited_nodes:
            return v2.index == pi[v1.index]

        visited_nodes.append(v1)
        # check prev edge
        valid = True
        valid &= explore_edges(v1.next, v2.next)
        valid &= explore_edges(v1.prev, v2.prev)

        return valid

    for i in range(n_inputs):
        v1 = g1.nodes[g1.input_nodes[i]]
        v2 = g2.nodes[g2.input_nodes[i]]

        valid &= traverse_from_nodes(v1, v2)

    if any([i == -1 for i in pi]):
        raise ValueError(f"Permutation incomplete: {pi}")

    return (valid, pi, edge_map)
