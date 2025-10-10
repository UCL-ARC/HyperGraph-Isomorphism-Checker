from proofChecker_python_serial.hypergraph import OpenHypergraph, Node, HyperEdge
import random


def permute_graph(g: OpenHypergraph) -> tuple[list[int], OpenHypergraph]:
    """Return a permuted version of the input graph and the permutation used"""

    permutation = list(range(len(g.nodes)))
    random.shuffle(permutation)

    nodes = [Node(i, g.nodes[permute].label) for i, permute in enumerate(permutation)]
    input_nodes = [permutation[i] for i in g.input_nodes]
    output_nodes = [permutation[o] for o in g.output_nodes]

    edges = []

    for index, edge in enumerate(g.edges):
        sources = [permutation[src] for src in edge.sources]
        targets = [permutation[tgt] for tgt in edge.targets]
        edges.append(HyperEdge(sources, targets, edge.label, index))

    permuted_graph = OpenHypergraph(nodes, edges, input_nodes, output_nodes)
    return permutation, permuted_graph


def update_mapping(p: list[int], i: int, j: int) -> bool:
    """Update the mapping p with i -> j, return False if inconsistent"""

    if i < 0 or i >= len(p):
        raise ValueError(f"Index {i} out of bounds for permutation of size {len(p)}")
    if j < 0 or j >= len(p):
        raise ValueError(f"Index {j} out of bounds for permutation of size {len(p)}")

    if p[i] in (-1, j):
        p[i] = j
        return True

    return False


def check_edge_compatibility(
    e1: int, e2: int, g1: OpenHypergraph, g2: OpenHypergraph
) -> bool:
    print("Check equality", e1, e2)
    if (e1 is None) or (e2 is None):
        return e1 is None and e2 is None

    edge1 = g1.edges[e1]
    edge2 = g2.edges[e2]

    return (
        (edge1.label == edge2.label)
        and (len(edge1.sources) == len(edge2.sources))
        and (len(edge1.targets) == len(edge2.targets))
    )


def explore_edges(
    e1,
    e2,
    visited_edges: list[int],
    pi: list[int],
    edge_map: list[int],
    g1: OpenHypergraph,
    g2: OpenHypergraph,
) -> bool:
    if e1 in visited_edges:
        return edge_map[e1] == e2

    if not check_edge_compatibility(e1, e2):
        return False

    if e1 is None:
        return True
    else:
        visited_edges.append(e1)
        valid = update_mapping(edge_map, e1, e2)
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
                valid &= update_mapping(pi, v1.index, v2.index)
                valid &= traverse_from_nodes(v1, v2)
            # check targets
            for t in range(n_targets):
                t1 = g1.edges[e1].targets[t]
                t2 = g2.edges[e2].targets[t]
                v1, v2 = g1.nodes[t1], g2.nodes[t2]
                print(f"Prev nodes = {v1, v2}")
                valid &= update_mapping(pi, v1.index, v2.index)
                valid &= traverse_from_nodes(v1, v2)
        return valid


def traverse_from_nodes(
    v1: Node,
    v2: Node,
    visited_nodes: list[Node],
    pi: list[int],
    g1: OpenHypergraph,
    g2: OpenHypergraph,
) -> bool:
    print(f"Traversing {v1, v2}")
    if v1.index in visited_nodes:
        return v2.index == pi[v1.index]

    visited_nodes.append(v1)
    valid = True
    valid &= explore_edges(v1.next, v2.next, visited_nodes, pi, g1, g2)
    valid &= explore_edges(v1.prev, v2.prev, visited_nodes, pi, g1, g2)

    return valid


def MC_isomorphism(g1: OpenHypergraph, g2: OpenHypergraph) -> bool:
    """Check for graph isomorphism in monogamous, cartesian case"""

    if not (
        len(g1.nodes) == len(g2.nodes)
        and len(g1.input_nodes) == len(g2.input_nodes)
        and len(g1.output_nodes) == len(g2.output_nodes)
        and len(g1.edges) == len(g2.edges)
    ):
        return False

    n = len(g1.nodes)
    pi = [-1] * n  # list for the permutations
    E = len(g1.edges)
    edge_map = [-1] * E

    visited_nodes = []
    visited_edges = []

    n_inputs = len(g1.input_nodes)
    n_outputs = len(g2.output_nodes)

    # We can begin with the input and output nodes
    for i in range(n_inputs):
        valid = update_mapping(pi, g1.input_nodes[i], g2.input_nodes[i])
        if not valid:
            return (False, [], [])
    for o in range(n_outputs):
        valid = update_mapping(pi, g1.output_nodes[o], g2.output_nodes[o])
        if not valid:
            return (False, [], [])

    for i in range(n_inputs):
        v1 = g1.nodes[g1.input_nodes[i]]
        v2 = g2.nodes[g2.input_nodes[i]]

        valid &= traverse_from_nodes(v1, v2, visited_nodes, pi)

    for o in range(n_outputs):
        v1 = g1.nodes[g1.output_nodes[o]]
        v2 = g2.nodes[g2.output_nodes[o]]

        valid &= traverse_from_nodes(v1, v2, visited_nodes, pi)

    if any([i == -1 for i in pi]):
        raise ValueError(f"Permutation incomplete: {pi}")

    return (valid, pi, edge_map)
