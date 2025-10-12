from proofChecker_python_serial.hypergraph import OpenHypergraph, Node, HyperEdge
import random

from dataclasses import dataclass, field


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


@dataclass(slots=True)
class Isomorphism:

    graphs: tuple[OpenHypergraph, OpenHypergraph]
    node_mapping: list[int] = field(init=False)
    edge_mapping: list[int] = field(init=False)
    is_isomorphic: bool = field(init=False)

    visited_nodes: list[int] = field(default_factory=list, init=False)
    visited_edges: list[int] = field(default_factory=list, init=False)

    @property
    def n_nodes(self) -> int:
        return len(self.graphs[0].nodes)

    @property
    def n_edges(self) -> int:
        return len(self.graphs[0].edges)

    def __post_init__(self):
        g1, g2 = self.graphs
        if not (
            len(g1.nodes) == len(g2.nodes)
            and len(g1.input_nodes) == len(g2.input_nodes)
            and len(g1.output_nodes) == len(g2.output_nodes)
            and len(g1.edges) == len(g2.edges)
        ):
            self.is_isomorphic = False
            self.node_mapping = []
            self.edge_mapping = []
        else:
            self.node_mapping = [-1] * self.n_nodes
            self.edge_mapping = [-1] * self.n_edges
            self.is_isomorphic = True

    def update_mapping(self, i: int, j: int, mode: str):
        """Update the mapping p with i -> j, return False if inconsistent"""

        if mode == "node":
            mapping = self.node_mapping
        elif mode == "edge":
            mapping = self.edge_mapping
        else:
            raise ValueError(f"Mode must be 'node' or 'edge', got {mode}")

        if i < 0 or i >= len(mapping):
            raise ValueError(
                f"Index {i} out of bounds for permutation of size {len(mapping)}"
            )
        if j < 0 or j >= len(mapping):
            raise ValueError(
                f"Index {j} out of bounds for permutation of size {len(mapping)}"
            )

        if mapping[i] in (-1, j):
            mapping[i] = j
            self.is_isomorphic = True
        else:
            self.is_isomorphic = False

    def update_mapping_list(self, list1: list[int], list2: list[int], mode: str):
        """Update the mapping p with i -> j for all i in list1 and j in list2, return False if inconsistent"""

        if len(list1) != len(list2):
            raise ValueError(
                f"Lists must be of same length, got {len(list1)} and {len(list2)}"
            )

        for i, j in zip(list1, list2):
            if self.is_isomorphic:
                self.update_mapping(i, j, mode)

    def check_edge_compatibility(self, e1: int, e2: int) -> bool:
        print("Check equality", e1, e2)

        # If one is None, the other must be None too for the return to be True
        if (e1 is None) or (e2 is None):
            return e1 is None and e2 is None

        edge1 = self.graphs[0].edges[e1]
        edge2 = self.graphs[1].edges[e2]

        return (
            (edge1.label == edge2.label)
            and (len(edge1.sources) == len(edge2.sources))
            and (len(edge1.targets) == len(edge2.targets))
        )

    def explore_edges(self, e1: int, e2: int) -> bool:
        if e1 in self.visited_edges:
            return self.edge_mapping[e1] == e2

        if not self.check_edge_compatibility(e1, e2):
            return False

        if e1 is None:
            return True
        else:
            self.visited_edges.append(e1)
            valid = self.update_mapping(self.edge_mapping, e1, e2)
            if not valid:
                return False

            # print(g1.edges[e1], g2.edges[e2])
            print(self.graphs[0].edges[e1], self.graphs[1].edges[e2])
            n_sources = len(self.graphs[0].edges[e1].sources)
            n_targets = len(self.graphs[0].edges[e1].targets)

            # check sources
            for s in range(n_sources):
                s1 = self.graphs[0].edges[e1].sources[s]
                s2 = self.graphs[1].edges[e2].sources[s]
                v1, v2 = self.graphs[0].nodes[s1], self.graphs[1].nodes[s2]
                print(f"Prev nodes = {v1, v2}")
                valid &= self.update_mapping(self.node_mapping, v1.index, v2.index)
                valid &= self.traverse_from_nodes(v1, v2)

            # check targets
            for t in range(n_targets):
                t1 = self.graphs[0].edges[e1].targets[t]
                t2 = self.graphs[1].edges[e2].targets[t]
                v1, v2 = self.graphs[0].nodes[t1], self.graphs[1].nodes[t2]
                print(f"Prev nodes = {v1, v2}")
                valid &= self.update_mapping(self.node_mapping, v1.index, v2.index)
                valid &= self.traverse_from_nodes(v1, v2)
            return valid

    def traverse_from_nodes(self, v1: Node, v2: Node) -> bool:
        print(f"Traversing {v1, v2}")
        if v1.index in self.visited_nodes:
            return v2.index == self.node_mapping[v1.index]

        self.visited_nodes.append(v1)
        valid = True
        valid &= self.explore_edges(v1.next, v2.next)
        valid &= self.explore_edges(v1.prev, v2.prev)

        return valid

    def check_MC_isomorphism(self) -> tuple[bool, list[int], list[int]]:
        """Check for graph isomorphism in monogamous, cartesian case"""

        g1, g2 = self.graphs

        # We can begin with mapping the input and output nodes only
        if self.is_isomorphic:
            self.update_mapping_list(g1.input_nodes, g2.input_nodes, mode="node")
        if self.is_isomorphic:
            self.update_mapping_list(g1.output_nodes, g2.output_nodes, mode="node")

        # Transverse from the input nodes
        for input_1, input_2 in zip(g1.input_nodes, g2.input_nodes):
            v1 = g1.nodes[input_1]
            v2 = g2.nodes[input_2]

            valid &= self.traverse_from_nodes(v1, v2)

        # Transverse from the output nodes
        for output_1, output_2 in zip(g1.output_nodes, g2.output_nodes):
            v1 = g1.nodes[output_1]
            v2 = g2.nodes[output_2]

            valid &= self.traverse_from_nodes(v1, v2)

        if any([i == -1 for i in self.node_mapping]):
            raise ValueError(f"Permutation incomplete: {self.node_mapping}")

        return (valid, self.node_mapping, self.edge_mapping)
