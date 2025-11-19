from IsomorphismChecker_python_serial.hypergraph import (
    OpenHypergraph,
    Node,
    HyperEdge,
    SubGraph,
)
import random
import logging

from dataclasses import dataclass, field
from enum import Enum

# Set up logger for this module
logger = logging.getLogger(__name__)


@dataclass
class IsomorphismData:
    isomorphic: bool
    p_nodes: list[int]
    p_edges: list[int]


NonIso = IsomorphismData(False, [], [])


def permute_graph(g: OpenHypergraph) -> tuple[list[int], list[int], OpenHypergraph]:
    """Return a permuted version of the input graph and the permutation used"""

    def invert_permutation(p):
        return [p.index(i) for i in range(len(p))]

    node_permutation = list(range(len(g.nodes)))
    random.shuffle(node_permutation)
    node_reverse_permutation = invert_permutation(node_permutation)

    nodes = [
        Node(i, g.nodes[permute].label)
        for i, permute in enumerate(node_reverse_permutation)
    ]
    input_nodes = [node_permutation[i] for i in g.input_nodes]
    output_nodes = [node_permutation[o] for o in g.output_nodes]

    edges = []

    edge_permutation = list(range(len(g.edges)))
    random.shuffle(edge_permutation)
    edge_reverse_permutation = invert_permutation(edge_permutation)
    for i in range(len(g.edges)):
        edge = g.edges[edge_reverse_permutation[i]]
        sources = [node_permutation[src] for src in edge.sources]
        targets = [node_permutation[tgt] for tgt in edge.targets]
        edges.append(HyperEdge(sources, targets, edge.label, i))

    permuted_graph = OpenHypergraph(nodes, edges, input_nodes, output_nodes)
    return node_permutation, edge_permutation, permuted_graph


class MappingMode(Enum):
    NODE = "node"
    EDGE = "edge"


class BiMap:
    def __init__(self):
        self.map = {}
        self.inverse = {}

    def insert(self, i, j):
        if (i in self.map and self.map[i] != j) or (
            j in self.inverse and self.inverse[j] != i
        ):
            return False
        else:
            self.map[i] = j
            self.inverse[j] = i
            return True


@dataclass(slots=True)
class Isomorphism:
    """Class to check isomorphism between two hypergraphs."""

    graphs: tuple[OpenHypergraph, OpenHypergraph]
    node_mapping: list[int] = field(init=False)
    edge_mapping: list[int] = field(init=False)
    mapping_valid: bool = field(init=False)

    visited_nodes: list[int] = field(default_factory=list, init=False)
    visited_edges: list[int] = field(default_factory=list, init=False)

    @property
    def n_nodes(self) -> int:
        """Return the number of nodes in the graphs."""
        return len(self.graphs[0].nodes)

    @property
    def n_edges(self) -> int:
        """Return the number of edges in the graphs."""
        return len(self.graphs[0].edges)

    def __post_init__(self):
        """Post-initialization checks for isomorphism. Includes basic size checks."""

        g1, g2 = self.graphs
        if not (
            len(g1.nodes) == len(g2.nodes)
            and len(g1.input_nodes) == len(g2.input_nodes)
            and len(g1.output_nodes) == len(g2.output_nodes)
            and len(g1.edges) == len(g2.edges)
        ):
            self.mapping_valid = False
            self.node_mapping = []
            self.edge_mapping = []
        else:
            self.node_mapping = [-1] * self.n_nodes
            self.edge_mapping = [-1] * self.n_edges
            self.mapping_valid = True

    def update_mapping(self, i: int, j: int, mode: MappingMode):
        """Update the mapping p with i -> j, Sets is_isomorphic to False if inconsistent"""

        if mode == MappingMode.NODE:
            mapping = self.node_mapping
        elif mode == MappingMode.EDGE:
            mapping = self.edge_mapping
        else:
            raise ValueError(f"Mode must be 'node' or 'edge', got {mode}")

        logger.debug(f"Current mapping {mode}: {mapping}")
        logger.debug(f"Updating mapping {mode}: {i} -> {j}")

        if i < 0 or i >= len(mapping):
            raise ValueError(
                f"Index {i} out of bounds for permutation of size {len(mapping)}"
            )
        if j < 0 or j >= len(mapping):
            raise ValueError(
                f"Index {j} out of bounds for permutation of size {len(mapping)}"
            )

        if mapping[i] == j:
            return
        elif mapping[i] == -1:
            if j in mapping:
                # Another node is already mapped to j
                self.mapping_valid = False
                logger.debug(
                    f"Contradiction mapping {mode} {i} -> {j}; existing mapping {mapping.index(j)} -> {j}"
                )
            else:
                mapping[i] = j
                logger.debug(f"Mapping {mode} {i} -> {j}")
        else:
            self.mapping_valid = False

    def update_mapping_list(
        self, list1: list[int], list2: list[int], mode: MappingMode
    ):
        """Update the mapping p with i -> j for all i in list1 and j in list2, Sets is_isomorphic to False if inconsistent"""

        if len(list1) != len(list2):
            raise ValueError(
                f"Lists must be of same length, got {len(list1)} and {len(list2)}"
            )

        for i, j in zip(list1, list2):
            if self.mapping_valid:
                self.update_mapping(i, j, mode)

    def check_edge_compatibility(self, e1: int | None, e2: int | None) -> bool:
        """Checks the length of sources and targets and labels of two edges for compatibility"""

        logger.debug(f"Check equality: e1={e1}, e2={e2}")

        if e1 is None and e2 is None:
            return True

        if e1 is None or e2 is None:
            return False

        edge1 = self.graphs[0].edges[e1]
        edge2 = self.graphs[1].edges[e2]

        return (
            (edge1.label == edge2.label)
            and (len(edge1.sources) == len(edge2.sources))
            and (len(edge1.targets) == len(edge2.targets))
        )

    def explore_edges(self, e1: int, e2: int):
        """Explore edges e1 and e2, updating mappings and traversing connected nodes."""

        logger.debug(f"Exploring edges {e1, e2}")
        if e1 in self.visited_edges:
            self.mapping_valid = (
                False if self.edge_mapping[e1] != e2 else self.mapping_valid
            )
            return

        if not self.check_edge_compatibility(e1, e2):
            self.mapping_valid = False
            return

        self.visited_edges.append(e1)
        logger.debug(f"Mapping edge {e1} -> {e2}")
        logger.debug(f"Currently is_isomorphic = {self.mapping_valid}")
        self.update_mapping(e1, e2, MappingMode.EDGE)
        logger.debug(f"Currently is_isomorphic = {self.mapping_valid}")
        logger.debug("Checkpoint 1")
        if not self.mapping_valid:
            return False
        logger.debug("Checkpoint 2")

        # Commented out: logger.debug(g1.edges[e1], g2.edges[e2])
        logger.debug(f"Edges: {self.graphs[0].edges[e1]}, {self.graphs[1].edges[e2]}")
        n_sources = len(self.graphs[0].edges[e1].sources)
        n_targets = len(self.graphs[0].edges[e1].targets)

        logger.debug(f"n_sources, n_targets = {n_sources, n_targets}")

        # check sources
        for s in range(n_sources):
            s1 = self.graphs[0].edges[e1].sources[s]
            s2 = self.graphs[1].edges[e2].sources[s]
            v1, v2 = self.graphs[0].nodes[s1], self.graphs[1].nodes[s2]
            logger.debug(f"Prev nodes = {v1, v2}")
            self.update_mapping(v1.index, v2.index, mode=MappingMode.NODE)
            if self.mapping_valid:
                self.traverse_from_nodes(v1, v2)
            else:
                return

        # check targets
        for t in range(n_targets):
            t1 = self.graphs[0].edges[e1].targets[t]
            t2 = self.graphs[1].edges[e2].targets[t]
            v1, v2 = self.graphs[0].nodes[t1], self.graphs[1].nodes[t2]
            logger.debug(f"Prev nodes = {v1, v2}")
            self.update_mapping(v1.index, v2.index, mode=MappingMode.NODE)
            if self.mapping_valid:
                self.traverse_from_nodes(v1, v2)
            else:
                return

    def traverse_from_nodes(self, v1: Node, v2: Node):
        """Traverse the graph from nodes v1 and v2, exploring connected edges and nodes."""

        logger.debug(f"Traversing {v1, v2}")
        if v1.index in self.visited_nodes:
            if v2.index != self.node_mapping[v1.index]:
                self.mapping_valid = False
            return

        if v1.label != v2.label:
            self.mapping_valid = False
            return

        self.visited_nodes.append(v1.index)

        if (v1.next is None) or (v2.next is None):
            if v1.next != v2.next:
                self.mapping_valid = False
                return
        else:
            self.explore_edges(v1.next.index, v2.next.index)

        if (v1.prev is None) or (v2.prev is None):
            if v1.prev != v2.prev:
                self.mapping_valid = False
                return
        else:
            self.explore_edges(v1.prev.index, v2.prev.index)

    def check_subgraph_isomorphism(
        self, v1: int, v2: int, subgraph1, subgraph2
    ) -> IsomorphismData:
        """Check for disconnected subgraph isomorphism where no nodes connect
        to global inputs or outputs"""

        g1, g2 = self.graphs

        if (v1 < 0 or v1 > max(subgraph1.nodes)) or (
            v2 < 0 or v2 > max(subgraph2.nodes)
        ):
            raise ValueError(
                f"Node pair {(v1, v2)} not in the node sets for graph pair."
            )

        # beginning by mapping v1 to v2 as a first attempt
        self.update_mapping(v1, v2, MappingMode.NODE)

        self.traverse_from_nodes(g1.nodes[v1], g2.nodes[v2])

        if self.mapping_valid:
            if any([self.node_mapping[i] == -1 for i in subgraph1.nodes]):
                raise ValueError(f"Permutation incomplete: {self.node_mapping}")

        return IsomorphismData(self.mapping_valid, self.node_mapping, self.edge_mapping)

    def check_MC_isomorphism(self) -> tuple[bool, list[int], list[int]]:
        """Check for graph isomorphism in monogamous, cartesian case"""

        g1, g2 = self.graphs

        # We can begin with mapping the input and output nodes only
        if self.mapping_valid:
            self.update_mapping_list(
                g1.input_nodes, g2.input_nodes, mode=MappingMode.NODE
            )
        if self.mapping_valid:
            self.update_mapping_list(
                g1.output_nodes, g2.output_nodes, mode=MappingMode.NODE
            )

        # Transverse from the input nodes
        for input_1, input_2 in zip(g1.input_nodes, g2.input_nodes):
            v1 = g1.nodes[input_1]
            v2 = g2.nodes[input_2]

            logger.debug(f"Starting traversal from input nodes {v1, v2}")

            self.traverse_from_nodes(v1, v2)

        # Transverse from the output nodes
        for output_1, output_2 in zip(g1.output_nodes, g2.output_nodes):
            v1 = g1.nodes[output_1]
            v2 = g2.nodes[output_2]

            self.traverse_from_nodes(v1, v2)

        if self.mapping_valid:
            if any([i == -1 for i in self.node_mapping]):
                raise ValueError(f"Permutation incomplete: {self.node_mapping}")

        return (self.mapping_valid, self.node_mapping, self.edge_mapping)


def get_connected_subgraphs(
    g: OpenHypergraph,
) -> tuple[list[SubGraph], dict]:
    num_nodes = len(g.nodes)
    num_edges = len(g.edges)

    node_subgraph_map = {}
    current_sub_graph = 0

    added_nodes = [False] * num_nodes
    added_edges = [False] * num_edges

    def traverse_connected_graph(
        node_idx: int, node_list: list[int], edge_list: list[int]
    ):
        if added_nodes[node_idx]:
            return

        node_list.append(node_idx)
        node_subgraph_map[node_idx] = current_sub_graph
        added_nodes[node_idx] = True
        next_edge = g.nodes[node_idx].next
        if next_edge is not None:
            traverse_connected_graph_from_edge(next_edge.index, node_list, edge_list)

        prev_edge = g.nodes[node_idx].prev
        if prev_edge is not None:
            traverse_connected_graph_from_edge(prev_edge.index, node_list, edge_list)

    def traverse_connected_graph_from_edge(edge_idx, node_list, edge_list):
        if added_edges[edge_idx]:
            return

        edge_list.append(edge_idx)
        added_edges[edge_idx] = True
        edge = g.edges[edge_idx]
        for s in edge.sources:
            traverse_connected_graph(s, node_list, edge_list)
        for t in edge.targets:
            traverse_connected_graph(t, node_list, edge_list)

    subgraphs = []

    for i in range(num_nodes):
        if not added_nodes[i]:
            node_list: list[int] = []
            edge_list: list[int] = []
            traverse_connected_graph(i, node_list, edge_list)
            subgraphs.append(SubGraph(node_list, edge_list))
            current_sub_graph += 1

    # also explore edges in case any subgraph is a disconnected edge w/no inputs or outputs
    for i in range(num_edges):
        if not added_edges[i]:
            node_list = []
            edge_list = []
            traverse_connected_graph_from_edge(i, node_list, edge_list)
            subgraphs.append(SubGraph(node_list, edge_list))
            current_sub_graph += 1

    return subgraphs, node_subgraph_map


def disconnected_subgraph_isomorphism(g1: OpenHypergraph, g2: OpenHypergraph):
    """Compares two monogamous subgraph isomorphism candidates (g1, g2) which
    have no paths to global inputs and outputs, and checks for ismorphism."""

    g1_subgraphs, subgraph_map_1 = get_connected_subgraphs(g1)
    g2_subgraphs, subgraph_map_2 = get_connected_subgraphs(g2)

    ## Check basic sizes to begin with
    num_nodes = len(g1.nodes)
    num_edges = len(g1.edges)
    if (num_nodes != len(g2.nodes)) or (num_edges != len(g2.edges)):
        return NonIso

    # Check number of subgraphs and sizes of subgraphs.
    g1_sizes = [(len(sg.nodes), len(sg.edges)) for sg in g1_subgraphs]
    g2_sizes = [(len(sg.nodes), len(sg.edges)) for sg in g2_subgraphs]
    if sorted(g1_sizes) != sorted(g2_sizes):
        return NonIso

    ## Start by eliminating all graphs that are connected to global interface
    if (len(g1.input_nodes) != len(g2.input_nodes)) | (
        len(g1.output_nodes) != len(g2.output_nodes)
    ):
        return NonIso

    paired_subgraphs = BiMap()
    subgraph_start_point = {}

    def update_mapping_from_interface(nodes1, nodes2):
        for i in range(len(nodes1)):
            sg1 = subgraph_map_1[nodes1[i]]
            sg2 = subgraph_map_2[nodes2[i]]
            if not paired_subgraphs.insert(sg1, sg2):
                return False
            else:
                subgraph_start_point[i] = (nodes1[i], nodes2[i])
        return True

    if not update_mapping_from_interface(g1.input_nodes, g2.input_nodes):
        return NonIso
    if not update_mapping_from_interface(g1.output_nodes, g2.output_nodes):
        return NonIso

    iso = Isomorphism((g1, g2))

    isomorphic = IsomorphismData(True, [-1] * num_nodes, [-1] * num_edges)

    for sg1, sg2 in paired_subgraphs.map.items():
        v1, v2 = subgraph_start_point[sg1]
        sub_isomorphic = iso.check_subgraph_isomorphism(
            v1, v2, g1_subgraphs[sg1], g2_subgraphs[sg2]
        )
        if not sub_isomorphic:
            return NonIso
        else:
            merge_isomorphism(isomorphic, sub_isomorphic)

    def check_subgraph_pair(sg1: SubGraph, sg2: SubGraph):
        # another disconnected subgraph; check for isomorphism by depth first search
        if len(sg1.nodes) != len(sg2.nodes) or len(sg1.edges) != len(sg2.edges):
            return NonIso  # these can't be isomorphic if sizes don't match

        # # Find most unique nodes by local neighbourhood (next/previous)
        neighbour_map1 = construct_neighbour_map(sg1, g1)
        neighbour_map2 = construct_neighbour_map(sg2, g2)
        if neighbour_map1 != neighbour_map2:
            print(neighbour_map1, neighbour_map2)
            return NonIso

        optimal_key = sorted(neighbour_map1.items(), key=lambda x: x[1])[0][
            0
        ]  # optimal key has the fewest matching nodes
        print(optimal_key)
        starters_1 = list(
            filter(lambda x: construct_node_key(x, g1) == optimal_key, sg1.nodes)
        )
        starters_2 = list(
            filter(lambda x: construct_node_key(x, g2) == optimal_key, sg2.nodes)
        )

        v1 = starters_1[0]
        for v2 in starters_2:
            iso = Isomorphism((g1, g2))
            sub_isomorphic = iso.check_subgraph_isomorphism(v1, v2, sg1, sg2)
            if sub_isomorphic.isomorphic:
                if not paired_subgraphs.insert(i, j):
                    return NonIso
                else:
                    return sub_isomorphic
        return NonIso

    def construct_neighbour_map(sg: SubGraph, g: OpenHypergraph):
        neighbour_map: dict[str, int] = {}
        for v in sg.nodes:

            key = construct_node_key(v, g)
            if key in neighbour_map:
                neighbour_map[key] += 1
            else:
                neighbour_map[key] = 1
        return neighbour_map

    def construct_node_key(v: int, g: OpenHypergraph):
        e_n = g.nodes[v].next
        e_p = g.nodes[v].prev

        def edge_sig(e):
            if e is None:
                return "___"
            else:
                return e.label + str(e.port)

        key = "i:" + edge_sig(e_n) + "_o:" + edge_sig(e_p)
        return key

    for i, sg1 in enumerate(g1_subgraphs):
        if i not in paired_subgraphs.map:
            # disconnected subgraph
            for j, sg2 in enumerate(g2_subgraphs):
                if j not in paired_subgraphs.inverse:
                    sub_isomorphic = check_subgraph_pair(sg1, sg2)
                    if sub_isomorphic.isomorphic:
                        merge_isomorphism(isomorphic, sub_isomorphic)
                        break

    if len(paired_subgraphs.map) != len(g1_subgraphs):
        return NonIso
    else:
        return isomorphic


def merge_isomorphism(iso_main: IsomorphismData, iso_contribution: IsomorphismData):
    if not iso_contribution.isomorphic:
        iso_main = NonIso
        return

    if len(iso_main.p_nodes) != len(iso_contribution.p_nodes) or len(
        iso_main.p_edges
    ) != len(iso_contribution.p_edges):
        raise ValueError("Mismatched isomorphism mapping sizes")

    for i, (x, y) in enumerate(zip(iso_main.p_nodes, iso_contribution.p_nodes)):
        iso_main.p_nodes[i] = y if x == -1 else x

    for i, (x, y) in enumerate(zip(iso_main.p_edges, iso_contribution.p_edges)):
        iso_main.p_edges[i] = y if x == -1 else x


def MC_isomorphism(
    g1: OpenHypergraph, g2: OpenHypergraph
) -> tuple[bool, list[int], list[int]]:

    iso = Isomorphism((g1, g2))

    ret = iso.check_MC_isomorphism() if iso.mapping_valid else (False, [], [])

    logger.info(f"Visited nodes: {iso.visited_nodes}")

    return ret
