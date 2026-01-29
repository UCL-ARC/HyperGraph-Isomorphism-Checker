from IsomorphismChecker_python_serial.hypergraph import (
    OpenHypergraph,
    Node,
    HyperEdge,
    SubGraph,
)
import random
import logging

from IsomorphismChecker_python_serial.diagram import Diagram
from IsomorphismChecker_python_serial.colouring import Colouring, ColourMap
from IsomorphismChecker_python_serial.util import MappingMode

from dataclasses import dataclass, field

# Set up logger for this module
logger = logging.getLogger(__name__)


@dataclass
class IsomorphismData:
    isomorphic: bool
    p_nodes: list[int]
    p_edges: list[int]


NonIso = IsomorphismData(False, [], [])


def head(some_list: list):
    return None if not some_list else some_list[0]


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


class BiMap:
    def __init__(self):
        self.map: dict[int, int] = {}
        self.inverse: dict[int, int] = {}

    def insert(self, i: int, j: int) -> bool:
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
        print(f"Traverse from edges {e1, e2}")
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

        print(f"Traverse from nodes {v1, v2}")
        logger.debug(f"Traversing {v1, v2}")
        if v1.index in self.visited_nodes:
            if v2.index != self.node_mapping[v1.index]:
                self.mapping_valid = False
            return

        if v1.label != v2.label:
            self.mapping_valid = False
            return

        self.visited_nodes.append(v1.index)

        print("check splitting")
        # nodes require the same amount of splitting in each directoin
        if len(v1.next) != len(v2.next) or (len(v1.prev) != len(v2.prev)):
            self.mapping_valid = False
            return

        # num_nexts = len(v1.next)
        # matching_paths: list[list[int]] = [[] for i in range(num_nexts)]
        # print(f"Nexts = {v1.next, v2.next}")
        # eliminated_paths = {i: False for i in range(num_nexts)}
        if len(v1.next) > 1:
            raise RuntimeError("Branching paths algorithm in progress")

            # explore paths starting with the minimal branching
        else:
            next1, next2 = head(v1.next), head(v2.next)
            if self.check_edges_for_continuation(next1, next2):
                self.explore_edges(next1.index, next2.index)

            prev1, prev2 = head(v1.prev), head(v2.prev)
            if self.check_edges_for_continuation(prev1, prev2):
                self.explore_edges(prev1.index, prev2.index)

    def check_edges_for_continuation(self, next1, next2):
        if next1 is None or next2 is None:
            print(f"ONe of them is none: {next1, next2}")
            if next1 != next2:
                print(f"Only one of them was none: {next1, next2}")
                self.mapping_valid = False
            return False  # no need to continue path if edges are None or don't match

        if next1.label != next2.label:
            print(f"Label mismatch {next1, next2}")
            self.mapping_valid = False
            return False

        if next1.port != next2.port:
            print(f"Port mismatch {next1, next2}")
            self.mapping_valid = False
            return False  # no need to continue path if ports don't match

        return True

    def check_subgraph_isomorphism(
        self, v1: int, v2: int, subgraph1: SubGraph, subgraph2: SubGraph
    ) -> IsomorphismData:
        """Check for disconnected subgraph isomorphism where no nodes connect
        to global inputs or outputs"""
        print("Check subgraph isomorphism")
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
        next_edge = (
            None
            if head(g.nodes[node_idx].next) is None
            else head(g.nodes[node_idx].next).index
        )
        if next_edge is not None:
            traverse_connected_graph_from_edge(next_edge, node_list, edge_list)

        prev_edge = (
            None
            if head(g.nodes[node_idx].prev) is None
            else head(g.nodes[node_idx].prev).index
        )
        if prev_edge is not None:
            traverse_connected_graph_from_edge(prev_edge, node_list, edge_list)

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

    subgraphs: list[SubGraph] = []

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
    print("disconnected graph isomorphism")

    g1_subgraphs, subgraph_map_1 = get_connected_subgraphs(g1)
    g2_subgraphs, subgraph_map_2 = get_connected_subgraphs(g2)

    ## Check basic sizes to begin with
    print("size check")
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
    print("interface check")
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

    for sg1_idx, sg2_idx in paired_subgraphs.map.items():
        v1, v2 = subgraph_start_point[sg1_idx]
        sub_isomorphic = iso.check_subgraph_isomorphism(
            v1, v2, g1_subgraphs[sg1_idx], g2_subgraphs[sg2_idx]
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
        neighbour_map1 = construct_neighbour_map(g1, sg1)
        neighbour_map2 = construct_neighbour_map(g2, sg2)
        if neighbour_map1 != neighbour_map2:
            return NonIso

        optimal_key = sorted(neighbour_map1.items(), key=lambda x: x[1])[0][
            0
        ]  # optimal key has the fewest matching nodes
        logger.debug(optimal_key)
        starters_1 = list(
            filter(lambda x: construct_node_key(g1.nodes[x]) == optimal_key, sg1.nodes)
        )
        starters_2 = list(
            filter(lambda x: construct_node_key(g2.nodes[x]) == optimal_key, sg2.nodes)
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


def construct_neighbour_map(g: OpenHypergraph, sg: SubGraph | None = None):
    neighbour_map: dict[str, int] = {}
    nodes = g.nodes if sg is None else [g.nodes[v] for v in sg.nodes]
    for v in nodes:
        key = construct_node_key(v)
        if key in neighbour_map:
            neighbour_map[key] += 1
        else:
            neighbour_map[key] = 1
    return neighbour_map


def construct_node_key(v: Node):
    e_n = head(v.next)  # TODO: Update to handle non-monogamous?
    e_p = head(v.prev)

    def edge_sig(e):
        if e is None:
            return "___"
        else:
            return e.label + str(e.port)

    key = "i:" + edge_sig(e_n) + "_o:" + edge_sig(e_p)
    return key


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


def Comparative_Graph_Colouring(
    g1: OpenHypergraph, g2: OpenHypergraph, filename: str = "", draw_steps=False
):
    n_nodes = len(g1.nodes)
    n_edges = len(g2.edges)

    if len(g2.nodes) != n_nodes:
        return NonIso
    if len(g2.edges) != n_edges:
        return NonIso

    colours1, colours2 = Colouring(g1), Colouring(g2)

    if len(g1.input_nodes) != len(g2.input_nodes):
        return NonIso

    if len(g1.output_nodes) != len(g2.output_nodes):
        return NonIso

    # Unique colours for input/output nodes
    # Sorting is not strictly necessary but I think will make the
    # parallelisation easier later on
    c_running = 0  # current colour to apply to updated nodes
    for (i1, i2) in zip(
        g1.input_nodes + g1.output_nodes, g2.input_nodes + g2.output_nodes
    ):
        # v1 and v2 must be a viable match
        v1, v2 = g1.nodes[i1], g2.nodes[i2]
        if v1.label != v2.label:
            return NonIso
        if len(v1.next) != len(v2.next):
            return NonIso
        if len(v1.prev) != len(v2.prev):
            return NonIso

        # c1, c2 give current colourings for v1, v2
        c1 = colours1.node_colouring.colouring[i1]
        c2 = colours2.node_colouring.colouring[i2]
        if (c1 == -1) and (c2 == -1):
            colours1.set_colour(i1, c_running, MappingMode.NODE)
            colours2.set_colour(i2, c_running, MappingMode.NODE)
            c_running += 1
        elif c1 != c2:
            # contradiction if existing node colours don't match
            return NonIso

    # Initial node and edge colourings
    if not set_initial_label_map(g1, g2, colours1, colours2, c_running):
        return NonIso

    if draw_steps:
        d = Diagram(g1, colouring=colours1)
        d.render(filename + "_g1_" + "0")
        d = Diagram(g2, colouring=colours2)
        d.render(filename + "_g2_" + "0")

    # Initial colouring complete, begin iterative updates
    viable, iteration = Update_Colouring_Pair(
        g1, g2, filename, colours1, colours2, 1, draw_steps
    )
    if not viable:
        return NonIso

    # Symmetry breaking
    (nodes_unique, node_symmetry), (
        edges_unique,
        edge_symmetry,
    ) = colours1.check_uniqueness()
    while (not nodes_unique) or (not edges_unique):
        if not nodes_unique:
            # check colour sets match across graphs
            cset1 = colours1.node_colouring.colour_map[node_symmetry]
            cset2 = colours2.node_colouring.colour_map[node_symmetry]
            if len(cset1) != len(cset2):
                return NonIso
            break_symmetry(colours1.node_colouring, node_symmetry)
            break_symmetry(colours2.node_colouring, node_symmetry)
        elif not edges_unique:
            cset1 = colours1.edge_colouring.colour_map[edge_symmetry]
            cset2 = colours2.edge_colouring.colour_map[edge_symmetry]
            if len(cset1) != len(cset2):
                return NonIso
            break_symmetry(colours1.edge_colouring, edge_symmetry)
            break_symmetry(colours2.node_colouring, node_symmetry)

        viable, iteration = Update_Colouring_Pair(
            g1, g2, filename, colours1, colours2, iteration, draw_steps
        )
        if not viable:
            return NonIso
        (nodes_unique, node_symmetry), (
            edges_unique,
            edge_symmetry,
        ) = colours1.check_uniqueness()

    # check that g2 has also be coloured uniquely
    (nodes_unique, node_symmetry), (
        edges_unique,
        edge_symmetry,
    ) = colours2.check_uniqueness()
    if not (nodes_unique and edges_unique):
        return NonIso

    # Construct isomorphism from colouring
    # Map from g1 -> g2; get the colours of node/edge i and then find the node/edge with that colour in g2
    node_map = [
        colours2.node_colouring.colour_map[colours1.node_colouring.colouring[i]].pop()
        for i in range(n_nodes)
    ]
    edge_map = [
        colours2.edge_colouring.colour_map[colours1.edge_colouring.colouring[i]].pop()
        for i in range(n_edges)
    ]
    return IsomorphismData(True, node_map, edge_map)


def break_symmetry_pair(cmap1: ColourMap, cmap2: ColourMap, symmetry_colour: int):
    cgroup1 = cmap1.colour_map[symmetry_colour]  # nodes in g1 with symm colour
    cgroup2 = cmap2.colour_map[symmetry_colour]  # and for g2

    n_group = len(cgroup1)
    if len(cgroup2) != n_group:
        # This should not be possible unless there is a bug
        raise Exception("Colour groups are different sizes.")


def Update_Colouring_Pair(
    g1: OpenHypergraph,
    g2: OpenHypergraph,
    filename: str,
    colours1: Colouring,
    colours2: Colouring,
    iteration: int,
    draw_steps=False,
):
    static = False
    while not static:
        ## Update node colourings
        static_nodes = True
        for c in colours1.node_colouring.colour_map.keys():
            viable, static_set = refine_colour_set(
                c, g1, g2, colours1, colours2, MappingMode.NODE
            )
            if not viable:
                return False, iteration
            static_nodes = static_nodes and static_set

        colours1.node_colouring.mergeUpdates()
        colours2.node_colouring.mergeUpdates()

        static_edges = True
        for c in colours1.edge_colouring.colour_map.keys():
            viable, static_set = refine_colour_set(
                c, g1, g2, colours1, colours2, MappingMode.EDGE
            )
            if not viable:
                return False, iteration
            static_edges = static_edges and static_set

        static = static_nodes and static_edges

        colours1.edge_colouring.mergeUpdates()
        colours2.edge_colouring.mergeUpdates()

        iteration += 1
        if draw_steps:
            d = Diagram(g1, colouring=colours1)
            d.render(filename + "_g1_" + str(iteration))
            d = Diagram(g2, colouring=colours2)
            d.render(filename + "_g2_" + str(iteration))

    return True, iteration


def refine_colour_set(
    c: int,
    g1: OpenHypergraph,
    g2: OpenHypergraph,
    colours1: Colouring,
    colours2: Colouring,
    mode: MappingMode,
):
    """Returns a (bool, bool) tuple; the first bool is true if isomorphism is still possible and false if there is a conflict.
    The second bool is true if the refinement is stable and false if there has been an update. (The second bool is arbitrary if
    a conflict has occurred.)"""

    # get corresponding colour group in g1 and g2
    colouring1 = colours1.get_map(mode)
    colouring2 = colours2.get_map(mode)
    cset1 = colouring1.colour_map[c]
    cset2 = colouring2.colour_map[c]

    if len(cset1) != len(cset2):
        return (False, False)
    if len(cset1) == 1:
        return (True, True)

    # attempt to split the colour groups
    # these colour keys can be stored in a segmented array for all nodes/edges
    # rather than being constructed like this
    def get_key(x, y, z):
        if mode == MappingMode.NODE:
            return GetNodeColourKey(x, y.nodes[z])
        else:
            return GetEdgeColourKey(x, y.edges[z])

    indexed_keys1 = [(v, get_key(colours1, g1, v)) for v in cset1]
    indexed_keys1.sort(key=lambda x: x[1])
    indexed_keys2 = [(v, get_key(colours2, g2, v)) for v in cset2]
    indexed_keys2.sort(key=lambda x: x[1])

    # these colour keys should be the same for both graphs else non isomorphic
    if [k for (_, k) in indexed_keys1] != [k for (_, k) in indexed_keys2]:
        return (False, False)
    # assign new colours
    static1 = AssignColours(colouring1, c, indexed_keys1)
    static2 = AssignColours(colouring2, c, indexed_keys2)
    if static1 != static2:
        raise Exception("Applying recolouring for two graphs was inconsistent")
    return (True, static1)


def set_initial_label_map(
    g1: OpenHypergraph,
    g2: OpenHypergraph,
    colours1: Colouring,
    colours2: Colouring,
    start_colour: int,
):
    # Initial label map can be constructed from a histogram of node and egde labels
    # Constructing historgrams like this should be efficient on the GPU here we can
    # just construct a list for simplicity
    node_type_list1 = [
        (i, v.label)
        for i, v in enumerate(g1.nodes)
        if colours1.node_colouring.colouring[i] == -1
    ]
    node_type_list2 = [
        (i, v.label)
        for i, v in enumerate(g2.nodes)
        if colours2.node_colouring.colouring[i] == -1
    ]
    node_type_list1.sort(key=lambda z: z[1])
    node_type_list2.sort(key=lambda z: z[1])

    # The histograms need to match to be valid
    if [label for (_, label) in node_type_list1] != [
        label for (_, label) in node_type_list2
    ]:
        return False

    AssignColours(colours1.node_colouring, start_colour, node_type_list1)
    colours1.node_colouring.mergeUpdates()
    AssignColours(colours2.node_colouring, start_colour, node_type_list2)
    colours2.node_colouring.mergeUpdates()

    edge_type_list1 = [(i, e.label) for i, e in enumerate(g1.edges)]
    edge_type_list1.sort(key=lambda z: z[1])
    edge_type_list2 = [(i, e.label) for i, e in enumerate(g2.edges)]
    edge_type_list2.sort(key=lambda z: z[1])

    if [label for (_, label) in edge_type_list1] != [
        label for (_, label) in edge_type_list2
    ]:
        return False

    AssignColours(colours1.edge_colouring, 0, edge_type_list1)
    colours1.edge_colouring.mergeUpdates()
    AssignColours(colours2.edge_colouring, 0, edge_type_list2)
    colours2.edge_colouring.mergeUpdates()

    return True


def Get_Canonical_Graph_Colouring(
    g: OpenHypergraph, filename: str, draw_steps=False
) -> Colouring:
    # iso = Isomorphism((g1, g2))

    # Need a dimension check

    # Colourings for nodes (do we also need to colour edges?)
    colours = Colouring(g)

    # To start let's try to colour a single graph

    # Unique colours for input/output nodes
    c = 0
    for v in g.input_nodes + g.output_nodes:
        ## assign colour if unassigned: a node can appear as both input and output
        ## and can also appear in the input/output list more than once
        if colours.node_colouring.colouring[v] == -1:
            colours.node_colouring.colouring[v] = c
            colours.node_colouring.colour_map[c] = set(
                [v]
            )  ## every colour group should be a singleton
            c += 1

    # Initial colouring of remaining nodes and edges by their labels
    InitialiseColours(g, colours, c)

    ## Output initial colouring
    iteration = 0
    if draw_steps:
        d = Diagram(g, colouring=colours)
        d.render(filename + str(iteration))

    # initialise update maps to avoid dictionary changing size during iterations

    # Updating egde and node colours by their labels
    # Only need to update colours on nodes which are not uniquely coloured
    iteration = Update_Colourings(g, filename, colours, iteration, draw_steps)

    # After first update the graph should be uniquely coloured up to automorphism groups
    # Automorphism groups need to be broken manually to arrive at an isomorphism
    (nodes_unique, node_symmetry), (
        edges_unique,
        edge_symmetry,
    ) = colours.check_uniqueness()
    while (not nodes_unique) or (not edges_unique):
        if not nodes_unique:
            break_symmetry(colours.node_colouring, node_symmetry)
        elif not edges_unique:
            break_symmetry(colours.edge_colouring, edge_symmetry)

        iteration = Update_Colourings(g, filename, colours, iteration, draw_steps)
        (nodes_unique, node_symmetry), (
            edges_unique,
            edge_symmetry,
        ) = colours.check_uniqueness()

    return colours


def break_symmetry(cmap: ColourMap, symmetry_colour: int):
    colour_group = cmap.colour_map[symmetry_colour]
    n_group = len(colour_group)
    break_idx = colour_group.pop()
    new_colour = symmetry_colour + n_group - 1
    cmap.colouring[break_idx] = new_colour
    cmap.colour_map[new_colour] = set([break_idx])


def Update_Colourings(g1, filename, colours: Colouring, iteration, draw_steps=False):
    static = False
    while not static:
        ## Update node colouring
        static_nodes = True
        for (colour, colour_set) in colours.node_colouring.colour_map.items():
            if len(colour_set) > 1:
                # attempt to split colour group
                indexed_keys = [
                    (v, GetNodeColourKey(colours, g1.nodes[v])) for v in colour_set
                ]
                # sort the indexed colour keys
                indexed_keys.sort(key=lambda x: x[1])
                # assign new colours
                static_set = AssignColours(colours.node_colouring, colour, indexed_keys)
                static_nodes = static_nodes and static_set
        colours.node_colouring.mergeUpdates()
        iteration += 1
        if draw_steps:
            d = Diagram(g1, colouring=colours)
            d.render(filename + str(iteration))

        ## Update edge colouring
        static_edges = True
        for (colour, colour_set) in colours.edge_colouring.colour_map.items():
            if len(colour_set) > 1:
                # attempt to split colour group
                indexed_keys = [
                    (e, GetEdgeColourKey(colours, g1.edges[e])) for e in colour_set
                ]
                # sort the indexed colour keys
                indexed_keys.sort(key=lambda x: x[1])
                # assign new colours
                static_set = AssignColours(colours.edge_colouring, colour, indexed_keys)
                static_edges = static_edges and static_set
        colours.edge_colouring.mergeUpdates()

        static = static_nodes & static_edges
        iteration += 1
        if draw_steps:
            d = Diagram(g1, colouring=colours)
            d.render(filename + str(iteration))
    return iteration


def AssignColours(
    cmap: ColourMap, start_colour: int, indexed_keys: list[tuple[int, str]]
):
    static = True
    c_running = start_colour
    c = start_colour
    key = ""
    for (i, k) in indexed_keys:
        if c_running != cmap.colouring[i]:
            static = False
        if k == key:
            cmap.colouring[i] = c_running
            cmap.update_map[c_running].add(i)
        else:  # new key --> new colour
            cmap.colouring[i] = c
            cmap.update_map[c] = set([i])
            c_running = c
            key = k
        c += 1
    return static


def GetNodeColourKey(colours: Colouring, v: Node):
    """Gather the colours of adjoining edges and convert it into a hashable key"""
    prevs = [(colours.edge_colouring.colouring[e.index], e.port) for e in v.prev]
    prevs.sort()
    nexts = [(colours.edge_colouring.colouring[e.index], e.port) for e in v.next]
    nexts.sort()
    key = f"{prevs}:{nexts}"
    return key


def GetEdgeColourKey(colours: Colouring, e: HyperEdge):
    """Gather the colours of source and target nodes and convert to hashable key"""
    sources = [colours.node_colouring.colouring[v] for v in e.sources]
    targets = [colours.node_colouring.colouring[v] for v in e.targets]
    # Unlike nodes there is no sorting here because they are already ordered
    key = f"{sources}:{targets}"
    return key


def InitialiseColours(g1: OpenHypergraph, colours: Colouring, start_colour: int):
    node_type_list = [
        (i, v.label)
        for i, v in enumerate(g1.nodes)
        if colours.node_colouring.colouring[i] == -1
    ]
    node_type_list.sort(key=lambda z: z[1])
    AssignColours(colours.node_colouring, start_colour, node_type_list)
    colours.node_colouring.mergeUpdates()

    edge_type_list = [(i, e.label) for i, e in enumerate(g1.edges)]
    edge_type_list.sort(key=lambda z: z[1])
    AssignColours(colours.edge_colouring, 0, edge_type_list)
    colours.edge_colouring.mergeUpdates()
