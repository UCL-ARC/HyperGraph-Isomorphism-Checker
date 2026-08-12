"""Module to define hypergraphs and related structures."""
import numpy as np
from dataclasses import dataclass, field

from IsomorphismChecker_python_serial.hyperedge import HyperEdge
from IsomorphismChecker_python_serial.node import Node, EdgeInfo


@dataclass
class SubGraph:
    nodes: list[int]
    edges: list[int]


@dataclass
class SegmentedArray:
    elements: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    initials: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    sizes: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))

    @classmethod
    def reserving(cls, n_segments: int, n_total: int):
        return cls(
            initials=np.zeros(n_segments, dtype=np.int64),
            sizes=np.zeros(n_segments, dtype=np.int64),
            elements=np.zeros(n_total, dtype=np.int64),
        )

    def __eq__(self, other):
        if not isinstance(other, SegmentedArray):
            return NotImplemented

        return (
            np.array_equal(self.elements, other.elements)
            and np.array_equal(self.initials, other.initials)
            and np.array_equal(self.sizes, other.sizes)
        )


class FlatHypergraph:
    """An open hypergraph with a flat np.array structure for data parallel implementations"""

    def __init__(
        self,
        node_labels,
        node_sources,
        node_s_ports,
        node_targets,
        node_t_ports,
        edge_labels,
        edge_sources,
        edge_targets,
        num_inputs,
        global_interface,
    ):
        self.node_labels = np.array(node_labels)
        self.node_sources: SegmentedArray = node_sources
        self.node_s_ports: SegmentedArray = node_s_ports
        self.node_t_ports: SegmentedArray = node_t_ports
        self.node_targets: SegmentedArray = node_targets
        self.num_nodes = self.node_labels.size
        self.edge_labels = np.array(edge_labels)
        self.edge_sources: SegmentedArray = edge_sources
        self.edge_targets: SegmentedArray = edge_targets
        self.num_edges = self.edge_labels.size
        self.num_inputs = num_inputs
        self.global_interface = np.array(global_interface)

        self.n_connections = node_sources.elements.size + node_targets.elements.size
        ## node keys are packed ints to account for colour AND port information
        self.node_keys = SegmentedArray().reserving(self.num_nodes, self.n_connections)
        start = 0
        for i in range(self.num_nodes):
            self.node_keys.initials[i] = start
            self.node_keys.sizes[i] = (
                self.node_sources.sizes[i] + self.node_targets.sizes[i]
            )
            start += self.node_keys.sizes[i]

        self.edge_keys = SegmentedArray().reserving(self.num_edges, self.n_connections)
        start = 0
        for i in range(self.num_edges):
            self.edge_keys.initials[i] = start
            self.edge_keys.sizes[i] = (
                self.edge_sources.sizes[i] + self.edge_targets.sizes[i]
            )
            start += self.edge_keys.sizes[i]

        self.node_cell_keys = SegmentedArray().reserving(
            self.num_nodes, self.n_connections
        )
        self.edge_cell_keys = SegmentedArray().reserving(
            self.num_edges, self.n_connections
        )


def constructLabelMap(labels):
    label_set = list(set(labels))
    label_set.sort()
    label_map = {l: i for (i, l) in enumerate(label_set)}
    return label_map


@dataclass
class OpenHypergraph:
    """An open hypergraph with input and output nodes."""

    nodes: list[Node] = field(default_factory=list)
    edges: list[HyperEdge] = field(default_factory=list)

    input_nodes: list[int] = field(default_factory=list)
    output_nodes: list[int] = field(default_factory=list)

    def flatten(self, vertex_label_map=None, edge_label_map=None):
        """Construct array representation used in data parallel approach"""

        ## Convert readable string labels to compact integer labels
        if vertex_label_map is None:
            vertex_label_map = constructLabelMap([v.label for v in self.nodes])
        if edge_label_map is None:
            edge_label_map = constructLabelMap([e.label for e in self.edges])

        (
            node_labels,
            node_sources,
            node_targets,
            source_ports,
            target_ports,
        ) = self.flatten_elements(
            self.nodes, len(self.nodes), vertex_label_map, nodes=True
        )
        edge_labels, edge_sources, edge_targets, _, _ = self.flatten_elements(
            self.edges, len(self.edges), edge_label_map, nodes=False
        )

        num_inputs = len(self.input_nodes)
        global_interface = self.input_nodes + self.output_nodes
        return FlatHypergraph(
            node_labels,
            node_sources,
            source_ports,
            node_targets,
            target_ports,
            edge_labels,
            edge_sources,
            edge_targets,
            num_inputs,
            global_interface,
        )

    def flatten_elements(self, elements, N, label_map, nodes=True):
        labels = [0] * N
        total_sources = sum([len(x.sources) for x in elements])
        total_targets = sum([len(x.targets) for x in elements])
        sources = SegmentedArray.reserving(N, total_sources)
        targets = SegmentedArray.reserving(N, total_targets)
        s_ports = SegmentedArray.reserving(N, total_sources)
        t_ports = SegmentedArray.reserving(N, total_targets)
        source_count = 0
        target_count = 0
        for i in range(N):
            labels[i] = label_map[elements[i].label]
            sources.initials[i] = source_count
            targets.initials[i] = target_count
            s_ports.initials[i] = source_count
            t_ports.initials[i] = target_count
            for j in range(len(elements[i].sources)):
                src = elements[i].sources[j]
                if nodes:
                    sources.elements[source_count] = src.index
                    s_ports.elements[source_count] = src.port
                else:
                    sources.elements[source_count] = src
                source_count += 1

            for j in range(len(elements[i].targets)):
                tgt = elements[i].targets[j]
                if nodes:
                    targets.elements[target_count] = tgt.index
                    t_ports.elements[target_count] = tgt.port
                else:
                    targets.elements[target_count] = tgt
                target_count += 1

            sources.sizes[i] = source_count - sources.initials[i]
            targets.sizes[i] = target_count - targets.initials[i]
        return labels, sources, targets, s_ports, t_ports

    # TODO: Improve efficiency by caching results and invalidating on changes
    def is_valid(self) -> bool:
        """Check if the hypergraph is valid."""
        if not self.nodes:
            return False

        if not self.edges:
            return False

        return True

    def check_nodes_in_graph(self, nodes) -> bool:
        """Check if all nodes are in the hypergraph."""
        return all(node < len(self.nodes) for node in nodes)

    def set_next_prev(self, edge: HyperEdge):
        """Set the next and previous edges for nodes based on edges in the hypergraph."""

        for i, v in enumerate(edge.sources):
            node = self.nodes[v]
            node.targets.append(EdgeInfo(edge.index, i, edge.label))
        #    else:
        #        raise ValueError(
        #            f"Source node {node.label} of edge {edge.label} already has a next edge. This is not currently supported."
        #        )

        for i, v in enumerate(edge.targets):
            node = self.nodes[v]
            node.sources.append(EdgeInfo(edge.index, i, edge.label))
        #    else:
        #        raise ValueError(
        #            f"Target node {node.label} of edge {edge.label} already has a previous edge. This is not currently supported."
        #        )

    def __post_init__(self):

        for edge in self.edges:

            if not self.check_nodes_in_graph(edge.sources + edge.targets):
                raise ValueError(f"Edge {edge.label} has nodes not in hypergraph nodes")

            self.set_next_prev(edge)
