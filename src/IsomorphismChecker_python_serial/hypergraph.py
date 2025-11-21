"""Module to define hypergraphs and related structures."""

import warnings

from dataclasses import dataclass, field

from IsomorphismChecker_python_serial.hyperedge import HyperEdge
from IsomorphismChecker_python_serial.node import Node, EdgeInfo


@dataclass
class SubGraph:
    nodes: list[int]
    edges: list[int]


@dataclass(slots=True)
class OpenHypergraph:
    """An open hypergraph with input and output nodes."""

    nodes: list[Node] = field(default_factory=list)
    edges: list[HyperEdge] = field(default_factory=list)

    input_nodes: list[int] = field(default_factory=list)
    output_nodes: list[int] = field(default_factory=list)

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

        for (i, v) in enumerate(edge.sources):
            node = self.nodes[v]
            if node.next is None:
                node.next = EdgeInfo(edge.index, i, edge.label)
            else:
                warnings.warn(
                    "Nodes with multiple outgoing edges are not currently fully supported. Use with caution.",
                    UserWarning,
                )

        for (i, v) in enumerate(edge.targets):
            node = self.nodes[v]
            if node.prev is None:
                node.prev = EdgeInfo(edge.index, i, edge.label)
            else:
                warnings.warn(
                    "Nodes with multiple incoming edges are not currently fully supported. Use with caution.",
                    UserWarning,
                )

    def __post_init__(self):

        for edge in self.edges:

            if not self.check_nodes_in_graph(edge.sources + edge.targets):
                raise ValueError(f"Edge {edge.label} has nodes not in hypergraph nodes")

            self.set_next_prev(edge)
