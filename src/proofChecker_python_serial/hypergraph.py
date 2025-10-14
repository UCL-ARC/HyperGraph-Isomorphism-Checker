"""Module to define hypergraphs and related structures."""

from dataclasses import dataclass, field

from proofChecker_python_serial.hyperedge import HyperEdge
from proofChecker_python_serial.node import Node


@dataclass(slots=True)
class OpenHypergraph:
    """An open hypergraph with input and output nodes."""

    nodes: list[Node] = field(default_factory=list)
    edges: list[HyperEdge] = field(default_factory=list)

    input_nodes: list[int] = field(default_factory=list)
    output_nodes: list[int] = field(default_factory=list)

    # TODO: Improve efficiency by caching results and invalidating on changes
    def is_valid(self) -> bool:
        """Check if the hypergraph is valid. For detailed error messages, use validate()."""
        if not self.nodes:
            return False

        if not self.edges:
            return False

        for edge in self.edges:
            for node in edge.sources + edge.targets:
                if node >= len(self.nodes):
                    return False

        return True

    def validate(self) -> list[str]:
        """Return list of validation error messages. Empty list means valid."""
        errors: list[str] = []

        if not self.nodes:
            errors.append("Hypergraph must contain at least one node")

        if not self.edges:
            errors.append("Hypergraph must contain at least one edge")

        return errors

    def add_node(self, node: Node):
        """Add a node to the hypergraph."""
        self.nodes.append(node)

    def add_edge(self, edge: HyperEdge):
        """Add a hyperedge to the hypergraph."""
        self.edges.append(edge)

    def add_nodes(self, nodes: list[Node]):
        """Add multiple nodes to the hypergraph."""
        self.nodes.extend(nodes)

    def add_edges(self, edges: list[HyperEdge]):
        """Add multiple edges to the hypergraph."""
        self.edges.extend(edges)

    def check_nodes_in_graph(self, nodes) -> bool:
        """Check if all nodes are in the hypergraph."""
        return all(node < len(self.nodes) for node in nodes)

    def set_next_prev(self, edge: HyperEdge):
        """Set the next and previous edges for nodes based on edges in the hypergraph."""

        for v in edge.sources:
            print(v)
            node = self.nodes[v]
            if node.next is None:
                node.next = edge.index
            else:
                raise ValueError(
                    f"Source node {node.label} of edge {edge.label} already has a next edge. This is not currently supported."
                )

        for v in edge.targets:
            node = self.nodes[v]
            if node.prev is None:
                node.prev = edge.index
            else:
                raise ValueError(
                    f"Target node {node.label} of edge {edge.label} already has a previous edge. This is not currently supported."
                )

    def __post_init__(self):

        for edge in self.edges:

            if not self.check_nodes_in_graph(edge.sources + edge.targets):
                raise ValueError(f"Edge {edge.label} has nodes not in hypergraph nodes")

            self.set_next_prev(edge)
