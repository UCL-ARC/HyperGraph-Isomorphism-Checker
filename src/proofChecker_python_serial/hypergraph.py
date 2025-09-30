"""Module to define hypergraphs and related structures."""

from dataclasses import dataclass, field

from proofChecker_python_serial.hyperedge import HyperEdge
from proofChecker_python_serial.node import Node


@dataclass(slots=True)
class OpenHypergraph:
    """An open hypergraph with input and output nodes."""

    nodes: list[Node] = field(default_factory=list)
    edges: list[HyperEdge] = field(default_factory=list)

    input_nodes: list[Node] = field(default_factory=list, init=False)
    output_nodes: list[Node] = field(default_factory=list, init=False)
    isolated_nodes: list[Node] = field(default_factory=list, init=False)

    def set_input_nodes(self) -> list[Node]:
        """Nodes with no incoming edges."""
        self.input_nodes = [node for node in self.nodes if node.prev is None]
        return self.input_nodes

    def set_output_nodes(self) -> list[Node]:
        """Nodes with no outgoing edges."""
        self.output_nodes = [node for node in self.nodes if node.next is None]
        return self.output_nodes

    def set_isolated_nodes(self) -> list[Node]:
        """Nodes with no incoming or outgoing edges."""
        self.isolated_nodes = [
            node for node in self.nodes if node.prev is None and node.next is None
        ]
        return self.isolated_nodes

    # TODO: Improve efficiency by caching results and invalidating on changes
    def is_valid(self) -> bool:
        """Check if the hypergraph is valid. For detailed error messages, use validate()."""
        if not self.nodes:
            return False

        if not self.edges:
            return False

        if not self.input_nodes:
            return False

        if not self.output_nodes:
            return False

        if self.isolated_nodes:
            return False

        for edge in self.edges:
            for node in edge.sources + edge.targets:
                if node not in self.nodes:
                    return False

        return True

    def validate(self) -> list[str]:
        """Return list of validation error messages. Empty list means valid."""
        errors: list[str] = []

        if not self.nodes:
            errors.append("Hypergraph must contain at least one node")

        if not self.edges:
            errors.append("Hypergraph must contain at least one edge")

        if not self.input_nodes:
            errors.append("Hypergraph must have at least one input node")

        if not self.output_nodes:
            errors.append("Hypergraph must have at least one output node")

        isolated = self.isolated_nodes
        if isolated:
            isolated_labels = [node.label for node in isolated]
            errors.append(f"Hypergraph contains isolated nodes: {isolated_labels}")

        # Check for nodes referenced in edges but not in the node list
        missing_nodes: list[Node] = []
        for edge in self.edges:
            for node in edge.sources + edge.targets:
                if node not in self.nodes and node not in missing_nodes:
                    missing_nodes.append(node)

        if missing_nodes:
            missing_labels = [node.label for node in missing_nodes]
            errors.append(f"Edges reference nodes not in hypergraph: {missing_labels}")

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

    def check_nodes_in_graph(self, nodes: list[Node]) -> bool:
        """Check if all nodes are in the hypergraph."""
        return all(node in self.nodes for node in nodes)

    @staticmethod
    def set_next_prev(edge: HyperEdge):
        """Set the next and previous edges for nodes based on edges in the hypergraph."""

        for node in edge.sources:
            if node.next is None:
                node.next = {edge.signature}
            else:
                raise ValueError(
                    f"Source node {node.label} of edge {edge.label} already has a next edge. This is not currently supported."
                )

        for node in edge.targets:
            if node.prev is None:
                node.prev = {edge.signature}
            else:
                raise ValueError(
                    f"Target node {node.label} of edge {edge.label} already has a previous edge. This is not currently supported."
                )

    def __post_init__(self):

        for edge in self.edges:

            if not self.check_nodes_in_graph(edge.sources + edge.targets):
                raise ValueError(f"Edge {edge.label} has nodes not in hypergraph nodes")

            self.set_next_prev(edge)

        self.set_input_nodes()
        self.set_output_nodes()
        self.set_isolated_nodes()
