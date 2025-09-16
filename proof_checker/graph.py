"""Module to define graphs and related structures."""

from dataclasses import dataclass, field

from proof_checker.edge import Edge
from proof_checker.node import Node


@dataclass
class Graph:
    """A graph with single-source, single-target edges."""

    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    @property
    def input_nodes(self) -> list[Node]:
        """Nodes with no incoming edges."""
        return [node for node in self.nodes if node.is_input]

    @property
    def output_nodes(self) -> list[Node]:
        """Nodes with no outgoing edges."""
        return [node for node in self.nodes if node.is_output]

    @property
    def isolated_nodes(self) -> list[Node]:
        """Nodes with no incoming or outgoing edges."""
        return [node for node in self.nodes if node.is_isolated]

    def is_valid(self) -> bool:
        """Check if the graph is valid. For detailed error messages, use validate()."""
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

        # Check that all edge endpoints are in the node list
        for edge in self.edges:
            if edge.source not in self.nodes or edge.target not in self.nodes:
                return False

        return True

    def validate(self) -> list[str]:
        """Return list of validation error messages. Empty list means valid."""
        errors: list[str] = []

        if not self.nodes:
            errors.append("Graph must contain at least one node")

        if not self.edges:
            errors.append("Graph must contain at least one edge")

        if not self.input_nodes:
            errors.append("Graph must have at least one input node")

        if not self.output_nodes:
            errors.append("Graph must have at least one output node")

        isolated = self.isolated_nodes
        if isolated:
            isolated_labels = [node.label for node in isolated]
            errors.append(f"Graph contains isolated nodes: {isolated_labels}")

        # Check for nodes referenced in edges but not in the node list
        missing_nodes: list[Node] = []
        for edge in self.edges:
            if edge.source not in self.nodes and edge.source not in missing_nodes:
                missing_nodes.append(edge.source)
            if edge.target not in self.nodes and edge.target not in missing_nodes:
                missing_nodes.append(edge.target)

        if missing_nodes:
            missing_labels = [node.label for node in missing_nodes]
            errors.append(f"Edges reference nodes not in graph: {missing_labels}")

        return errors

    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        if node not in self.nodes:
            self.nodes.append(node)

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)

    def add_nodes(self, nodes: list[Node]) -> None:
        """Add multiple nodes to the graph."""
        for node in nodes:
            self.add_node(node)

    def add_edges(self, edges: list[Edge]) -> None:
        """Add multiple edges to the graph."""
        self.edges.extend(edges)
