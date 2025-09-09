"""Module to define hypergraphs and related structures."""

from dataclasses import dataclass, field
from typing import NamedTuple, Optional


@dataclass(slots=True)
class Node:
    """A node in a hypergraph."""

    label: str
    prev: Optional[int] = field(default=None)
    next: Optional[int] = field(default=None)

    def __post_init__(self):
        if self.prev is not None and self.prev < 0:
            raise ValueError("Previous node index must be non-negative or None.")
        if self.next is not None and self.next < 0:
            raise ValueError("Next node index must be non-negative or None.")

    @property
    def is_isolated(self) -> bool:
        return self.prev is None and self.next is None

    @property
    def is_input(self) -> bool:
        return self.prev is None and self.next is not None

    @property
    def is_output(self) -> bool:
        return self.next is None and self.prev is not None


class Signature(NamedTuple):
    """A signature for a hypergraph, defining the types of nodes and edges."""

    sources: list[Node]
    targets: list[Node]


@dataclass(slots=True)
class HyperEdge:
    """A hyperedge in a hypergraph."""

    sources: list[Node]
    targets: list[Node]
    label: str

    @property
    def signature(self) -> Signature:
        return Signature(sources=self.sources, targets=self.targets)


@dataclass
class OpenHypergraph:
    """An open hypergraph with input and output nodes."""

    nodes: list[Node] = field(default_factory=list)
    edges: list[HyperEdge] = field(default_factory=list)

    @property
    def input_nodes(self) -> list[Node]:
        return [node for node in self.nodes if node.is_input]

    @property
    def output_nodes(self) -> list[Node]:
        return [node for node in self.nodes if node.is_output]

    @property
    def isolated_nodes(self) -> list[Node]:
        return [node for node in self.nodes if node.is_isolated]

    def is_valid(self) -> bool:

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

    def add_node(self, node: Node):
        self.nodes.append(node)

    def add_edge(self, edge: HyperEdge):
        self.edges.append(edge)

    def add_nodes(self, nodes: list[Node]):
        self.nodes.extend(nodes)

    def add_edges(self, edges: list[HyperEdge]):
        self.edges.extend(edges)
