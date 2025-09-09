"""Module to define hypergraphs and related structures."""

from dataclasses import dataclass, field
from typing import NamedTuple, Optional


@dataclass
class Node:
    """A node in a hypergraph."""

    label: str
    prev: Optional[int] = None
    next: Optional[int] = None

    def __post_init__(self):
        if self.prev is not None and self.prev < 0:
            raise ValueError("Previous node index must be non-negative or None.")
        if self.next is not None and self.next < 0:
            raise ValueError("Next node index must be non-negative or None.")


class Signature(NamedTuple):
    """A signature for a hypergraph, defining the types of nodes and edges."""

    sources: list[Node]
    targets: list[Node]


@dataclass
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
    input_nodes: list[Node] = field(default_factory=list)
    output_nodes: list[Node] = field(default_factory=list)

    def add_node(self, node: Node):
        self.nodes.append(node)

    def add_edge(self, edge: HyperEdge):
        self.edges.append(edge)

    def add_nodes(self, nodes: list[Node]):
        self.nodes.extend(nodes)

    def add_edges(self, edges: list[HyperEdge]):
        self.edges.extend(edges)
