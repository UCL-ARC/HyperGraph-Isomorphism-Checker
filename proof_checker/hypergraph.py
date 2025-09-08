"""Module to define hypergraphs and related structures."""

from dataclasses import dataclass
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
