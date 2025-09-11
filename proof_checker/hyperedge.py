"""Module defining HyperEdge and Signature in a hypergraph."""

from dataclasses import dataclass
from typing import NamedTuple

from proof_checker.node import Node


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
