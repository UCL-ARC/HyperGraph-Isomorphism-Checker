"""Module defining an edge in a graph."""

from dataclasses import dataclass
from typing import NamedTuple

from proofChecker_python_serial.node import Node


class EdgeSignature(NamedTuple):
    """A signature for a graph edge, defining the types of nodes."""

    sources: Node
    targets: Node


@dataclass(slots=True)
class Edge:
    """An edge in a graph."""

    source: Node
    target: Node
    label: str

    @property
    def signature(self) -> EdgeSignature:
        return EdgeSignature(sources=self.source, targets=self.target)
