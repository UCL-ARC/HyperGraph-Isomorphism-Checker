"""Module defining HyperEdge and Signature in a hypergraph."""

import hashlib
from dataclasses import dataclass, field

from proofChecker_python_serial.node import Node


@dataclass(slots=True)
class HyperEdgeSignature:
    """A signature for a hypergraph, defining the types of nodes and edges."""

    sources: list[Node]
    targets: list[Node]

    signature: str = field(default="")
    signature_hash: int = field(default=0, init=False)

    def __post_init__(self):
        if self.signature == "":
            source_labels = ",".join(node.label for node in self.sources)
            target_labels = ",".join(node.label for node in self.targets)
            self.signature = f"({source_labels})->({target_labels})"

        self.signature_hash = self.create_hash(self.signature)

    @staticmethod
    def create_hash(signature: str) -> int:
        """Create a simple hash for the signature string."""
        return int(hashlib.md5(signature.encode()).hexdigest(), 16)


@dataclass(slots=True)
class HyperEdge:
    """A hyperedge in a hypergraph."""

    sources: list[Node]
    targets: list[Node]
    label: str

    @property
    def signature(self) -> HyperEdgeSignature:
        return HyperEdgeSignature(sources=self.sources, targets=self.targets)
