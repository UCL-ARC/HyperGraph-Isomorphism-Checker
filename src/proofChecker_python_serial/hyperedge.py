"""Module defining HyperEdge and Signature in a hypergraph."""

import hashlib
from dataclasses import dataclass, field

from proofChecker_python_serial.node import Node


@dataclass(slots=True)
class HyperEdge:
    """A hyperedge in a hypergraph."""

    sources: list[Node]
    targets: list[Node]
    label: str
    index: int

    signature: str = field(default="")
    signature_hash: int = field(default=0, init=False)

    def __post_init__(self):
        if self.signature == "":

            sorted_sources = sorted(self.sources, key=lambda node: node.index)
            sorted_targets = sorted(self.targets, key=lambda node: node.index)

            source_labels = ",".join(node.label for node in sorted_sources)
            target_labels = ",".join(node.label for node in sorted_targets)

            self.signature = (
                f"{self.label}, {self.index} ({source_labels})->({target_labels})"
            )

        self.signature_hash = self.create_hash(self.signature)

    # TODO: Use a better hash function if needed: https://arxiv.org/pdf/1611.00029
    @staticmethod
    def create_hash(signature: str) -> int:
        """Create a simple hash for the signature string."""
        return int(hashlib.md5(signature.encode()).hexdigest(), 16)
