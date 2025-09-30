"""Module defining HyperEdge and Signature in a hypergraph."""

from dataclasses import dataclass, field

from proofChecker_python_serial.node import Node


@dataclass(slots=True)
class HyperEdge:
    """A hyperedge in a hypergraph."""

    sources: list[Node]
    targets: list[Node]
    label: str
    index: int

    signature: int = field(init=False)
    display_label: str = field(init=False)

    def create_display_label(self) -> str:

        sorted_sources = sorted(self.sources, key=lambda node: node.index)
        sorted_targets = sorted(self.targets, key=lambda node: node.index)

        source_labels = ",".join(node.label for node in sorted_sources)
        target_labels = ",".join(node.label for node in sorted_targets)

        return f"{self.label}, {self.index} ({source_labels})->({target_labels})"

    def create_signature(self) -> int:

        source_node_index = [node.index for node in self.sources]
        target_node_index = [node.index for node in self.targets]

        return self.create_hash(self.index, source_node_index, target_node_index)

    def __post_init__(self):

        self.signature = self.create_signature()
        self.display_label = self.create_display_label()

    # TODO: Use a better hash function if needed: https://arxiv.org/pdf/1611.00029
    @staticmethod
    def create_hash(
        index: int, source_node_index: list[int], target_node_index: list[int]
    ) -> int:
        """Create a simple hash for the hyperedge."""

        string = f"{index}:{source_node_index}->{target_node_index}"
        return int(string.encode().hex(), 16) % (10**8)
