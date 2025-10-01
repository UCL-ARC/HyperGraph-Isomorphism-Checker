"""Module to handle signatures of hypergraphs."""

from dataclasses import dataclass
import re

from proofChecker_python_serial.regex_check import matches_signature_pattern

# import re


@dataclass(slots=True)
class Signature:
    """A signature for a hypergraph."""

    signature: str

    def __post_init__(self):
        if not isinstance(self.signature, str):
            raise ValueError("Signature must be a string.")

        # Validate that the signature follows the expected pattern
        if not matches_signature_pattern(self.signature):
            raise ValueError(
                "Signature must follow the pattern: 'word (space word)*' "
                "where word is 'unit-group-group'"
            )

    @property
    def signature_elements(self) -> list[str]:
        """Return the elements of the signature split by spaces, hyphens, or underscores."""
        return re.split(r"[ \-_]", self.signature.strip())

    @property
    def signature_length(self) -> int:
        """Return the length of the signature string."""
        return len(self.signature_elements)

    @property
    def num_edges(self) -> int:
        """Return the number of edges in the signature."""
        return len(self.edges)

    @property
    def edges(self) -> list[str]:
        """Return the list of edge signatures."""
        return self.signature.split(" ")

    @property
    def node_signatures(self) -> list[str]:
        """Return the list of node signatures."""
        nodes = set()
        for edge in self.edges:
            _, sources, targets = edge.split("-")
            nodes.update(sources.split("_"))
            nodes.update(targets.split("_"))
        return list(nodes)

    @property
    def num_sources_all_edges(self) -> list[int]:
        """Return the list of number of sources for each edge."""

        num: list[int] = []
        for edge in self.edges:
            _, sources, _ = edge.split("-")
            num.append(len(sources.split("_")))
        return num

    @property
    def num_targets_all_edges(self) -> list[int]:
        """Return the list of number of targets for each edge."""

        num: list[int] = []
        for edge in self.edges:
            _, _, targets = edge.split("-")
            num.append(len(targets.split("_")))
        return num

    @property
    def num_sources_and_targets_all_edges(self) -> list[int]:
        """Return the list of (number of sources, number of targets) for each edge."""

        num: list[int] = []
        for edge in self.edges:
            num.append(edge.count("_") + 2)  # +2 for the two groups
        return num
