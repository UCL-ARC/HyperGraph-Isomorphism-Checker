"""Multiple signature comparison functions."""

from dataclasses import dataclass, field

from proofChecker_python_serial.signature import Signature


@dataclass(slots=True)
class CompareSignatures:
    """Class to compare two signatures."""

    signature1: Signature
    signature2: Signature
    are_equal: bool = field(init=False)
    details: str = field(init=False)

    def are_equal_lengths(self) -> bool:
        """Check if the two signatures have equal lengths."""
        return self.signature1.signature_length == self.signature2.signature_length

    def are_equal_num_edges(self) -> bool:
        """Check if the two signatures have equal number of edges."""
        return self.signature1.num_edges == self.signature2.num_edges

    def are_equal_num_sources_targets(self) -> bool:
        """Check if the two signatures have equal number of sources and targets for each edge."""
        return sorted(self.signature1.num_sources_and_targets_all_edges) == sorted(
            self.signature2.num_sources_and_targets_all_edges
        )

    def are_equal_num_sources_all_edges(self) -> bool:
        """Check if the two signatures have equal number of sources for each edge."""
        return sorted(self.signature1.num_sources_all_edges) == sorted(
            self.signature2.num_sources_all_edges
        )

    def are_equal_num_targets_all_edges(self) -> bool:
        """Check if the two signatures have equal number of targets for each edge."""
        return sorted(self.signature1.num_targets_all_edges) == sorted(
            self.signature2.num_targets_all_edges
        )

    def __post_init__(self):

        details_list = []

        if not self.are_equal_lengths():
            details_list.append(
                f"Signatures have different lengths: "
                f"{self.signature1.signature_length} vs {self.signature2.signature_length}"
            )

        if not self.are_equal_num_edges():
            details_list.append(
                f"Signatures have different number of edges: "
                f"{self.signature1.num_edges} vs {self.signature2.num_edges}"
            )

        if not self.are_equal_num_sources_targets():
            details_list.append(
                f"Signatures have different number of sources and targets: "
                f"{self.signature1.num_sources_and_targets_all_edges} vs {self.signature2.num_sources_and_targets_all_edges}"
            )

        if not self.are_equal_num_sources_all_edges():
            details_list.append(
                f"Signatures have different number of sources for edges: "
                f"{self.signature1.num_sources_all_edges} vs {self.signature2.num_sources_all_edges}"
            )

        if not self.are_equal_num_targets_all_edges():
            details_list.append(
                f"Signatures have different number of targets for edges: "
                f"{self.signature1.num_targets_all_edges} vs {self.signature2.num_targets_all_edges}"
            )

        if details_list:
            self.details = "\n".join(details_list)
            self.are_equal = False
        else:
            self.details = "Signatures are equal."
            self.are_equal = True
