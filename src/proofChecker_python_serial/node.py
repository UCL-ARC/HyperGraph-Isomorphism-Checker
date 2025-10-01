"""Module defining a Node in a hypergraph."""

from dataclasses import dataclass, field
from typing import Optional

from proofChecker_python_serial.validation import validate_common_fields


@dataclass(slots=True)
class Node:
    """A node in a hypergraph."""

    index: int
    label: str
    display_label: str = field(init=False)
    id: str = field(default="", init=False)

    # For user-input validation purposes only
    prev: Optional[set[int]] = field(default=None, init=False)
    next: Optional[set[int]] = field(default=None, init=False)

    def __post_init__(self):

        validate_common_fields(self, allow_digits_in_label=False)

        self.display_label = f"{self.label}, {self.index}"
        self.id = f"{self.index}{self.label}"
