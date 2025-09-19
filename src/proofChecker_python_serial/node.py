"""Module defining a Node in a hypergraph."""

from dataclasses import dataclass, field
from typing import Optional


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

        if not isinstance(self.index, int) or self.index < 0:
            raise ValueError("Index must be a non-negative integer.")

        if not isinstance(self.label, str):
            raise ValueError("Label must be a string.")

        if self.label == "":
            raise ValueError("Label must be a non-empty string.")

        if "0123456789" in self.label:
            raise ValueError("Label must not contain digits.")

        self.display_label = f"{self.label}, {self.index}"
        self.id = f"{self.label}{self.index}"
