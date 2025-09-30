"""Module defining a Node in a hypergraph."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class Node:
    """A node in a hypergraph."""

    index: int
    label: str
    display_label: str = field(init=False)

    # For user-input validation purposes only
    prev: Optional[set[int]] = field(default=None, init=False)
    next: Optional[set[int]] = field(default=None, init=False)

    def __post_init__(self):
        self.display_label = f"{self.label}, {self.index}"
