"""Module defining a Node in a hypergraph."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class Node:
    """A node in a hypergraph."""

    label: str
    prev: Optional[int] = field(default=None)
    next: Optional[int] = field(default=None)

    def __post_init__(self):
        if self.prev is not None and self.prev < 0:
            raise ValueError("Previous node index must be non-negative or None.")
        if self.next is not None and self.next < 0:
            raise ValueError("Next node index must be non-negative or None.")

    @property
    def is_isolated(self) -> bool:
        return self.prev is None and self.next is None

    @property
    def is_input(self) -> bool:
        return self.prev is None and self.next is not None

    @property
    def is_output(self) -> bool:
        return self.next is None and self.prev is not None
