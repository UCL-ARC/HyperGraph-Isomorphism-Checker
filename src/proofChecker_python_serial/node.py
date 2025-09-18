"""Module defining a Node in a hypergraph."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class Node:
    """A node in a hypergraph."""

    index: int
    label: str

    # For user-input validation purposes only
    prev: Optional[int] = field(default=None, init=False)
    next: Optional[int] = field(default=None, init=False)
