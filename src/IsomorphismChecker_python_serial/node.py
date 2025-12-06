"""Module defining a Node in a hypergraph."""

from dataclasses import dataclass, field


@dataclass
class EdgeInfo:
    index: int
    port: int
    label: str


@dataclass(slots=True)
class Node:
    """A node in a hypergraph."""

    index: int
    label: str
    display_label: str = field(init=False)

    prev: list[EdgeInfo] = field(default_factory=list, init=False)
    next: list[EdgeInfo] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.display_label = f"{self.label}, {self.index}"
