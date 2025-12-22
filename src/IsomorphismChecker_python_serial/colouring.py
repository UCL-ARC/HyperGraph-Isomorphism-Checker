from dataclasses import dataclass, field
from IsomorphismChecker_python_serial.hypergraph import OpenHypergraph


@dataclass
class ColourMap:
    """
    Holds colour information for a list of elements (node or edges)
    The `colouring` maps indices to colours
    The `colour_map` maps colours to lists of indices with that colour
    """

    size: int
    colouring: list[int] = field(init=False)
    colour_map: dict[int, set[int]] = field(init=False)
    update_map: dict[int, set[int]] = field(init=False)

    def __post_init__(self):
        """Post-initialization to set up the colouring and maps."""
        if self.size < 0:
            raise ValueError("Size of ColourMap must be non-negative.")

        self.colouring: list[int] = [-1] * self.size
        self.colour_map: dict[int, set[int]] = {}
        self.update_map: dict[int, set[int]] = {}

    def mergeUpdates(self):
        for colour, group in self.update_map.items():
            self.colour_map[colour] = group
        self.update_map.clear()


@dataclass
class Colouring:
    """Class to manage the colouring of nodes and edges in a hypergraph."""

    graph: OpenHypergraph
    colour: int = field(default=0)
    n_nodes: int = field(init=False)
    n_edges: int = field(init=False)
    node_colouring: ColourMap = field(init=False)
    edge_colouring: ColourMap = field(init=False)

    def __post_init__(self):
        self.n_nodes = len(self.graph.nodes)
        self.n_edges = len(self.graph.edges)
        self.node_colouring = ColourMap(self.n_nodes)
        self.edge_colouring = ColourMap(self.n_edges)

    def get_new_colour(self):
        self.colour += 1
        return self.colour

    def check_uniqueness(self):
        nodes_unique, edges_unique = (True, -1), (True, -1)
        for colour, group in self.node_colouring.colour_map.items():
            if len(group) > 1:
                nodes_unique = (False, colour)
                break

        for colour, group in self.edge_colouring.colour_map.items():
            if len(group) > 1:
                edges_unique = (False, colour)
                break

        return nodes_unique, edges_unique
