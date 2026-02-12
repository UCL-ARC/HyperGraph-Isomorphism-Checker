from IsomorphismChecker_python_serial.hypergraph import OpenHypergraph
from IsomorphismChecker_python_serial.util import MappingMode


class ColourMap:
    """
    Holds colour information for a list of elements (node or edges)
    The `colouring` maps indices to colours
    The `colour_map` maps colours to lists of indices with that colour
    """

    def __init__(self, size):
        self.colouring: list[int] = [-1] * size
        self.colour_map: dict[int, set[int]] = {}
        self.update_map: dict[int, set[int]] = {}

    def mergeUpdates(self):
        for (colour, group) in self.update_map.items():
            self.colour_map[colour] = group
        self.update_map.clear()


class Colouring:
    def __init__(self, g: OpenHypergraph):
        self.g1 = g
        self.colour = 0
        self.n_nodes = len(g.nodes)
        self.n_edges = len(g.edges)
        self.node_colouring = ColourMap(self.n_nodes)
        self.edge_colouring = ColourMap(self.n_edges)

    def get_map(self, mode: MappingMode):
        if mode == MappingMode.NODE:
            return self.node_colouring
        else:
            return self.edge_colouring

    def set_colour(self, v, c, mode: MappingMode):
        colouring = self.get_map(mode)
        colouring.colouring[v] = c
        if c in colouring.colour_map.keys():
            colouring.colour_map[c].add(v)
        else:
            colouring.colour_map[c] = set([v])

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
