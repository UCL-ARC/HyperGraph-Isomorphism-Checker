"""Module for diagram representations of Hypergraphs."""

from dataclasses import dataclass, field
from enum import Enum

from graphviz import Digraph

from IsomorphismChecker_python_serial.hypergraph import OpenHypergraph
from IsomorphismChecker_python_serial.hyperedge import HyperEdge
from IsomorphismChecker_python_serial.node import Node
from IsomorphismChecker_python_serial.isomorphisms import Colouring


class ElementType(Enum):
    NODE = "node"
    EDGE = "edge"


class Orientation(Enum):
    TOP_TO_BOTTOM = "TB"
    LEFT_TO_RIGHT = "LR"


@dataclass(slots=True)
class Diagram:
    """A diagram representation of a hypergraph."""

    openHyperGraph: OpenHypergraph

    nodes: list[Node] = field(init=False)
    hyperEdges: list[HyperEdge] = field(init=False)
    graphRep: Digraph = field(init=False)
    orientation: Orientation = field(default=Orientation.TOP_TO_BOTTOM)
    highlighted_nodes: list[int] = field(default_factory=list)
    highlighted_edges: list[int] = field(default_factory=list)

    @staticmethod
    def diagram_label(label: str, index: int, type: ElementType, hash: int = -1) -> str:

        if type == ElementType.NODE:
            joiner = ";"
        elif type == ElementType.EDGE:
            joiner = ","
        else:
            raise ValueError("Type must be ElementType.NODE or ElementType.EDGE.")

        final = joiner.join([label, str(index)])

        if type == ElementType.EDGE:
            final = f"{final}\n\n{hash}"

        return final

    def drawArrows(self, hyperEdge: HyperEdge, edge_label: str, nodes: list[Node]):

        # Draw arrows from nodes to edges
        for s in hyperEdge.sources:
            self.graphRep.edge(
                self.diagram_label(self.nodes[s].label, s, ElementType.NODE),
                edge_label,
            )

        # Draw arrows from edges to nodes
        for t in hyperEdge.targets:
            self.graphRep.edge(
                edge_label,
                self.diagram_label(self.nodes[t].label, t, ElementType.NODE),
            )

    def drawGraph(self, colouring: Colouring | None = None):

        for node in self.nodes:
            node_label = self.diagram_label(
                node.label, self.nodes.index(node), ElementType.NODE
            )

            if node.index in self.highlighted_nodes:
                color = "red"
            else:
                color = "black"

            if colouring is not None:
                colour = "/set19/" + str(
                    colouring.node_colouring.colouring[node.index] + 1
                )
                self.graphRep.node(
                    node_label, shape="circle", style="filled", fillcolor=colour
                )
            else:
                self.graphRep.node(node_label, shape="circle", fontcolor=color)

        for hyperEdge in self.hyperEdges:
            edge_label = self.diagram_label(
                hyperEdge.label,
                self.hyperEdges.index(hyperEdge),
                ElementType.EDGE,
                hash=hyperEdge.signature,
            )

            if hyperEdge.index in self.highlighted_edges:
                color = "red"
            else:
                color = "black"
            if colouring is not None:
                colour = "/paired12/" + str(
                    colouring.edge_colouring.colouring[hyperEdge.index] + 1
                )
                self.graphRep.node(
                    edge_label, shape="box", style="filled", fillcolor=colour
                )
            else:
                self.graphRep.node(edge_label, shape="box", fontcolor=color)

            self.drawArrows(hyperEdge, edge_label, self.nodes)

        return self.graphRep

    def __post_init__(self):

        if not self.openHyperGraph.is_valid():
            raise ValueError("The provided OpenHypergraph is not valid.")

        self.nodes = self.openHyperGraph.nodes
        self.hyperEdges = self.openHyperGraph.edges

        self.graphRep = Digraph(format="png")
        self.drawGraph()

    def render(self, filename: str = "hypergraph_diagram") -> None:
        """Render the diagram to a file."""
        self.graphRep.render(filename, view=False, cleanup=True)

    def source(self) -> str:
        """Return the source of the diagram."""
        return self.graphRep.source
