"""Main module for the data-parallel proof checker."""

import json

from proofChecker_python_serial.hyperedge import HyperEdge
from proofChecker_python_serial.node import Node
from proofChecker_python_serial.hypergraph import OpenHypergraph
from proofChecker_python_serial.diagram import Diagram


def create_hypergraph(filepath: str) -> OpenHypergraph:
    """Create a hypergraph from a JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    nodes = [
        Node(index=i, label=node["type_label"]) for i, node in enumerate(data["nodes"])
    ]

    edges = [
        HyperEdge(
            sources=[nodes[src] for src in edge["source_nodes"]],
            targets=[nodes[tgt] for tgt in edge["target_nodes"]],
            label=edge["type_label"],
            index=i,
        )
        for i, edge in enumerate(data["hyperedges"])
    ]
    hypergraph = OpenHypergraph(nodes=nodes, edges=edges)
    return hypergraph


def draw_graph(hypergraph: OpenHypergraph):
    """Demonstrate hypergraph creation and rendering."""

    diagram = Diagram(openHyperGraph=hypergraph)
    diagram.render("hypergraph_diagram")
    source = diagram.source()
    print("Diagram source:")
    print(source)


def is_valid_dimensions(
    hypergraph1: OpenHypergraph, hypergraph2: OpenHypergraph
) -> bool:

    if len(hypergraph1.nodes) != len(hypergraph2.nodes) or len(
        hypergraph1.edges
    ) != len(hypergraph2.edges):
        return False

    return True


def main():

    hypergraph = create_hypergraph("InputFormat.json")
    draw_graph(hypergraph)

    hypergraph1 = create_hypergraph("tests/inputs/graph1.json")
    hypergraph2 = create_hypergraph("tests/inputs/graph2.json")

    validity = is_valid_dimensions(hypergraph1, hypergraph2)
    print(f"Input graphs have valid dimensions: {validity}")


if __name__ == "__main__":
    main()
