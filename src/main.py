"""Main module for the data-parallel proof checker."""

import json

from proofChecker_python_serial.hyperedge import HyperEdge
from proofChecker_python_serial.node import Node
from proofChecker_python_serial.hypergraph import OpenHypergraph
from proofChecker_python_serial.diagram import Diagram


def main():
    """Main function to demonstrate hypergraph creation and rendering."""

    with open("InputFormat.json", "r") as f:
        data = json.load(f)
        print("Loaded hypergraph data:")
        print(data)

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

    diagram = Diagram(openHyperGraph=hypergraph)
    diagram.render("hypergraph_diagram")
    source = diagram.source()
    print("Diagram source:")
    print(source)


if __name__ == "__main__":
    main()
