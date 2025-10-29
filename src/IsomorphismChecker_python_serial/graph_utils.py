from IsomorphismChecker_python_serial.hypergraph import OpenHypergraph, HyperEdge, Node
import json
from pathlib import Path


def create_hypergraph(filepath: str | Path) -> OpenHypergraph:
    """Create a hypergraph from a JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    nodes = [
        Node(index=i, label=node["type_label"]) for i, node in enumerate(data["nodes"])
    ]

    edges = [
        HyperEdge(
            sources=[src for src in edge["source_nodes"]],
            targets=[tgt for tgt in edge["target_nodes"]],
            label=edge["type_label"],
            index=i,
        )
        for i, edge in enumerate(data["hyperedges"])
    ]
    inputs = data["Inputs"]
    outputs = data["Outputs"]
    hypergraph = OpenHypergraph(
        nodes=nodes, edges=edges, input_nodes=inputs, output_nodes=outputs
    )
    return hypergraph
