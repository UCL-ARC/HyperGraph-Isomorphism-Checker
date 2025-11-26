"""Module to generate random hypergraphs on demand."""

import json
import os
import time
from typing import Any
import networkx as nx
import random
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def generate_random_hypergraph(
    num_nodes: int,
    num_edges: int,
    num_inputs: int,
    num_outputs: int,
    seed: int = 42,
    wire_labels: list[str] = ["wire"],
    box_labels: list[str] = ["box"],
) -> nx.DiGraph:
    """Generates a random hypergraph represented as a bipartite graph.

    Args:
        num_nodes (int): Number of nodes in the hypergraph.
        num_edges (int): Number of hyperedges in the hypergraph.
        num_inputs (int): Number of input nodes in the hypergraph.
        num_outputs (int): Number of output nodes in the hypergraph.
        seed (int, optional): Seed for random number generator. Defaults to None.

    Returns:
        networkx.Graph: A bipartite graph representing the hypergraph.
    """

    if seed is not None:
        random.seed(seed)

    graph = nx.DiGraph()

    wire_labels = [random.choice(wire_labels) for _ in range(num_nodes)]
    box_labels = [random.choice(box_labels) for _ in range(num_edges)]

    wires = [f"{label}_{i}" for i, label in enumerate(wire_labels)]
    boxes = [f"{label}_{j}" for j, label in enumerate(box_labels)]
    graph.add_nodes_from(wires, bipartite=0, input=False, output=False)
    graph.add_nodes_from(boxes, bipartite=1)

    input_wires = wires[:num_inputs]
    output_wires = wires[-num_outputs:]
    internal_wires = wires[num_inputs : num_nodes - num_outputs]

    for wire in input_wires:
        graph.nodes[wire]["input"] = True
        box = random.choice(boxes)
        in_degree = graph.in_degree(box)
        logger.debug(
            f"Connecting input wire {wire} to box {box} with in_degree {in_degree}"
        )
        graph.add_edge(wire, box, port=in_degree)
        logger.debug(f"Added edge from {wire} to {box} with port {in_degree}")

    for wire in output_wires:
        graph.nodes[wire]["output"] = True
        box = random.choice(boxes)
        out_degree = graph.out_degree(box)
        logger.debug(
            f"Connecting output wire {wire} to box {box} with out_degree {out_degree}"
        )
        graph.add_edge(box, wire, port=out_degree)
        logger.debug(f"Added edge from {box} to {wire} with port {out_degree}")

    for wire in internal_wires:
        edge1 = random.choice(boxes)
        in_degree = graph.in_degree(edge1)
        logger.debug(
            f"Connecting internal wire {wire} to box {edge1} with in_degree {in_degree}"
        )
        graph.add_edge(wire, edge1, port=in_degree)
        logger.debug(f"Added edge from {wire} to {edge1} with port {in_degree}")
        edge2 = random.choice(boxes)
        out_degree = graph.out_degree(edge2)
        logger.debug(
            f"Connecting internal wire {wire} to box {edge2} with out_degree {out_degree}"
        )
        graph.add_edge(edge2, wire, port=out_degree)

        logger.debug(f"Added edge from {edge2} to {wire} with port {out_degree}")

    for node, data in graph.nodes(data=True):
        if data["bipartite"] == 0:
            print(f"Node: {node}, Data: {data}, Degree: {graph.degree(node)}")  # type: ignore

    exit()

    return graph


def graph_to_json_serializable(
    graph: nx.DiGraph,
    file_name: str = "random_hypergraph",
    directory: str = "trial_graphs",
) -> dict[str, Any]:
    """Converts a NetworkX graph to a JSON-serializable dictionary.

    Args:
        graph (networkx.Graph): The input graph.

    Returns:
        dict: A JSON-serializable representation of the graph.
    """

    data: dict[str, Any] = {}
    data["graph_name"] = "random_hypergraph"
    data["comment"] = "This is a randomly generated hypergraph."

    nodes = [
        {"type_label": node.split("_")[0]}
        for node, data in graph.nodes(data=True)
        if data["bipartite"] == 0
    ]

    hyperedges = [
        node for node, data in graph.nodes(data=True) if data["bipartite"] == 1
    ]

    data["hyperedges"] = []
    for hyperedge in hyperedges:
        source_nodes = [int(i.split("_")[1]) for i, j in graph.in_edges(hyperedge)]
        target_nodes = [int(j.split("_")[1]) for i, j in graph.out_edges(hyperedge)]

        edge_dict = {
            "type_label": hyperedge.split("_")[0],
            "source_nodes": source_nodes,
            "target_nodes": target_nodes,
        }

        # data.setdefault("hyperedges", []).append(edge_dict)
        data["hyperedges"].append(edge_dict)

    data["nodes"] = nodes

    input_nodes = [
        int(node.split("_")[1])
        for node, data in graph.nodes(data=True)
        if data["bipartite"] == 0 and data["input"]
    ]

    output_nodes = [
        int(node.split("_")[1])
        for node, data in graph.nodes(data=True)
        if data["bipartite"] == 0 and data["output"]
    ]

    data["Inputs"] = input_nodes
    data["Outputs"] = output_nodes

    os.makedirs(directory, exist_ok=True)

    with open(f"{directory}/{file_name}.json", "w") as f:
        json.dump(data, f, indent=4)

    return data


if __name__ == "__main__":

    initial_time = time.time()
    hg = generate_random_hypergraph(200, 55, 20, 20, seed=42)
    final_time = time.time()

    logger.debug("Generated hypergraph edges:")
    logger.debug(hg.edges)
    logger.debug("Generated hypergraph nodes:")
    logger.debug(hg.nodes)
    for node, data in hg.nodes(data=True):
        if data["bipartite"] == 0:
            logger.debug(f"Node: {node}, Data: {data}, Degree: {hg.degree(node)}")  # type: ignore

    for edge in hg.edges:
        logger.debug(f"Edge: {edge}")

    logger.debug(
        f"Time taken to generate hypergraph: {(final_time - initial_time) * 1000} milliseconds"
    )

    json_serializable_hg = graph_to_json_serializable(hg)
    logger.debug("JSON-serializable hypergraph:")
    logger.debug(json_serializable_hg)
    logger.debug("Done.")
