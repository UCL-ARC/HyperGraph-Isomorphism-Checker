"""Module to generate random hypergraphs on demand."""

import time
import networkx as nx
import random


def generate_random_hypergraph(
    num_nodes: int,
    num_edges: int,
    num_inputs: int,
    num_outputs: int,
    seed: int = 42,
    node_labels: list[str] = ["node"],
    edge_labels: list[str] = ["edge"],
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

    node_labels = [random.choice(node_labels) for _ in range(num_nodes)]
    edge_labels = [random.choice(edge_labels) for _ in range(num_edges)]

    hypergraph_nodes = [f"{label}_{i}" for i, label in enumerate(node_labels)]
    hypergraph_edges = [f"{label}_{j}" for j, label in enumerate(edge_labels)]
    graph.add_nodes_from(hypergraph_nodes, bipartite=0)
    graph.add_nodes_from(hypergraph_edges, bipartite=1)

    input_nodes = hypergraph_nodes[:num_inputs]
    output_nodes = hypergraph_nodes[-num_outputs:]
    internal_nodes = hypergraph_nodes[num_inputs : num_nodes - num_outputs]

    for node in input_nodes:
        edge = random.choice(hypergraph_edges)
        graph.add_edge(node, edge)

    for node in output_nodes:
        edge = random.choice(hypergraph_edges)
        graph.add_edge(edge, node)

    for node in internal_nodes:
        edge1 = random.choice(hypergraph_edges)
        graph.add_edge(node, edge1)
        edge2 = random.choice(hypergraph_edges)
        graph.add_edge(edge2, node)

    return graph


def graph_to_json_serializable(graph: nx.DiGraph) -> dict:
    """Converts a NetworkX graph to a JSON-serializable dictionary.

    Args:
        graph (networkx.Graph): The input graph.

    Returns:
        dict: A JSON-serializable representation of the graph.
    """

    data = {}
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

    for hyperedge in hyperedges:
        source_nodes = [int(i.split("_")[1]) for i, j in graph.in_edges(hyperedge)]
        target_nodes = [int(j.split("_")[1]) for i, j in graph.out_edges(hyperedge)]

        edge_dict = {
            "type_label": hyperedge.split("_")[0],
            "source_nodes": source_nodes,
            "target_nodes": target_nodes,
        }

        data.setdefault("edges", []).append(edge_dict)

    data["nodes"] = nodes
    # data["edges"] = edges

    return data


if __name__ == "__main__":

    initial_time = time.time()
    hg = generate_random_hypergraph(10_000, 1_000, 2, 2, seed=42)
    final_time = time.time()

    print("Generated hypergraph edges:")
    print(hg.edges)
    print("Generated hypergraph nodes:")
    print(hg.nodes)

    for node, data in hg.nodes(data=True):
        if data["bipartite"] == 0:
            print(f"Node: {node}, Data: {data}, Degree: {hg.degree(node)}")  # type: ignore

    for edge in hg.edges:
        print(f"Edge: {edge}")

    print(
        f"Time taken to generate hypergraph: {(final_time - initial_time) * 1000} milliseconds"
    )

    json_serializable_hg = graph_to_json_serializable(hg)
    print("JSON-serializable hypergraph:")
    print(json_serializable_hg["edges"])
    print("Done.")
