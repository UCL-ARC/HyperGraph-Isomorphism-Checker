"""Main module for the data-parallel proof checker."""

from proofChecker_python_serial.hypergraph import OpenHypergraph, create_hypergraph
from proofChecker_python_serial.diagram import draw_graph


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
