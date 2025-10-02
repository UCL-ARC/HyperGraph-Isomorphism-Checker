"""Main module for the data-parallel proof checker."""

from proofChecker_python_serial.hypergraph import OpenHypergraph
from proofChecker_python_serial.diagram import Diagram
from proofChecker_python_serial.isomorphisms import MC_isomorphism, permute_graph
from proofChecker_python_serial.graph_utils import create_hypergraph


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


def print_graph(G1):
    print(
        [(n.index, n.label) for n in G1.nodes],
        [(e.label, e.sources, e.targets) for e in G1.edges],
        G1.input_nodes,
        G1.output_nodes,
    )


def main():

    G1 = create_hypergraph("Graph1.json")
    # draw_graph(G1)

    (p, G2) = permute_graph(G1)
    draw_graph(G2)

    print(p)

    print_graph(G1)
    print_graph(G2)

    isomorphic, p_nodes, p_edges = MC_isomorphism(G1, G2)

    print(
        f"G1 and G2 isomorphic = {'True' if isomorphic else 'False', p_nodes, p_edges}"
    )


if __name__ == "__main__":
    main()
