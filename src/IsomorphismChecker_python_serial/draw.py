from IsomorphismChecker_python_serial.hypergraph import OpenHypergraph
from IsomorphismChecker_python_serial.diagram import Diagram


def draw_graph(
    hypergraph: OpenHypergraph,
    filename: str = "hypergraph_diagram",
    verbose: bool = False,
) -> None:
    """Demonstrate hypergraph creation and rendering."""

    diagram = Diagram(openHyperGraph=hypergraph)
    diagram.render(filename)
    source = diagram.source()
    if verbose:
        print("Diagram source:")
        print(source)


def print_graph(G1: OpenHypergraph):
    print(
        [(n.index, n.label) for n in G1.nodes],
        [(e.label, e.sources, e.targets) for e in G1.edges],
        G1.input_nodes,
        G1.output_nodes,
    )
