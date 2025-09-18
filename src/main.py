from proofChecker_python_serial.hyperedge import HyperEdge
from proofChecker_python_serial.node import Node
from proofChecker_python_serial.hypergraph import OpenHypergraph
from proofChecker_python_serial.diagram import Diagram


def main():
    print("Hello from data-parallel-proof-checker-1368!")

    # Example usage
    n1 = Node(index=0, label="a")
    n2 = Node(index=1, label="b")
    e1 = HyperEdge(sources=[n1], targets=[n2], label="F", index=0)

    hypergraph = OpenHypergraph(nodes=[n1, n2], edges=[e1])
    diagram = Diagram(openHyperGraph=hypergraph)

    diagram.render("example_hypergraph")
    source = diagram.source()
    print("Diagram source:")
    print(source)


if __name__ == "__main__":
    main()
