from proofChecker_python_serial.hyperedge import HyperEdge
from proofChecker_python_serial.node import Node
from proofChecker_python_serial.hypergraph import OpenHypergraph
from proofChecker_python_serial.diagram import Diagram


def main():
    print("Hello from data-parallel-proof-checker-1368!")

    # Example usage
    n1 = Node(label="A", prev=None, next=0)
    n2 = Node(label="B", prev=0, next=None)
    e1 = HyperEdge(sources=[n1], targets=[n2], label="f")

    hypergraph = OpenHypergraph(nodes=[n1, n2], edges=[e1])
    diagram = Diagram(openHyperGraph=hypergraph)

    diagram.render("example_hypergraph")
    source = diagram.source()
    print("Diagram source:")
    print(source)


if __name__ == "__main__":
    main()
