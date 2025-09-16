from proof_checker.hyperedge import HyperEdge
from proof_checker.node import Node
from proof_checker.hypergraph import OpenHypergraph
from proof_checker.diagram import Diagram


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
