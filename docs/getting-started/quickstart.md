# Quick Start

Graph Input example.

## Basic Graph Creation

### Creating Nodes

```python
from IsomorphismChecker_python_serial.node import Node

# Create input node (no previous connections)
input_node = Node(index=0, label="X")

# Create intermediate node
intermediate = Node(index=1, label="Y")

# Create output node (no next connections)
output_node = Node(index=2, label="Z")
```

### Creating a Simple Graph

```python
from IsomorphismChecker_python_serial.graph import Graph
from IsomorphismChecker_python_serial.edge import Edge

# Create edges
edge1 = Edge(source=input_node, target=intermediate, label="f")
edge2 = Edge(source=intermediate, target=output_node, label="g")

# Create graph
graph = Graph(
    nodes=[input_node, intermediate, output_node],
    edges=[edge1, edge2]
)

# Validate the graph
if graph.is_valid():
    print("✓ Graph is valid!")
else:
        print(f"✗ Graph is invalid!")
```

## Working with Hypergraphs

### Creating a Hypergraph

```python
from IsomorphismChecker_python_serial.hypergraph import OpenHypergraph
from IsomorphismChecker_python_serial.hyperedge import HyperEdge

# Create nodes
nodes = [
    Node(index=0, label="a"),
    Node(index=1, label="b"),
    Node(index=2, label="c")
]

# Create hyperedge (multiple sources/targets)
hyperedge = HyperEdge(
    index=0,
    label="F",
    sources=[nodes[0], nodes[1]],  # Multiple sources
    targets=[nodes[2]]              # Single target
)

# Create hypergraph
hypergraph = OpenHypergraph(
    nodes=nodes,
    edges=[hyperedge]
)
```

### Loading from JSON

```python
from IsomorphismChecker_python_serial.hypergraph import create_hypergraph

# Load from JSON file
with open("path/to/hypergraph.json", "r") as f:
    data = json.load(f)

hypergraph = create_hypergraph(data)
```

## Visualization

### Rendering Graphs

```python
from IsomorphismChecker_python_serial.diagram import Diagram

# Create diagram from graph
diagram = Diagram(openHyperGraph=graph)

# Render to file
diagram.render("output_graph")  # Creates output_graph.pdf
```

## Validation

### Graph Validation

```python
# Quick validation
is_valid = graph.is_valid()
```

### Signature Validation

```python
from IsomorphismChecker_python_serial.signature import Signature

# Create and validate signature
sig = Signature("a-b c-d")

# Check if valid
if sig.is_valid():
    print(f"Signature has {sig.num_edges()} edges")
```

## Next Steps

- Explore the [API Reference](../api/node.md) for detailed documentation
- Check out [Examples](../examples.md) for more complex use cases
- Read the [Contributing Guide](../contributing.md) to contribute to the project
