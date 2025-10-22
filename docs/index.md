# Data Parallel Proof Checker

Welcome to the documentation for the Data Parallel Proof Checker project!

## Overview

The Data Parallel Proof Checker is a Python library for working with hypergraphs, directed graphs, and proof verification. It provides a robust framework for representing and manipulating graph structures commonly used in formal verification and proof checking.

## Key Features

- 🔗 **Graph Structures**: Support for both simple directed graphs and hypergraphs
- ✅ **Validation**: Comprehensive validation of graph structures and signatures
- 🎨 **Visualization**: Built-in support for rendering graphs using Graphviz
- 🧪 **Well-Tested**: Extensive test coverage with pytest
- 🚀 **Type-Safe**: Full type hints for better IDE support and code quality

## Quick Example

```python
from proofChecker_python_serial.node import Node
from proofChecker_python_serial.edge import Edge
from proofChecker_python_serial.graph import Graph

# Create nodes
n1 = Node(index=0, label="A")
n2 = Node(index=1, label="B")

# Create edge
edge = Edge(source=n1, target=n2, label="f")

# Create graph
graph = Graph(nodes=[n1, n2], edges=[edge])

# Validate
if graph.is_valid():
    print("Graph is valid!")
```

## Getting Started

Check out the [Installation Guide](getting-started/installation.md) to get started with the Data Parallel Proof Checker.

## Project Structure

```
src/proofChecker_python_serial/
├── node.py              # Node definitions
├── edge.py              # Simple edge definitions
├── hyperedge.py         # Hyperedge definitions
├── graph.py             # Directed graph implementation
├── hypergraph.py        # Hypergraph implementation
├── signature.py         # Signature handling
├── diagram.py           # Graph visualization
├── validation.py        # Validation utilities
└── ...
```

## Contributing

We welcome contributions! See our [Contributing Guide](contributing.md) for details.
