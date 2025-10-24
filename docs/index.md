# HyperGraph Isomorphism Checker

Welcome to the documentation for the Hypergraph Isomorphism Checker (HgIC) project!

## Overview

The current version is a serial Python library for directed hypergraphs.

## Key Features

- 🔗 **Graph Structures**: Support for directed hypergraphs/graphs
- ✅ **Validation**: Comprehensive validation of graph structures and signatures
- 🎨 **Visualization**: Built-in support for rendering graphs using Graphviz
- 🧪 **Well-Tested**: Extensive test coverage with pytest
- 🚀 **Type-Safe**: Full type hints for better IDE support and code quality

## Use Case

```python
from IsomorphismChecker_python_serial.node import Node
from IsomorphismChecker_python_serial.edge import Edge
from IsomorphismChecker_python_serial.graph import Graph

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

Check out the [Installation Guide](getting-started/installation.md) to get started with the HgIC.

## Project Structure

```
src/IsomorphismChecker_python_serial/
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


## Table of Contents

- **Introduction**
    - [Overview](#overview)
    - [Key Features](#key-features)
    - [Use Case](#use-case)
- **Getting Started**
    - [Installation](getting-started/installation.md)
    - [Quick Start](getting-started/quickstart.md)
    - [Algorithm](getting-started/algorithm.md)
- **API Reference**
    - [Node](api/node.md)
    - [Hyperedge](api/hyperedge.md)
    - [Hypergraph](api/hypergraph.md)
    - [Diagram](api/diagram.md)
    - [Drawing](api/draw.md)
    - [Graph Utilities](api/graph_utils.md)
    - [Isomorphism](api/isomorphisms.md)
    - [Validation](api/validation.md)
- [Contributing](contributing.md)
