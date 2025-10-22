# Graph Isomorphisms

This module provides functionality for checking graph isomorphisms between hypergraphs, particularly for monogamous, cartesian (MC) hypergraphs.

## Overview

Graph isomorphism checking determines whether two graphs are structurally identical, meaning there exists a bijection between their nodes and edges that preserves the graph structure.

## Classes

### Isomorphism

::: proofChecker_python_serial.isomorphisms.Isomorphism
    options:
      show_source: true
      members:
        - __init__
        - __post_init__
        - n_nodes
        - n_edges
        - update_mapping
        - update_mapping_list
        - check_edge_compatibility
        - explore_edges
        - traverse_from_nodes
        - check_MC_isomorphism

### MappingMode

::: proofChecker_python_serial.isomorphisms.MappingMode
    options:
      show_source: true

## Functions

### MC_isomorphism

Main function for checking monogamous, cartesian hypergraph isomorphism.

```python
from proofChecker_python_serial.isomorphisms import MC_isomorphism
from proofChecker_python_serial.hypergraph import OpenHypergraph

# Check if two hypergraphs are isomorphic
is_iso, node_mapping, edge_mapping = MC_isomorphism(graph1, graph2)

if is_iso:
    print(f"Graphs are isomorphic!")
    print(f"Node mapping: {node_mapping}")
    print(f"Edge mapping: {edge_mapping}")
else:
    print("Graphs are not isomorphic")
```

**Parameters:**
- `g1` (OpenHypergraph): First hypergraph
- `g2` (OpenHypergraph): Second hypergraph

**Returns:**
- `tuple[bool, list[int], list[int]]`:
  - Boolean indicating if graphs are isomorphic
  - Node mapping (permutation) if isomorphic, empty list otherwise
  - Edge mapping (permutation) if isomorphic, empty list otherwise

### permute_graph

Creates a randomly permuted version of a hypergraph for testing isomorphism algorithms.

```python
from proofChecker_python_serial.isomorphisms import permute_graph

# Create a permuted version of a graph
permutation, permuted_graph = permute_graph(original_graph)

# The permutation list shows how nodes were reordered
print(f"Permutation used: {permutation}")
```

**Parameters:**
- `g` (OpenHypergraph): Input hypergraph to permute

**Returns:**
- `tuple[list[int], OpenHypergraph]`:
  - The permutation used (list of node indices)
  - The permuted hypergraph

## Algorithm Details

### MC Isomorphism Algorithm

The `check_MC_isomorphism` method implements a traversal-based algorithm for checking graph isomorphism:

1. **Initialization**: Validates that both graphs have the same number of nodes, edges, inputs, and outputs
2. **Input/Output Mapping**: Maps corresponding input and output nodes
3. **Graph Traversal**:
   - Starts from input nodes
   - Explores connected edges and nodes
   - Maintains node and edge mappings
   - Validates structural compatibility at each step
4. **Validation**: Checks if all nodes and edges have been consistently mapped

### Key Properties

- **Monogamous**: Each edge connects to unique source/target nodes
- **Cartesian**: The graph structure follows cartesian product properties
- **Preserves Labels**: Node and edge labels must match in the isomorphism
- **Preserves Structure**: Source/target relationships are maintained

## Usage Examples

### Example 1: Basic Isomorphism Check

```python
from proofChecker_python_serial.hypergraph import OpenHypergraph, Node, HyperEdge
from proofChecker_python_serial.isomorphisms import MC_isomorphism

# Create first graph
nodes1 = [Node(i, label) for i, label in enumerate(['a', 'b', 'c'])]
edges1 = [HyperEdge([0], [1], 'f', 0), HyperEdge([1], [2], 'g', 1)]
graph1 = OpenHypergraph(nodes1, edges1, [0], [2])

# Create isomorphic graph with different node order
nodes2 = [Node(i, label) for i, label in enumerate(['c', 'b', 'a'])]
edges2 = [HyperEdge([1], [0], 'g', 0), HyperEdge([2], [1], 'f', 1)]
graph2 = OpenHypergraph(nodes2, edges2, [2], [0])

# Check isomorphism
is_iso, node_map, edge_map = MC_isomorphism(graph1, graph2)
print(f"Isomorphic: {is_iso}")
print(f"Node mapping: {node_map}")
print(f"Edge mapping: {edge_map}")
```

### Example 2: Testing with Random Permutations

```python
from proofChecker_python_serial.isomorphisms import permute_graph, MC_isomorphism

# Create a random permutation of the graph
permutation, permuted = permute_graph(original_graph)

# Check if we can detect the isomorphism
is_iso, node_map, edge_map = MC_isomorphism(original_graph, permuted)

# Verify the mapping matches the permutation
assert is_iso == True
assert node_map == permutation
```

### Example 3: Non-Isomorphic Graphs

```python
# Create two structurally different graphs
graph1 = create_linear_graph(['a', 'b', 'c'])  # a -> b -> c
graph2 = create_branching_graph(['a', 'b', 'c'])  # a -> b, a -> c

# Check isomorphism
is_iso, _, _ = MC_isomorphism(graph1, graph2)
assert is_iso == False  # Different structures
```

## Logging

The isomorphism checker uses Python's logging module for detailed debugging:

```python
import logging

# Enable debug logging to see traversal details
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('proofChecker_python_serial.isomorphisms')

# Now run isomorphism check with detailed logs
MC_isomorphism(graph1, graph2)
```

Log levels:
- **DEBUG**: Detailed traversal information, mapping updates
- **INFO**: High-level results like visited nodes

## Performance Considerations

- **Time Complexity**: O(V + E) for MC hypergraphs where V = vertices, E = edges
- **Space Complexity**: O(V + E) for storing mappings and visited nodes
- **Optimization**: The algorithm short-circuits on the first incompatibility found

## See Also

- [Hypergraph API](hypergraph.md) - Core hypergraph structures
- [Node API](node.md) - Node definitions
- [HyperEdge API](hyperedge.md) - Edge definitions
