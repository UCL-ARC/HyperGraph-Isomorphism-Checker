# Graph Isomorphisms

This module provides functionality for checking graph isomorphisms between hypergraphs, particularly for monogamous, cartesian (MC) hypergraphs.

## Overview

Graph isomorphism checking determines whether two graphs are structurally identical, meaning there exists a bijection between their nodes and edges that preserves the graph structure.

## Classes

### Isomorphism

::: IsomorphismChecker_python_serial.isomorphisms.Isomorphism
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

::: IsomorphismChecker_python_serial.isomorphisms.MappingMode
    options:
      show_source: true

## Functions

### MC_isomorphism

Main function for checking monogamous, cartesian hypergraph isomorphism.

```python
from IsomorphismChecker_python_serial.isomorphisms import MC_isomorphism
from IsomorphismChecker_python_serial.hypergraph import OpenHypergraph

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
from IsomorphismChecker_python_serial.isomorphisms import permute_graph

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

## How Isomorphism is Computed

This section explains the algorithm used in the `MC_isomorphism` function in terms of code logic and steps. In order to understand the mathematical reasoning behind the algorithm, please refer to the [algorithm documentation](../getting-started/algorithm.md).

The `check_MC_isomorphism` method implements a **graph traversal-based algorithm** specifically designed for monogamous, cartesian (MC) hypergraphs. Here's a detailed explanation of how it works:

### Step-by-Step Algorithm

#### 1. **Initial Validation**

```python
# Check basic structural properties
if self.n_nodes[0] != self.n_nodes[1]:
    return False, [], []  # Different number of nodes
if self.n_edges[0] != self.n_edges[1]:
    return False, [], []  # Different number of edges
```

The algorithm first performs quick checks to ensure both graphs have:
- Same number of nodes
- Same number of edges
- Same number of input nodes
- Same number of output nodes

If any of these checks fail, the graphs cannot be isomorphic.

#### 2. **Input/Output Node Mapping**

```python
# Map input nodes (sources with no incoming edges)
for i, input_index in enumerate(self.graphs[0].input_nodes):
    corresponding_input = self.graphs[1].input_nodes[i]
    self.update_mapping(input_index, corresponding_input, MappingMode.NODES)

# Similarly for output nodes
```

The algorithm establishes an initial mapping by aligning:
- Input nodes (nodes with no predecessors) in order
- Output nodes (nodes with no successors) in order

This provides anchor points for the traversal.

#### 3. **Graph Traversal from Input Nodes**

```python
def traverse_from_nodes(self, start_nodes: list[int], graph_id: int) -> bool:
    """Traverse graph starting from given nodes, building mappings."""

    queue = deque(start_nodes)
    visited_nodes = set(start_nodes)

    while queue:
        current_node = queue.popleft()

        # Find outgoing edges from current node
        outgoing_edges = [e for e in graph.edges if current_node in e.sources]

        for edge in outgoing_edges:
            # Try to map this edge and its connected nodes
            if not self.explore_edges(edge, graph_id):
                return False  # Incompatible structure

            # Add newly mapped target nodes to queue
            for target in edge.targets:
                if target not in visited_nodes:
                    queue.append(target)
                    visited_nodes.add(target)

    return True
```

**Key points:**
- Uses **breadth-first search (BFS)** starting from input nodes
- Maintains a queue of nodes to visit
- For each node, explores all outgoing edges
- Attempts to map edges and their connected nodes
- Continues until all reachable nodes are visited

#### 4. **Edge Exploration and Mapping**

```python
def explore_edges(self, edge: HyperEdge, graph_id: int) -> bool:
    """Explore an edge and try to map it to the other graph."""

    # Get corresponding edge in the other graph
    other_graph = self.graphs[1 - graph_id]

    # Find candidate edges in other graph that could match
    candidates = [e for e in other_graph.edges
                  if self.check_edge_compatibility(edge, e, graph_id)]

    if not candidates:
        return False  # No compatible edge found

    # Use the first compatible edge (deterministic for MC graphs)
    matched_edge = candidates[0]

    # Update edge mapping
    self.update_mapping(edge.index, matched_edge.index, MappingMode.EDGES)

    # Update node mappings for sources and targets
    for src1, src2 in zip(edge.sources, matched_edge.sources):
        self.update_mapping(src1, src2, MappingMode.NODES)

    for tgt1, tgt2 in zip(edge.targets, matched_edge.targets):
        self.update_mapping(tgt1, tgt2, MappingMode.NODES)

    return True
```

**The exploration process:**

1. For each edge in graph 1, finds compatible edges in graph 2
2. Checks compatibility based on: (a) Edge labels must match (b)Number of sources must match (c)Number of targets must match (d)Already-mapped nodes must correspond correctly
3. Updates both edge and node mappings
4. Returns false if no compatible edge exists

#### 5. **Edge Compatibility Checking**

```python
def check_edge_compatibility(self, edge1: HyperEdge, edge2: HyperEdge,
                             graph_id: int) -> bool:
    """Check if two edges are compatible for mapping."""

    # Labels must match
    if edge1.label != edge2.label:
        return False

    # Structure must match
    if len(edge1.sources) != len(edge2.sources):
        return False
    if len(edge1.targets) != len(edge2.targets):
        return False

    # Check if already-mapped nodes correspond correctly
    for src1, src2 in zip(edge1.sources, edge2.sources):
        if src1 in self.node_mappings[graph_id]:
            if self.node_mappings[graph_id][src1] != src2:
                return False  # Conflict with existing mapping

    # Similar check for targets
    for tgt1, tgt2 in zip(edge1.targets, edge2.targets):
        if tgt1 in self.node_mappings[graph_id]:
            if self.node_mappings[graph_id][tgt1] != tgt2:
                return False

    return True
```

**Compatibility requires:**
- Identical edge labels
- Same number of sources and targets
- Consistency with existing node mappings
- No conflicts with previously established correspondences

#### 6. **Bidirectional Traversal**

```python
# Traverse from inputs (forward direction)
if not self.traverse_from_nodes(self.graphs[0].input_nodes, 0):
    return False, [], []

# Traverse from outputs (backward direction)
if not self.traverse_from_nodes(self.graphs[0].output_nodes, 0):
    return False, [], []
```

The algorithm performs **two traversals**:
1. **Forward**: Starting from input nodes, following edge directions
2. **Backward**: Starting from output nodes, following edges in reverse

This ensures all nodes and edges are covered, even in graphs with cycles.

#### 7. **Final Validation**

```python
# Verify all nodes are mapped
if len(self.node_mappings[0]) != self.n_nodes[0]:
    return False, [], []

# Verify all edges are mapped
if len(self.edge_mappings[0]) != self.n_edges[0]:
    return False, [], []

# Convert mappings to permutation lists
node_permutation = [self.node_mappings[0][i] for i in range(self.n_nodes[0])]
edge_permutation = [self.edge_mappings[0][i] for i in range(self.n_edges[0])]

return True, node_permutation, edge_permutation
```

**Final checks ensure:**
- Every node in graph 1 has been mapped to a node in graph 2
- Every edge in graph 1 has been mapped to an edge in graph 2
- The mappings form valid permutations (bijections)

### Algorithm Complexity

**Time Complexity:** O(V + E)
- Each node is visited exactly once during traversal
- Each edge is examined exactly once
- V = number of vertices (nodes)
- E = number of edges

**Space Complexity:** O(V + E)
- Stores node mappings: O(V)
- Stores edge mappings: O(E)
- BFS queue: O(V) in worst case
- Visited set: O(V)

### Why This Algorithm Works for MC Hypergraphs

**Monogamous Property:**
- Each source/target appears in exactly one edge
- This makes edge matching deterministic
- No backtracking needed

**Cartesian Property:**
- Graph structure follows cartesian product rules
- Traversal order doesn't affect correctness
- Single forward pass is sufficient

**Key Insight:** The combination of monogamous and cartesian properties means that once we establish the input/output node mapping, the rest of the graph structure is uniquely determined. The traversal simply verifies this unique structure exists in both graphs.

### Example Execution Trace

Consider two simple isomorphic graphs:

**Graph 1:** `a --f--> b --g--> c`
**Graph 2:** `x --f--> y --g--> z`

```
Step 1: Initial validation ✓
  - Both have 3 nodes, 2 edges

Step 2: Map inputs/outputs
  - Map: a → x (both are inputs)
  - Map: c → z (both are outputs)

Step 3: Forward traversal from 'a'
  - Visit node 'a' (already mapped to 'x')
  - Find edge 'f' from 'a' to 'b'
  - Find edge 'f' from 'x' in Graph 2 (goes to 'y')
  - Map: edge f₁ → f₂
  - Map: b → y (target nodes)

Step 4: Continue from 'b'
  - Visit node 'b' (mapped to 'y')
  - Find edge 'g' from 'b' to 'c'
  - Find edge 'g' from 'y' in Graph 2 (goes to 'z')
  - Map: edge g₁ → g₂
  - Node 'c' already mapped to 'z' ✓

Step 5: Final validation
  - All 3 nodes mapped ✓
  - All 2 edges mapped ✓
  - Return: (True, [0,1,2], [0,1])
```

### When the Algorithm Fails

The algorithm returns `False` if:

1. **Structural mismatch:**
   ```python
   Graph 1: a --> b --> c
   Graph 2: a --> b, b --> c, a --> c  # Extra edge
   # Fails: Different number of edges
   ```

2. **Label mismatch:**
   ```python
   Graph 1: a --f--> b
   Graph 2: a --g--> b  # Different edge label
   # Fails: Edge compatibility check
   ```

3. **Incompatible structure:**
   ```python
   Graph 1: a --f--> b --g--> c
   Graph 2: a --f--> c --g--> b  # Different connectivity
   # Fails: Node mapping conflict
   ```

4. **Incomplete traversal:**
   ```python
   # If graph has unreachable nodes
   # Fails: Not all nodes mapped
   ```
