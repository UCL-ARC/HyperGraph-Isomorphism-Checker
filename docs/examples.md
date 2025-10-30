# Examples

## Example 1: Simple Linear Pipeline

Create a simple linear processing pipeline:

```python
from IsomorphismChecker_python_serial.node import Node
from IsomorphismChecker_python_serial.edge import Edge
from IsomorphismChecker_python_serial.graph import Graph

# Create pipeline nodes
input_node = Node(index=0, label="input")
process1 = Node(index=1, label="process1")
process2 = Node(index=2, label="process2")
output_node = Node(index=3, label="output")

# Create edges
edges = [
    Edge(source=input_node, target=process1, label="f"),
    Edge(source=process1, target=process2, label="g"),
    Edge(source=process2, target=output_node, label="h")
]

# Create graph
pipeline = Graph(
    nodes=[input_node, process1, process2, output_node],
    edges=edges
)

# Validate and visualize
if pipeline.is_valid():
    from IsomorphismChecker_python_serial.diagram import Diagram
    diagram = Diagram(openHyperGraph=pipeline)
    diagram.render("pipeline")
```

## Example 2: Hypergraph with Multiple Sources/Targets

```python
from IsomorphismChecker_python_serial.hypergraph import OpenHypergraph
from IsomorphismChecker_python_serial.hyperedge import HyperEdge
from IsomorphismChecker_python_serial.node import Node

# Create nodes for a merge-split pattern
nodes = [
    Node(index=0, label="a"),
    Node(index=1, label="b"),
    Node(index=2, label="c"),
    Node(index=3, label="d"),
    Node(index=4, label="e")
]

# Merge: two sources -> one target
merge_edge = HyperEdge(
    index=0,
    label="M",
    sources=[nodes[0], nodes[1]],
    targets=[nodes[2]]
)

# Split: one source -> two targets
split_edge = HyperEdge(
    index=1,
    label="S",
    sources=[nodes[2]],
    targets=[nodes[3], nodes[4]]
)

# Create hypergraph
hypergraph = OpenHypergraph(
    nodes=nodes,
    edges=[merge_edge, split_edge]
)

print(f"Input nodes: {[n.label for n in hypergraph.input_nodes]}")
print(f"Output nodes: {[n.label for n in hypergraph.output_nodes]}")
```

## Example 3: Loading and Processing JSON Files

```python
from IsomorphismChecker_python_serial.json_utils import read_all_jsons_in_directory
from IsomorphismChecker_python_serial.hypergraph import create_hypergraph

# Load all hypergraph JSON files
json_files = read_all_jsons_in_directory("path/to/jsons")

# Process each hypergraph
for filename, data in json_files.items():
    try:
        hypergraph = create_hypergraph(data)

        if hypergraph.is_valid():
            print(f"✓ {filename}: Valid hypergraph")
            print(f"  Nodes: {len(hypergraph.nodes)}")
            print(f"  Edges: {len(hypergraph.edges)}")
        else:
            print(f"✗ {filename}: Invalid")
    except Exception as e:
        print(f"✗ {filename}: Error - {e}")
```

## Example 4: Signature Comparison

```python
from IsomorphismChecker_python_serial.signature import Signature
from IsomorphismChecker_python_serial.compare_signatures import CompareSignatures

# Create signatures
sig1 = Signature("a-b c-d")
sig2 = Signature("x-y z-w")

# Compare signatures
comparator = CompareSignatures(sig1, sig2)

if comparator.compare():
    print("Signatures match!")
    print(f"Mapping: {comparator.get_mapping()}")
else:
    print("Signatures don't match")
```

## Example 5: Custom Validation

```python
from IsomorphismChecker_python_serial.graph import Graph
from IsomorphismChecker_python_serial.validation import validate_label, validate_index

# Custom graph creation with validation
def create_validated_graph(nodes_data, edges_data):
    nodes = []
    for node_data in nodes_data:
        # Validate before creating
        validate_index(node_data['index'])
        validate_label(node_data['label'], allow_digits=False)

        node = Node(**node_data)
        nodes.append(node)

    # Create graph...
    graph = Graph(nodes=nodes, edges=edges)

    # Get detailed validation results

    return graph
```
