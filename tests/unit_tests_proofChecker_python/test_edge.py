# """Tests for Edge class in edge module."""

# from proofChecker_python_serial.edge import Edge
# from proofChecker_python_serial.node import Node


# def test_edge_creation():
#     """Test basic edge creation and properties."""
#     node1 = Node(index=0, label="a")
#     node2 = Node(index=1, label="b")

#     edge = Edge(source=node1, target=node2, label="f")

#     assert edge.source == node1
#     assert edge.target == node2
#     assert edge.label == "f"
#     assert edge.signature.sources == node1
#     assert edge.signature.targets == node2


# def test_edge_signature():
#     """Test edge signature properties."""
#     source_node = Node(label="X", prev=None, next=0)
#     target_node = Node(label="Y", prev=0, next=None)

#     edge = Edge(source=source_node, target=target_node, label="g")
#     signature = edge.signature

#     assert signature.sources == source_node
#     assert signature.targets == target_node


# def test_edge_with_same_source_and_target():
#     """Test edge creation where source and target are the same node."""
#     node = Node(label="A", prev=0, next=0)
#     edge = Edge(source=node, target=node, label="self")

#     assert edge.source == node
#     assert edge.target == node
#     assert edge.label == "self"


# def test_edge_equality():
#     """Test edge equality comparison."""
#     node1 = Node(label="A", prev=None, next=0)
#     node2 = Node(label="B", prev=0, next=None)

#     edge1 = Edge(source=node1, target=node2, label="f")
#     edge2 = Edge(source=node1, target=node2, label="f")
#     edge3 = Edge(source=node1, target=node2, label="g")

#     assert edge1 == edge2
#     assert edge1 != edge3


# def test_edge_with_different_node_types():
#     """Test edges connecting different types of nodes."""
#     input_node = Node(label="I", prev=None, next=0)
#     output_node = Node(label="O", prev=0, next=None)
#     intermediate_node = Node(label="M", prev=0, next=1)

#     # Input to intermediate
#     edge1 = Edge(source=input_node, target=intermediate_node, label="a")
#     assert edge1.source.is_input
#     assert not edge1.target.is_input and not edge1.target.is_output

#     # Intermediate to output
#     edge2 = Edge(source=intermediate_node, target=output_node, label="b")
#     assert edge2.target.is_output
#     assert not edge2.source.is_input and not edge2.source.is_output


# def test_edge_label_variations():
#     """Test edges with different label types."""
#     node1 = Node(label="A", prev=None, next=0)
#     node2 = Node(label="B", prev=0, next=None)

#     # Single character label
#     edge1 = Edge(source=node1, target=node2, label="f")
#     assert edge1.label == "f"

#     # Multi-character label
#     edge2 = Edge(source=node1, target=node2, label="function")
#     assert edge2.label == "function"

#     # Numeric label
#     edge3 = Edge(source=node1, target=node2, label="1")
#     assert edge3.label == "1"
