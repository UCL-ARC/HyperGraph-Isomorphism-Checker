# # type: ignore

# from collections import defaultdict
# from dataclasses import dataclass, field
# from typing import NamedTuple, list


# # --------------------------------------------------------------------------------------------------
# # @Arindam  replace these with includes I just put them here for this mock code
# class Node:
#     """A node in a hypergraph."""

#     label: str

#     # Note: Properties like is_input, is_output can be derived
#     # but are not directly needed for the WL test itself.
#     def __init__(self, label: str):
#         self.label = label

#     # Making Node hashable by its ID for use in maps
#     def __hash__(self):
#         return id(self)

#     def __eq__(self, other):
#         return id(self) == id(other)


# class HyperEdgeSignature(NamedTuple):
#     """A signature for a hypergraph, defining the types of nodes and edges."""

#     sources: list[Node]
#     targets: list[Node]


# @dataclass(slots=True)
# class HyperEdge:
#     """A hyperedge in a hypergraph."""

#     sources: list[Node]
#     targets: list[Node]
#     label: str

#     @property
#     def signature(self) -> HyperEdgeSignature:
#         return HyperEdgeSignature(sources=self.sources, targets=self.targets)


# @dataclass
# class OpenHypergraph:
#     """An open hypergraph with input and output nodes."""

#     nodes: list[Node] = field(default_factory=list)
#     edges: list[HyperEdge] = field(default_factory=list)


# # --------------------------------------------------------------------------------------------------


# # --------------------------------------------------------------------------------------------------
# #  Weisfeiler-Lehman Implementation
# # --------------------------------------------------------------------------------------------------
# def wl_test_hypergraph(h1: OpenHypergraph, h2: OpenHypergraph) -> bool:
#     """
#     Performs a Weisfeiler-Lehman test to check for hypergraph non-isomorphism.
#     Args:
#         h1: The first open hypergraph.
#         h2: The second open hypergraph.

#     Returns:
#         True: If the hypergraphs are determined to be non-isomorphic.
#         False: If the test is inconclusive (they might be isomorphic).
#     """

#     # --------------------------------------------------------------------------
#     # 0. Initial Check: Basic properties must match.
#     if len(h1.nodes) != len(h2.nodes) or len(h1.edges) != len(h2.edges):
#         return True
#     # --------------------------------------------------------------------------

#     num_nodes = len(h1.nodes)
#     num_edges = len(h1.edges)

#     # --------------------------------------------------------------------------
#     # Create mappings from Node object ID to a consistent integer index (0 to N-1)
#     h1_node_to_idx = {node: i for i, node in enumerate(h1.nodes)}
#     h2_node_to_idx = {node: i for i, node in enumerate(h2.nodes)}

#     # 1. Helper maps for efficient lookup: node_idx -> list of edge_indices
#     def build_connection_maps(hypergraph: OpenHypergraph, node_to_idx_map: dict):
#         source_map = defaultdict(list)
#         target_map = defaultdict(list)
#         for i, edge in enumerate(hypergraph.edges):
#             for source_node in edge.sources:
#                 source_map[node_to_idx_map[source_node]].append(i)
#             for target_node in edge.targets:
#                 target_map[node_to_idx_map[target_node]].append(i)
#         return source_map, target_map

#     h1_source_map, h1_target_map = build_connection_maps(h1, h1_node_to_idx)
#     h2_source_map, h2_target_map = build_connection_maps(h2, h2_node_to_idx)
#     # --------------------------------------------------------------------------

#     # 2. Initialization: Assign label '1' to all nodes and edges.
#     node_labels1 = {i: 1 for i in range(num_nodes)}
#     edge_labels1 = {i: 1 for i in range(num_edges)}
#     node_labels2 = {i: 1 for i in range(num_nodes)}
#     edge_labels2 = {i: 1 for i in range(num_edges)}

#     # --------------------------------------------------------------------------
#     # 3. Iterative Refinement Loop

#     for _ in range(num_nodes + num_edges):
#         fingerprint1 = (
#             tuple(sorted(node_labels1.values())),
#             tuple(sorted(edge_labels1.values())),
#         )
#         fingerprint2 = (
#             tuple(sorted(node_labels2.values())),
#             tuple(sorted(edge_labels2.values())),
#         )

#         if fingerprint1 != fingerprint2:
#             return True

#         # A] Signature Calculation
#         node_signatures1, edge_signatures1 = {}, {}
#         node_signatures2, edge_signatures2 = {}, {}

#         # B] Create hyperedge signatures from node labels
#         for i in range(num_edges):
#             s_labels1 = tuple(
#                 sorted(node_labels1[h1_node_to_idx[s]] for s in h1.edges[i].sources)
#             )
#             t_labels1 = tuple(
#                 sorted(node_labels1[h1_node_to_idx[t]] for t in h1.edges[i].targets)
#             )
#             edge_signatures1[i] = (edge_labels1[i], s_labels1, t_labels1)

#             s_labels2 = tuple(
#                 sorted(node_labels2[h2_node_to_idx[s]] for s in h2.edges[i].sources)
#             )
#             t_labels2 = tuple(
#                 sorted(node_labels2[h2_node_to_idx[t]] for t in h2.edges[i].targets)
#             )
#             edge_signatures2[i] = (edge_labels2[i], s_labels2, t_labels2)

#         # C] Create node signatures from hyperedge labels
#         for i in range(num_nodes):
#             source_of_labels1 = tuple(sorted(edge_labels1[e] for e in h1_source_map[i]))
#             target_of_labels1 = tuple(sorted(edge_labels1[e] for e in h1_target_map[i]))
#             node_signatures1[i] = (
#                 node_labels1[i],
#                 source_of_labels1,
#                 target_of_labels1,
#             )

#             source_of_labels2 = tuple(sorted(edge_labels2[e] for e in h2_source_map[i]))
#             target_of_labels2 = tuple(sorted(edge_labels2[e] for e in h2_target_map[i]))
#             node_signatures2[i] = (
#                 node_labels2[i],
#                 source_of_labels2,
#                 target_of_labels2,
#             )

#         # D] Label Compression
#         all_node_sigs = sorted(
#             list(set(node_signatures1.values()) | set(node_signatures2.values()))
#         )
#         node_sig_map = {sig: i + 1 for i, sig in enumerate(all_node_sigs)}

#         all_edge_sigs = sorted(
#             list(set(edge_signatures1.values()) | set(edge_signatures2.values()))
#         )
#         edge_sig_map = {sig: i + 1 for i, sig in enumerate(all_edge_sigs)}

#         # E] Update labels for the next iteration
#         node_labels1 = {i: node_sig_map[s] for i, s in node_signatures1.items()}
#         node_labels2 = {i: node_sig_map[s] for i, s in node_signatures2.items()}
#         edge_labels1 = {i: edge_sig_map[s] for i, s in edge_signatures1.items()}
#         edge_labels2 = {i: edge_sig_map[s] for i, s in edge_signatures2.items()}

#     # End 3. Iterative Refinement Loop
#     # --------------------------------------------------------------------------

#     # If the loop completes, the test is inconclusive.
#     return False


# ###############################################


# # Map objects to stable integer indices for label tracking
# def func(g1: OpenHypergraph, g2: OpenHypergraph) -> bool:
#     g1_node_to_idx = {node: i for i, node in enumerate(g1.nodes)}
#     g1_edge_to_idx = {edge: i for i, edge in enumerate(g1.edges)}
#     g2_node_to_idx = {node: i for i, node in enumerate(g2.nodes)}
#     g2_edge_to_idx = {edge: i for i, edge in enumerate(g2.edges)}
#     # Initialize labels
#     node_labels1 = {i: 1 for i in range(num_nodes)}
#     edge_labels1 = {i: 1 for i in range(num_edges)}
#     node_labels2 = {i: 1 for i in range(num_nodes)}
#     edge_labels2 = {i: 1 for i in range(num_edges)}
#     # Iterative Refinement
#     for _ in range(num_nodes + num_edges):
#         fingerprint1 = (
#             tuple(sorted(node_labels1.values())),
#             tuple(sorted(edge_labels1.values())),
#         )
#         fingerprint2 = (
#             tuple(sorted(node_labels2.values())),
#             tuple(sorted(edge_labels2.values())),
#         )
#         if fingerprint1 != fingerprint2:
#             return True
#         # Signature Calculation
#         node_signatures1, edge_signatures1 = {}, {}
#         node_signatures2, edge_signatures2 = {}, {}
#         # 1. Create hyperedge signatures
#         for i, edge in enumerate(g1.edges):
#             s_labels = tuple(
#                 sorted(node_labels1[g1_node_to_idx[s]] for s in edge.sources)
#             )
#             t_labels = tuple(
#                 sorted(node_labels1[g1_node_to_idx[t]] for t in edge.targets)
#             )
#             edge_signatures1[i] = (edge_labels1[i], s_labels, t_labels)
#         for i, edge in enumerate(g2.edges):
#             s_labels = tuple(
#                 sorted(node_labels2[g2_node_to_idx[s]] for s in edge.sources)
#             )
#             t_labels = tuple(
#                 sorted(node_labels2[g2_node_to_idx[t]] for t in edge.targets)
#             )
#             edge_signatures2[i] = (edge_labels2[i], s_labels, t_labels)
#         # 2. Create node signatures
#         for i, node in enumerate(g1.nodes):
#             source_for_labels = tuple(
#                 sorted(edge_labels1[g1_edge_to_idx[e]] for e in node.next_edges)
#             )
#             target_for_labels = tuple(
#                 sorted(edge_labels1[g1_edge_to_idx[e]] for e in node.previous_edges)
#             )
#             node_signatures1[i] = (
#                 node_labels1[i],
#                 source_for_labels,
#                 target_for_labels,
#             )
#         for i, node in enumerate(g2.nodes):
#             source_for_labels = tuple(
#                 sorted(edge_labels2[g2_edge_to_idx[e]] for e in node.next_edges)
#             )
#             target_for_labels = tuple(
#                 sorted(edge_labels2[g2_edge_to_idx[e]] for e in node.previous_edges)
#             )
#             node_signatures2[i] = (
#                 node_labels2[i],
#                 source_for_labels,
#                 target_for_labels,
#             )
#         # Label Compression
#         all_node_sigs = sorted(
#             list(set(node_signatures1.values()) | set(node_signatures2.values()))
#         )
#         node_sig_map = {sig: i + 1 for i, sig in enumerate(all_node_sigs)}
#         all_edge_sigs = sorted(
#             list(set(edge_signatures1.values()) | set(edge_signatures2.values()))
#         )
#         edge_sig_map = {sig: i + 1 for i, sig in enumerate(all_edge_sigs)}
#         node_labels1 = {i: node_sig_map[s] for i, s in node_signatures1.items()}
#         node_labels2 = {i: node_sig_map[s] for i, s in node_signatures2.items()}
#         edge_labels1 = {i: edge_sig_map[s] for i, s in edge_signatures1.items()}
#         edge_labels2 = {i: edge_sig_map[s] for i, s in edge_signatures2.items()}
#     return False
