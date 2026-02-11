#ifndef GPU_DATA_STRUCTURES_H
#define GPU_DATA_STRUCTURES_H

#include <vector>
#include <string>
#include <algorithm>

/**
 * @brief Structure representing a hyperedge in the input graph.
 *
 * A hyperedge connects multiple source nodes to multiple target nodes and has a label.
 * Think of them as a function with multiple inputs and outputs. For example, a hyperedge
 * f(x, y) -> (z1, z2) takes inputs x and y and produces outputs z1 and z2.
 *
 * @member labelIndex Index of the hyperedge label in the edge labels database. 'f' in the example above.
 * @member sourceNodes Vector of source node indices (inputs). 'x' and 'y' in the example above.
 * @member targetNodes Vector of target node indices (outputs). 'z1' and 'z2' in the example above.
 */
/*-------------------------------------------------------------------------------------------------------------------*/
struct IO_hyperEdge
{
	int labelIndex;
	std::vector<uint> sourceNodes;
	std::vector<uint> targetNodes;
};
/*-------------------------------------------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------------------------------------------*/
struct InputGraph
{
	std::vector<std::string>  nodeLabelsDB;      /* Unique node labels */
	std::vector<uint>         nodeLabelIndex;    /* Label index for each node */

	// std::vector<uint>         node_EdgeSources;  /* Node edge source connections */
	// std::vector<uint>         node_EdgeTargets;  /* Node edge target connections */

	std::vector<std::string>  edgeLabelsDB;      /* Unique edge labels */
	std::vector<IO_hyperEdge> edges;             /* All hyperedges */

	std::vector<uint>         globalInputs;      /* Global input node IDs */
	std::vector<uint>         globalOutputs;     /* Global output node IDs */
};

/**
 * @brief GPU node data structure for Compressed Sparse Row (CSR) representation.
 *
 * Organizes all node-related data for GPU transfer and computation. Uses CSR format
 * for efficient storage and access of node connectivity information.
 *
 * **Metadata Fields:**
 * - numNodes: Total number of nodes in the graph
 * - numNodeLabelsDB: Number of unique node labels (label dictionary size)
 * - nodeEdgesPrevsSize: Total elements in the compact edgePrevs array
 * - nodeEdgesNextsSize: Total elements in the compact edgeNexts array
 *
 * **Node Label Information:**
 * - labelIndex: Array mapping node ID to its label in the database
 *
 * **CSR Previous Edges (Incoming Connections):**
 * - edgeStartPrevsStart: CSR offset array for previous edges
 * - edgeStartPrevsNum: CSR count array (degree) for previous edges
 * - edgePrevs: Compact array of previous edge IDs
 * - edgePrevsPort: Port index for each previous edge connection
 * - prevsFirstEdge: First edge label for each node (memoization)
 *
 * **CSR Next Edges (Outgoing Connections):**
 * - edgeStartNextsStart: CSR offset array for next edges
 * - edgeStartNextsNum: CSR count array (degree) for next edges
 * - edgeNexts: Compact array of next edge IDs
 * - edgeNextsPort: Port index for each next edge connection
 * - nextsFirstEdge: First edge label for each node (memoization)
 *
 * **Combined Edge Information:**
 * - totalEdges: Sum of incoming and outgoing edges per node
 *
 * **Metadata Tags:**
 * - ioTags: Input/Output classification (0=none, 1=input, 2=output, 3=both)
 */
struct CSR_NodeData
{
	uint numNodes;
	uint numNodeLabelsDB;
	uint nodeEdgesPrevsSize;
	uint nodeEdgesNextsSize;
	uint* labelIndex;
	int* prevsFirstEdge;
	int* nextsFirstEdge;
	uint* edgePrevs;
	uint* edgeNexts;
	int* edgePrevsPort;
	int* edgeNextsPort;
	uint* edgeStartPrevsStart;
	uint* edgeStartPrevsNum;
	uint* edgeStartNextsStart;
	uint* edgeStartNextsNum;
	uint* totalEdges;
	uint* ioTags;
};

/**
 * @brief GPU edge data structure for Compressed Sparse Row (CSR) representation.
 *
 * Organizes all edge-related data for GPU transfer and computation. Uses CSR format
 * for efficient storage of edge-to-node connectivity (source and target nodes).
 *
 * **Metadata Fields:**
 * - numEdges: Total number of hyperedges in the graph
 * - numEdgeLabelsDB: Number of unique edge labels (label dictionary size)
 * - edgeNodesSourceSize: Total elements in nodesSources compact array
 * - edgeNodesTargetSize: Total elements in nodesTargets compact array
 *
 * **Edge Labels:**
 * - labelIndex: Array mapping edge ID to its label in the database
 *
 * **CSR Source Nodes (Incoming/Driver Nodes):**
 * - nodeStartSourcesStart: CSR offset array for source nodes per edge
 * - nodeStartSourcesNum: CSR count array (number of source nodes per edge)
 * - nodesSources: Compact array of all source node IDs
 *
 * **CSR Target Nodes (Output/Consumer Nodes):**
 * - nodeStartTargetsStart: CSR offset array for target nodes per edge
 * - nodeStartTargetsNum: CSR count array (number of target nodes per edge)
 * - nodesTargets: Compact array of all target node IDs
 *
 * **Combined Node Information:**
 * - totalNodes: Sum of source and target nodes per edge
 */
struct CSR_EdgeData
{
	uint numEdges;
	uint numEdgeLabelsDB;
	uint edgeNodesSourceSize;
	uint edgeNodesTargetSize;
	uint* labelIndex;
	uint* nodesSources;
	uint* nodesTargets;
	uint* nodeStartSourcesStart;
	uint* nodeStartSourcesNum;
	uint* nodeStartTargetsStart;
	uint* nodeStartTargetsNum;
	uint* totalNodes;
};

/**
 * @brief Complete GPU graph data container with node, edge, and metadata information.
 *
 * Encapsulates all GPU-transferable graph data for a single graph instance.
 * Designed to be passed to GPU kernels for isomorphism checking and other computations.
 *
 * **Graph Identification:**
 * - graphIndex: Which graph this is (0 or 1 for two-graph comparison)
 * - gpu: GPU device ID (typically 0 for single GPU systems)
 *
 * **Graph Data:**
 * - nodeData: All node-related CSR arrays and metadata (see GPUNodeData)
 * - edgeData: All edge-related CSR arrays and metadata (see GPUEdgeData)
 *
 * **Usage:** This structure is populated on the host side with all necessary graph
 * information in CSR format, then transferred to the GPU for kernel execution.
 */
struct CSR_Graph
{
	uint graphIndex;
	CSR_NodeData nodeData;
	CSR_EdgeData edgeData;
};








#endif // GPU_DATA_STRUCTURES_H
