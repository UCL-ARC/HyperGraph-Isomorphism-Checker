#ifndef GPU_DATA_STRUCTURES_H
#define GPU_DATA_STRUCTURES_H

#include <vector>
#include <string>
#include <algorithm>
#include "DebugHistogram.h"

// Forward declaration for InputGraph structure
struct InputGraph;

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
struct GPUNodeData
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
struct GPUEdgeData
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
struct GPUGraphData
{
	uint graphIndex;
	GPUNodeData nodeData;
	GPUEdgeData edgeData;
	uint gpu;
};

/**
 * @brief Wrapper function to transfer GPU graph data to GPU device.
 * 
 * Unpacks the organized GPUGraphData structure and calls the low-level GPU
 * initialization function with all necessary arrays and metadata.
 * 
 * @param gpuData Complete GPU graph data structure containing all node/edge arrays
 * 
 * @see GPUGraphData, InitGPUArrays
 */
void TransferGraphToGPU(const GPUGraphData& gpuData);

/**
 * @brief Initialize GPU graph data structure with allocated arrays.
 * 
 * Allocates all necessary GPU arrays for node and edge data based on the input graph
 * structure. Sets up metadata fields and initializes special arrays (prevsFirstEdge,
 * nextsFirstEdge) to -1.
 * 
 * @param gInd Graph index (0 or 1) for identification
 * @param graph Input graph containing topology information
 * @param gpuData Output GPU graph data structure to be populated
 * 
 * @post All pointer fields in gpuData are allocated and ready for population.
 * 
 * @see DeallocateGPUGraphData
 */
void InitializeGPUGraphData(int gInd, const InputGraph& graph, GPUGraphData& gpuData);

/**
 * @brief Deallocate GPU graph data structure.
 * 
 * Frees all dynamically allocated arrays in the GPU graph data structure.
 * Performs null-safe deletion of all pointer members.
 * 
 * @param gpuData GPU graph data structure to deallocate
 * 
 * @pre gpuData should have been initialized with InitializeGPUGraphData
 * 
 * @see InitializeGPUGraphData
 */
void DeallocateGPUGraphData(GPUGraphData& gpuData);

#endif // GPU_DATA_STRUCTURES_H
