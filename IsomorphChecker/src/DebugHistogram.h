#ifndef DEBUG_HISTOGRAM_H
#define DEBUG_HISTOGRAM_H

typedef unsigned int uint;

/**
 * @brief Histogram of edge degree distribution by node count.
 * 
 * Tracks the frequency distribution of hyperedges categorized by:
 * - Number of source nodes
 * - Number of target nodes  
 * - Total node count
 * 
 * Used for statistical analysis and visualization of graph topology properties.
 * 
 * @member maxNodesSize Maximum number of nodes in any edge (determines histogram bin size)
 * @member sourceNodeCount Histogram array: frequency of edges with N source nodes
 * @member targetNodeCount Histogram array: frequency of edges with N target nodes
 * @member totalNodeCount Histogram array: frequency of edges with N total nodes
 */
struct EdgeHistogram
{
	uint   maxNodesSize;           /* Max number of nodes in any edge (for bin sizing) */
	uint  *sourceNodeCount;        /* Histogram: count of edges with N source nodes */
	uint  *targetNodeCount;        /* Histogram: count of edges with N target nodes */
	uint  *totalNodeCount;         /* Histogram: count of edges with N total nodes */
};

/**
 * @brief Histogram of node degree distribution by edge connectivity.
 * 
 * Tracks the frequency distribution of nodes categorized by:
 * - Number of incoming edges (previous/target edges)
 * - Number of outgoing edges (next/source edges)
 * - Total edge connectivity
 * - Input/Output tag designation
 * 
 * Used for statistical analysis and topology characterization.
 * 
 * @member maxEdgesSize Maximum number of edges any node connects to (determines histogram bin size)
 * @member prevCount Histogram array: frequency of nodes with N incoming edges
 * @member nextCount Histogram array: frequency of nodes with N outgoing edges
 * @member totalCount Histogram array: frequency of nodes with N total edges
 * @member ioTagCounts Histogram array: frequency of nodes by IO tag (0=none, 1=input, 2=output, 3=both)
 */
struct NodeHistogram
{
	uint   maxEdgesSize;           /* Max number of edges any node connects to (for bin sizing) */
	uint  *prevCount;              /* Histogram: count of nodes with N incoming edges */
	uint  *nextCount;              /* Histogram: count of nodes with N outgoing edges */
	uint  *totalCount;             /* Histogram: count of nodes with N total edges */
	uint  *ioTagCounts;            /* Histogram: count of nodes by IO tag (none/input/output/both) */
};

/**
 * @brief Container for all debug histograms (edge and node statistics).
 * 
 * Organizes histogram data structures for comprehensive graph topology analysis.
 * Histograms are allocated/deallocated together and accessed in parallel during
 * graph statistics reporting.
 * 
 * @member edge EdgeHistogram tracking node count distribution across edges
 * @member node NodeHistogram tracking edge connectivity distribution across nodes
 */
struct DebugHistogram
{
	EdgeHistogram edge;
	NodeHistogram node;
};

/**
 * @brief Allocate histogram arrays for debug statistics.
 * 
 * Allocates all dynamically-sized arrays within both EdgeHistogram and NodeHistogram
 * structures based on their maximum size parameters. Arrays are sized as (maxSize + 1)
 * to accommodate all possible bin indices.
 * 
 * @param debugHist DebugHistogram structure with maxNodesSize and maxEdgesSize already set
 * 
 * @pre debugHist.edge.maxNodesSize and debugHist.node.maxEdgesSize must be initialized
 * @post All histogram arrays are allocated and zero-initialized
 * 
 * @see DeallocateDebugHistograms
 */
static inline void AllocateDebugHistograms(DebugHistogram& debugHist)
{
	debugHist.edge.sourceNodeCount = new uint  [debugHist.edge.maxNodesSize +1]();
	debugHist.edge.targetNodeCount = new uint  [debugHist.edge.maxNodesSize +1]();
	debugHist.edge.totalNodeCount  = new uint  [debugHist.edge.maxNodesSize +1]();

	debugHist.node.prevCount   = new uint  [debugHist.node.maxEdgesSize +1]();
	debugHist.node.nextCount   = new uint  [debugHist.node.maxEdgesSize +1]();
	debugHist.node.totalCount  = new uint  [debugHist.node.maxEdgesSize +1]();
	debugHist.node.ioTagCounts = new uint  [debugHist.node.maxEdgesSize +1]();
}

/**
 * @brief Deallocate histogram arrays for debug statistics.
 * 
 * Frees all dynamically allocated arrays within both EdgeHistogram and NodeHistogram
 * structures. Performs null-safe deletion of all pointer members.
 * 
 * @param debugHist DebugHistogram structure to deallocate
 * 
 * @pre debugHist should have been allocated with AllocateDebugHistograms
 * 
 * @see AllocateDebugHistograms
 */
static inline void DeallocateDebugHistograms(DebugHistogram& debugHist)
{
	delete [] debugHist.edge.sourceNodeCount;
	delete [] debugHist.edge.targetNodeCount;
	delete [] debugHist.edge.totalNodeCount;

	delete [] debugHist.node.prevCount;
	delete [] debugHist.node.nextCount;
	delete [] debugHist.node.totalCount;
	delete [] debugHist.node.ioTagCounts;
}

#endif // DEBUG_HISTOGRAM_H
