
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
#include "json.hpp"

#include "GPU_Solver/CUDA_Functions.h" /* CUDA Solver */


using namespace std;


/* Configuation */

bool debugSort = false;
string defaultFile1 = "../Input/DrugALarge.json";
string defaultFile2 = "../Input/DrugBLarge.json";

/* End Configuration */

/* XML Reader for C++ */
using json = nlohmann::json;
struct IO_hyperEdge
{
	int labelIndex;
	std::vector<uint> sourceNodes;
	std::vector<uint> targetNodes;
};


/* Comparator for hyperedges used in multiple sorts */
namespace {
static inline bool HyperEdgeLess(const IO_hyperEdge& a, const IO_hyperEdge& b)
{
	// Compute totals once
	int total_a = static_cast<int>(a.sourceNodes.size() + a.targetNodes.size());
	int total_b = static_cast<int>(b.sourceNodes.size() + b.targetNodes.size());

	// Sort by source nodes count first
	if (a.sourceNodes.size() != b.sourceNodes.size())
	{
		return a.sourceNodes.size() < b.sourceNodes.size();
	}

	// Then by target nodes count
	if (a.targetNodes.size() != b.targetNodes.size())
	{
		return a.targetNodes.size() < b.targetNodes.size();
	}

	// Then by total nodes
	if (total_a != total_b)
	{
		return total_a < total_b;
	}

	// Finally by label index
	return a.labelIndex < b.labelIndex;
}
} // anonymous namespace




/*-------------------------------------------------------------------------------------------------------------------*/
/* Input Graph Structure - encapsulates all IO data for a single graph */
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

/* Graph Details [2] means we will store 2 graphs can be made to as many as needed */
/* MOVED: InputGraph IO_graphs[2]; is now declared locally in main() */

/*-------------------------------------------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------------------------------------------*/
/* Debug Histogram Structures - organize histograms for analyzing graph topology */
/*-------------------------------------------------------------------------------------------------------------------*/
struct EdgeHistogram
{
	uint   maxNodesSize;           /* Max number of nodes in any edge (for bin sizing) */
	uint  *sourceNodeCount;        /* Histogram: count of edges with N source nodes */
	uint  *targetNodeCount;        /* Histogram: count of edges with N target nodes */
	uint  *totalNodeCount;         /* Histogram: count of edges with N total nodes */
};

struct NodeHistogram
{
	uint   maxEdgesSize;           /* Max number of edges any node connects to (for bin sizing) */
	uint  *prevCount;              /* Histogram: count of nodes with N incoming edges */
	uint  *nextCount;              /* Histogram: count of nodes with N outgoing edges */
	uint  *totalCount;             /* Histogram: count of nodes with N total edges */
	uint  *ioTagCounts;            /* Histogram: count of nodes by IO tag (none/input/output/both) */
};

struct DebugHistogram
{
	EdgeHistogram edge;
	NodeHistogram node;
};

/*-------------------------------------------------------------------------------------------------------------------*/




/*-------------------------------------------------------------------------------------------------------------------*/
/* GPU Data Transfer Structures - organize node and edge data for GPU initialization */
/*-------------------------------------------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------------------------------------------*/
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

struct GPUGraphData
{
	uint graphIndex;
	GPUNodeData nodeData;
	GPUEdgeData edgeData;
	uint gpu;
};

/* Wrapper function to transfer GPU data using organized struct */
static inline void TransferGraphToGPU(const GPUGraphData& gpuData)
{
	InitGPUArrays( gpuData.graphIndex,
	               gpuData.nodeData.numNodes, gpuData.nodeData.labelIndex,
	               gpuData.nodeData.prevsFirstEdge, gpuData.nodeData.nextsFirstEdge,
	               gpuData.nodeData.nodeEdgesPrevsSize, gpuData.nodeData.nodeEdgesNextsSize,
	               gpuData.nodeData.edgePrevs, gpuData.nodeData.edgeNexts,
	               gpuData.nodeData.edgePrevsPort, gpuData.nodeData.edgeNextsPort,
	               gpuData.nodeData.edgeStartPrevsStart, gpuData.nodeData.edgeStartPrevsNum,
	               gpuData.nodeData.edgeStartNextsStart, gpuData.nodeData.edgeStartNextsNum,
	               gpuData.nodeData.totalEdges, gpuData.nodeData.ioTags,
	               gpuData.edgeData.numEdges, gpuData.edgeData.labelIndex,
	               gpuData.edgeData.edgeNodesSourceSize, gpuData.edgeData.edgeNodesTargetSize,
	               gpuData.edgeData.nodesSources, gpuData.edgeData.nodesTargets,
	               gpuData.edgeData.nodeStartSourcesStart, gpuData.edgeData.nodeStartSourcesNum,
	               gpuData.edgeData.nodeStartTargetsStart, gpuData.edgeData.nodeStartTargetsNum,
	               gpuData.edgeData.totalNodes, gpuData.gpu );
}

/*-------------------------------------------------------------------------------------------------------------------*/

/* Debug helper: builds a permutation index for edges and prints the mapping.
 * NOTE: If called after IO_edges[gInd] is sorted, permutation will be identity.
 * To get original->sorted positions, invoke before sorting or capture the original order. */
static void DebugEdgeIndexMapping(int gInd, const InputGraph* graphs, uint** edgeLabelIndexOrg)
{
	uint numEdgesS = graphs[gInd].edges.size();
	printf(" EdgeSortIndex %u \n", numEdgesS);

	// Allocate index array (identity permutation)
	edgeLabelIndexOrg[gInd] = new uint[numEdgesS]();
	for (uint i = 0; i < numEdgesS; ++i)
	{
		edgeLabelIndexOrg[gInd][i] = i;
	}

	// Sort the index array using the same comparator (redundant post edge sort)
	const auto& edges_to_compare = graphs[gInd].edges;
	std::sort(
		edgeLabelIndexOrg[gInd],
		edgeLabelIndexOrg[gInd] + numEdgesS,
		[&edges_to_compare](uint ia, uint ib)
		{
			return HyperEdgeLess(edges_to_compare[ia], edges_to_compare[ib]);
		}
	);

	// Print mapping
	for (uint i = 0; i < numEdgesS; ++i)
	{
		printf(" %u EdgeLabMap %u \n", i, edgeLabelIndexOrg[gInd][i]);
	}
}

/* Parse command-line arguments for input filenames, with fallback to defaults */
static void ParseInputFilenames(int argc, char* argv[], string filenames[2])
{
	// Parse command-line arguments or use defaults
	if (argc >= 3)
	{
		filenames[0] = argv[1];
		filenames[1] = argv[2];
	}
	else
	{
		// Default filenames if not provided
		filenames[0] = defaultFile1;
		filenames[1] = defaultFile2;

		if (argc > 1)
		{
			std::cerr << "Usage: " << argv[0] << " <graph1.json> <graph2.json>" << std::endl;
			std::cerr << "Using default input files." << std::endl;
		}
	}
}

/* Process edge nodes (source or target) and populate CSR arrays */
static inline void ProcessEdgeNodes(
	int gInd,
	uint e,
	const std::vector<uint>& nodeList,
	uint* edgeNodeArray,
	int& edgeCounter,
	uint** nodeEdgeArray,
	int** nodeEdgePortArray,
	uint* debugNodeCount,
	int** nodeFirstEdgeArray,
	int edgeLabelIndex)
{
	for (uint i = 0; i < nodeList.size(); i++)
	{
		uint nID = nodeList[i];
		
		/* Write node into compact array */
		edgeNodeArray[edgeCounter] = nID;
		edgeCounter++;
		
		/* Fill the node edge list and port */
		nodeEdgeArray[gInd][debugNodeCount[nID]] = e;
		nodeEdgePortArray[gInd][debugNodeCount[nID]] = i;
		
		/* Store first port label */
		if (debugNodeCount[nID] == 0)
		{
			nodeFirstEdgeArray[gInd][nID] = edgeLabelIndex;
		}
		
		debugNodeCount[nID]++;
	}
}

/* Initialize GPU graph data structure with allocated arrays */
static inline void InitializeGPUGraphData(int gInd, const InputGraph& graph, GPUGraphData& gpuData)
{
	gpuData.graphIndex = gInd;
	gpuData.gpu = 0;

	/* Initialize node data metadata */
	gpuData.nodeData.numNodes = graph.nodeLabelIndex.size();
	gpuData.nodeData.numNodeLabelsDB = graph.nodeLabelsDB.size();
	gpuData.nodeData.nodeEdgesPrevsSize = 0;
	gpuData.nodeData.nodeEdgesNextsSize = 0;

	/* Allocate node arrays */
	gpuData.nodeData.labelIndex = new uint[gpuData.nodeData.numNodes]();
	gpuData.nodeData.ioTags = new uint[gpuData.nodeData.numNodes]();
	gpuData.nodeData.edgeStartPrevsNum = new uint[gpuData.nodeData.numNodes]();
	gpuData.nodeData.edgeStartNextsNum = new uint[gpuData.nodeData.numNodes]();
	gpuData.nodeData.totalEdges = new uint[gpuData.nodeData.numNodes]();
	gpuData.nodeData.edgeStartPrevsStart = new uint[gpuData.nodeData.numNodes]();
	gpuData.nodeData.edgeStartNextsStart = new uint[gpuData.nodeData.numNodes]();
	gpuData.nodeData.prevsFirstEdge = new int[gpuData.nodeData.numNodes]();
	gpuData.nodeData.nextsFirstEdge = new int[gpuData.nodeData.numNodes]();

	/* Initialize first edge markers */
	std::fill_n(gpuData.nodeData.prevsFirstEdge, gpuData.nodeData.numNodes, -1);
	std::fill_n(gpuData.nodeData.nextsFirstEdge, gpuData.nodeData.numNodes, -1);

	/* Initialize edge data metadata */
	gpuData.edgeData.numEdges = graph.edges.size();
	gpuData.edgeData.numEdgeLabelsDB = graph.edgeLabelsDB.size();
	gpuData.edgeData.edgeNodesSourceSize = 0;
	gpuData.edgeData.edgeNodesTargetSize = 0;

	/* Allocate edge arrays */
	gpuData.edgeData.labelIndex = new uint[gpuData.edgeData.numEdges]();
	gpuData.edgeData.nodeStartSourcesNum = new uint[gpuData.edgeData.numEdges]();
	gpuData.edgeData.nodeStartTargetsNum = new uint[gpuData.edgeData.numEdges]();
	gpuData.edgeData.totalNodes = new uint[gpuData.edgeData.numEdges]();
	gpuData.edgeData.nodeStartSourcesStart = new uint[gpuData.edgeData.numEdges]();
	gpuData.edgeData.nodeStartTargetsStart = new uint[gpuData.edgeData.numEdges]();
}

/* Deallocate GPU graph data structure */
static inline void DeallocateGPUGraphData(GPUGraphData& gpuData)
{
	delete[] gpuData.nodeData.labelIndex;
	delete[] gpuData.nodeData.ioTags;
	delete[] gpuData.nodeData.edgeStartPrevsNum;
	delete[] gpuData.nodeData.edgeStartNextsNum;
	delete[] gpuData.nodeData.totalEdges;
	delete[] gpuData.nodeData.edgeStartPrevsStart;
	delete[] gpuData.nodeData.edgeStartNextsStart;
	delete[] gpuData.nodeData.prevsFirstEdge;
	delete[] gpuData.nodeData.nextsFirstEdge;
	delete[] gpuData.nodeData.edgePrevs;
	delete[] gpuData.nodeData.edgeNexts;
	delete[] gpuData.nodeData.edgePrevsPort;
	delete[] gpuData.nodeData.edgeNextsPort;

	delete[] gpuData.edgeData.labelIndex;
	delete[] gpuData.edgeData.nodeStartSourcesNum;
	delete[] gpuData.edgeData.nodeStartTargetsNum;
	delete[] gpuData.edgeData.totalNodes;
	delete[] gpuData.edgeData.nodeStartSourcesStart;
	delete[] gpuData.edgeData.nodeStartTargetsStart;
	delete[] gpuData.edgeData.nodesSources;
	delete[] gpuData.edgeData.nodesTargets;
}

/* Compute compact array metadata (A0 + A1 + A2 phases)
 * Counts edge nodes, computes CSR offsets, and assigns IO tags
 * Returns the computed sizes needed for allocation
 */
static inline void ComputeCompactArrayMetadata(
	int gInd,
	const InputGraph& graph,
	GPUGraphData& gpuData,
	DebugHistogram* debugHist)
{
	/* A0] Count edge nodes to determine CSR array sizes */
	for (uint e = 0; e < gpuData.edgeData.numEdges; e++)
	{
		/* Each edge will increment its source nodes as a Next */
		for (uint i = 0; i < graph.edges.at(e).sourceNodes.size(); i++)
		{
			/* Primary: Number of source nodes  */
			gpuData.edgeData.edgeNodesSourceSize++;

			/* Secondary: NodeCompactList Inc the Node counter also */
			gpuData.nodeData.edgeStartNextsNum[graph.edges.at(e).sourceNodes.at(i)]++;
		}

		/* Each edge will increment its target nodes as a Prev */
		for (uint i = 0; i < graph.edges.at(e).targetNodes.size(); i++)
		{
			gpuData.edgeData.edgeNodesTargetSize++;

			gpuData.nodeData.edgeStartPrevsNum[graph.edges.at(e).targetNodes.at(i)]++;
		}
	}

	/* A1] Compute CSR offsets and total edges per node using prefix sum */
	for (uint n = 0; n < gpuData.nodeData.numNodes; n++)
	{
		gpuData.nodeData.labelIndex[n] = graph.nodeLabelIndex.at(n);

		/* Compact Array of node Prevs Start */
		gpuData.nodeData.edgeStartPrevsStart[n] = gpuData.nodeData.nodeEdgesPrevsSize;
		/* Compact Array of node Prevs Size */
		gpuData.nodeData.nodeEdgesPrevsSize += gpuData.nodeData.edgeStartPrevsNum[n];

		/* Next Array */
		gpuData.nodeData.edgeStartNextsStart[n] = gpuData.nodeData.nodeEdgesNextsSize;
		gpuData.nodeData.nodeEdgesNextsSize += gpuData.nodeData.edgeStartNextsNum[n];

		gpuData.nodeData.totalEdges[n] = gpuData.nodeData.edgeStartPrevsNum[n] + gpuData.nodeData.edgeStartNextsNum[n];

		/* Debug: track max edges for histogram binning */
		if ((gpuData.nodeData.edgeStartPrevsNum[n] + gpuData.nodeData.edgeStartNextsNum[n]) > debugHist[gInd].node.maxEdgesSize)
		{
			debugHist[gInd].node.maxEdgesSize = gpuData.nodeData.edgeStartPrevsNum[n] + gpuData.nodeData.edgeStartNextsNum[n];
		}
	}

	/* A2] Assign IO tags for global input/output nodes */
	for (uint n = 0; n < graph.globalInputs.size(); n++)
	{
		gpuData.nodeData.ioTags[graph.globalInputs.at(n)] = 1;
	}

	for (uint n = 0; n < graph.globalOutputs.size(); n++)
	{
		if (gpuData.nodeData.ioTags[graph.globalOutputs.at(n)] == 0)
		{
			gpuData.nodeData.ioTags[graph.globalOutputs.at(n)] = 2;
		}
		else if (gpuData.nodeData.ioTags[graph.globalOutputs.at(n)] == 1)
		{
			gpuData.nodeData.ioTags[graph.globalOutputs.at(n)] = 3;
		}
		else
		{
			gpuData.nodeData.ioTags[graph.globalOutputs.at(n)]++;
			cout << "ERROR NodeID: " << graph.globalOutputs.at(n) 
			     << " LabelIndex: " << gpuData.nodeData.labelIndex[graph.globalOutputs.at(n)]
			     << " Label: " << graph.nodeLabelsDB.at(gpuData.nodeData.labelIndex[graph.globalOutputs.at(n)]) << endl;
		}
	}
}

/* Allocate and populate compact CSR arrays (B phase)
 * Takes computed sizes from ComputeCompactArrayMetadata as parameters
 * Populates all edge-node and node-edge relationship arrays
 */
static inline void AllocateAndPopulateCompactArrays(
	int gInd,
	const InputGraph& graph,
	GPUGraphData& gpuData,
	GPUGraphData gpuGraphs[2],
	DebugHistogram* debugHist)
{
	/* Allocate node and edge CSR arrays using pre-computed sizes */
	gpuData.nodeData.edgePrevs = new uint[gpuData.nodeData.nodeEdgesPrevsSize]();
	gpuData.nodeData.edgeNexts = new uint[gpuData.nodeData.nodeEdgesNextsSize]();
	gpuData.nodeData.edgePrevsPort = new int[gpuData.nodeData.nodeEdgesPrevsSize]();
	gpuData.nodeData.edgeNextsPort = new int[gpuData.nodeData.nodeEdgesNextsSize]();

	std::fill_n(gpuData.nodeData.edgeNextsPort, gpuData.nodeData.nodeEdgesNextsSize, -1);
	std::fill_n(gpuData.nodeData.edgePrevsPort, gpuData.nodeData.nodeEdgesPrevsSize, -1);

	/* Allocate edge arrays */
	gpuData.edgeData.nodesSources = new uint[gpuData.edgeData.edgeNodesSourceSize]();
	gpuData.edgeData.nodesTargets = new uint[gpuData.edgeData.edgeNodesTargetSize]();

	uint *DEBUGnode_CountSources = new uint[gpuData.nodeData.numNodes]();
	uint *DEBUGnode_CountTargets = new uint[gpuData.nodeData.numNodes]();

	/* Create array pointers for ProcessEdgeNodes compatibility */
	uint* nodeEdgeNextsArray[2] = {gpuGraphs[0].nodeData.edgeNexts, gpuGraphs[1].nodeData.edgeNexts};
	int* nodeEdgeNextsPortArray[2] = {gpuGraphs[0].nodeData.edgeNextsPort, gpuGraphs[1].nodeData.edgeNextsPort};
	uint* nodeEdgePrevsArray[2] = {gpuGraphs[0].nodeData.edgePrevs, gpuGraphs[1].nodeData.edgePrevs};
	int* nodeEdgePrevsPortArray[2] = {gpuGraphs[0].nodeData.edgePrevsPort, gpuGraphs[1].nodeData.edgePrevsPort};
	int* nodeNextsFirstEdgeArray[2] = {gpuGraphs[0].nodeData.nextsFirstEdge, gpuGraphs[1].nodeData.nextsFirstEdge};
	int* nodePrevsFirstEdgeArray[2] = {gpuGraphs[0].nodeData.prevsFirstEdge, gpuGraphs[1].nodeData.prevsFirstEdge};

	int DEBUGedgeCounterSources = 0, DEBUGedgeCounterTargets = 0;
	debugHist[gInd].edge.maxNodesSize = 0;

	/* B1] Loop over sorted edges and populate CSR arrays */
	for (uint e = 0; e < gpuData.edgeData.numEdges; e++)
	{
		gpuData.edgeData.labelIndex[e] = graph.edges.at(e).labelIndex;

		/* 1. Start and Num for Edge Source Nodes */
		gpuData.edgeData.nodeStartSourcesStart[e] = DEBUGedgeCounterSources;
		gpuData.edgeData.nodeStartSourcesNum[e] = graph.edges.at(e).sourceNodes.size();

		/* Process source nodes */
		ProcessEdgeNodes(
			gInd, e,
			graph.edges.at(e).sourceNodes,
			gpuData.edgeData.nodesSources,
			DEBUGedgeCounterSources,
			nodeEdgeNextsArray,
			nodeEdgeNextsPortArray,
			DEBUGnode_CountTargets,
			nodeNextsFirstEdgeArray,
			graph.edges.at(e).labelIndex
		);

		/* 2. Start and Num for Edge Target Nodes */
		gpuData.edgeData.nodeStartTargetsStart[e] = DEBUGedgeCounterTargets;
		gpuData.edgeData.nodeStartTargetsNum[e] = graph.edges.at(e).targetNodes.size();

		/* Process target nodes */
		ProcessEdgeNodes(
			gInd, e,
			graph.edges.at(e).targetNodes,
			gpuData.edgeData.nodesTargets,
			DEBUGedgeCounterTargets,
			nodeEdgePrevsArray,
			nodeEdgePrevsPortArray,
			DEBUGnode_CountSources,
			nodePrevsFirstEdgeArray,
			graph.edges.at(e).labelIndex
		);

		gpuData.edgeData.totalNodes[e] = gpuData.edgeData.nodeStartSourcesNum[e] + gpuData.edgeData.nodeStartTargetsNum[e];

		if ((graph.edges.at(e).sourceNodes.size() + graph.edges.at(e).targetNodes.size()) > debugHist[gInd].edge.maxNodesSize)
		{
			debugHist[gInd].edge.maxNodesSize = graph.edges.at(e).sourceNodes.size() + graph.edges.at(e).targetNodes.size();
		}
	}

	cout << " DEBUG: EdgeSourceCountCSR " << DEBUGedgeCounterSources << " EdgeTargetCountCSR " << DEBUGedgeCounterTargets << endl;

	/* Error Checking */
	for (uint n = 0; n < gpuData.nodeData.numNodes; n++)
	{
		if (gpuData.nodeData.edgeStartPrevsNum[n] != DEBUGnode_CountSources[n])
		{
			cout << n << " Error SourceNodeEdgeMapping Got " << DEBUGnode_CountSources[n]
			     << " Expected " << gpuData.nodeData.edgeStartPrevsNum[n] << endl;
		}

		if (gpuData.nodeData.edgeStartNextsNum[n] != DEBUGnode_CountTargets[n])
		{
			cout << n << " Error TargetNodeEdgeMapping Got " << DEBUGnode_CountTargets[n]
			     << " Expected " << gpuData.nodeData.edgeStartNextsNum[n] << endl;
		}
	}

	/* Cleanup temp counter arrays */
	delete[] DEBUGnode_CountSources;
	delete[] DEBUGnode_CountTargets;
}

/* Allocate histogram arrays for debug statistics */
static inline void AllocateDebugHistograms(DebugHistogram& debugHist)
{
	debugHist.edge.sourceNodeCount = new uint  [debugHist.edge.maxNodesSize +1](); /* Arr13 */
	debugHist.edge.targetNodeCount = new uint  [debugHist.edge.maxNodesSize +1](); /* Arr14 */
	debugHist.edge.totalNodeCount  = new uint  [debugHist.edge.maxNodesSize +1](); /* Arr15 */

	debugHist.node.prevCount   = new uint  [debugHist.node.maxEdgesSize +1](); /* Arr16 */
	debugHist.node.nextCount   = new uint  [debugHist.node.maxEdgesSize +1](); /* Arr17 */
	debugHist.node.totalCount  = new uint  [debugHist.node.maxEdgesSize +1](); /* Arr18 */
	debugHist.node.ioTagCounts = new uint  [debugHist.node.maxEdgesSize +1](); /* Arr19 */
}

/* Deallocate histogram arrays for debug statistics */
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

/* Forward declarations */
void parseGraphJSON_global(std::istream& json_stream, InputGraph& graph, uint& maxNodesPerEdge);

/* Load input graphs from files */
static inline long long LoadGraphs(int argc, char* argv[], InputGraph* graphs, uint& maxNodesPerEdge)
{
	string filenames[2];
	ParseInputFilenames(argc, argv, filenames);

	auto start_io = std::chrono::high_resolution_clock::now();
    for (int gInd = 0;gInd<2;gInd++ )
	{
    	cout<< filenames[gInd]<<endl;
		std::ifstream file_stream(filenames[gInd]);
		if (!file_stream.is_open())
		{
			std::cerr << "Error: Could not open file " << filenames[gInd] << std::endl;
			return -1;
		}
		parseGraphJSON_global(file_stream, graphs[gInd], maxNodesPerEdge);
		file_stream.close();

		std::cout<<" IONodeLabels "<<graphs[gInd].nodeLabelsDB.size()<<" IONodes "<<graphs[gInd].nodeLabelIndex.size()
			<<" IOInputNodes "<<graphs[gInd].globalInputs.size()<<" IOOutputNodes "<<graphs[gInd].globalOutputs.size()
			<<" IOEdgeLabels "<<graphs[gInd].edgeLabelsDB.size()<<" IOEdges "<<graphs[gInd].edges.size()<<std::endl;

		std::cout<<std::endl;
	}
	auto end_io = std::chrono::high_resolution_clock::now();
	auto io_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_io - start_io).count();
	std::cout << "Total I/O and parsing time: " << io_time << " ms" << std::endl;
	
	return io_time;
}

/* Sort edges in input graphs */
static inline void SortGraphEdges(InputGraph* graphs, uint** edgeLabelIndexOrg)
{
    for (int gInd = 0;gInd<2;gInd++ )
	{
		/*-------------------------------------------------------------------------------------------*/
						/* Edge Sorting by key */
		auto start_sort = std::chrono::high_resolution_clock::now();
		std::sort(
			graphs[gInd].edges.begin(),
			graphs[gInd].edges.end(),
			HyperEdgeLess
		);
		auto end_sort = std::chrono::high_resolution_clock::now();
		auto sort_time = std::chrono::duration_cast<std::chrono::microseconds>(end_sort - start_sort).count();
		std::cout << "  Sorting time: " << std::fixed << std::setprecision(3) 
		          << sort_time / 1000.0 << " ms" << std::endl;
		std::cout<<std::endl;
		/*-------------------------------------------------------------------------------------------*/

	    /* Debug edge index mapping */
		if (debugSort) DebugEdgeIndexMapping(gInd, graphs, edgeLabelIndexOrg);
	}
}
/*-------------------------------------------------------------------------------------------------------------------*/
/* IO Mimic by reading a JSON file and creating the compact lists for node and edges */
/*-------------------------------------------------------------------------------------------------------------------*/
void parseGraphJSON_global(std::istream& json_stream, InputGraph& graph, uint& maxNodesPerEdge)
{
	json j;

	// Maps for tracking unique labels
	std::map<std::string, int> node_label_to_index;
	std::map<std::string, int> edge_label_to_index;

	// Clear all vectors in the struct
	graph.nodeLabelsDB.clear();
	graph.nodeLabelIndex.clear();

	graph.edgeLabelsDB.clear();
	graph.edges.clear();
	graph.globalInputs.clear();
	graph.globalOutputs.clear();


	try
	{


		json_stream >> j;

		/*-----------------------------------------------------------------------------*/
		/* 1. Read Node and extract unique labels */
		for (const auto& node_obj : j["nodes"])
		{
			std::string label = node_obj["type_label"];
			int index;
			auto it = node_label_to_index.find(label);

			if (it == node_label_to_index.end())
			{
				graph.nodeLabelsDB.push_back(label);
				index = graph.nodeLabelsDB.size() - 1;
				node_label_to_index[label] = index;
			}
			else
			{
				index = it->second;
			}
			graph.nodeLabelIndex.push_back(index);
		}
		/*-----------------------------------------------------------------------------*/



		/*-----------------------------------------------------------------------------*/
		/* 2. Extract Hyperedges */
		for (const auto& edge_obj : j["hyperedges"])
		{
			IO_hyperEdge edge;
			std::string label_str = edge_obj["type_label"];
			int index;
			auto it = edge_label_to_index.find(label_str);

			if (it == edge_label_to_index.end())
			{
				// It's a new edge label
				graph.edgeLabelsDB.push_back(label_str);
				index = graph.edgeLabelsDB.size() - 1;
				edge_label_to_index[label_str] = index;
			}
			else
			{
				// We've seen this label. Get its stored index.
				index = it->second;
			}

			// Store the index, not the string
			edge.labelIndex = index;
			// --- End Edge Label Logic ---

			/* Debug not used on GPU */
			edge.sourceNodes = edge_obj["source_nodes"].get<std::vector<uint>>();
			edge.targetNodes = edge_obj["target_nodes"].get<std::vector<uint>>();

			if(edge.sourceNodes.size()>maxNodesPerEdge)
			{
				maxNodesPerEdge = edge.sourceNodes.size();
			}

			if(edge.targetNodes.size()>maxNodesPerEdge)
			{
				maxNodesPerEdge = edge.targetNodes.size();
			}

			graph.edges.push_back(edge);
		}
		/*-----------------------------------------------------------------------------*/


		/*-----------------------------------------------------------------------------*/
		/* 3. Extract Global Inputs Nodes */
		graph.globalInputs = j["Inputs"].get<std::vector<uint>>();
		/*-----------------------------------------------------------------------------*/

		/*-----------------------------------------------------------------------------*/
		/* 4. Extract Global Output Nodes */
		graph.globalOutputs = j["Outputs"].get<std::vector<uint>>();
		/*-----------------------------------------------------------------------------*/

	}
	catch (json::parse_error& e)
	{
		std::cerr << "JSON parse error: " << e.what() << std::endl;
	}
	catch (json::type_error& e)
	{
		std::cerr << "JSON type error: " << e.what() << std::endl;
	}
	catch (std::exception& e)
	{
		std::cerr << "An unexpected error occurred: " << e.what() << std::endl;
	}
}
/*-------------------------------------------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------------------------------------------*/
/* Debug Print Stats IO */
/*-------------------------------------------------------------------------------------------------------------------*/
void printGraphStats(   // Node Info
						unsigned int numNodes,
						unsigned int numNodesInput,
						unsigned int numNodesOutput,

						// const uint*  node_LabelDBIndex,	
						unsigned int numNodeLabelsDB,
						// const uint* node_EdgeStartPrevsStart,
						const uint* node_EdgeStartPrevsNum,
						// const uint* node_EdgeStartNextsStart,
						const uint* node_EdgeStartNextsNum,
						const uint*  node_IOTag,

						// Edge Info
						unsigned int numEdges,
						// const uint* edge_LabelDBIndex, 
						unsigned int numEdgeLabelsDB,
						// const uint* edge_NodeStartSourcesStart,
						const uint* edge_NodeStartSourcesNum,
						// const uint* edge_NodeStartTargetsStart,
						const uint* edge_NodeStartTargetsNum,

						/* Debug Histograms */
						const DebugHistogram& debugHist )
{

    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "## 📊 Graph Overall Statistics ##" << std::endl;
    std::cout << std::left << std::setw(25) << "* Total Nodes:" << numNodes << std::endl;
    std::cout << std::left << std::setw(25) << "* Total Edges:" << numEdges << std::endl;
    std::cout << std::left << std::setw(25) << "* Unique Node Labels:" << numNodeLabelsDB << std::endl;
    std::cout << std::left << std::setw(25) << "* Unique Edge Labels:" << numEdgeLabelsDB << std::endl;
    std::cout << std::left << std::setw(25) << "* Global Input Nodes:" << numNodesInput << std::endl;
    std::cout << std::left << std::setw(25) << "* Global Output Nodes:" << numNodesOutput << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;




    /*--------------------------------------------------------------------------------*/
    /* Build Histo for edge props */
    for (unsigned int i = 0; i < numEdges; ++i)
    {
    	debugHist.edge.sourceNodeCount[ edge_NodeStartSourcesNum[i] ]++;
    	debugHist.edge.targetNodeCount[ edge_NodeStartTargetsNum[i] ]++;
    	debugHist.edge.totalNodeCount   [ edge_NodeStartSourcesNum[i] + edge_NodeStartTargetsNum[i] ]++;
    }

    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "## 🔗 Edge Degree Distribution ##" << std::endl;

    std::cout << "EdgeHist-CountSourceNodes" << std::endl;
    for (int i=0; i<(int)debugHist.edge.maxNodesSize+1;i++)
    {
    	if(debugHist.edge.sourceNodeCount[i]>0)
    	{
         std::cout << "* " << std::setw(3) << i << " sourcesNodes: " << debugHist.edge.sourceNodeCount[i] << " count" << std::endl;
    	}
    }
    std::cout << std::endl;

    std::cout << "EdgeHist-CountTargetNodes" << std::endl;
    for (int i=0; i<(int)debugHist.edge.maxNodesSize+1;i++)
	{
		if(debugHist.edge.targetNodeCount[i]>0)
		{
		 std::cout << "* " << std::setw(3) << i << " targetNodes: " << debugHist.edge.targetNodeCount[i] << " count" << std::endl;
		}
	}
    std::cout << std::endl;

    std::cout << "EdgeHist-TotNodes" << std::endl;
    for (int i=0; i<(int)debugHist.edge.maxNodesSize+1;i++)
	{
		if(debugHist.edge.totalNodeCount[i]>0)
		{
		 std::cout << "* " << std::setw(3) << i << " totNodes: " << debugHist.edge.totalNodeCount[i] << " count" << std::endl;
		}
	}
    std::cout << "----------------------------------------------------------------" << std::endl;
    /*--------------------------------------------------------------------------------*/


    /*--------------------------------------------------------------------------------*/
    std::cout << "## ↔️ Node Degree Distribution ##" << std::endl;

    for (unsigned int i = 0; i < numNodes; ++i)
    {
    	debugHist.node.prevCount  [ node_EdgeStartPrevsNum[i] ]++;
    	debugHist.node.nextCount  [ node_EdgeStartNextsNum[i] ]++;
    	debugHist.node.totalCount [ node_EdgeStartPrevsNum[i] + node_EdgeStartNextsNum[i] ]++;
    	debugHist.node.ioTagCounts   [ node_IOTag[i] ]++;
    }

    std::cout << "NodeHist-NodesIO" << std::endl;
    for (int i=0; i<(int)debugHist.node.maxEdgesSize+1;i++)
    {
    	if(debugHist.node.ioTagCounts[i]>0)
    	{
         std::cout << "* " << std::setw(3) << i << "NodesIO " << debugHist.node.ioTagCounts[i] << " count" << std::endl;
    	}
    }
    std::cout << "---" << std::endl;

    std::cout << "NodeHist-NumPrevs" << std::endl;
    for (int i=0; i<(int)debugHist.node.maxEdgesSize+1;i++)
	{
		if(debugHist.node.prevCount[i]>0)
		{
		 std::cout << "* " << std::setw(3) << i << "NodesPrev " << debugHist.node.prevCount[i] << " count" << std::endl;
		}
	}
    std::cout << std::endl;

    std::cout << "NodeHist-NumNexts" << std::endl;
    for (int i=0; i<(int)debugHist.node.maxEdgesSize+1;i++)
	{
		if(debugHist.node.nextCount[i]>0)
		{
		 std::cout << "* " << std::setw(3) << i << "NodesNext " << debugHist.node.nextCount[i] << " count" << std::endl;
		}
	}
    std::cout << std::endl;

    std::cout << "NodeHist-TotEdges" << std::endl;
    for (int i=0; i<(int)debugHist.node.maxEdgesSize+1;i++)
	{
		if(debugHist.node.totalCount[i]>0)
		{
		 std::cout << "* " << std::setw(3) << i << "NodesIO " << debugHist.node.totalCount[i] << " count" << std::endl;
		}
	}
    std::cout << "----------------------------------------------------------------" << std::endl;
    /*--------------------------------------------------------------------------------*/

}
/*-------------------------------------------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------------------------------------------*/
/* Debug Print Stats Connections*/
/*-------------------------------------------------------------------------------------------------------------------*/
void printGraphStatsConn(const InputGraph* graphs, const GPUGraphData* gpuGraphs, DebugHistogram* debugHist)
{
	bool isIso = true;

	/*-------------------------------------------------------------------------------------------*/
	 /* Construct Histogram */
	for (int gInd = 0;gInd<2;gInd++ )
	{
		std::cout <<" HistMaxEdgeBins "<< debugHist[gInd].edge.maxNodesSize<<" HistMaxNodeBins "<< debugHist[gInd].node.maxEdgesSize<<endl;
		std::cout << "\n--- Calling printGraphStats ---\n";
		AllocateDebugHistograms(debugHist[gInd]);


    	printGraphStats(    // Node Args
								gpuGraphs[gInd].nodeData.numNodes, graphs[gInd].globalInputs.size(), graphs[gInd].globalOutputs.size(),
								// gpuGraphs[gInd].nodeData.labelIndex, 
								gpuGraphs[gInd].nodeData.numNodeLabelsDB,
								// gpuGraphs[gInd].nodeData.edgeStartPrevsStart,
								gpuGraphs[gInd].nodeData.edgeStartPrevsNum,
								// gpuGraphs[gInd].nodeData.edgeStartNextsStart,
								gpuGraphs[gInd].nodeData.edgeStartNextsNum,
								gpuGraphs[gInd].nodeData.ioTags,
								// Edge Args
								gpuGraphs[gInd].edgeData.numEdges,
								// gpuGraphs[gInd].edgeData.labelIndex, 
								gpuGraphs[gInd].edgeData.numEdgeLabelsDB,
								// gpuGraphs[gInd].edgeData.nodeStartSourcesStart,
								gpuGraphs[gInd].edgeData.nodeStartSourcesNum,
								// gpuGraphs[gInd].edgeData.nodeStartTargetsStart,
								gpuGraphs[gInd].edgeData.nodeStartTargetsNum,

								/* Pass the entire debug histogram struct */
								debugHist[gInd] );
			std::cout << "--- Finished printGraphStats ---\n";
	 }
	/*-------------------------------------------------------------------------------------------*/

	 /*-------------------------------------------------------------------------------------------*/
		std::cout << " Basic Iso Tests \n";

		if (graphs[0].globalInputs.size() != graphs[1].globalInputs.size())
		{
			std::cout<<"NotIso: GlobalInputCount "<<endl;
			isIso = false;
		}

		if (graphs[0].globalOutputs.size() != graphs[1].globalOutputs.size())
		{
			std::cout<<"NotIso: GlobalOutputCount "<<endl;
			isIso = false;
		}


		if (graphs[0].nodeLabelsDB.size() != graphs[1].nodeLabelsDB.size())
		{
			std::cout<<"NotIso: NodeLabelCount "<<endl;
			isIso = false;
		}

		if (graphs[0].nodeLabelIndex.size() != graphs[1].nodeLabelIndex.size())
		{
			std::cout<<"NotIso: NodeCount "<<endl;
			isIso = false;
		}

		if (graphs[0].edgeLabelsDB.size() != graphs[1].edgeLabelsDB.size())
		{
			std::cout<<"NotIso: EdgeLabelCount "<<endl;
			isIso = false;
		}

		if (graphs[0].edges.size() != graphs[1].edges.size())
		{
			std::cout<<"NotIso: EdgeCount "<<endl;
			isIso = false;
		}
		std::cout << " Basic Iso Tests Done \n";

		if(isIso)
		{
			std::cout<<" Graphs Pass Basic Iso Tests "<<endl;
		}
		/*-------------------------------------------------------------------------------------------*/


	for (int gInd = 0;gInd<2;gInd++ )
	{
		DeallocateDebugHistograms(debugHist[gInd]);
	}
}
/*-------------------------------------------------------------------------------------------------------------------*/



/*-------------------------------------------------------------------------------------------------------------------*/
int main(int argc, char* argv[])
{
	InputGraph IO_graphs[2];
	GPUGraphData gpuGraphs[2] = {};
	DebugHistogram m_DebugHist[2] = {};
	uint MaxNodesPerEdge = 0;
	uint* m_Edge_LabelDBIndexOrg[2] = {};

	auto start_total = std::chrono::high_resolution_clock::now();	
	LoadGraphs(argc, argv, IO_graphs, MaxNodesPerEdge);
	SortGraphEdges(IO_graphs, m_Edge_LabelDBIndexOrg);

	auto start_compact = std::chrono::high_resolution_clock::now();
     for (int gInd = 0;gInd<2;gInd++ )
	 {
    	 cout<<" Create Compact Arrays " <<gInd<<endl;

		/* Initialize GPU graph data structure */
		InitializeGPUGraphData(gInd, IO_graphs[gInd], gpuGraphs[gInd]);
		m_DebugHist[gInd].node.maxEdgesSize = 0; /* Debug host variable for histo on CPU */

		/* Compute metadata: edge counts, CSR offsets, and IO tags (A0 + A1 + A2) */
		ComputeCompactArrayMetadata(gInd, IO_graphs[gInd], gpuGraphs[gInd], m_DebugHist);

		cout<<" EdgeSourceCSR: "<<gpuGraphs[gInd].edgeData.edgeNodesSourceSize<<"  EdgeTargetCSR: "<<gpuGraphs[gInd].edgeData.edgeNodesTargetSize
			<<" NodePrevsCSR:  "<<gpuGraphs[gInd].nodeData.nodeEdgesPrevsSize <<"  NodeNextsCSR:  "<<gpuGraphs[gInd].nodeData.nodeEdgesNextsSize<<endl;

		/* Allocate and populate compact arrays using computed metadata (B phase) */
		cout<<" Create Edge Compact Arrays " <<gInd<<endl;
		AllocateAndPopulateCompactArrays(gInd, IO_graphs[gInd], gpuGraphs[gInd], gpuGraphs, m_DebugHist);
    }
	auto end_compact = std::chrono::high_resolution_clock::now();
	auto compact_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_compact - start_compact).count();
	std::cout << "Compact array creation time: " << compact_time << " ms" << std::endl;
 	/*===========================================================================================*/
 				          /* End Create compact arrays and pass to the GPU */
 	/*===========================================================================================*/


    printGraphStatsConn(IO_graphs, gpuGraphs, m_DebugHist);

	auto start_gpu = std::chrono::high_resolution_clock::now();
    /* Transfer GPU data to device */
    for (int gInd = 0;gInd<2;gInd++ )
	{
    	/* Transfer graph to GPU using organized struct */
    	TransferGraphToGPU(gpuGraphs[gInd]);
    }
	auto end_gpu_init = std::chrono::high_resolution_clock::now();
	auto gpu_init_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu_init - start_gpu).count();
	std::cout << "GPU initialization time: " << gpu_init_time << " ms" << std::endl;

	auto start_gpu_compute = std::chrono::high_resolution_clock::now();
    CreateGraphBinsGPU(); /* hist Binning on GPU */

    if(MaxNodesPerEdge<8)
    {
      printf(" Check EdgeNodes %d \n", MaxNodesPerEdge);
      CompareEdgesGPU();
    }
    else
    {
    	printf(" Cannot use NetworkSort on GPU 8 Exceeded \n");
    }
	auto end_gpu_compute = std::chrono::high_resolution_clock::now();
	auto gpu_compute_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu_compute - start_gpu_compute).count();
	std::cout << "GPU computation time: " << gpu_compute_time << " ms" << std::endl;

    /* TODO WL-1 Test */

    for (int gInd = 0;gInd<2;gInd++ )
	{
      FreeGPUArrays(gInd,0);
	  DeallocateGPUGraphData(gpuGraphs[gInd]);
	  if (m_Edge_LabelDBIndexOrg[gInd] != nullptr)
	  {
	  	delete[] m_Edge_LabelDBIndexOrg[gInd];
	  }
	}

	auto end_total = std::chrono::high_resolution_clock::now();
	auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count();
	std::cout << "\n==================================================" << std::endl;
	std::cout << "Total execution time: " << total_time << " ms" << std::endl;
	std::cout << "==================================================" << std::endl;

	return 0;
}
