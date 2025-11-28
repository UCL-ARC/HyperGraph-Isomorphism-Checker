
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
InputGraph IO_graphs[2];

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

/* Debug histograms [2] for two graphs */
DebugHistogram m_DebugHist[2] = {};
/*-------------------------------------------------------------------------------------------------------------------*/




/*-------------------------------------------------------------------------------------------------------------------*/
/* A] Node Struct compact list that we will copy to GPU */
/*-------------------------------------------------------------------------------------------------------------------*/
uint m_numNodes           [2] = {};                                 /* Node Total  */
uint m_numNodeLabelsDB    [2] = {};                                 /* Node Type Total  */
uint m_nodeEdgesPrevsSize [2] = {}, m_nodeEdgesNextsSize[2] = {};   /* Size of the compact arrays for node edges */

/* Per Node Array Storage */
uint  *m_Node_LabelDBIndex [2];       /* 1] index of the label that identifies the node  */
uint  *m_Node_IOTag        [2];       /* 2] 0 none 1 GInput 2 GOut  3 Both */

uint *m_Node_EdgeStartPrevsNum   [2]; /* 3] count in node_EdgePrevs array  */
uint *m_Node_EdgeStartNextsNum   [2]; /* 4] count in node_EdgeNexts array  */
uint *m_Node_TotEdges            [2]; /* 5] Sum of Next and Prevs */
uint *m_Node_EdgeStartPrevsStart [2]; /* 6] start index in node_EdgePrevs array  */
uint *m_Node_EdgeStartNextsStart [2]; /* 7] start index in node_EdgeNexts array  */


/* Each node will write its input and output edges into these compact arrays */
uint  *m_Node_EdgePrevs          [2]; /* 8] CSR "From Edge Sources " */
uint  *m_Node_EdgeNexts          [2]; /* 9] CSR "From Edge Targets " */

/* Node Edge Connections */
int  *m_Node_EdgePrevsPort       [2]; /* 10] CSR ports for edge connections */
int  *m_Node_EdgeNextsPort      [2]; /* 11] CSR ports for edge connections  " */

int  *m_Node_PrevsFirstEdge      [2];  /* 12] used for signature */
int  *m_Node_NextsFirstEdge      [2];  /* 13] used for signature */


/*-------------------------------------------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------------------------------------------*/
/* B] Edge struct compact list */
/*-------------------------------------------------------------------------------------------------------------------*/
uint m_numEdges[2]={};                                          /* Edge Total */
uint m_numEdgeLabelsDB[2] = {};                                 /* Edge Type Total */
uint m_EdgeNodesSourceSize[2] = {}, m_edgeNodesTargetSize[2] = {}; /* Size of the compact arrays for edge nodes */

/* Per Edge Storage */
uint  *m_Edge_LabelDBIndex         [2]; /* 14] index of the label that identifies the node  */

/* Edge Node Connections */
uint *m_Edge_NodeStartSourcesNum   [2]; /* 15] start index in edge_NodesSources array  */
uint *m_Edge_NodeStartTargetsNum   [2]; /* 16] count in edge_NodesTargets array  */
uint *m_Edge_TotNodes              [2]; /* 17] Sum of Next and Prevs */
uint *m_Edge_NodeStartSourcesStart [2]; /* 18] start index in edge_NodesSources array  */
uint *m_Edge_NodeStartTargetsStart [2]; /* 19] count in edge_NodesTargets array  */

/* Each edge will write its source and target nodes into these compact arrays */
uint *m_Edge_NodesSources          [2]; /* 20] CSR Source Node List */
uint *m_Edge_NodesTargets          [2]; /* 21] CSR Target Node List */

uint MaxNodesPerEdge = 0;

uint  *m_Edge_LabelDBIndexOrg         [2]; /* unsorted */

/*-------------------------------------------------------------------------------------------------------------------*/

/* Debug helper: builds a permutation index for edges and prints the mapping.
 * NOTE: If called after IO_edges[gInd] is sorted, permutation will be identity.
 * To get original->sorted positions, invoke before sorting or capture the original order. */
static void DebugEdgeIndexMapping(int gInd)
{
	uint numEdgesS = IO_graphs[gInd].edges.size();
	printf(" EdgeSortIndex %u \n", numEdgesS);

	// Allocate index array (identity permutation)
	m_Edge_LabelDBIndexOrg[gInd] = new uint[numEdgesS]();
	for (uint i = 0; i < numEdgesS; ++i)
	{
		m_Edge_LabelDBIndexOrg[gInd][i] = i;
	}

	// Sort the index array using the same comparator (redundant post edge sort)
	const auto& edges_to_compare = IO_graphs[gInd].edges;
	std::sort(
		m_Edge_LabelDBIndexOrg[gInd],
		m_Edge_LabelDBIndexOrg[gInd] + numEdgesS,
		[&edges_to_compare](uint ia, uint ib)
		{
			return HyperEdgeLess(edges_to_compare[ia], edges_to_compare[ib]);
		}
	);

	// Print mapping
	for (uint i = 0; i < numEdgesS; ++i)
	{
		printf(" %u EdgeLabMap %u \n", i, m_Edge_LabelDBIndexOrg[gInd][i]);
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


/*-------------------------------------------------------------------------------------------------------------------*/
/* IO Mimic by reading a JSON file and creating the compact lists for node and edges */
/*-------------------------------------------------------------------------------------------------------------------*/
void parseGraphJSON_global(std::istream& json_stream, InputGraph& graph)
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

			if(edge.sourceNodes.size()>MaxNodesPerEdge)
			{
				MaxNodesPerEdge = edge.sourceNodes.size();
			}

			if(edge.targetNodes.size()>MaxNodesPerEdge)
			{
				MaxNodesPerEdge = edge.targetNodes.size();
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
void printGraphStatsConn()
{
	bool isIso = true;

	/*-------------------------------------------------------------------------------------------*/
	 /* Construct Histogram */
	for (int gInd = 0;gInd<2;gInd++ )
	{
		std::cout <<" HistMaxEdgeBins "<< m_DebugHist[gInd].edge.maxNodesSize<<" HistMaxNodeBins "<< m_DebugHist[gInd].node.maxEdgesSize<<endl;
		std::cout << "\n--- Calling printGraphStats ---\n";
		m_DebugHist[gInd].edge.sourceNodeCount = new uint  [m_DebugHist[gInd].edge.maxNodesSize +1](); /* Arr13 */
		m_DebugHist[gInd].edge.targetNodeCount = new uint  [m_DebugHist[gInd].edge.maxNodesSize +1](); /* Arr14 */
		m_DebugHist[gInd].edge.totalNodeCount  = new uint  [m_DebugHist[gInd].edge.maxNodesSize +1](); /* Arr15 */

		m_DebugHist[gInd].node.prevCount   = new uint  [m_DebugHist[gInd].node.maxEdgesSize +1](); /* Arr16 */
		m_DebugHist[gInd].node.nextCount   = new uint  [m_DebugHist[gInd].node.maxEdgesSize +1](); /* Arr17 */
		m_DebugHist[gInd].node.totalCount  = new uint  [m_DebugHist[gInd].node.maxEdgesSize +1](); /* Arr18 */
		m_DebugHist[gInd].node.ioTagCounts = new uint  [m_DebugHist[gInd].node.maxEdgesSize +1](); /* Arr19 */


    	printGraphStats(    // Node Args
								m_numNodes[gInd], IO_graphs[gInd].globalInputs.size(), IO_graphs[gInd].globalOutputs.size(),
								// m_Node_LabelDBIndex[gInd], 
								m_numNodeLabelsDB[gInd],
								// m_Node_EdgeStartPrevsStart[gInd],
								m_Node_EdgeStartPrevsNum[gInd],
								// m_Node_EdgeStartNextsStart[gInd],
								m_Node_EdgeStartNextsNum[gInd],
								m_Node_IOTag[gInd],
								// Edge Args
								m_numEdges[gInd],
								// m_Edge_LabelDBIndex[gInd],
								m_numEdgeLabelsDB[gInd],
								// m_Edge_NodeStartSourcesStart[gInd],
								m_Edge_NodeStartSourcesNum[gInd],
								// m_Edge_NodeStartTargetsStart[gInd],
								m_Edge_NodeStartTargetsNum[gInd],

								/* Pass the entire debug histogram struct */
								m_DebugHist[gInd] );
			std::cout << "--- Finished printGraphStats ---\n";
	 }
	/*-------------------------------------------------------------------------------------------*/

	 /*-------------------------------------------------------------------------------------------*/
		std::cout << " Basic Iso Tests \n";

		if (IO_graphs[0].globalInputs.size() != IO_graphs[1].globalInputs.size())
		{
			std::cout<<"NotIso: GlobalInputCount "<<endl;
			isIso = false;
		}

		if (IO_graphs[0].globalOutputs.size() != IO_graphs[1].globalOutputs.size())
		{
			std::cout<<"NotIso: GlobalOutputCount "<<endl;
			isIso = false;
		}


		if (IO_graphs[0].nodeLabelsDB.size() != IO_graphs[1].nodeLabelsDB.size())
		{
			std::cout<<"NotIso: NodeLabelCount "<<endl;
			isIso = false;
		}

		if (IO_graphs[0].nodeLabelIndex.size() != IO_graphs[1].nodeLabelIndex.size())
		{
			std::cout<<"NotIso: NodeCount "<<endl;
			isIso = false;
		}

		if (IO_graphs[0].edgeLabelsDB.size() != IO_graphs[1].edgeLabelsDB.size())
		{
			std::cout<<"NotIso: EdgeLabelCount "<<endl;
			isIso = false;
		}

		if (IO_graphs[0].edges.size() != IO_graphs[1].edges.size())
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
		delete [] m_DebugHist[gInd].edge.sourceNodeCount;
		delete [] m_DebugHist[gInd].edge.targetNodeCount;
		delete [] m_DebugHist[gInd].edge.totalNodeCount;

		delete [] m_DebugHist[gInd].node.prevCount;
		delete [] m_DebugHist[gInd].node.nextCount;
		delete [] m_DebugHist[gInd].node.totalCount;
		delete [] m_DebugHist[gInd].node.ioTagCounts;
	}
}
/*-------------------------------------------------------------------------------------------------------------------*/



/*-------------------------------------------------------------------------------------------------------------------*/
int main(int argc, char* argv[])
{

	/*===========================================================================================*/
	                                   /* _Mimic Input_ */
	/*===========================================================================================*/
	auto start_total = std::chrono::high_resolution_clock::now();
	
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
			return 1;
		}
		parseGraphJSON_global(file_stream, IO_graphs[gInd]);
		file_stream.close();

		std::cout<<" IONodeLabels "<<IO_graphs[gInd].nodeLabelsDB.size()<<" IONodes "<<IO_graphs[gInd].nodeLabelIndex.size()
			<<" IOInputNodes "<<IO_graphs[gInd].globalInputs.size()<<" IOOutputNodes "<<IO_graphs[gInd].globalOutputs.size()
			<<" IOEdgeLabels "<<IO_graphs[gInd].edgeLabelsDB.size()<<" IOEdges "<<IO_graphs[gInd].edges.size()<<std::endl;

		std::cout<<std::endl;
		
		/*-------------------------------------------------------------------------------------------*/
							/* Edge Sorting by key */
		auto start_sort = std::chrono::high_resolution_clock::now();
		std::sort(
			IO_graphs[gInd].edges.begin(),
			IO_graphs[gInd].edges.end(),
			HyperEdgeLess
		);
		auto end_sort = std::chrono::high_resolution_clock::now();
		auto sort_time = std::chrono::duration_cast<std::chrono::microseconds>(end_sort - start_sort).count();
		std::cout << "  Sorting time: " << std::fixed << std::setprecision(3) 
		          << sort_time / 1000.0 << " ms" << std::endl;
		std::cout<<std::endl;
		/*-------------------------------------------------------------------------------------------*/

	    /* Debug edge index mapping */
		if (debugSort) DebugEdgeIndexMapping(gInd);
	}
	auto end_io = std::chrono::high_resolution_clock::now();
	auto io_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_io - start_io).count();
	std::cout << "Total I/O and parsing time: " << io_time << " ms" << std::endl;
										  /* End Edge Sorting */
	/*===========================================================================================*/
									/* End _Mimic Input_ */
	/*===========================================================================================*/


	/*===========================================================================================*/
				          /* Create compact arrays and pass to the GPU */
	/*===========================================================================================*/
	auto start_compact = std::chrono::high_resolution_clock::now();
     for (int gInd = 0;gInd<2;gInd++ )
	 {
    	 cout<<" Create Compact Arrays " <<gInd<<endl;
 		/*-------------------------------------------------------------------------------------------*/
 		/* Set Global Vars to IO Value for nodes */
 		m_numNodes        [gInd] = IO_graphs[gInd].nodeLabelIndex.size();
 		m_numNodeLabelsDB [gInd] = IO_graphs[gInd].nodeLabelsDB.size();


		/* Set Global Vars to IO Value for edges */
		m_numEdges        [gInd]  = IO_graphs[gInd].edges.size();
		m_numEdgeLabelsDB [gInd]  = IO_graphs[gInd].edgeLabelsDB.size();
		/*-------------------------------------------------------------------------------------------*/


		/* ** Note: This can be done when reading/creating the graph input, saving this iteration */

		/*-------------------------------------------------------------------------------------------*/
		/* A0] Create Compact List for nodes to store the edges it connects by incrementing each nodes counter  */

		m_Node_LabelDBIndex        [gInd] = new uint [m_numNodes[gInd]](); /* Arr 1] */
		m_Node_IOTag               [gInd] = new uint [m_numNodes[gInd]](); /* Arr 2] */

		m_Node_EdgeStartPrevsNum   [gInd] = new uint [m_numNodes[gInd]](); /* Arr 3] */
		m_Node_EdgeStartNextsNum   [gInd] = new uint [m_numNodes[gInd]](); /* Arr 4] */
		m_Node_TotEdges            [gInd] = new uint [m_numNodes[gInd]](); /* Arr 5] */
		m_Node_EdgeStartPrevsStart [gInd] = new uint [m_numNodes[gInd]](); /* Arr 6] */
		m_Node_EdgeStartNextsStart [gInd] = new uint [m_numNodes[gInd]](); /* Arr 7] */

		m_Node_PrevsFirstEdge      [gInd] = new int [m_numNodes[gInd]]();  /* Arr 12] */
		m_Node_NextsFirstEdge      [gInd] = new int [m_numNodes[gInd]]();  /* Arr 13] */
		std::fill_n(m_Node_PrevsFirstEdge[gInd], m_numNodes[gInd], -1);
		std::fill_n(m_Node_NextsFirstEdge[gInd], m_numNodes[gInd], -1);

		/* Loop over edges */
		for (uint e=0;e<m_numEdges[gInd];e++)
		{
			/* Each edge will increment its source nodes as a Next */
			for(uint i=0; i<IO_graphs[gInd].edges.at(e).sourceNodes.size(); i++)
			{
				/* Primary: Number of source nodes  */
				m_EdgeNodesSourceSize    [gInd] ++;

				/* Secondary: NodeCompactList Inc the Node counter also */
				m_Node_EdgeStartNextsNum [gInd] [ IO_graphs[gInd].edges.at(e).sourceNodes.at(i) ]++;
			}

			/* Each edge will increment its target nodes as a Prev */
			for(uint i=0;i<IO_graphs[gInd].edges.at(e).targetNodes.size();i++)
			{
				m_edgeNodesTargetSize [gInd] ++;

				m_Node_EdgeStartPrevsNum [gInd] [ IO_graphs[gInd].edges.at(e).targetNodes.at(i) ]++;
			}
		}
		/*-------------------------------------------------------------------------------------------*/

		m_DebugHist[gInd].node.maxEdgesSize = 0; /* Debug host varible for histo on CPU */

		/*-------------------------------------------------------------------------------------------*/
		/* A1] Loop over all nodes and complete the locations of where each needs to read its "next" and "prev" from using a running sum */
		for (uint n=0;n<m_numNodes[gInd];n++)
		{
			m_Node_LabelDBIndex [gInd] [n] = IO_graphs[gInd].nodeLabelIndex.at(n);

			/* Compact Array of node Prevs Start */
			m_Node_EdgeStartPrevsStart [gInd] [n] = m_nodeEdgesPrevsSize[gInd];
			/* Compact Array of node Prevs Size */
			m_nodeEdgesPrevsSize       [gInd]    += m_Node_EdgeStartPrevsNum[gInd][n];

			/* Next Array*/
			m_Node_EdgeStartNextsStart [gInd][n] = m_nodeEdgesNextsSize[gInd];
			m_nodeEdgesNextsSize       [gInd]   += m_Node_EdgeStartNextsNum [gInd] [n];

			m_Node_TotEdges            [gInd][n] = m_Node_EdgeStartPrevsNum[gInd][n] + m_Node_EdgeStartNextsNum [gInd] [n]; /* Total Counter for easy hashing */

			/* Debug For host binning stats find the node with the most edges */
			if ( (m_Node_EdgeStartPrevsNum [gInd] [n] + m_Node_EdgeStartNextsNum [gInd] [n])> m_DebugHist[gInd].node.maxEdgesSize)
			{
				m_DebugHist[gInd].node.maxEdgesSize = m_Node_EdgeStartPrevsNum [gInd] [n] + m_Node_EdgeStartNextsNum [gInd] [n];
			}
		}
		cout<<" EdgeSourceCSR: "<<m_EdgeNodesSourceSize[gInd]<<"  EdgeTargetCSR: "<<m_edgeNodesTargetSize[gInd]
			<<" NodePrevsCSR:  "<<m_nodeEdgesPrevsSize[gInd] <<"  NodeNextsCSR:  "<<m_nodeEdgesNextsSize[gInd]<<endl;


		/*-------------------------------------------------------------------------------------------*/
		/* A2 IO Tag sent node status */
		// int isError = -1;
		for (uint n=0;n<IO_graphs[gInd].globalInputs.size();n++)
		{
			m_Node_IOTag[gInd][IO_graphs[gInd].globalInputs.at(n)] = 1;
		}

		for (uint n=0;n<IO_graphs[gInd].globalOutputs.size();n++)
	    {
		  if(m_Node_IOTag[gInd][IO_graphs[gInd].globalOutputs.at(n)]==0)
		  {
		    m_Node_IOTag[gInd][IO_graphs[gInd].globalOutputs.at(n)] = 2;
		  }
		  else if(m_Node_IOTag[gInd][IO_graphs[gInd].globalOutputs.at(n)]==1)
		  {
			  m_Node_IOTag[gInd][IO_graphs[gInd].globalOutputs.at(n)]=3;
		  }
		  else
		  {
			 m_Node_IOTag[gInd][IO_graphs[gInd].globalOutputs.at(n)]++;
			 cout<<"ERROR NodeID: "<<IO_graphs[gInd].globalOutputs.at(n)<<" LabelIndex: "<<m_Node_LabelDBIndex[gInd][IO_graphs[gInd].globalOutputs.at(n)]
																 <<" Label: "<< IO_graphs[gInd].nodeLabelsDB.at( m_Node_LabelDBIndex[gInd][IO_graphs[gInd].globalOutputs.at(n)]  )<<endl;
		  }
		}
		/*-------------------------------------------------------------------------------------------*/



		/*-------------------------------------------------------------------------------------------*/
				  /* B] Populate compact list for edges and nodes by looping over edges   */
		 cout<<" Create Edge Compact Arrays " <<gInd<<endl;

		/*-------------------------------------------------------------------------------------------*/
		/* Used for the node running sum to store elements and also a debug counter vs A1] values*/
		m_Node_EdgePrevs      [gInd] = new uint [m_nodeEdgesPrevsSize[gInd]](); /* Arr 8]  */
		m_Node_EdgeNexts      [gInd] = new uint [m_nodeEdgesNextsSize[gInd]](); /* Arr 9]  */
		m_Node_EdgeNextsPort  [gInd] = new int  [m_nodeEdgesNextsSize[gInd]](); /* Arr 10] */
		m_Node_EdgePrevsPort  [gInd] = new int  [m_nodeEdgesPrevsSize[gInd]](); /* Arr 11] */

		std::fill_n(m_Node_EdgeNextsPort[gInd], m_nodeEdgesNextsSize[gInd], -1);
		std::fill_n(m_Node_EdgePrevsPort[gInd], m_nodeEdgesPrevsSize[gInd], -1);

		uint *DEBUGnode_CountSources = new uint [m_numNodes[gInd]](); /* Arr 22] */
		uint *DEBUGnode_CountTargets = new uint [m_numNodes[gInd]](); /* Arr 23] */
		/*-------------------------------------------------------------------------------------------*/



		/*-------------------------------------------------------------------------------------------*/
		/* Populate Edges */
		m_Edge_LabelDBIndex          [gInd] = new uint [m_numEdges[gInd]](); /* Arr 14] */

		m_Edge_NodeStartSourcesNum   [gInd] = new uint [m_numEdges[gInd]](); /* Arr 15] */
		m_Edge_NodeStartTargetsNum   [gInd] = new uint [m_numEdges[gInd]](); /* Arr 16] */

		m_Edge_TotNodes              [gInd] = new uint [m_numEdges[gInd]](); /* Arr 17] */
		m_Edge_NodeStartSourcesStart [gInd] = new uint [m_numEdges[gInd]](); /* Arr 18] */
		m_Edge_NodeStartTargetsStart [gInd] = new uint [m_numEdges[gInd]](); /* Arr 19]  */

		m_Edge_NodesSources          [gInd] = new uint [m_EdgeNodesSourceSize[gInd]]; /* Arr 20]  */
		m_Edge_NodesTargets          [gInd] = new uint [m_edgeNodesTargetSize[gInd]]; /* Arr 21] */
		/*-------------------------------------------------------------------------------------------*/


		int DEBUGedgeCounterSources ={}, DEBUGedgeCounterTargets={}; /* Local Counter but also used as DEBUG to check counters match */

		 m_DebugHist[gInd].edge.maxNodesSize =0;
		/* B1] Loop over sorted edges */
		for (uint e=0;e<m_numEdges[gInd];e++)
		{
			m_Edge_LabelDBIndex [gInd] [e] = IO_graphs[gInd].edges.at(e).labelIndex;

			/* 1. Start and Num for Edge Source Nodes */
			m_Edge_NodeStartSourcesStart[gInd][e]  = DEBUGedgeCounterSources;
			m_Edge_NodeStartSourcesNum  [gInd][e]  = IO_graphs[gInd].edges.at(e).sourceNodes.size();

			/* Process source nodes */
			ProcessEdgeNodes(
				gInd, e,
				IO_graphs[gInd].edges.at(e).sourceNodes,
				m_Edge_NodesSources[gInd],
				DEBUGedgeCounterSources,
				m_Node_EdgeNexts,
				m_Node_EdgeNextsPort,
				DEBUGnode_CountTargets,
				m_Node_NextsFirstEdge,
				IO_graphs[gInd].edges.at(e).labelIndex
			);

			/* 2. Start and Num for Edge Target Nodes */
			m_Edge_NodeStartTargetsStart[gInd][e] = DEBUGedgeCounterTargets;
			m_Edge_NodeStartTargetsNum[gInd][e]  = IO_graphs[gInd].edges.at(e).targetNodes.size();

			/* Process target nodes */
			ProcessEdgeNodes(
				gInd, e,
				IO_graphs[gInd].edges.at(e).targetNodes,
				m_Edge_NodesTargets[gInd],
				DEBUGedgeCounterTargets,
				m_Node_EdgePrevs,
				m_Node_EdgePrevsPort,
				DEBUGnode_CountSources,
				m_Node_PrevsFirstEdge,
				IO_graphs[gInd].edges.at(e).labelIndex
			);

			m_Edge_TotNodes          [gInd][e] = m_Edge_NodeStartSourcesNum  [gInd][e] + m_Edge_NodeStartTargetsNum[gInd][e] ;

			if ( (IO_graphs[gInd].edges.at(e).sourceNodes.size() + IO_graphs[gInd].edges.at(e).targetNodes.size() )> m_DebugHist[gInd].edge.maxNodesSize)
			{
				m_DebugHist[gInd].edge.maxNodesSize = IO_graphs[gInd].edges.at(e).sourceNodes.size() + IO_graphs[gInd].edges.at(e).targetNodes.size();
			}

		}
		cout<<" DEBUG: EdgeSourceCountCSR "<<DEBUGedgeCounterSources<<" EdgeTargetCountCSR "<<DEBUGedgeCounterTargets<<endl;

		         /* End B] Populate compact list for edges and nodes by looping over edges   */
	    /*-------------------------------------------------------------------------------------------*/



		/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
		                                  /* Error Checking  */
		for (uint n=0;n<m_numNodes[gInd];n++)
		{
			if( m_Node_EdgeStartPrevsNum[gInd][n] != DEBUGnode_CountSources[n])
			{
				cout<<n<<" Error SourceNodeEdgeMapping Got "<<DEBUGnode_CountSources[n]<<" Expected "<<m_Node_EdgeStartPrevsNum[gInd][n] <<endl;
			}

			if( m_Node_EdgeStartNextsNum[gInd][n] != DEBUGnode_CountTargets[n])
			{
				cout<<n<<" Error TargetNodeEdgeMapping Got "<<DEBUGnode_CountTargets[n]<<" Expected "<<m_Node_EdgeStartNextsNum[gInd][n] <<endl;
			}
		}
		/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/



		/* Temp CounterArrays */
		delete [] DEBUGnode_CountSources;
		delete [] DEBUGnode_CountTargets;
    }
	auto end_compact = std::chrono::high_resolution_clock::now();
	auto compact_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_compact - start_compact).count();
	std::cout << "Compact array creation time: " << compact_time << " ms" << std::endl;
 	/*===========================================================================================*/
 				          /* End Create compact arrays and pass to the GPU */
 	/*===========================================================================================*/


    printGraphStatsConn();

	auto start_gpu = std::chrono::high_resolution_clock::now();
    /* Free CPU Memory */
    for (int gInd = 0;gInd<2;gInd++ )
	{

    	/* Copy to GPU */
    	InitGPUArrays( gInd,
    			       m_numNodes[gInd], m_Node_LabelDBIndex[gInd],
					   m_Node_PrevsFirstEdge[gInd], m_Node_NextsFirstEdge[gInd],
					   m_nodeEdgesPrevsSize[gInd],
					   m_nodeEdgesNextsSize[gInd],

					   m_Node_EdgePrevs[gInd],
					   m_Node_EdgeNexts[gInd],
					   m_Node_EdgePrevsPort[gInd], m_Node_EdgeNextsPort[gInd],

					   m_Node_EdgeStartPrevsStart[gInd],
					   m_Node_EdgeStartPrevsNum[gInd],
					   m_Node_EdgeStartNextsStart[gInd],
					   m_Node_EdgeStartNextsNum[gInd],
					   m_Node_TotEdges[gInd],
					   m_Node_IOTag[gInd],

                       m_numEdges[gInd],
					   m_Edge_LabelDBIndex[gInd],

					   m_EdgeNodesSourceSize[gInd],
					   m_edgeNodesTargetSize[gInd],

					   m_Edge_NodesSources[gInd],
					   m_Edge_NodesTargets[gInd],

					   m_Edge_NodeStartSourcesStart[gInd],
					   m_Edge_NodeStartSourcesNum[gInd],

					   m_Edge_NodeStartTargetsStart[gInd],
					   m_Edge_NodeStartTargetsNum[gInd],

					   m_Edge_TotNodes[gInd],
					   0 );

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


		delete [] m_Edge_LabelDBIndex[gInd];
		delete [] m_Edge_TotNodes[gInd];
		delete [] m_Edge_NodesSources[gInd];
		delete [] m_Edge_NodesTargets[gInd];
		delete [] m_Edge_NodeStartSourcesNum[gInd];
		delete [] m_Edge_NodeStartTargetsNum[gInd];
		delete [] m_Edge_NodeStartSourcesStart[gInd];
		delete [] m_Edge_NodeStartTargetsStart[gInd];

		delete [] m_Node_LabelDBIndex[gInd];
		delete [] m_Node_IOTag[gInd];
		delete [] m_Node_TotEdges[gInd];
		delete [] m_Node_EdgePrevs[gInd];
		delete [] m_Node_EdgeNexts[gInd];

		delete [] m_Node_EdgeNextsPort[gInd];
		delete [] m_Node_EdgePrevsPort[gInd];

		delete [] m_Node_PrevsFirstEdge[gInd];
		delete [] m_Node_NextsFirstEdge[gInd];

		delete [] m_Node_EdgeStartPrevsNum[gInd];
		delete [] m_Node_EdgeStartNextsNum[gInd];

		delete [] m_Node_EdgeStartPrevsStart[gInd];
		delete [] m_Node_EdgeStartNextsStart[gInd];
	}

	auto end_total = std::chrono::high_resolution_clock::now();
	auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count();
	std::cout << "\n==================================================" << std::endl;
	std::cout << "Total execution time: " << total_time << " ms" << std::endl;
	std::cout << "==================================================" << std::endl;

	return 0;
}
