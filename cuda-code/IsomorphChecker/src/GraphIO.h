/*
 * GraphIO.h
 *
 *  Created on: Dec 1, 2025
 *      Author: blaze
 */

#ifndef GRAPHIO_H_
#define GRAPHIO_H_


/* XML Reader for C++ */
using json = nlohmann::json;
using namespace std;

/*-------------------------------------------------------------------------------------------------------------------*/
/**
 * @brief Container for all input graph data from JSON parsing.
 *
 * Organizes a complete hypergraph representation including node labels, edge definitions,
 * and global input/output node designations. This structure serves as the intermediate
 * representation between JSON file parsing and GPU data marshaling.
 *
 * @member nodeLabelsDB Database of unique node label strings
 * @member nodeLabelIndex Label index assigned to each node
 * @member edgeLabelsDB Database of unique edge label strings
 * @member edges Vector of hyperedges (each can connect multiple source/target nodes)
 * @member globalInputs Node IDs designated as graph inputs
 * @member globalOutputs Node IDs designated as graph outputs
 */


bool m_PrintDebugSortEID = false;

/*===================================================================================================================*/
/**
 * @brief Comparator for hyperedges using multi-level sort criteria.
 *
 * Sorts hyperedges by:
 * 1. Number of source nodes (ascending)
 * 2. Number of target nodes (ascending)
 * 3. Total node count (ascending)
 * 4. Label index (ascending)
 *
 * This ordering is used consistently throughout the codebase for deterministic edge sorting.
 *
 * @param a First hyperedge to compare
 * @param b Second hyperedge to compare
 * @return true if a should come before b in sorted order, false otherwise
 */
static inline bool HyperEdgeLessThanCompare(const IO_hyperEdge& a, const IO_hyperEdge& b)
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
/*===================================================================================================================*/





/*===================================================================================================================*/
/** Debug helper: builds a permutation index for edges and prints the mapping.
 * NOTE: If called after graph edges are sorted, permutation will be identity.
 * To get original->sorted positions, invoke before sorting or capture the original order. */

/**
 * @brief Debug helper to build a permutation index for edges and print the mapping.
 *
 * The function builds a permutation index that maps original edge positions to their sorted positions.
 * @note If called after graph edges are sorted, permutation will be identity.
 *
 * @param graph The input graph containing edges.
 * @param edgeLabelIndexOrg Output array to store the original edge indices.
 */
 static void DebugEdgeIndexMappingPrint(const InputGraph& graph, uint*& edgeLabelIndexOrg)
{
	uint numEdgesS = graph.edges.size();
	printf(" EdgeSortIndex %u \n", numEdgesS);

	// Allocate index array (identity permutation)
	edgeLabelIndexOrg = new uint[numEdgesS]();
	for (uint i = 0; i < numEdgesS; ++i)
	{
		edgeLabelIndexOrg[i] = i;
	}

	// Sort the index array using the same comparator (redundant post edge sort)
	const auto& edges_to_compare = graph.edges;
	std::sort(
		edgeLabelIndexOrg,
		edgeLabelIndexOrg + numEdgesS,
		[&edges_to_compare](uint ia, uint ib)
		{
			return HyperEdgeLessThanCompare(edges_to_compare[ia], edges_to_compare[ib]);
		}
	);

	// Print mapping
	for (uint i = 0; i < numEdgesS; ++i)
	{
		printf(" %u EdgeLabMap %u \n", i, edgeLabelIndexOrg[i]);
	}
}
/*===================================================================================================================*/


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

/*===================================================================================================================*/
/** Parse command-line arguments for input filenames, with fallback to defaults */
/*===================================================================================================================*/
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
		filenames[0] = "../../Input/DrugALarge.json";
		filenames[1] = "../../Input/DrugBLarge.json";

		if (argc > 1)
		{
			std::cerr << "Usage: " << argv[0] << " <graph1.json> <graph2.json>" << std::endl;
			std::cerr << "Using default input files." << std::endl;
		}
	}
}
/*===================================================================================================================*/

/*===================================================================================================================*/
/** IO Mimic by reading a JSON file and creating the compact lists for node and edges */
/*===================================================================================================================*/
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
		/** 1. Read Node and extract unique labels */
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
		/** 2. Extract Hyperedges */
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
				index = it->second;
			}
			edge.labelIndex = index;


			/** Debug not used on GPU */
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
		/** 3. Extract Global Inputs Nodes */
		graph.globalInputs = j["Inputs"].get<std::vector<uint>>();
		/*-----------------------------------------------------------------------------*/

		/*-----------------------------------------------------------------------------*/
		/** 4. Extract Global Output Nodes */
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
/*===================================================================================================================*/


/*===================================================================================================================*/
/** Load input graphs from user input files or the default files we provide for unit test */
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
/*===================================================================================================================*/

/*===================================================================================================================*/
/** Sort edges in input graphs */
static inline void SortGraphEdges(InputGraph* graphs, uint** edgeLabelIndexOrg)
{
    for (int gInd = 0;gInd<2;gInd++ )
	{
		/*-------------------------------------------------------------------------------------------*/
						/** Edge Sorting by key */
		auto start_sort = std::chrono::high_resolution_clock::now();
		std::sort(
			graphs[gInd].edges.begin(),
			graphs[gInd].edges.end(),
			HyperEdgeLessThanCompare
		);
		auto end_sort = std::chrono::high_resolution_clock::now();
		auto sort_time = std::chrono::duration_cast<std::chrono::microseconds>(end_sort - start_sort).count();
		std::cout << "  Sorting time: " << std::fixed << std::setprecision(3)
		          << sort_time / 1000.0 << " ms" << std::endl;
		std::cout<<std::endl;
		/*-------------------------------------------------------------------------------------------*/

	    /** Debug edge index mapping */
		if (m_PrintDebugSortEID){ DebugEdgeIndexMappingPrint(graphs[gInd], edgeLabelIndexOrg[gInd]); }
	}
}
/*===================================================================================================================*/


/*===================================================================================================================*/
/** Process edge nodes (source or target) to create its connections that is not in the input json */
static inline void ProcessEdgeNodes(    uint e,
										const std::vector<uint>& nodeList,
										uint* edgeNodeArray,        // Where we write the nodes involved in this edge
										int& edgeCounter,           // Counter for edgeNodeArray
										uint* nodeEdgeList,
										int* nodeEdgePorts,
										uint* debugNodeCount,       // Tracks how many edges a node has seen so far
										int* nodeFirstEdge,
										int edgeLabelIndex)
{
    for (uint i = 0; i < nodeList.size(); i++)
    {
        uint nID = nodeList[i];

        /** Write node into compact array */
        edgeNodeArray[edgeCounter] = nID;
        edgeCounter++;

        /* Fill the node edge list and port */
        // We no longer need [gInd] because 'nodeEdgeList' points directly
        // to the specific array for this graph.
        nodeEdgeList[debugNodeCount[nID]] = e;
        nodeEdgePorts[debugNodeCount[nID]] = i;

        /** Store first port label */
        if (debugNodeCount[nID] == 0)
        {
            nodeFirstEdge[nID] = edgeLabelIndex;
        }

        debugNodeCount[nID]++;
    }
}
/*===================================================================================================================*/


/*===================================================================================================================*/
/** Debug Print Stats IO */
/*===================================================================================================================*/
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
    /** Build Histo for edge props */
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
/*===================================================================================================================*/

/*===================================================================================================================*/
/* Debug Print Stats Connections*/
/*===================================================================================================================*/
void printGraphStatsConn(const InputGraph* graphs, const CSR_Graph* gpuGraphs, DebugHistogram* debugHist)
{
	bool isIso = true;

	/*-------------------------------------------------------------------------------------------*/
	/** Construct Histogram */
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
/*===================================================================================================================*/



#endif /* GRAPHIO_H_ */
