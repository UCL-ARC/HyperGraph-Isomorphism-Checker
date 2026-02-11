
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
#include "json.hpp"

typedef unsigned int uint;
#include "GPU_Solver/CUDA_Functions.h" /* CUDA Solver */
#include "InputDataStructures.h" /* Compact Arrays the GPU will use*/
#include "GraphIO.h"

using namespace std;



/*===================================================================================================================*/
/** A] Counts edge nodes, computes CSR offsets, and assigns IO tags
 * Returns the computed sizes needed for allocation Compute compact array metadata (A0 + A1 + A2 phases)
 */
static inline void ComputeCompactArrayMetadata( int gInd,
												const InputGraph &graphIO,
												CSR_Graph        &CSRGraphDataOut,
												DebugHistogram   *debugHist)
{
	/*---------------------------------------------------------------------------------*/
	/** Edge Loop: A0] Count edge nodes to determine CSR array sizes */
	for (uint e = 0; e < CSRGraphDataOut.edgeData.numEdges; e++)
	{
		/* Each edge will increment its source nodes as a Next */
		for (uint i = 0; i < graphIO.edges.at(e).sourceNodes.size(); i++)
		{
			/* Primary: Number of source nodes  */
			CSRGraphDataOut.edgeData.edgeNodesSourceSize++;

			/* Secondary: NodeCompactList Inc the Node counter also */
			CSRGraphDataOut.nodeData.edgeStartNextsNum[graphIO.edges.at(e).sourceNodes.at(i)]++;
		}

		/* Each edge will increment its target nodes as a Prev */
		for (uint i = 0; i < graphIO.edges.at(e).targetNodes.size(); i++)
		{
			CSRGraphDataOut.edgeData.edgeNodesTargetSize++;

			CSRGraphDataOut.nodeData.edgeStartPrevsNum[graphIO.edges.at(e).targetNodes.at(i)]++;
		}
	}
	/*---------------------------------------------------------------------------------*/

	/*---------------------------------------------------------------------------------*/
	/** Node Loop: A1] Compute CSR offsets and total edges per node using prefix sum */
	for (uint n = 0; n < CSRGraphDataOut.nodeData.numNodes; n++)
	{
		CSRGraphDataOut.nodeData.labelIndex[n] = graphIO.nodeLabelIndex.at(n);

		/* Compact Array of node Prevs Start */
		CSRGraphDataOut.nodeData.edgeStartPrevsStart[n] = CSRGraphDataOut.nodeData.nodeEdgesPrevsSize;
		/* Compact Array of node Prevs Size */
		CSRGraphDataOut.nodeData.nodeEdgesPrevsSize += CSRGraphDataOut.nodeData.edgeStartPrevsNum[n];

		/* Next Array */
		CSRGraphDataOut.nodeData.edgeStartNextsStart[n] = CSRGraphDataOut.nodeData.nodeEdgesNextsSize;
		CSRGraphDataOut.nodeData.nodeEdgesNextsSize += CSRGraphDataOut.nodeData.edgeStartNextsNum[n];

		CSRGraphDataOut.nodeData.totalEdges[n] = CSRGraphDataOut.nodeData.edgeStartPrevsNum[n] + CSRGraphDataOut.nodeData.edgeStartNextsNum[n];

		/* Debug: track max edges for histogram binning */
		if ((CSRGraphDataOut.nodeData.edgeStartPrevsNum[n] + CSRGraphDataOut.nodeData.edgeStartNextsNum[n]) > debugHist[gInd].node.maxEdgesSize)
		{
			debugHist[gInd].node.maxEdgesSize = CSRGraphDataOut.nodeData.edgeStartPrevsNum[n] + CSRGraphDataOut.nodeData.edgeStartNextsNum[n];
		}
	}
	/*---------------------------------------------------------------------------------*/


	/*---------------------------------------------------------------------------------*/
	/** Global Input: Loop A2] Assign IO tags for global input/output nodes */
	for (uint n = 0; n < graphIO.globalInputs.size(); n++)
	{
		CSRGraphDataOut.nodeData.ioTags[graphIO.globalInputs.at(n)] = 1;
	}

	/*---------------------------------------------------------------------------------*/
	/** Global Output Loop */
	for (uint n = 0; n < graphIO.globalOutputs.size(); n++)
	{
		if (CSRGraphDataOut.nodeData.ioTags[graphIO.globalOutputs.at(n)] == 0)
		{
			CSRGraphDataOut.nodeData.ioTags[graphIO.globalOutputs.at(n)] = 2;
		}
		else if (CSRGraphDataOut.nodeData.ioTags[graphIO.globalOutputs.at(n)] == 1)
		{
			CSRGraphDataOut.nodeData.ioTags[graphIO.globalOutputs.at(n)] = 3;
		}
		else
		{
			CSRGraphDataOut.nodeData.ioTags[graphIO.globalOutputs.at(n)]++;
			cout << "ERROR NodeID: " << graphIO.globalOutputs.at(n)
			     << " LabelIndex: " << CSRGraphDataOut.nodeData.labelIndex[graphIO.globalOutputs.at(n)]
			     << " Label: " << graphIO.nodeLabelsDB.at(CSRGraphDataOut.nodeData.labelIndex[graphIO.globalOutputs.at(n)]) << endl;
		}
	}
	/*---------------------------------------------------------------------------------*/
}
/*===================================================================================================================*/



/*===================================================================================================================*/
/** B] Allocate and populate compact CSR arrays (B phase)
 * Takes computed sizes from ComputeCompactArrayMetadata as parameters
 * Populates all edge-node and node-edge relationship arrays
 */
static inline void AllocateAndPopulateCompactArrays( const InputGraph &IOgraph,
													 CSR_Graph        &CSRGraphDataOut,
													 DebugHistogram   *debugHisto)
{
    /** 1. Allocate node and edge CSR arrays using pre-computed sizes */
    CSRGraphDataOut.nodeData.edgePrevs     = new uint[CSRGraphDataOut.nodeData.nodeEdgesPrevsSize]();
    CSRGraphDataOut.nodeData.edgeNexts     = new uint[CSRGraphDataOut.nodeData.nodeEdgesNextsSize]();
    CSRGraphDataOut.nodeData.edgePrevsPort = new int [CSRGraphDataOut.nodeData.nodeEdgesPrevsSize]();
    CSRGraphDataOut.nodeData.edgeNextsPort = new int [CSRGraphDataOut.nodeData.nodeEdgesNextsSize]();

    /** Initialize Port arrays to -1 */
    std::fill_n(CSRGraphDataOut.nodeData.edgeNextsPort, CSRGraphDataOut.nodeData.nodeEdgesNextsSize, -1);
    std::fill_n(CSRGraphDataOut.nodeData.edgePrevsPort, CSRGraphDataOut.nodeData.nodeEdgesPrevsSize, -1);

    /* Allocate edge arrays */
    CSRGraphDataOut.edgeData.nodesSources = new uint[CSRGraphDataOut.edgeData.edgeNodesSourceSize]();
    CSRGraphDataOut.edgeData.nodesTargets = new uint[CSRGraphDataOut.edgeData.edgeNodesTargetSize]();

    /** Allocate Temp Debug Counters */
    uint *DEBUGnode_CountSources = new uint[CSRGraphDataOut.nodeData.numNodes]();
    uint *DEBUGnode_CountTargets = new uint[CSRGraphDataOut.nodeData.numNodes]();
    int DEBUGedgeCounterSources = 0;
    int DEBUGedgeCounterTargets = 0;
    debugHisto->edge.maxNodesSize = 0;

    /** 2. Loop over sorted edges and populate CSR arrays */
    for (uint e = 0; e < CSRGraphDataOut.edgeData.numEdges; e++)
    {
        CSRGraphDataOut.edgeData.labelIndex[e] = IOgraph.edges.at(e).labelIndex;

        /*-------------------------------------------------------------------------------------*/
        /* A. Process Source Nodes */
        CSRGraphDataOut.edgeData.nodeStartSourcesStart[e] = DEBUGedgeCounterSources;
        CSRGraphDataOut.edgeData.nodeStartSourcesNum  [e] = IOgraph.edges.at(e).sourceNodes.size();

        ProcessEdgeNodes(   e,
							IOgraph.edges.at(e).sourceNodes,
							CSRGraphDataOut.edgeData.nodesSources,
							DEBUGedgeCounterSources,
							CSRGraphDataOut.nodeData.edgeNexts,
							CSRGraphDataOut.nodeData.edgeNextsPort,
							DEBUGnode_CountTargets,
							CSRGraphDataOut.nodeData.nextsFirstEdge,
							IOgraph.edges.at(e).labelIndex             );
        /*-------------------------------------------------------------------------------------*/

        /*-------------------------------------------------------------------------------------*/
        /** B. Process Target Nodes */
        CSRGraphDataOut.edgeData.nodeStartTargetsStart[e] = DEBUGedgeCounterTargets;
        CSRGraphDataOut.edgeData.nodeStartTargetsNum[e] = IOgraph.edges.at(e).targetNodes.size();

        ProcessEdgeNodes(   e,
							IOgraph.edges.at(e).targetNodes,
							CSRGraphDataOut.edgeData.nodesTargets,
							DEBUGedgeCounterTargets,
							CSRGraphDataOut.nodeData.edgePrevs,
							CSRGraphDataOut.nodeData.edgePrevsPort,
							DEBUGnode_CountSources,
							CSRGraphDataOut.nodeData.prevsFirstEdge,
							IOgraph.edges.at(e).labelIndex );
        /*-------------------------------------------------------------------------------------*/

        /*-------------------------------------------------------------------------------------*/
        /** Metadata updates */
        CSRGraphDataOut.edgeData.totalNodes[e] = CSRGraphDataOut.edgeData.nodeStartSourcesNum[e] + CSRGraphDataOut.edgeData.nodeStartTargetsNum[e];
        if ((IOgraph.edges.at(e).sourceNodes.size() + IOgraph.edges.at(e).targetNodes.size()) > debugHisto->edge.maxNodesSize)
        {
            debugHisto->edge.maxNodesSize = IOgraph.edges.at(e).sourceNodes.size() + IOgraph.edges.at(e).targetNodes.size();
        }
        /*-------------------------------------------------------------------------------------*/

        //cout<<e<<" ELabel "<<" "<<CSRGraphDataOut.edgeData.labelIndex[e]<<endl;
    }

    // cout << " DEBUG: EdgeSourceCountCSR " << DEBUGedgeCounterSources << " EdgeTargetCountCSR " << DEBUGedgeCounterTargets << endl;

    /** 4. Error Checking */
    for (uint n = 0; n < CSRGraphDataOut.nodeData.numNodes; n++)
    {


        if (CSRGraphDataOut.nodeData.edgeStartPrevsNum[n] != DEBUGnode_CountSources[n])
        {
            cout << n << " Error SourceNodeEdgeMapping Got " << DEBUGnode_CountSources[n]
                 << " Expected " << CSRGraphDataOut.nodeData.edgeStartPrevsNum[n] << endl;
        }

        if (CSRGraphDataOut.nodeData.edgeStartNextsNum[n] != DEBUGnode_CountTargets[n])
        {
            cout << n << " Error TargetNodeEdgeMapping Got " << DEBUGnode_CountTargets[n]
                 << " Expected " << CSRGraphDataOut.nodeData.edgeStartNextsNum[n] << endl;
        }
    }

    /** Cleanup temp counter arrays */
    delete[] DEBUGnode_CountSources;
    delete[] DEBUGnode_CountTargets;
}
/*===================================================================================================================*/


/*===================================================================================================================*/
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
void TransferGraphToGPU(const CSR_Graph& data, int gpuIndex)
{
	GPU_InitArrays( data.graphIndex,
	               data.nodeData.numNodes,            data.nodeData.labelIndex,
	               data.nodeData.prevsFirstEdge,      data.nodeData.nextsFirstEdge,
	               data.nodeData.nodeEdgesPrevsSize,  data.nodeData.nodeEdgesNextsSize,
	               data.nodeData.edgePrevs,           data.nodeData.edgeNexts,
	               data.nodeData.edgePrevsPort,       data.nodeData.edgeNextsPort,
	               data.nodeData.edgeStartPrevsStart, data.nodeData.edgeStartPrevsNum,
	               data.nodeData.edgeStartNextsStart, data.nodeData.edgeStartNextsNum,
	               data.nodeData.totalEdges,          data.nodeData.ioTags,
				   /* Edges */
	               data.edgeData.numEdges,              data.edgeData.labelIndex,
	               data.edgeData.edgeNodesSourceSize,   data.edgeData.edgeNodesTargetSize,
	               data.edgeData.nodesSources,          data.edgeData.nodesTargets,
	               data.edgeData.nodeStartSourcesStart, data.edgeData.nodeStartSourcesNum,
	               data.edgeData.nodeStartTargetsStart, data.edgeData.nodeStartTargetsNum,
	               data.edgeData.totalNodes,            gpuIndex );
}
/*===================================================================================================================*/



/*===================================================================================================================*/
/** Allocates the Compact (CSR) arrays that will be used for compute * @see DeallocateGPUGraphData*/
void AllocateCSRGraphData(int gInd, const InputGraph& graph, CSR_Graph& data)
{
	data.graphIndex = gInd;

	/* Initialize node data metadata */
	data.nodeData.numNodes           = graph.nodeLabelIndex.size();
	data.nodeData.numNodeLabelsDB    = graph.nodeLabelsDB.size();
	data.nodeData.nodeEdgesPrevsSize = 0;
	data.nodeData.nodeEdgesNextsSize = 0;

	/* Allocate node arrays */
	data.nodeData.labelIndex          = new uint[data.nodeData.numNodes]();
	data.nodeData.ioTags              = new uint[data.nodeData.numNodes]();
	data.nodeData.edgeStartPrevsNum   = new uint[data.nodeData.numNodes]();
	data.nodeData.edgeStartNextsNum   = new uint[data.nodeData.numNodes]();
	data.nodeData.totalEdges          = new uint[data.nodeData.numNodes]();
	data.nodeData.edgeStartPrevsStart = new uint[data.nodeData.numNodes]();
	data.nodeData.edgeStartNextsStart = new uint[data.nodeData.numNodes]();
	data.nodeData.prevsFirstEdge      = new int[data.nodeData.numNodes]();
	data.nodeData.nextsFirstEdge      = new int[data.nodeData.numNodes]();

	/* Initialize first edge markers */
	std::fill_n(data.nodeData.prevsFirstEdge, data.nodeData.numNodes, -1);
	std::fill_n(data.nodeData.nextsFirstEdge, data.nodeData.numNodes, -1);

	/* Initialize edge data metadata */
	data.edgeData.numEdges = graph.edges.size();
	data.edgeData.numEdgeLabelsDB = graph.edgeLabelsDB.size();
	data.edgeData.edgeNodesSourceSize = 0;
	data.edgeData.edgeNodesTargetSize = 0;

	/* Allocate edge arrays */
	data.edgeData.labelIndex            = new uint[data.edgeData.numEdges]();
	data.edgeData.nodeStartSourcesNum   = new uint[data.edgeData.numEdges]();
	data.edgeData.nodeStartTargetsNum   = new uint[data.edgeData.numEdges]();
	data.edgeData.totalNodes            = new uint[data.edgeData.numEdges]();
	data.edgeData.nodeStartSourcesStart = new uint[data.edgeData.numEdges]();
	data.edgeData.nodeStartTargetsStart = new uint[data.edgeData.numEdges]();
}
/*===================================================================================================================*/



/*===================================================================================================================*/
/** Deallocate CSR Data @see InitializeGPUGraphData */
void DeallocateCSRGraphData(CSR_Graph& data)
{
	delete[] data.nodeData.labelIndex;
	delete[] data.nodeData.ioTags;
	delete[] data.nodeData.edgeStartPrevsNum;
	delete[] data.nodeData.edgeStartNextsNum;
	delete[] data.nodeData.totalEdges;
	delete[] data.nodeData.edgeStartPrevsStart;
	delete[] data.nodeData.edgeStartNextsStart;
	delete[] data.nodeData.prevsFirstEdge;
	delete[] data.nodeData.nextsFirstEdge;
	delete[] data.nodeData.edgePrevs;
	delete[] data.nodeData.edgeNexts;
	delete[] data.nodeData.edgePrevsPort;
	delete[] data.nodeData.edgeNextsPort;

	delete[] data.edgeData.labelIndex;
	delete[] data.edgeData.nodeStartSourcesNum;
	delete[] data.edgeData.nodeStartTargetsNum;
	delete[] data.edgeData.totalNodes;
	delete[] data.edgeData.nodeStartSourcesStart;
	delete[] data.edgeData.nodeStartTargetsStart;
	delete[] data.edgeData.nodesSources;
	delete[] data.edgeData.nodesTargets;
}
/*===================================================================================================================*/


/*--------------------------------------------------------------------------------------------------------------------*/
/** Processed Arrays to be stored on the heap */
InputGraph     m_IO_graphs[2] = {};  /** Raw IO */
CSR_Graph      m_CSRGraphs[2] = {};  /** Processed Graphs */

uint          *m_DebugEdge_LabelDBIndexOrg [2] = {};  /** Since we sort edges this allows us to map back to the IO Order   */
DebugHistogram m_DebugHist                 [2] = {};  /** Stats counter for the graph to check input was processed correct */
/*--------------------------------------------------------------------------------------------------------------------*/



int main(int argc, char* argv[])
{
	int targetGPU = 0; /** User provided GPU to run on */


	uint MaxNodesPerEdge = 0;  /** We assume max of 8 so that we can use faster sort methods */
	auto Timer_Start = std::chrono::high_resolution_clock::now();

	LoadGraphs    (argc, argv, m_IO_graphs, MaxNodesPerEdge); /** Open and process the json file or pass the arrays from another binary (RUST) */
	//SortGraphEdges(m_IO_graphs, m_DebugEdge_LabelDBIndexOrg); /** Sort Edges based on counts */



 	/*===========================================================================================*/
 				          /** Start Create compact arrays and pass to the GPU */
 	/*===========================================================================================*/
	auto Timer_CSR = std::chrono::high_resolution_clock::now();
	for (int gInd = 0;gInd<2;gInd++ )
	 {
    	 cout<<" Create Compact Arrays " <<gInd<<endl;

		/** Initialize GPU graph data structure */
		AllocateCSRGraphData(gInd, m_IO_graphs[gInd],  m_CSRGraphs[gInd]);
		m_DebugHist[gInd].node.maxEdgesSize = 0; /* Debug host variable for histo on CPU */

		/** Compute metadata: edge counts, CSR offsets, and IO tags (A0 + A1 + A2) */
		ComputeCompactArrayMetadata(gInd, m_IO_graphs[gInd],  m_CSRGraphs[gInd], m_DebugHist);

		cout<<" EdgeSourceCSR: "<< m_CSRGraphs[gInd].edgeData.edgeNodesSourceSize<<"  EdgeTargetCSR: "<< m_CSRGraphs[gInd].edgeData.edgeNodesTargetSize
			<<" NodePrevsCSR:  "<< m_CSRGraphs[gInd].nodeData.nodeEdgesPrevsSize <<"  NodeNextsCSR:  "<< m_CSRGraphs[gInd].nodeData.nodeEdgesNextsSize<<endl;

		/** Allocate and populate compact arrays using computed metadata (B phase) */
		cout<<" Create Edge Compact Arrays " <<gInd<<endl;
		AllocateAndPopulateCompactArrays(m_IO_graphs[gInd],  m_CSRGraphs[gInd],  m_DebugHist);
    }

	auto Time_CSREnd = std::chrono::high_resolution_clock::now();
	auto compact_time = std::chrono::duration_cast<std::chrono::milliseconds>(Time_CSREnd - Timer_CSR).count();
	std::cout << "Compact array creation time: " << compact_time << " ms" << std::endl;
	/*===========================================================================================*/
 				          /* End Create compact arrays and pass to the GPU */
 	/*===========================================================================================*/


    printGraphStatsConn(m_IO_graphs,  m_CSRGraphs, m_DebugHist); /* Debug Stats Printing */


	auto start_gpu = std::chrono::high_resolution_clock::now();
    /** Transfer GPU data to device */
    for (int gInd = 0; gInd<2; gInd++ )
	{
    	/* Transfer graph to GPU using organized struct */
    	TransferGraphToGPU( m_CSRGraphs[gInd], targetGPU);
    }

	auto gpu_init_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_gpu).count();
	std::cout << "GPU initialization time: " << gpu_init_time << " ms" << std::endl;

	/*===========================================================================================*/
	/** GPU Calculation */
	/*===========================================================================================*/
	auto start_gpu_compute = std::chrono::high_resolution_clock::now();

	GPU_CheckHypergraphIsomorphism();

	//RunDeterminismStressTest(100);

	auto end_gpu_compute = std::chrono::high_resolution_clock::now();
	auto gpu_compute_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu_compute - start_gpu_compute).count();
	std::cout << "GPU computation time: " << gpu_compute_time << " ms" << std::endl;


    for (int gInd = 0;gInd<2;gInd++ )
	{
      GPU_FreeInitArrays(gInd,0);

	  DeallocateCSRGraphData( m_CSRGraphs[gInd]);
	  if (m_DebugEdge_LabelDBIndexOrg[gInd] != nullptr)
	  {
	  	delete[] m_DebugEdge_LabelDBIndexOrg[gInd];
	  }
	}
	/*===========================================================================================*/
	/** End GPU Calculation */
	/*===========================================================================================*/

	auto Timer_End = std::chrono::high_resolution_clock::now();
	auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(Timer_End - Timer_Start).count();
	std::cout << "\n==================================================" << std::endl;
	std::cout << "Total execution time: " << total_time << " ms" << std::endl;
	std::cout << "==================================================" << std::endl;

	return 0;
}
