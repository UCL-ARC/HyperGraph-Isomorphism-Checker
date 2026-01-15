/*
 * CUDA_Functions.cu
 *
 *  Created on: Oct 23, 2025
 *  Author: Nicolin Govender UCL-ARC
 */




#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdio.h>
#include <iomanip>    // <-- For std::setw and std::left
#include <sstream>    // <-- For std::stringstream (which we also used)
#include <vector>     // <-- For std::vector (which we also used)
#include <map>

#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/equal.h>
#include <thrust/device_vector.h>


/*-------------------------------------------------------------------------------------------------------------------*/
/** 16 Bytes Aligned:  Node signatures Label, IO, numNexts, numPrevs, OPT NextEdgeLab, PrevEdgeLab */
typedef thrust::tuple<uint, uint, uint, uint> NodeKeyTuple;
NodeKeyTuple MAX_TUPLEH = thrust::make_tuple(UINT_MAX, 0, 0, 0); /** Host Side {default value} */
/*-------------------------------------------------------------------------------------------------------------------*/

#include "CUDA_Kernels.cuh"

/*-------------------------------------------------------------------------------------------------------------------*/
/** A] Node Struct compact list that we will copy to GPU */
/*-------------------------------------------------------------------------------------------------------------------*/
uint  m_numNodes               [2] = {};                             /** Total Number of Nodes in a graph  */
uint  m_nodeEdgesPrevsSize     [2] = {}, nodeEdgesNextsSize[2] = {}; /** Size of the compact arrays for node Edges Prevs and Nexts */

/** Per Node */
uint *dram_Node_labelDBIndex   [2];    /** index of the label that identifies the node type  */
uint *dram_Node_IOTag          [2];    /** is the node a global input,output or both   */

/** Node Edge Connections */
uint *dram_Node_edgePrevsStart [2];  /** start index in node_EdgePrevs array  */
uint *dram_Node_edgePrevsNum   [2];  /** count in node_EdgePrevs array  */
uint *dram_Node_edgeNextsStart [2];  /** start index in node_EdgeNexts array  */
uint *dram_Node_edgeNextsNum   [2];  /** count in node_EdgeNexts array  */

int  *dram_Node_PrevsFirstEdge [2];  /** Sig -1 if empty else 1st port edge label TODO: MM */
int  *dram_Node_NextsFirstEdge [2];  /** Sig -1 if empty else 1st port edge label TODO: MM */

/** Each node will write its input and output edges into these compact arrays */
uint *dram_Node_edgePrevs      [2];  /** "Compact Created From Edge Sources " */
uint *dram_Node_edgeNexts      [2];  /** "Compact Created From Edge Targets " */

int  *dram_Node_edgePrevsPorts [2];  /** "Compact Created From Edge Sources store the port on the edge it connects to " */
int  *dram_Node_edgeNextsPorts [2];  /** "Compact Created From Edge Targets store the port on the edge it connects to " */
/*-------------------------------------------------------------------------------------------------------------------*/


/*-------------------------------------------------------------------------------------------------------------------*/
/** B] Edge struct compact list */
/*-------------------------------------------------------------------------------------------------------------------*/
uint m_numEdges[2]={}; /** Total Number of Edges in a graph  */

/** Size of the compact arrays */
uint m_edgeNodesSourceSize[2]={}, m_edgeNodesTargetSize[2]={};

/** Per Edge */
uint *dram_Edge_labelDBIndex[2];    /** index of the label that identifies the node  */

/** Edge Node Connections */
uint *dram_Edge_nodeSourcesStart [2]; /** start in edge_NodesSources array  */
uint *dram_Edge_nodeSourcesNum   [2]; /** count in edge_NodesSources array  */
uint *dram_Edge_nodeTargetsStart [2]; /** start in edge_NodesTargets array  */
uint *dram_Edge_nodeTargetsNum   [2]; /** count in edge_NodesTargets array  */

uint *dram_Edge_nodeTot          [2];  /** Sig: Node total for faster look up  */

/** Each edge will write its source and target nodes into these compact arrays */
uint *dram_Edge_nodeSources      [2];
uint *dram_Edge_nodeTargets      [2];
/*-------------------------------------------------------------------------------------------------------------------*/


NodeKeyTuple *dram_NodeColorHashes    [2];/** Node Signature (Color) that we create on the GPU */


/*-------------------------------------------------------------------------------------------------------------------*/
/** ISOMorph: WL-1 Test */
bool  m_isWL1Alloc [2] = {0,0};
bool  m_isWL2Alloc [2] = {0,0};

uint       m_WL_BinCount          [2] = {0,0}; /* Number of Bins for each graph */
uint64_t  *dram_WL_BinsColorKeys  [2]; /** Pointers to the final WL Histogram Keys (Hashes) in GPU Memory */
uint      *dram_WL_BinsNumCount   [2]; /** Pointers to the final WL Histogram Counts in GPU Memory */
/*-------------------------------------------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------------------------------------------*/
/** ISOMorph: WL-2 Test */
/* We store pairs of nodes so NNode*NNodes matrix */
/** Stores the size (number of bins) of the WL-2 histogram for each graph */
uint64_t *dram_WL2_MatrixColors  [2] = {NULL, NULL};
/*-------------------------------------------------------------------------------------------------------------------*/


/*-----------------------------------------------------------------------------*/
/** CUDA Specific Flags */
/*-----------------------------------------------------------------------------*/
struct launchParms
{
    dim3 dimGrid;
    dim3 dimBlock;
};

/** 64 = Max GPUS for future use */
launchParms   ThreadsAllNodes  [64]; /** One Thread Per Node */
launchParms   ThreadsAllEdges  [64]; /** One Thread Per Edge */

launchParms   ThreadsNodePairs [64]; /** Node Pairs for adj matrix */
/*-----------------------------------------------------------------------------*/

double m_MaxGPUMemoryMB = 6000.0;   /** Query GPU Memory */

/*-----------------------------------------------------------------------------*/
//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
        cudaError_t e=cudaGetLastError();                                 \
        if(e!=cudaSuccess) {                                              \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }
/*-----------------------------------------------------------------------------*/
/* End CUDA Specific Flags */
/*-----------------------------------------------------------------------------*/

/*===================================================================================================================*/
/** Init GPU Arrays */
/*===================================================================================================================*/
void InitGPUArrays( uint gIndex,

		            uint numNodesH,
					uint *NodeLabelIndexH,
					int  *NodePrevsFirstEdge, int *NodeNextsFirstEdge,
					uint nodePrevsArraySizeH,  uint nodeNextsArraySizeH,

					uint *NodePrevsH, uint *NodeNextsH,
					int *NodePrevsPortsH, int *NodeNextsPortsH,

					uint *NodeStartPrevsStartH,
					uint *NodeStartPrevsCountH,
					uint *NodeStartNextsStartH,
					uint *NodeStartNextsCountH,
					uint *NodeEdgeCountH,
					uint *NodeIOTagsH,

					uint numEdgesH,
					uint *EdgeLabelIndexH,

					uint edgeSourcesArraySizeH, uint edgeTargetsArraySizeH,
					uint *EdgesSourcesH, uint *EdgeTargetsH,

					uint *EdgeStartSourcesStartH,
					uint *EdgeStartSourcesCountH,
					uint *EdgeStartTargetsStartH,
					uint *EdgeStartTargetsCountH,
					uint *EdgeNodeCountH,

					uint gpu               )

{

	/*-----------------------------------------------------------------------------*/
	double bytesNodes = (sizeof(uint)*numNodesH) + (sizeof(uint)*numNodesH) + (sizeof(uint2)*numNodesH) + (sizeof(uint2)*numNodesH) +
			            (sizeof(uint)*nodePrevsArraySizeH) + (sizeof(uint)*nodeNextsArraySizeH);
	double bytesEdges = (sizeof(uint)*numEdgesH) + (sizeof(uint2)*numEdgesH) + (sizeof(uint2)*numEdgesH) + (sizeof(uint)*edgeSourcesArraySizeH) +
			            (sizeof(uint)*edgeTargetsArraySizeH);
	std::cout<<" Memory Required (mb): "<< (bytesNodes+bytesEdges)*1E-6<<" Nodes (mb): "<< (bytesNodes)*1E-6 <<" Edges (mb): "<< (bytesEdges)*1E-6<<std::endl;

    /** Set Array sizes that will be passed to GPU Per Graph Nodes */
	m_numNodes           [gIndex] = numNodesH;
	m_nodeEdgesPrevsSize [gIndex] = nodePrevsArraySizeH;
	nodeEdgesNextsSize [gIndex] = nodeNextsArraySizeH;

    /** Set Array sizes that will be passed to GPU Per Graph Edges */
    m_numEdges            [gIndex] = numEdgesH;
    m_edgeNodesSourceSize [gIndex] = edgeSourcesArraySizeH;
    m_edgeNodesTargetSize [gIndex] = edgeTargetsArraySizeH;


	printf("%d Nodes %d CSRNexts %d CSRPrevs %d  Edges %d CSRSources %d CSRTargets %d \n",
			gIndex,m_numNodes [gIndex], nodeEdgesNextsSize [gIndex] , m_nodeEdgesPrevsSize [gIndex],
			m_numEdges [gIndex], m_edgeNodesSourceSize [gIndex] , m_edgeNodesTargetSize [gIndex]);
	/*-----------------------------------------------------------------------------*/


	/*-----------------------------------------------------------------------------*/
	/** A] Set Target GPU */
	cudaDeviceSynchronize();
	cudaSetDevice(gpu);

	/** B] Allocate memory GPU for nodes  */
	cudaMalloc( (void**) &dram_Node_labelDBIndex   [gIndex],  sizeof(uint)*numNodesH);
	cudaMalloc( (void**) &dram_Node_IOTag          [gIndex],  sizeof(uint)*numNodesH);
	cudaMalloc( (void**) &dram_Node_edgePrevsStart [gIndex],  sizeof(uint)*numNodesH);
	cudaMalloc( (void**) &dram_Node_edgePrevsNum   [gIndex],  sizeof(uint)*numNodesH);
	cudaMalloc( (void**) &dram_Node_edgeNextsStart [gIndex],  sizeof(uint)*numNodesH);
	cudaMalloc( (void**) &dram_Node_edgeNextsNum   [gIndex],  sizeof(uint)*numNodesH);

	cudaMalloc( (void**) &dram_Node_PrevsFirstEdge [gIndex],  sizeof(int)*numNodesH); /** Used for signature */
	cudaMalloc( (void**) &dram_Node_NextsFirstEdge [gIndex],  sizeof(int)*numNodesH); /** Used for signature */

	/** Compact Arrays */
	cudaMalloc( (void**) &dram_Node_edgePrevs      [gIndex],  sizeof(uint)*nodePrevsArraySizeH);
	cudaMalloc( (void**) &dram_Node_edgeNexts      [gIndex],  sizeof(uint)*nodeNextsArraySizeH);

	cudaMalloc( (void**) &dram_Node_edgePrevsPorts [gIndex],  sizeof(int)*nodePrevsArraySizeH);
	cudaMalloc( (void**) &dram_Node_edgeNextsPorts [gIndex],  sizeof(int)*nodeNextsArraySizeH);

	cudaDeviceSynchronize();
	cudaCheckError();

	/** C] Copy memory GPU for nodes  */
    cudaMemcpyAsync( dram_Node_labelDBIndex   [gIndex], NodeLabelIndexH,      sizeof(uint)*numNodesH,           cudaMemcpyHostToDevice );
    cudaMemcpyAsync( dram_Node_IOTag          [gIndex], NodeIOTagsH,          sizeof(uint)*numNodesH,           cudaMemcpyHostToDevice );

    cudaMemcpyAsync( dram_Node_PrevsFirstEdge [gIndex], NodePrevsFirstEdge,   sizeof(int)*numNodesH,           cudaMemcpyHostToDevice );
    cudaMemcpyAsync( dram_Node_NextsFirstEdge [gIndex], NodeNextsFirstEdge,   sizeof(int)*numNodesH,           cudaMemcpyHostToDevice );

    cudaMemcpyAsync( dram_Node_edgePrevsStart [gIndex], NodeStartPrevsStartH, sizeof(uint)*numNodesH,          cudaMemcpyHostToDevice );
    cudaMemcpyAsync( dram_Node_edgePrevsNum   [gIndex], NodeStartPrevsCountH, sizeof(uint)*numNodesH,          cudaMemcpyHostToDevice );

    cudaMemcpyAsync( dram_Node_edgeNextsStart [gIndex], NodeStartNextsStartH, sizeof(uint)*numNodesH,          cudaMemcpyHostToDevice );
    cudaMemcpyAsync( dram_Node_edgeNextsNum   [gIndex], NodeStartNextsCountH, sizeof(uint)*numNodesH,          cudaMemcpyHostToDevice );


    cudaMemcpyAsync( dram_Node_edgePrevs      [gIndex], NodePrevsH,           sizeof(uint)*nodePrevsArraySizeH, cudaMemcpyHostToDevice );
    cudaMemcpyAsync( dram_Node_edgeNexts      [gIndex], NodeNextsH,           sizeof(uint)*nodeNextsArraySizeH, cudaMemcpyHostToDevice );

    cudaMemcpyAsync( dram_Node_edgePrevsPorts [gIndex], NodePrevsPortsH,      sizeof(int)*nodePrevsArraySizeH, cudaMemcpyHostToDevice );
    cudaMemcpyAsync( dram_Node_edgeNextsPorts [gIndex], NodeNextsPortsH,      sizeof(int)*nodeNextsArraySizeH, cudaMemcpyHostToDevice );


	cudaDeviceSynchronize();
	cudaCheckError();


	/** B1] Allocate memory for edges  */
	cudaMalloc( (void**) &dram_Edge_labelDBIndex     [gIndex],  sizeof(uint)*numEdgesH);
	cudaMalloc( (void**) &dram_Edge_nodeTot          [gIndex],  sizeof(uint)*numEdgesH);
	cudaMalloc( (void**) &dram_Edge_nodeSourcesStart [gIndex],  sizeof(uint)*numEdgesH);
	cudaMalloc( (void**) &dram_Edge_nodeSourcesNum   [gIndex],  sizeof(uint)*numEdgesH);
	cudaMalloc( (void**) &dram_Edge_nodeTargetsStart [gIndex],  sizeof(uint)*numEdgesH);
	cudaMalloc( (void**) &dram_Edge_nodeTargetsNum   [gIndex],  sizeof(uint)*numEdgesH);

	cudaMalloc( (void**) &dram_Edge_nodeSources      [gIndex],  sizeof(uint)*edgeSourcesArraySizeH);
	cudaMalloc( (void**) &dram_Edge_nodeTargets      [gIndex],  sizeof(uint)*edgeTargetsArraySizeH);
	cudaDeviceSynchronize();
	cudaCheckError();

	/** C1] Copy memory for edges  */
    cudaMemcpyAsync( dram_Edge_labelDBIndex     [gIndex], EdgeLabelIndexH,        sizeof(uint)*numEdgesH,   cudaMemcpyHostToDevice );
    cudaMemcpyAsync( dram_Edge_nodeTot          [gIndex], EdgeNodeCountH,         sizeof(uint)*numEdgesH,   cudaMemcpyHostToDevice );

    cudaMemcpyAsync( dram_Edge_nodeSourcesStart [gIndex], EdgeStartSourcesStartH, sizeof(uint)*numEdgesH,   cudaMemcpyHostToDevice );
    cudaMemcpyAsync( dram_Edge_nodeSourcesNum   [gIndex], EdgeStartSourcesCountH, sizeof(uint)*numEdgesH,   cudaMemcpyHostToDevice );

    cudaMemcpyAsync( dram_Edge_nodeTargetsStart [gIndex], EdgeStartTargetsStartH, sizeof(uint)*numEdgesH,   cudaMemcpyHostToDevice );
    cudaMemcpyAsync( dram_Edge_nodeTargetsNum   [gIndex], EdgeStartTargetsCountH, sizeof(uint)*numEdgesH,   cudaMemcpyHostToDevice );

    cudaMemcpyAsync( dram_Edge_nodeSources      [gIndex], EdgesSourcesH, sizeof(uint)*edgeSourcesArraySizeH,   cudaMemcpyHostToDevice );
    cudaMemcpyAsync( dram_Edge_nodeTargets      [gIndex], EdgeTargetsH,  sizeof(uint)*edgeTargetsArraySizeH,   cudaMemcpyHostToDevice );


    /** D] Wait till GPU is done and check for errors */
	cudaDeviceSynchronize();
	cudaCheckError();
	/*-----------------------------------------------------------------------------*/


	/*-----------------------------------------------------------------------------*/
	/** D] CUDA Threads Launch Config */
	/** Hardware Specific Query " Like splitting a problem over MPI and then OMP, think of a SM "Streaming MultiProcessor as a node" */
	int numThreadsBlock = 32; /** Threads per block equal to wrap size to reduce divergence cost */
	int num_sm          = 1;

	/** CUDA Query to get Hardware Details also returns free memory etc */
	cudaDeviceProp DevProp;
	cudaGetDeviceProperties(&DevProp,gpu);
	m_MaxGPUMemoryMB = DevProp.totalGlobalMem*1E-6;
	std::cout<<"INFO-D: Using Device: "<<gpu<<" with "<<DevProp.multiProcessorCount<<" SM "<<DevProp.name
			<<" SM "<< m_MaxGPUMemoryMB <<std::endl;
	num_sm = DevProp.multiProcessorCount;

	int numItemsSM  = (int)ceilf(numNodesH / (float)num_sm);   /** Threads split over SMs */
	int numBlocksSM = (int)ceilf(numItemsSM / (float)numThreadsBlock); /** Each SM splits its threads into blocks */

	/** For very small problems ( we should not have such) reduce block size */
	if (numItemsSM < 32)
	{
		numThreadsBlock = numItemsSM; /** Single block is sufficient */
	}

	/** Threads that we want to launch */
	if (gIndex==0 || (m_numNodes [gIndex] > m_numNodes[0]) )
	{
		ThreadsAllNodes[gpu].dimBlock = make_uint3(numThreadsBlock, 1, 1);
		ThreadsAllNodes[gpu].dimGrid  = make_uint3(numBlocksSM * num_sm, 1, 1);
	}

	/** Threads that we want to launch for edges */
    numItemsSM  = (int)ceilf(numEdgesH / (float)num_sm);
	numBlocksSM = (int)ceilf(numItemsSM / (float)numThreadsBlock);
	/** default block size is too big */
	if (numItemsSM < 32)
	{
		numThreadsBlock = numItemsSM; /** Single block is sufficient */
	}

	if (gIndex==0 || (m_numEdges [gIndex] > m_numEdges[0]) )
	{
	 ThreadsAllEdges[gpu].dimBlock = make_uint3(numThreadsBlock, 1, 1);
	 ThreadsAllEdges[gpu].dimGrid  = make_uint3(numBlocksSM * num_sm, 1, 1);
	}
	/*-----------------------------------------------------------------------------*/

	std::cout<<" NodeNexts Print \n";
	printItem<<<1,1>>> (dram_Node_edgeNexts[gIndex], dram_Node_edgeNextsStart[gIndex],dram_Node_edgeNextsNum[gIndex], dram_Node_labelDBIndex[gIndex], m_numNodes[gIndex], 1);
	cudaDeviceSynchronize();

	std::cout<<" NodePrevs Print \n";
	printItem<<<1,1>>> (dram_Node_edgePrevs[gIndex], dram_Node_edgePrevsStart[gIndex],dram_Node_edgePrevsNum[gIndex], dram_Node_labelDBIndex[gIndex], m_numNodes[gIndex], 1);

	std::cout<<" EdgeSources Print \n";
	printItem<<<1,1>>> (dram_Edge_nodeSources[gIndex], dram_Edge_nodeSourcesStart[gIndex], dram_Edge_nodeSourcesNum[gIndex], dram_Edge_labelDBIndex[gIndex], m_numEdges[gIndex], 1);
	cudaDeviceSynchronize();

	std::cout<<" EdgeTargets Print \n";
	printItem<<<1,1>>> (dram_Edge_nodeTargets[gIndex], dram_Edge_nodeTargetsStart[gIndex],dram_Edge_nodeTargetsNum[gIndex], dram_Edge_labelDBIndex[gIndex], m_numEdges[gIndex], 1);


	numItemsSM  = (int)ceilf(numNodesH*numNodesH / (float)num_sm);   /** Threads split over SMs */
	numBlocksSM = (int)ceilf(numNodesH*numNodesH / (float)numThreadsBlock); /** Each SM splits its threads into blocks */
	/** For very small problems ( we should not have such) reduce block size */
	if (numItemsSM < 32)
	{
		numThreadsBlock = numItemsSM; /** Single block is sufficient */
	}
	/** Threads that we want to launch */
	if (gIndex==0 || (m_numNodes [gIndex] > m_numNodes[0]) )
	{
		ThreadsNodePairs[gpu].dimBlock = make_uint3(numThreadsBlock, 1, 1);
		ThreadsNodePairs[gpu].dimGrid  = make_uint3(numBlocksSM * num_sm, 1, 1);
	}


	cudaDeviceSynchronize();
	cudaCheckError();
}
/*===================================================================================================================*/


/*===================================================================================================================*/
/** Free GPU Arrays */
/*===================================================================================================================*/
void GPU_FreeArrays (uint gIndex, uint gpu)
{
	cudaDeviceSynchronize();
	cudaSetDevice(gpu);
	std::cout<<"GPU Free Graph "<<gIndex<<std::endl;

	cudaFree (dram_Node_labelDBIndex[gIndex]);
	cudaFree (dram_Node_IOTag[gIndex]);
	cudaFree (dram_Node_PrevsFirstEdge[gIndex]);
	cudaFree (dram_Node_NextsFirstEdge[gIndex]);

	cudaFree (dram_Node_edgePrevsStart[gIndex]);
	cudaFree (dram_Node_edgeNextsStart[gIndex]);

	cudaFree (dram_Node_edgePrevsNum[gIndex]);
	cudaFree (dram_Node_edgeNextsNum[gIndex]);
	cudaFree (dram_Node_edgePrevs[gIndex]);
	cudaFree (dram_Node_edgeNexts[gIndex]);

	cudaFree (dram_Node_edgePrevsPorts[gIndex]);
	cudaFree (dram_Node_edgeNextsPorts[gIndex]);

	/** edge arrays */
	cudaFree (dram_Edge_labelDBIndex[gIndex]);
	cudaFree (dram_Edge_nodeTot[gIndex]);
	cudaFree (dram_Edge_nodeSourcesStart[gIndex]);
	cudaFree (dram_Edge_nodeTargetsStart[gIndex]);

	cudaFree (dram_Edge_nodeSourcesNum[gIndex]);
	cudaFree (dram_Edge_nodeTargetsNum[gIndex]);
	cudaFree (dram_Edge_nodeSources[gIndex]);
	cudaFree (dram_Edge_nodeTargets[gIndex]);


	/* Colors Created on GPU */
	if (m_isWL1Alloc[gIndex])
	{
	 cudaFree(dram_NodeColorHashes[gIndex]);
	}

	if (m_isWL2Alloc[gIndex])
	{
	  cudaFree(dram_WL2_MatrixColors[gIndex]);
	}

	cudaDeviceSynchronize();
	cudaError_t errormsg=cudaGetLastError();
	if(errormsg>0)
	{
	 std::cout<<"GPU Free Memory "<<cudaGetErrorString(errormsg)<<std::endl;
	 exit(1);
	}

}

void GPU_FreeWLBins ()
{
	for (int gIndex=0;gIndex<2;gIndex++)
	{
		std::cout<<"GPU Free WL Bins "<<gIndex<<std::endl;
		cudaFree (dram_WL_BinsColorKeys[gIndex]);
		cudaFree (dram_WL_BinsNumCount[gIndex]);
	}
}
/*===================================================================================================================*/


/*===================================================================================================================*/
/** A] Does the binning on the GPU for the feature counts of each edge and node
 *  GlobalOutput: dram_NodeColorHashes which is the "hash/color" is the output for other functions */
/*===================================================================================================================*/
bool GPU_CompareSignatureCountsBetweenGraphs()
{
	bool isDebug = true;

	bool isPossibleIsomorphic = true; /* Assume True */

    /** We use the Thrust lib, you can also write a custom kernel for this binning if rust does not have a boost type lib equv */
	/*===============================================================================================================*/
	                                      /** Start Edge Histogram */
	/*===============================================================================================================*/
	/** Edge Signature key source,target,tot,label */
	typedef thrust::tuple<uint, uint, uint, uint> EdgeKeyTuple;


	std::cout<<"------------------------------------------------------------------------"<<std::endl;
	std::cout<<"GPU Bulk Test 0 Signature counts "<<std::endl;
	std::cout<<"------------------------------------------------------------------------"<<std::endl;

	/** Output Histo */
	EdgeKeyTuple *d_HistoEdgeKeys      [2]; /** Sorted keys array */
	uint         *d_HistoEdgeKeyCounts [2]; /** Bin counts for sorted keys array */
	int          numUniqueKeyBinsGraph [2]; /** Store Tot bins per graph */

	/** Create a histogram of edge signatures  */
	for (int gIndex=0;gIndex<2;gIndex++)
	{
		cudaDeviceSynchronize();
		cudaError_t errormsg=cudaGetLastError();
		if(errormsg>0)
		{
		 std::cout<<"GPU GraphBins Memory "<<cudaGetErrorString(errormsg)<<std::endl;
		 exit(1);
		}
		std::cout << "NumEdges" << m_numEdges[0] << std::endl; /** Debug print */


		cudaMalloc((void**)&d_HistoEdgeKeys      [gIndex], m_numEdges[gIndex]*sizeof(EdgeKeyTuple));
		cudaMalloc((void**)&d_HistoEdgeKeyCounts [gIndex], m_numEdges[gIndex]*sizeof(uint));


        /** Pointer Wrapping */
		thrust::device_ptr<EdgeKeyTuple> d_ptr_output_ColorHashkeys  (d_HistoEdgeKeys   [gIndex]);
		thrust::device_ptr<uint>         d_ptr_output_counts(d_HistoEdgeKeyCounts [gIndex]);

		std::cout<<"Start Zip IT "<<gIndex<<std::endl;
        /** Zip Iterator to create color hash */
		auto keys_begin   = thrust::make_zip_iterator(thrust::make_tuple( thrust::device_ptr<uint> ( dram_Edge_nodeSourcesNum [gIndex]),
				                                                          thrust::device_ptr<uint> ( dram_Edge_nodeTargetsNum [gIndex]),
																		  thrust::device_ptr<uint> ( dram_Edge_nodeTot        [gIndex]),
																		  thrust::device_ptr<uint> ( dram_Edge_labelDBIndex   [gIndex])	)) ;
		auto keys_end     = keys_begin + m_numEdges[gIndex];
		auto values_begin = thrust::make_constant_iterator(1u); /** Set to 1 */

	     /** Bin Counting */
		std::cout<<"Start Binning "<<gIndex<<std::endl;
		auto new_end = thrust::reduce_by_key( keys_begin,
											  keys_end,
											  values_begin,

											  d_ptr_output_ColorHashkeys,
											  d_ptr_output_counts  	);
		std::cout<<"End Binning "<<gIndex<<std::endl;

	    /** Get Number of bins for each signature type */
		numUniqueKeyBinsGraph[gIndex] = new_end.first - d_ptr_output_ColorHashkeys;
		std::cout<<"Bins "<<numUniqueKeyBinsGraph[gIndex] <<std::endl;

	}/** End Bin Creation for both graphs */
	/*===============================================================================================================*/
	                                      /** End Edge Histogram */
	/*===============================================================================================================*/

	/*----------------------------------------------------------------------------------*/
    /** Compare Edge Histo counts between graphs  */
	/*----------------------------------------------------------------------------------*/
	std::cout << "GPU Binning Complete" << std::endl;
	std::cout << "Graph 0 unique bins: " << numUniqueKeyBinsGraph[0] << std::endl;
	std::cout << "Graph 1 unique bins: " << numUniqueKeyBinsGraph[1] << std::endl;

	if (numUniqueKeyBinsGraph[0] != numUniqueKeyBinsGraph[1])
	{
	    std::cout << "Result: NOT Isomorphic (Edge Bin counts differ)" << std::endl;
	    isPossibleIsomorphic = false;
	    if (!isDebug)
	    {
	    	return false;
	    }
	}
	else
	{
	    /** The bin counts are the same. Now we must compare keys and bin counts on the GPU */
	    std::cout << "Bin counts match. Comparing arrays on GPU" << std::endl;
	    int num_bins = numUniqueKeyBinsGraph[0];

	    /** Wrap for Thrust */
	    thrust::device_ptr<EdgeKeyTuple> keys_A(d_HistoEdgeKeys[0]);
	    thrust::device_ptr<EdgeKeyTuple> keys_B(d_HistoEdgeKeys[1]);

	    thrust::device_ptr<uint>     counts_A(d_HistoEdgeKeyCounts[0]);
	    thrust::device_ptr<uint>     counts_B(d_HistoEdgeKeyCounts[1]);

	    /** GPU Check 1: Compare the Key arrays returns a bool */
	    bool areKeysEqual = thrust::equal( keys_A, keys_A + num_bins,
										   keys_B  );

	    if (!areKeysEqual)
	    {
	        std::cout << "Result: NOT Isomorphic (Edge Bin keys do not match)" << std::endl;
	        isPossibleIsomorphic = false;
	        if (!isDebug)
			{
				return false;
			}
	    }
	    else/** Keys are the same so check counts */
	    {
	        /** GPU Check 2: Compare the Count arrays */
	        std::cout << "Bin keys match. Comparing Edge counts on GPU " << std::endl;

	        bool are_counts_equal = thrust::equal( counts_A, counts_A + num_bins,
	        		                               counts_B );

	        if (!are_counts_equal)
	        {
	            std::cout << "Result:  NOT Isomorphic (Edge Bin counts do not match) " << std::endl;
	            isPossibleIsomorphic = false;
	            if (!isDebug)
				{
					return false;
				}
	        }
	        else
	        {
	            std::cout << "Result:  Possible Isomorphic (Edge Keys and counts match)" << std::endl;
	            isPossibleIsomorphic = true;
	            if (!isDebug)
				{
					return true;
				}
	        }
	    }
	}/** End Loop over graphs */
	std::cout << " Free GPU Memory HistoEdgeCounts " << std::endl;
	for (int gIndex=0;gIndex<2;gIndex++)
	{
	  cudaFree(d_HistoEdgeKeys[gIndex]);
	  cudaFree(d_HistoEdgeKeyCounts[gIndex]);
	}
	/*----------------------------------------------------------------------------------*/


	/** Only If Edges find possible Isomorphic do we count bins  */
	if (isPossibleIsomorphic)
	{
	/*===============================================================================================================*/
	                                      /** Start Node Histogram */
	/*===============================================================================================================*/
    /** Now we compare Node signatures Label, IO, numNexts, numPrevs */
	    NodeKeyTuple *d_HistoNodeKeys       [2];
	    uint         *d_HistoNodeKeyCounts  [2];
	    int           numUniqueKeyBinsNodes [2];


	    for (int gIndex = 0; gIndex < 2; gIndex++)
	    {
	        std::cout << "\nProcessing Node Histogram for gIndex " << gIndex << std::endl;

	        uint *d_temp_labels;
	        uint *d_temp_IOTag;
	        uint *d_temp_numNexts;
	        uint *d_temp_numPrevs;
	        int  *d_temp_PrevsEdge;
	        int  *d_temp_NextsEdge;

	        /** Allocate output arrays and store pointers*/
	        cudaMalloc((void**)&d_HistoNodeKeys      [gIndex], m_numNodes[gIndex] * sizeof(NodeKeyTuple));
	        cudaMalloc((void**)&d_HistoNodeKeyCounts [gIndex], m_numNodes[gIndex] * sizeof(uint));

	        cudaMalloc((void**) &d_temp_labels,    m_numNodes[gIndex]* sizeof(uint));
	        cudaMalloc((void**) &d_temp_IOTag,     m_numNodes[gIndex]* sizeof(uint));
	        cudaMalloc((void**) &d_temp_numNexts,  m_numNodes[gIndex]* sizeof(uint));
	        cudaMalloc((void**) &d_temp_numPrevs,  m_numNodes[gIndex]* sizeof(uint));

	        cudaMalloc((void**) &d_temp_NextsEdge, m_numNodes[gIndex]* sizeof(int));
	        cudaMalloc((void**) &d_temp_PrevsEdge, m_numNodes[gIndex]* sizeof(int));

	        /** fast device-to-device copies */
	        cudaMemcpy(d_temp_labels,   dram_Node_labelDBIndex[gIndex],    m_numNodes[gIndex]* sizeof(uint), cudaMemcpyDeviceToDevice);
	        cudaMemcpy(d_temp_IOTag,    dram_Node_IOTag[gIndex],           m_numNodes[gIndex]* sizeof(uint), cudaMemcpyDeviceToDevice);
	        cudaMemcpy(d_temp_numNexts, dram_Node_edgeNextsNum[gIndex],    m_numNodes[gIndex]* sizeof(uint), cudaMemcpyDeviceToDevice);
	        cudaMemcpy(d_temp_numPrevs, dram_Node_edgePrevsNum[gIndex],    m_numNodes[gIndex]* sizeof(uint), cudaMemcpyDeviceToDevice);

	        cudaMemcpy(d_temp_NextsEdge, dram_Node_NextsFirstEdge[gIndex], m_numNodes[gIndex]* sizeof(int), cudaMemcpyDeviceToDevice);
	        cudaMemcpy(d_temp_PrevsEdge, dram_Node_PrevsFirstEdge[gIndex], m_numNodes[gIndex]* sizeof(int), cudaMemcpyDeviceToDevice);


	        std::cout <<"Sorting 4-part Signature" << std::endl;
	        /** Create the 4-part zip iterator */
	        auto node_keys_begin = thrust::make_zip_iterator(thrust::make_tuple( thrust:: device_ptr<uint>(d_temp_labels),
																				 thrust::device_ptr<uint> (d_temp_IOTag),
																				 thrust::device_ptr<uint> (d_temp_numNexts),
																				 thrust::device_ptr<uint> (d_temp_numPrevs)));
	        																	 //thrust::device_ptr<int>  (d_temp_NextsEdge),
																				 //thrust::device_ptr<int>  (d_temp_PrevsEdge)));
	        auto node_keys_end = node_keys_begin + m_numNodes[gIndex];

	        /** Store The node signatures */
	        cudaMalloc((void**)&dram_NodeColorHashes  [gIndex], m_numNodes[gIndex]*sizeof(NodeKeyTuple));
	        m_isWL1Alloc[gIndex] = true;

	        thrust::device_ptr<NodeKeyTuple> d_ptr_output_lookup(dram_NodeColorHashes[gIndex]);
	        thrust::copy(node_keys_begin, node_keys_end, d_ptr_output_lookup);
	        cudaDeviceSynchronize();


	        /** Sort the temporary buffers */
	        thrust::sort(node_keys_begin, node_keys_end);

	        cudaDeviceSynchronize();
	        std::cout << "  Pass 2: Complete " << std::endl;
	        std::cout << "  Pass 3: Binning Signatures into Histogram" << std::endl;



	        /** Now we can start to bin the nodes */
	        thrust::device_ptr<NodeKeyTuple> d_ptr_node_hist_keys(d_HistoNodeKeys[gIndex]);
	        thrust::device_ptr<uint>         d_ptr_node_hist_counts(d_HistoNodeKeyCounts[gIndex]);

	        /** Create  values to sum (just a stream of '1's) */
	        auto values_begin = thrust::make_constant_iterator(1u);

	        auto hist_end = thrust::reduce_by_key(  node_keys_begin,
													node_keys_end,
													values_begin,
													d_ptr_node_hist_keys,
													d_ptr_node_hist_counts );

	        cudaDeviceSynchronize();

	        /**  Store the number of unique bins found */
	        numUniqueKeyBinsNodes[gIndex] = hist_end.first - d_ptr_node_hist_keys;
	        std::cout << "  Pass 3: Complete: Found " << numUniqueKeyBinsNodes[gIndex] << " unique node bins " << std::endl;

	        cudaFree(d_temp_labels);
	        cudaFree(d_temp_IOTag);
	        cudaFree(d_temp_numNexts);
	        cudaFree(d_temp_numPrevs);
	        cudaFree(d_temp_PrevsEdge);
	        cudaFree(d_temp_NextsEdge);

	    }/** End Loop over graphs */
		/*===============================================================================================================*/
		                                      /** End Node Histogram */
		/*===============================================================================================================*/


	    /*----------------------------------------------------------------------------------*/
	    /** Compare Node Histo counts between graphs  */
	    /*----------------------------------------------------------------------------------*/
	    std::cout << "\n Node Histogram Comparison " << std::endl;
		if (numUniqueKeyBinsNodes[0] != numUniqueKeyBinsNodes[1])
		{
			std::cout << "Result: NOT Isomorphic (Node bin counts differ) " << std::endl;
			isPossibleIsomorphic = false;
            if (!isDebug)
			{
				return false;
			}
		}
		else
		{
			int num_bins = numUniqueKeyBinsNodes[0];
			std::cout << "Node bin counts match (" << num_bins << ") Comparing arrays " << std::endl;

			thrust::device_ptr<NodeKeyTuple> keys_A(d_HistoNodeKeys[0]);
			thrust::device_ptr<NodeKeyTuple> keys_B(d_HistoNodeKeys[1]);
			bool keys_match = thrust::equal(keys_A, keys_A + num_bins, keys_B);

			if (!keys_match)
			{
				std::cout << "Result: NOT Isomorphic (Node keys differ)" << std::endl;
				isPossibleIsomorphic = false;
	            if (!isDebug)
				{
					return false;
				}
			}
			else
			{
				thrust::device_ptr<uint> counts_A(d_HistoNodeKeyCounts[0]);
				thrust::device_ptr<uint> counts_B(d_HistoNodeKeyCounts[1]);
				bool counts_match = thrust::equal(counts_A, counts_A + num_bins, counts_B);

				if (counts_match)
				{
					std::cout << "Result: Node histograms are Isomorphic" << std::endl;
					isPossibleIsomorphic = true;
		            if (!isDebug)
					{
						return true;
					}
				}
				else
				{
					std::cout << "Result: NOT Isomorphic (Node counts differ)" << std::endl;
					isPossibleIsomorphic = false;
		            if (!isDebug)
					{
						return false;
					}
				}
			}
		}
		/*----------------------------------------------------------------------------------*/


        /*----------------------------------------------------------------------------------*/
        /* DEBUG Print */
        /* We will store the host-side copies here */
//        std::vector<NodeKeyTuple> h_keys[2];
//        std::vector<uint> h_counts[2];
//
//        for (int gIndex = 0; gIndex < 2; gIndex++)
//        {
//            std::cout << "\n Printing Results for Graph " << gIndex << "  " << std::endl;
//
//            int num_bins = numUniqueKeyBinsNodes[gIndex];
//            std::cout << "  Found " << num_bins << " unique node bins." << std::endl;
//
//            if (num_bins == 0)
//            {
//                continue; // Nothing to print
//            }
//
//
//            h_keys[gIndex].resize(num_bins);
//            h_counts[gIndex].resize(num_bins);
//
//            /** Copy histogram keys from Device (GPU) to Host (CPU) */
//            cudaMemcpy( h_keys[gIndex].data(),   d_HistoNodeKeys[gIndex],      num_bins * sizeof(NodeKeyTuple), cudaMemcpyDeviceToHost);
//            cudaMemcpy( h_counts[gIndex].data(), d_HistoNodeKeyCounts[gIndex], num_bins * sizeof(uint),         cudaMemcpyDeviceToHost );
//
//            std::cout << "  " << std::setw(30) << std::left << "Signature (Lab, IO, Nexts, Prevs)" << " | Count" << std::endl;
//            std::cout << "  ----------------------------------- | -----" << std::endl;
//
//            for (int i = 0; i < num_bins; ++i)
//            {
//                NodeKeyTuple key   = h_keys[gIndex][i];
//                uint         count = h_counts[gIndex][i];
//
//                std::stringstream ss;
//                ss << "( " << thrust::get<0>(key) << ", "
//                           << thrust::get<1>(key) << ", "
//                           << thrust::get<2>(key) << ", "
//                           << thrust::get<3>(key) << ")";
//
//                std::cout << "  " << std::setw(30) << std::left << ss.str() << " | " << count << std::endl;
//            }
//        }
        /*----------------------------------------------------------------------------------*/

	    std::cout << "Cleaning up final node histogram memory " << std::endl;
		for (int gIndex=0;gIndex<2;gIndex++)
		{
		  cudaFree(d_HistoNodeKeys[gIndex]);
		  cudaFree(d_HistoNodeKeyCounts[gIndex]);
		}
	}

	std::cout<<"------------------------------------------------------------------------"<<std::endl;
	std::cout<<""<<std::endl;
	std::cout<<"------------------------------------------------------------------------"<<std::endl;

   return isPossibleIsomorphic;
}
/*===================================================================================================================*/



/*===================================================================================================================*/
/** B] Uses the Node Keys of an edge to create a hash assumes **max 8 nodes per edge target or source
 *  Assumes SignatureCounts was called that created this as  GlobalInput: dram_NodeColorHashes[gIndex] */
/*===================================================================================================================*/
bool GPU_CompareEdgesSignaturesBetweenGraphs(int MaxNodesPerEdge)
{
  bool isPossibleIsomorphic= true;

  /** Hashes are stored here*/
  uint64_t  *d_temp_EdgeHashSources[2];
  uint64_t  *d_temp_EdgeHashTargets[2];

	std::cout<<"------------------------------------------------------------------------"<<std::endl;
	std::cout<<"GPU Structure Test 1 Edge Colors "<<std::endl;
	std::cout<<"------------------------------------------------------------------------"<<std::endl;

  /*=========================================================================*/
  /** A]  Create Hashes for the edges */
  /*=========================================================================*/
  for (int gIndex = 0; gIndex < 2; gIndex++)
  {
    /** Allocate arrays */
	cudaMalloc((void**)&d_temp_EdgeHashSources      [gIndex], m_numEdges[gIndex] * sizeof(uint64_t));
	cudaMalloc((void**)&d_temp_EdgeHashTargets      [gIndex], m_numEdges[gIndex] * sizeof(uint64_t));
	cudaDeviceSynchronize();
	cudaCheckError();

	printf("Graph %d Threads %d \n", gIndex, ThreadsAllEdges[0].dimGrid.x*ThreadsAllEdges[0].dimBlock.x);


	if(MaxNodesPerEdge<=8)
	{
		Kernel_EdgeHashesSortedBoundFlat <<<ThreadsAllEdges[0].dimGrid ,ThreadsAllEdges[0].dimBlock>>>(
													 m_numEdges[gIndex],

													dram_Edge_nodeSources[gIndex],
													dram_Edge_nodeSourcesStart[gIndex],
													dram_Edge_nodeSourcesNum[gIndex],

													dram_Edge_nodeTargets[gIndex],
													dram_Edge_nodeTargetsStart[gIndex],
													dram_Edge_nodeTargetsNum[gIndex],


													dram_NodeColorHashes[gIndex],

													d_temp_EdgeHashSources[gIndex],
													d_temp_EdgeHashTargets[gIndex],
													MAX_TUPLEH );
	}
	else if(MaxNodesPerEdge<=64)
	{
			Kernel_EdgeHashesSortedBubble64L <<<ThreadsAllEdges[0].dimGrid ,ThreadsAllEdges[0].dimBlock>>>(
														 m_numEdges[gIndex],

														dram_Edge_nodeSources[gIndex],
														dram_Edge_nodeSourcesStart[gIndex],
														dram_Edge_nodeSourcesNum[gIndex],

														dram_Edge_nodeTargets[gIndex],
														dram_Edge_nodeTargetsStart[gIndex],
														dram_Edge_nodeTargetsNum[gIndex],


														dram_NodeColorHashes[gIndex],

														d_temp_EdgeHashSources[gIndex],
														d_temp_EdgeHashTargets[gIndex],
														MAX_TUPLEH );
	}
	else
	{
			Kernel_EdgeHashes_MixHashNoSort<<<ThreadsAllEdges[0].dimGrid ,ThreadsAllEdges[0].dimBlock>>>(
														 m_numEdges[gIndex],

														dram_Edge_nodeSources[gIndex],
														dram_Edge_nodeSourcesStart[gIndex],
														dram_Edge_nodeSourcesNum[gIndex],

														dram_Edge_nodeTargets[gIndex],
														dram_Edge_nodeTargetsStart[gIndex],
														dram_Edge_nodeTargetsNum[gIndex],


														dram_NodeColorHashes[gIndex],

														d_temp_EdgeHashSources[gIndex],
														d_temp_EdgeHashTargets[gIndex] );
	}


	 cudaDeviceSynchronize();
	 cudaCheckError();


	/*DD---------------------------------------------------------------------------------------------------------------------DD*/
	/*  Debug Printing */
	/*DD---------------------------------------------------------------------------------------------------------------------DD*/
	    /*DD---------------------------------------------------------------------------------------------------------------------DD*/
//			std::cout << "\n Edge Neighborhood Hashes for Graph " << gIndex << std::endl;
//			// 2. Debug Print Hash List on Host
//			int num_edges = numEdges[gIndex];
//			 uint64_t  *h_source_hashes;
//			 h_source_hashes = new uint64_t [num_edges];
//			cudaMemcpy(h_source_hashes, d_temp_EdgeHashSources[gIndex], num_edges * sizeof(uint64_t), cudaMemcpyDeviceToHost);
//
//			 uint64_t  *h_target_hashes;
//			 h_target_hashes = new uint64_t [num_edges];
//			cudaMemcpy( h_target_hashes, d_temp_EdgeHashTargets[gIndex], num_edges * sizeof(uint64_t), cudaMemcpyDeviceToHost);
//
//			cudaDeviceSynchronize();
//			cudaCheckError();
//
//		    for (int i = 0; i < num_edges; i++)
//		    {
//		       printf(" %d HashSource 0x%016llx HashTarget 0x%016llx  \n",i, h_source_hashes[i], h_target_hashes[i]);
//		        PrintEdge<<<1,1 >>>(  i, dram_Edge_labelDBIndex[gIndex],
//
//												dram_NodeKeys[gIndex],
//												dram_Edge_nodeSources[gIndex],
//		                                        dram_Edge_nodeSourcesStart[gIndex],
//												dram_Edge_nodeSourcesNum[gIndex],
//
//												dram_Edge_nodeTargets[gIndex],
//												dram_Edge_nodeTargetsStart[gIndex],
//												dram_Edge_nodeTargetsNum[gIndex],MAX_TUPLEH    );
//		  	    cudaDeviceSynchronize();
//		  	    cudaCheckError();
//		    }
//		    delete []h_source_hashes;
//		    delete []h_target_hashes;
		    /*DD---------------------------------------------------------------------------------------------------------------------DD*/


  } /*  End Loop over graphs */

  /*=========================================================================*/
  /** End A]  Create Hashes for the edges */
  /*=========================================================================*/

  /*=========================================================================*/
   /** B]  Sort the hashes for each graph: linked edge index is optional  */
  /*=========================================================================*/
	int num_edges = m_numEdges[0];

    /*DD----------------------------------------------------------------------*/
    /** Optional for debugging to keep track of edge index*/
    uint      *d_original_indices_srcA;
    uint      *d_original_indices_tgtA;
    uint      *d_original_indices_srcB;
    uint      *d_original_indices_tgtB;
	cudaMalloc((void**)&d_original_indices_srcA, m_numEdges[0] * sizeof(uint));
	cudaMalloc((void**)&d_original_indices_tgtA, m_numEdges[0] * sizeof(uint));

	thrust::device_ptr<uint> d_ptr_indices_src_A(d_original_indices_srcA);
	thrust::device_ptr<uint> d_ptr_indices_tgt_A(d_original_indices_tgtA);

	thrust::sequence(d_ptr_indices_src_A, d_ptr_indices_src_A + m_numEdges[0]);
	thrust::sequence(d_ptr_indices_tgt_A, d_ptr_indices_tgt_A + m_numEdges[0]);


	cudaMalloc((void**)&d_original_indices_srcB, m_numEdges[1] * sizeof(uint));
	cudaMalloc((void**)&d_original_indices_tgtB, m_numEdges[1] * sizeof(uint));

	thrust::device_ptr<uint> d_ptr_indices_src_B(d_original_indices_srcB);
	thrust::device_ptr<uint> d_ptr_indices_tgt_B(d_original_indices_tgtB);

	thrust::sequence(d_ptr_indices_src_B, d_ptr_indices_src_B + m_numEdges[1]);
	thrust::sequence(d_ptr_indices_tgt_B, d_ptr_indices_tgt_B + m_numEdges[1]);

    cudaDeviceSynchronize();
    cudaCheckError();
    /*DD----------------------------------------------------------------------*/


    /*-----------------------------------------------------------------------*/
	std::cout<< "Sorting source and target hash arrays " << std::endl;

	/** Wrap Pointers */
    thrust::device_ptr<uint64_t> d_ptr_source_hash_A(d_temp_EdgeHashSources[0]);
    thrust::device_ptr<uint64_t> d_ptr_source_hash_B(d_temp_EdgeHashSources[1]);
    thrust::device_ptr<uint64_t> d_ptr_target_hash_A(d_temp_EdgeHashTargets[0]);
    thrust::device_ptr<uint64_t> d_ptr_target_hash_B(d_temp_EdgeHashTargets[1]);

	thrust::sort_by_key(d_ptr_source_hash_A, d_ptr_source_hash_A + num_edges, d_ptr_indices_src_A);
	thrust::sort_by_key(d_ptr_source_hash_B, d_ptr_source_hash_B + num_edges, d_ptr_indices_src_B);

	/** Debug Sort the target hash arrays with the indices */
	thrust::sort_by_key(d_ptr_target_hash_A, d_ptr_target_hash_A + num_edges, d_ptr_indices_tgt_A);
	thrust::sort_by_key(d_ptr_target_hash_B, d_ptr_target_hash_B + num_edges, d_ptr_indices_tgt_B);
	/*-----------------------------------------------------------------------*/
	cudaDeviceSynchronize();
	std::cout <<"Sorting complete" << std::endl;
	/*=========================================================================*/
	/** End B]  Sort the hashes for each graph: linked edge index is optional  */
	/*=========================================================================*/


	/*=========================================================================*/
	/** C] Compare Hashes if they don't match it is not isomorphic */
	/*=========================================================================*/
	bool source_hashes_match = false;
	bool target_hashes_match = false;


	source_hashes_match = thrust::equal( d_ptr_source_hash_A, d_ptr_source_hash_A + num_edges,d_ptr_source_hash_B  );

	if (!source_hashes_match)
	{
		  std::cout << "Result: NOT Isomorphic (Edge Source neighborhood hashes differ)" << std::endl;
	}
	else
	{
	  std::cout << "Source neighborhood hashes match " << std::endl;

	  target_hashes_match = thrust::equal( d_ptr_target_hash_A, d_ptr_target_hash_A + num_edges, d_ptr_target_hash_B);

	  if (!target_hashes_match)
	  {
		  std::cout << "Result: NOT Isomorphic (Edge Target neighborhood hashes differ) " << std::endl;
	  }
	}
	/*=========================================================================*/


	/*DD---------------------------------------------------------------------------------------------------------------------DD*/
	/** Debug print non matching */
	/*DD---------------------------------------------------------------------------------------------------------------------DD*/

    if (!source_hashes_match || !target_hashes_match)
	{
	  std::cout << "Result: NOT Isomorphic (Source neighborhood hashes differ)" << std::endl;
	  std::cout << "\n  --- Printing Source Mismatches ---" << std::endl;
	  std::cout << std::hex << std::setfill('0');

	  std::vector<uint64_t> h_source_A(num_edges), h_source_B(num_edges);
	  std::vector<uint> h_indices_A(num_edges), h_indices_B(num_edges);

	  cudaMemcpy(h_source_A.data(),  d_ptr_source_hash_A.get(), num_edges * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	  cudaMemcpy(h_source_B.data(),  d_ptr_source_hash_B.get(), num_edges * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	  cudaMemcpy(h_indices_A.data(), d_ptr_indices_src_A.get(), num_edges * sizeof(uint),     cudaMemcpyDeviceToHost);
	  cudaMemcpy(h_indices_B.data(), d_ptr_indices_src_B.get(), num_edges * sizeof(uint),     cudaMemcpyDeviceToHost);

	  int i = 0, j = 0;
	  while (i < num_edges || j < num_edges)
	  {
		  if (i < num_edges && j < num_edges)
		  {
			  if (h_source_A[i] < h_source_B[j])
			  {
				  std::cout << "    Only in G0: 0x" << std::setw(16) << h_source_A[i]
							<< " (from original edge " << std::dec << h_indices_A[i] << std::hex << ")" << std::endl;
				  i++;
			  }
			  else if (h_source_B[j] < h_source_A[i])
			  {
				  std::cout << "    Only in G1: 0x" << std::setw(16) << h_source_B[j]
							<< " (from original edge " << std::dec << h_indices_B[j] << std::hex << ")" << std::endl;
				  j++;
			  }
			  else
			  {
				  i++; j++; // Match
			  }
		  }
		  else if (i < num_edges)
		  {
			  std::cout << "    Only in G0: 0x" << std::setw(16) << h_source_A[i]
						<< " (from original edge " << std::dec << h_indices_A[i] << std::hex << ")" << std::endl;
			  i++;
		  }
		  else if (j < num_edges)
		  {
			  std::cout << "    Only in G1: 0x" << std::setw(16) << h_source_B[j]
						<< " (from original edge " << std::dec << h_indices_B[j] << std::hex << ")" << std::endl;
			  j++;
		  }
	  }
	  std::cout << std::dec << std::setfill(' '); // Reset cout
    }
    /** B] Compare Targets */
	{
	  std::cout << "Source neighborhood hashes match " << std::endl;

	  bool target_hashes_match = thrust::equal( d_ptr_target_hash_A, d_ptr_target_hash_A + num_edges, d_ptr_target_hash_B);

	  if (!target_hashes_match)
	  {
		  std::cout << "Result: NOT Isomorphic (Target neighborhood hashes differ) " << std::endl;
		  std::cout << "Printing Target Mismatches" << std::endl;

		  std::cout << std::hex << std::setfill('0'); // Set hex format
		  std::vector<uint64_t> h_target_A(num_edges), h_target_B(num_edges);
		  std::vector<uint> h_indices_A(num_edges), h_indices_B(num_edges);

		  cudaMemcpy(h_target_A.data(), d_ptr_target_hash_A.get(), num_edges * sizeof(uint64_t), cudaMemcpyDeviceToHost);
		  cudaMemcpy(h_target_B.data(), d_ptr_target_hash_B.get(), num_edges * sizeof(uint64_t), cudaMemcpyDeviceToHost);
		  cudaMemcpy(h_indices_A.data(), d_ptr_indices_tgt_A.get(), num_edges * sizeof(uint), cudaMemcpyDeviceToHost);
		  cudaMemcpy(h_indices_B.data(), d_ptr_indices_tgt_B.get(), num_edges * sizeof(uint), cudaMemcpyDeviceToHost);

		  // "Diff" loop
		  int i = 0, j = 0;
		  while (i < num_edges || j < num_edges)
		  {
			  if (i < num_edges && j < num_edges)
			  {
				  if (h_target_A[i] < h_target_B[j])
				  {
					  std::cout << "    Only in G0: 0x" << std::setw(16) << h_target_A[i]
								<< " (from original edge " << std::dec << h_indices_A[i] << std::hex << ")" << std::endl;
					  i++;
				  } else if (h_target_B[j] < h_target_A[i]) {
					  std::cout << "    Only in G1: 0x" << std::setw(16) << h_target_B[j]
								<< " (from original edge " << std::dec << h_indices_B[j] << std::hex << ")" << std::endl;
					  j++;
				  } else
				  {
					  i++; j++; // Match
				  }
			  }
			  else if (i < num_edges)
			  { // G1 ran out, G0 has extras
				  std::cout << "    Only in G0: 0x" << std::setw(16) << h_target_A[i]
							<< " (from original edge " << std::dec << h_indices_A[i] << std::hex << ")" << std::endl;
				  i++;
			  }
			  else if (j < num_edges)
			  { // G0 ran out, G1 has extras
				  std::cout << "    Only in G1: 0x" << std::setw(16) << h_target_B[j]
							<< " (from original edge " << std::dec << h_indices_B[j] << std::hex << ")" << std::endl;
				  j++;
			  }
		  }
		  std::cout << std::dec << std::setfill(' '); // Reset cout
	  }
	  else
	  {
		  std::cout << "Target neighborhood hashes match" << std::endl;
	  }
	}



  for (int gIndex = 0; gIndex < 2; gIndex++)
  {
	cudaFree(d_temp_EdgeHashSources[gIndex]);
	cudaFree(d_temp_EdgeHashTargets[gIndex]);
  }

	std::cout<<"------------------------------------------------------------------------"<<std::endl;
	std::cout<<""<<std::endl;
	std::cout<<"------------------------------------------------------------------------"<<std::endl;

  return isPossibleIsomorphic;
}
/*===================================================================================================================*/


/*---------------------------------------------------------------------------------------------------------------------*/
/** WL Stability Check to see if graph cannot be divided into anymore colors */
/*---------------------------------------------------------------------------------------------------------------------*/
bool CheckStability( size_t    num_elements,       /** Total size*/
                     uint64_t *d_new_colors,       // Input

                     uint64_t *d_temp_sort_buffer, // Scratch: Temp buffer to sort into

                     /** Histogram Outputs (The Graph Signature) */
                     uint64_t *d_histo_ColorHashKeys,       // Unique Hash Values found
                     uint     *d_histo_counts,     // Output: How many times each Hash appeared
                     int      &h_num_bins_curr,    // Output: Total number of unique groups found this round

                     int       h_num_bins_prev     /** The bin count from the PREVIOUS loop (-1 if first run) */ )
{

    /** Copy Data
      * We must copy to a temp buffer because Thrust::sort is in-place
      * We cannot sort 'd_new_colors' directly because the pixel position (index)
      * matters for the NEXT iteration's geometry */
    cudaMemcpy(d_temp_sort_buffer, d_new_colors, num_elements * sizeof(uint64_t), cudaMemcpyDeviceToDevice);


    thrust::device_ptr<uint64_t> d_ptr_sort_buffer         (d_temp_sort_buffer);
    thrust::device_ptr<uint64_t> d_ptr_histo_ColorHashKeys (d_histo_ColorHashKeys);
    thrust::device_ptr<uint>     d_ptr_histo_counts        (d_histo_counts);


    /** Sort (Grouping)
     * Brings identical structural hashes together.
     * Example: [A, C, A, B, A] -> [A, A, A, B, C] */
    try
    {
        thrust::sort(thrust::device, d_ptr_sort_buffer, d_ptr_sort_buffer + num_elements);
    }
    catch (thrust::system_error &e)
    {
        std::cerr << "Thrust Sort Error: " << e.what() << std::endl;
        return false;
    }


    /** Reduce By Key (Binning)
      * Compresses the sorted array into unique keys and their counts
      * Example: [A, A, A, B, C] -> Keys: [A, B, C], Counts: [3, 1, 1] */
    auto new_end = thrust::reduce_by_key(thrust::device,
                                         d_ptr_sort_buffer,                  // Input Keys (Sorted)
                                         d_ptr_sort_buffer + num_elements,   // Input End
                                         thrust::make_constant_iterator(1u), // Input Values (1 for each occurrence)
                                         d_ptr_histo_ColorHashKeys,                   // Output Unique Keys
                                         d_ptr_histo_counts);                // Output Counts

    // The iterator 'new_end.first' points to the end of the unique keys
    h_num_bins_curr = new_end.first - d_ptr_histo_ColorHashKeys;

     // Optional: Print status for debugging
     std::cout << "  Refinement Step -> Bins [Prev: " << h_num_bins_prev<< ", Curr: " << h_num_bins_curr << "]" << std::endl;


    // Case A: First Run (Prev is -1) or Empty Graph
    if (h_num_bins_prev == -1 || h_num_bins_curr == 0)
    {
        return false; // Not stable yet, keep going
    }

    /** Case B: Bin Count Increased
     *  This means we found new sub-structures. The graph is "splitting" further
     * We must continue refining */

    if (h_num_bins_curr > h_num_bins_prev)
    {
        return false;
    }

    /** Case C: Bin Count Unchanged (STABLE)
     * Due to Monotonic Refinement, if the count didn't grow, the partition cannot change anymore */
    if (h_num_bins_curr == h_num_bins_prev)
    {
        return true;
    }

    /** Fallback (Should typically not be reached in valid WL execution) */
    return false;
}
/*---------------------------------------------------------------------------------------------------------------------*/


/*===================================================================================================================*/
/** C] Possible Isomorphism Check
 *  Compares the stable WL-1/2 histograms between two graphs.
 *  Returns TRUE if graphs are structurally identical ( Possible Isomorphic)
 */
/*===================================================================================================================*/
bool GPU_AreGraphsPossibleIsomorphic()
{
	std::cout<<"------------------------------------------------------------------------"<<std::endl;
	std::cout<<"GPU Structure Test Between Graphs "<<std::endl;
	std::cout<<"------------------------------------------------------------------------"<<std::endl;

    /* Fail Fast: Bin Count Mismatch */
    if (m_WL_BinCount[0] != m_WL_BinCount[1])
    {
        std::cout << "Mismatch: Graphs have different complexity (Bins: "
                  << m_WL_BinCount[0] << " vs " <<m_WL_BinCount[1] << ")" << std::endl;
        GPU_FreeWLBins ();
        return false;
    }

   /* Wrap th*/
    thrust::device_ptr<uint64_t> th_keys_A(dram_WL_BinsColorKeys[0]);
    thrust::device_ptr<uint>     th_counts_A(dram_WL_BinsNumCount[0]);

    thrust::device_ptr<uint64_t> th_keys_B(dram_WL_BinsColorKeys[1]);
    thrust::device_ptr<uint>     th_counts_B(dram_WL_BinsNumCount[1]);


    /** Compare Keys (Structure Types): Do both graphs contain the same connections? */
    bool keys_match = thrust::equal(thrust::device,
                                    th_keys_A,
                                    th_keys_A + m_WL_BinCount[0],
                                    th_keys_B);

    if (!keys_match)
    {
        std::cout << "Mismatch: Graphs contain different structural shapes (Keys differ)!" << std::endl;
        GPU_FreeWLBins ();
        return false;
    }

    /** Compare Counts */
    bool counts_match = thrust::equal(thrust::device,
                                      th_counts_A,
                                      th_counts_A + m_WL_BinCount[0],
                                      th_counts_B);

    if (!counts_match)
    {
        std::cout << "Mismatch: Structures appear with different frequencies (Counts differ)!" << std::endl;
        GPU_FreeWLBins ();
        return false;
    }

    GPU_FreeWLBins ();

	std::cout<<"------------------------------------------------------------------------"<<std::endl;
	std::cout<<" "<<std::endl;
	std::cout<<"------------------------------------------------------------------------"<<std::endl;

    return true; /** Both are the same possible isomorphic */
}
/*===================================================================================================================*/

/*===================================================================================================================*/
/** WL-1 Test: Iterative Color Refinement returns a histogram of node colors
 * Assumes SignatureCounts was called that created this as input: dram_NodeKeys[gIndex]  */
/*===================================================================================================================*/
bool GPU_WL1GraphColorHashIT( int gIndex, int MAX_ITERATIONS )
{
	int nodeSizeN = m_numNodes[gIndex];
	int edgeSizeN = m_numEdges[gIndex];


    /** 0] Initial base Colors */
    uint64_t* d_node_Colors_Init;
    uint64_t* d_edge_Colors_Init;

	/** 1] Colors Updated each iteration */
	uint64_t* d_node_Colors;
	uint64_t* d_edge_Colors;

	/** 2] Node Bins for Stability Check */
	uint64_t *d_NodeHisto_keys;
	uint     *d_NodeHisto_counts;
    /** 2.1] Temp Array for sorting */
    uint64_t *d_temp_sort_buffer;

    /*-----------------------------------------------------------------*/
    cudaMalloc((void**)&d_node_Colors_Init,  nodeSizeN*sizeof(uint64_t));
    cudaMalloc((void**)&d_edge_Colors_Init,  edgeSizeN*sizeof(uint64_t));

    cudaMalloc((void**)&d_node_Colors,       nodeSizeN*sizeof(uint64_t));
    cudaMalloc((void**)&d_edge_Colors,       edgeSizeN*sizeof(uint64_t));

    cudaMalloc((void**)&d_NodeHisto_keys,        nodeSizeN*sizeof(uint64_t));
    cudaMalloc((void**)&d_NodeHisto_counts,      nodeSizeN*sizeof(uint));

    cudaMalloc((void**)&d_temp_sort_buffer,  nodeSizeN*sizeof(uint64_t));
    cudaDeviceSynchronize();
    cudaCheckError();
    /*-----------------------------------------------------------------*/

	std::cout<<"------------------------------------------------------------------------"<<std::endl;
	std::cout<<"GPU WL1 Class 1st NN Structure Test "<<std::endl;
	std::cout<<"------------------------------------------------------------------------"<<std::endl;

    /*===================================================================================================================*/
    std::cout << "WL1: Initial Coloring Node HashTuple for gIndex " << gIndex << std::endl;
	Kernel_InitNodeColorWL1<<<ThreadsAllNodes[0].dimGrid, ThreadsAllNodes[0].dimBlock>>>( nodeSizeN, dram_NodeColorHashes[gIndex], d_node_Colors_Init);

    std::cout << "WL1: Initial Coloring Edge Label " << std::endl;
	Kernel_InitEdgeColorWL1<<<ThreadsAllEdges[0].dimGrid, ThreadsAllEdges[0].dimBlock>>>(edgeSizeN, dram_Edge_labelDBIndex[gIndex], d_edge_Colors_Init );

    cudaDeviceSynchronize();
	cudaCheckError();
	/*===================================================================================================================*/


    /** First Step current is Init */
    cudaMemcpy(d_node_Colors, d_node_Colors_Init, nodeSizeN * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_edge_Colors, d_edge_Colors_Init, edgeSizeN * sizeof(uint64_t), cudaMemcpyDeviceToDevice);

	int iteration       = 0;
	bool is_stable      = false;
	int h_num_bins      = -1;
	int h_num_bins_Prev = -1;

	/** Will Fail first step  */
	CheckStability(
	            nodeSizeN,
	            d_node_Colors,       // Current Colors
	            d_temp_sort_buffer,  // Temp buffer

	            d_NodeHisto_keys,    // Output: Histogram Keys
	            d_NodeHisto_counts,  // Output: Histogram Counts
	            h_num_bins,          // Output: Current Bin Count

	            h_num_bins_Prev       );

	h_num_bins_Prev = h_num_bins;

	/*===================================================================================================================*/
    /** Color Iteration Loop */
	while (!is_stable && iteration < MAX_ITERATIONS)
	{
	  std::cout << "\n WL1 Iteration " << iteration << " (gIndex " << gIndex << ") " << std::endl;

	  /*===================================================================================================================*/
	  /** 1] Update Edge Colors */
      Kernel_EdgeColorsWL1_SortedBound<<<ThreadsAllEdges[0].dimGrid, ThreadsAllEdges[0].dimBlock>>>(   edgeSizeN, /** Num Edges */

																						d_edge_Colors_Init,  // Input Init Value

																						/** Edge Color comes from the node colors */
																						dram_Edge_nodeSources      [gIndex],
																						dram_Edge_nodeSourcesStart [gIndex],
																						dram_Edge_nodeSourcesNum   [gIndex],
																						d_node_Colors,         //Input (Current Node Colors)

																						dram_Edge_nodeTargets[gIndex],
																						dram_Edge_nodeTargetsStart[gIndex],
																						dram_Edge_nodeTargetsNum[gIndex],

																						d_edge_Colors /* Output*/);
	  cudaDeviceSynchronize();
	  cudaCheckError();
	  /*===================================================================================================================*/

	  /*DD---------------------------------------------------------------------------------------------------------------------DD*/
	  /** Debug Printing */
	  /*DD---------------------------------------------------------------------------------------------------------------------DD*/
			  printf("\n");
			  std::cout << "\n StartEdgeColors for Graph " << gIndex << std::endl;
			  uint64_t *h_EdgeColors;
			  h_EdgeColors = new uint64_t [edgeSizeN];

			  cudaMemcpy(h_EdgeColors, d_edge_Colors, edgeSizeN*sizeof(uint64_t), cudaMemcpyDeviceToHost);
			  cudaDeviceSynchronize();
			  cudaCheckError();
			  for (int i = 0; i < 4; i++)
			  {
				printf(" %d EdgeColor 0x%016llx \n",i, h_EdgeColors[i]);
			  }
			  delete []h_EdgeColors;
			  std::cout << "\n EndEdgeColors for Graph " << gIndex << std::endl;
			  printf("\n");
	  /** DD---------------------------------------------------------------------------------------------------------------------DD*/

	  /*===================================================================================================================*/
	  /**  2] Update Node Colors: If nodes have a large number of edges this can be slow */
	  Kernel_NodeColorsWL1<<<ThreadsAllNodes[0].dimGrid, ThreadsAllNodes[0].dimBlock>>>(   nodeSizeN,

																						d_node_Colors_Init,   // Input Init Value

																						dram_Node_edgePrevs      [gIndex],
																						dram_Node_edgePrevsStart [gIndex],
																						dram_Node_edgePrevsNum   [gIndex],

																						edgeSizeN,
																						dram_Node_edgeNexts[gIndex],
																						dram_Node_edgeNextsStart[gIndex],
																						dram_Node_edgeNextsNum[gIndex],
																						d_edge_Colors,    // Input (Current Edge Colors)

																						d_node_Colors /* Output */ );
	  cudaDeviceSynchronize();
	  cudaCheckError();
	  /*===================================================================================================================*/

	  /** 3. Stability Check */
      is_stable = CheckStability(nodeSizeN, d_node_Colors, d_temp_sort_buffer,
                               d_NodeHisto_keys, d_NodeHisto_counts, h_num_bins,
                                h_num_bins_Prev);


		if (is_stable)
		{
			std::cout << "  Graph is stable." << std::endl;
		}
		else
		{
			h_num_bins_Prev = h_num_bins;
			iteration++;
		}

	} /** End Loop over iterations */
	/*===================================================================================================================*/


    /** Save Histo for the graph  */
	 if (is_stable)
	 {
		std::cout << "WL-1 Stabilized after " << iteration << " iterations." << std::endl;

		/** Store the final size */
		m_WL_BinCount[gIndex] = (uint)h_num_bins_Prev;

		/** Copy to permanent storage */
		if (h_num_bins_Prev > 0)
		{
			size_t keys_bytes   = h_num_bins_Prev * sizeof(uint64_t);
			size_t counts_bytes = h_num_bins_Prev * sizeof(uint);

			cudaMalloc((void**)&dram_WL_BinsColorKeys[gIndex],  keys_bytes);
			cudaMalloc((void**)&dram_WL_BinsNumCount[gIndex], counts_bytes);

			std::cout << "Copy Color Histogram " << std::endl;
	        cudaMemcpy( dram_WL_BinsColorKeys[gIndex],  d_NodeHisto_keys,  keys_bytes,   cudaMemcpyDeviceToDevice );
	        cudaMemcpy( dram_WL_BinsNumCount[gIndex],  d_NodeHisto_counts, counts_bytes, cudaMemcpyDeviceToDevice );

	   	    cudaDeviceSynchronize();
	        cudaCheckError();
	     }
	 }
	 else
	 {
		std::cout << "WL-1 FAILED TO STABILIZE after " << MAX_ITERATIONS << std::endl;
		m_WL_BinCount[gIndex] = 0;
		return false;
	 }

     /** Cleanup */
	 cudaFree(d_node_Colors);
	 cudaFree(d_edge_Colors);
     cudaFree(d_node_Colors_Init);
     cudaFree(d_edge_Colors_Init);
	 cudaFree(d_temp_sort_buffer);

	 /** Free since  dram_WL1_BinsKeys and dram_WL1_BinsCount has the data */
	 cudaFree(d_NodeHisto_keys);
	 cudaFree(d_NodeHisto_counts);

	 cudaDeviceSynchronize();
	 cudaCheckError();

	 std::cout<<"------------------------------------------------------------------------"<<std::endl;
	 std::cout<<" "<<std::endl;
	 std::cout<<"------------------------------------------------------------------------"<<std::endl;

	 return true;
}
/*===================================================================================================================*/

/*===================================================================================================================*/
/** WL-2 Test: Iterative Pair Color Refinement (N^2 Complexity): Assumes SignatureCounts was called that created this as input: dram_NodeKeys[gIndex] */
/*===================================================================================================================*/
bool GPU_WL2GraphPairColoring(int gIndex, int MAX_ITERATIONS)
{
    int nodeSizeN = m_numNodes[gIndex];
    int edgeSizeN = m_numEdges[gIndex];

	std::cout<<"------------------------------------------------------------------------"<<std::endl;
	std::cout<<"GPU WL2 Class Pair Coloring "<<std::endl;
	std::cout<<"------------------------------------------------------------------------"<<std::endl;

    /*-----------------------------------------------------------------*/
    /* WL-2 operates on N*N pairs. We use size_t to prevent overflow for large N */
    size_t NodeMatrixSize = (size_t)nodeSizeN * (size_t)nodeSizeN;

    /* Calculate Real VRAM Usage:
           - 3x uint64 arrays (Matrix, StepUpdate, HistoKeys) = 24 bytes per pair
           - 1x uint32 array (HistoCounts) = 4 bytes per pair
           - Total: 28 bytes * N^2  */
     double required_MB = (double)NodeMatrixSize * 28.0 / (1024.0 * 1024.0);
    if (required_MB > 0.75*m_MaxGPUMemoryMB )
    {
        std::cout << "WL-2 Skipped: Graph too large (" << nodeSizeN << " nodes)." << std::endl;
        m_WL_BinCount[gIndex] = 0;
        return false;
    }

    /* 1] One 64Bit color per element pair */
    uint64_t *d_MatrixEleColors;
    cudaMalloc((void**)&d_MatrixEleColors,    NodeMatrixSize * sizeof(uint64_t));
    cudaDeviceSynchronize();
    cudaCheckError();
    /*-----------------------------------------------------------------*/

    /*===================================================================================================================*/
    /* A] Initialization (Phase 0) Thread per node */
    std::cout << "WL-2 Init: Background for gIndex " << gIndex << std::endl;
    Kernel_WL2_Init_Background<<<ThreadsNodePairs[0].dimGrid, ThreadsNodePairs[0].dimBlock>>>(NodeMatrixSize, d_MatrixEleColors);
    cudaDeviceSynchronize();
    cudaCheckError();

    /** A1] Thread Per Node: Init Diagonals of d_pair_Colors_In which are the same node using dram_NodeKeys    */
    std::cout << "WL-2 Init: Diagonals " << std::endl;
    Kernel_WL2_Init_Diagonal<<<ThreadsAllNodes[0].dimGrid, ThreadsAllNodes[0].dimBlock>>>(  nodeSizeN,
																							dram_NodeColorHashes[gIndex],
																							d_MatrixEleColors	);

    bool isDebugBrute = true;

    if(isDebugBrute)
    {
		/** A2] Thread Per Edge: Init Hyperedges (Expand Sources x Targets into SubMatrix of d_MatrixEleColors)
		 *  Brute Force Seriel Atomic Updates */
		std::cout << "WL-2 Init: Hyperedges (Brute Force)" << std::endl;
		Kernel_WL2_Init_HyperEdges_Brute<<<ThreadsAllEdges[0].dimGrid, ThreadsAllEdges[0].dimBlock>>>(
			edgeSizeN,
			d_MatrixEleColors,
			nodeSizeN,
			dram_Edge_nodeSources      [gIndex],
			dram_Edge_nodeSourcesStart [gIndex],
			dram_Edge_nodeSourcesNum   [gIndex],
			dram_Edge_nodeTargets      [gIndex],
			dram_Edge_nodeTargetsStart [gIndex],
			dram_Edge_nodeTargetsNum   [gIndex],
			dram_Edge_labelDBIndex     [gIndex]  );
    }
    else
    {
        std::cout << "WL-2 Init: Hyperedges (Warp Optimized SM)" << std::endl;
		/* Configuration for Warp-Centric Kernel */
		int threadsPerBlock = 256;             // Standard block size
		int warpsPerBlock   = threadsPerBlock / 32; // = 8 Warps per block

		/* We need 1 Warp per Edge.
		   Total Warps needed = NumEdges.
		   Total Blocks = NumEdges / 8 */
		int numBlocks = (edgeSizeN + warpsPerBlock - 1) / warpsPerBlock;

		Kernel_WL2_Init_HyperEdges_WarpOptimized<<<numBlocks, threadsPerBlock>>>(
			edgeSizeN,
			d_MatrixEleColors,
			nodeSizeN,
			dram_Edge_nodeSources      [gIndex],
			dram_Edge_nodeSourcesStart [gIndex],
			dram_Edge_nodeSourcesNum   [gIndex],
			dram_Edge_nodeTargets      [gIndex],
			dram_Edge_nodeTargetsStart [gIndex],
			dram_Edge_nodeTargetsNum   [gIndex],
			dram_Edge_labelDBIndex     [gIndex]
		);
    }

    cudaDeviceSynchronize();
    cudaCheckError();
    /*===================================================================================================================*/



    /*-----------------------------------------------------------------*/
    /* B0] Temp Array for sorting and Writing colors  */
    uint64_t *d_MatrixEleColorsStepUpdate;
    cudaMalloc((void**)&d_MatrixEleColorsStepUpdate,   NodeMatrixSize * sizeof(uint64_t));
    /*-----------------------------------------------------------------*/

    /*-----------------------------------------------------------------*/
    /* B0.1] Initial Stability Check */
    int iteration       = 0;
    bool is_stable      = false;
    int h_num_bins      = -1;
    int h_num_bins_Prev = -1;

    /* B2] Stability Check BuffersSame logic as WL-1 */
    uint64_t *d_Histo_keys;
    uint     *d_Histo_counts;
    cudaMalloc((void**)&d_Histo_keys,        NodeMatrixSize * sizeof(uint64_t));
    cudaMalloc((void**)&d_Histo_counts,      NodeMatrixSize * sizeof(uint));
    /*-----------------------------------------------------------------*/



    /* Will Fail first step reuses WL1 Check */
    is_stable = CheckStability(NodeMatrixSize, d_MatrixEleColors, d_MatrixEleColorsStepUpdate,
                               d_Histo_keys, d_Histo_counts, h_num_bins,
                               h_num_bins_Prev);

    h_num_bins_Prev = h_num_bins;


    /** CUDA:  Define 2D Grid for N*N Matrix (16x16 blocks) */
    dim3 dimBlock2D(16, 16);
    dim3 dimGrid2D((nodeSizeN + dimBlock2D.x - 1) / dimBlock2D.x,
                   (nodeSizeN + dimBlock2D.y - 1) / dimBlock2D.y);

    /*===================================================================================================================*/
    while (!is_stable && iteration < MAX_ITERATIONS)
    {
        std::cout << "\n WL-2 Iteration " << iteration << " (gIndex " << gIndex << ") " << std::endl;

        /** Read From d_MatrixEleColors and then Write to  d_MatrixEleColorsStepUpdate:
         *  Each element checks if by scanning its row/col if it can find a non zero location this tells use if we can do
         *  A->B->C  if both are true then we can do C->B->A also and it is a triple (discarded for now) */
        Kernel_WL2_UpdatePairs_Tiled<<<dimGrid2D, dimBlock2D>>>(nodeSizeN, d_MatrixEleColors, d_MatrixEleColorsStepUpdate);
        cudaDeviceSynchronize();
        cudaCheckError();

        /* Check if this step is stable  */
        is_stable = CheckStability(NodeMatrixSize, d_MatrixEleColorsStepUpdate, d_MatrixEleColors,
                                   d_Histo_keys, d_Histo_counts, h_num_bins, h_num_bins_Prev);

        if (is_stable)
        {
			uint64_t* temp = d_MatrixEleColors;
			d_MatrixEleColors = d_MatrixEleColorsStepUpdate;
			d_MatrixEleColorsStepUpdate = temp;

			std::cout << "  Graph Stable at Iteration " << iteration << std::endl;
			break; // Exit Loop
         }
         else
         {
			// Standard Loop Swap
			uint64_t* temp = d_MatrixEleColors;
			d_MatrixEleColors = d_MatrixEleColorsStepUpdate;
			d_MatrixEleColorsStepUpdate = temp;
			h_num_bins_Prev = h_num_bins;
			iteration++;
         }
    }
    /*===================================================================================================================*/



    /* C] Save Results */
    if (is_stable && h_num_bins_Prev > 0)
    {
        std::cout << "WL-2 Stabilized after " << iteration << " iterations." << std::endl;

        /* Store WL2 Histogram Size */
        m_WL_BinCount[gIndex] = (uint)h_num_bins_Prev;

        /* Copy to permanent storage */
        size_t keys_bytes   = h_num_bins_Prev * sizeof(uint64_t);
        size_t counts_bytes = h_num_bins_Prev * sizeof(uint);

        cudaMalloc((void**)&dram_WL_BinsColorKeys[gIndex],  keys_bytes);
        cudaMalloc((void**)&dram_WL_BinsNumCount[gIndex], counts_bytes);

        std::cout << "Copy WL-2 Color Histogram " << std::endl;
        cudaMemcpy( dram_WL_BinsColorKeys[gIndex],  d_Histo_keys,   keys_bytes,   cudaMemcpyDeviceToDevice );
        cudaMemcpy( dram_WL_BinsNumCount[gIndex], d_Histo_counts, counts_bytes, cudaMemcpyDeviceToDevice );

        dram_WL2_MatrixColors[gIndex] = d_MatrixEleColors;
        m_isWL2Alloc[gIndex] = true;

        cudaDeviceSynchronize();
        cudaCheckError();
    }
    else
    {
        std::cout << "WL-2 FAILED TO STABILIZE after " << MAX_ITERATIONS << std::endl;
        m_WL_BinCount[gIndex] = 0;
        return false;
    }

    /* Cleanup */
    cudaFree(d_MatrixEleColorsStepUpdate);
    cudaFree(d_Histo_keys);
    cudaFree(d_Histo_counts);

    cudaDeviceSynchronize();
    cudaCheckError();

	std::cout<<"------------------------------------------------------------------------"<<std::endl;
	std::cout<<" "<<std::endl;
	std::cout<<"------------------------------------------------------------------------"<<std::endl;


    return true;
}
/*===================================================================================================================*/


