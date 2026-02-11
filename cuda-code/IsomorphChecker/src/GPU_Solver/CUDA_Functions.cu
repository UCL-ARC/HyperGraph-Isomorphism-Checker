/*
 * CUDA_Functions.cu
 *
 * Created on: Oct 23, 2025
 * Updated on: Feb 05, 2026 (Formal Proof Checker - Dense WL-2 / Sparse WL-3)
 * Author: Nicolin Govender UCL-ARC
 */

/*-------------------------------------------------------------------------------------------------------------------*/
/* CUDA */
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/* CUDA Boost like lib for sorting and other standard ops */
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/equal.h>

#include <thrust/inner_product.h>
#include <thrust/extrema.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/binary_search.h>
#include <thrust/unique.h>

#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <sstream>
#include <vector>
#include <map>
#include <chrono>
#include <unordered_map>
/*-------------------------------------------------------------------------------------------------------------------*/



/*===================================================================================================================*/
/* NG WL Matrix Based Solver */
/*===================================================================================================================*/

/*-------------------------------------------------------------------------------------------------------------------*/
/** 128 bit: 16 Bytes Aligned:  Node signatures Label, IO, numNexts, numPrevs  */
/*-------------------------------------------------------------------------------------------------------------------*/
typedef thrust::tuple<uint, uint, uint, uint> NodeKeyTuple;
NodeKeyTuple MAX_TUPLEH = thrust::make_tuple(UINT_MAX, 0, 0, 0); /** Host Side {default value} */
/*-------------------------------------------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------------------------------------------*/
/* Sparse WL-3 Data Structures                                                                                       */
/*-------------------------------------------------------------------------------------------------------------------*/
/* Sparse triplet entry: Only store non-zero relationships */
struct SparseTriplet
{
    uint u, v, w;           /* 12 bytes: Triplet indices */
    uint64_t color;         /* 8 bytes: Hash color */
};  /* Total: 20 bytes per triplet */

/* Metadata for sparse tensor */
struct SparseTensorInfo
{
    int num_triplets;       /* Number of non-zero triplets */
    int capacity;           /* Allocated capacity */
    SparseTriplet *d_data;  /* Device pointer to triplet array */
};

/* Used to check if it is worth to run a WL3 test in the first place */
struct WL3SymmetryProfile
{
    int    num_bins;
    int    largest_bin;
    int    smallest_bin;
    double avg_bin_size;
    double symmetry_score;  /* 0.0 = discrete, 1.0 = complete symmetry */
    std::vector<int> bin_sizes;
};
/*===================================================================================================================*/


/*-------------------------------------------------------------------------------------------------------------------*/
/* Hashing Constants */
/*-------------------------------------------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------------------------------------------*/


#include "CUDA_Kernels.cuh"


/* Note: dram means it persistent in GPU memory  and d_ means it is allocated just within that function */

/*===================================================================================================================*/
/* Z] Memory needed for Graph input storage in CSR Format O (N) O(NE) Note [2] means 2 Graphs */
/*===================================================================================================================*/

/*-------------------------------------------------------------------------------------------------------------------*/
/** Z1] Node Struct compact list that we will copy to GPU */
/*-------------------------------------------------------------------------------------------------------------------*/
uint  m_numNodes               [2] = {};                             /** Total Number of Nodes in a graph  */
uint  m_nodeEdgesPrevsSize     [2] = {}, nodeEdgesNextsSize[2] = {}; /** Size of the compact arrays for node Edges Prevs and Nexts */

/** Per Node */
uint *dram_Node_labelDBIndex   [2];    /** index of the label that identifies the node type  */
uint *dram_Node_IOTag          [2];    /** 0 = none is the node a 1= global input, 2= output or 3= both */

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
/** Z2] Edge struct compact list */
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

/*===================================================================================================================*/
/* End 0] Graph Input Memory */
/*===================================================================================================================*/



/*===================================================================================================================*/
/* Y] Memory needed for our WL1/2/3 Implementation N=numNodes */
/*===================================================================================================================*/

NodeKeyTuple *dram_NodeColorHashes    [2];/** Y1] Node Signature Color: 128bit per Node O(N) */

/*-------------------------------------------------------------------------------------------------------------------*/
/** ISOMorph: WL-1 Test */
uint64_t  *dram_WL_BinsColorKeys  [2]; /** Final WL Histogram Keys (Hashes) max O(N) */
uint      *dram_WL_BinsNumCount   [2]; /** Final WL Histogram Counts max O(N) */
uint       m_WL_BinCount          [2] = {0,0}; /* Number of Bins for each graph */
/*-------------------------------------------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------------------------------------------*/
/** ISOMorph: WL-2 Test uses adjacency matrix: For sparse cases we can use COO or CSR which will cost performance on GPU */
uint64_t      *dram_WL2_MatrixColors    [2] = {0, 0};     /* Y2] We store pairs of nodes so NNode*NNodes matrix for dense  */
/*-------------------------------------------------------------------------------------------------------------------*/

/*===================================================================================================================*/
/* End 1] Memory needed for our WL1/2/3 Implementation  */
/*===================================================================================================================*/

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

launchParms   ThreadsWLMatrix2DTiled [64]; /** Node Pairs for adj matrix */
/*-----------------------------------------------------------------------------*/

double m_MaxGPUMemoryMB = 8000.0;   /** Query GPU Memory */


/*-----------------------------------------------------------------------------*/
/* Host Side Structs for WL implementation */
/*-----------------------------------------------------------------------------*/
/* Stack Frame: Keeps track of where we are in the decision tree for IR check  */
struct IR_StackFrame
{
    int pivotNode;              /* The node we chose to "Pin" at this depth */
    int candidateIndex;         /* The index in the candidate list we are currently testing */
    std::vector<int> candidates; /* The list of valid nodes in G1 to map to */
};

struct BinInfoGPU
{
    int isTargetColorFound = 0; /* Boolean flag (0 or 1) */
    int pivotNode = -1;
    std::vector<int> candidates; /* Tiny vector (indices only) */
};
/*-----------------------------------------------------------------------------*/



//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
        cudaError_t e=cudaGetLastError();                                 \
        if(e!=cudaSuccess) {                                              \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }

bool  m_isWL1Alloc       [2] = {0,0};
bool  m_isWL2Alloc       [2] = {0,0};
/*-----------------------------------------------------------------------------*/


/*===================================================================================================================*/
/** AA] Init GPU Arrays for input stoarge */
/*===================================================================================================================*/
void GPU_InitArrays( uint gIndex,

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

	if (gIndex==0 || (m_numEdges [gIndex] >= m_numEdges[0]) )
	{
	  ThreadsAllEdges[gpu].dimBlock = make_uint3(numThreadsBlock, 1, 1);
	  ThreadsAllEdges[gpu].dimGrid  = make_uint3(numBlocksSM * num_sm, 1, 1);

	  int totalBlocks = numBlocksSM * num_sm;
	  ThreadsAllEdges[gpu].dimGrid  = make_uint3(totalBlocks > 0 ? totalBlocks : 1, 1, 1);
	}
	/*-----------------------------------------------------------------------------*/

//	std::cout<<" NodeNexts Print \n";
//	printItem<<<1,1>>> (dram_Node_edgeNexts[gIndex], dram_Node_edgeNextsStart[gIndex],dram_Node_edgeNextsNum[gIndex], dram_Node_labelDBIndex[gIndex], m_numNodes[gIndex], 1);
//	cudaDeviceSynchronize();
//
//	std::cout<<" NodePrevs Print \n";
//	printItem<<<1,1>>> (dram_Node_edgePrevs[gIndex], dram_Node_edgePrevsStart[gIndex],dram_Node_edgePrevsNum[gIndex], dram_Node_labelDBIndex[gIndex], m_numNodes[gIndex], 1);
//
//	std::cout<<" EdgeSources Print \n";
//	printItem<<<1,1>>> (dram_Edge_nodeSources[gIndex], dram_Edge_nodeSourcesStart[gIndex], dram_Edge_nodeSourcesNum[gIndex], dram_Edge_labelDBIndex[gIndex], m_numEdges[gIndex], 1);
//	cudaDeviceSynchronize();
//
//	std::cout<<" EdgeTargets Print \n";
//	printItem<<<1,1>>> (dram_Edge_nodeTargets[gIndex], dram_Edge_nodeTargetsStart[gIndex],dram_Edge_nodeTargetsNum[gIndex], dram_Edge_labelDBIndex[gIndex], m_numEdges[gIndex], 1);

//	std::cout<<" EdgeTargets Print \n";
//	printItem<<<1,m_numEdges[gIndex]>>> (dram_Edge_nodeTargets[gIndex], dram_Edge_nodeTargetsStart[gIndex],dram_Edge_nodeTargetsNum[gIndex], dram_Edge_labelDBIndex[gIndex], m_numEdges[gIndex], m_numEdges[gIndex]);


	/* Configure 2D Grid for Matrix Operations (WL-2 Refinement) */
	    /* We use 16x16 tiles (256 threads) to match WARP_CACHESIZE in the kernel */
	int TILE_DIM = 16;
	dim3 dimBlock2D(TILE_DIM, TILE_DIM, 1);

	/* Calculate grid size to cover the whole matrix */
	int numTiles = (m_numNodes[gIndex] + TILE_DIM - 1) / TILE_DIM;
	dim3 dimGrid2D(numTiles, numTiles, 1);

	/* Save to your launch parameter struct */
	/* Assuming you renamed ThreadsNodePairs to ThreadsWLMatrix2DTiled */
	ThreadsWLMatrix2DTiled[gpu].dimBlock = dimBlock2D;
	ThreadsWLMatrix2DTiled[gpu].dimGrid  = dimGrid2D;

	cudaDeviceSynchronize();
	cudaCheckError();
}
/*===================================================================================================================*/


/*===================================================================================================================*/
/** AB] Free Init GPU Arrays */
/*===================================================================================================================*/
void GPU_FreeInitArrays (uint gIndex, uint gpu)
{
	cudaDeviceSynchronize();
	cudaSetDevice(gpu);

	std::cout<<"GPU Free Graph "<<gIndex<<std::endl;

	/*-----------------------------------------------------------------------------*/
	/* Input Storage */
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
	/*-----------------------------------------------------------------------------*/


	cudaDeviceSynchronize();
	cudaCheckError();
	cudaSetDevice(gpu);

	/*-----------------------------------------------------------------------------*/
	/* NG WL Method */
	std::cout<<"GPU Free WL Memory "<<gIndex<<std::endl;
	/* Colors Created on GPU */
	if (m_isWL1Alloc[gIndex])
	{
	 cudaFree(dram_NodeColorHashes[gIndex]);
	 cudaCheckError();
	}

	/* Dense Matrix */
	if (m_isWL2Alloc[gIndex])
	{
	  cudaFree(dram_WL2_MatrixColors[gIndex]);
	  cudaCheckError();
	}
	/*-----------------------------------------------------------------------------*/


	cudaDeviceSynchronize();
	cudaError_t errormsg=cudaGetLastError();
	cudaCheckError();
	if(errormsg>0)
	{
	 std::cout<<"GPU Free Memory "<<cudaGetErrorString(errormsg)<<std::endl;
	 exit(1);
	}

}
/*===================================================================================================================*/

/*===================================================================================================================*/
/** AC] Free GPU Arrays */
/*===================================================================================================================*/
void GPU_FreeWLBins ()
{
	for (int gIndex=0;gIndex<2;gIndex++)
	{
		std::cout<<"GPU Free WL Bins "<<gIndex<<std::endl;
		cudaFree (dram_WL_BinsColorKeys[gIndex]);
		cudaFree (dram_WL_BinsNumCount[gIndex]);
		cudaDeviceSynchronize();
		cudaCheckError();
	}
}
/*===================================================================================================================*/


/*===================================================================================================================*/
/* A] Does the binning on the GPU for the feature counts of each edge and node
 * GlobalOutput: dram_NodeColorHashes which is the "hash/color" is the output for other functions
 *
 * Computational Cost: O(E log E + N log N)
 * - Sorting is the bottleneck. Edge sorting O(E log E), Node sorting O(N log N)
 * - Reduction and equality checks are linear O(N + E)
 * Memory Cost: O(N + E)
 * - Requires temporary buffers for Keys and Counts for both nodes and edges */
/*===================================================================================================================*/
bool GPU_CompareSignatureCountsBetweenGraphs()
{
    bool isPossibleIsomorphic = false;

    std::cout<<"------------------------------------------------------------------------"<<std::endl;
    std::cout<<"GPU Bulk Test 0 Signature counts "<<std::endl;
    std::cout<<"------------------------------------------------------------------------"<<std::endl;

    /*---------------------------------------------------------------------------------------------------------------*/
    /* A1] Start Edge Histogram */
    /*---------------------------------------------------------------------------------------------------------------*/
    typedef thrust::tuple<uint, uint, uint, uint> EdgeKeyTuple;

    EdgeKeyTuple *d_HistoEdgeKeys      [2]; /* Sorted keys array */
    uint         *d_HistoEdgeKeyCounts [2]; /* Bin counts for sorted keys array */
    int          numUniqueKeyBinsGraph [2]; /* Store Tot bins per graph */

    for (int gIndex=0; gIndex<2; gIndex++)
    {
        cudaDeviceSynchronize();
        cudaCheckError();

        std::cout << "\nProcessing Node Histogram for graph " << gIndex << " Edges "<< m_numEdges[gIndex] <<std::endl;
        cudaMalloc((void**)&d_HistoEdgeKeys[gIndex], m_numEdges[gIndex]*sizeof(EdgeKeyTuple));
        cudaMalloc((void**)&d_HistoEdgeKeyCounts[gIndex], m_numEdges[gIndex]*sizeof(uint));
        cudaCheckError();

        thrust::device_ptr<EdgeKeyTuple> d_ptr_output_ColorHashkeys(d_HistoEdgeKeys[gIndex]);
        thrust::device_ptr<uint>         d_ptr_output_counts(d_HistoEdgeKeyCounts[gIndex]);

        //std::cout<<"Creating zip iterator for graph "<<gIndex<<std::endl;

        /* 1] Create tuple (SourceDeg, TargetDeg, TotalNodes, Label) */
        auto keys_begin = thrust::make_zip_iterator(thrust::make_tuple(
                                                    thrust::device_ptr<uint>(dram_Edge_nodeSourcesNum[gIndex]),
                                                    thrust::device_ptr<uint>(dram_Edge_nodeTargetsNum[gIndex]),
                                                    thrust::device_ptr<uint>(dram_Edge_nodeTot[gIndex]),
                                                    thrust::device_ptr<uint>(dram_Edge_labelDBIndex[gIndex]) ) );
        auto keys_end = keys_begin + m_numEdges[gIndex];
        auto values_begin = thrust::make_constant_iterator(1u);

        /* 2] Sort keys before reducing to ensure correct histogram logic */
        //std::cout<<"Sorting edge keys for graph "<<gIndex<<std::endl;
        thrust::sort(keys_begin, keys_end);
        cudaDeviceSynchronize();

        //std::cout<<"Binning edge signatures for graph "<<gIndex<<std::endl;
        auto new_end = thrust::reduce_by_key( keys_begin, keys_end, values_begin,
                                              d_ptr_output_ColorHashkeys, d_ptr_output_counts );

        cudaDeviceSynchronize();
        cudaCheckError();

        numUniqueKeyBinsGraph[gIndex] = new_end.first - d_ptr_output_ColorHashkeys;
        std::cout<<"Graph "<<gIndex<<" numUniqueEdgeBins: "<<numUniqueKeyBinsGraph[gIndex]<<std::endl;
    }
    std::cout << "GPU Edge Binning Complete" << std::endl;


    /* Compare Edge Histo counts */
    bool edgesMatch = false;
    if (numUniqueKeyBinsGraph[0] != numUniqueKeyBinsGraph[1])
    {
        std::cout << "Result: NOT Isomorphic - Edge bin counts differ" << std::endl;
    }
    else
    {
        std::cout << "Edge bin counts match: Comparing arrays" << std::endl;
        int num_bins = numUniqueKeyBinsGraph[0];

        thrust::device_ptr<EdgeKeyTuple> keys_A(d_HistoEdgeKeys[0]);
        thrust::device_ptr<EdgeKeyTuple> keys_B(d_HistoEdgeKeys[1]);
        thrust::device_ptr<uint> counts_A(d_HistoEdgeKeyCounts[0]);
        thrust::device_ptr<uint> counts_B(d_HistoEdgeKeyCounts[1]);

        bool areKeysEqual = thrust::equal(keys_A, keys_A + num_bins, keys_B);

        if (!areKeysEqual)
        {
            std::cout << "Result: NOT Isomorphic - Edge bin keys differ" << std::endl;
        }
        else
        {
            std::cout << "Edge keys match - Comparing counts" << std::endl;
            bool are_counts_equal = thrust::equal(counts_A, counts_A + num_bins, counts_B);

            if (!are_counts_equal)
            {
                std::cout << "Result: NOT Isomorphic - Edge bin counts differ" << std::endl;
            }
            else
            {
                std::cout << "Result: Edge histograms match - Possible isomorphic " << std::endl;
                edgesMatch = true;
            }
        }
    }

    for (int gIndex=0; gIndex<2; gIndex++)
    {
        cudaFree(d_HistoEdgeKeys[gIndex]);
        cudaFree(d_HistoEdgeKeyCounts[gIndex]);
    }
    /*---------------------------------------------------------------------------------------------------------------*/
    /* End A1] Edge Histogram */
    /*---------------------------------------------------------------------------------------------------------------*/


    /*---------------------------------------------------------------------------------------------------------------*/
    /* A2] Start Node Histogram */
    /*---------------------------------------------------------------------------------------------------------------*/
    if (edgesMatch)
    {
        NodeKeyTuple *d_HistoNodeKeys[2];
        uint *d_HistoNodeKeyCounts[2];
        int numUniqueKeyBinsNodes[2];

        for (int gIndex = 0; gIndex < 2; gIndex++)
        {
            std::cout << "\nProcessing Node Histogram for graph " << gIndex<< " Nodes "<< m_numNodes[gIndex]<<std::endl;

            uint *d_temp_labels;
            uint *d_temp_IOTag;
            uint *d_temp_numNexts;
            uint *d_temp_numPrevs;

            cudaMalloc((void**)&d_HistoNodeKeys[gIndex], m_numNodes[gIndex] * sizeof(NodeKeyTuple));
            cudaMalloc((void**)&d_HistoNodeKeyCounts[gIndex], m_numNodes[gIndex] * sizeof(uint));

            cudaMalloc((void**)&d_temp_labels, m_numNodes[gIndex] * sizeof(uint));
            cudaMalloc((void**)&d_temp_IOTag, m_numNodes[gIndex] * sizeof(uint));
            cudaMalloc((void**)&d_temp_numNexts, m_numNodes[gIndex] * sizeof(uint));
            cudaMalloc((void**)&d_temp_numPrevs, m_numNodes[gIndex] * sizeof(uint));
            cudaCheckError();

            cudaMemcpy(d_temp_labels, dram_Node_labelDBIndex[gIndex], m_numNodes[gIndex]*sizeof(uint), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_temp_IOTag, dram_Node_IOTag[gIndex], m_numNodes[gIndex]*sizeof(uint), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_temp_numNexts, dram_Node_edgeNextsNum[gIndex], m_numNodes[gIndex]*sizeof(uint), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_temp_numPrevs, dram_Node_edgePrevsNum[gIndex], m_numNodes[gIndex]*sizeof(uint), cudaMemcpyDeviceToDevice);
            cudaCheckError();

            //std::cout << "Creating 4-tuple node signature" << std::endl;
            auto node_keys_begin = thrust::make_zip_iterator(thrust::make_tuple(
                thrust::device_ptr<uint>(d_temp_labels),
                thrust::device_ptr<uint>(d_temp_IOTag),
                thrust::device_ptr<uint>(d_temp_numNexts),
                thrust::device_ptr<uint>(d_temp_numPrevs)
            ));
            auto node_keys_end = node_keys_begin + m_numNodes[gIndex];

            /* 1] Store node signatures for later use (WL-1 Init) */
            cudaMalloc((void**)&dram_NodeColorHashes[gIndex], m_numNodes[gIndex]*sizeof(NodeKeyTuple));
            m_isWL1Alloc[gIndex] = true;

            thrust::device_ptr<NodeKeyTuple> d_ptr_output_lookup(dram_NodeColorHashes[gIndex]);
            thrust::copy(node_keys_begin, node_keys_end, d_ptr_output_lookup);
            cudaDeviceSynchronize();

            /* 2] Sort for binning */
            //std::cout << "Sorting node signatures" << std::endl;
            thrust::sort(node_keys_begin, node_keys_end);
            cudaDeviceSynchronize();

            //std::cout << "Binning node signatures into histogram" << std::endl;
            thrust::device_ptr<NodeKeyTuple> d_ptr_node_hist_keys(d_HistoNodeKeys[gIndex]);
            thrust::device_ptr<uint> d_ptr_node_hist_counts(d_HistoNodeKeyCounts[gIndex]);

            auto values_begin = thrust::make_constant_iterator(1u);

            auto hist_end = thrust::reduce_by_key( node_keys_begin, node_keys_end, values_begin,
                                                    d_ptr_node_hist_keys, d_ptr_node_hist_counts );

            cudaDeviceSynchronize();
            cudaCheckError();

            numUniqueKeyBinsNodes[gIndex] = hist_end.first - d_ptr_node_hist_keys;
            std::cout<<"Graph "<<gIndex<<" numUniqueNodeBins: "<<numUniqueKeyBinsNodes[gIndex] <<std::endl;

            cudaFree(d_temp_labels);
            cudaFree(d_temp_IOTag);
            cudaFree(d_temp_numNexts);
            cudaFree(d_temp_numPrevs);
        }

        /* Compare Node Histo counts */
        std::cout << "\nNode Histogram Comparison" << std::endl;
        if (numUniqueKeyBinsNodes[0] != numUniqueKeyBinsNodes[1])
        {
            std::cout << "Result: NOT Isomorphic - Node bin counts differ" << std::endl;
        }
        else
        {
            int num_bins = numUniqueKeyBinsNodes[0];
            std::cout << "Node bin counts match " << num_bins << " Comparing arrays" << std::endl;

            thrust::device_ptr<NodeKeyTuple> keys_A(d_HistoNodeKeys[0]);
            thrust::device_ptr<NodeKeyTuple> keys_B(d_HistoNodeKeys[1]);
            bool keys_match = thrust::equal(keys_A, keys_A + num_bins, keys_B);

            if (!keys_match)
            {
                std::cout << "Result: NOT Isomorphic - Node keys differ" << std::endl;
            }
            else
            {
                thrust::device_ptr<uint> counts_A(d_HistoNodeKeyCounts[0]);
                thrust::device_ptr<uint> counts_B(d_HistoNodeKeyCounts[1]);
                bool counts_match = thrust::equal(counts_A, counts_A + num_bins, counts_B);

                if (counts_match)
                {
                    std::cout << "Result: Node histograms match - Possible Isomorphic" << std::endl;
                    isPossibleIsomorphic = true;
                }
                else
                {
                    std::cout << "Result: NOT Isomorphic - Node counts differ" << std::endl;
                }
            }
        }

        /* Clean up node histograms */
        for (int gIndex=0; gIndex<2; gIndex++)
        {
            cudaFree(d_HistoNodeKeys[gIndex]);
            cudaFree(d_HistoNodeKeyCounts[gIndex]);
        }
    }
    /*---------------------------------------------------------------------------------------------------------------*/
    /* End A2] Node Histogram */
    /*---------------------------------------------------------------------------------------------------------------*/

    std::cout<<"------------------------------------------------------------------------"<<std::endl;
    std::cout<<""<<std::endl;
    std::cout<<"------------------------------------------------------------------------"<<std::endl;
    std::cout<<""<<std::endl;
    return isPossibleIsomorphic;
}
/*===================================================================================================================*/

/*-----------------------------------------------------------------------------------------------*/
/* Helper: Analyzes edge signatures to report structural diversity                               */
/* Usage: Call after sorting edge hashes in Phase 2                                              */
/*-----------------------------------------------------------------------------------------------*/
void LogEdgeRefinementStats(int numEdges, uint64_t* d_EdgeHashes, const char* stageLabel)
{
	/* Download edge hashes to Host for canonical analysis */
	    std::vector<uint64_t> h_Edges(numEdges);
	    cudaMemcpy(h_Edges.data(), d_EdgeHashes, numEdges * sizeof(uint64_t), cudaMemcpyDeviceToHost);

	    /* Sort to group identical edge signatures canonically */
	    std::sort(h_Edges.begin(), h_Edges.end());

	    /* Compute Stats */
	    int totalBins = 0;
	    int uniqueEdges = 0;
	    int maxBinSize = 0;
	    int currentBinSize = 0;

	    if (numEdges > 0)
	    {
	        currentBinSize = 1;
	        totalBins = 1;
	        for (int i = 1; i < numEdges; i++)
	        {
	            if (h_Edges[i] == h_Edges[i-1])
	            {
	                currentBinSize++;
	            }
	            else
	            {
	                if (currentBinSize == 1) uniqueEdges++;
	                if (currentBinSize > maxBinSize) maxBinSize = currentBinSize;
	                currentBinSize = 1;
	                totalBins++;
	            }
	        }
	        /* Handle final edge bin */
	        if (currentBinSize == 1) uniqueEdges++;
	        if (currentBinSize > maxBinSize) maxBinSize = currentBinSize;
	    }

	    int ambiguousEdges = numEdges - uniqueEdges;

	    /* Print Formatted Report */
	    std::cout << "------------------------------------------------------------------------" << std::endl;
	    std::cout << " " << stageLabel << " Edge Structural Diversity: " << std::endl;
	    std::cout << "   - Unique Edge Signatures: " << totalBins << std::endl;
	    std::cout << "   - Distinctly Identified:  " << uniqueEdges << std::endl;
	    std::cout << "   - Signature Collisions:    " << ambiguousEdges << std::endl;
	    std::cout << "   - Max Collision Bin:       " << maxBinSize << std::endl;
	    std::cout << "------------------------------------------------------------------------" << std::endl;
}
/*-----------------------------------------------------------------------------------------------*/

/*===================================================================================================================*/
/* B] Uses the Node Keys of an edge to create a hash
 * Assumes SignatureCounts was called that created this as GlobalInput: dram_NodeColorHashes[gIndex]
 *
 * Computational Cost: O(E log E) due to sorting for comparison. Hashing is O(E)
 * Memory Cost: O(E) for temporary source/target hash arrays */
/*===================================================================================================================*/
bool GPU_CompareEdgesSignaturesBetweenGraphs()
{
  bool isPossibleIsomorphic = false;

  /* Hashes are stored here */
  uint64_t *d_temp_EdgeHashSources[2];
  uint64_t *d_temp_EdgeHashTargets[2];

  std::cout << "------------------------------------------------------------------------" << std::endl;
  std::cout << "GPU Structure Test 1 Edge Colors " << std::endl;
  std::cout << "------------------------------------------------------------------------" << std::endl;

  /*-------------------------------------------------------------------------*/
  /* B1] Create Hashes for the edges */
  for (int gIndex = 0; gIndex < 2; gIndex++)
  {
    /* Allocate arrays */
    cudaMalloc((void**)&d_temp_EdgeHashSources[gIndex], m_numEdges[gIndex] * sizeof(uint64_t));
    cudaMalloc((void**)&d_temp_EdgeHashTargets[gIndex], m_numEdges[gIndex] * sizeof(uint64_t));
    cudaDeviceSynchronize();
    cudaCheckError();

    /*  Enforces Tuple Order (A,B) != (B,A) */
    Kernel_EdgeHashes_PortOrderPreserving<<<ThreadsAllEdges[0].dimGrid, ThreadsAllEdges[0].dimBlock>>>(
        m_numEdges[gIndex],

        dram_Edge_nodeSources[gIndex],
        dram_Edge_nodeSourcesStart[gIndex],
        dram_Edge_nodeSourcesNum[gIndex],

        dram_Edge_nodeTargets[gIndex],
        dram_Edge_nodeTargetsStart[gIndex],
        dram_Edge_nodeTargetsNum[gIndex],

        dram_NodeColorHashes[gIndex],

        d_temp_EdgeHashSources[gIndex],
        d_temp_EdgeHashTargets[gIndex]
    );

    cudaDeviceSynchronize();
    cudaCheckError();

  } /* End Loop over graphs */

  /* End B1] Create Hashes for the edges */
  /*-------------------------------------------------------------------------*/


  /*-------------------------------------------------------------------------*/
  /* B2] Sort the hashes for each graph */
  /*-------------------------------------------------------------------------*/
  int num_edges = m_numEdges[0];

  std::cout << "Sorting source and target hash arrays " << std::endl;

  /* Wrap Pointers */
  thrust::device_ptr<uint64_t> d_ptr_source_hash_A(d_temp_EdgeHashSources[0]);
  thrust::device_ptr<uint64_t> d_ptr_source_hash_B(d_temp_EdgeHashSources[1]);
  thrust::device_ptr<uint64_t> d_ptr_target_hash_A(d_temp_EdgeHashTargets[0]);
  thrust::device_ptr<uint64_t> d_ptr_target_hash_B(d_temp_EdgeHashTargets[1]);

  /* Sort source hashes */
  thrust::sort(d_ptr_source_hash_A, d_ptr_source_hash_A + num_edges);
  thrust::sort(d_ptr_source_hash_B, d_ptr_source_hash_B + num_edges);

  /* Sort target hashes */
  thrust::sort(d_ptr_target_hash_A, d_ptr_target_hash_A + num_edges);
  thrust::sort(d_ptr_target_hash_B, d_ptr_target_hash_B + num_edges);

  cudaDeviceSynchronize();
  std::cout << "Sorting complete" << std::endl;

  LogEdgeRefinementStats(num_edges, d_temp_EdgeHashSources[0], "EdgeBins-Source");
  LogEdgeRefinementStats(num_edges, d_temp_EdgeHashTargets[0], "EdgeBins-Target");
  /* End B2] Sort the hashes for each graph */
  /*-----------------------------------------------------------------------*/


  /*-----------------------------------------------------------------------*/
  /* C] Compare Hashes - if they don't match it is not isomorphic */
  bool source_hashes_match = false;
  bool target_hashes_match = false;

  source_hashes_match = thrust::equal(d_ptr_source_hash_A, d_ptr_source_hash_A + num_edges, d_ptr_source_hash_B);

  if (!source_hashes_match)
  {
    std::cout << "Result: NOT Isomorphic: Edge Source neighborhood hashes differ " << std::endl;
  }
  else
  {
    std::cout << "Source neighborhood hashes match " << std::endl;

    target_hashes_match = thrust::equal(d_ptr_target_hash_A, d_ptr_target_hash_A + num_edges, d_ptr_target_hash_B);

    if (!target_hashes_match)
    {
      std::cout << "Result: NOT Isomorphic: Edge Target neighborhood hashes differ " << std::endl;
    }
    else
    {
      std::cout << "Target neighborhood hashes match" << std::endl;
      isPossibleIsomorphic = true;
    }
  }
  /* End C] Compare Hashes */
  /*-----------------------------------------------------------------------*/


  /*-----------------------------------------------------------------------*/
  /* Cleanup */
  for (int gIndex = 0; gIndex < 2; gIndex++)
  {
    cudaFree(d_temp_EdgeHashSources[gIndex]);
    cudaFree(d_temp_EdgeHashTargets[gIndex]);
  }

  std::cout << "------------------------------------------------------------------------" << std::endl;
  std::cout << "" << std::endl;
  std::cout << "------------------------------------------------------------------------" << std::endl;

  return isPossibleIsomorphic;
}
/*===================================================================================================================*/

void LogRefinementStats_FromHistogram(int N, uint* d_Counts, int numBins, const char* stageLabel)
{
    // 1. Download only the counts (tiny transfer, e.g., 50 integers)
    std::vector<uint> h_Counts(numBins);
    cudaMemcpy(h_Counts.data(), d_Counts, numBins * sizeof(uint), cudaMemcpyDeviceToHost);

    // 2. Compute Stats directly from counts
    int solvedNodes = 0;
    int maxBinSize = 0;

    for (int count : h_Counts) {
        if (count == 1) solvedNodes++;
        if (count > maxBinSize) maxBinSize = count;
    }

    int ambiguousNodes = N - solvedNodes;
    float percentDone = (N > 0) ? ((float)solvedNodes / N * 100.0f) : 0.0f;

    // 3. Print
    std::cout << "------------------------------------------------------------------------" << std::endl;
    std::cout << " " << stageLabel << " Refinement Quality (Histogram): " << std::endl;
    std::cout << "   - Total Bins:      " << numBins << std::endl;
    std::cout << "   - Solved Nodes:    " << solvedNodes << " (" << std::fixed << std::setprecision(0) << percentDone << "% Done)" << std::endl;
    std::cout << "   - Ambiguous Nodes: " << ambiguousNodes << " Needs Backtracking" << std::endl;
    std::cout << "   - Max Bin Size:    " << maxBinSize << "  Worst-Case before pruning" << std::endl;
    std::cout << "------------------------------------------------------------------------" << std::endl;
}


/*-----------------------------------------------------------------------------------------------*/
/* Helper: Analyzes node colors to report solver progress                                        */
/* Usage: Call after any coloring kernel where a full histogram is not yet available             */
/*-----------------------------------------------------------------------------------------------*/
void LogRefinementStats(int N, uint64_t* d_Nodes, const char* stageLabel)
{
    /* 1. Download colors to Host (Fast for N <= 10,000) */
    std::vector<uint64_t> h_Nodes(N);
    cudaMemcpy(h_Nodes.data(), d_Nodes, N * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    /* 2. Sort to compute histogram canonically */
    std::sort(h_Nodes.begin(), h_Nodes.end());

    /* 3. Compute Stats */
    int totalBins = 0;
    int solvedNodes = 0;
    int maxBinSize = 0;
    int currentBinSize = 0;

    if (N > 0)
    {
        currentBinSize = 1;
        totalBins = 1;
        for (int i = 1; i < N; i++)
        {
            if (h_Nodes[i] == h_Nodes[i-1])
            {
                currentBinSize++;
            }
            else
            {
                if (currentBinSize == 1) solvedNodes++;
                if (currentBinSize > maxBinSize) maxBinSize = currentBinSize;
                currentBinSize = 1;
                totalBins++;
            }
        }
        /* Handle final bin after loop exit */
        if (currentBinSize == 1) solvedNodes++;
        if (currentBinSize > maxBinSize) maxBinSize = currentBinSize;
    }

    int ambiguousNodes = N - solvedNodes;
    float percentDone = (N > 0) ? ((float)solvedNodes / (float)N * 100.0f) : 0.0f;

    /* 4. Print Formatted Report matching requested style */
    std::cout << "------------------------------------------------------------------------" << std::endl;
    std::cout << " " << stageLabel << " Refinement Quality: " << std::endl;
    std::cout << "   - Total Bins:      " << totalBins << std::endl;
    std::cout << "   - Solved Nodes:    " << solvedNodes << " (" << std::fixed << std::setprecision(0) << percentDone << "% Done)" << std::endl;
    std::cout << "   - Ambiguous Nodes: " << ambiguousNodes << " Needs Backtracking" << std::endl;
    std::cout << "   - Max Bin Size:    " << maxBinSize << "  Worst-Case before pruning" << std::endl;
    std::cout << "------------------------------------------------------------------------" << std::endl;
}
/*-----------------------------------------------------------------------------------------------*/


/*---------------------------------------------------------------------------------------------------------------------*/
/* C] WL1 Stability Check: Computes Histogram and compares Bin Counts only                                             */
/*---------------------------------------------------------------------------------------------------------------------*/
bool WL1CheckStability(  size_t    num_elements,       /* Total size */
                         uint64_t *d_new_colors,       /* Input current color hashes */

                         uint64_t *d_temp_sort_buffer, /* Scratch: Temp buffer to sort into */

                         /* Histogram Outputs: Graph Signature */
                         uint64_t *d_histo_ColorHashKeys,       /* Unique Hash Values found */
                         uint     *d_histo_counts,     /* Output: How many times each Hash appeared */
                         int      &h_num_bins_curr,    /* Output: Total number of unique groups found this round */

                         int       h_num_bins_prev     /* The bin count from the Previous loop */ )
{
    /* 1] Snapshot for Sorting */
    cudaMemcpy(d_temp_sort_buffer, d_new_colors, num_elements * sizeof(uint64_t), cudaMemcpyDeviceToDevice);

    thrust::device_ptr<uint64_t> d_ptr_sort_buffer         (d_temp_sort_buffer);
    thrust::device_ptr<uint64_t> d_ptr_histo_ColorHashKeys (d_histo_ColorHashKeys);
    thrust::device_ptr<uint>     d_ptr_histo_counts        (d_histo_counts);

    /* 2] Group Identical Hashes */
    try
    {
        thrust::sort(thrust::device, d_ptr_sort_buffer, d_ptr_sort_buffer + num_elements);
    }
    catch (thrust::system_error &e)
    {
        std::cerr << "Thrust Sort Error: " << e.what() << std::endl;
        return false;
    }

    /* 3] Generate Histogram */
    auto new_end = thrust::reduce_by_key(thrust::device,
                                         d_ptr_sort_buffer,                  /* Input Keys */
                                         d_ptr_sort_buffer + num_elements,   /* Input End */
                                         thrust::make_constant_iterator(1u), /* Input Values */
                                         d_ptr_histo_ColorHashKeys,          /* Output Unique Keys */
                                         d_ptr_histo_counts);                /* Output Counts */

    h_num_bins_curr = new_end.first - d_ptr_histo_ColorHashKeys;

    /* 4] Check for Convergence (Bin Count Only) */
    if (h_num_bins_prev == -1 || h_num_bins_curr == 0) return false;

    /* If partition refined (more bins), it's not stable */
    if (h_num_bins_curr > h_num_bins_prev) return false;

    /* If bin count is steady, we return true */
    if (h_num_bins_curr == h_num_bins_prev) return true;

    return false;
}
/*-------------------------------------------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------------------------------------------*/
/* C1] WL-1 Test: Iterative Hash Refinement returns a histogram of node hashes
 * Assumes SignatureCounts was called that created this as input: dram_NodeColorHashes[gIndex]
 *
 * Computational Cost: O(Iterations * (E + N log N))
 * Memory Cost: O(N) for node colors, O(E) for edge colors */
/*-------------------------------------------------------------------------------------------------------------------*/
bool GPU_WL1GraphColorHashIT( int gIndex, int MAX_ITERATIONS )
{
    int nodeSizeN = m_numNodes[gIndex];
    int edgeSizeN = m_numEdges[gIndex];

    /* 0] Allocations  */
    uint64_t* d_node_Colors_Init;
    uint64_t* d_edge_Colors_Init;
    uint64_t* d_node_Colors;
    uint64_t* d_edge_Colors;
    uint64_t *d_NodeHisto_keys;
    uint     *d_NodeHisto_counts;
    uint64_t *d_temp_sort_buffer;

    cudaMalloc((void**)&d_node_Colors_Init,  nodeSizeN*sizeof(uint64_t));
    cudaMalloc((void**)&d_edge_Colors_Init,  edgeSizeN*sizeof(uint64_t));
    cudaMalloc((void**)&d_node_Colors,       nodeSizeN*sizeof(uint64_t));
    cudaMalloc((void**)&d_edge_Colors,       edgeSizeN*sizeof(uint64_t));
    cudaMalloc((void**)&d_NodeHisto_keys,    nodeSizeN*sizeof(uint64_t));
    cudaMalloc((void**)&d_NodeHisto_counts,  nodeSizeN*sizeof(uint64_t));
    cudaMalloc((void**)&d_temp_sort_buffer,  nodeSizeN*sizeof(uint64_t));
    cudaDeviceSynchronize();

    std::cout<<"------------------------------------------------------------------------"<<std::endl;
    std::cout<<"GPU WL1 Class 1st NN Structure Test Graph " << gIndex  <<std::endl;
    std::cout<<"------------------------------------------------------------------------"<<std::endl;

    /* Initialize Nodes and Edges */
    Kernel_InitNodeHashWL1<<<ThreadsAllNodes[0].dimGrid, ThreadsAllNodes[0].dimBlock>>>( nodeSizeN, dram_NodeColorHashes[gIndex], d_node_Colors_Init);
    if (edgeSizeN > 0)
    {
        Kernel_InitEdgeHashWL1<<<ThreadsAllEdges[0].dimGrid, ThreadsAllEdges[0].dimBlock>>>(edgeSizeN, dram_Edge_labelDBIndex[gIndex], d_edge_Colors_Init );
    }
    cudaDeviceSynchronize();

    /* Copy Init to Current */
    cudaMemcpy(d_node_Colors, d_node_Colors_Init, nodeSizeN * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_edge_Colors, d_edge_Colors_Init, edgeSizeN * sizeof(uint64_t), cudaMemcpyDeviceToDevice);

    int iteration       = 0;
    bool is_stable      = false;
    int h_num_bins      = -1;
    int h_num_bins_Prev = -1;

    /* NEW: Streak Counter to prevent premature stopping on symmetric graphs */
    int stable_streak   = 0;
    const int FalgStableIT = 2;

    /* Initial Histogram (Iteration 0) */
    WL1CheckStability(nodeSizeN, d_node_Colors, d_temp_sort_buffer, d_NodeHisto_keys, d_NodeHisto_counts, h_num_bins, h_num_bins_Prev);
    h_num_bins_Prev = h_num_bins;

    /*===================================================================================================================*/
    /* Color Iteration Loop */
    while (!is_stable && iteration < MAX_ITERATIONS)
    {
      /* 1] Update Edge Hashes */
      if (edgeSizeN > 0)
      {
          Kernel_EdgeColorsWL1_Hypergraph<<<ThreadsAllEdges[0].dimGrid, ThreadsAllEdges[0].dimBlock>>>(
              edgeSizeN, d_edge_Colors_Init,
              dram_Edge_nodeSources[gIndex], dram_Edge_nodeSourcesStart[gIndex], dram_Edge_nodeSourcesNum[gIndex],
              d_node_Colors,
              dram_Edge_nodeTargets[gIndex], dram_Edge_nodeTargetsStart[gIndex], dram_Edge_nodeTargetsNum[gIndex],
              d_edge_Colors
          );
          cudaDeviceSynchronize();
      }

      /* 2] Update Node Colors */
      Kernel_NodeHashWL1<<<ThreadsAllNodes[0].dimGrid, ThreadsAllNodes[0].dimBlock>>>(
          nodeSizeN,
          d_node_Colors_Init,
          dram_Node_edgePrevs[gIndex], dram_Node_edgePrevsStart[gIndex], dram_Node_edgePrevsNum[gIndex],
          edgeSizeN,
          dram_Node_edgeNexts[gIndex], dram_Node_edgeNextsStart[gIndex], dram_Node_edgeNextsNum[gIndex],
          d_edge_Colors,
          d_node_Colors );
      cudaDeviceSynchronize();

      /* 3. Check Stability via Bin Counts */
      bool bins_constant = WL1CheckStability( nodeSizeN, d_node_Colors, d_temp_sort_buffer,
                                              d_NodeHisto_keys, d_NodeHisto_counts, h_num_bins,
                                              h_num_bins_Prev);

      if (bins_constant)
      {
          /* If bin count matched previous step, increment streak */
          stable_streak++;
          if (stable_streak >= FalgStableIT)
          {
              is_stable = true;
          }
      }
      else
      {
          /* Bin count changed or first run */
          stable_streak = 0;
          h_num_bins_Prev = h_num_bins;
      }

      iteration++;
    }
    /* End Loop */
    /*===================================================================================================================*/

    if (is_stable)
    {
        std::cout << "WL-1 Stabilized Partition after " << iteration << " iterations" << std::endl;
        m_WL_BinCount[gIndex] = (uint)h_num_bins_Prev;

        if (h_num_bins_Prev > 0)
        {
            size_t keys_bytes   = h_num_bins_Prev * sizeof(uint64_t);
            size_t counts_bytes = h_num_bins_Prev * sizeof(uint);

            cudaMalloc((void**)&dram_WL_BinsColorKeys[gIndex],  keys_bytes);
            cudaMalloc((void**)&dram_WL_BinsNumCount[gIndex], counts_bytes);

            cudaMemcpy( dram_WL_BinsColorKeys[gIndex],  d_NodeHisto_keys,  keys_bytes,   cudaMemcpyDeviceToDevice );
            cudaMemcpy( dram_WL_BinsNumCount[gIndex],  d_NodeHisto_counts, counts_bytes, cudaMemcpyDeviceToDevice );

            LogRefinementStats_FromHistogram( m_numNodes[gIndex], dram_WL_BinsNumCount[gIndex], h_num_bins_Prev, "WL-1Final");
        }
    }
    else
    {
        std::cout << "WL-1 FAILED TO STABILIZE after " << MAX_ITERATIONS << " iterations" << std::endl;
        m_WL_BinCount[gIndex] = 0;
        return false;
    }

    /* Cleanup */
    cudaFree(d_node_Colors);       cudaFree(d_edge_Colors);
    cudaFree(d_node_Colors_Init);  cudaFree(d_edge_Colors_Init);
    cudaFree(d_temp_sort_buffer);  cudaFree(d_NodeHisto_keys);
    cudaFree(d_NodeHisto_counts);

    return true;
}
/*-------------------------------------------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------------------------------------------*/
/* D] Possible Isomorphism Check for Single Iteration
 * Compares the stable WL-1/2 histograms between two graphs
 * Returns TRUE if graphs are structurally identical: Possible Isomorphic
 * * Computational Cost: O(K) where K is number of unique bins (K <= N)
 * - Thrust equality checks are linear scan O(K)
 * - Memory transfer for reporting is O(K)
 * Memory Cost: O(1) persistent
 * - Operates on pre-allocated dram_WL_Bins pointers */
/*-------------------------------------------------------------------------------------------------------------------*/
bool WL_CompareBinCountsInitState()
{
	std::cout<<"------------------------------------------------------------------------"<<std::endl;
	std::cout<<"GPU Structure Test Between Graphs "<<std::endl;
	std::cout<<"------------------------------------------------------------------------"<<std::endl;

    /* Fail Fast: Bin Count Mismatch */
    if (m_WL_BinCount[0] != m_WL_BinCount[1])
    {
        std::cout << "Mismatch: Graphs have different complexity Bins: "
                  << m_WL_BinCount[0] << " vs " <<m_WL_BinCount[1] << "" << std::endl;
        GPU_FreeWLBins ();
        return false;
    }

   /* Wrap thrust pointers */
    thrust::device_ptr<uint64_t> th_keys_A(dram_WL_BinsColorKeys[0]);
    thrust::device_ptr<uint>     th_counts_A(dram_WL_BinsNumCount[0]);

    thrust::device_ptr<uint64_t> th_keys_B(dram_WL_BinsColorKeys[1]);
    thrust::device_ptr<uint>     th_counts_B(dram_WL_BinsNumCount[1]);


    /* Compare Keys (Structure Types): Do both graphs contain the same connections? */
    bool keys_match = thrust::equal(thrust::device,
                                    th_keys_A,
                                    th_keys_A + m_WL_BinCount[0],
                                    th_keys_B);

    if (!keys_match)
    {
        std::cout << "Mismatch: Graphs contain different structural shapes- Keys differ !" << std::endl;
        GPU_FreeWLBins ();
        return false;
    }

    /* Compare Counts */
    bool counts_match = thrust::equal(thrust::device,
                                      th_counts_A,
                                      th_counts_A + m_WL_BinCount[0],
                                      th_counts_B);

    if (!counts_match)
    {
        std::cout << "Mismatch: Structures appear with different frequencies - Counts differ !" << std::endl;
        GPU_FreeWLBins ();
        return false;
    }

	GPU_FreeWLBins ();

	std::cout<<"------------------------------------------------------------------------"<<std::endl;
	std::cout<<" Match: Graphs have identical structural histograms"<<std::endl;
	std::cout<<"------------------------------------------------------------------------"<<std::endl;

	return true; /* Both are the same possible isomorphic */
}
/*-------------------------------------------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------------------------------------------*/
/* D1] Compare Histograms of Two Graphs for the Iterative Solve
 *
 * Computational Cost: O(N log N)
 * - Sorting two arrays of size N dominates the cost
 * - Equality check is O(N)
 * Memory Cost: O(0)
 * - Uses pre-allocated scratch buffer passed from parent (No dynamic allocation) */
/*-------------------------------------------------------------------------------------------------------------------*/
bool WL2_CompareBinCountsIR(int numNodes, uint64_t* d_Colors_G0, uint64_t* d_Colors_G1, uint64_t* d_Scratch_Buffer)
{
    /* Use the pre-allocated scratch buffer,  We need 2 * numNodes space
       We assume d_Scratch_Buffer is at least 2*N * sizeof(uint64_t) */
    uint64_t* d_diag0 = d_Scratch_Buffer;
    uint64_t* d_diag1 = d_Scratch_Buffer + numNodes;

    /* Extract diagonals */
    Kernel_WL2_ExtractDiagonals<<<ThreadsAllNodes[0].dimGrid, ThreadsAllNodes[0].dimBlock>>>(
        numNodes, d_Colors_G0, d_diag0);
    cudaDeviceSynchronize();
	cudaCheckError();

    Kernel_WL2_ExtractDiagonals<<<ThreadsAllNodes[0].dimGrid, ThreadsAllNodes[0].dimBlock>>>(
        numNodes, d_Colors_G1, d_diag1);
    cudaDeviceSynchronize();
	cudaCheckError();

    /* Wrap raw pointers for Thrust */
    thrust::device_ptr<uint64_t> t_diag0(d_diag0);
    thrust::device_ptr<uint64_t> t_diag1(d_diag1);

    /* Sort to create canonical multisets */
    thrust::sort(t_diag0, t_diag0 + numNodes);
    thrust::sort(t_diag1, t_diag1 + numNodes);

    cudaDeviceSynchronize();
	cudaCheckError();

    /*Compare the sorted histograms */
    bool isEqual = thrust::equal(t_diag0, t_diag0 + numNodes, t_diag1);

	/* Note: IR loop does not persist keys/counts to global DRAM to save bandwidth,
	   but we log the stats here for depth analysis. */
	if (isEqual)
	{
		/* Calculate number of unique bins (transitions + 1) */
		/* Since t_diag0 is sorted, we just count where t[i] != t[i+1] */
		int numBins = thrust::inner_product(
			t_diag0, t_diag0 + numNodes - 1,  // Range 1: [0, N-1]
			t_diag0 + 1,                      // Range 2: [1, N]
			1,                                // Initial value (first bin)
			thrust::plus<int>(),              // Accumulator
			thrust::not_equal_to<uint64_t>()  // Comparator (1 if different, 0 if same)
		);
		std::cout << "IR Match! Bins: " << numBins << " / " << numNodes << std::endl;
	}

        return isEqual;
}
/*-------------------------------------------------------------------------------------------------------------------*/


/*-----------------------------------------------------------------------------------------------*/
/* Helper: Detects and logs symmetry classes to prune redundant IR branches                      */
/* Returns: Number of distinct structural patterns found                                         */
/*-----------------------------------------------------------------------------------------------*/
int LogSymmetryPruningStats(int N, uint64_t* d_NodeProfiles, const char* stageLabel)
{
	/* 1. Download node profiles to Host */
	    std::vector<uint64_t> h_Profiles(N);
	    cudaMemcpy(h_Profiles.data(), d_NodeProfiles, N * sizeof(uint64_t), cudaMemcpyDeviceToHost);

	    /* 2. Pair profiles with node indices to preserve identity after sorting */
	    std::vector<std::pair<uint64_t, int>> h_NodePatterns(N);
	    for(int i = 0; i < N; i++)
	    {
	        h_NodePatterns[i] = {h_Profiles[i], i};
	    }

	    /* 3. Sort to group structurally equivalent nodes */
	    std::sort(h_NodePatterns.begin(), h_NodePatterns.end());

	    /* 4. Analyze Classes and Print Preview */
	    std::cout << "------------------------------------------------------------------------" << std::endl;
	    std::cout << " " << stageLabel << " Symmetry Analysis: " << std::endl;

	    int distinctPatterns = 0;
	    int maxClassSize = 0;
	    const int MAX_PREVIEW = 16; /* Print at most 16 elements */

	    if (N > 0)
	    {
	        int currentClassSize = 1;
	        distinctPatterns = 1;
	        int printedCount = 0;
	        bool isPrinting = true;

	        std::cout << "   - Pattern Map: [";

	        /* Print first element */
	        std::cout << "{" << h_NodePatterns[0].second;
	        printedCount++;

	        for (int i = 1; i < N; i++)
	        {
	            bool sameClass = (h_NodePatterns[i].first == h_NodePatterns[i-1].first);

	            /* A] Update Statistics */
	            if (sameClass)
	            {
	                currentClassSize++;
	            }
	            else
	            {
	                if (currentClassSize > maxClassSize) maxClassSize = currentClassSize;
	                currentClassSize = 1;
	                distinctPatterns++;
	            }

	            /* B] Update Print Output (Capped at MAX_PREVIEW) */
	            if (isPrinting)
	            {
	                if (printedCount >= MAX_PREVIEW)
	                {
	                    std::cout << "...";
	                    isPrinting = false; /* Stop printing, but continue stats loop */
	                }
	                else
	                {
	                    if (sameClass)
	                    {
	                        std::cout << "," << h_NodePatterns[i].second;
	                    }
	                    else
	                    {
	                        std::cout << "} {" << h_NodePatterns[i].second;
	                    }
	                    printedCount++;
	                }
	            }
	        }

	        /* Close the pattern map bracket */
	        if (isPrinting) std::cout << "}]" << std::endl;
	        else std::cout << "}]" << std::endl;

	        /* Finalize max class size */
	        if (currentClassSize > maxClassSize) maxClassSize = currentClassSize;
	    }

	    /* 5. Report Summary stats */
	    std::cout << "   - Symmetry Classes Found:  " << distinctPatterns << " distinct patterns" << std::endl;

	    if (distinctPatterns == N)
	    {
	        std::cout << "   - Symmetry Status:         No symmetries detected (All nodes unique)" << std::endl;
	    }
	    else
	    {
	        int redundantNodes = N - distinctPatterns;
	        std::cout << "   - Symmetry Status:         Symmetry Detected (" << redundantNodes << " nodes prunable)" << std::endl;
	        std::cout << "   - Largest Equivalence:     Class Size " << maxClassSize << std::endl;
	    }
	    std::cout << "------------------------------------------------------------------------" << std::endl;

	    return distinctPatterns;
}

/*-------------------------------------------------------------------------------------------------------------------*/
/** E1] Check Node Profile Equivalence - Fast Row Comparison !! Core Method @MM this should be what you need also or L2  ?
 *
 * Checks if two nodes have the same structural profile by comparing their WL-2 matrix rows
 * Used for fast pruning in IR search and for detecting symmetry classes
 *
 * Computational Cost: O(N) average, O(N log N) worst case
 *   - Stage 1 (hash filter): O(N) - rejects 99% of non-matches
 *   - Stage 2 (exact verify): O(N log N) - confirms potential matches
 *
 * Memory Cost: O(N)
 *   - Uses pre-allocated workspace (2*N elements)
 *
 * Parameters:
 *   - numNodes: Number of nodes in graph
 *   - dram_Matrix_G0: WL-2 matrix for graph 0
 *   - nodeIndexG0: First node to compare
 *   - dram_Matrix_G1: WL-2 matrix for graph 1
 *   - nodeIndexG1: Second node to compare
 *   - d_temp_workspace: GPU workspace (must be at least 2*N*sizeof(uint64_t))
 *
 * Returns:
 *   - true if nodes have identical structural profiles (rows match)
 *   - false if profiles differ
 *   - No false negatives (never rejects matching profiles) */
/*-------------------------------------------------------------------------------------------------------------------*/
bool WL2_CheckNodeProfileEquivalence(
    int numNodes,
    uint64_t *dram_Matrix_G0, int nodeIndexG0,
    uint64_t *dram_Matrix_G1, int nodeIndexG1,
    uint64_t *d_temp_workspace)
{
	/*--------------------------------------------------*/
    /** Fast hash filter: Hash match does NOT guarantee equality */
    thrust::device_ptr<uint64_t> row_G0_ptr(dram_Matrix_G0 + nodeIndexG0 * numNodes);
    thrust::device_ptr<uint64_t> row_G1_ptr(dram_Matrix_G1 + nodeIndexG1 * numNodes);

    uint64_t hash_G0 = thrust::reduce(row_G0_ptr, row_G0_ptr + numNodes, (uint64_t)0, thrust::plus<uint64_t>());
    uint64_t hash_G1 = thrust::reduce(row_G1_ptr, row_G1_ptr + numNodes, (uint64_t)0, thrust::plus<uint64_t>());

    /** Fast rejection if hashes differ (no collision risk - definitely different) */
    if (hash_G0 != hash_G1)
    {
        return false;
    }
    /*--------------------------------------------------*/

    /*--------------------------------------------------*/
    /** More expensive check: Exact verification, hashes matched, but could be collision */
    /** Extract and sort rows for exact comparison */
    uint64_t *d_row_G0 = d_temp_workspace;
    uint64_t *d_row_G1 = d_temp_workspace + numNodes;

    cudaMemcpy(d_row_G0, dram_Matrix_G0 + nodeIndexG0 * numNodes,
               numNodes * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_row_G1, dram_Matrix_G1 + nodeIndexG1 * numNodes,
               numNodes * sizeof(uint64_t), cudaMemcpyDeviceToDevice);

    thrust::device_ptr<uint64_t> sorted_G0_ptr(d_row_G0);
    thrust::device_ptr<uint64_t> sorted_G1_ptr(d_row_G1);

    thrust::sort(sorted_G0_ptr, sorted_G0_ptr + numNodes);
    thrust::sort(sorted_G1_ptr, sorted_G1_ptr + numNodes);
    /** Exact comparison guaranteed no false negatives */
    return thrust::equal(sorted_G0_ptr, sorted_G0_ptr + numNodes, sorted_G1_ptr);
    /*--------------------------------------------------*/
}
/*-------------------------------------------------------------------------------------------------------------------*/




/*-------------------------------------------------------------------------------------------------------------------*/
/** E2] Checks for automorphism in a bin (Symmetry Class Detection)
 *
 * Identifies nodes with identical structural neighborhoods relative to the current stable coloring
 * - Uses exact sorted row comparison (Canonical Certificate)
 * - No hash collisions; mathematically sound for formal verification
 *
 * Context:
 * This function operates on the accumulated Matrix state
 * - If called after Phase 3, it detects WL-2 symmetries
 * - If called after Phase 3.5, it detects WL-3 symmetries (triplet-aware
 *
 * Theory:
 * A "Symmetry Class" is a set of nodes indistinguishable by the current refinement
 * Since the matrix row represents a node's full view of the graph:
 * Signature(u) = Sort( { Matrix[u][v] | for all v in V } )
 *
 * If Signature(u) == Signature(v) (Exact Vector Comparison), then u and v are
 * automorphically equivalent relative to the solver's current depth (WL-2 or WL-3)
 *
 * Search Reduction:
 * - Nodes in the same class are interchangeable
 * - We only need to test One representative per class in the backtracking search
 * - Prunes exponential branches
 *
 * Computational Cost: O(N^2 log N)
 * - Sorting N rows of size N: O(N^2 log N)
 * - Exact Vector Comparison: O(N^2)
 *
 * Memory Cost: O(N^2)
 * - Requires a full copy of the matrix for sorting
 *
 * Parms:
 * - numNodes: Number of nodes in graph
 * - d_Matrix: Current Stable Matrix (WL-2 or WL-3 refined) on GPU
 *
 * Return:
 * - Vector of representatives (one per symmetry class) */
/*-------------------------------------------------------------------------------------------------------------------*/
std::vector<int> WL2_DetectSymmetryClasses(int numNodes, uint64_t* d_InputMatrixColors)
{
   // std::cout << "Start Symmetry Classes Finding structurally equivalent nodes" << std::endl;
    bool enableDebug = true;
    /** 1] Allocate workspace for sorted rows */
    uint64_t *d_SortedMatrix;
    cudaMalloc(&d_SortedMatrix, numNodes * numNodes * sizeof(uint64_t));

    /** 2] Copy matrix and sort each row to create canonical form */
    cudaMemcpy(d_SortedMatrix, d_InputMatrixColors, numNodes * numNodes * sizeof(uint64_t), cudaMemcpyDeviceToDevice);

    /** Sort each row individually */
    for (int row = 0; row < numNodes; row++)
    {
        thrust::device_ptr<uint64_t> row_ptr(d_SortedMatrix + row * numNodes);
        thrust::sort(row_ptr, row_ptr + numNodes);
    }
    cudaDeviceSynchronize();

    /** 3] Copy sorted rows to host for grouping */
    std::vector<uint64_t> h_SortedMatrix(numNodes * numNodes);
    cudaMemcpy(h_SortedMatrix.data(), d_SortedMatrix, numNodes * numNodes * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_SortedMatrix);

    /** 4] Group nodes by their sorted row signature (exact comparison) */
    /** Use vector<uint64_t> as key (represents sorted row) - no hash collisions */
    std::map<std::vector<uint64_t>, std::vector<int>> symmetry_classes;

    /* Optional: Storage for debug hashes */

        std::vector<uint64_t> h_ProfileHashes;
        if (enableDebug) h_ProfileHashes.resize(numNodes);

    for (int i = 0; i < numNodes; i++)
    {
        /** Extract sorted row for node i */
        std::vector<uint64_t> sorted_row(
            h_SortedMatrix.begin() + i * numNodes,
            h_SortedMatrix.begin() + (i + 1) * numNodes
        );

        /* Debug: Generate profile hashes only if requested */
			if (enableDebug)
			{
				uint64_t row_hash = 0xCBF29CE484222325ULL;
				for (uint64_t val : sorted_row)
				{
					row_hash ^= val;
					row_hash *= 0x100000001B3ULL;
				}
				h_ProfileHashes[i] = row_hash;
			}

        /** Group by exact sorted row - nodes with identical rows are in same class */
        symmetry_classes[sorted_row].push_back(i);
    }

    /** 5] Select one representative per symmetry class */
    std::vector<int> representatives;
    int total_classes_with_symmetry = 0;
    int total_symmetric_nodes = 0;

   // std::cout << "Symmetry Found " << symmetry_classes.size() << " distinct structural patterns" << std::endl;

    for (auto& [sorted_row, nodes] : symmetry_classes)
    {
        representatives.push_back(nodes[0]);  /** First node represents the class/bin */

        if (nodes.size() > 1)
        {
            total_classes_with_symmetry++;
            total_symmetric_nodes += nodes.size();

            //std::cout << "Symmetry Class " << nodes.size() << " equivalent nodes: {";
            for (size_t i = 0; i < std::min((size_t)5, nodes.size()); i++)
            {
                std::cout << nodes[i];
                if (i < std::min((size_t)5, nodes.size()) - 1) std::cout << ", ";
            }
            if (nodes.size() > 5) std::cout << ", ...";
            //std::cout << "} -> representative: node " << nodes[0] << std::endl;
        }
    }

    /* 6] Encased Debug: Log statistics and symmetry patterns */
        if (enableDebug)
        {
            uint64_t *d_ProfileHashes;
            cudaMalloc(&d_ProfileHashes, numNodes * sizeof(uint64_t));
            cudaMemcpy(d_ProfileHashes, h_ProfileHashes.data(), numNodes * sizeof(uint64_t), cudaMemcpyHostToDevice);

            /* This function already prints:
               - "Symmetry Classes Found"
               - "Symmetry Status" (including "No symmetries detected")
               - "Pattern Map" (if enabled)
            */
            int distinctPatterns = LogSymmetryPruningStats(numNodes, d_ProfileHashes, "IR Symmetry Detection");

            /* ONLY print the extra math if symmetries exist.
               We don't need an 'else' block because LogSymmetryPruningStats handles the failure case. */
            if (total_classes_with_symmetry > 0)
            {
                /* Calculate reduction only for the detailed trace */
                double reduction_factor = (double)numNodes / representatives.size();

                std::cout << "   - Reduction Math:          Reduced from " << numNodes
                          << " to " << representatives.size() << " reps ("
                          << std::fixed << std::setprecision(1) << reduction_factor << "x)" << std::endl;
            }

            cudaFree(d_ProfileHashes);
        }

    return representatives;
}
/*-------------------------------------------------------------------------------------------------------------------*/


/*-------------------------------------------------------------------------------------------------------------------*/
/** F] Pivot Selection with Symmetry Pruning
 *
 * Selects pivot candidates using the smallest bin with symmetry pruning.
 * This is the core of the individualization-refinement search strategy.
 *
 * Strategy:
 *   1. Find color class with minimum size > 1
 *   2. Extract all nodes from that class
 *   3. Detect symmetry classes among candidates
 *   4. Return only symmetry class representatives
 *
 * Theory:
 *   - Smaller cells provide stronger individualization
 *   - Binary branching (size 2) is theoretically optimal
 *   - Symmetry pruning: Nodes in same class are interchangeable
 *   - Combined: Exponential to polynomial search reduction
 *
 * Computational Cost: O(N log N) + O(N^2 log N) with symmetry detection
 *   - Diagonal extraction: O(N)
 *   - Sort by color: O(N log N)
 *   - Reduce_by_key: O(N)
 *   - SCT selection: O(k) where k = number of color classes
 *   - Candidate extraction: O(bin_size)
 *   - Symmetry detection: O(N^2 log N) - called once per depth level
 *   - Symmetry filtering: O(bin_size × num_classes)
 *   - Total: O(N^2 log N) dominated by symmetry detection
 *
 * Memory Cost: O(N^2)
 *   - Thrust temporary vectors: O(N)
 *   - Symmetry detection workspace: O(N^2)
 *   - Candidate lists: O(bin_size)
 *
 * Parameters:
 *   - numNodes: Number of nodes in graph
 *   - d_Matrix: WL-2 matrix on GPU (N×N)
 *   - d_Scratch_Diag: Scratch buffer for diagonal extraction (2*N elements)
 *   - d_IOTags: I/O tags for nodes (not currently used in selection)
 *
 * Returns:
 *   - BinInfoGPU struct containing:
 *     - isTargetColorFound: 0 if discrete, 1 if non-discrete
 *     - pivotNode: First candidate from smallest bin (or -1 if discrete)
 *     - candidates: Symmetry-pruned candidate list */
/*-------------------------------------------------------------------------------------------------------------------*/
BinInfoGPU WL2_SelectPivotFromBin( int numNodes,
								   uint64_t *d_Matrix,
								   uint64_t *d_Scratch_Diag,
								   uint     *d_IOTags         )
{
    BinInfoGPU info;
    info.isTargetColorFound = 0;

    /** 1] Extract Matrix Diagonals: node self-colors  */
    Kernel_WL2_ExtractDiagonals<<<ThreadsAllNodes[0].dimGrid, ThreadsAllNodes[0].dimBlock>>>( numNodes, d_Matrix, d_Scratch_Diag );
    cudaDeviceSynchronize();

    /** 2] Prepare Structural Data for Analysis: Create (color, index) pairs for sorting and binning */
    thrust::device_vector<uint64_t> d_colors(d_Scratch_Diag, d_Scratch_Diag + numNodes);
    thrust::device_vector<int> d_indices(numNodes);
    thrust::sequence(d_indices.begin(), d_indices.end());

    /** 3] Sort by Color to Group Same-Colored Nodes */
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(d_colors.begin(), d_indices.begin()));
    auto zip_end   = thrust::make_zip_iterator(thrust::make_tuple(d_colors.end(), d_indices.end()));
    thrust::sort(zip_begin, zip_end);

    /** 4] Generate Color Histogram Compress sorted array into unique colors and their counts */
    thrust::device_vector<uint64_t> d_unique_keys(numNodes);
    thrust::device_vector<int>      d_counts(numNodes);
    auto end_pair = thrust::reduce_by_key(  d_colors.begin(), d_colors.end(),
											thrust::constant_iterator<int>(1),
											d_unique_keys.begin(), d_counts.begin() );

    int num_unique = end_pair.first - d_unique_keys.begin();

    /** 5] Check for Discrete Partition */
    if (num_unique == numNodes)
    {
        /** All nodes have unique colors - partition is discrete */
        return info;
    }

    /** 6] Transfer Histogram to Host for SCT Selection */
    std::vector<uint64_t> h_unique_keys(num_unique);
    std::vector<int> h_counts(num_unique);
    cudaMemcpy(h_unique_keys.data(), thrust::raw_pointer_cast(d_unique_keys.data()),
               num_unique * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_counts.data(), thrust::raw_pointer_cast(d_counts.data()),
               num_unique * sizeof(int), cudaMemcpyDeviceToHost);

    /** 7] Smallest bin Pivot Selection: Find the non-discrete color class with minimum size */
    int best_bin_idx = -1;
    int min_bin_size = numNodes + 1;

    for (int i = 0; i < num_unique; i++)
    {
        if (h_counts[i] > 1)
        {
            if (h_counts[i] < min_bin_size)
            {
                min_bin_size = h_counts[i];
                best_bin_idx = i;
            }
            /** Early exit: Size 2 is theoretically optimal for branching */
            if (min_bin_size == 2) break;
        }
    }

    /** 8] Extract Initial Candidate List */
    if (best_bin_idx == -1)
    {
        /** No non-discrete bins found - shouldn't happen but handle gracefully :-) */
        return info;
    }

    info.isTargetColorFound = 1;
    uint64_t target_color = h_unique_keys[best_bin_idx];

    /** Find where this color starts in the sorted array */
    auto lower = thrust::lower_bound(d_colors.begin(), d_colors.end(), target_color);
    int start_offset = lower - d_colors.begin();

    /** Extract all candidates from this bin */
    std::vector<int> all_candidates(min_bin_size);
    thrust::copy(
        d_indices.begin() + start_offset,
        d_indices.begin() + start_offset + min_bin_size,
        all_candidates.begin()
    );

    /** 9] Detect symmetry classes and reduce candidates to representatives only */
    std::vector<int> symmetry_representatives = WL2_DetectSymmetryClasses(numNodes, d_Matrix);

    /** 9.1] Filter candidates: keep only symmetry class representatives */
    std::vector<int> pruned_candidates;
    for (int candidate : all_candidates)
    {
        if (std::find(symmetry_representatives.begin(), symmetry_representatives.end(), candidate)
            != symmetry_representatives.end())
        {
            pruned_candidates.push_back(candidate);
        }
    }

    /** 9.2] Report pruning statistics */
    if (pruned_candidates.size() < all_candidates.size())
    {
        std::cout << "Symmetry Prunning Candidates reduced: "
                  << all_candidates.size() << " → " << pruned_candidates.size();

        if (all_candidates.size() > 0)
        {
            double prune_percent = 100.0 * (all_candidates.size() - pruned_candidates.size())
                                 / all_candidates.size();
            std::cout << " (" << std::fixed << std::setprecision(1)<< prune_percent << "% pruned)";
        }
        std::cout << std::endl;
    }

    /** 10] Finalize Candidate List */
    if (!pruned_candidates.empty())
    {
        info.candidates = pruned_candidates;
    }
    else
    {
        /** Fallback: If pruning removed everything (shouldn't happen), use full set */
        std::cout << "Warning Symmetry pruning removed all candidates Error Debug NG - using full set" << std::endl;
        info.candidates = all_candidates;
    }

    /** Set first candidate as pivot */
    info.pivotNode = info.candidates[0];

    return info;
}
/*-------------------------------------------------------------------------------------------------------------------*/


/*-------------------------------------------------------------------------------------------------------------------*/
/* G] Canonical Map Transformation:
 * Normalizes high-entropy 64-bit hashes into a dense, ordered integer set [0, K-1]
 * Hash-Drift:
 * By sorting the global population of hashes, we define a Canonical Map "
 * Even if Graph A and Graph B produce different raw hash values for the same
 * logical structure (drift), they will occupy the same relative position (Rank)
 * within their respective sorted populations. This forces isomorphic structures
 * into identical integer states
 *
 * Hash Colission:
 * If two distinct structural features accidentally produce the same hash ("merge"),
 * they are assigned the same rank, effectively becoming a single bin.
 *
 * Solution: The solver treats this as "artificial symmetry" and is resolved by the
 * IR-Backtracking loop and a final Set-Verification Function [L],
 * any invalid mapping caused by a hash merge will eventually trigger a
 * contradiction check, forcing the serach to revert and correct the path
 *
 * Monotonic Order-Preservation:
 * The transformation is monotonic: if Hash(A) < Hash(B), then Rank(A) < Rank(B)
 * This ensures that the relative "distance" and hierarchy between structural
 * features are preserved across iterations, maintaining mathematical stability */
/*-------------------------------------------------------------------------------------------------------------------*/
void WL2_CanonicalRelabelIR(size_t totalElements, uint64_t* d_data, uint64_t* d_map)
{
    if (totalElements == 0) return;

    /* 1] Copy original hashes into map buffer to preserve the original matrix during sort */
    cudaMemcpy(d_map, d_data, totalElements * sizeof(uint64_t), cudaMemcpyDeviceToDevice);

    thrust::device_ptr<uint64_t> t_ranks = thrust::device_pointer_cast(d_map);
    thrust::device_ptr<uint64_t> t_data  = thrust::device_pointer_cast(d_data);

    /* 2] Sort the map to group identical structural signatures */
    thrust::sort(t_ranks, t_ranks + totalElements);

    /* 3] Identify unique structural classes (The Canonical Alphabet) */
    auto new_end = thrust::unique(t_ranks, t_ranks + totalElements);
    size_t num_unique = new_end - t_ranks;

    /* 4] Project original hashes onto their integer rank [0...numUnique-1]
          This transforms d_data from 'hashes' into categorical integers */
    thrust::lower_bound(t_ranks, t_ranks + num_unique,
                        t_data, t_data + totalElements,
                        t_data);
}
/*-------------------------------------------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------------------------------------------*/
/* H] Count unique integers in a sorted/relabeled array                                              */
/*-------------------------------------------------------------------------------------------------------------------*/
int WL2_CountUniqueBinsGPU(size_t totalElements, uint64_t* d_Data, uint64_t* d_Workspace)
{
	if (totalElements == 0) return 0;
	    cudaMemcpy(d_Workspace, d_Data, totalElements * sizeof(uint64_t), cudaMemcpyDeviceToDevice);

	    /* 1] Wrap the workspace pointer for Thrust */
	    thrust::device_ptr<uint64_t> t_ptr = thrust::device_pointer_cast(d_Workspace);

	    /* 2] Sort and Unique in-place on the workspace */
	    thrust::sort(t_ptr, t_ptr + totalElements);
	    auto new_end = thrust::unique(t_ptr, t_ptr + totalElements);

	    /* 3] Return the number of unique structural classes K */
	    return (int)thrust::distance(t_ptr, new_end);
}
/*-------------------------------------------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------------------------------------------*/
/* I] O(N2) due to sort */
/*-------------------------------------------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------------------------------------------*/
/* J] Helper: Initialize WL-2 Matrix Outside Loop                                           */
/* Allocates memory, sets diagonals, injects edges, counts triangles, and does initial relabel    */
/*-------------------------------------------------------------------------------------------------------------------*/
bool GPU_WL2_InitMatrix(int gIndex)
{
    int nodeSizeN = m_numNodes[gIndex];
    int edgeSizeN = m_numEdges[gIndex];
    size_t NodeMatrixSize = (size_t)nodeSizeN * (size_t)nodeSizeN;

    /* 1] Allocate and Clear */
    if (m_isWL2Alloc[gIndex]) cudaFree(dram_WL2_MatrixColors[gIndex]);

    cudaMalloc((void**)&dram_WL2_MatrixColors[gIndex], NodeMatrixSize * sizeof(uint64_t));
    cudaMemset(dram_WL2_MatrixColors[gIndex], 0, NodeMatrixSize * sizeof(uint64_t));
    m_isWL2Alloc[gIndex] = true;

    uint64_t* d_Mat = dram_WL2_MatrixColors[gIndex];

    /* 2] Diagonals and Self-Loops */
    Kernel_WL2_Init_Diagonal<<<ThreadsAllNodes[0].dimGrid, ThreadsAllNodes[0].dimBlock>>>(
        nodeSizeN, dram_Node_IOTag[gIndex], dram_NodeColorHashes[gIndex], d_Mat
    );
    Kernel_WL2_DetectSelfLoops<<<ThreadsAllNodes[0].dimGrid, ThreadsAllNodes[0].dimBlock>>>(
        nodeSizeN, dram_Node_edgePrevs[gIndex], dram_Node_edgePrevsStart[gIndex], dram_Node_edgePrevsNum[gIndex],
        dram_Node_edgeNexts[gIndex], dram_Node_edgeNextsStart[gIndex], dram_Node_edgeNextsNum[gIndex], d_Mat
    );

    /* 3] HyperEdges */
    if (edgeSizeN > 0) {
        int threads = 256;
        int blocks = (edgeSizeN + 7) / 8;
        Kernel_WL2_InitHyperEdges_Tiled<<<blocks, threads>>>(
            edgeSizeN, d_Mat, nodeSizeN,
            dram_Edge_nodeSources[gIndex], dram_Edge_nodeSourcesStart[gIndex], dram_Edge_nodeSourcesNum[gIndex],
            dram_Edge_nodeTargets[gIndex], dram_Edge_nodeTargetsStart[gIndex], dram_Edge_nodeTargetsNum[gIndex],
            dram_Edge_labelDBIndex[gIndex]
        );
    }

    /* 4] Triangles and Initial Relabel */
    uint64_t *d_Temp;
    cudaMalloc(&d_Temp, NodeMatrixSize * sizeof(uint64_t));

    /* For safety, just copy d_Mat to d_Temp if skipping triangles */
    cudaMemcpy(d_Temp, d_Mat, NodeMatrixSize * sizeof(uint64_t), cudaMemcpyDeviceToDevice);

    /* Initial Canonical Relabel (Map hashes to 0..K integers) */
    WL2_CanonicalRelabelIR(NodeMatrixSize, d_Temp, d_Mat);
    cudaMemcpy(d_Mat, d_Temp, NodeMatrixSize * sizeof(uint64_t), cudaMemcpyDeviceToDevice);


    /* 5]Capture refinement stats of the initialized matrix */
        /* Allocate temporary buffer to extract diagonal node colors */
        uint64_t* d_diag;
        cudaMalloc(&d_diag, nodeSizeN * sizeof(uint64_t));

        /* Extract diagonal elements to analyze node distinction */
        Kernel_WL2_ExtractDiagonals<<<ThreadsAllNodes[0].dimGrid, ThreadsAllNodes[0].dimBlock>>>(
            nodeSizeN, d_Mat, d_diag
        );

        /* Create label and log stats */
        char labelBuf[64];
        sprintf(labelBuf, "WL-2 Init Matrix (Graph %d)", gIndex);
        LogRefinementStats(nodeSizeN, d_diag, labelBuf);

        /* Cleanup temporary buffers */
        cudaFree(d_diag);


    cudaFree(d_Temp);
    return true;
}
/*-------------------------------------------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------------------------------------------*/
/* K1] Refines G0 and G1 simultaneously using a shared hash pool                                     */
/* Ensures that Identical Structure -> Identical Integer Color across both graphs                */
/*-------------------------------------------------------------------------------------------------------------------*/
bool WL2_RefineBothGraphsUnified(int numNodes, uint64_t* d_G0, uint64_t* d_G1, int MAX_ITERATIONS)
{
    size_t MatrixSize = (size_t)numNodes * (size_t)numNodes;
    size_t TotalElements = MatrixSize * 2;
    int prev_bins = -1;

    /* 1] Allocate Combined Workspace */
    /* We need space for G0+G1 combined, and a Map/Scratch buffer of the same size */
    uint64_t *d_Combined, *d_Map;
    if (cudaMalloc(&d_Combined, TotalElements * sizeof(uint64_t)) != cudaSuccess)
    {
        cudaDeviceSynchronize();
    	cudaCheckError();
    	return false;
    }
    if (cudaMalloc(&d_Map,      TotalElements * sizeof(uint64_t)) != cudaSuccess)
    {
        cudaDeviceSynchronize();
    	cudaCheckError();
    	cudaFree(d_Combined);
    	return false;
    }

    for (int i = 0; i < MAX_ITERATIONS; ++i)
    {
        /* 2] Refine Steps, Write to Combined Buffer */
        /* Update G0  First half of Combined */
        Kernel_WL2_UpdatePairs_Tiled<<<ThreadsWLMatrix2DTiled[0].dimGrid, ThreadsWLMatrix2DTiled[0].dimBlock>>>(
            numNodes, d_G0, d_Combined
        );
        cudaDeviceSynchronize();
    	cudaCheckError();


        /* Update G1  Second half of Combined */
        Kernel_WL2_UpdatePairs_Tiled<<<ThreadsWLMatrix2DTiled[0].dimGrid, ThreadsWLMatrix2DTiled[0].dimBlock>>>(
            numNodes, d_G1, d_Combined + MatrixSize
        );
		cudaDeviceSynchronize();
		cudaCheckError();

        /* 3] Unified Canonical Relabeling */
        /* Sorts the entire population (G0 + G1) to define global ranks */
        WL2_CanonicalRelabelIR(TotalElements, d_Combined, d_Map);
        cudaDeviceSynchronize();
    	cudaCheckError();


        /* 4] Write back the new categorical colors to the respective graphs */
        cudaMemcpy(d_G0, d_Combined,              MatrixSize * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_G1, d_Combined + MatrixSize, MatrixSize * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
		cudaDeviceSynchronize();
		cudaCheckError();
        /* 5] Stability Check, count unique bins in the global population */
        /* Reuse d_Map as the scratch workspace for counting */
        int current_bins = WL2_CountUniqueBinsGPU(TotalElements, d_Combined, d_Map);

        if (i < 4)
		{
        	/* Calculate Normalized Resolution */
        	    float nodeEquiv = (float)current_bins / (float)numNodes;

        	    std::cout << "    > WL2 Refine Step " << i
        	              << ": Pair Bins = " << current_bins
        	              << " Node Resolution: " << std::fixed << std::setprecision(1) << nodeEquiv << "/" << numNodes << std::endl;
		}

        if (current_bins == prev_bins && i > 4) break;
        prev_bins = current_bins;
    }

    cudaFree(d_Combined);
    cudaFree(d_Map);
	cudaDeviceSynchronize();
	cudaCheckError();
    return true;
}
/*-------------------------------------------------------------------------------------------------------------------*/


/*-------------------------------------------------------------------------------------------------------------------*/
/* K2] Restores base state and replays the (Inject -> Refine) sequence up to targetDepth          */
/*-------------------------------------------------------------------------------------------------------------------*/
void WL2_RevertPrevDepthBothGraphsIR(
    int targetDepth, int N, const std::vector<IR_StackFrame>& stack,
    const std::vector<uint64_t>& h_Base0, const std::vector<uint64_t>& h_Base1,
    uint64_t* d_G0, uint64_t* d_G1, int MAX_ITER)
{
    size_t bytes = N * N * sizeof(uint64_t);

    /* 1] Restore Base State (Depth 0) */
    /* We must overwrite the current state completely */
    cudaMemcpy(d_G0, h_Base0.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_G1, h_Base1.data(), bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
	cudaCheckError();


    /* 2] Replay History Layer by Layer */
    /* We must replicate the exact sequence of "Inject -> Refine" that got us here */
    for(int d=0; d < targetDepth; d++)
    {
        /* A] Re-Inject Pivot for this depth */
        /* Note: magic constant includes depth 'd' to ensure unique coloring per level */
        uint64_t magic = 0xF000000000000000ULL | (uint64_t)d;

        /* G0 Pivot */
        Kernel_WL2_InjectUniqueColor<<<1,1>>>(N, stack[d].pivotNode, d_G0, magic);
        cudaDeviceSynchronize();
    	cudaCheckError();


        /* G1 Candidate (The one we chose at this depth) */
        int candidateNode = stack[d].candidates[stack[d].candidateIndex];
        Kernel_WL2_InjectUniqueColor<<<1,1>>>(N, candidateNode, d_G1, magic);
        cudaDeviceSynchronize();
    	cudaCheckError();


        /* B] Unified Refinement */
        /* We must refine at every step to recreate the exact color evolution */
        WL2_RefineBothGraphsUnified(N, d_G0, d_G1, MAX_ITER);
        cudaDeviceSynchronize();
    	cudaCheckError();
    }
}
/*-------------------------------------------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------------------------------------------*/
/** L] TODO Check that is is correct @MM */
/*-------------------------------------------------------------------------------------------------------------------*/
bool Host_WL2VerifyHypergraphIR(int numNodes, int numEdges,
                                /* G0 Topology Pointers */
                                const uint* d_src0, const uint* d_srcStart0, const uint* d_srcNum0,
                                const uint* d_tgt0, const uint* d_tgtStart0, const uint* d_tgtNum0,
                                const uint* d_labels0, /* NEW: Graph 0 Labels */

                                /* G1 Topology Pointers */
                                const uint* d_src1, const uint* d_srcStart1, const uint* d_srcNum1,
                                const uint* d_tgt1, const uint* d_tgtStart1, const uint* d_tgtNum1,
                                const uint* d_labels1, /* NEW: Graph 1 Labels */

                                /* Final Canonical Ranks - NxN Matrix */
                                uint64_t* d_Refined_G0, uint64_t* d_Refined_G1)
{
	using HyperEdge = std::tuple<std::vector<uint>, std::vector<uint>, uint>;

	    /* A] Safety Checks */
	    if (numNodes <= 0) return true;
	    if (numEdges > 0) {
	        if (!d_src0 || !d_src1) {
	            std::cout << "Error: HostVerify received NULL pointers." << std::endl;
	            return false;
	        }
	    }

	    /*--------------------------------------------------------------------------------*/
	    /* Step 1: Extract Diagonals & Build Bijection P[u] -> v                          */
	    /*--------------------------------------------------------------------------------*/
	    uint64_t* d_DiagBuffer = nullptr;
	    cudaMalloc(&d_DiagBuffer, 2 * numNodes * sizeof(uint64_t));
	    cudaDeviceSynchronize();

	    /* Extract diagonals */
	    int threads = 256;
	    int blocks = (numNodes + threads - 1) / threads;
	    Kernel_WL2_ExtractDiagonals<<<blocks, threads>>>(numNodes, d_Refined_G0, d_DiagBuffer);
	    Kernel_WL2_ExtractDiagonals<<<blocks, threads>>>(numNodes, d_Refined_G1, d_DiagBuffer + numNodes);
        cudaDeviceSynchronize();
    	cudaCheckError();

	    std::vector<uint64_t> h_Diagonals(2 * numNodes);
	    cudaMemcpy(h_Diagonals.data(), d_DiagBuffer, 2 * numNodes * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	    cudaFree(d_DiagBuffer);
        cudaDeviceSynchronize();
    	cudaCheckError();

	    uint64_t* h_Diagonal_G0 = h_Diagonals.data();
	    uint64_t* h_Diagonal_G1 = h_Diagonals.data() + numNodes;

	    /* Build Mapping P: G0_Node -> G1_Node */
	    std::vector<int> P(numNodes);
	    std::unordered_map<uint64_t, int> map_G1_Rank_to_Idx;
	    map_G1_Rank_to_Idx.reserve(numNodes);

	    for (int i = 0; i < numNodes; i++) {
	        uint64_t rank = h_Diagonal_G1[i];
	        if (map_G1_Rank_to_Idx.count(rank)) return false;
	        map_G1_Rank_to_Idx[rank] = i;
	    }

	    for (int i = 0; i < numNodes; i++) {
	        uint64_t rank = h_Diagonal_G0[i];
	        auto it = map_G1_Rank_to_Idx.find(rank);
	        if (it == map_G1_Rank_to_Idx.end()) return false;
	        P[i] = it->second;
	    }
	    /*--------------------------------------------------------------------------------*/

	    /*--------------------------------------------------------------------------------*/
	    /* Step 2: Fetch Hyperedge Topology Safely (Lambda)                               */
	    /*--------------------------------------------------------------------------------*/
	    bool fetch_success = true;

	    auto FetchGraphParallel = [&](int numE,
	                                  const uint* d_src, const uint* d_sStart, const uint* d_sNum,
	                                  const uint* d_tgt, const uint* d_tStart, const uint* d_tNum,
	                                  const uint* d_lbl, const char* label, int gIdx) -> std::vector<HyperEdge>
	    {
	        std::vector<HyperEdge> edges;
	        if (!fetch_success || numE == 0) return edges;

	        /* 2.1 Bulk Metadata Copy */
	        std::vector<uint> h_sStart(numE), h_sNum(numE), h_tStart(numE), h_tNum(numE), h_Labels(numE);

	        cudaMemcpy(h_sStart.data(), d_sStart, numE * sizeof(uint), cudaMemcpyDeviceToHost);
	        cudaMemcpy(h_sNum.data(),   d_sNum,   numE * sizeof(uint), cudaMemcpyDeviceToHost);
	        cudaMemcpy(h_tStart.data(), d_tStart, numE * sizeof(uint), cudaMemcpyDeviceToHost);
	        cudaMemcpy(h_tNum.data(),   d_tNum,   numE * sizeof(uint), cudaMemcpyDeviceToHost);
	        cudaMemcpy(h_Labels.data(), d_lbl,    numE * sizeof(uint), cudaMemcpyDeviceToHost);
	        cudaDeviceSynchronize();
	    	cudaCheckError();


	        uint totalSrc = 0;
	        uint totalTgt = 0;

	        #pragma omp parallel for reduction(max:totalSrc, totalTgt)
	        for(int i = 0; i < numE; i++) {
	            uint sEnd = h_sStart[i] + h_sNum[i];
	            uint tEnd = h_tStart[i] + h_tNum[i];
	            if (sEnd > totalSrc) totalSrc = sEnd;
	            if (tEnd > totalTgt) totalTgt = tEnd;
	        }


	        uint allocatedSrc = m_edgeNodesSourceSize[gIdx];
	        uint allocatedTgt = m_edgeNodesTargetSize[gIdx];

	        if (totalSrc > allocatedSrc) {
	            std::cout << "Warning: " << label << " Metadata requires " << totalSrc
	                      << " src-indices, but only " << allocatedSrc << " allocated. Clamping." << std::endl;
	            totalSrc = allocatedSrc; // Clamp to prevent crash
	        }
	        if (totalTgt > allocatedTgt) {
	             std::cout << "Warning: " << label << " Metadata requires " << totalTgt
	                       << " tgt-indices, but only " << allocatedTgt << " allocated. Clamping." << std::endl;
	             totalTgt = allocatedTgt; // Clamp to prevent crash
	        }

	        /* 2.3 Safe Data Copy */
	        std::vector<uint> h_srcData, h_tgtData;

	        if(totalSrc > 0) {
	             h_srcData.resize(totalSrc);
	             cudaError_t e = cudaMemcpy(h_srcData.data(), d_src, totalSrc * sizeof(uint), cudaMemcpyDeviceToHost);
	             if (e != cudaSuccess) {
	                 std::cout << "Error: " << label << " Src Data Copy failed: " << cudaGetErrorString(e) << std::endl;
	                 fetch_success = false; return edges;
	             }
	        }

	        if(totalTgt > 0) {
	             h_tgtData.resize(totalTgt);
	             cudaError_t e = cudaMemcpy(h_tgtData.data(), d_tgt, totalTgt * sizeof(uint), cudaMemcpyDeviceToHost);
	             if (e != cudaSuccess) {
	                 std::cout << "Error: " << label << " Tgt Data Copy failed: " << cudaGetErrorString(e) << std::endl;
	                 fetch_success = false; return edges;
	             }
	        }
	        cudaDeviceSynchronize();
	    	cudaCheckError();

	        /* 2.4 Construct Edges */
	        edges.resize(numE);

	        #pragma omp parallel for
	        for (int i = 0; i < numE; i++)
	        {
	            auto& sources = std::get<0>(edges[i]);
	            auto& targets = std::get<1>(edges[i]);

	            sources.reserve(h_sNum[i]);
	            targets.reserve(h_tNum[i]);

	            /* Extract Indices - Bounds Checked */
	            uint sStart = h_sStart[i];
	            for (uint k = 0; k < h_sNum[i]; k++) {
	                if (sStart + k < h_srcData.size()) sources.push_back(h_srcData[sStart + k]);
	            }

	            uint tStart = h_tStart[i];
	            for (uint k = 0; k < h_tNum[i]; k++) {
	                if (tStart + k < h_tgtData.size()) targets.push_back(h_tgtData[tStart + k]);
	            }

	            std::get<2>(edges[i]) = h_Labels[i];
	        }
	        return edges;
	    };

	    /* Pass gIndex (0 or 1) to allow lookup of global allocation sizes */
	    auto Edges0 = FetchGraphParallel(numEdges, d_src0, d_srcStart0, d_srcNum0, d_tgt0, d_tgtStart0, d_tgtNum0, d_labels0, "Graph0", 0);
	    auto Edges1 = FetchGraphParallel(numEdges, d_src1, d_srcStart1, d_srcNum1, d_tgt1, d_tgtStart1, d_tgtNum1, d_labels1, "Graph1", 1);

	    if (!fetch_success) return false;
	    /*--------------------------------------------------------------------------------*/

	    /*--------------------------------------------------------------------------------*/
	    /* Step 3: Canonical HyperEdge Verification                                       */
	    /*--------------------------------------------------------------------------------*/
	    std::map<HyperEdge, int> counts_G1;
	    for (const auto& edge : Edges1) counts_G1[edge]++;

	    std::map<HyperEdge, int> counts_G0_permuted;

	    #pragma omp parallel
	    {
	        std::map<HyperEdge, int> local_counts;

	        #pragma omp for nowait
	        for (int i = 0; i < numEdges; i++)
	        {
	            HyperEdge permuted;
	            auto& p_src = std::get<0>(permuted);
	            auto& p_tgt = std::get<1>(permuted);

	            const auto& orig_src = std::get<0>(Edges0[i]);
	            const auto& orig_tgt = std::get<1>(Edges0[i]);

	            p_src.reserve(orig_src.size());
	            p_tgt.reserve(orig_tgt.size());

	            /* Apply Permutation P */
	            for (uint u : orig_src) p_src.push_back(P[u]);
	            for (uint v : orig_tgt) p_tgt.push_back(P[v]);

	            /* Directional: NO SORTING */
	            std::get<2>(permuted) = std::get<2>(Edges0[i]);
	            local_counts[permuted]++;
	        }

	        #pragma omp critical
	        {
	            for (const auto& kv : local_counts) counts_G0_permuted[kv.first] += kv.second;
	        }
	    }

	    if (counts_G0_permuted.size() != counts_G1.size()) return false;

	    for (const auto& kv : counts_G0_permuted)
	    {
	        auto it = counts_G1.find(kv.first);
	        if (it == counts_G1.end()) return false;
	        if (it->second != kv.second) return false;
	    }
        cudaDeviceSynchronize();
    	cudaCheckError();

	    return true;
}
/*-------------------------------------------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------------------------------------------*/
/* M1] Used to determine is a WL3 check would improve color splitting  */
/*-------------------------------------------------------------------------------------------------------------------*/
WL3SymmetryProfile WL3_AnalyzeSymmetry(int numNodes, uint64_t* d_Matrix, uint64_t* d_Scratch)
{
    WL3SymmetryProfile profile;

    /* Extract diagonal colors */
    thrust::device_vector<uint64_t> d_colors(numNodes);
    Kernel_WL2_ExtractDiagonals<<<ThreadsAllNodes[0].dimGrid, ThreadsAllNodes[0].dimBlock>>>(
        numNodes, d_Matrix, thrust::raw_pointer_cast(d_colors.data())
    );
    cudaDeviceSynchronize();
	cudaCheckError();

    thrust::device_vector<int> d_indices(numNodes);
    thrust::sequence(d_indices.begin(), d_indices.end());

    /* Sort by color */
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(d_colors.begin(), d_indices.begin()));
    thrust::sort(zip_begin, zip_begin + numNodes);

    /* Count bin sizes */
    thrust::device_vector<uint64_t> d_unique_keys(numNodes);
    thrust::device_vector<int> d_counts(numNodes);

    auto end_pair = thrust::reduce_by_key(
        d_colors.begin(), d_colors.end(),
        thrust::constant_iterator<int>(1),
        d_unique_keys.begin(),
        d_counts.begin()
    );

    int num_bins = end_pair.first - d_unique_keys.begin();

    /* Copy to host */
    std::vector<int> h_counts(num_bins);
    cudaMemcpy(h_counts.data(), thrust::raw_pointer_cast(d_counts.data()),
               num_bins * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
	cudaCheckError();
    /* Analyze */
    profile.num_bins = num_bins;
    profile.bin_sizes = h_counts;
    profile.largest_bin = *std::max_element(h_counts.begin(), h_counts.end());
    profile.smallest_bin = *std::min_element(h_counts.begin(), h_counts.end());
    profile.avg_bin_size = (double)numNodes / num_bins;

    /* Symmetry score: How far from discrete partition? */
    /* 0.0 = fully discrete (num_bins == numNodes) */
    /* 1.0 = single bin (num_bins == 1) */
    profile.symmetry_score = (numNodes > 1) ?
        (1.0 - ((double)num_bins - 1.0) / ((double)numNodes - 1.0)) : 0.0;

    return profile;
}
/*-------------------------------------------------------------------------------------------------------------------*/


/*-------------------------------------------------------------------------------------------------------------------*/
/* M2] WL-3 Refinement: Deterministic Sorted Sequence Extraction                                                     */
/*-------------------------------------------------------------------------------------------------------------------*/
bool WL3_GraphTripletColoring(int gIndex, int MAX_ITERATIONS)
{
    int N = m_numNodes[gIndex];
    int E = m_numEdges[gIndex];

    std::cout << "------------------------------------------------------------------------" << std::endl;
    std::cout << "WL-3: Deterministic Hash Squence " << std::endl;
    std::cout << "------------------------------------------------------------------------" << std::endl;

    /*--------------------------------------------------------------------*/
    /*1] Build distance-3 symmetric mask with 8bit values */
    uint8_t *d_M1, *d_M2, *d_M3;
    size_t memSize = N * N * sizeof(uint8_t);
    cudaMalloc(&d_M1, memSize);
    cudaMalloc(&d_M2, memSize);
    cudaMalloc(&d_M3, memSize);
    cudaMemset(d_M1, 0, memSize);
    cudaDeviceSynchronize();
	cudaCheckError();

    int threads = 256;
    int blocksMask = (E + threads - 1) / threads;
    Kernel_BuildAdjacencyMask_Symmetric<<<blocksMask, threads>>>(   E, dram_Edge_nodeSources[gIndex], dram_Edge_nodeSourcesStart[gIndex], dram_Edge_nodeSourcesNum[gIndex],
																	dram_Edge_nodeTargets[gIndex], dram_Edge_nodeTargetsStart[gIndex], dram_Edge_nodeTargetsNum[gIndex],
																	d_M1, N );
    cudaDeviceSynchronize();
	cudaCheckError();

    dim3 grid2D((N + 15) / 16, (N + 15) / 16);
    dim3 block2D(16, 16);
    Kernel_BooleanMatrixMultiply<<<grid2D, block2D>>>(N, d_M1, d_M1, d_M2);
    Kernel_BooleanMatrixMultiply<<<grid2D, block2D>>>(N, d_M2, d_M1, d_M3);
    cudaDeviceSynchronize();
	cudaCheckError();
    cudaFree(d_M1); cudaFree(d_M2);

    uint8_t* d_AdjMask = d_M3; /* Final mask */
    /*--------------------------------------------------------------------*/

    /*--------------------------------------------------------------------*/
    /*2] Count and initialize sparse triplets */
    int* d_tot; cudaMalloc(&d_tot, sizeof(int)); cudaMemset(d_tot, 0, sizeof(int));
    Kernel_WL3_CountNonZeroTriplets<<<grid2D, block2D>>>(N, d_AdjMask, d_tot);
    cudaDeviceSynchronize();
	cudaCheckError();

    int actual_triplets = 0;
    cudaMemcpy(&actual_triplets, d_tot, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
	cudaCheckError();
    if (actual_triplets == 0)
    {
        cudaDeviceSynchronize();
    	cudaCheckError();
        cudaFree(d_AdjMask); cudaFree(d_tot);
        return false;
    }

    SparseTriplet *d_triplets;
    cudaMalloc(&d_triplets, actual_triplets * sizeof(SparseTriplet));
    int* d_ni; cudaMalloc(&d_ni, sizeof(int)); cudaMemset(d_ni, 0, sizeof(int));
    Kernel_WL3_InitSparseTriplets<<<grid2D, block2D>>>(N, dram_WL2_MatrixColors[gIndex], d_AdjMask, d_triplets, d_ni);
    cudaDeviceSynchronize();
	cudaCheckError();
    /*--------------------------------------------------------------------*/

    /*--------------------------------------------------------------------*/
    /*3] Build mapping for refinement, temporary offsets */
    int* d_keys; cudaMalloc(&d_keys, actual_triplets * sizeof(int));
    int* d_values; cudaMalloc(&d_values, actual_triplets * sizeof(int));
    thrust::device_ptr<SparseTriplet> d_tri_ptr(d_triplets);

    thrust::transform(d_tri_ptr, d_tri_ptr + actual_triplets, thrust::device_ptr<int>(d_keys), GetTripletU());
    thrust::sequence(thrust::device_ptr<int>(d_values), thrust::device_ptr<int>(d_values) + actual_triplets);
    thrust::sort_by_key(thrust::device_ptr<int>(d_keys), thrust::device_ptr<int>(d_keys) + actual_triplets, thrust::device_ptr<int>(d_values));

    int* d_offsets; cudaMalloc(&d_offsets, (N + 1) * sizeof(int));
    thrust::lower_bound(thrust::device_ptr<int>(d_keys), thrust::device_ptr<int>(d_keys) + actual_triplets,
                        thrust::make_counting_iterator(0), thrust::make_counting_iterator(N),
                        thrust::device_ptr<int>(d_offsets));
    cudaMemcpy(d_offsets + N, &actual_triplets, sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
	cudaCheckError();
    /*--------------------------------------------------------------------*/

    /*--------------------------------------------------------------------*/
    /*4] Iterative refinement */
    uint64_t *d_scratch; cudaMalloc(&d_scratch, actual_triplets * sizeof(uint64_t));
    for (int iter = 0; iter < MAX_ITERATIONS; iter++)
    {
        Kernel_WL3_UpdateSparseTriplets_Fast<<<(actual_triplets + 255)/256, 256>>>(actual_triplets, d_triplets, d_offsets, d_values, d_scratch);
        Kernel_WL3_WritebackColors<<<(actual_triplets + 255)/256, 256>>>(actual_triplets, d_triplets, d_scratch);
        cudaDeviceSynchronize();
        cudaDeviceSynchronize();
    	cudaCheckError();
    }
    /*--------------------------------------------------------------------*/

    /*--------------------------------------------------------------------*/
    /*5] Deterministic canonical sorting, using functor */
    std::cout << "WL-3: Canonical Sorting" << std::endl;
    thrust::sort(d_tri_ptr, d_tri_ptr + actual_triplets, TripletComparator());
    cudaDeviceSynchronize();
	cudaCheckError();
    /*--------------------------------------------------------------------*/

    /*--------------------------------------------------------------------*/
    /*6] Recalculate node offsets for sequence hashing */
    int* d_node_offsets; cudaMalloc(&d_node_offsets, (N + 1) * sizeof(int));
    auto it = thrust::make_transform_iterator(d_tri_ptr, GetTripletU());
    thrust::lower_bound(it, it + actual_triplets, thrust::make_counting_iterator(0), thrust::make_counting_iterator(N), thrust::device_ptr<int>(d_node_offsets));
    cudaMemcpy(d_node_offsets + N, &actual_triplets, sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
	cudaCheckError();
    /*--------------------------------------------------------------------*/

    /*--------------------------------------------------------------------*/
    /*7] Sequence hashing */
    uint64_t* d_Nodes; cudaMalloc(&d_Nodes, N * sizeof(uint64_t));
    Kernel_WL3_DeterministicSequenceHash<<<(N + 255)/256, 256>>>(N, d_triplets, d_node_offsets, d_Nodes);
    cudaDeviceSynchronize();
	cudaCheckError();
    /*--------------------------------------------------------------------*/

    /* 8] Results and injection into the WL2 Color Matrix  */
        uint64_t* d_CountScratch; cudaMalloc(&d_CountScratch, N * sizeof(uint64_t));

        char labelBuf[64];
        sprintf(labelBuf, "WL-3 Final Results (Graph %d)", gIndex);
        LogRefinementStats(N, d_Nodes, labelBuf);

        int wl3_node_bins = WL2_CountUniqueBinsGPU(N, d_Nodes, d_CountScratch);
        std::cout << "WL-3 Final Results: " << wl3_node_bins << " / " << N << " bins" << std::endl;

        bool isImproved = (wl3_node_bins > m_WL_BinCount[gIndex]);
        if (isImproved)
        {
            std::cout << "Success: Symmetry Improved :-)" << std::endl;

            int threads = 256;
            int blocks = (N + threads - 1) / threads;

            Kernel_Wl2_InjectDiagonalColors<<<blocks, threads>>>(N, d_Nodes, dram_WL2_MatrixColors[gIndex]);

            cudaDeviceSynchronize();
        	cudaCheckError();

            m_WL_BinCount[gIndex] = wl3_node_bins;
        }
        cudaDeviceSynchronize();
    	cudaCheckError();
        /* 9] Cleanup */
        /* Only free what was actually allocated in this scope */
        cudaFree(d_triplets); cudaFree(d_scratch); cudaFree(d_node_offsets);
        cudaFree(d_Nodes);    cudaFree(d_AdjMask); cudaFree(d_CountScratch);
        cudaFree(d_tot);      cudaFree(d_ni); cudaFree(d_keys); cudaFree(d_values); cudaFree(d_offsets);
        cudaDeviceSynchronize();
    	cudaCheckError();

    return isImproved;
}
/*===================================================================================================================*/

/*===================================================================================================================*/
/** M] Init GPU WL-2 Graph Pair Coloring
 *
 * Performs the 2-Dimensional Weisfeiler-Leman refinement
 * Unlike WL-1 (which uses raw hashes), WL-2 operates on "Colors" (Canonical Integers)
 *
 * 1. The input 'd_Matrix' contains the current stable coloring (integers 0..K)
 * 2. We update the color of every pair (u, v) based on the colors of its neighbors
 * NewColor(u, v) = Hash( OldColor(u, v), Mset{ (Color(u, w), Color(w, v)) | w in V } )
 * 3. These new signatures are then re-mapped to dense integers (Colors) for the next iteration
 *
 * Parms:
 * - numNodes: Number of nodes in the graph (N)
 * - d_Matrix_G0: The NxN color matrix for Graph 0
 * - d_Matrix_G1: The NxN color matrix for Graph 1
 * - d_Colors_Map: The mapping table from Hash -> Integer Color
 *
 * Updates dram_WL2_MatrixColors Returns true if stable */
/*===================================================================================================================*/
bool GPU_WL2GraphInitColoring(int gIndex, int MAX_ITERATIONS)
{
	    int nodeSizeN = m_numNodes[gIndex];
	    int edgeSizeN = m_numEdges[gIndex];
	    size_t NodeMatrixSize = (size_t)nodeSizeN * (size_t)nodeSizeN;

	    std::cout << "------------------------------------------------------------------------" << std::endl;
	    std::cout << "GPU WL2 Init With Opt WL3 Graph " << gIndex << std::endl;
	    std::cout << "------------------------------------------------------------------------" << std::endl;

	    cudaDeviceSynchronize();
	    /* 0] Memory Safety Check */
	    double required_MB = (double)NodeMatrixSize * 16.0 / (1024.0 * 1024.0);
	    if (required_MB > 0.75 * m_MaxGPUMemoryMB)
	    {
	        std::cout << "Error Insufficient GPU memory: Required " << required_MB
	                  << " MB, Available " << (0.75 * m_MaxGPUMemoryMB) << " MB" << std::endl;
	        m_WL_BinCount[gIndex] = 0;
	        return false;
	    }
	    cudaDeviceSynchronize();
		cudaCheckError();

	    /* 1] Allocate and Clear */
	    /* Free existing if re-initializing */
	    if (m_isWL2Alloc[gIndex])
	    {
	        cudaFree(dram_WL2_MatrixColors[gIndex]);
	        cudaDeviceSynchronize();
	    	cudaCheckError();
	    }

	    cudaMalloc((void**)&dram_WL2_MatrixColors[gIndex], NodeMatrixSize * sizeof(uint64_t));
	    cudaMemset(dram_WL2_MatrixColors[gIndex], 0, NodeMatrixSize * sizeof(uint64_t));
	    cudaDeviceSynchronize();
		cudaCheckError();

	    m_isWL2Alloc[gIndex] = true;
	    cudaDeviceSynchronize();
		cudaCheckError();

	    uint64_t *d_MatrixEleColors = dram_WL2_MatrixColors[gIndex];
	    /*----------------------------------------------------------------------------------------------*/

	    /*----------------------------------------------------------------------------------------------*/
	    /* 2] Set Matrix using hashes */


	    /* 2.1] Diagonals */
	    Kernel_WL2_Init_Diagonal<<<ThreadsAllNodes[0].dimGrid, ThreadsAllNodes[0].dimBlock>>>(
	        nodeSizeN, dram_Node_IOTag[gIndex], dram_NodeColorHashes[gIndex], d_MatrixEleColors );
	    cudaDeviceSynchronize();
		cudaCheckError();

	    /* 2.2] Self-Loops */
	    Kernel_WL2_DetectSelfLoops<<<ThreadsAllNodes[0].dimGrid, ThreadsAllNodes[0].dimBlock>>>(
	        nodeSizeN,
	        dram_Node_edgePrevs[gIndex], dram_Node_edgePrevsStart[gIndex], dram_Node_edgePrevsNum[gIndex],
	        dram_Node_edgeNexts[gIndex], dram_Node_edgeNextsStart[gIndex], dram_Node_edgeNextsNum[gIndex],
	        d_MatrixEleColors );
	    cudaDeviceSynchronize();
		cudaCheckError();

	    /* 2.3] HyperEdges */
	    if (edgeSizeN > 0)
	    {
	        int threadsPerBlock = 256;
	        int numBlocks = (edgeSizeN + (threadsPerBlock/32) - 1) / (threadsPerBlock/32);
	        if (numBlocks > 0)
	        {
	            Kernel_WL2_InitHyperEdges_Tiled<<<numBlocks, threadsPerBlock>>>(
	                edgeSizeN, d_MatrixEleColors, nodeSizeN,
	                dram_Edge_nodeSources[gIndex], dram_Edge_nodeSourcesStart[gIndex], dram_Edge_nodeSourcesNum[gIndex],
	                dram_Edge_nodeTargets[gIndex], dram_Edge_nodeTargetsStart[gIndex], dram_Edge_nodeTargetsNum[gIndex],
	                dram_Edge_labelDBIndex[gIndex] );
	            cudaDeviceSynchronize();
	        	cudaCheckError();
	        }
	    }
	    else
	    {
	        std::cout << " Graph " << gIndex << " has 0 edges; skipping edge init" << std::endl;
	    }
	    cudaDeviceSynchronize();
		cudaCheckError();
	    /*----------------------------------------------------------------------------------------------*/

	    /*----------------------------------------------------------------------------------------------*/
	    /* 2.4] Triangle Counts (Conditional) */
	    uint64_t *d_Temp;
	    cudaMalloc((void**)&d_Temp, NodeMatrixSize * sizeof(uint64_t));
	    cudaDeviceSynchronize();
		cudaCheckError();

	    size_t non_zero_count = thrust::count_if(thrust::device_pointer_cast(d_MatrixEleColors),
	                                             thrust::device_pointer_cast(d_MatrixEleColors + NodeMatrixSize),
	                                             thrust::placeholders::_1 != 0);
	    double density = (double)non_zero_count / (double)NodeMatrixSize;

	    if (density > 0.01 && density < 0.5)
	    {
	        std::cout << "TriangleCountDensity = " << (density * 100.0) << "%" << std::endl;
	        dim3 dimBlock(TRIANGLE_TILE_SIZE, TRIANGLE_TILE_SIZE);
	        dim3 dimGrid((nodeSizeN + TRIANGLE_TILE_SIZE - 1) / TRIANGLE_TILE_SIZE,
	                     (nodeSizeN + TRIANGLE_TILE_SIZE - 1) / TRIANGLE_TILE_SIZE);
	        Kernel_WL2_InjectTriangleCounts_Tiled<<<dimGrid, dimBlock>>>(nodeSizeN, d_MatrixEleColors, d_Temp);

	        /* Copy result back to Main Matrix */
	        cudaMemcpy(d_MatrixEleColors, d_Temp, NodeMatrixSize * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
	        cudaDeviceSynchronize();
	    	cudaCheckError();
	    }
	    else
	    {
	        std::cout << "Skipping: TriangleCountDensity =" << (density * 100.0) << "%" << std::endl;
	        /* Copy Matrix to Temp just to prep for Relabel input/output convention */
	        cudaMemcpy(d_Temp, d_MatrixEleColors, NodeMatrixSize * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
	        cudaDeviceSynchronize();
	    	cudaCheckError();
	    }
	    /*----------------------------------------------------------------------------------------------*/


	    /*----------------------------------------------------------------------------------------------*/
	    /* 3] Initial Canonical Relabel */
	    /* Map raw hashes to integers 0..K so the refinement starts clean */
	    WL2_CanonicalRelabelIR(NodeMatrixSize, d_Temp, d_MatrixEleColors);
	    cudaDeviceSynchronize();
		cudaCheckError();

	    /* Ensure final result is in d_MatrixEleColors */
	    cudaMemcpy(d_MatrixEleColors, d_Temp, NodeMatrixSize * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
	    cudaDeviceSynchronize();
		cudaCheckError();
	    /*----------------------------------------------------------------------------------------------*/

	    cudaFree(d_Temp);
	    return true;
}
/*===================================================================================================================*/


/*===================================================================================================================*/

/*-------------------------------------------------------------------------------------------------------------------*/
/* Debug Helper: Write WL-2 Matrix to File (Format: Row Col ColorHash)                            */
/*-------------------------------------------------------------------------------------------------------------------*/
void WriteWL2Log(int gIndex, int numNodes, uint64_t* d_Matrix, const std::string& filename)
{
    std::cout << " Debug Writing WL-2 State for Graph " << gIndex << " to " << filename << std::endl;

    size_t MatrixSize = (size_t)numNodes * (size_t)numNodes;
    std::vector<uint64_t> h_Matrix(MatrixSize);

    cudaMemcpy(h_Matrix.data(), d_Matrix, MatrixSize * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    FILE* fp = fopen(filename.c_str(), "w");
    if (!fp) {
        std::cerr << "Error opening file " << filename << std::endl;
        return;
    }

    fprintf(fp, "%% Graph %d WL-2 Matrix Log\n", gIndex);
    fprintf(fp, "%% Node Count: %d\n", numNodes);
    fprintf(fp, "%% Columns: Row Col Value(Hex)\n");
    fprintf(fp, "--------------------------------------------------\n");

    long long nz_count = 0;
    for (int i = 0; i < numNodes; i++) {
        for (int j = 0; j < numNodes; j++) {
            uint64_t val = h_Matrix[i * numNodes + j];
            if (val != 0)
            {
                fprintf(fp, "%d %d %016llx\n", i, j, (unsigned long long)val);
                nz_count++;
            }
        }
    }
    fclose(fp);
    std::cout << " Debug Write Complete (" << nz_count << " entries)" << std::endl;
}
/*-------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------*/
/* Helper: Logs recursive search steps with tree-style indentation                               */
/* Usage: Call inside Recursive_IsomorphismSearch to visualize the backtracking tree             */
/*-----------------------------------------------------------------------------------------------*/
void LogRecursiveSearchBranch(int depth, int pivot, int target, const char* status, bool isSuccess)
{
    /* 1. Create indentation string based on current depth */
    /* e.g., Depth 0 = "", Depth 1 = "  | ", Depth 2 = "  |   | " */
    for (int i = 0; i < depth; i++)
    {
        std::cout << "  | ";
    }

    /* 2. Print Tree Branch marker */
    std::cout << "|-- Depth " << depth << ": ";

    /* 3. Log the specific action being taken */
    if (target >= 0)
    {
        std::cout << "Map P" << pivot << " -> T" << target << " ";
    }
    else
    {
        /* Target -1 implies we are just selecting the pivot or backtracking */
        std::cout << "Pivot P" << pivot << " ";
    }

    /* 4. Append Status Tag */
    /* e.g., [Refining], [Stable], [Conflict], [Pruned] */
    std::cout << "[" << status << "]";

    /* 5. Add visual cue for result */
    if (isSuccess)
    {
        std::cout << " -> OK" << std::endl;
    }
    else
    {
        std::cout << " -> Backtrack" << std::endl;
    }
}
/*-----------------------------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------*/
/* N] Iterative-Refinement WL2 Search with State Restoration
 *
 *
 * 1. Select a Pivot Bin (a set of nodes that look locally identical but might be different)
 * 2. Pick a "Pivot" node from G0's bin
 * 3. Try mapping it to a candidate in G1's bin ("Individualization")
 * 4. Refine (WL2) both graphs to propagate this constraint
 * - If graphs diverge (counts mismatch) -> Backtrack (Wrong guess)
 * - If graphs stabilize matching -> Advance (Deepen search)
 * - If partition is discrete (N bins) -> Full Check (Verify Isomorphism)
 *
 * Frame:
 * - Represents one level of "splitting"
 * - frame.pivotNode: The node in G0 we are currently fixing
 * - frame.candidates: The list of possible matches in G1
 * - frame.candidateIndex: Which G1 candidate we are currently testing */
/*-------------------------------------------------------------------------------------------------------------------*/
bool GPU_WL2Refinement_IsIso(int max_depth, int max_pivots, int timeout_seconds)
{
    auto start_time = std::chrono::steady_clock::now();
    int numNodes = m_numNodes[0];

    if (m_numNodes[0] != m_numNodes[1])
    {
        std::cout << "Node count mismatch " << m_numNodes[0] << " vs " << m_numNodes[1]  << std::endl;
        return false;
    }

    size_t MatrixBytes = (size_t)numNodes * (size_t)numNodes * sizeof(uint64_t);

    /*---------------------------------------------------------------------------------------------------------------*/
    /* Stats Counters */
    int maxRefineSteps            = 0;
    long long totalPivots         = 0;
    long long totalBacktracks     = 0;
    long long totalRowPrunes      = 0;
    long long totalAmbiguousNodes = 0;

    /* Create Base Snapshots */
    std::vector<uint64_t> h_Base_G0(numNodes * numNodes);
    std::vector<uint64_t> h_Base_G1(numNodes * numNodes);
    cudaMemcpy(h_Base_G0.data(), dram_WL2_MatrixColors[0], MatrixBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Base_G1.data(), dram_WL2_MatrixColors[1], MatrixBytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
	cudaCheckError();

    /* Allocate scratch space */
    uint64_t *d_Scratch_Diag;
    cudaMalloc(&d_Scratch_Diag, 2 * numNodes * sizeof(uint64_t));
    cudaDeviceSynchronize();
	cudaCheckError();
    /*---------------------------------------------------------------------------------------------------------------*/

    /*---------------------------------------------------------------------------------------------------------------*/
    /** Phase 1: Initial Symmetry Check for nodes that are the same in the bin to reduce pivots */
    /*---------------------------------------------------------------------------------------------------------------*/
    std::cout << "\n IR1: Checking for bin AutoMorpshim " << std::endl;

    std::vector<int> initalSymmClasses_G0 = WL2_DetectSymmetryClasses(numNodes, dram_WL2_MatrixColors[0]);
    std::vector<int> initalSymmClasses_G1 = WL2_DetectSymmetryClasses(numNodes, dram_WL2_MatrixColors[1]);

    if (initalSymmClasses_G0.size() != initalSymmClasses_G1.size())
    {
        std::cout << " IR1: AutoMorpshim Symmetry Mismatch!" << std::endl;
        std::cout << "  G0 Symmetry: " << initalSymmClasses_G0.size() << std::endl;
        std::cout << "  G1 Symmetry: " << initalSymmClasses_G1.size() << std::endl;
        cudaFree(d_Scratch_Diag);
        cudaDeviceSynchronize();
    	cudaCheckError();
        return false;
    }
    /*---------------------------------------------------------------------------------------------------------------*/

    /* Initialize Search Stack */
    std::vector<IR_StackFrame> hostStack;
    hostStack.reserve(max_depth);
    hostStack.push_back({ -1, -1, {} }); /* Root Frame */
    int refineStep = 0;
    bool isIsomorphic = false;

    std::cout << "\n IR2: Starting Refinement of  " <<initalSymmClasses_G0.size()<<" Bins "<< std::endl;
    /*---------------------------------------------------------------------------------------------------------------*/
    /* Loop: Iterative Refinement with single step backtrack safety */
    /*---------------------------------------------------------------------------------------------------------------*/
    while (refineStep >= 0)
    {
        /* Stats */
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start_time ).count();
        if (elapsed > timeout_seconds)
        {
            std::cout << "\n IR2 TimeExceeded " << timeout_seconds << "s | Checked " << totalPivots << " pivots" << std::endl;
            break;
        }
        if (refineStep > maxRefineSteps) maxRefineSteps = refineStep;
        /* ----------------------- */

        IR_StackFrame &frame = hostStack[refineStep];

        /*--------------------------------------------------------------*/
        /* Select a Pivot Bin to Split */
        /* executed only when entering a new depth (frame.pivotNode == -1) */
        /*--------------------------------------------------------------*/
        if (frame.pivotNode == -1)
        {
            /* 1] Find the smallest ambiguous color bin in G0 (Deterministic Choice) */
            BinInfoGPU bin = WL2_SelectPivotFromBin(numNodes, dram_WL2_MatrixColors[0], d_Scratch_Diag, dram_Node_IOTag[0]);

            /* [Case A] No bins left to split? Partition is Discrete! */
            if (bin.isTargetColorFound == 0)
            {
                /* 2] Perform Exact Isomorphism Check on Host: do the edges match: @MM this is based on nauty so check */
            	bool isVerified = Host_WL2VerifyHypergraphIR(
            	                    numNodes, m_numEdges[0],
            	                    dram_Edge_nodeSources[0], dram_Edge_nodeSourcesStart[0], dram_Edge_nodeSourcesNum[0],
            	                    dram_Edge_nodeTargets[0], dram_Edge_nodeTargetsStart[0], dram_Edge_nodeTargetsNum[0],
            	                    dram_Edge_labelDBIndex[0],

            	                    dram_Edge_nodeSources[1], dram_Edge_nodeSourcesStart[1], dram_Edge_nodeSourcesNum[1],
            	                    dram_Edge_nodeTargets[1], dram_Edge_nodeTargetsStart[1], dram_Edge_nodeTargetsNum[1],
									dram_Edge_labelDBIndex[1],

            	                    dram_WL2_MatrixColors[0], dram_WL2_MatrixColors[1] );

                if (isVerified)
                {
                    std::cout << "Isomorphism Verified at Depth " << refineStep << std::endl;
                    isIsomorphic = true;
                    break; /* Found it! Exit Loop */
                }
                else
                {
                    /* Backtrack Ghost Mapping: Colors match, Structure doesn't */
                    totalBacktracks++;
                    hostStack.pop_back(); /* Kill this dead branch */
                    refineStep--;

                    /* Restore State of Parent Frame */
                    if(refineStep >= 0)
                    {
                        hostStack[refineStep].candidateIndex++; /* Move parent to next guess */

                        /* Revert BOTH graphs because color space is shared */
                        WL2_RevertPrevDepthBothGraphsIR(
                            refineStep, numNodes, hostStack, h_Base_G0, h_Base_G1,
                            dram_WL2_MatrixColors[0], dram_WL2_MatrixColors[1], 50
                        );
                    }
                    continue; /* Jump to top; loop will pick up Parent Frame at Phase 2 */
                }
            }

            /* We found a bin to split */
            frame.pivotNode = bin.pivotNode;
            totalAmbiguousNodes += bin.candidates.size();
            LogRecursiveSearchBranch(refineStep, frame.pivotNode, -1, "Pivot Selected", true);

            /* Find matching bin in G1 */
            BinInfoGPU binG1 = WL2_SelectPivotFromBin(numNodes, dram_WL2_MatrixColors[1], d_Scratch_Diag, dram_Node_IOTag[1]);

            /* If bin sizes differ, this path is impossible */
            if (binG1.isTargetColorFound == 0 || binG1.candidates.size() != bin.candidates.size())
            {
                totalBacktracks++;
                hostStack.pop_back();
                refineStep--;
                if(refineStep >= 0)
                {
                    /* Revert Both graphs */
                    hostStack[refineStep].candidateIndex++;
                    WL2_RevertPrevDepthBothGraphsIR(refineStep, numNodes, hostStack, h_Base_G0, h_Base_G1,
                                                  dram_WL2_MatrixColors[0], dram_WL2_MatrixColors[1], 100);
                }
                continue;
            }

            frame.candidates = binG1.candidates;
            frame.candidateIndex = 0;
        }
        /*--------------------------------------------------------------*/
        /* End Phase 1 */
        /*--------------------------------------------------------------*/

        /*--------------------------------------------------------------*/
        /* Phase 2: Try Candidates (Pivot G0 -> Candidate G1) */
        /*--------------------------------------------------------------*/
        /* 0] Have we run out of candidates in G1? */
        if (frame.candidateIndex >= (int)frame.candidates.size())
        {
        	LogRecursiveSearchBranch(refineStep, frame.pivotNode, -1, "Backtrack", false);
            /* Exhausted all options at this depth, The mistake is higher up */
            totalBacktracks++;
            hostStack.pop_back();
            refineStep--;
            if (refineStep >= 0)
            {
                hostStack[refineStep].candidateIndex++;
                /* Revert Both graphs */
                WL2_RevertPrevDepthBothGraphsIR(refineStep, numNodes, hostStack, h_Base_G0, h_Base_G1,
                                                dram_WL2_MatrixColors[0], dram_WL2_MatrixColors[1], 100);
            }
            continue;
        }

        /* Check Safety Limit */
        if (totalPivots > max_pivots)
        {
            std::cout << "\n PivotLimitExceeded " << max_pivots << std::endl;
            break;
        }

        /* 1] Pick Next Candidate */
        totalPivots++;
        int current_G1_node = frame.candidates[frame.candidateIndex];

        LogRecursiveSearchBranch(refineStep, frame.pivotNode, current_G1_node, "Refining", true);
        /* 3] Full Refinement Test */

        /* 3a] Reset State to current depth (Wipe any previous failed candidate effects) */
        /* This is expensive but necessary for correctness with Unified Refinement */
        WL2_RevertPrevDepthBothGraphsIR( refineStep, numNodes, hostStack, h_Base_G0, h_Base_G1,
                                         dram_WL2_MatrixColors[0], dram_WL2_MatrixColors[1], 100 );

        /* 3b] Inject Pivots*/
        /* Use depth-dependent magic color to ensure unique constraints per level */
        uint64_t magic = 0xF000000000000000ULL | (uint64_t)refineStep;

        /* Pin G0 Pivot */
        Kernel_WL2_InjectUniqueColor<<<1,1>>>(numNodes, frame.pivotNode, dram_WL2_MatrixColors[0], magic);
        cudaDeviceSynchronize();
    	cudaCheckError();


        /* Pin G1 Candidate */
        Kernel_WL2_InjectUniqueColor<<<1,1>>>(numNodes, current_G1_node, dram_WL2_MatrixColors[1], magic);
        cudaDeviceSynchronize();
    	cudaCheckError();


        /* 3c] Refine Both Graphs Together (Propagate Constraints) */
        /* This ensures G0 and G1 drift in the exact same direction */
        WL2_RefineBothGraphsUnified(numNodes, dram_WL2_MatrixColors[0], dram_WL2_MatrixColors[1], 100);
        cudaDeviceSynchronize();
    	cudaCheckError();

        /* 4] Compare Partitions */
        if (WL2_CompareBinCountsIR(numNodes, dram_WL2_MatrixColors[0], dram_WL2_MatrixColors[1], d_Scratch_Diag))
        {
        	LogRecursiveSearchBranch(refineStep, frame.pivotNode, current_G1_node, "Stable", true);
            /* Consistent split, Dive deeper */
            refineStep++;
            hostStack.push_back({ -1, -1, {} }); /* New empty frame */
        }
        else
        {
        	LogRecursiveSearchBranch(refineStep, frame.pivotNode, current_G1_node, "Conflict", false);
            /* Inconsistent split, Try next candidate */
            frame.candidateIndex++;
        }
        /*--------------------------------------------------------------*/
    } /* End while loop */

    cudaDeviceSynchronize();
	cudaCheckError();


    /* Cleanup */
    if (d_Scratch_Diag != nullptr)
    {
            cudaFree(d_Scratch_Diag);
            d_Scratch_Diag = nullptr; // Prevent double-free
    }

    cudaDeviceSynchronize();
	cudaCheckError();

    /* Restore Original State */
    cudaMemcpy(dram_WL2_MatrixColors[0], h_Base_G0.data(), MatrixBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dram_WL2_MatrixColors[1], h_Base_G1.data(), MatrixBytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
	cudaCheckError();
    return isIsomorphic;
}
/*-------------------------------------------------------------------------------------------------------------------*/


/*-------------------------------------------------------------------------------------------------------------------*/
/* Helper: Check Node Isomorphism Status (Extracts Diagonals for Correct Stats)                   */
/*-------------------------------------------------------------------------------------------------------------------*/
bool WL2_CheckNodeIsomorphismStatus(int N, uint64_t* d_MatrixG0, uint64_t* d_MatrixG1)
{
    /* 1] Allocate Scratch for Diagonals (Size N, not N*N) */
    uint64_t *d_DiagG0, *d_DiagG1;
    cudaMalloc(&d_DiagG0, N * sizeof(uint64_t));
    cudaMalloc(&d_DiagG1, N * sizeof(uint64_t));

    /* 2] Extract Diagonals Nodes Only */
    /* This kernel is O(N) and reads only the diagonal elements (i,i) */
    Kernel_WL2_ExtractDiagonals<<<ThreadsAllNodes[0].dimGrid, ThreadsAllNodes[0].dimBlock>>>(
        N, d_MatrixG0, d_DiagG0
    );
    Kernel_WL2_ExtractDiagonals<<<ThreadsAllNodes[0].dimGrid, ThreadsAllNodes[0].dimBlock>>>(
        N, d_MatrixG1, d_DiagG1
    );
    cudaDeviceSynchronize();

    /* 3] Histogram Node Colors */
    /* Sort Diagonals */
    thrust::device_ptr<uint64_t> t_g0(d_DiagG0);
    thrust::device_ptr<uint64_t> t_g1(d_DiagG1);
    thrust::sort(t_g0, t_g0 + N);
    thrust::sort(t_g1, t_g1 + N);

    /* Check for Exact Match */
    bool match = thrust::equal(t_g0, t_g0 + N, t_g1);


    if (!match)
    {
        std::cout << "\n Diagonal Mismatch :" << std::endl;
        std::vector<uint64_t> h_g0(N), h_g1(N);

        /* Copy diagonals back to host */
        cudaMemcpy(h_g0.data(), d_DiagG0, N * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_g1.data(), d_DiagG1, N * sizeof(uint64_t), cudaMemcpyDeviceToHost);

        /* Sort them on CPU to match the GPU comparison logic */
        std::sort(h_g0.begin(), h_g0.end());
        std::sort(h_g1.begin(), h_g1.end());

        /* Print the first 10 differences */
        int diff_count = 0;
        for(int i = 0; i < N; i++)
        {
            if (h_g0[i] != h_g1[i])
            {
                std::cout << "  Idx " << std::setw(4) << i
                          << ": G0=" << h_g0[i]
                          << " != G1=" << h_g1[i] << std::endl;
                diff_count++;
                if (diff_count >= 10) break;
            }
        }
        std::cout << "  Total mismatched nodes: " << diff_count << std::endl;
    }

    /* 4] Cleanup */
    cudaFree(d_DiagG0);
    cudaFree(d_DiagG1);

    return match;
}
/*-------------------------------------------------------------------------------------------------------------------*/



/*===================================================================================================================*/
/* O] Iterative-Refinement WL2 Search with State Restoration
 * Phase 0:   [O(N)]      Fast Rejection - Signature histogram comparison
 * Phase 1:   [O(N log N)] WL-1 Refinement - Node coloring iteration
 * Phase 2:   [O(E log E)] Edge Verification - Port-ordered edge hash comparison
 * Phase 3:   [O(N^2)]     WL-2 Initialization - Matrix construction & Triangle Injection
 * Phase 3.5: [O(M^2)]     WL-3 Triplet Refinement - (Sparse) Higher order refinement
 * Phase 4:   [Exp]        IR Search - Individualization-Refinement with symmetry pruning */
/*===================================================================================================================*/
bool GPU_CheckHypergraphIsomorphism()
{
    auto total_start = std::chrono::steady_clock::now();

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "GPU Isomorphsim Checker "<< std::endl;
    std::cout << std::string(80, '=') << std::endl;

    /*--------------------------------------------------------------------------------*/
    /** Phase 0: Fast Rejection via Signature Histogram Comparison */
    /*--------------------------------------------------------------------------------*/
    std::cout << "\nPhase 0: Signature Histogram Comparison" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    auto p0_start = std::chrono::steady_clock::now();

    bool phase0_pass = GPU_CompareSignatureCountsBetweenGraphs();

    if (!phase0_pass)
    {
        std::cout << "Phase0: Signature histograms differ" << std::endl;
        return false;
    }
    std::cout << " Phase0: Signatures match " << std::endl;


    /*--------------------------------------------------------------------------------*/
	/** Phase 1: Port-Ordered Edge Hash Verification Fail fast on structural edge mismatches */
	/*--------------------------------------------------------------------------------*/
        std::cout << "\nPhase1: Port-Ordered Edge Verification" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        auto p2_start = std::chrono::steady_clock::now();
        if (m_numEdges[0] > 0)
        {

            if (!GPU_CompareEdgesSignaturesBetweenGraphs())
            {
                std::cout << "Edge signatures differ" << std::endl;
                return false;
            }
            std::cout << " [PASS] Edge hashes match" << std::endl;
        }
        else
        {
            std::cout << " [SKIP] No edges to verify. Proceeding to WL-1 Node Refinement " << std::endl;
        }
   /*--------------------------------------------------------------------------------*/

	/*--------------------------------------------------------------------------------*/
	/** Phase 2: WL-1 Iterative Refinement (Node Color Refinement) */
	/*--------------------------------------------------------------------------------*/
	std::cout << "\nPhase2: WL-1 Node Color Refinement" << std::endl;
	std::cout << std::string(80, '-') << std::endl;
	auto p1_start = std::chrono::steady_clock::now();

	const int WL1_ITER = 100;
	/* Run refinement on Graph 0 */
	if (!GPU_WL1GraphColorHashIT(0, WL1_ITER)) return false;
	/* Run refinement on Graph 1 */
	if (!GPU_WL1GraphColorHashIT(1, WL1_ITER)) return false;

	/* Check if WL-1 histograms match */
	if (!WL_CompareBinCountsInitState())
	{
		std::cout << " Node color histograms differ after WL-1" << std::endl;
		return false;
	}
	std::cout << " Phase2:  WL-1 stable and matching" << std::endl;

	/*--------------------------------------------------------------------------------*/

    /*--------------------------------------------------------------------------------*/
	/** Phase 3: WL-2 Matrix Construction and Refinement  */
	/*--------------------------------------------------------------------------------*/
	std::cout << "\nPhase3: WL-2 Matrix Construction" << std::endl;
	std::cout << std::string(80, '-') << std::endl;

    auto p3_start = std::chrono::steady_clock::now();
	/* Initialize Matrices */
	if (!GPU_WL2_InitMatrix(0)) return false;
	if (!GPU_WL2_InitMatrix(1)) return false;

	/* Run Unified Refinement to prevent hash drift */
	/* This ensures G0 and G1 hashes map to the SAME integers */
	const int WL2_ITER = 100;
	if (!WL2_RefineBothGraphsUnified(m_numNodes[0], dram_WL2_MatrixColors[0], dram_WL2_MatrixColors[1], WL2_ITER))
	{
		 std::cout << " Error: Unified Refining Failed Phase3 Exit " << std::endl;
		return false;
	}

	/* Verify Node Profiles Match Between graphs */
	if (!WL2_CheckNodeIsomorphismStatus(m_numNodes[0], dram_WL2_MatrixColors[0], dram_WL2_MatrixColors[1]))
	{
		 std::cout << " Node Profiles differ after WL-2 Error " << std::endl;
		 /* Dump Logs for Debugging */
		 WriteWL2Log(0, m_numNodes[0], dram_WL2_MatrixColors[0], "dump_g0_wl2_fail.txt");
		 WriteWL2Log(1, m_numNodes[1], dram_WL2_MatrixColors[1], "dump_g1_wl2_fail.txt");
		 return false;
	}
	std::cout << " Phase3: WL-2 matrices stable and matching" << std::endl;
	/*--------------------------------------------------------------------------------*/


    /*--------------------------------------------------------------------------------*/
    /** Phase 3.5: WL-3 Triplet Refinement, Conditional */
    /*--------------------------------------------------------------------------------*/
    /* Only run for moderate sizes (N < 5000) where matrix is ambiguous */
    int numNodes = m_numNodes[0];
    auto p35_start = std::chrono::steady_clock::now();
    bool ran_wl3 = false;

    if (numNodes < 1000)
    {
            /* Analyze Symmetry of G0 */
            uint64_t *d_Scratch;
            cudaMalloc(&d_Scratch, numNodes * numNodes * sizeof(uint64_t));
            WL3SymmetryProfile profile0 = WL3_AnalyzeSymmetry(numNodes, dram_WL2_MatrixColors[0], d_Scratch);
            cudaFree(d_Scratch);

            if (profile0.symmetry_score > 0.05 && profile0.largest_bin > 1)
            {
                std::cout << "\nPhase 3.5: WL-3 Triplet Refinement (High Symmetry Detected)" << std::endl;
                std::cout << std::string(80, '-') << std::endl;

                /* Run independent triplet injection */
                bool improved_g0 = WL3_GraphTripletColoring(0, 20);
                bool improved_g1 = WL3_GraphTripletColoring(1, 20);
                ran_wl3 = true;

                if (improved_g0 != improved_g1) {
                    std::cout << "WL-3 improved one graph but not the other" << std::endl;
                    return false;
                }
                /* Re-sync the graphs to ensure the new triplet colors match exactly */
                std::cout << "Re-Syncing color spaces after WL-3" << std::endl;
                if (!WL2_RefineBothGraphsUnified(m_numNodes[0], dram_WL2_MatrixColors[0], dram_WL2_MatrixColors[1], WL2_ITER))
                {
                	std::cout << " Error: Unified Refining Failed Phase3.5 Exit " << std::endl;
                    return false;
                }
            }
    }
    if(!ran_wl3) std::cout << "\nPhase 3.5: Skipped WL-2 partition is sufficient " << std::endl;
    /*--------------------------------------------------------------------------------*/


    /*--------------------------------------------------------------------------------*/
    /** Phase 4: Individualization-Refinement (IR) Search which should be Definite */
    /*--------------------------------------------------------------------------------*/
    std::cout << "\nPhase 4: IR Search with Symmetry Pruning" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    auto p4_start = std::chrono::steady_clock::now();

    /* Adaptive Search Parameters */
    int max_depth   = 100;
    int max_pivots  = 100000;
    int timeout_sec = 300;
    if (numNodes > 1000) { max_pivots = 1000000; timeout_sec = 600; }

    bool isIsomorphic = GPU_WL2Refinement_IsIso(max_depth, max_pivots, timeout_sec); /* This is a loop that will return a final answer */
    /*--------------------------------------------------------------------------------*/



    /*--------------------------------------------------------------------------------*/
    /** Result */
    auto total_end = std::chrono::steady_clock::now();
    long total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "FINAL RESULT: " << (isIsomorphic ? "ISOMORPHIC :-)" : "NOT ISOMORPHIC :-(") << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    std::cout << "Pipeline Timing:" << std::endl;
    std::cout << "  Phase 0 (Sig):   " << std::setw(6) << std::chrono::duration_cast<std::chrono::milliseconds>(p1_start - p0_start).count() << " ms" << std::endl;
    std::cout << "  Phase 1 (Edge):  " << std::setw(6) << std::chrono::duration_cast<std::chrono::milliseconds>(p3_start - p2_start).count() << " ms" << std::endl;
    std::cout << "  Phase 2 (WL-1):  " << std::setw(6) << std::chrono::duration_cast<std::chrono::milliseconds>(p2_start - p1_start).count() << " ms" << std::endl;
    std::cout << "  Phase 3 (WL-2):  " << std::setw(6) << std::chrono::duration_cast<std::chrono::milliseconds>(p35_start - p3_start).count() << " ms" << std::endl;
    if(ran_wl3) std::cout << "  Phase 3.5 (WL3): " << std::setw(6) << std::chrono::duration_cast<std::chrono::milliseconds>(p4_start - p35_start).count() << " ms" << std::endl;
    std::cout << "  Phase 4 (IR):    " << std::setw(6) << std::chrono::duration_cast<std::chrono::milliseconds>(total_end - p4_start).count() << " ms" << std::endl;
    std::cout << "  TOTAL TIME:      " << std::setw(6) << total_ms << " ms" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    /*--------------------------------------------------------------------------------*/

	cudaDeviceSynchronize();
	cudaCheckError();
    return isIsomorphic;
}
/*===================================================================================================================*/


/*================================================================================================*/
/* Determinism Verification                                                          */
/* Runs the exact same isomorphism check repeatedly to catch race conditions                     */
/* If the hash is truly deterministic, it must pass 100% of the time                             */
/*================================================================================================*/
void RunDeterminismStressTest(int iterations)
{
    std::cout << "\n" << std::string(80, '#') << std::endl;
    std::cout << "Starting Determinism Stress Test " << iterations << " iterations" << std::endl;
    std::cout << std::string(80, '#') << std::endl;

    int failures = 0;

    for (int i = 0; i < iterations; i++)
    {
        std::cout << "\n>>> IT " << (i + 1) << " / " << iterations << " <<<" << std::endl;

        /* 1. Run the Solver */
        bool result = GPU_CheckHypergraphIsomorphism();

        /* 2. Check Result */
        if (!result)
        {
            std::cout << " Failed: Race Condition Detected! Run " << (i + 1) << " returned NOT ISOMORPHIC" << std::endl;
            failures++;
            /* Optional: Break immediately on failure to debug */
            // break;
        }
        else
        {
            std::cout << " Passed Run " << (i + 1) << " matches" << std::endl;
        }
        GPU_FreeWLBins(); // Clears WL-1 Histograms

        for(int g=0; g<2; g++)
        {
            /* Free WL-1 Node Hashes */
            if (m_isWL1Alloc[g]) {
                cudaFree(dram_NodeColorHashes[g]);
                m_isWL1Alloc[g] = false;
            }

            /* Free WL-2 Dense Matrices */
            if (m_isWL2Alloc[g]) {
                cudaFree(dram_WL2_MatrixColors[g]);
                m_isWL2Alloc[g] = false;
            }
        }

        /* Reset Bin Counts */
        m_WL_BinCount[0] = 0;
        m_WL_BinCount[1] = 0;

        cudaDeviceSynchronize();
    }

    std::cout << "\n" << std::string(80, '#') << std::endl;
    if (failures == 0)
    {
        std::cout << " All Good Passed " << iterations << "/" << iterations << " runs |  Solver is Deterministic" << std::endl;
    }
    else
    {
        std::cout << " Error Solver failed " << failures << " times Race conditions persist" << std::endl;
    }
    std::cout << std::string(80, '#') << std::endl;
}
