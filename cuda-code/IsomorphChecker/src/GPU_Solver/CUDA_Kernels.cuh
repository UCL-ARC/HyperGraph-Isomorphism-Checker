/*
 * CUDA_Kernels.cuh
 *
 *  Created on: Nov 4, 2025
 *      Author: Nicolin Govender UCL-ARC
 */

#ifndef GPU_SOLVER_CUDA_KERNELS_CUH_
#define GPU_SOLVER_CUDA_KERNELS_CUH_
#include <thrust/tuple.h>

static_assert(sizeof(NodeKeyTuple) == 16, "Structure contains padding! Unsafe for raw hashing ");

#define FNV_OFFSET_BASIS 0xcbf29ce484222325ULL
#define FNV_PRIME 0x100000001b3ULL

/*-----------------------------------------------------------------------------------------------*/
/* Test Kernel */
/*-----------------------------------------------------------------------------------------------*/
__global__ void printItem(uint *CSR, uint *Start, uint *Num, uint *indexLabel, uint MaxArraySize, uint numPrint)
{
	uint tid = blockIdx.x*blockDim.x  + threadIdx.x; /* Gives the ThreadIndex */

	/* Check we are in memory range */
	if(tid<MaxArraySize)
    {
		printf("Item %d LabelInd %d,  StartNum %d %d \n", tid,indexLabel[tid], Start[tid] , Num[tid] );

		uint start =  Start[tid];

		for(int i=start;i<start+Num[tid];i++)
		{
			printf(" %d ", CSR[i] );
		}
		printf("\n");
    }

}
/*-----------------------------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------------------------*/
/* Debug prints the edge hashes that do not match */
/*-----------------------------------------------------------------------------------------------*/
__global__ void PrintEdge(  int edge_id, const uint* d_edge_LabelA,


									const NodeKeyTuple *d_node_sig,

									const uint* d_edge_SourceNodes,
									const uint* d_edge_SourcesStart,
									const uint* d_edge_SourcesNum,

									const uint* d_edge_TargetNodes,
									const uint* d_edge_TargetsStart,
									const uint* d_edge_TargetsNum,
									NodeKeyTuple MAX_TUPLE)
{



	 printf("MaxTupe %u %u %u %u \n",
	        thrust::get<0>(MAX_TUPLE),
	        thrust::get<1>(MAX_TUPLE),
	        thrust::get<2>(MAX_TUPLE),
	        thrust::get<3>(MAX_TUPLE)
	 );
				printf(" %d  LabelIndA %d \n", edge_id, d_edge_LabelA[edge_id]  );

				printf(" SourceNodes \n");
				uint start =  d_edge_SourcesStart[edge_id];
				for(int i=start;i<start+d_edge_SourcesNum[edge_id];i++)
				{
					NodeKeyTuple sig = d_node_sig[d_edge_SourceNodes[i]];
					printf(" Node %d Sig %d %d %d %d \n", d_edge_SourceNodes[i],
							   thrust::get<0>(sig),
		                       thrust::get<1>(sig),
		                       thrust::get<2>(sig),
		                       thrust::get<3>(sig) );
				}
				printf("\n");

				printf(" TargetNodes \n");
				start =  d_edge_TargetsStart[edge_id];
				for(int i=start;i<start+d_edge_TargetsNum[edge_id];i++)
				{
					NodeKeyTuple sig = d_node_sig[d_edge_SourceNodes[i]];
										printf(" Node %d Sig %d %d %d %d \n", d_edge_TargetNodes[i],
												   thrust::get<0>(sig),
							                       thrust::get<1>(sig),
							                       thrust::get<2>(sig),
							                       thrust::get<3>(sig) );
				}
				printf("\n");


}
/*-----------------------------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------------------------*/
/** 0] Fast Computes a 64bit hash Fowler–Noll–Vo "a" variant */
/*-----------------------------------------------------------------------------------------------*/
__device__ uint64_t fnv1a_hash_64(const void* data, size_t num_bytes)
{
    uint64_t hash = FNV_OFFSET_BASIS;
    const unsigned char* bytes = (const unsigned char*)data;

    for (size_t i = 0; i < num_bytes; ++i)
    {
        hash = hash ^ bytes[i];
        hash = hash * FNV_PRIME;
    }

    return hash;
}
/*-----------------------------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------------------------*/
/** 0] SplitMix64 variant for unsorted input */
/*-----------------------------------------------------------------------------------------------*/
__device__ __forceinline__ uint64_t mix_color(uint64_t k)
{
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}
/*-----------------------------------------------------------------------------------------------*/


/*-----------------------------------------------------------------------------------------------*/
/** 0.1] Mixes a single NodeKeyTuple into an existing 64-bit hash */
__device__ __forceinline__ uint64_t hash_update_tuple(uint64_t hash, const NodeKeyTuple& t)
{
    // Extract values to registers (ignoring any padding in the struct)
    uint32_t data[4];
    data[0] = thrust::get<0>(t); // Label
    data[1] = thrust::get<1>(t); // IO
    data[2] = thrust::get<2>(t); // NumNexts
    data[3] = thrust::get<3>(t); // NumPrevs

    // FNV-1a Mix for these 16 bytes
    // We unroll the byte loop manually for speed
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t val = data[i];

        // Byte 0
        hash ^= (val & 0xFF);
        hash *= FNV_PRIME;
        // Byte 1
        hash ^= ((val >> 8) & 0xFF);
        hash *= FNV_PRIME;
        // Byte 2
        hash ^= ((val >> 16) & 0xFF);
        hash *= FNV_PRIME;
        // Byte 3
        hash ^= ((val >> 24) & 0xFF);
        hash *= FNV_PRIME;
    }
    return hash;
}
/*-----------------------------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------------------------*/
/** 0.2] Fast 64bit hash of two uint64_t hashes Fowler–Noll–Vo "a" variant */
__device__ inline uint64_t hash_pair(uint64_t a, uint64_t b)
{
    uint64_t hash = FNV_OFFSET_BASIS;

    /** Hash first integer (A) byte by byte */
    unsigned char* p = (unsigned char*)&a;
    for (int i=0; i<8; i++)
    {
        hash = hash ^ p[i];
        hash = hash * FNV_PRIME;
    }

    /**  Hash second integer (B) byte by byte */
    p = (unsigned char*)&b;
    for (int i=0; i<8; i++)
    {
        hash = hash ^ p[i];
        hash = hash * FNV_PRIME;
    }
    return hash;
}
/*-----------------------------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------------------------*/
/** 1] Fixed Network ordering for Max 8 elements */
/*-----------------------------------------------------------------------------------------------*/
template <typename T>
__device__ __forceinline__ void compareAndSwap(T& a, T& b)
{
    if (b < a)
    {
        T temp = a;
        a = b;
        b = temp;
    }
}
template <typename T>
__device__ void sorting_network_8(T data[8])
{
    // Stage 1 (Odd)
    compareAndSwap(data[1], data[2]);
    compareAndSwap(data[3], data[4]);
    compareAndSwap(data[5], data[6]);
    // Stage 2 (Even)
    compareAndSwap(data[0], data[1]);
    compareAndSwap(data[2], data[3]);
    compareAndSwap(data[4], data[5]);
    compareAndSwap(data[6], data[7]);
    // Stage 3 (Odd)
    compareAndSwap(data[1], data[2]);
    compareAndSwap(data[3], data[4]);
    compareAndSwap(data[5], data[6]);
    // Stage 4 (Even)
    compareAndSwap(data[0], data[1]);
    compareAndSwap(data[2], data[3]);
    compareAndSwap(data[4], data[5]);
    compareAndSwap(data[6], data[7]);
    // Stage 5 (Odd)
    compareAndSwap(data[1], data[2]);
    compareAndSwap(data[3], data[4]);
    compareAndSwap(data[5], data[6]);
    // Stage 6 (Even)
    compareAndSwap(data[0], data[1]);
    compareAndSwap(data[2], data[3]);
    compareAndSwap(data[4], data[5]);
    compareAndSwap(data[6], data[7]);
    // Stage 7 (Odd)
    compareAndSwap(data[1], data[2]);
    compareAndSwap(data[3], data[4]);
    compareAndSwap(data[5], data[6]);
    // Stage 8 (Even)
    compareAndSwap(data[0], data[1]);
    compareAndSwap(data[2], data[3]);
    compareAndSwap(data[4], data[5]);
    compareAndSwap(data[6], data[7]);
}
/*-----------------------------------------------------------------------------------------------*/


/*-----------------------------------------------------------------------------------------------*/
/* 1.1] Returns true if A > B (Lexicographical order) */
__device__ __forceinline__ bool greater_than(const NodeKeyTuple& a, const NodeKeyTuple& b)
{
    // Compare Label (Index 0)
    if (thrust::get<0>(a) != thrust::get<0>(b))
        return thrust::get<0>(a) > thrust::get<0>(b);

    // Compare IO (Index 1)
    if (thrust::get<1>(a) != thrust::get<1>(b))
        return thrust::get<1>(a) > thrust::get<1>(b);

    // Compare NumNexts (Index 2)
    if (thrust::get<2>(a) != thrust::get<2>(b))
        return thrust::get<2>(a) > thrust::get<2>(b);

    // Compare NumPrevs (Index 3)
    return thrust::get<3>(a) > thrust::get<3>(b);
}

__device__ void sort_local_buffer(NodeKeyTuple* arr, int n)
{
    // Simple Bubble Sort
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (greater_than(arr[j], arr[j + 1])) {
                // Swap
                NodeKeyTuple temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}
/*-----------------------------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------------------------*/
/* A] TODO Example for UCL Upskilling Both source and target nodes: For each edge sorts the nodes and creates a 64 bit hash using Max 8 network sort Canonical Order Unique Hash*/
/*-----------------------------------------------------------------------------------------------*/
__global__ void Kernel_EdgeHashesSortedBound(    int numEdges, // Number of threads

												const uint* __restrict__ d_edge_SourceNodes,
												const uint* __restrict__ d_edge_TargetNodes,

												const uint* __restrict__ d_edge_SourcesStart,
												const uint* __restrict__ d_edge_SourcesNum,

												const uint* __restrict__  d_edge_TargetsStart,
												const uint*  __restrict__ d_edge_TargetsNum,

												const NodeKeyTuple* __restrict__  d_node_sig, /* Color */

                                                uint64_t* edge_SourceNodeHashes,
                                                uint64_t* edge_TargetNodeHashes,

												NodeKeyTuple MAX_TUPLE    )
{
    // One thread per edge
    int edgeIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (edgeIdx < numEdges)
    {
		NodeKeyTuple NodeSigs[8]; /** Fixed local array */

		/*---------------------------------------------------------------------------------------------------*/
		/* A] Source Nodes Signatures */
		int srcStartIdx = d_edge_SourcesStart [edgeIdx];
		int srcNumNodes = d_edge_SourcesNum   [edgeIdx];

		/* Copy in local Array */
		for (int i = 0; i < 8; i++)
		{
			if( i<srcNumNodes)
			{
			  NodeSigs[i] = d_node_sig[ d_edge_SourceNodes[srcStartIdx + i] ];
			}
			else
			{
				NodeSigs[i] = MAX_TUPLE;
			}
		}

		/* B] Sort Nodes Signatures create a hash for the edge  */
		sorting_network_8(NodeSigs);
		if(srcNumNodes<8)
		{
			//printf(" %d numSourceNodes %d \n", edge_id, num_source_nodes);
			edge_SourceNodeHashes[edgeIdx] = fnv1a_hash_64( NodeSigs, 8*sizeof(NodeKeyTuple) );
		}
		else
		{
			printf (" Error Too many source nodes \n");

		}
        /*---------------------------------------------------------------------------------------------------*/

		/*---------------------------------------------------------------------------------------------------*/
		/* B] Target Nodes Signatures */
		int trgStartIdx = d_edge_TargetsStart [edgeIdx];
		int trgNumNodes = d_edge_TargetsNum   [edgeIdx];

		for (int i = 0; i < 8; i++)
		{
			if( i<trgNumNodes)
			{
				NodeSigs[i] =  d_node_sig[ d_edge_TargetNodes[ trgStartIdx + i ]];
			}
			else
			{
				NodeSigs[i] = MAX_TUPLE;
			}
		}

		sorting_network_8(NodeSigs);

		if(trgNumNodes<8)
		{
		   //printf(" %d numTargetNodes %d \n", edge_id, num_target_nodes);
			edge_TargetNodeHashes[edgeIdx] = fnv1a_hash_64(NodeSigs, 8*sizeof(NodeKeyTuple));
		}
		else
		{
		   printf (" Error Too Many target nodes \n");
		}
		/*---------------------------------------------------------------------------------------------------*/

    } /* Valid TID */
}
/*-----------------------------------------------------------------------------------------------*/

/*===================================================================================================================*/
/* A1] Same as A] but uses a rolling hash for few global memory reads */
/*===================================================================================================================*/
__global__ void Kernel_EdgeHashesSortedBoundFlat(
                                                int numEdges,
                                                const uint* __restrict__ d_edge_SourceNodes,
                                                const uint* __restrict__ d_edge_SourcesStart,
                                                const uint* __restrict__ d_edge_SourcesNum,

                                                const uint* __restrict__ d_edge_TargetNodes,
                                                const uint* __restrict__ d_edge_TargetsStart,
                                                const uint* __restrict__ d_edge_TargetsNum,

                                                const NodeKeyTuple* __restrict__ d_node_sig, /* Offset combined src and targets */

                                                uint64_t* edge_SourceNodeHashes,
                                                uint64_t* edge_TargetNodeHashes,

                                                NodeKeyTuple MAX_TUPLE)
{
    // One thread per edge
    int edgeIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (edgeIdx < numEdges)
    {
        // Register-allocated temporary buffer for sorting
        NodeKeyTuple NodeSigs[8];

        /*===========================================================================*/
        /* PART A: SOURCE NODES                                                      */
        /*===========================================================================*/
        int srcStartIdx = d_edge_SourcesStart[edgeIdx];
        int srcNumNodes = d_edge_SourcesNum[edgeIdx];

        /** Load source node siguatures */
        #pragma unroll
        for (int i = 0; i < 8; i++)
        {
            if (i < srcNumNodes)
            {
                NodeSigs[i] = d_node_sig[d_edge_SourceNodes[srcStartIdx + i]];
            }
            else
            {
                NodeSigs[i] = MAX_TUPLE; // Padding for Sort
            }
        }
        // Sort (Canonical Order)
        sorting_network_8(NodeSigs);

        /** Hash the source nodes */
        if (srcNumNodes <= 8)
        {
            uint64_t running_hash = FNV_OFFSET_BASIS;

            // We hash ALL 8 slots (including MAX_TUPLE padding) to ensure
            // that an edge with 3 nodes is distinct from an edge with 4 nodes.
            #pragma unroll
            for (int i = 0; i < 8; i++)
            {
                running_hash = hash_update_tuple(running_hash, NodeSigs[i]);
            }
            edge_SourceNodeHashes[edgeIdx] = running_hash;
        }
        /* Error Fallback */
        else
        {
            edge_SourceNodeHashes[edgeIdx] = UINT64_MAX;
            printf("Error: Too many source nodes at Edge %d\n", edgeIdx);
        }

        /*===========================================================================*/
        /* PART B: TARGET NODES                                                      */
        /*===========================================================================*/
        int trgStartIdx = d_edge_TargetsStart[edgeIdx];
        int trgNumNodes = d_edge_TargetsNum[edgeIdx];

        // 1. Load Data
        #pragma unroll
        for (int i = 0; i < 8; i++)
        {
            if (i < trgNumNodes)
            {
                NodeSigs[i] = d_node_sig[d_edge_TargetNodes[trgStartIdx + i]];
            }
            else
            {
                NodeSigs[i] = MAX_TUPLE;
            }
        }

        // 2. Sort
        sorting_network_8(NodeSigs);

        // 3. Hash
        if (trgNumNodes <= 8)
        {
            uint64_t running_hash = FNV_OFFSET_BASIS;

            #pragma unroll
            for (int i = 0; i < 8; i++)
            {
                running_hash = hash_update_tuple(running_hash, NodeSigs[i]);
            }
            edge_TargetNodeHashes[edgeIdx] = running_hash;
        }
        else
        {
            edge_TargetNodeHashes[edgeIdx] = UINT64_MAX;
            // printf("Error: Too many target nodes at Edge %d\n", edgeIdx);
        }
    }
}
/*===================================================================================================================*/

/*-----------------------------------------------------------------------------------------------*/
/* A2] Same as A1] but handles edges >8 hashes up to 64 using a slower bubble sort  */
/*-----------------------------------------------------------------------------------------------*/
#define MAX_LOCAL_CAPACITY 64 // Upper limit for this kernel approach
__global__ void Kernel_EdgeHashesSortedBubble64L(
    int numEdges,
    const uint* __restrict__ d_edge_SourceNodes,
    const uint* __restrict__ d_edge_SourcesStart,
    const uint* __restrict__ d_edge_SourcesNum,

    const uint* __restrict__ d_edge_TargetNodes,
    const uint* __restrict__ d_edge_TargetsStart,
    const uint* __restrict__ d_edge_TargetsNum,

    const NodeKeyTuple* __restrict__ d_node_sig,

    uint64_t* edge_SourceNodeHashes,
    uint64_t* edge_TargetNodeHashes,

    NodeKeyTuple MAX_TUPLE)
{
    int edgeIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (edgeIdx < numEdges)
    {

        int srcStartIdx = d_edge_SourcesStart[edgeIdx];
        int srcNumNodes = d_edge_SourcesNum  [edgeIdx];

        if (srcNumNodes <= MAX_LOCAL_CAPACITY)
        {
            /** Allocate in Local Memory (L1 Cache)
            /* Note: This increases stack frame size but only used by threads that enter */

            NodeKeyTuple LocalSigs[MAX_LOCAL_CAPACITY];

            /** Load into larger local memory */
            for (int i = 0; i < srcNumNodes; i++)
            {
                LocalSigs[i] = d_node_sig[d_edge_SourceNodes[srcStartIdx + i]];
            }

            // 2. Sort (Bubble)
            sort_local_buffer(LocalSigs, srcNumNodes);

            // 3. Hash
            uint64_t running_hash = FNV_OFFSET_BASIS;
            for (int i = 0; i < srcNumNodes; i++)
            {
                running_hash = hash_update_tuple(running_hash, LocalSigs[i]);
            }
            edge_SourceNodeHashes[edgeIdx] = running_hash;
        }
        else
        {
            // Mark for CPU or specialized kernel handling
        	edge_SourceNodeHashes[edgeIdx] = UINT64_MAX;
            printf("Overflow Src Edge %d: %d nodes\n", edgeIdx, srcNumNodes);
        }


        /* Same logic on Target nodes */
        int trgStartIdx = d_edge_TargetsStart[edgeIdx];
        int trgNumNodes = d_edge_TargetsNum[edgeIdx];

        if (trgNumNodes <= MAX_LOCAL_CAPACITY)
        {
            NodeKeyTuple LocalSigs[MAX_LOCAL_CAPACITY];
            for (int i = 0; i < trgNumNodes; i++)
            {
                LocalSigs[i] = d_node_sig[d_edge_TargetNodes[trgStartIdx + i]];
            }
            sort_local_buffer(LocalSigs, trgNumNodes);

            uint64_t running_hash = FNV_OFFSET_BASIS;
            for (int i = 0; i < trgNumNodes; i++)
            {
                running_hash = hash_update_tuple(running_hash, LocalSigs[i]);
            }
            edge_TargetNodeHashes[edgeIdx] = running_hash;
        }
        else
        {
        	edge_TargetNodeHashes[edgeIdx] = UINT64_MAX;
        	printf("Overflow Trg Edge %d: %d nodes\n", edgeIdx, trgNumNodes);
        }
    }
}
/*-----------------------------------------------------------------------------------------------*/

/*===================================================================================================================*/
/** A3] Weak Hash for color but sorting not needed for edges with large node numbers  */
/* KERNEL: Commutative Edge Hashing (Sources & Targets)                                          */
/* Logic: Linearly scans neighbors and accumulates their hashes.                                 */
/* Uses SUM + XOR mixing to guarantee order-independence (No Sorting required)           */
/*===================================================================================================================*/
__global__ void Kernel_EdgeHashes_MixHashNoSort( int numEdges,

														// Source Inputs
														const uint* __restrict__ d_edge_SourceNodes,
														const uint* __restrict__ d_edge_SourcesStart,
														const uint* __restrict__ d_edge_SourcesNum,

														// Target Inputs
														const uint* __restrict__ d_edge_TargetNodes,
														const uint* __restrict__ d_edge_TargetsStart,
														const uint* __restrict__ d_edge_TargetsNum,

														// Node Signatures
														const NodeKeyTuple* __restrict__ d_node_sig,

														// Outputs
														uint64_t* edge_SourceNodeHashes,
														uint64_t* edge_TargetNodeHashes)
{
    int edgeIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (edgeIdx < numEdges)
    {

    	/* Source Nodes */
        int start = d_edge_SourcesStart[edgeIdx];
        int num   = d_edge_SourcesNum[edgeIdx];

        uint64_t sum_acc = 0;
        uint64_t xor_acc = 0;

        for (int i = 0; i < num; i++)
        {
            /** 1] Fetch */
            NodeKeyTuple sig = d_node_sig[d_edge_SourceNodes[start + i]];

            /** 2] Hash (Treat as independent item by resetting seed) */
            uint64_t node_h = hash_update_tuple(FNV_OFFSET_BASIS, sig);

            /** 3] Accumulate Commutatively */
            sum_acc += node_h;
            xor_acc ^= mix_color(node_h); // Mix to prevent XOR self-cancellation
        }

        /** 4] Finalize Source Hash */
        edge_SourceNodeHashes[edgeIdx] = hash_pair(sum_acc, xor_acc);


       /* Target Nodes */
        start = d_edge_TargetsStart[edgeIdx];
        num   = d_edge_TargetsNum[edgeIdx];

        sum_acc = 0;
        xor_acc = 0;

        for (int i = 0; i < num; i++)
        {
            NodeKeyTuple sig = d_node_sig[d_edge_TargetNodes[start + i]];
            uint64_t node_h = hash_update_tuple(FNV_OFFSET_BASIS, sig);
            sum_acc += node_h;
            xor_acc ^= mix_color(node_h);
        }
        edge_TargetNodeHashes[edgeIdx] = hash_pair(sum_acc, xor_acc);
    }
}
/*===================================================================================================================*/

                                             /* WL Color Test Start */


/*===================================================================================================================*/
/** WL1: Canonical Order Unique Hash */
__global__ void Kernel_InitNodeColorWL1( int numNodes, const NodeKeyTuple*   __restrict__  d_nodeKey, uint64_t *nodeColors )
{
    int nodeIdx = blockIdx.x*blockDim.x + threadIdx.x;

    if (nodeIdx < numNodes)
    {
        /* Hash the 16-byte tuple to a single 64-bit color */
    	NodeKeyTuple sig    = d_nodeKey[nodeIdx];
    	nodeColors[nodeIdx] = fnv1a_hash_64(&sig, sizeof(NodeKeyTuple));
    }
}
/*===================================================================================================================*/

/*===================================================================================================================*/
/** WL1: Non Unique Hash unsorted */
/*===================================================================================================================*/
__global__ void Kernel_InitEdgeColorWL1( int numEdges, const uint*  __restrict__  d_edge_labels, uint64_t *nodeColors )
{
    int edgeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (edgeIdx < numEdges)
    {
    	nodeColors[edgeIdx] = (uint64_t)d_edge_labels[edgeIdx];
    }
}
/*===================================================================================================================*/


/*===================================================================================================================*/
/** WL1: Color Iteration for edges  Max 8 nodes for target or source per edge  Canonical Order Unique Hash */
/*===================================================================================================================*/
__global__ void Kernel_EdgeColorsWL1_SortedBound(   int numEdges,
												const uint64_t* __restrict__ d_edge_Colors_Initial, // The static C0 color
												const uint*     __restrict__ d_edge_SourceNodes,
												const uint*     __restrict__ d_edge_SourcesStart,
												const uint*     __restrict__ d_edge_SourcesNum,

												const  uint64_t* __restrict__ d_node_Colors,         // Current neighbor colors
												const  uint*     __restrict__ d_edge_TargetNodes,
												const  uint*     __restrict__ d_edge_TargetsStart,
												const  uint*     __restrict__ d_edge_TargetsNum,
												uint64_t *edgeColors )
{
    int edgeIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (edgeIdx < numEdges)
    {
		uint64_t NodeNNColors[8]; /* Should fit in registers else manual unroll */

		/*----------------------------------------------*/
		/* A] Source Nodes Signatures */
		int start_index      = d_edge_SourcesStart [edgeIdx];
		int num_source_nodes = d_edge_SourcesNum   [edgeIdx];

		/* Fixed Loop unrolling */
		for (int i = 0; i < 8; i++)
		{
			if( i < num_source_nodes)
			{
				NodeNNColors[i] = d_node_Colors[ d_edge_SourceNodes[start_index + i] ];
			}
			else
			{
				NodeNNColors[i] = UINT64_MAX;
			}
		}
		sorting_network_8(NodeNNColors);
		uint64_t source_hash = fnv1a_hash_64(NodeNNColors, num_source_nodes*sizeof(uint64_t)); /* Unique Hash */
       /*----------------------------------------------*/

		/*----------------------------------------------*/
		/* B] Target Nodes Signatures */
		int start_indexT     = d_edge_TargetsStart [edgeIdx];
		int num_target_nodes = d_edge_TargetsNum   [edgeIdx];

		for (int i = 0; i < 8; i++)
		{
			if( i < num_target_nodes)
			{
				NodeNNColors[i] = d_node_Colors[ d_edge_TargetNodes[start_indexT + i] ];
			}
			else
			{
				NodeNNColors[i] = UINT64_MAX;
			}
		}
		sorting_network_8(NodeNNColors);
		uint64_t target_hash = fnv1a_hash_64(NodeNNColors, num_target_nodes*sizeof(uint64_t)); /* Unique Hash */
		/*----------------------------------------------*/

		/*----------------------------------------------*/
		/* C] Combine and create new color for edge */
		uint64_t combined_hash_data[3] = { d_edge_Colors_Initial[edgeIdx], source_hash, target_hash };
		edgeColors[edgeIdx] = fnv1a_hash_64(combined_hash_data, 3 * sizeof(uint64_t)); /* Unique Hash */
        /*----------------------------------------------*/
    }
}
/*===================================================================================================================*/


/*===================================================================================================================*/
/** WL-1 Node Color Update Kernel Unbounded loops over array of nodes nexts and prevs */
/*===================================================================================================================*/
__global__ void Kernel_NodeColorsWL1( int numNodes,
									const  __restrict__ uint64_t *d_node_Colors_Initial, // The static C0 color
									const  __restrict__ uint *d_node_edgePrevs,
									const  __restrict__ uint *d_node_edgePrevsStart,
									const  __restrict__ uint *d_node_edgePrevsNum,

									int    numEdges,
									const  __restrict__ uint *d_node_edgeNexts,
									const  __restrict__ uint *d_node_edgeNextsStart,
									const  __restrict__ uint *d_node_edgeNextsNum,
									const  __restrict__ uint64_t *d_edge_Colors, // Current neighbor colors

									uint64_t   *nodeColors                         )
{
	int nodeIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (nodeIdx < numNodes)
	{
		/* A] Node Prevs */
		int num_prev_edges = d_node_edgePrevsNum[nodeIdx];
		int start_index    = d_node_edgePrevsStart[nodeIdx];

		uint64_t prev_hash = 0;
		for (int i = 0; i < num_prev_edges; i++)
		{
			uint64_t neighbor_col = d_edge_Colors[ d_node_edgePrevs[start_index + i] ];
			prev_hash += mix_color(neighbor_col);
		}

		/* B] Node Nexts */
		int num_next_edges = d_node_edgeNextsNum[nodeIdx];
		int start_indexN   = d_node_edgeNextsStart[nodeIdx];

		uint64_t next_hash = 0;
		for (int i = 0; i < num_next_edges; ++i)
		{
			uint64_t neighbor_col = d_edge_Colors[ d_node_edgeNexts[start_indexN + i] ];
		    next_hash += mix_color(neighbor_col);
		}

		/* C] Write New Color */
		uint64_t combined_hash_data[3] = { d_node_Colors_Initial[nodeIdx], prev_hash, next_hash };
		nodeColors[nodeIdx] = fnv1a_hash_64(combined_hash_data, 3 * sizeof(uint64_t));  /* Unique Hash */
	}
}
/*===================================================================================================================*/


                                   /* Start of WL-2 Kernels */

/*===================================================================================================================*/
/** 1. Set All matrix elements to 0 */
/*===================================================================================================================*/
__global__ void Kernel_WL2_Init_Background(size_t num_pairs, uint64_t *MatrixEleColor )
{
    // Use size_t for index to prevent overflow on massive grids
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_pairs)
    {
    	MatrixEleColor[idx] = 0;
    }
}

/*-----------------------------------------------------------------------------------------------*/
/** 2. Set Diagonal: Each node updates its own cell (i, i) */
/*-----------------------------------------------------------------------------------------------*/
__global__ void Kernel_WL2_Init_Diagonal(int numNodes, const NodeKeyTuple*  __restrict__ d_node_keys, uint64_t *MatrixEleColor)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numNodes)
    {
        // 1. Retrieve the tuple
        NodeKeyTuple k = d_node_keys[idx];

        // 2. Hash the tuple (Safe because tuple is strictly 4x uints)
        uint64_t node_hash = fnv1a_hash_64(&k, sizeof(NodeKeyTuple));

        // 3. Store in Diagonal (Row idx, Col idx)
        size_t addr = (size_t)idx * (size_t)numNodes + (size_t)idx;

        MatrixEleColor[addr] = node_hash;
    }
}

/*===================================================================================================================*/
/* WL-2 Kernel: Each Edge does an atomic update of the hash for the nodes it is connects to */
/*===================================================================================================================*/
__global__ void Kernel_WL2_Init_HyperEdges_Brute(  int numEdges,
												   uint64_t *MatrixEleColor,

												   int numNodes,
												   const  __restrict__ uint *d_edge_sources,
												   const  __restrict__ uint *d_edge_sources_start,
												   const  __restrict__ uint *d_edge_sources_num,
												   const  __restrict__ uint *d_edge_targets,
												   const  __restrict__ uint *d_edge_targets_start,
												   const  __restrict__ uint *d_edge_targets_num,
												   const  __restrict__ uint *d_edge_labels          )
{
    int edgeIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (edgeIdx < numEdges)
    {
        uint label = d_edge_labels[edgeIdx];

        // 1. Hash the Edge Label
        uint64_t edge_hash = fnv1a_hash_64(&label, sizeof(uint));

        // 2. Get Source Range (Inputs)
        int srcStartIdx = d_edge_sources_start[edgeIdx];
        int srcNumNodes      = d_edge_sources_num[edgeIdx];

        // 3. Get Target Range (Outputs)
        int trgStartIdx = d_edge_targets_start[edgeIdx];
        int trgNumNodes = d_edge_targets_num[edgeIdx];

        // 4. Cartesian Product expansion brute force
        for (int s = 0; s < srcNumNodes; s++)
        {
            int u = d_edge_sources[srcStartIdx + s];

            for (int t = 0; t < trgNumNodes; t++)
            {
                int v = d_edge_targets[trgStartIdx + t];

                if (u < numNodes && v < numNodes)
                {
                    // Calculate address
                    uint64_t* addr = &MatrixEleColor[u * numNodes + v];

                    // SAFE: 64bitAtomic XOR mixes this edge's hash with existing hash value it is commutative
                    atomicXor((unsigned long long*)addr, (unsigned long long)edge_hash);
                }
            }
        }
    }
}
/*===================================================================================================================*/


/*===================================================================================================================*/
/* WL-2 Optimized Kernel: Warp-Centric Hyperedge Init */
/* ----------------------------------------------------------------------------------------------------------------- */
/* Optimization Strategy: */
/* 1. Warp-Level Parallelism: 32 threads work on one edge together. */
/* 2. Shared Memory Tiling: Caches Source/Target lists to avoid scattered global reads. */
/* 3. Fallback: Handles edges larger than the cache with direct global memory reads */
/*===================================================================================================================*/

#define WARP_SIZE       32
#define WARPS_PER_BLOCK 8        // 256 threads / 32 = 8 warps
#define MAX_CACHESIZE   64       // 64 per wrap = 512 32bit ints per warp
/* Kernel Launches numEdges*32 threads  */
__global__ void Kernel_WL2_Init_HyperEdges_WarpOptimized(  int numEdges,
														   uint64_t *MatrixEleColor,
														   int numNodes,

														   const  __restrict__ uint *d_edge_sources,
														   const  __restrict__ uint *d_edge_sources_start,
														   const  __restrict__ uint *d_edge_sources_num,
														   const  __restrict__ uint *d_edge_targets,
														   const  __restrict__ uint *d_edge_targets_start,
														   const  __restrict__ uint *d_edge_targets_num,
														   const  __restrict__ uint *d_edge_labels)
{

    int warpInBlockIdx = threadIdx.x / WARP_SIZE; /** The wrap Index local to the block it is in */
    int warpThreadIdx  = threadIdx.x % WARP_SIZE; /** Thread Index local to the Wrap 0-31 */

    /** Global Edge Index: Block Offset + Local Warp Offset */
    int edgeIdx      = blockIdx.x * WARPS_PER_BLOCK + warpInBlockIdx;

    /** Shared Memory Allocation (Per Warp) [WarpID][CacheIndex] */
    __shared__ int smem_src[WARPS_PER_BLOCK][MAX_CACHESIZE];
    __shared__ int smem_tgt[WARPS_PER_BLOCK][MAX_CACHESIZE];

    /** Check that the edge index we are accessing is valid */
    if (edgeIdx < numEdges)
    {
    	/*-----------------------------------------------*/
        /*  warpThreadIdx 0 reads global memory and writes into shared memory */
        uint label, numSnodes, numTnodes, srcStartIdx, tgtStartIdx;
        if (warpThreadIdx == 0)
        {
            label       = d_edge_labels        [edgeIdx];
            srcStartIdx = d_edge_sources_start [edgeIdx];
            numSnodes   = d_edge_sources_num   [edgeIdx];

            tgtStartIdx = d_edge_targets_start [edgeIdx];
            numTnodes   = d_edge_targets_num   [edgeIdx];
        }
        /* Broadcasts to the warp using Shuffle and sync so it is a blocking call */
        // Sync ensures everyone gets the counters
        label       = __shfl_sync(0xFFFFFFFF, label,       0);
        numSnodes   = __shfl_sync(0xFFFFFFFF, numSnodes,   0);
        numTnodes   = __shfl_sync(0xFFFFFFFF, numTnodes,   0);
        srcStartIdx = __shfl_sync(0xFFFFFFFF, srcStartIdx, 0);
        tgtStartIdx = __shfl_sync(0xFFFFFFFF, tgtStartIdx, 0);
        /*-----------------------------------------------*/

        /* All threads compute hash based on the label */
        uint64_t edge_hash = fnv1a_hash_64(&label, sizeof(uint));


        /** Each thread writes into shared memory for max of 64 nodes it is two value per thread in the wrap
         *  if there are more nodes than 64 then they will be accessed directly from global memory */

        // Load Sources Max 64
        for (int i = warpThreadIdx; i < numSnodes && i < MAX_CACHESIZE; i += WARP_SIZE)
        {
            smem_src[warpInBlockIdx][i] = d_edge_sources[srcStartIdx + i];
        }

        // Load Targets Max 64
        for (int i = warpThreadIdx; i < numTnodes && i < MAX_CACHESIZE; i += WARP_SIZE)
        {
            smem_tgt[warpInBlockIdx][i] = d_edge_targets[tgtStartIdx + i];
        }

        /** Wait for all loads to finish */
        __syncwarp();

        /**  Each Edge Warp (32 threads) will compute its product within its own matrix of numSNodes*numTnodes and make the same number of atomic update */
        int total_pairs = numSnodes * numTnodes;
        for (int i = warpThreadIdx; i < total_pairs; i += WARP_SIZE)
        {
            /** Decode 1D index back to 2D */
            int s_idx = i / numTnodes;
            int t_idx = i % numTnodes;

            int u, v;

            /**  Fetch Source 'u' from shared mem if it is within cache limit */
            if (s_idx < MAX_CACHESIZE)
            {
                u = smem_src[warpInBlockIdx][s_idx];
            }
            /** Global Memory fetch */
            else
            {
                u = d_edge_sources[srcStartIdx + s_idx];
            }

            /**  Fetch Target 'v' from shared mem if it is within cache limit */
            if (t_idx < MAX_CACHESIZE)
            {
                v = smem_tgt[warpInBlockIdx][t_idx];
            }
            else
            {
                v = d_edge_targets[tgtStartIdx + t_idx];
            }

            /* Atomic Update */
            if (u < numNodes && v < numNodes)
            {
                // AtomicXor is required for hyper edges
            	size_t addr = (size_t)u * (size_t)numNodes + (size_t)v;
            	atomicXor((unsigned long long*)&MatrixEleColor[addr], (unsigned long long)edge_hash);
            }
        }
    }
}
/*===================================================================================================================*/



/*===================================================================================================================*/
/**
 * Kernel: WL2 Pair Refinement (Tiled Implementation) Per Edge Note: Other option is to do Sparse Matrix Mul
 * Updates the color of every pair of nodes (u, v) based on the colors of
 * all 2NN "2nd-Degree Nodes"  (u, w, v) connected to it
 *
 * Algorithm Strategy (Tiled Matrix Multiplication):
 * Similar to Matrix Multiplication (C = A * B), instead of multiply-add  we do Hash-Accumulate
 *
 * For every pair (u, v) Scan rows and colors for non zero Loop is O(N) in numNodes over a single Row :
 * 3. We hash them together: H = hash( C(u,w), C(w,v) )
 * 4. We accumulate these hashes into a single signature for node (u, v)
 *
 * Performance Note:
 * - Global Memory is slow
 * - We use "Tiling" to load a 16x16 block of colors into fast Shared Memory
 * - We reuse these 256 loaded values for 256 computations, reducing global
 * memory bandwidth pressure by a factor of 16 (TILE_WIDTH)
 */
/*===================================================================================================================*/

#define WARP_CACHESIZE 16 //Wrap Size = 16^2 = 256 x 2 = 512 64 bit uints:  must match block size
__global__ void Kernel_WL2_UpdatePairs_Tiled(int numNodes,
		                                     const uint64_t*  __restrict__  MatrixEleColor,
                                             uint64_t* MatrixEleColorWrite )
{
    int row = blockIdx.y * WARP_CACHESIZE + threadIdx.y; /** Node 'u' */
    int col = blockIdx.x * WARP_CACHESIZE + threadIdx.x; /** Node 'v' */


    /** Accumulator for the structural signature of pair (u, v) */
    uint64_t dotProdAccumulator = 0;

    __shared__ uint64_t tile_A[WARP_CACHESIZE][WARP_CACHESIZE];   /** tile_A stores a chunk of Row 'u' (interactions u -> w) */
    __shared__ uint64_t tile_B[WARP_CACHESIZE][WARP_CACHESIZE];   /** tile_B stores a chunk of Col 'v' (interactions w -> v) */


    /* m iterates across the intermediate dimension (all nodes 'w' in the graph) in shard mem chunks loading 16 cols and 16 rows at a time */
    for (int indexStride = 0; indexStride < (numNodes + WARP_CACHESIZE - 1) / WARP_CACHESIZE; ++indexStride)
    {

        /*-----------------------------------------------------------------------------------*/
        /** identifies which 'w' (column index) it needs to load for Tile A */
        int indexColA = indexStride * WARP_CACHESIZE + threadIdx.x;

        /**  Load interactions for (u, w) */
        if (row < numNodes && indexColA < numNodes)
        {
            tile_A[threadIdx.y][threadIdx.x] = MatrixEleColor[row * numNodes + indexColA];
        }
        else
        {
            tile_A[threadIdx.y][threadIdx.x] = 0; // Padding for boundary safety
        }

        /** A thread identifies which 'w' (row index) it needs to load for Tile B */
        int indexRowB = indexStride * WARP_CACHESIZE + threadIdx.y;

        /** Load interactions for (w, v) */
        if (indexRowB < numNodes && col < numNodes)
        {
            tile_B[threadIdx.y][threadIdx.x] = MatrixEleColor[indexRowB * numNodes + col];
        }
        else
        {
            tile_B[threadIdx.y][threadIdx.x] = 0;
        }

        /** Barrier: Wait for all threads in block to finish loading the tiles */
        __syncthreads();
        /*-----------------------------------------------------------------------------------*/

        /** Iterate through the loaded tile (intermediate nodes w_local) */
        #pragma unroll
        for (int k = 0; k < WARP_CACHESIZE; ++k)
        {
            uint64_t c1 = tile_A[threadIdx.y][k];            // c1 = Color(u, w)  [Row 'ty', Col 'k' inside tile A]
            uint64_t c2 = tile_B[k][threadIdx.x];            // c2 = Color(w, v)  [Row 'k', Col 'tx' inside tile B]

            /** Hash Combination:
             *  If we do u->w (color c1) and w->v (color c2), what does that path look like ?
             *  hash with  XOR/Mix this into the accumulator!
            /*  Note: if it is a valid 2nd degree connection then but c1 and C2 must be non-zero however for the WL2 test it just wants structure so 0 (missing element} gets hashed */

            if (c1 != 0 || c2 != 0) // Optimization: Skip empty space/padding
            {
                dotProdAccumulator += hash_pair(c1, c2);
            }
        }

        /** Barrier: Wait for everyone to finish reading before we overwrite tiles in the next loop*/
        __syncthreads();
    }

    /** Each thread Updates Global Memory */
    if (row < numNodes && col < numNodes)
    {
        /** WL Logic: NewColor(u,v) = Hash( OldColor(u,v), NeighborhoodHash ) */
        uint64_t my_old_color = MatrixEleColor[row * numNodes + col];
        MatrixEleColorWrite[row * numNodes + col] = hash_pair(my_old_color, dotProdAccumulator);
    }
}
/*===================================================================================================================*/



/*===================================================================================================================*/
/* Helper 1: Extract Node Colors from WL-2 Matrix Diagonal */
/* WL-2 stores Pair colors: The color of Node 'u' is stored at Matrix[u][u] */
/*===================================================================================================================*/
__global__ void Kernel_Extract_Diagonals(int numNodes,
		                                 const  __restrict__  uint64_t* MatrixEleColor,
                                         uint64_t* MatrixEleColorWrite)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < numNodes)
    {
        /* Diagonal index: Row u, Col u */
        size_t diag_idx = (size_t)u * numNodes + u;
        MatrixEleColorWrite[u] = MatrixEleColor[diag_idx];
    }
}
/*===================================================================================================================*/

/*===================================================================================================================*/
/* TODO NG WIP: Helper 2: Definite Permutation Verification */
/* Checks: "For every edge (u, v) in G1, does edge (Map[u], Map[v]) exist in G2?" */
/* Returns: 0 if mismatch found, 1 if exact match. */
/*===================================================================================================================*/
__global__ void Kernel_Verify_Isomorphism(int numEdges1,
                                          const uint* __restrict__ d_src1,
                                          const uint* __restrict__ d_tgt1,
                                          const int* __restrict__  d_map_G1_to_G2,
                                          const uint* __restrict__ d_row_ptr2,
                                          const uint* __restrict__ d_col_ind2,
                                          int* d_is_isomorphic)
{
    int edgeIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check flag at the start to skip work if another block already failed
    if (edgeIdx < numEdges1 && *d_is_isomorphic == 1)
    {

        int u1 = d_src1[edgeIdx];
        int v1 = d_tgt1[edgeIdx];

        /** 1] Translate to G2 Nodes */
        int u2 = d_map_G1_to_G2[u1];
        int v2 = d_map_G1_to_G2[v1];

        /** 2] Scan neighbors of u2 in G2 */
        bool found = false;
        int start = d_row_ptr2[u2];
        int end   = d_row_ptr2[u2 + 1];

        for (int i = start; i < end; i++)
        {
            // Optional: If neighbor lists are sorted, we can stop early
            // if (d_col_ind2[i] > v2) break;

            if (d_col_ind2[i] == v2)
            {
                found = true;
                break;
            }
        }

        /** 3] Update Flag safely */
        if (!found)
        {
            // Only write if the flag hasn't been flipped yet
            if (*d_is_isomorphic == 1) *d_is_isomorphic = 0;
        }
    }
}





/*===================================================================================================================*/
/*TODO NG WIP:  GPU Helper: Permute G1 Edges using the Map */
/* Logic: Converts edge (u, v) -> (Map[u], Map[v]) so it matches G2's numbering */
/*===================================================================================================================*/
__global__ void Kernel_Permute_Edges(size_t total_pairs,
                                     uint64_t* d_edges,
                                     const int* d_map)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total_pairs)
    {
        uint64_t packed = d_edges[tid];

        // Unpack High 32 (u) and Low 32 (v)
        uint u = (uint)(packed >> 32);
        uint v = (uint)(packed & 0xFFFFFFFF);

        // Apply Map P[u] -> u'
        // If map is -1 (error case), this will crash or produce garbage,
        // but our host code ensures map is valid before calling this.
        uint u_prime = (uint)d_map[u];
        uint v_prime = (uint)d_map[v];

        // Repack
        d_edges[tid] = ((uint64_t)u_prime << 32) | (uint64_t)v_prime;
    }
}

/*===================================================================================================================*/
/* TODO NG WIP: Optimized Expansion Kernel: Warp-Centric Tiling */
/* Logic: 1 Warp (32 threads) processes 1 Hyperedge */
/* Uses Shared Memory to cache Sources/Targets */
/* Writes (u,v) pairs into the flat array in parallel (Coalesced) */
/*===================================================================================================================*/

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8     // 256 threads / 32 = 8 warps
#define MAX_S_T_CACHE 64      // Typical Limit

__global__ void Kernel_Expand_HyperEdges_WarpOptimized(
                                         int numEdges,
                                         const uint* __restrict__ d_src_start,
                                         const uint* __restrict__ d_src_num,
                                         const uint* __restrict__ d_src_data,
                                         const uint* __restrict__ d_tgt_start,
                                         const uint* __restrict__ d_tgt_num,
                                         const uint* __restrict__ d_tgt_data,
                                         const uint* __restrict__ d_edge_write_offsets, // Calculated Prefix Sum
                                         uint64_t* d_flat_edges)                        // Output Array
{
    // 1. Identify Warp and Edge ID
    int warpInBlockIdx = threadIdx.x / WARP_SIZE;
    int warpThreadIdx  = threadIdx.x % WARP_SIZE;

    // Global Edge Index
    int edgeIdx = blockIdx.x * WARPS_PER_BLOCK + warpInBlockIdx;

    // 2. Shared Memory Tiling (Per Warp)
    __shared__ int smem_src[WARPS_PER_BLOCK][MAX_S_T_CACHE];
    __shared__ int smem_tgt[WARPS_PER_BLOCK][MAX_S_T_CACHE];

    if (edgeIdx < numEdges)
    {

    	/*--------------------------------------------------------------------*/
        /* 1st Thread Load limits for the warp */
        uint s_start, s_num, t_start, t_num, write_start_idx;
        if (warpThreadIdx == 0)
        {
            s_start         = d_src_start[edgeIdx];
            s_num           = d_src_num[edgeIdx];
            t_start         = d_tgt_start[edgeIdx];
            t_num           = d_tgt_num[edgeIdx];
            write_start_idx = d_edge_write_offsets[edgeIdx];
        }

        /* 1st Thread Send to others  */
        s_start         = __shfl_sync(0xFFFFFFFF, s_start, 0);
        s_num           = __shfl_sync(0xFFFFFFFF, s_num, 0);
        t_start         = __shfl_sync(0xFFFFFFFF, t_start, 0);
        t_num           = __shfl_sync(0xFFFFFFFF, t_num, 0);
        write_start_idx = __shfl_sync(0xFFFFFFFF, write_start_idx, 0);
        /*--------------------------------------------------------------------*/

        /*--------------------------------------------------------------------*/
        /* Everyone Load Data (Max 64) */
        for (int i = warpThreadIdx; i < s_num && i < MAX_S_T_CACHE; i += WARP_SIZE)
        {
            smem_src[warpInBlockIdx][i] = d_src_data[s_start + i];
        }

        for (int i = warpThreadIdx; i < t_num && i < MAX_S_T_CACHE; i += WARP_SIZE)
        {
            smem_tgt[warpInBlockIdx][i] = d_tgt_data[t_start + i];
        }
        __syncwarp();
       /*--------------------------------------------------------------------*/

        /*--------------------------------------------------------------------*/
        /* Write Back */
        int total_pairs = s_num * t_num;
        // Stride loop over the Cartesian Product
        for (int i = warpThreadIdx; i < total_pairs; i += WARP_SIZE)
        {
        	/* A] Decode 1D index -> 2D (s, t)*/
            int s_idx = i / t_num;
            int t_idx = i % t_num;

            int u, v;

            /* B] Fetch Source u (Hybrid Read: Cache vs Global) */
            if (s_idx < MAX_S_T_CACHE)
            {
            	u = smem_src[warpInBlockIdx][s_idx];
            }
            else
            {
            	u = d_src_data[s_start + s_idx];
            }

            /* C] Fetch Target v (Hybrid Read: Cache or Global) */
            if (t_idx < MAX_S_T_CACHE)
            {
            	v = smem_tgt[warpInBlockIdx][t_idx];
            }
            else
            {
            	v = d_tgt_data[t_start + t_idx];
            }
            /* D] Write to Output (Perfectly Coalesced) Per Wrap  write to sequential addresses */
            uint64_t packed = ((uint64_t)u << 32) | (uint64_t)v;
            d_flat_edges[write_start_idx + i] = packed;
        }
        /*--------------------------------------------------------------------*/
    }
}


#endif /* GPU_SOLVER_CUDA_KERNELS_CUH_ */
