/*
 * CUDA_KernelsWL1.cuh
 *
 *  Created on: Feb 4, 2026
 *      Author: blaze
 */

#ifndef ISOMORPHCHECKER_SRC_GPU_SOLVER_CUDA_KERNELSWL1_CUH_
#define ISOMORPHCHECKER_SRC_GPU_SOLVER_CUDA_KERNELSWL1_CUH_


/*===================================================================================================================*/
/** WA] WL1: Init Node Hash */
__global__ void Kernel_InitNodeHashWL1( int numNodes, const NodeKeyTuple*   __restrict__  d_nodeKey, uint64_t *nodeHashes )
{
	int nodeIdx = blockIdx.x*blockDim.x + threadIdx.x;

	if (threadIdx.x == 0 && blockIdx.x == 0) {
	    NodeKeyTuple debug_t = thrust::make_tuple(0xFFFFFFFF, 0xFFFFFFFF, 0, 0);

	    /* If packing works, packed_A should be 0xFFFFFFFFFFFFFFFF (all ones) */
	    uint64_t packed_A = (uint64_t)thrust::get<0>(debug_t) | ((uint64_t)thrust::get<1>(debug_t) << 32);

	    if (packed_A != 0xFFFFFFFFFFFFFFFFULL) {
	        printf("Critical: Tuple Packing Logic Failed! Result: %llx\n", packed_A);
	    }
	}

	    if (nodeIdx < numNodes)
	    {
	        NodeKeyTuple sig    = d_nodeKey[nodeIdx];

	        /* FNV_OFFSET_BASIS is seed */
	        nodeHashes[nodeIdx] = hashAddTuple(FNV_OFFSET_BASIS, sig);
	    }
}
/*===================================================================================================================*/

/*===================================================================================================================*/
/** WB] WL1: Init Edge Hash */
/*===================================================================================================================*/
__global__ void Kernel_InitEdgeHashWL1( int numEdges, const uint*  __restrict__  d_edge_labels, uint64_t *nodeHashes )
{
    int edgeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (edgeIdx < numEdges)
    {
    	//printf(" %d EdgeLabel %d \n",edgeIdx, d_edge_labels[edgeIdx]);
    	nodeHashes[edgeIdx] = (uint64_t)d_edge_labels[edgeIdx];
    }
}
/*===================================================================================================================*/

/*===================================================================================================================*/
/** WC] WL1: Compute Edge Hash (Port-Order Preserving)
 * We use Sum(Hash(Node) ^ Salt(PortIndex)) This ensures that Edge(A, B) != Edge(B, A) */
/*===================================================================================================================*/
__global__ void Kernel_EdgeColorsWL1_Hypergraph      (  int numEdges,
														const uint64_t* __restrict__ d_edge_HashesInit,

														/* Source Nodes */
														const uint* __restrict__ d_edge_SourceNodes,
														const uint* __restrict__ d_edge_SourcesStart,
														const uint* __restrict__ d_edge_SourcesNum,

														const uint64_t* __restrict__ d_node_Hashes,

														/* Target Nodes */
														const uint* __restrict__ d_edge_TargetNodes,
														const uint* __restrict__ d_edge_TargetsStart,
														const uint* __restrict__ d_edge_TargetsNum,

														uint64_t *edgeHashes)
{
    int edgeIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (edgeIdx < numEdges)
    {
        /*----------------------------------------------*/
        /* A] Process Source Nodes (Ordered Tuple)      */
        /*----------------------------------------------*/
        int start_src = d_edge_SourcesStart[edgeIdx];
        int num_src   = d_edge_SourcesNum[edgeIdx];

        uint64_t src_acc1 = 0;
        uint64_t src_acc2 = 0;

        for (int i = 0; i < num_src; i++)
        {
            uint64_t hash = d_node_Hashes[d_edge_SourceNodes[start_src + i]];

            /* Mix Port Index 'i' into the hash this distinguishes Arg0 from Arg1 */
            uint64_t port_salt = (uint64_t)i * 0x517cc1b727220a95ULL;

            /* Combine Node Hash + Position ID */
            uint64_t h = hashPair(hash, port_salt);

            /* Stream 1: Linear Sum of Position-Aware Hashes */
            src_acc1 += h;

            /* Stream 2: Scrambled Sum */
            src_acc2 += (h ^ PRIME_A) * PRIME_B;
        }
        uint64_t source_hash = hashPair(src_acc1, src_acc2);
        /*----------------------------------------------*/

        /*----------------------------------------------*/
        /* B] Process Target Nodes (Ordered Tuple)      */
        /*----------------------------------------------*/
        int start_tgt = d_edge_TargetsStart[edgeIdx];
        int num_tgt   = d_edge_TargetsNum[edgeIdx];

        uint64_t tgt_acc1 = 0;
        uint64_t tgt_acc2 = 0;

        for (int i = 0; i < num_tgt; i++)
        {
            uint64_t hash = d_node_Hashes[d_edge_TargetNodes[start_tgt + i]];

            /*  Mix Port Index 'i' */
            uint64_t port_salt = (uint64_t)i * 0x9e3779b97f4a7c15ULL;

            uint64_t h = hashPair(hash, port_salt);

            tgt_acc1 += h;
            tgt_acc2 += (h ^ PRIME_A) * PRIME_B;
        }

        uint64_t target_hash = hashPair(tgt_acc1, tgt_acc2);
        /*----------------------------------------------*/


        /*----------------------------------------------*/
        /* C] Final Mix                                 */
        /*----------------------------------------------*/
        uint64_t static_label_hash = d_edge_HashesInit[edgeIdx];

        uint64_t final_hash = hashPair(static_label_hash, source_hash);
        final_hash = hashPair(final_hash, target_hash);

        edgeHashes[edgeIdx] = final_hash;
        /*----------------------------------------------*/
    }
}
/*===================================================================================================================*/

/*===================================================================================================================*/
/** WD] WL-1 Node Color Update                                                      */
/*===================================================================================================================*/
__global__ void Kernel_NodeHashWL1( int numNodes,
        const  __restrict__ uint64_t *d_node_Hashes_Initial,
        const  __restrict__ uint *d_node_edgePrevs,
        const  __restrict__ uint *d_node_edgePrevsStart,
        const  __restrict__ uint *d_node_edgePrevsNum,

        int    numEdges,
        const  __restrict__ uint *d_node_edgeNexts,
        const  __restrict__ uint *d_node_edgeNextsStart,
        const  __restrict__ uint *d_node_edgeNextsNum,
        const  __restrict__ uint64_t *d_edge_Hashes,

        uint64_t   *nodeHashes                         )
{
	int nodeIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (nodeIdx < numNodes)
	{
		/*------------------------------------------------------------------------*/
		/* A] Node Prevs: Unordered Sum */
		int num_prev_edges = d_node_edgePrevsNum[nodeIdx];
		int start_index    = d_node_edgePrevsStart[nodeIdx];
		uint64_t prev_hash_acc = 0;
		for (int i = 0; i < num_prev_edges; i++)
		{
			/* 2] Hash of the Edge connecting to us */
			uint64_t edge_h = d_edge_Hashes[ d_node_edgePrevs[start_index + i] ];

			/* 3] Use WyHash Unary Mixer to scramble before commutative add */
			prev_hash_acc += hashMix(edge_h);
		}
		/*------------------------------------------------------------------------*/

		/*------------------------------------------------------------------------*/
		/* B] Node Nexts: we keep new vars so for small nexts the instructions gets unrolled */
		int num_next_edges = d_node_edgeNextsNum[nodeIdx];
		int start_indexN   = d_node_edgeNextsStart[nodeIdx];

		uint64_t next_hash_acc = 0;
		for (int i = 0; i < num_next_edges; ++i)
		{
			uint64_t edge_h = d_edge_Hashes[ d_node_edgeNexts[start_indexN + i] ];
			next_hash_acc += hashMix(edge_h);
		}
		/*------------------------------------------------------------------------*/

		/*------------------------------------------------------------------------*/
		/* C] Write New Hash using Chain Mix Initial_State -> Input_Edges -> Output_Edges */
		uint64_t final_hash = d_node_Hashes_Initial[nodeIdx];
		/* Mix Prevs  */
		final_hash = hashPair(final_hash, prev_hash_acc);
		/* Mix Nexts  */
		final_hash = hashPair(final_hash, next_hash_acc);
		nodeHashes[nodeIdx] = final_hash;
		/*------------------------------------------------------------------------*/

	}
}
/*===================================================================================================================*/



/*===================================================================================================================*/
/* For Teaching WL-2 Kernel: Each Edge does an atomic update of the hash for the nodes it is connects to */
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

	        // 1. Base Hash for the Edge Type
	        uint64_t edge_hash = hashMix((uint64_t)label);

	        int srcStartIdx = d_edge_sources_start[edgeIdx];
	        int srcNumNodes = d_edge_sources_num[edgeIdx];
	        int trgStartIdx = d_edge_targets_start[edgeIdx];
	        int trgNumNodes = d_edge_targets_num[edgeIdx];

	        for (int s = 0; s < srcNumNodes; s++)
	        {
	            int u = d_edge_sources[srcStartIdx + s];

	            // Generate a salt for the source port index
	            uint64_t s_salt = (uint64_t)s * 0x517cc1b727220a95ULL;

	            for (int t = 0; t < trgNumNodes; t++)
	            {
	                int v = d_edge_targets[trgStartIdx + t];

	                if (u < numNodes && v < numNodes)
	                {
	                    // Generate a salt for the target port index
	                    uint64_t t_salt = (uint64_t)t * 0x9e3779b97f4a7c15ULL;

	                    // 2. Position-Aware Hash: unique to (EdgeLabel, SourcePort, TargetPort)
	                    uint64_t positional_hash = edge_hash ^ s_salt ^ t_salt;

	                    uint64_t* addr = &MatrixEleColor[(size_t)u * numNodes + v];

	                    /* AtomicAdd preserves edge multiplicity and structural signatures */
	                    atomicAdd((unsigned long long*)addr, (unsigned long long)positional_hash);
	                }
	            }
	        }
	    }
}
/*===================================================================================================================*/






#endif /* ISOMORPHCHECKER_SRC_GPU_SOLVER_CUDA_KERNELSWL1_CUH_ */
