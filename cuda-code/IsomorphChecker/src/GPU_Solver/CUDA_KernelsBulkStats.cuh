/*
 * CUDA_KernelsBulkStats.cuh
 *
 *  Created on: Feb 4, 2026
 *      Author: blaze
 */

#ifndef ISOMORPHCHECKER_SRC_GPU_SOLVER_CUDA_KERNELSBULKSTATS_CUH_
#define ISOMORPHCHECKER_SRC_GPU_SOLVER_CUDA_KERNELSBULKSTATS_CUH_



/*===================================================================================================================*/
/** BA] Preserves Port Order for Directed Hypergraphs                                          */
/** Works for ANY edge size without sorting - O(N) scan with positional salting                                    */
/*===================================================================================================================*/
__global__ void Kernel_EdgeHashes_PortOrderPreserving(
    int numEdges,

    const uint* __restrict__ d_edge_SourceNodes,
    const uint* __restrict__ d_edge_SourcesStart,
    const uint* __restrict__ d_edge_SourcesNum,

    const uint* __restrict__ d_edge_TargetNodes,
    const uint* __restrict__ d_edge_TargetsStart,
    const uint* __restrict__ d_edge_TargetsNum,

    const NodeKeyTuple* __restrict__ d_node_sig,

    uint64_t* edge_SourceNodeHashes,
    uint64_t* edge_TargetNodeHashes)
{
    int edgeIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (edgeIdx < numEdges)
    {
        /*===========================================================================*/
        /* Source Nodes - Hash in sequential port order                     */
        /*===========================================================================*/
        int srcStartIdx = d_edge_SourcesStart[edgeIdx];
        int srcNumNodes = d_edge_SourcesNum[edgeIdx];

        uint64_t src_hash = FNV_OFFSET_BASIS;

        for (int port_idx = 0; port_idx < srcNumNodes; port_idx++)
        {
        	/* 1] Get the node signature at this port */
            uint nodeID = d_edge_SourceNodes[srcStartIdx + port_idx];
            NodeKeyTuple sig = d_node_sig[nodeID];

            /* 2] Salt the port position to differentiate port0 vs port1 */
            uint64_t port_salt = (uint64_t)port_idx * 0x517cc1b727220a95ULL;
            src_hash = hashPair(src_hash, port_salt);

            /* 3] Mix in the node signature */
            src_hash = hashAddTuple(src_hash, sig);
        }

        edge_SourceNodeHashes[edgeIdx] = src_hash;

        /*===========================================================================*/
        /* Target Nodes - Hash in sequential port order                     */
        /*===========================================================================*/
        int trgStartIdx = d_edge_TargetsStart[edgeIdx];
        int trgNumNodes = d_edge_TargetsNum[edgeIdx];

        uint64_t tgt_hash = FNV_OFFSET_BASIS;

        for (int port_idx = 0; port_idx < trgNumNodes; port_idx++)
        {
            uint nodeID = d_edge_TargetNodes[trgStartIdx + port_idx];
            NodeKeyTuple sig = d_node_sig[nodeID];

            /* 4] Use different salt for target ports to distinguish source vs target */
            uint64_t port_salt = (uint64_t)port_idx * 0x9e3779b97f4a7c15ULL;
            tgt_hash = hashPair(tgt_hash, port_salt);
            tgt_hash = hashAddTuple(tgt_hash, sig);
        }

        edge_TargetNodeHashes[edgeIdx] = tgt_hash;
    }
}
/*===================================================================================================================*/


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



#endif /* ISOMORPHCHECKER_SRC_GPU_SOLVER_CUDA_KERNELSBULKSTATS_CUH_ */
