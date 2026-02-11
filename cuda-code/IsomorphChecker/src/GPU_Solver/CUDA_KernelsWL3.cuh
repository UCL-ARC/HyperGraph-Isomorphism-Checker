/** CUDA_KernelsWL3.cuh  */
#ifndef ISOMORPHCHECKER_SRC_GPU_SOLVER_CUDA_KERNELSWL3_CUH_
#define ISOMORPHCHECKER_SRC_GPU_SOLVER_CUDA_KERNELSWL3_CUH_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*-----------------------------------------------------------------------------------------------*/
/* Populates the NxN uint8_t mask from the sparse edge list                                      */
/* We build a symmetric mask to find ALL physically connected triplets               */
/* The directionality is later recovered via the directed Edge Colors in InitSparseTriplets      */
/*-----------------------------------------------------------------------------------------------*/
__global__ void Kernel_BuildAdjacencyMask_Symmetric
(
    int num_edges, const uint* d_sources, const uint* d_srcStart, const uint* d_srcNum,
    const uint* d_targets, const uint* d_tgtStart, const uint* d_tgtNum,
    uint8_t* d_AdjMask, int N)
{
	int edgeIdx = blockIdx.x * blockDim.x + threadIdx.x;
	    if (edgeIdx < num_edges)
	    {
	        /* Get bounds for this hyperedge */
	        uint sStart = d_srcStart[edgeIdx];
	        uint sNum   = d_srcNum[edgeIdx];
	        uint tStart = d_tgtStart[edgeIdx];
	        uint tNum   = d_tgtNum[edgeIdx];

	        /* Cartesian Product: Connect every Source to every Target */
	        /* If tNum is 0 (Rook Graph), the inner loop never runs, preventing the crash */
	        for (uint i = 0; i < sNum; i++)
	        {
	            int u = d_sources[sStart + i];

	            for (uint j = 0; j < tNum; j++)
	            {
	                int v = d_targets[tStart + j];

	                /* Bounds check for safety */
	                if (u < N && v < N)
	                {
	                    d_AdjMask[u * N + v] = 1;
	                    d_AdjMask[v * N + u] = 1;
	                }
	            }
	        }
	    }
}
/*-----------------------------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------------------------*/
/* Computes C = (A * B) > 0 using uint8_t storage                                                */
/* Finds Distance-2 and Distance-3 connections to populate the Sparse Triplet Map         */
/*-----------------------------------------------------------------------------------------------*/
__global__ void Kernel_BooleanMatrixMultiply(int N, const uint8_t* A, const uint8_t* B, uint8_t* C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        uint8_t val = 0;
        for (int k = 0; k < N; ++k)
        {
            if (A[row * N + k] != 0 && B[k * N + col] != 0)
            {
                val = 1;
                break;
            }
        }
        /* Result includes new paths (val), existing paths (A), and self-loops */
        C[row * N + col] = (val || A[row * N + col] || (row == col)) ? 1 : 0;
    }
}
/*-----------------------------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------------------------*/
/* Reads uint8_t mask to count valid triplets                                                    */
/* Counts how many 'w' exist for every valid 'u,v' pair in the mask                              */
/*-----------------------------------------------------------------------------------------------*/
__global__ void Kernel_WL3_CountNonZeroTriplets(int N, const uint8_t* d_AdjMask, int* d_total_count)
{
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	/* Bounds check and sparsity check */
	if (u < N && v < N && d_AdjMask[u * N + v] != 0)
	{
		int local_count = 0;
		for (int w = 0; w < N; w++)
		{
			if (d_AdjMask[v * N + w] != 0)
			{
				local_count++;
			}
		}

		if (local_count > 0)
		{
			atomicAdd(d_total_count, local_count);
		}
	    }
}
/*-----------------------------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------------------------*/
/* Reads uint8_t mask to initialize sparse triplet structures                                    */
/* Captures ordering by using non-commutative hashPair logic on the cycle        */
/* The triplet u->v->w gets a different hash than v->u->w                                        */
/*-----------------------------------------------------------------------------------------------*/
__global__ void Kernel_WL3_InitSparseTriplets( int N, const uint64_t* d_ColorMatrix, const uint8_t* d_AdjMask,
                                               SparseTriplet* d_triplets, int* d_next_index )
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u < N && v < N)
    {
        if (d_AdjMask[u * N + v] != 0)
        {
            for (int w = 0; w < N; w++)
            {
                if (d_AdjMask[v * N + w] != 0)
                {
                    int idx = atomicAdd(d_next_index, 1);
                    d_triplets[idx].u = u;
                    d_triplets[idx].v = v;
                    d_triplets[idx].w = w;

                    /* Fetch Edge Colors which already encode directionality from WL-2 */
                    uint64_t h1 = d_ColorMatrix[u * N + v];
                    uint64_t h2 = d_ColorMatrix[v * N + w];
                    uint64_t h3 = d_ColorMatrix[w * N + u];

                    /* Chain Mix: Order u->v->w is baked into this hash via non-commutative pairing */
                    d_triplets[idx].color = hashPair(h1, hashPair(h2, h3));
                }
            }
        }
    }
}
/*-----------------------------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------------------------*/
/* Iterative Refinement Step (Pebble Game Logic)                                                 */
/* Buckets neighbors by which face of the triangle they share (uv, vw, or wu)                    */
/*-----------------------------------------------------------------------------------------------*/
__global__ void Kernel_WL3_UpdateSparseTriplets_Fast( int num_triplets, const SparseTriplet* d_triplets,
                                                      const int* d_offsets, const int* d_tripletList,
                                                      uint64_t* d_color_scratch )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_triplets)
    {
        SparseTriplet myT = d_triplets[idx];
        uint64_t acc_uv = 0, acc_vw = 0, acc_wu = 0;
        int nodes[3] = {(int)myT.u, (int)myT.v, (int)myT.w};

        /* For each node in the triangle */
        for (int i = 0; i < 3; i++)
        {
            int node = nodes[i];
            int start = d_offsets[node];
            int end   = d_offsets[node + 1];

            /* Check all triplets touching this node */
            for (int j = start; j < end; j++)
            {
                int otherIdx = d_tripletList[j];
                SparseTriplet otherT = d_triplets[otherIdx];

                /* Check intersection */
                bool m_u = (otherT.u==myT.u || otherT.v==myT.u || otherT.w==myT.u);
                bool m_v = (otherT.u==myT.v || otherT.v==myT.v || otherT.w==myT.v);
                bool m_w = (otherT.u==myT.w || otherT.v==myT.w || otherT.w==myT.w);

                /* Accumulate based on Shared Edge */
                /* hashMix is commutative (summation) but the bucket (uv/vw/wu) is not */
                if (m_u && m_v)      acc_uv += hashMix(otherT.color);
                else if (m_v && m_w) acc_vw += hashMix(otherT.color);
                else if (m_w && m_u) acc_wu += hashMix(otherT.color);
            }
        }
        /* Final Chain: Preserves the orientation of the update */
        d_color_scratch[idx] = hashPair(myT.color, hashPair(acc_uv, hashPair(acc_vw, acc_wu)));
    }
}
/*-----------------------------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------------------------*/
/* Syncs scratch memory back to the main triplet array                                           */
/*-----------------------------------------------------------------------------------------------*/
__global__ void Kernel_WL3_WritebackColors(int num_triplets, SparseTriplet* d_triplets, const uint64_t* d_color_scratch)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_triplets)
    {
        d_triplets[idx].color = d_color_scratch[idx];
    }
}
/*-----------------------------------------------------------------------------------------------*/


/*-----------------------------------------------------------------------------------------------*/
/* Functor for canonical triplet sorting                                                         */
/* Defines the canonical order of triplets (u -> v -> w -> color)                         */
/* Critical for the Deterministic Sequence Hash to be stable across runs                         */
/*-----------------------------------------------------------------------------------------------*/
struct TripletComparator
{
    __device__ bool operator()(const SparseTriplet& a, const SparseTriplet& b) const
    {
        if (a.u != b.u) return a.u < b.u;
        if (a.v != b.v) return a.v < b.v;
        if (a.w != b.w) return a.w < b.w;
        return a.color < b.color;
    }
};

/*-----------------------------------------------------------------------------------------------*/
/* Functor to extract the 'u' component for offset calculation                                   */
/*-----------------------------------------------------------------------------------------------*/
struct GetTripletU
{
    __device__ int operator()(const SparseTriplet& t) const
    {
        return (int)t.u;
    }
};
/*-----------------------------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------------------------*/
/* Deterministic Sequential Hashing (Aggregator)                                                 */
/* Compresses sorted triplet sequence into a single 64-bit Node Color  */
/* Captures global connections by chain-hashing the sorted sequence           */
/*-----------------------------------------------------------------------------------------------*/
__global__ void Kernel_WL3_DeterministicSequenceHash(
    int N,
    const SparseTriplet* __restrict__ d_triplets,
    const int* __restrict__ d_offsets,
    uint64_t* d_NodeColors)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < N)
    {
        int start = d_offsets[u];
        int end = d_offsets[u + 1];

        uint64_t node_h = 0xCBF29CE484222325ULL; /* FNV Basis */

        /* Sequential Chain */
        for (int i = start; i < end; i++)
        {
            node_h = hashPair(node_h, d_triplets[i].color);
        }
        d_NodeColors[u] = node_h;
    }
}
/*-----------------------------------------------------------------------------------------------*/

#endif // ISOMORPHCHECKER_SRC_GPU_SOLVER_CUDA_KERNELSWL3_CUH_
