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

/* Prime constants to differentiate the two streams */
#define PRIME_A 0xbf58476d1ce4e5b9ULL
#define PRIME_B 0x94d049bb133111ebULL

/* Sentinel  if a hash collision with the zero-vector occurs */
#define HASH_SENTINEL 0xDEADBEEFCAFEBABEULL
#define  _wyp  0xa0761d6478bd642fULL
#define  _wyp0 0xa0761d6478bd642fULL
#define  _wyp1 0xe7037ed1a0b428dbULL
#define  _wyp2 0xbf58476d1ce4e5b9ULL


/*-----------------------------------------------------------------------------------------------*/
/** 0] Non-Commutative WyHash                                                                    */
/*-----------------------------------------------------------------------------------------------*/
__device__ inline uint64_t hashMix(uint64_t k)
{

    uint64_t x  = k ^ _wyp;
    uint64_t lo = x * _wyp;
    uint64_t hi = __umul64hi(x, _wyp);
    uint64_t res = lo ^ hi;
    return (res == 0) ? HASH_SENTINEL : res;
}
/*-----------------------------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------------------------*/
/** 0.1] Non-Commutative WyHash Pairing                                                          */
/*-----------------------------------------------------------------------------------------------*/
__device__ inline uint64_t hashPair(uint64_t a, uint64_t b)
{

	/* 1] Position-Aware Rotation: Breaks symmetry in A/B pairings */
	uint64_t b_rot = (b << 19) | (b >> 45);

	/* 2] Round 1: Multiplication and XOR-Fold */
	uint64_t x = a ^ _wyp0;
	uint64_t y = b_rot ^ _wyp1;
	uint64_t h1 = (x * y) ^ __umul64hi(x, y);

	/* 3] Round 2: Re-mixing with third prime */
	/* Forces another non-linear transformation to ensure bit-dispersion */
	uint64_t h2  = h1 ^ _wyp2;
	uint64_t res = (h2 * _wyp0) ^ __umul64hi(h2, _wyp0);

	return (res == 0) ? 0xDEADBEEFCAFEBABEULL : res;
}
/*-----------------------------------------------------------------------------------------------*/


#define SALT_PORT_0 0x9E3779B97F4A7C15ULL
#define SALT_PORT_1 0xBF58476D1CE4E5B9ULL
#define SALT_PORT_2 0x94D049BB133111EBULL
#define SALT_PORT_3 0x845ED68B12032517ULL

/*-----------------------------------------------------------------------------------------------*/
/** 0.2] Mixes a single NodeKeyTuple into an existing 64-bit hash */
__device__ __inline__ uint64_t hashAddTuple(uint64_t hash, const NodeKeyTuple& t)
{
	    /* Chain mix each port value */
	    hash = hashPair(hash, (uint64_t)thrust::get<0>(t) ^SALT_PORT_0);
	    hash = hashPair(hash, (uint64_t)thrust::get<1>(t) ^ SALT_PORT_1);
	    hash = hashPair(hash, (uint64_t)thrust::get<2>(t) ^ SALT_PORT_2);
	    hash = hashPair(hash, (uint64_t)thrust::get<3>(t) ^ SALT_PORT_3);

	    return hash;
}
/*-----------------------------------------------------------------------------------------------*/

#include "CUDA_KernelsBulkStats.cuh"
#include "CUDA_KernelsWL1.cuh"
#include "CUDA_KernelsWL3.cuh"



                                   /* Start of WL-2 Kernels */

/*================================================================================================*/
/** 1.1] Self-Loop Detection Kernel                                                               */
/** Detects if a node connects to itself and mixes this into the diagonal color                   */
/*================================================================================================*/
__global__ void Kernel_WL2_DetectSelfLoops(
    int numNodes,

    /* Node's incoming edges (prevs) */
    const uint* __restrict__ d_node_edgePrevs,
    const uint* __restrict__ d_node_edgePrevsStart,
    const uint* __restrict__ d_node_edgePrevsNum,

	/* Node's outgoing edges (nexts) */
    const uint* __restrict__ d_node_edgeNexts,
    const uint* __restrict__ d_node_edgeNextsStart,
    const uint* __restrict__ d_node_edgeNextsNum,

	/* Matrix to update diagonal only */
    uint64_t* __restrict__ d_Matrix)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;

    if (u < numNodes)
    {
        bool has_self_loop = false;
        int self_loop_count = 0;

        /* 1] Check incoming edges: Does any "prev" edge connect u to itself? */
        int prevStart = d_node_edgePrevsStart[u];
        int prevNum   = d_node_edgePrevsNum[u];

        for (int i = 0; i < prevNum; i++)
        {
            uint neighbor = d_node_edgePrevs[prevStart + i];
            if (neighbor == u)
            {
                has_self_loop = true;
                self_loop_count++;
            }
        }

        /* 2] Check outgoing edges: Does any "next" edge connect u to itself? */
        int nextStart = d_node_edgeNextsStart[u];
        int nextNum   = d_node_edgeNextsNum[u];

        for (int i = 0; i < nextNum; i++)
        {
            uint neighbor = d_node_edgeNexts[nextStart + i];
            if (neighbor == u)
            {
                has_self_loop = true;
                self_loop_count++;
            }
        }

        /* 3] If self-loop exists, mix it into the diagonal color */
        if (has_self_loop)
        {
            size_t diag_idx = (size_t)u * numNodes + u;
            uint64_t current_color = d_Matrix[diag_idx];

            /* Use a unique constant + count to distinguish:
             * - No self-loop: 0
             * - 1 self-loop: hashPair(color, SELF_LOOP_SALT + 1)
             * - 2 self-loops: hashPair(color, SELF_LOOP_SALT + 2)
             * etc */
            const uint64_t SELF_LOOP_SALT = 0xDEADBEEFCAFEBABEULL;
            uint64_t self_loop_hash = SELF_LOOP_SALT + (uint64_t)self_loop_count;

            d_Matrix[diag_idx] = hashPair(current_color, self_loop_hash);
        }
    }
}
/*================================================================================================*/

/*================================================================================================*/
/** 1.2] Set Diagonal: Each node updates its own cell (i, i) using Register-Based WyHash O(N) */
/*================================================================================================*/
__global__ void Kernel_WL2_Init_Diagonal(int numNodes, const uint* __restrict__ Node_IOTag,
		                                 const NodeKeyTuple* __restrict__ d_node_keys, uint64_t *MatrixEleColor)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numNodes)
    {
        NodeKeyTuple k = d_node_keys[idx];

        /* Mix the IOTag (input/output/both) into the base before the tuple to separate node functional roles */
        uint64_t role_salt = (uint64_t)Node_IOTag[idx] * 0x3141592653589793ULL;
        uint64_t node_hash = hashAddTuple(role_salt, k);

        size_t addr = (size_t)idx * (size_t)numNodes + (size_t)idx;
        MatrixEleColor[addr] = node_hash;
    }
}
/*================================================================================================*/

/*================================================================================================*/
/* 1.3] Sets the diagonal color of a specific node to a unique "Super Color"                      */
/*================================================================================================*/
__global__ void Kernel_WL2_InjectUniqueColor(int nodeSizeN, int nodeID, uint64_t* d_Matrix, uint64_t magic_color)
{
    /* The matrix is N*N. The node's color is stored on the diagonal [nodeID * N + nodeID] */
    size_t idx = (size_t)nodeID * (size_t)nodeSizeN + (size_t)nodeID;
    d_Matrix[idx] = magic_color;
}
/*================================================================================================*/

/*================================================================================================*/
/* 1.4] Batch Diagonal Injection: Update all diagonal elements from an array                                              */
/*================================================================================================*/
__global__ void Kernel_Wl2_InjectDiagonalColors(
    int nodeSizeN,
    const uint64_t* __restrict__ d_NewColors,
    uint64_t* d_Matrix)
{
    int nodeID = blockIdx.x * blockDim.x + threadIdx.x;

    if (nodeID < nodeSizeN)
    {
        size_t idx = (size_t)nodeID * (size_t)nodeSizeN + (size_t)nodeID;
        d_Matrix[idx] = d_NewColors[nodeID];
    }
}
/*================================================================================================*/

/*================================================================================================*/
/* 1.5] Extract Diagonals                                               */
/*================================================================================================*/
__global__ void Kernel_WL2_ExtractDiagonals(int numNodes, const uint64_t* d_Matrix, uint64_t* d_Diagonals)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numNodes)
    {
        size_t idx = (size_t)tid * (size_t)numNodes + (size_t)tid;
        d_Diagonals[tid] = d_Matrix[idx];
    }
}
/*================================================================================================*/


/*================================================================================================*/
/* 2] Triangle Counting (Local WL-3 Feature) Only Done Once */
/* Optimized Triangle Counting with Shared Memory Tiling                                         */
/* Strategy: Cache row u and column v in shared memory to reduce global memory traffic          */
/* Cost: O(N3)-> O(N3 / TILE_SIZE) global reads                                                 */
/*================================================================================================*/
#define TRIANGLE_TILE_SIZE 32  // Match warp size for coalesced access
__global__ void Kernel_WL2_InjectTriangleCounts_Tiled( int numNodes,
												   const uint64_t* __restrict__ d_In,
												   uint64_t* __restrict__ d_Out)
{
    /* Thread coordinates in the N×N output matrix */
    int u = blockIdx.y * TRIANGLE_TILE_SIZE + threadIdx.y;
    int v = blockIdx.x * TRIANGLE_TILE_SIZE + threadIdx.x;

    /* Shared memory for caching tiles */
    __shared__ uint64_t tile_row_u[TRIANGLE_TILE_SIZE][TRIANGLE_TILE_SIZE];  // Cache u's row
    __shared__ uint64_t tile_col_v[TRIANGLE_TILE_SIZE][TRIANGLE_TILE_SIZE];  // Cache v's column

    uint64_t triangle_count = 0;

    /*  Number of tiles needed to cover all 'w' in [0, numNodes) */
    int numTiles = (numNodes + TRIANGLE_TILE_SIZE - 1) / TRIANGLE_TILE_SIZE;

    /*------------------------------------------------------------------------------------*/
    /* Iterate over tiles of the intermediate dimension 'w'                               */
    /*------------------------------------------------------------------------------------*/
    for (int tileIdx = 0; tileIdx < numTiles; tileIdx++)
    {
        int w_base = tileIdx * TRIANGLE_TILE_SIZE;

        /*--------------------------------------------------------------------------------*/
        /* Cooperatively load row and column into shared memory             */
        /*--------------------------------------------------------------------------------*/
        // Load tile_row_u: row u, columns [w_base, w_base + TILE_SIZE)
        int w_load = w_base + threadIdx.x;
        if (u < numNodes && w_load < numNodes)
        {
            tile_row_u[threadIdx.y][threadIdx.x] = d_In[u * numNodes + w_load];
        }
        else
        {
            tile_row_u[threadIdx.y][threadIdx.x] = 0;  // Padding
        }

        /*  Load tile_col_v: column v, rows [w_base, w_base + TILE_SIZE) */
        int w_row = w_base + threadIdx.y;
        if (w_row < numNodes && v < numNodes)
        {
            tile_col_v[threadIdx.y][threadIdx.x] = d_In[w_row * numNodes + v];
        }
        else
        {
            tile_col_v[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();  // Ensure all threads have loaded their data
        /*--------------------------------------------------------------------------------*/

        /*--------------------------------------------------------------------------------*/
        /* Count triangles using cached data                             */
        /*--------------------------------------------------------------------------------*/
        #pragma unroll
        for (int k = 0; k < TRIANGLE_TILE_SIZE; k++)
        {
            uint64_t c_uw = tile_row_u[threadIdx.y][k];  // From shared memory
            uint64_t c_wv = tile_col_v[k][threadIdx.x];  // From shared memory

            if (c_uw != 0 && c_wv != 0)
            {
            	/*  Hash the relationship pair, preserves edge type information */
                triangle_count += hashPair(c_uw, c_wv);
            }
        }
        __syncthreads();  // Ensure all threads finish before loading next tile
        /*--------------------------------------------------------------------------------*/
    }
    /*------------------------------------------------------------------------------------*/

    /*------------------------------------------------------------------------------------*/
    /* Mix triangle count into the original color                          */
    /*------------------------------------------------------------------------------------*/
    if (u < numNodes && v < numNodes)
    {
        int idx = u * numNodes + v;
        uint64_t my_color = d_In[idx];

        if (triangle_count > 0)
        {
            my_color = hashPair(my_color, triangle_count);
        }

        d_Out[idx] = my_color;
    }
    /*------------------------------------------------------------------------------------*/
}
/*================================================================================================*/



/*===================================================================================================================*/
/* 3] Project HyperEdges into N*N space using a clique-expansion
 * Warp-Level Parallelism: 32 threads work on one edge together  (Tiled GEMM Pattern) */
/* Shared Memory Tiling: Caches Source/Target lists to avoid scattered global reads */
/* Fallback: Handles edges larger than the cache with direct global memory reads  O(N*E) */
/* Kernel Launches numEdges*32 threads  */
/* - Distinct Salts for Source vs Target ports (Domain Separation)                                                   */
/* - Zero-Hash protection (Edges never sum to 0)                                                                     */
/* - Full Avalanche Mixing (High entropy inputs for atomicAdd)                                                       */
/*===================================================================================================================*/
#define WARP_SIZE       32
#define WARPS_PER_BLOCK 8        // 256 threads / 32 = 8 warps
#define MAX_CACHESIZE   64       // 64 per wrap = 512 32bit ints per warp
__global__ void Kernel_WL2_InitHyperEdges_Tiled(
    int numEdges,
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
    /* Calculate Warp-Global ID */
    int warpInBlockIdx = threadIdx.x / WARP_SIZE;
    int warpThreadIdx  = threadIdx.x % WARP_SIZE;
    int edgeIdx        = blockIdx.x * WARPS_PER_BLOCK + warpInBlockIdx;

    /* Shared Memory Cache for Source/Target lists */
    __shared__ int smem_src[WARPS_PER_BLOCK][MAX_CACHESIZE];
    __shared__ int smem_tgt[WARPS_PER_BLOCK][MAX_CACHESIZE];

    if (edgeIdx < numEdges)
    {
        /* 1. Load Metadata */
        uint label, numSnodes, numTnodes, srcStartIdx, tgtStartIdx;
        if (warpThreadIdx == 0)
        {
            label       = d_edge_labels        [edgeIdx];
            srcStartIdx = d_edge_sources_start [edgeIdx];
            numSnodes   = d_edge_sources_num   [edgeIdx];
            tgtStartIdx = d_edge_targets_start [edgeIdx];
            numTnodes   = d_edge_targets_num   [edgeIdx];
        }

        /* Sync to ensure all threads have the data */
        label       = __shfl_sync(0xFFFFFFFF, label,       0);
        numSnodes   = __shfl_sync(0xFFFFFFFF, numSnodes,   0);
        numTnodes   = __shfl_sync(0xFFFFFFFF, numTnodes,   0);
        srcStartIdx = __shfl_sync(0xFFFFFFFF, srcStartIdx, 0);
        tgtStartIdx = __shfl_sync(0xFFFFFFFF, tgtStartIdx, 0);

        /* 2. Collaborative Load into Shared Memory */
        /* Load Sources */
        for (int i = warpThreadIdx; i < numSnodes && i < MAX_CACHESIZE; i += WARP_SIZE)
        {
            smem_src[warpInBlockIdx][i] = d_edge_sources[srcStartIdx + i];
        }
        /* Load Targets */
        for (int i = warpThreadIdx; i < numTnodes && i < MAX_CACHESIZE; i += WARP_SIZE)
        {
            smem_tgt[warpInBlockIdx][i] = d_edge_targets[tgtStartIdx + i];
        }
        __syncwarp();

        /* 3. Process Pairs */
        size_t total_pairs = (size_t)numSnodes * (size_t)numTnodes;

        for (size_t i = warpThreadIdx; i < total_pairs; i += WARP_SIZE)
        {
            size_t s_idx = i / numTnodes;
            size_t t_idx = i % numTnodes;

            int u, v;

            /* Fetch u, v */
            if (s_idx < MAX_CACHESIZE) u = smem_src[warpInBlockIdx][s_idx];
            else                       u = d_edge_sources[srcStartIdx + s_idx];

            if (t_idx < MAX_CACHESIZE) v = smem_tgt[warpInBlockIdx][t_idx];
            else                       v = d_edge_targets[tgtStartIdx + t_idx];

            if (u < numNodes && v < numNodes)
            {
                /* A] Mix Label: Start with entropy from the edge type */
                uint64_t val = hashMix((uint64_t)label);

                /* B] Mix Source Port: Multiply by Large Prime A to separate domain */
                /* 0x517cc is a randomly chosen 64-bit prime */
                uint64_t s_salt = (uint64_t)s_idx * 0x517cc1b727220a95ULL;
                val = hashPair(val, s_salt);

                /* C] Mix Target Port: Multiply by Large Prime B to separate domain */
                /* 0x9e377 is a different 64-bit prime (Golden Ratio const) */
                uint64_t t_salt = (uint64_t)t_idx * 0x9e3779b97f4a7c15ULL;
                val = hashPair(val, t_salt);

                /* D] Final Avalanche: Ensure single-bit changes affect all output bits */
                val = hashMix(val);

                /* E] Zero-Blindness Check:  Force a non-zero pattern */
                if (val == 0) val = HASH_SENTINEL;

//                /* Atomic Add: Commutative and Safe */
//                size_t addr = (size_t)u * (size_t)numNodes + (size_t)v;
//                atomicAdd((unsigned long long*)&MatrixEleColor[addr], (unsigned long long)val);
//
                /* Atomic XOR: Commutative and robust against edge order */
				/* Prevents "Unsorted" discrepancy by ensuring result is order-independent */
				size_t addr = (size_t)u * (size_t)numNodes + (size_t)v;
				atomicXor((unsigned long long*)&MatrixEleColor[addr], (unsigned long long)val);
            }
        }
    }
}
/*===================================================================================================================*/



/*===================================================================================================================*/
/**
 * 4] Same clique-expansion as 3] works on the N*N matrix
 * Per Edge Note: Other option is to do Sparse Matrix Mul
 * Updates the color of every pair of nodes (u, v) based on the colors of
 * all 2NN "2nd-Degree Nodes"  (u, w, v) connected to it
 *
 * Algorithm Strategy (Tiled Matrix Multiplication): (Tiled GEMM Pattern)
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
#define WARP_CACHESIZE 16
__global__ void Kernel_WL2_UpdatePairs_Tiled(int numNodes,
		                                     const uint64_t*  __restrict__  MatrixEleColor,
                                             uint64_t* MatrixEleColorWrite )
{
	/* 1] Calculate Global Coordinates */
	int row = blockIdx.y * WARP_CACHESIZE + threadIdx.y; /** Node 'u' */
	int col = blockIdx.x * WARP_CACHESIZE + threadIdx.x; /** Node 'v' */

	/* 2] High-Accuracy Accumulators to avoid hash clash */
	uint64_t acc1 = 0;
	uint64_t acc2 = 0;

	/* 3] Shared Memory Buffers */
	__shared__ uint64_t tile_A[WARP_CACHESIZE][WARP_CACHESIZE];
	__shared__ uint64_t tile_B[WARP_CACHESIZE][WARP_CACHESIZE];


	/*-------------------------------------------------------------------------------*/
	/* 4] Loop over tiles */
	/* indexStride iterates across the intermediate dimension 'w' */
	int numTiles = (numNodes + WARP_CACHESIZE - 1) / WARP_CACHESIZE;

	for (int indexStride = 0; indexStride < numTiles; ++indexStride)
	{
		/* 4.1] Load Data into Shared Memory */
		int indexColA = indexStride * WARP_CACHESIZE + threadIdx.x; // Col for A
		int indexRowB = indexStride * WARP_CACHESIZE + threadIdx.y; // Row for B

		/* Load tile_A (row, k) */
		if (row < numNodes && indexColA < numNodes)
			tile_A[threadIdx.y][threadIdx.x] = MatrixEleColor[row * numNodes + indexColA];
		else
			tile_A[threadIdx.y][threadIdx.x] = 0;

		/* Load tile_B (k, col) */
		if (indexRowB < numNodes && col < numNodes)
			tile_B[threadIdx.y][threadIdx.x] = MatrixEleColor[indexRowB * numNodes + col];
		else
			tile_B[threadIdx.y][threadIdx.x] = 0;

		__syncthreads();

		/* 4.2] Compute High-Accuracy Hash for this tile */
		#pragma unroll
		for (int k = 0; k < WARP_CACHESIZE; ++k)
		{
			uint64_t c1 = tile_A[threadIdx.y][k]; // A[row][k]
			uint64_t c2 = tile_B[k][threadIdx.x]; // B[k][col]

			if (c1 != 0 || c2 != 0)
			{
				/* Use Robust Mixer, WyHash Logic */
				uint64_t path_hash = hashPair(c1, c2);

				/* Stream 1: Direct Sum */
				acc1 += path_hash;

				/* Stream 2: Scrambled Sum, Mathematically Orthogonal */
				acc2 += (path_hash ^ PRIME_A) * PRIME_B;
			}
		}
		__syncthreads();
	}
	/*-------------------------------------------------------------------------------*/


	/* 5] Final Write */
	if (row < numNodes && col < numNodes)
	{
		uint64_t my_old_color = MatrixEleColor[row * numNodes + col];

		/* Mix old color + Acc1 + Acc2 */
		uint64_t final_hash = hashPair(my_old_color, acc1);
		final_hash = hashPair(final_hash, acc2);

		MatrixEleColorWrite[row * numNodes + col] = final_hash;
	}
}
/*===================================================================================================================*/


#endif /* GPU_SOLVER_CUDA_KERNELS_CUH_ */
