/**
 * hyperGraph.h
 *
 *  Created on: Oct 23, 2025
 *      Author: Nicolin Govender UCL-ARC
 */

#ifndef GPU_SOLVER_CUDA_FUNCTIONS_H_
#define GPU_SOLVER_CUDA_FUNCTIONS_H_

typedef unsigned int uint;

/** Entry point of the GPU Solver, Sets the memory of GPU and the thread configuration  */
void GPU_InitArrays( uint gIndex,
					uint numNodesH,
					uint *NodeLabelIndexH,
					int *NodePrevsFirstEdge, int *NodeNextsFirstEdge,
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
					uint gpu               );


void GPU_FreeInitArrays (uint gIndex, uint gpu); /** Free Allocation memory on the GPU */


/* Entry Point calls all needed functions */
bool GPU_CheckHypergraphIsomorphism();


/* Debug */
void RunDeterminismStressTest(int iterations);

bool GPU_CompareSignatureCountsBetweenGraphs(); /** Does the binning on the GPU for the feature counts of each edge and node */
bool GPU_CompareEdgesSignaturesBetweenGraphs(); /** Compare edge signatures based on direct feature hashing  */
bool GPU_WL1GraphColorHashIT( int gIndex, int MAX_ITERATIONS ); /** Iterative Color based on hashing of edge and nodes  */
bool GPU_WL2GraphPairColorInit_Dense(int gIndex, int MAX_ITERATIONS);
bool WL_CompareBinCountsInitState();




#endif /* GPU_SOLVER_CUDA_FUNCTIONS_H_ */
