/*
 * hyperGraph.h
 *
 *  Created on: Oct 23, 2025
 *      Author: Nicolin Govender UCL-ARC
 */

#ifndef GPU_SOLVER_CUDA_FUNCTIONS_H_
#define GPU_SOLVER_CUDA_FUNCTIONS_H_

typedef unsigned int uint;

void InitGPUArrays( uint gIndex,
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

void GPU_FreeArrays (uint gIndex, uint gpu);

bool GPU_CompareSignatureCountsBetweenGraphs();
bool GPU_CompareEdgesSignaturesBetweenGraphs();

void GPU_WL1GraphColorHashIT( int gIndex, int MAX_ITERATIONS );



#endif /* GPU_SOLVER_CUDA_FUNCTIONS_H_ */
