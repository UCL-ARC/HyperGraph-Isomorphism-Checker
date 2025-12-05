# Hypergraph Isomorphism & GPU Architecture

## 1. Coding Philosophy: Minimal C
To ensure the core functionality remains transparent and portable, this project adheres to a **Minimal C** coding style.
* **Transparency:** Logic is exposed clearly without heavy abstraction layers.
* **Portability:** The design facilitates future implementation in **Rust** or other systems languages.
* **Simplicity:** Complex C++ features are avoided in favor of flat data structures and simple pointers.

---

## 2. Input Pipeline

### Raw Input
To facilitate standalone testing, a simple JSON reader for hypergraphs is used. This matches the format of the Python serial version in the upstream repository, allowing for the reuse of existing frameworks and tooling.

The raw data is parsed into the following C++ structures:

```cpp
struct IO_hyperEdge
{
    int labelIndex;
    std::vector<uint> sourceNodes;
    std::vector<uint> targetNodes;
};

struct InputGraph
{
    std::vector<std::string> nodeLabelsDB;  /* Unique node labels */
    std::vector<uint> nodeLabelIndex;       /* Label index for each node */

    std::vector<std::string> edgeLabelsDB;  /* Unique edge labels */
    std::vector<IO_hyperEdge> edges;        /* All hyperedges */

    std::vector<uint> globalInputs;         /* Global input node IDs */
    std::vector<uint> globalOutputs;        /* Global output node IDs */
};
```

### Working Input (CSR Optimization)
For efficient parallel computing on the GPU, the object-based structures above are processed into **Flat/Compact Arrays** using the **CSR (Compressed Sparse Row)** format.



**Optimization Strategy:**
1.  **Flattening:** Data is converted to compact arrays required for the devised algorithms.
2.  **Sorting:** Data is sorted by edge tuple before storage to improve cache efficiency.

**Initialization Logic (`IsomorphsimChecker.cpp`):**
```cpp
/* 1. Open and process the json file or pass arrays from binary (RUST) */
LoadGraphs(argc, argv, m_IO_graphs, MaxNodesPerEdge); 

/* 2. Sort Edges based on counts for cache optimization */
SortGraphEdges(m_IO_graphs, m_DebugEdge_LabelDBIndexOrg); 

/* 3. Create compact arrays and pass to the GPU */
for (int gInd = 0; gInd < 2; gInd++)
{
    /* Initialize GPU graph data structure */
    AllocateCSRGraphData(gInd, m_IO_graphs[gInd], m_CSRGraphs[gInd]);

    /* Compute metadata: edge counts, CSR offsets, and IO tags (A0 + A1 + A2) */
    ComputeCompactArrayMetadata(gInd, m_IO_graphs[gInd], m_CSRGraphs[gInd], m_DebugHist);

    /* Allocate and populate compact arrays using computed metadata (B phase) */
    AllocateAndPopulateCompactArrays(m_IO_graphs[gInd], m_CSRGraphs[gInd], m_DebugHist);
}
```

---

## 3. Compact Array Reference
The following arrays represent the flattened graph structure. The index `[2]` denotes storage for the two graphs being compared.

### A] Node Struct Compact List
```cpp
/* --- Per Node Array Storage --- */
uint *m_Node_LabelDBIndex [2];         /* [1] Index of the label that identifies the node */
uint *m_Node_IOTag [2];                /* [2] 0: None, 1: GInput, 2: GOut, 3: Both */

uint *m_Node_EdgeStartPrevsNum [2];    /* [3] Count in node_EdgePrevs array */
uint *m_Node_EdgeStartNextsNum [2];    /* [4] Count in node_EdgeNexts array */
uint *m_Node_TotEdges [2];             /* [5] Sum of Next and Prevs */
uint *m_Node_EdgeStartPrevsStart [2];  /* [6] Start index in node_EdgePrevs array */
uint *m_Node_EdgeStartNextsStart [2];  /* [7] Start index in node_EdgeNexts array */

/* --- Node Edge Connections --- */
/* Each node writes its input and output edges into these compact arrays */
uint *m_Node_EdgePrevs [2];            /* [8] CSR "From Edge Sources" */
uint *m_Node_EdgeNexts [2];            /* [9] CSR "From Edge Targets" */

/* --- Connection Ports --- */
int *m_Node_EdgePrevsPort [2];         /* [10] CSR ports for edge connections */
int *m_Node_EdgeNextsPort [2];         /* [11] CSR ports for edge connections */

/* --- Signatures --- */
int *m_Node_PrevsFirstEdge [2];        /* [12] Used for signature generation */
int *m_Node_NextsFirstEdge [2];        /* [13] Used for signature generation */
```

### B] Edge Struct Compact List
```cpp
/* --- Per Edge Storage --- */
uint *m_Edge_LabelDBIndex [2];            /* [14] Index of the label that identifies the edge */

/* --- Edge Node Connections --- */
uint *m_Edge_NodeStartSourcesNum [2];     /* [15] Start index in edge_NodesSources array */
uint *m_Edge_NodeStartTargetsNum [2];     /* [16] Count in edge_NodesTargets array */
uint *m_Edge_TotNodes [2];                /* [17] Sum of Next and Prevs */
uint *m_Edge_NodeStartSourcesStart [2];   /* [18] Start index in edge_NodesSources array */
uint *m_Edge_NodeStartTargetsStart [2];   /* [19] Count in edge_NodesTargets array */

/* --- CSR Lists --- */
/* Each edge writes its source and target nodes into these compact arrays */
uint *m_Edge_NodesSources [2];            /* [20] CSR Source Node List */
uint *m_Edge_NodesTargets [2];            /* [21] CSR Target Node List */
```

---

## 4. GPU Implementation
The GPU implementation utilizes the **Thrust** library (the GPU equivalent of Boost) for standardized parallel tasks such as sorting and counting. The isomorphism check is performed in two phases.

### Phase 1: Bulk Signature Tests
We aim to rule out isomorphism rapidly by launching CUDA kernels that compare node and edge signature counts.
1.  **Histogram Comparison:** Creates histograms of signatures and compares counts between graphs.
2.  **Detailed Edge Signature:** A second, more computationally expensive test comparing detailed edge signatures.

```cpp
bool GPU_CompareSignatureCountsBetweenGraphs();
bool GPU_CompareEdgesSignaturesBetweenGraphs();
```

### Phase 2: Structural Analysis (WL1)
If the graphs cannot be distinguished by the "bulk" tests, we initiate computationally intensive tests to explore the structure of the connectivity.

The current implementation utilizes the **Weisfeiler-Lehman (WL1)** algorithm. It explores first-degree connectivity and propagates data iteratively to generate color hashes.



```cpp
/*
 * gIndex: Index of the graph (0 or 1)
 * MAX_ITERATIONS: Depth of propagation/color refinement
 */
void GPU_WL1GraphColorHashIT(int gIndex, int MAX_ITERATIONS);
```
