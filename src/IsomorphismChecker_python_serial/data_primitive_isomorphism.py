from IsomorphismChecker_python_serial.hypergraph import (
    FlatHypergraph,
    SegmentedArray,
)

import numpy as np
from IsomorphismChecker_python_serial import data_parallel_primitives as dpp


def InitialCompare(g1: FlatHypergraph, g2: FlatHypergraph):
    ## Check size and type compatibility of vertices and edges
    if g1.num_nodes != g2.num_nodes:
        return False

    if g1.num_edges != g2.num_edges:
        return False

    getNodeTypes = lambda g: [
        (g.node_labels[i], g.node_sources.sizes[i], g.node_targets.sizes[i])
        for i in range(g.num_nodes)
    ]
    node_types1 = getNodeTypes(g1)
    node_types2 = getNodeTypes(g2)
    dpp.sort(node_types1)
    dpp.sort(node_types2)
    if node_types1 != node_types2:
        return False

    getEdgeTypes = lambda g: [
        (g.edge_labels[i], g.edge_sources.sizes[i], g.edge_targets.sizes[i])
        for i in range(g.num_edges)
    ]
    edge_types1 = getEdgeTypes(g1)
    edge_types2 = getEdgeTypes(g2)
    dpp.sort(edge_types1)
    dpp.sort(edge_types2)
    if edge_types1 != edge_types2:
        return False

    ## Check size and type compatibility of global interface
    if g1.num_inputs != g2.num_inputs:
        return False

    if len(g1.global_interface) != len(g2.global_interface):
        return False

    for i in range(len(g1.global_interface)):
        if (
            g1.node_labels[g1.global_interface[i]]
            != g2.node_labels[g2.global_interface[i]]
        ):
            return False

    # pass all checks
    return True


class ColourData:
    def __init__(self, N: int):
        self.v2c = np.array([-1] * N)
        self.c2v = np.array([-1] * N)
        self.c2v_dummy = np.array([-1] * N)
        self.c_sizes = np.array([0] * N)
        self.c_sizes_dummy = np.array([0] * N)
        self.deltas = np.array([(-1, -1)] * N)
        self.Delta = np.array([0] * N)
        self.Delta_t = np.array([-1] * N)
        self.size = N

    def __repr__(self):
        return (
            "ColourData(\n"
            f"  v2c={self.v2c},\n"
            f"  c2v={self.c2v},\n"
            f"  c_sizes={self.c_sizes},\n"
            f"  deltas={self.deltas}\n"
            ")"
        )


class ColouredGraph:
    def __init__(self, g: FlatHypergraph):
        self.g = g
        self.vertexColours = ColourData(g.num_nodes)
        self.edgeColours = ColourData(g.num_edges)


def ColourGlobalInterface(g: FlatHypergraph, colouring: ColourData):
    N_int = len(g.global_interface)
    P = np.arange(N_int, dtype=np.int64)
    I = g.global_interface
    I, P = dpp.stable_sort_by_key(I, P)

    B = dpp.genChangeArray(N_int, I)

    S = dpp.prefix_sum(B)  ## S[i] <= i

    c_max = S[N_int - 1]
    P_prime = np.zeros(c_max + 1, dtype=np.int64)
    I_prime = np.zeros(c_max + 1, dtype=np.int64)

    ## not trivially parallelisable if done in place (reusing I & P)
    for i in range(N_int):
        if B[i]:
            I_prime[S[i]] = I[i]
            P_prime[S[i]] = P[i]

    P_prime, I_prime = dpp.sort_by_key(P_prime, I_prime)
    ## trivially parallelisable
    for i in range(c_max + 1):
        c = I_prime[i]
        colouring.c2v[i] = I_prime[i]
        colouring.c_sizes[i] = 1
        colouring.v2c[c] = i
        colouring.Delta[c] = 1
        colouring.Delta_t[c] = 0
        colouring.deltas[i] = (i, 1)

    return c_max


def InitialColouring(
    g: FlatHypergraph,
    node_colouring: ColourData,
    edge_colouring: ColourData,
    c_max: int,
):
    """Assigns colours to the remaining nodes and edges after the interface"""
    ## Assign the remaining node colours
    nodeKeys = np.array(
        [(-1, -1, -1)] * g.num_nodes
    )  # In practice this would be e.g. 3 16-bit ints packed into a 64 bit value
    ## trivially parallelisable
    for i in range(g.num_nodes):
        if node_colouring.v2c[i] == -1:
            nodeKeys[i] = [
                g.node_labels[i],
                g.node_sources.sizes[i],
                g.node_targets.sizes[i],
            ]

    i_max_nodes = initialiseColoursFromKeys(
        g.num_nodes, node_colouring, c_max, nodeKeys
    )

    edgeKeys = np.array([(-1, -1, -1)] * g.num_edges)
    ## trivially parallelisable
    for i in range(g.num_edges):
        edgeKeys[i] = [
            g.edge_labels[i],
            g.edge_sources.sizes[i],
            g.edge_targets.sizes[i],
        ]

    i_max_edges = initialiseColoursFromKeys(g.num_edges, edge_colouring, -1, edgeKeys)

    return (c_max + i_max_nodes, i_max_edges)


def initialiseColoursFromKeys(N, colouring, c_max, keys):
    """c_max the the most recently filled after the global interface. It is -1 if nothing has been coloured.
    All assigned colours up to this point are guaranteed to be unique."""
    c_next = c_max + 1
    subsize = N - (c_next)  # number of elements yet to be coloured

    P = np.arange(N, dtype=np.int64)
    keys, P = dpp.sort_packed_by_key(keys, P)

    B = dpp.genChangeArray(subsize, keys[c_next:], lambda a, b: np.array_equal(a, b))
    S = dpp.prefix_sum(B)

    i_max = S[subsize - 1] + 1
    workspace = np.array([0] * (i_max + 1))  # size at most N+1
    workspace[i_max] = N

    ## trivially parallelisable
    for i in range(subsize):
        if i == 0 or B[i] == 1:
            workspace[S[i]] = i + c_next

    ## trivially parallelisable and collapsible
    for i in range(i_max):
        n = workspace[i + 1] - workspace[i]
        c = workspace[i]
        colouring.c_sizes[c] = n
        colouring.Delta[c] = N
        colouring.Delta[c] = 1
        colouring.deltas[c_max + i] = (c, n)
        for j in range(n):
            colouring.c2v[c + j] = P[c + j]
            colouring.v2c[P[c + j]] = c
    return i_max  ## next delta entry index


def setupColourCellKeyArrays(
    N,
    sources: SegmentedArray,
    targets: SegmentedArray,
    cell_keys: SegmentedArray,
    colouring: ColourData,
):
    """This needs to be called after the initial colouring but before colour refinement. This allows us to know the size of the key
    required for each different colour that will be assigned. (When a colour cell is split, all new colours within that block will
    still have the same size key, so the initial colouring suffices.)"""
    for c in range(N):
        colour_rep = colouring.c2v[c]
        key_size = sources.sizes[colour_rep] + targets.sizes[colour_rep]
        cell_keys.sizes[c] = key_size
    cell_keys.initials = dpp.prefix_sum(cell_keys.sizes) - cell_keys.sizes[0]


def constructEdgeColourKeys(
    N: int,
    keys: SegmentedArray,
    sources: SegmentedArray,
    targets: SegmentedArray,
    neighbour_colouring: ColourData,
):
    """Trivially paralellisable function to construct all node keys. Each node can have its
    key constructed in parallel. The bottleneck is the sort operations for each key which
    can be avoided if hashing is used."""
    for i in range(N):
        start_idx = keys.initials[i]

        ## construct and sort the source subarray
        source_size = sources.sizes[i]
        source_idx = sources.initials[i]
        local_sources = sources.elements[source_idx : source_idx + source_size]
        segment = keys.elements[start_idx : start_idx + source_size]
        segment[:] = [neighbour_colouring.v2c[e] for e in local_sources]

        # do the same for targets
        target_size = targets.sizes[i]
        target_idx = targets.initials[i]
        local_targets = targets.elements[target_idx : target_idx + target_size]
        segment = keys.elements[
            start_idx + source_size : start_idx + source_size + target_size
        ]
        segment[:] = [neighbour_colouring.v2c[e] for e in local_targets]


def constructNodeColourKeys(
    N: int,
    keys: SegmentedArray,
    sources: SegmentedArray,
    s_ports: SegmentedArray,
    targets: SegmentedArray,
    t_ports: SegmentedArray,
    neighbour_colouring: ColourData,
):
    """Trivially paralellisable function to construct all node keys. Each node can have its
    key constructed in parallel. The bottleneck is the sort operations for each key which
    can be avoided if hashing is used."""
    for i in range(N):
        start_idx = keys.initials[i]

        ## construct and sort the source subarray
        source_size = sources.sizes[i]
        source_idx = sources.initials[i]
        local_sources = sources.elements[source_idx : source_idx + source_size]
        local_s_ports = s_ports.elements[source_idx : source_idx + source_size]
        segment = keys.elements[start_idx : start_idx + source_size]
        for j in range(source_size):
            segment[j] = (
                neighbour_colouring.v2c[local_sources[j]] << 16
            ) | local_s_ports[j]
        segment[:] = dpp.sort(segment)

        # do the same for targets
        target_size = targets.sizes[i]
        print(f"Size of targets for {i} is {target_size} = {targets.sizes[i]}")
        target_idx = targets.initials[i]
        local_targets = targets.elements[target_idx : target_idx + target_size]
        local_t_ports = t_ports.elements[target_idx : target_idx + target_size]
        segment = keys.elements[
            start_idx + source_size : start_idx + source_size + target_size
        ]
        for j in range(target_size):
            segment[j] = (
                neighbour_colouring.v2c[local_targets[j]] << 16
            ) | local_t_ports[j]
        segment[:] = dpp.sort(segment)


def colourSetDecomposition(
    N: int,
    cellKeys: SegmentedArray,
    keys: SegmentedArray,
    colouring: ColourData,
    t: int,
):
    ## for each colour we need to decompose the set if the size > 1
    for c in range(N):
        print(c)
        if colouring.c_sizes[c] > 1:
            cell_size = colouring.c_sizes[c]
            segment = colouring.c2v[c : c + cell_size]
            cellKeys_start_idx = cellKeys.initials[c]
            key_size = cellKeys.sizes[c]
            ## copy all keys for this cell into their segment
            # for i in range(size):
            #    cell_idx = cellKeys_start_idx + i*key_size
            #    v = segment[i]
            #    vKey_idx = keys.initials[v]
            #    key = keys.elements[vKey_idx:vKey_idx+key_size]
            #    cellKeys.element[cell_idx:cell_idx+key_size] = key

            ## for simpler sequential sorting arrange the keys (a, b, c...) as
            ## [a0, b0, c0, ..., a1, b1, c1, ...]
            ## Then we can sort the string by stable sorting the segments left to
            ## right
            for i in range(cell_size):
                cell_idx = cellKeys_start_idx + i
                v = segment[i]
                vKey_idx = keys.initials[v]
                key = keys.elements[vKey_idx : vKey_idx + key_size]
                for j in range(key_size):
                    cellKeys.elements[cell_idx + j * cell_size] = key[j]

            ## sort these keys
            ## for key of length k this does k sort operations on n elements
            key_segment = cellKeys.elements[
                cellKeys_start_idx : cellKeys_start_idx + (cell_size * key_size)
            ]
            print(key_segment, cell_size, key_size)
            P = dpp.sort_str_by_key(key_segment, cell_size, key_size)
            print(P, cell_size, P.size, key_size)

            ## generating the counting array is now linear in the size of the key
            ## due to the equality check over an array
            B = np.array([0] * cell_size)  # combined this is just an array of length N
            for i in range(1, cell_size):
                v_i = P[i]
                v_im1 = P[i - 1]
                ki_idx = keys.initials[v_i]
                kim1_idx = keys.initials[v_im1]
                key_i = keys.elements[ki_idx : ki_idx + key_size]
                key_im1 = keys.elements[kim1_idx : kim1_idx + key_size]
                if not np.array_equal(key_i, key_im1):
                    B[i] = i

            S = dpp.max_scan(B)  ## S now contains the offsets from the original colour

            # helper function to avoid code repetition
            def recordNewCell(colouring, t, c, size):
                colouring.c_sizes[c] = size
                colouring.Delta[c] = size
                colouring.Delta_t[c] = t

            # Only need to proceed if there is at least one key that is different
            if S[cell_size - 1] != 0:
                # make a copy of the relevant data from colouring so that we can read and update
                # in parallel; only necessary if this loop needs to be parallelised
                colouring.c2v_dummy[c : c + cell_size] = colouring.c2v[
                    c : c + cell_size
                ]
                for j in range(cell_size):
                    v = colouring.c2v_dummy[c + P[j]]
                    colouring.v2c[v] = c + S[j]
                    colouring.c2v[c + j] = v
                    if B[j] != 0:
                        diff = S[j] - S[j - 1]
                        recordNewCell(colouring, t, c + j - diff, diff)
                diff = cell_size - S[cell_size - 1]
                recordNewCell(colouring, t, c + cell_size - diff, diff)


def refineColouring(
    g: FlatHypergraph, vertex_colours: ColourData, edge_colours: ColourData, t: int
):
    """Perform a single step of colour refinement on a graph"""
    constructNodeColourKeys(
        g.num_nodes,
        g.node_keys,
        g.node_sources,
        g.node_s_ports,
        g.node_targets,
        g.node_t_ports,
        edge_colours,
    )
    colourSetDecomposition(
        g.num_nodes, g.node_cell_keys, g.node_keys, vertex_colours, t
    )

    constructEdgeColourKeys(
        g.num_edges, g.edge_keys, g.edge_sources, g.edge_targets, vertex_colours
    )
    colourSetDecomposition(g.num_edges, g.edge_cell_keys, g.edge_keys, edge_colours, t)


def convergeColouring(
    g: FlatHypergraph, vertex_colours: ColourData, edge_colours: ColourData, t: int
):
    """Apply colour refinement until it has stabilised"""
    # Convergence criterion could also be implemented by setting a flag when updating
    # colours and then performing a reduction over those values to detect changes
    def converged():
        vertices_converged = np.all(vertex_colours.Delta_t < t)
        edges_converged = np.all(edge_colours.Delta_t < t)
        return vertices_converged and edges_converged

    while not converged():
        t += 1
        refineColouring(g, vertex_colours, edge_colours, t)
    return t


def checkCompleteness(vertex_colours: ColourData, edge_colours: ColourData):
    """Check whether the colouring is discrete"""
    vertices_discrete = np.all(vertex_colours.c_sizes == 1)
    edges_discrete = np.all(edge_colours.c_sizes == 1)
    return (vertices_discrete, edges_discrete)


def processStableColourings(cg1: ColouredGraph, cg2: ColouredGraph, t: int) -> bool:
    (v_discrete, e_discrete) = checkCompleteness(cg1.vertexColours, cg1.edgeColours)
    # at this point since the histories are identical the same must be true of cg2

    if v_discrete and e_discrete:
        # we're done!
        return True
    elif not v_discrete:
        c = selectTargetCell(cg1.vertexColours)
        return exploreBranches(cg1, cg2, c, t)
    else:
        # need to implement explore for edges
        c = selectTargetCell(cg1.edgeColours)
        return exploreBranches(cg1, cg2, c, t)


def selectTargetCell(colouring: ColourData):
    """Select a colour to force refinement"""
    S, C = dpp.stable_sort_by_key(colouring.c_sizes, np.arange(colouring.c_sizes.size))
    for s, c in zip(S, C):
        if s > 1:
            return c
    raise LookupError("No valid target cells found.")


def exploreBranches(
    cg1: ColouredGraph,
    cg2: ColouredGraph,
    target_colour: int,
    t: int,
    recolour_edges=False,
) -> bool:
    """Explores possibilities in g2 for matching g1
    If no branches give a positive match then the graphs are not isomorphic"""
    # start by recolouring an element of the target cell in g1
    target_cell_size = cg1.vertexColours.c_sizes[target_colour]
    new_colour = target_colour + target_cell_size - 1
    target_vertex = cg1.vertexColours.c2v[new_colour]
    recolourTarget(cg1.vertexColours, target_colour, new_colour, target_vertex, t + 1)

    # Propragate the consequences in g1
    t1 = convergeColouring(cg1.g, cg1.vertexColours, cg1.edgeColours, t + 1)

    # search for a matching solution in g2
    for i in range(target_cell_size):
        if checkBranch(cg1, cg2, target_colour, new_colour, i, t1, t):
            return True
        else:
            # unroll changes to c2 before trying again!
            rollBackColouring(cg2.vertexColours, t)
            rollBackColouring(cg2.edgeColours, t)

    return False


def checkBranch(cg1, cg2, target_colour, new_colour, i, t1, t) -> bool:
    target_vertex2 = cg2.vertexColours.c2v[target_colour + i]
    recolourTarget(cg2.vertexColours, target_colour, new_colour, target_vertex2, t + 1)
    t2 = convergeColouring(cg2.g, cg2.vertexColours, cg2.edgeColours, t + 1)
    if t1 != t2:
        return False
    # Compare the results of the recolouring to see if this is valid so far
    if not compareNodeInvariant(cg1, cg2):
        return False
    else:
        # Recursively check for more refinements to be made.
        return processStableColourings(cg1, cg2, t1)


def recolourTarget(colouring1, target_colour, new_colour, target_vertex, t):
    colouring1.v2c[target_vertex] = new_colour
    colouring1.c_sizes[new_colour] = 1
    colouring1.c_sizes[target_colour] -= 1
    colouring1.Delta[new_colour] = 1
    colouring1.Delta_t[new_colour] = t


def compareNodeInvariant(cg1: ColouredGraph, cg2: ColouredGraph) -> bool:
    """Check that the histories of the two graphs are sufficiently similar"""
    match_v_history = np.all(
        cg1.vertexColours.Delta == cg2.vertexColours.Delta
    ) and np.all(cg1.vertexColours.Delta_t == cg2.vertexColours.Delta_t)
    match_e_history = np.all(cg1.edgeColours.Delta == cg2.edgeColours.Delta) and np.all(
        cg1.edgeColours.Delta_t == cg2.edgeColours.Delta_t
    )
    return match_v_history and match_e_history


def rollBackColouring(colouring: ColourData, t: int):
    """Roll back the colouring to step t"""
    for i in range(colouring.size):
        if colouring.Delta_t[i] > t:
            cell_size = colouring.c_sizes[i]
            # search for its previous colour
            previous_colour = 0
            for j in range(i - 1, -1, -1):
                if colouring.Delta_t[j] <= t:
                    previous_colour = j
            for j in range(cell_size):
                v = colouring.c2v[j]
                colouring.v2c[v] = previous_colour
            colouring.Delta[i] = 0
            colouring.Delta_t[i] = -1
            # this would need to reformulated as a reduction
            colouring.c_sizes[previous_colour] += cell_size


def checkIsomorphism(cg1: ColouredGraph, cg2: ColouredGraph) -> bool:
    """Checks that two given graphs with discrete colourings are isomorphic"""
    # Check the nodes are compatible
    # Start with the interface
    if (cg1.g.num_inputs != cg2.g.num_inputs) or (
        len(cg1.g.global_interface) != len(cg2.g.global_interface)
    ):
        return False

    for i in range(cg1.g.num_inputs):
        v1 = cg1.g.global_interface[i]
        v2 = cg2.g.global_interface[i]
        if cg1.vertexColours.v2c[v1] != cg2.vertexColours.v2c[v2]:
            return False

    for c in range(cg1.g.num_nodes):
        v1 = cg1.vertexColours.c2v[c]
        v2 = cg2.vertexColours.c2v[c]
        if cg1.g.node_labels[v1] != cg2.g.node_labels[v2]:
            return False

    # Check the edges
    for c in range(cg1.g.num_edges):
        e1 = cg1.edgeColours.c2v[c]
        e2 = cg1.edgeColours.c2v[c]
        if cg1.g.edge_labels[e1] != cg2.g.edge_labels[e2]:
            return False
        if cg1.g.edge_sources.sizes[e1] != cg2.g.edge_sources.sizes[e2]:
            return False
        if cg1.g.edge_targets.sizes[e1] != cg2.g.edge_targets.sizes[e2]:
            return False
        s_idx1 = cg1.g.edge_sources.initials[e1]
        s_idx2 = cg2.g.edge_sources.initials[e2]
        for s in range(cg1.g.edge_sources.sizes[e1]):
            v1 = cg1.g.edge_sources.elements[s_idx1 + s]
            v2 = cg2.g.edge_sources.elements[s_idx2 + s]
            if cg1.vertexColours.v2c[v1] != cg2.vertexColours.v2c[v2]:
                return False

        t_idx1 = cg1.g.edge_targets.initials[e1]
        t_idx2 = cg2.g.edge_targets.initials[e2]
        for t in range(cg1.g.edge_targets.sizes[e1]):
            v1 = cg1.g.edge_targets.elements[t_idx1 + t]
            v2 = cg2.g.edge_targets.elements[t_idx2 + t]
            if cg1.vertexColours.v2c[v1] != cg2.vertexColours.v2c[v2]:
                return False

    return True


def determineIsomorphism(g1: FlatHypergraph, g2: FlatHypergraph):
    """Complete isomorphism procedure for two hypergraphs."""

    ## Initial compatibility checks
    if not InitialCompare(g1, g2):
        return False

    ## Initial colourings
    cg1 = ColouredGraph(g1)
    cg2 = ColouredGraph(g2)

    cmax1 = ColourGlobalInterface(cg1.g, cg1.vertexColours)
    cmax2 = ColourGlobalInterface(cg2.g, cg2.vertexColours)

    if not (cmax1 == cmax2):
        return False

    InitialColouring(cg1.g, cg1.vertexColours, cg1.edgeColours, cmax1)
    InitialColouring(cg2.g, cg2.vertexColours, cg2.edgeColours, cmax2)

    if not compareNodeInvariant(cg1, cg2):
        return False

    ## Initial convergence for both graphs; we can put more checks in if we combine these into one function
    t1 = convergeColouring(cg1.g, cg1.vertexColours, cg1.edgeColours, 2)
    t2 = convergeColouring(cg2.g, cg2.vertexColours, cg2.edgeColours, 2)
    if not (t1 == t2):
        return False
    if not compareNodeInvariant(cg1, cg2):
        return False

    ## Recursive tree search
    if not (
        checkCompleteness(cg1.vertexColours, cg1.edgeColours)
        and checkCompleteness(cg2.vertexColours, cg2.edgeColours)
    ):
        if not processStableColourings(cg1, cg2, t1):
            return False

    ## Isomorphism check
    return checkIsomorphism(cg1, cg2)
