import numpy as np


def sort_by_key(keys, values):
    """This can be done on a GPU using e.g. thrust::sort_by_key
    For short fixed length key like three numbers they can be
    compacted into single e.g. 48 or 96-bit datatype and use radix-sort"""
    permutation = np.argsort(keys)
    return keys[permutation], values[permutation]


def sort_packed_by_key(keys, values):
    permutation = np.lexsort(np.rot90(keys))
    return keys[permutation], values[permutation]


def sort_str_by_key(key_array, num_keys, key_size):
    permutation = np.arange(num_keys, dtype=np.int64)
    print(permutation, key_array)
    for i in range(key_size - 1, -1, -1):  # moving from left to right
        print(i, num_keys, key_size)
        key_digits = key_array[i * num_keys : (i + 1) * num_keys]
        key_digits = key_digits[permutation]
        print(key_digits)
        P = np.argsort(key_digits, kind="stable")
        print(P)
        permutation = permutation[P]
    return permutation


def stable_sort_by_key(keys, values):
    permutation = np.argsort(keys, kind="stable")
    return keys[permutation], values[permutation]


def prefix_sum(A):
    return np.cumulative_sum(A)


def max_scan(A):
    return np.maximum.accumulate(A)


def unique(inputs):
    """Analogous to thrust::unique / thrust::unique_by_key"""


def sort(inputs):
    """A standard comparison sort algorithm."""
    return np.sort(inputs)


def radix_sort(inputs):
    """Radix sort can be used more efficiently than comparison sort on fixed length data."""
    return np.sort(inputs)


def filter_unique(indices):
    """We can do the initial colouring of the interface by sorting and then
    doing a filter / stream compaction which is also O(log N)"""
    n_indices = len(indices)
    indices_and_pos = zip(indices, range(n_indices))
    sort(indices_and_pos)
    ## filter on indices to remove repeats
    unique(indices_and_pos)
    ## sort on position again (second value)


def genChangeArray(N, V, eq=lambda a, b: a == b):
    """Assigns a bool array which is 1 wherever the input array V changes value"""
    B = np.zeros(N, dtype=np.int64)
    ## Trivially parallelisable
    for i in range(1, N):
        B[i] = 0 if eq(V[i], V[i - 1]) else 1
    return B


def stream_compaction():
    """Stream compaction can be implemented using sort/scan operations"""
