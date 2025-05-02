from typing import TypeAlias

# Chunk size for a 2D return array, after normalization
NormalizedChunks2D: TypeAlias = tuple[tuple[int, ...], tuple[int, ...]]
# Chunk size for a 2D return array. It can be anything accepted by Dask
# (a positive integer, a tuple of two ints, or a tuple of two tuples of ints)
# for the output shape or None to request a NumPy array.
Chunks2D: TypeAlias = int | tuple[int, int] | NormalizedChunks2D
