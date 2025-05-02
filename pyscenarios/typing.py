from typing import TypeAlias

# Chunk size for a return array, which has shape (samples, dimensions). It can be
# anything accepted by dask (a positive integer, a tuple of two ints, or a tuple of two
# tuples of ints) for the output shape or None to request a numpy array.
NormalizedChunks2D: TypeAlias = tuple[tuple[int, ...], tuple[int, ...]]
Chunks2D: TypeAlias = None | int | tuple[int, int] | NormalizedChunks2D
