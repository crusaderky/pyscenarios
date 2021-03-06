from typing import Tuple, Union

# Chunk size for a return array, which has shape (samples, dimensions). It can be
# anything accepted by dask (a positive integer, a tuple of two ints, or a tuple of two
# tuples of ints) for the output shape or None to request a numpy array.
Chunks2D = Union[None, int, Tuple[int, int], Tuple[Tuple[int, ...], Tuple[int, ...]]]
NormalizedChunks2D = Tuple[Tuple[int, ...], Tuple[int, ...]]
