from __future__ import annotations

# TODO use builtins (requires Python >=3.9)
from typing import TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:  # pragma: nocover
    # TODO move to typing (requires Python >=3.10)
    from typing_extensions import TypeAlias

# Chunk size for a return array, which has shape (samples, dimensions). It can be
# anything accepted by dask (a positive integer, a tuple of two ints, or a tuple of two
# tuples of ints) for the output shape or None to request a numpy array.
Chunks2D: TypeAlias = Union[
    None, int, Tuple[int, int], Tuple[Tuple[int, ...], Tuple[int, ...]]
]
NormalizedChunks2D: TypeAlias = Tuple[Tuple[int, ...], Tuple[int, ...]]
