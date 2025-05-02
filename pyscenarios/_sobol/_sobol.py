from functools import cache
from typing import overload

import dask.array as da
import numpy as np
import numpy.typing as npt
from dask.array.core import normalize_chunks

from pyscenarios.typing import Chunks2D, NormalizedChunks2D


@cache
def _use_numba() -> bool:
    """Check if Numba is available"""
    try:
        import numba  # noqa: F401

        return True
    except ImportError:
        return False


def sobol_kernel(
    samples: int, dimensions: int, s0: int, d0: int
) -> npt.NDArray[np.float64]:
    """Kernel for :func:`sobol`. Uses Numba is available, otherwise
    falls back to a pure NumPy implementation.

    :returns:
        points 2D array.

        points[i, j] = the jth component of the ith point
        with i indexed from 0 to N-1 and j indexed from 0 to D-1

    .. note::
        This import design allows

        - having Numba installed on the Dask client, but not on the workers
        - not importing Numba if not needed; e.g. when using
        pyscenarios.copula with rng="Mersenne Twister"
        - Unit testing both implementations while Numba is installed
    """
    if _use_numba():
        import pyscenarios._sobol._kernel_numba as kernel
    else:
        import pyscenarios._sobol._kernel_numpy as kernel  # type: ignore[no-redef]

    return kernel.sobol_kernel(samples, dimensions, s0, d0)


@overload
def sobol(
    size: int | tuple[int, int], d0: int = 0, *, chunks: None = None
) -> npt.NDArray[np.float64]: ...


@overload
def sobol(
    size: int | tuple[int, int], d0: int = 0, *, chunks: Chunks2D
) -> da.Array: ...


def sobol(
    size: int | tuple[int, int], d0: int = 0, *, chunks: Chunks2D | None = None
) -> npt.NDArray[np.float64] | da.Array:
    """Sobol points generator based on Gray code order

    This is a Python reimplementation of a C++ algorithm by
    `Stephen Joe and Frances Y. Kuo <https://web.maths.unsw.edu.au/~fkuo/sobol/>`_,
    using directions from the file ``new-joe-kuo-6.21201`` linked above.

    :param int | tuple[int, int] size:
        number of samples (cannot be greater than :math:`2^{32}`) to extract
        from a single dimension, or tuple (samples, dimensions).
        To guarantee uniform distribution, the number of samples should
        always be :math:`2^{n} - 1`.
    :param int d0:
        first dimension. This can be used as a functional equivalent of a
        a random seed. dimensions + d0 can't be greater than
        :func:`max_sobol_dimensions()` - 1.
    :param chunks:
        If omitted or None, return a NumPy array.

        If not None, return a Dask array with the given chunk size.
        It can be anything accepted by Dask (a positive integer, a
        tuple of two ints, or a tuple of two tuples of ints) for the output
        shape (see result below). e.g. either ``(16384, 50)`` or
        ``((16384, 16383),  (50, 50, 50))`` could be used together with
        ``size=(32767, 150)``.
    :returns:
        If size is an int, a 1-dimensional array of samples.
        If size is a tuple, a 2-dimensional array POINTS, where
        ``POINTS[i, j]`` is the ith sample of the jth dimension.
        Each dimension is a uniform (0, 1) distribution.
    :rtype:
        If chunks is not None, :class:`dask.array.Array`; else
        :class:`numpy.ndarray`

    .. note::
        This function will try accelerating the calculation with
        `numba <http://numba.pydata.org>`_ if installed, and fall back
        to a slower pure NumPy implementation otherwise.
    """
    if isinstance(size, int):
        samples = size
        dimensions = 1
    else:
        samples, dimensions = size

    if not 0 < samples < 2**32:
        raise ValueError("samples must be between 1 and 2**32")
    if not 0 < dimensions + d0 <= max_sobol_dimensions():
        raise ValueError(
            f"(dimensions + d0) must be between 1 and {max_sobol_dimensions()}"
        )

    if chunks is None:
        np_res = sobol_kernel(samples, dimensions, 0, d0)
        if isinstance(size, int):
            np_res = np_res[:, 0]
        return np_res

    # Dask-specific code
    norm_chunks: NormalizedChunks2D = normalize_chunks(
        chunks, shape=(samples, dimensions)
    )
    name = f"sobol-{samples}-{dimensions}-{d0}"
    dsk = {}

    offset_i = 0
    for i, size_i in enumerate(norm_chunks[0]):
        offset_j = 0
        for j, size_j in enumerate(norm_chunks[1]):
            dsk[name, i, j] = (sobol_kernel, size_i, size_j, offset_i, d0 + offset_j)
            offset_j += size_j
        offset_i += size_i

    da_res = da.Array(dsk, name=name, dtype=float, chunks=norm_chunks)
    if isinstance(size, int):
        da_res = da_res[:, 0]
    return da_res


def max_sobol_dimensions() -> int:
    """Return number of dimensions available. When invoking :func:`sobol`,
    ``size[1] + d0`` must be smaller than this.
    """
    # Expensive import; don't do it until we need it
    from pyscenarios._sobol._vmatrix import V

    return V.shape[0]
