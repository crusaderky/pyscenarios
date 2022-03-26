"""Sobol sequence generator

This is a reimplementation of a C++ algorithm by
`Stephen Joe and Frances Y. Kuo <http://web.maths.unsw.edu.au/~fkuo/sobol/>`_.
Directions are based on :file:`new-joe-kuo-6.21201` from the URL above.
"""
from __future__ import annotations

import lzma
import pkgutil
from typing import cast

import dask.array as da
import numpy as np
from dask.array.core import normalize_chunks
from numba import jit

from pyscenarios.typing import Chunks2D, NormalizedChunks2D

DIRECTIONS = "new-joe-kuo-6.21201.txt.xz"


_v_cache = None


def _load_v() -> np.ndarray:
    """Load V from the original author's file. This function is executed
    automatically the first time you call the :func:`sobol` function.
    When using a dask backend, the V array is only loaded when
    actually needed by the kernel; this results in smaller pickle files.
    When using dask distributed, V is loaded locally on the workers instead of
    being transferred over the network.
    """
    global _v_cache
    if _v_cache is None:
        directions = _load_directions(DIRECTIONS)
        _v_cache = _calc_v(directions)
    return _v_cache


def _load_directions(resource_fname: str) -> np.ndarray:
    """Load input file containing direction numbers.
    The file must one of those available on the website of the
    original author, or formatted like one.

    :returns:
        contents of the input file as a 2D numpy array.
        Every row contains the information for the matching dimension.
        Column 0 contains the a values, while columns 1+ contain the m values.
        The m values are padded on the right with zeros.
    """
    data_bin = pkgutil.get_data("pyscenarios", resource_fname)
    assert data_bin
    data_bin = lzma.decompress(data_bin)
    data_txt = data_bin.decode("ascii")
    del data_bin
    rows = [row.split() for row in data_txt.splitlines()]
    del data_txt

    # Add padding at end of rows
    # Drop first 2 columns
    # Replace header with element for d=1
    rowlen = len(rows[-1])
    for row in rows:
        row[:] = row[2:] + ["0"] * (rowlen - len(row))
    rows[0] = ["0"] + ["1"] * (rowlen - 3)
    return np.array(rows, dtype="uint32")


@jit("uint32[:,:](uint32[:,:])", nopython=True, nogil=True, cache=True)
def _calc_v(directions: np.ndarray) -> np.ndarray:
    """Calculate V matrix from directions"""
    # Initialise temp array of direction numbers
    v = np.empty((directions.shape[0], 32), dtype=np.uint32)

    for j in range(directions.shape[0]):
        s = 0
        # Compute direction numbers
        for s in range(directions.shape[1] - 1):
            if directions[j, s + 1] == 0:
                break
            v[j, s] = directions[j, s + 1] * 2 ** (31 - s)
        else:
            # need a C-style for loop
            # for(s=0; s<m.size; s++)
            # where at the end of the loop s == m.size
            s += 1

        for t in range(s, 32):
            v[j, t] = v[j, t - s] ^ (v[j, t - s] // 2**s)
            for k in range(1, s):
                v[j, t] ^= ((directions[j, 0] // 2 ** (s - 1 - k)) & 1) * v[j, t - k]

    return v


def _sobol_kernel(samples: int, dimensions: int, s0: int, d0: int) -> np.ndarray:
    """Numba kernel for :func:`sobol`

    :returns:
        points 2D array.

        points[i, j] = the jth component of the ith point
        with i indexed from 0 to N-1 and j indexed from 0 to D-1
    """
    output = np.empty((samples, dimensions), order="F")
    _sobol_kernel_jit(samples, dimensions, s0, d0, _load_v(), output)
    return output


@jit(
    "void(uint32, uint32, uint32, uint32, uint32[:, :], float64[:, :])",
    nopython=True,
    nogil=True,
    cache=True,
)
def _sobol_kernel_jit(
    samples: int, dimensions: int, s0: int, d0: int, V: np.ndarray, output: np.ndarray
) -> None:
    """Jit-compiled core of :func:`_sobol_kernel

    When running in dask and there are multiple chunks on the
    samples, calculate and then discard the first s0 samples.
    This is inefficient but preferable to transferring a state
    vector across the graph, which would introduce cross-chunks dependencies.
    """
    for j in range(dimensions):
        state = 0
        for i in range(s0 + samples):
            # c = index from the right of the first zero bit of i
            c = 0
            mask = 1
            while i & mask:
                mask *= 2
                c += 1

            state ^= V[j + d0, c]
            if i >= s0:
                output[i - s0, j] = np.double(state) / np.double(2**32)


def sobol(
    size: int | tuple[int, int], d0: int = 0, chunks: Chunks2D = None
) -> np.ndarray | da.Array:
    """Sobol points generator based on Gray code order

    :param size:
        number of samples (cannot be greater than :math:`2^{32}`) to extract
        from a single dimension, or tuple (samples, dimensions).
        To guarantee uniform distribution, the number of samples should
        always be :math:`2^{n} - 1`.
    :param int d0:
        first dimension. This can be used as a functional equivalent of a
        a random seed. dimensions + d0 can't be greater than
        :func:`max_sobol_dimensions()` - 1.
    :param chunks:
        If None, return a numpy array.

        If set, return a dask array with the given chunk size.
        It can be anything accepted by dask (a positive integer, a
        tuple of two ints, or a tuple of two tuples of ints) for the output
        shape (see result below). e.g. either ``(16384, 50)`` or
        ``((16384, 16383),  (50, 50, 50))`` could be used together with
        ``size=(32767, 150)``.

        .. note::
           The algorithm is not efficient if there are multiple chunks on axis
           0. However, if you do need them, it is typically better to require
           them here than re-chunking afterwards, particularly if (most of) the
           subsequent algorithm is embarassingly parallel.
    :returns:
        If size is an int, a 1-dimensional array of samples.
        If size is a tuple, a 2-dimensional array POINTS, where
        ``POINTS[i, j]`` is the ith sample of the jth dimension.
        Each dimension is a uniform (0, 1) distribution.
    :rtype:
        If chunks is not None, :class:`dask.array.Array`; else
        :class:`numpy.ndarray`
    """
    if isinstance(size, int):
        samples = size
        dimensions = 1
    else:
        samples, dimensions = size

    if not 0 < samples < 2**32:
        raise ValueError("samples must be between 1 and 2^32")
    if not 0 < dimensions + d0 <= max_sobol_dimensions():
        raise ValueError(
            "(dimensions + d0) must be between 1 and %d" % max_sobol_dimensions()
        )

    if chunks is None:
        res = _sobol_kernel(samples, dimensions, 0, d0)
        if isinstance(size, int):
            res = res[:, 0]
        return res

    # dask-specific code
    chunks = cast(
        NormalizedChunks2D, normalize_chunks(chunks, shape=(samples, dimensions))
    )
    name = "sobol-%d-%d-%d" % (samples, dimensions, d0)
    dsk = {}

    offset_i = 0
    for i, size_i in enumerate(chunks[0]):
        offset_j = 0
        for j, size_j in enumerate(chunks[1]):
            dsk[name, i, j] = (_sobol_kernel, size_i, size_j, offset_i, d0 + offset_j)
            offset_j += size_j
        offset_i += size_i

    res = da.Array(dsk, name=name, dtype=float, chunks=chunks)
    if isinstance(size, int):
        res = res[:, 0]
    return res


def max_sobol_dimensions() -> int:
    """Return number of dimensions available. When invoking :func:`sobol`,
    ``size[1] + d0`` must be smaller than this.
    """
    return _load_v().shape[0]


def max_dimensions() -> int:
    import warnings

    warnings.warn(
        "pyscenarios.sobol.max_dimensions has been moved to "
        "pyscenarios.max_sobol_dimensions",
        DeprecationWarning,
    )
    return max_sobol_dimensions()
