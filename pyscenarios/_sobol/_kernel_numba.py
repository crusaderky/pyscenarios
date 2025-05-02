import numpy as np
import numpy.typing as npt
from numba import jit

from pyscenarios._sobol._vmatrix import V


def sobol_kernel(
    samples: int, dimensions: int, s0: int, d0: int
) -> npt.NDArray[np.float64]:
    """Numba kernel for :func:`sobol`

    :returns:
        points 2D array.

        points[i, j] = the jth component of the ith point
        with i indexed from 0 to N-1 and j indexed from 0 to D-1
    """
    output = np.empty((samples, dimensions), dtype=np.float64, order="F")
    _sobol_kernel_jit(samples, dimensions, s0, d0, V, output)
    return output


@jit(
    "void(uint32, uint32, uint32, uint32, uint32[:, :], float64[:, :])",
    nopython=True,
    nogil=True,
    cache=True,
)
def _sobol_kernel_jit(
    samples: int,
    dimensions: int,
    s0: int,
    d0: int,
    V: npt.NDArray[np.uint32],
    output: npt.NDArray[np.float64],
) -> None:  # coverage: ignore
    """Jit-compiled core of sobol_kernel

    When running in Dask, if there are multiple chunks on the
    samples, calculate and then discard the first s0 samples.

    This is inefficient but preferable to transferring the final state
    over the network, as in most cases it would cause nodes to sit idle
    until the state arrives from the node left of them.
    """
    c = np.empty(s0 + samples, dtype=np.uint8)
    for i in range(s0 + samples):
        # ci = index from the right of the first zero bit of i
        ci = 0
        while i & (1 << ci):
            ci += 1
        c[i] = ci

    for j in range(dimensions):
        state = 0
        for i in range(s0):
            state ^= V[j + d0, c[i]]
        for i in range(s0, s0 + samples):
            state ^= V[j + d0, c[i]]
            output[i - s0, j] = np.double(state) / np.double(2**32)
