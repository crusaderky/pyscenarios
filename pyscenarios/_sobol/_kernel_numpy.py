import math

import numpy as np
import numpy.typing as npt

from pyscenarios._sobol._vmatrix import V


def sobol_kernel(
    samples: int, dimensions: int, s0: int, d0: int
) -> npt.NDArray[np.float64]:
    """Pure NumPy variant kernel for :func:`sobol`, when Numba is not available.

    :returns:
        points 2D array.

        points[i, j] = the jth component of the ith point
        with i indexed from 0 to N-1 and j indexed from 0 to D-1
    """
    c = _calc_c(s0 + samples)

    # NOTE: transposition order is crucial for performance.
    # dimensions must be on axis 1 **in C order** to allow for
    # efficient SIMD auto-vectorization.
    # This would be *much* slower:
    # states = xp.take(V[d0:d0 + dimensions, :], c, axis=1).T
    VT = V[d0 : d0 + dimensions, :].T

    # On Dask, recalculate the initial state from the samples of previous chunks.
    # This is inefficient but preferable to transferring the final state
    # over the network, as in most cases it would cause nodes to sit idle
    # until the state arrives from the node left of them.
    prev_state = None
    scratch = np.empty((samples, VT.shape[1]), dtype=np.uint32)

    for i in range(0, s0, samples):
        if i + samples <= s0:
            states = np.take(VT, c[i : i + samples], axis=0, out=scratch)
        else:
            # Final chunk is shorter than the others, or
            # explicitly requested odd-sized chunks, e.g. (5, 6, 4).
            states = np.take(VT, c[i:s0], axis=0, out=scratch[: s0 - i])  # type: ignore[arg-type]

        if prev_state is not None:
            states[0, :] ^= prev_state
        prev_state = np.bitwise_xor.reduce(states, axis=0, out=prev_state)

    states = np.take(VT, c[s0:], axis=0, out=scratch)
    if prev_state is not None:
        states[0, :] ^= prev_state

    states = np.bitwise_xor.accumulate(states, axis=0, out=scratch)  # type: ignore[assignment]
    return states.astype(np.float64) / 2**32


def _calc_c(samples: int) -> npt.NDArray[np.intp]:
    """c[i] = index from the right of the first zero bit of sample index i"""
    samples_range = np.arange(samples, dtype=np.intp)
    c_max = int(math.log(samples, 2))
    out = np.full(samples, c_max, dtype=np.intp)
    for c in range(c_max + 1, -1, -1):
        mask = samples_range & (1 << c) == 0
        out = np.where(mask, c, out)  # type: ignore[assignment]
    return out
