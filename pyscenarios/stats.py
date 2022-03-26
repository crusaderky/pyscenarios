"""Statistical functions
"""
from __future__ import annotations

from typing import Any, TypeVar

import dask.array as da
import numpy as np
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from numba import guvectorize

from pyscenarios import duck

T = TypeVar("T", np.ndarray, da.Array)


def tail_dependence(x: Any, y: Any, q: Any) -> np.ndarray | da.Array:
    r"""Calculate `tail dependence
    <https://en.wikipedia.org/wiki/Tail_dependence>`_
    between vectors x and y.

    :param x:
        1D array-like or dask array containing samples from a
        uniform (0, 1) distribution.
    :param y:
        other array to compare against
    :param q:
        quantile(s) (0 < q < 1).
        Either a scalar or a ND array-like or dask array.
    :returns:
        array of the same shape and type as q, containing:

        .. math::

            \cases{
                P(y < q | x < q) | q < 0.5 \cr
                P(y \geq q | x \geq q) | q \geq 0.5
            }
    """
    x = duck.array(x)
    y = duck.array(y)
    q = duck.array(q)

    assert x.size == y.size
    assert x.ndim == y.ndim == 1

    q = q[..., np.newaxis]
    x_lt_q = x < q
    x_ge_q = x >= q
    xcount = duck.where(q < 0.5, x_lt_q, x_ge_q)
    ltail = x_lt_q & (y < q)
    htail = x_ge_q & (y >= q)
    tail = duck.where(q < 0.5, ltail, htail)

    return tail.sum(axis=-1) / xcount.sum(axis=-1)


def clusterization(samples: T) -> T:
    """Given an array of m points in a n-dimensional space with domain [0, 1) along each
    dimension, where each row of the array indicates the cartesian coordinates of each
    point, calculate the measure of how clustered each point is to its neighbours:

    #. For each sample, calculate the r^2 from each other sample:
       sum[i=0...d] (xi - yi)^2
    #. Then sum[j=0...s] 1 / rj^2
    #. Finally, normalize the output by dividing it by its own median

    :param samples:
        numpy or dask array with dimensions (sample, dimension)
    :returns:
        array with dimensions (sample, )

    TODO This is a quadratic algorithm. It's possible to define a linear approximation:
         #. group points into n-dimensional buckets
         #. calculate interaction only between points inside a bucket and its neighbours
    """
    samples2 = samples.reshape((samples.shape[0], 1, -1))

    if isinstance(samples, np.ndarray):
        rq = square_radius(samples, samples2)
        # The diagonal (squared distance between a point and itself) is now full of
        # zeros; set it to inf so that 1 / r^2 = 0.
        eye = np.eye(rq.shape[0], dtype=bool)
        irq = 1.0 / np.where(eye, np.inf, rq)

    elif isinstance(samples, da.Array):
        rq = da.map_blocks(
            square_radius,
            samples,
            samples2,
            keepdims=True,
            chunks=(samples.chunks[0], samples.chunks[0], (1,) * samples.numblocks[1]),
            dtype="f8",
        ).sum(axis=-1)
        eye = da.eye(rq.shape[0], chunks=rq.chunks[0][0])
        rq_zero_eye = da.where(eye, np.inf, rq)

        # dask-specific optimizations:
        # - Do not apply the eye filter on chunks that aren't along the diagonal
        # - Do not run the square_radius kernel for chunks that are below the diagonal
        # This relies on the fact that dask optimizer drops unused chunks.
        #
        # The whole paragraph is functionally identical to
        # irq = 1.0 / rq_zero_eye

        irq_base = 1.0 / rq
        irq_t = irq_base.T
        irq_zero_eye = 1.0 / rq_zero_eye

        dsk = {}
        name = "irq_oz-" + tokenize(rq)
        for i in range(rq.numblocks[0]):
            for j in range(rq.numblocks[1]):
                if i > j:
                    parent = irq_base
                elif i < j:
                    parent = irq_t
                else:
                    parent = irq_zero_eye
                dsk[name, i, j] = (parent.name, i, j)

        irq = da.Array(
            dask=HighLevelGraph.from_collections(
                name, dsk, dependencies=[irq_base, irq_t, irq_zero_eye]
            ),
            name=name,
            chunks=rq.chunks,
            dtype=float,
        )
        # End optimizations

    else:
        raise TypeError(samples)

    out = irq.sum(axis=1)
    return out / np.median(out, axis=0)


@guvectorize(["f8[:],f8[:],f8[:]"], "(d),(d)->()", nopython=True, cache=True)
def square_radius(x: np.ndarray, y: np.ndarray, out: np.ndarray) -> None:
    """Numba kernel of :func:`clusterization`.

    Given 2 points in hyperspace X(x1, x2, ... xi) and Y(y1, y2, ... yi),
    calculate the squared distance between the two.
    Assume that all points are in a [0, 1] space that wraps around the corners.
    """
    acc = 0.0
    for i in range(x.size):
        di = abs(x[i] - y[i])
        if di > 0.5:
            di = 1.0 - di
        acc += di**2
    out[0] = acc
