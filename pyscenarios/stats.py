"""Statistical functions"""

import dask.array as da
import numpy.typing as npt

from pyscenarios import duck


def tail_dependence(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    q: npt.ArrayLike,
) -> npt.NDArray | da.Array:
    r"""Calculate `tail dependence
    <https://en.wikipedia.org/wiki/Tail_dependence>`_
    between vectors x and y.

    :param x:
        1D array-like or Dask array containing samples from a
        uniform (0, 1) distribution.
    :param y:
        other 1D array-like or Dask array to compare against
    :param q:
        quantile(s) (`0 < q < 1`).
        Either a scalar or an array-like or Dask array.
    :returns:
        Array of the same shape and type as `q`, containing:

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

    q = q[..., None]
    x_lt_q = x < q
    x_ge_q = x >= q
    xcount = duck.where(q < 0.5, x_lt_q, x_ge_q)
    ltail = x_lt_q & (y < q)
    htail = x_ge_q & (y >= q)
    tail = duck.where(q < 0.5, ltail, htail)

    return tail.sum(axis=-1) / xcount.sum(axis=-1)
