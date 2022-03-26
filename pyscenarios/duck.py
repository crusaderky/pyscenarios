"""Duck-typed functions that call numpy or dask depending on the inputs
"""
from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

import dask.array as da
import numpy as np
import scipy.stats

from pyscenarios.typing import Chunks2D


def array(x: Any) -> np.ndarray | da.Array:
    """Convert x to numpy array, unless it's a da.array"""
    if isinstance(x, (np.ndarray, da.Array)):
        return x
    return np.array(x)


def _apply_unary(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Apply a function to an array-like. If the argument is a dask array, wrap
    it in :func:`dask.array.blockwise`
    """

    @wraps(func)
    def wrapper(x):
        if isinstance(x, da.Array):
            sig = tuple(range(x.ndim))
            return da.blockwise(func, sig, x, sig, dtype=float)
        return func(x)

    return wrapper


def _apply_binary(func: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
    """Apply a function to two array-likes. If either argument is a dask array,
    wrap it in :func:`dask.array.blockwise`
    """

    @wraps(func)
    def wrapper(x, y):
        x = array(x)
        y = array(y)
        if isinstance(x, da.Array) or isinstance(y, da.Array):
            out_ndim = max(x.ndim, y.ndim)
            return da.blockwise(
                func,
                tuple(range(out_ndim)),
                x,
                tuple(range(out_ndim - x.ndim, out_ndim)),
                y,
                tuple(range(out_ndim - y.ndim, out_ndim)),
                dtype=float,
            )
        return func(x, y)

    return wrapper


def _toplevel(func_name: str) -> Callable[..., np.ndarray | da.Array]:
    """If any of the args is a Dask Array, invoke da.func_name; else invoke
    np.func_name
    """

    def wrapper(*args, **kwargs):
        if any(isinstance(arg, da.Array) for arg in args):
            func = getattr(da, func_name)
        else:
            func = getattr(np, func_name)
        return func(*args, **kwargs)

    return wrapper


norm_ppf = _apply_unary(scipy.stats.norm.ppf)
chi2_ppf = _apply_binary(scipy.stats.chi2.ppf)
t_cdf = _apply_binary(scipy.stats.t.cdf)
sqrt = np.sqrt  # NEP-18 gufunc
where = _toplevel("where")


class RandomState:
    """Wrapper around :class:`numpy.random.RandomState` and
    :class:`dask.array.random.RandomState`.

    For each method, if chunks=None invoke the numpy version, otherwise invoke
    the dask version.
    """

    def __init__(self, seed: int | None = None):
        self._dask_state = da.random.RandomState(seed)

    @property
    def _numpy_state(self) -> np.random.RandomState:
        return self._dask_state._numpy_state

    def _apply(
        self,
        func_name: str,
        size: tuple[int, int] | None = None,
        chunks: Chunks2D = None,
    ):
        if chunks is not None:
            func = getattr(self._dask_state, func_name)
            return func(size=size, chunks=chunks)
        else:
            func = getattr(self._numpy_state, func_name)
            return func(size=size)

    def uniform(self, size: tuple[int, int] | None = None, chunks: Chunks2D = None):
        return self._apply("uniform", size=size, chunks=chunks)

    def standard_normal(
        self, size: tuple[int, int] | None = None, chunks: Chunks2D = None
    ):
        return self._apply("standard_normal", size=size, chunks=chunks)
