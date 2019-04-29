"""Duck-typed functions that call numpy or dask depending on the inputs
"""
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Union

import dask.array as da
import numpy as np
import scipy.stats
from .typing import Chunks2D


def array(x: Any) -> Union[np.ndarray, da.Array]:
    """Convert x to numpy array, unless it's a da.array
    """
    if isinstance(x, (np.ndarray, da.Array)):
        return x
    return np.array(x)


def _map_blocks(func: Callable[..., np.ndarray]
                ) -> Callable[..., Union[np.ndarray, da.Array]]:
    """Wrap an arbitrary function that takes one or more arrays in input.
    If any is a Dask Array, invoke :func:`dask.array.map_blocks`, otherwise
    apply the function directly.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if any(isinstance(arg, da.Array) for arg in args):
            return da.map_blocks(func, *args, **kwargs, dtype=args[0].dtype)
        return func(*args, **kwargs)
    return wrapper


def _map_blocks_df(func: Callable[[Any, Any], np.ndarray]
                   ) -> Callable[[Any, Any], Union[np.ndarray, da.Array]]:
    """Specialized variant for functions with degrees of freedom - adds
    auto-chunking in case of mismatched arguments
    """
    @wraps(func)
    def wrapper(x, df):
        x = array(x)
        df = array(df)
        if isinstance(x, da.Array) or isinstance(df, da.Array):
            # map_blocks auto-broadcasting broken since dask 1.1
            # https://github.com/dask/dask/issues/4739
            x, df = da.broadcast_arrays(x, df)
            return da.map_blocks(func, x, df, dtype=float)
        return func(x, df)
    return wrapper


def _toplevel(func_name: str) -> Callable[..., Union[np.ndarray, da.Array]]:
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


norm_cdf = _map_blocks(scipy.stats.norm.cdf)
norm_ppf = _map_blocks(scipy.stats.norm.ppf)
chi2_cdf = _map_blocks_df(scipy.stats.chi2.cdf)
chi2_ppf = _map_blocks_df(scipy.stats.chi2.ppf)
t_cdf = _map_blocks_df(scipy.stats.t.cdf)
t_ppf = _map_blocks_df(scipy.stats.t.ppf)
dot = _toplevel('dot')
sqrt = _toplevel('sqrt')
where = _toplevel('where')


class RandomState:
    """Wrapper around :class:`numpy.random.RandomState` and
    :class:`dask.array.random.RandomState`.

    For each method, if chunks=None invoke the numpy version, otherwise invoke
    the dask version.
    """
    def __init__(self, seed: Optional[int] = None):
        self._dask_state = da.random.RandomState(seed)

    @property
    def _numpy_state(self) -> np.random.RandomState:
        return self._dask_state._numpy_state

    def seed(self, seed: Optional[int] = None) -> None:
        self._dask_state.seed(seed)

    def _apply(self, func_name: str, size: Optional[Tuple[int, int]] = None,
               chunks: Chunks2D = None):
        if chunks is not None:
            func = getattr(self._dask_state, func_name)
            return func(size=size, chunks=chunks)
        else:
            func = getattr(self._numpy_state, func_name)
            return func(size=size)

    def uniform(self, size: Optional[Tuple[int, int]] = None,
                chunks: Chunks2D = None):
        return self._apply('uniform', size=size, chunks=chunks)

    def standard_normal(self, size: Optional[Tuple[int, int]] = None,
                        chunks: Chunks2D = None):
        return self._apply('standard_normal', size=size, chunks=chunks)
