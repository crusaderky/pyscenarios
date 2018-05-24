"""Duck-typed functions that call numpy or dask depending on the inputs
"""
import dask.array as da
import numpy as np
import scipy.stats
from functools import wraps


def _map_blocks(func):
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


def _map_blocks_df(func):
    """Specialized variant for functions with degrees of freedom - adds
    auto-chunking in case of mismatched arguments
    """
    @wraps(func)
    def wrapper(x, df):
        if not isinstance(x, da.Array):
            x = np.array(x)
        if not isinstance(df, da.Array):
            df = np.array(df)
        if isinstance(x, da.Array) and isinstance(df, np.ndarray):
            df = da.from_array(df, chunks=(x.chunks[-1], ))
        if isinstance(x, np.ndarray) and isinstance(df, da.Array):
            xchunks = tuple((s, ) for s in x.shape[:-1]) + df.chunks
            x = da.from_array(x, chunks=xchunks)
        if isinstance(x, da.Array) or isinstance(df, da.Array):
            return da.map_blocks(func, x, df, dtype=float)
        return func(x, df)
    return wrapper


def _toplevel(func_name):
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


class RandomState:
    """Wrapper around :class:`numpy.random.RandomState` and
    :class:`dask.array.random.RandomState`.

    For each method, if chunks=None invoke the numpy version, otherwise invoke
    the dask version.
    """
    def __init__(self, seed=None):
        self._dask_state = da.random.RandomState(seed)

    @property
    def _numpy_state(self):
        return self._dask_state._numpy_state

    def seed(self, seed=None):
        self._dask_state.seed(seed)

    def _apply(self, func_name, *args, chunks=None, **kwargs):
        if chunks:
            func = getattr(self._dask_state, func_name)
            return func(*args, **kwargs, chunks=chunks)
        else:
            func = getattr(self._numpy_state, func_name)
            return func(*args, **kwargs)

    def standard_normal(self, size=None, chunks=None):
        return self._apply('standard_normal', size=size, chunks=chunks)

    def chisquare(self, df, size=None, chunks=None):
        return self._apply('chisquare', df=df, size=size, chunks=chunks)
