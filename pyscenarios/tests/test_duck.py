import pytest
import dask.array as da
import numpy as np
import scipy.stats
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from pyscenarios import duck


def test_array():
    for x in (1, [1, 2], np.array(1), np.array([1, 2])):
        y = duck.array(x)
        assert isinstance(y, np.ndarray)
        assert_equal(x, y)

    x = da.arange(3, chunks=2)
    y = duck.array(x)
    assert x is y


@pytest.mark.parametrize('chunk', [False, True])
@pytest.mark.parametrize('func,wrapped', [
    (duck.norm_cdf, scipy.stats.norm.cdf),
    (duck.norm_ppf, scipy.stats.norm.ppf),
    (duck.sqrt, np.sqrt),
])
def test_map_blocks(func, wrapped, chunk):
    x = np.random.rand(10)
    y = wrapped(x)

    if chunk:
        dx = da.from_array(x, chunks=5)
    else:
        dx = x
    dy = func(dx)

    assert_array_equal(y, dy)
    if chunk:
        assert dy.chunks == ((5, 5), )
    else:
        assert isinstance(dy, np.ndarray)


@pytest.mark.parametrize('df,chunk_df', [
    (3, False),
    ([1, 2, 3], False),
    (np.array([1, 2, 3]), False),
    (np.array([1, 2, 3]), True)
])
@pytest.mark.parametrize('x,chunk_x', [
    (np.random.rand(4).reshape(4, 1), False),
    (np.random.rand(4).reshape(4, 1), True),
    # dask fails to broadcast a np.ndarray with shape[i] != 1
    # against a da.Array with len(chunks[i]) != 1
    pytest.param(np.random.rand(30).reshape(10, 3), False,
                 marks=pytest.mark.xfail),
    pytest.param(np.random.rand(30).reshape(10, 3), True,
                 marks=pytest.mark.xfail),
    pytest.param([[1, 2, 3], [4, 5, 6]], False,
                 marks=pytest.mark.xfail),
])
@pytest.mark.parametrize('func,wrapped', [
    (duck.chi2_cdf, scipy.stats.chi2.cdf),
    (duck.chi2_ppf, scipy.stats.chi2.ppf),
    (duck.t_cdf, scipy.stats.t.cdf),
    (duck.t_ppf, scipy.stats.t.ppf),
])
def test_map_blocks_df(func, wrapped, x, chunk_x, df, chunk_df):
    y = wrapped(x, df)

    if chunk_x:
        dx = da.from_array(x, chunks=2)
    else:
        dx = x
    if chunk_df:
        ddf = da.from_array(df, chunks=2)
    else:
        ddf = df

    dy = func(dx, ddf)

    assert_array_equal(y, dy)
    if chunk_x or chunk_df:
        assert isinstance(dy, da.Array)
    else:
        assert isinstance(dy, np.ndarray)


@pytest.mark.parametrize('chunk', [False, True])
def test_dot(chunk):
    x = np.random.rand(16).reshape(4, 4)
    y = np.random.rand(24).reshape(4, 6)
    z = np.dot(x, y)

    if chunk:
        dx = da.from_array(x, chunks=2)
        dy = da.from_array(y, chunks=2)
    else:
        dx = x
        dy = y
    dz = duck.dot(dx, dy)
    assert_allclose(z, dz, 1e-12, 0)
    if chunk:
        assert dz.chunks == ((2, 2), (2, 2, 2))
    else:
        assert isinstance(dz, np.ndarray)


@pytest.mark.parametrize('chunk', [False, True])
def test_where(chunk):
    x = np.array([1, 2, 3])
    y = np.array([0, 5, 2])
    z = duck.where(x > y, x, y)

    if chunk:
        dx = da.from_array(x, chunks=2)
        dy = da.from_array(y, chunks=2)
    else:
        dx = x
        dy = y
    dz = duck.where(dx > dy, dx, dy)
    assert_array_equal(z, dz)
    if chunk:
        assert dz.chunks == ((2, 1),)
    else:
        assert isinstance(dz, np.ndarray)


@pytest.mark.parametrize('chunks', [None, 2, ((2, 1), (2, 1))])
def test_randomstate_uniform(chunks):
    state = duck.RandomState(123)
    ref_np = np.random.RandomState(123)
    ref_da = da.random.RandomState(123)

    seq = state.uniform(size=(3, 3), chunks=chunks)
    if chunks:
        assert_array_equal(
            seq, ref_da.uniform(size=(3, 3), chunks=chunks))
        assert seq.chunks == ((2, 1), (2, 1))
    else:
        assert_array_equal(seq, ref_np.uniform(size=(3, 3)))
        assert isinstance(seq, np.ndarray)


@pytest.mark.parametrize('chunks', [None, 2, ((2, 1), (2, 1))])
def test_randomstate_standard_normal(chunks):
    state = duck.RandomState(123)
    ref_np = np.random.RandomState(123)
    ref_da = da.random.RandomState(123)

    seq = state.standard_normal(size=(3, 3), chunks=chunks)
    if chunks:
        assert_array_equal(
            seq, ref_da.standard_normal(size=(3, 3), chunks=chunks))
        assert seq.chunks == ((2, 1), (2, 1))
    else:
        assert_array_equal(seq, ref_np.standard_normal(size=(3, 3)))
        assert isinstance(seq, np.ndarray)
