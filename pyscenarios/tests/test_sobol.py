import pickle

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyscenarios import max_sobol_dimensions, sobol


@pytest.fixture(params=["numpy", "numba"])
def kernel(request, monkeypatch):
    """Test both the Numba and NumPy implementations of the Sobol kernel"""
    if request.param == "numba":
        pytest.importorskip("numba")
    else:
        monkeypatch.setattr("pyscenarios._sobol._sobol._use_numba", lambda: False)


pytestmark = pytest.mark.usefixtures("kernel")


EXPECT = np.array(
    [
        [0.5, 0.5, 0.5, 0.5],
        [0.25, 0.75, 0.25, 0.25],
        [0.75, 0.25, 0.75, 0.75],
        [0.625, 0.875, 0.375, 0.125],
        [0.125, 0.375, 0.875, 0.625],
        [0.875, 0.125, 0.125, 0.375],
        [0.375, 0.625, 0.625, 0.875],
        [0.1875, 0.5625, 0.1875, 0.5625],
        [0.6875, 0.0625, 0.6875, 0.0625],
        [0.4375, 0.3125, 0.4375, 0.8125],
        [0.9375, 0.8125, 0.9375, 0.3125],
        [0.5625, 0.4375, 0.3125, 0.6875],
        [0.0625, 0.9375, 0.8125, 0.1875],
        [0.8125, 0.6875, 0.0625, 0.9375],
        [0.3125, 0.1875, 0.5625, 0.4375],
    ]
)


def test_max_sobol_dimensions():
    assert max_sobol_dimensions() == 21201
    assert sobol((4, 21201)).shape == (4, 21201)
    assert sobol((4, 201), d0=21000).shape == (4, 201)
    with pytest.raises(ValueError, match="must be between"):
        sobol((4, 202), d0=21000)
    assert sobol(4, d0=21200).shape == (4,)
    with pytest.raises(ValueError, match="must be between"):
        sobol(4, d0=21201)


def test_numpy_1d():
    output = sobol(15, d0=123)
    assert isinstance(output, np.ndarray)
    assert_array_equal(EXPECT[:, 0], output)


def test_numpy_2d():
    output = sobol((15, 4), d0=123)
    assert isinstance(output, np.ndarray)
    assert_array_equal(EXPECT, output)


@pytest.mark.parametrize(
    "chunks,expect_chunks",
    [
        (-1, ((15, ), )),
        (((-1, -1)), ((15, ), )),
        ((6, -1), ((6, 6, 3, ), )),
        ((6, 100), ((6, 6, 3, ), )),
        (((5, 6, 4), -1), ((5, 6, 4), )),
    ]
)
def test_dask_1d(chunks, expect_chunks):
    output = sobol(15, d0=123, chunks=chunks)
    assert output.chunks == expect_chunks
    assert_array_equal(EXPECT[:, 0], output.compute())


@pytest.mark.parametrize(
    "chunks,expect_chunks",
    [
        (-1, ((15, ), (4, ))),
        (100, ((15, ), (4, ))),
        ((-1, -1), ((15, ), (4, ))),
        ((100, 100), ((15, ), (4, ))),
        ((6, -1), ((6, 6, 3, ), (4, ))),
        ((-1, 3), ((15, ), (3, 1))),
        ((6, 3), ((6, 6, 3, ), (3, 1))),
        (((5, 6, 4), (1, 1, 2)), ((5, 6, 4), (1, 1, 2))),
    ]
)
def test_dask_2d(chunks, expect_chunks):
    output = sobol((15, 4), d0=123, chunks=chunks)
    assert output.chunks == expect_chunks
    assert_array_equal(EXPECT, output.compute())


@pytest.mark.parametrize("n", list(range(8, 13)))
def test_samepoints(n):
    """Given exactly 2^n-1 samples, all series produce exactly the same
    points in different order
    """
    # Use Dask to speed the test up
    s = sobol((2**n - 1, max_sobol_dimensions()), chunks=(-1, 2000))
    s = s.map_blocks(np.sort, axis=0)
    s = s.T - s[:, 0]
    assert not s.any()


@pytest.mark.parametrize("n", [-1, 0, (2**33)])
def test_bad_samples(n):
    with pytest.raises(ValueError, match="must be between"):
        sobol(n)


def test_dask_workers_without_numba():
    """Test that Sobol can run when the Dask client has Numba installed,
    but the workers do not.
    """
    output = sobol((15, 4), d0=123, chunks=(6, 3))
    pik = pickle.dumps(output)
    assert b"numba" not in pik
