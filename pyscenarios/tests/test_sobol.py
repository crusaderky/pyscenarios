import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyscenarios import max_sobol_dimensions, sobol
from pyscenarios.sobol import max_dimensions
from pyscenarios.tests import requires_jit

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
    with pytest.raises(ValueError):
        sobol((4, 202), d0=21000)
    assert sobol(4, d0=21200).shape == (4,)
    with pytest.raises(ValueError):
        sobol(4, d0=21201)


def test_max_dimensions():
    with pytest.warns(DeprecationWarning):
        assert max_dimensions() == 21201


def test_numpy_1d():
    output = sobol(15, d0=123)
    assert_array_equal(EXPECT[:, 0], output)


def test_dask_1d():
    output = sobol(15, d0=123, chunks=(10, 3))
    assert output.chunks == ((10, 5),)
    assert_array_equal(EXPECT[:, 0], output.compute())


def test_numpy_2d():
    output = sobol((15, 4), d0=123)
    assert_array_equal(EXPECT, output)


def test_dask_2d():
    output = sobol((15, 4), d0=123, chunks=(10, 3))
    assert output.chunks == ((10, 5), (3, 1))
    assert_array_equal(EXPECT, output.compute())


@requires_jit
@pytest.mark.parametrize("n", list(range(8, 13)))
def test_samepoints(n):
    """Given exactly 2^n-1 samples, all series produce exactly the same
    points in different order
    """
    s = sobol((2**n - 1, max_sobol_dimensions()), chunks=(2**n - 1, 2000))
    s = s.map_blocks(np.sort, axis=0)
    s = s.T - s[:, 0]
    assert not s.any()


@pytest.mark.parametrize("n", [-1, 0, int(2**33)])
def test_bad_samples(n):
    with pytest.raises(ValueError) as e:
        sobol(n)
    assert str(e.value) == "samples must be between 1 and 2^32"
