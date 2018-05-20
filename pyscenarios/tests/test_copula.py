import pytest
import numpy as np
from numpy.testing import assert_allclose
from pyscenarios.copula import gaussian_copula, t_copula

cov = [[1., 0.9, 0.7],
       [0.9, 1., 0.4],
       [0.7, 0.4, 1.]]


def test_gaussian_mersenne_np():
    actual = gaussian_copula(cov, scenarios=4, seed=123, chunks=None,
                             rng='Mersenne Twister')

    expect = [[-1.08563060, -1.50629471, -2.42667924, -0.86674040],
              [-0.54233474, -1.60787125, -2.37097000, -1.07598597],
              [-1.15002017,  0.04561073, -0.86315499, -0.29407627]]
    assert_allclose(expect, actual, 1e-6, 0)
    assert isinstance(actual, np.ndarray)


def test_gaussian_mersenne_da():
    actual = gaussian_copula(cov, scenarios=4, seed=123, chunks=2,
                             rng='Mersenne Twister')
    expect = [[1.61739599,  0.00936110, -0.88414686, -1.63883139],
              [1.34668479,  0.77307164, -1.07582330, -1.21794817],
              [0.85965138, -0.65639660, -0.47076357, -1.47348213]]
    assert_allclose(expect, actual, 1e-6, 0)
    assert actual.chunks == ((2, 1), (2, 2))


@pytest.mark.parametrize('chunks,expect_chunks', [
    (None, None),
    (2, ((2, 1), (2, 2))),
    (((2, 1), (3, 1)), ((2, 1), (3, 1))),
])
def test_gaussian_sobol(chunks, expect_chunks):
    actual = gaussian_copula(cov, scenarios=4, seed=123, chunks=chunks,
                             rng='SOBOL')
    expect = [[0., -0.67448975, 0.67448975,  0.31863936],
              [0., -0.31303751, 0.31303751,  0.78820110],
              [0., -1.15262386, 1.15262386, -0.53727912]]
    assert_allclose(expect, actual, 1e-6, 0)
    if chunks:
        assert actual.chunks == expect_chunks
    else:
        assert isinstance(actual, np.ndarray)


def test_student_t_mersenne_np():
    actual = t_copula(cov, df=3, scenarios=4, seed=123, chunks=None,
                      rng='Mersenne Twister')
    print(actual)
    expect = [[-1.08563060, -1.50629471, -2.42667924, -0.86674040],
              [-0.54233474, -1.60787125, -2.37097000, -1.07598597],
              [-1.15002017,  0.04561073, -0.86315499, -0.29407627]]
    assert_allclose(expect, actual, 1e-6, 0)
    assert isinstance(actual, np.ndarray)


def test_student_t_mersenne_da():
    actual = t_copula(cov, df=3, scenarios=4, seed=123, chunks=2,
                      rng='Mersenne Twister')
    print(actual)
    expect = [[1.61739599,  0.00936110, -0.88414686, -1.63883139],
              [1.34668479,  0.77307164, -1.07582330, -1.21794817],
              [0.85965138, -0.65639660, -0.47076357, -1.47348213]]
    assert_allclose(expect, actual, 1e-6, 0)
    assert actual.chunks == ((2, 1), (2, 2))


@pytest.mark.parametrize('chunks,expect_chunks', [
    (None, None),
    (2, ((2, 1), (2, 2))),
    (((2, 1), (3, 1)), ((2, 1), (3, 1))),
])
def test_student_t_sobol(chunks, expect_chunks):
    actual = t_copula(cov, df=3, scenarios=4, seed=123, chunks=chunks,
                      rng='SOBOL')
    print(actual)
    expect = [[0., -0.67448975, 0.67448975,  0.31863936],
              [0., -0.31303751, 0.31303751,  0.78820110],
              [0., -1.15262386, 1.15262386, -0.53727912]]
    assert_allclose(expect, actual, 1e-6, 0)
    if chunks:
        assert actual.chunks == expect_chunks
    else:
        assert isinstance(actual, np.ndarray)