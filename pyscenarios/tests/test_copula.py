import pytest
import numpy as np
from numpy.testing import assert_allclose
from pyscenarios.copula import gaussian_copula, t_copula

cov = [[1.0, 0.9, 0.7],
       [0.9, 1.0, 0.4],
       [0.7, 0.4, 1.0]]


def test_gaussian_mersenne_np():
    actual = gaussian_copula(cov, scenarios=4, seed=123, chunks=None,
                             rng='Mersenne Twister')
    expect = [[-1.08563060, -0.54233474, -1.15002017],
              [-1.50629471, -1.60787125,  0.04561073],  # noqa
              [-2.42667924, -2.37097000, -0.86315499],
              [-0.86674040, -1.07598597, -0.29407627]]
    assert_allclose(expect, actual, 1e-6, 0)
    assert isinstance(actual, np.ndarray)


def test_gaussian_mersenne_da():
    actual = gaussian_copula(cov, scenarios=4, seed=123, chunks=2,
                             rng='Mersenne Twister')
    expect = [[ 1.61739599,  1.34668479,  0.85965138],  # noqa
              [ 0.00936110,  0.77307164, -0.65639660],  # noqa
              [-0.88414686, -1.07582330, -0.47076357],
              [-1.63883139, -1.21794817, -1.47348213]]
    assert_allclose(expect, actual, 1e-6, 0)
    assert actual.chunks == ((2, 2), (2, 1))


@pytest.mark.parametrize('chunks,expect_chunks', [
    (None, None),
    (2, ((2, 2), (2, 1))),
    (((3, 1), (2, 1)), ((3, 1), (2, 1))),
])
def test_gaussian_sobol(chunks, expect_chunks):
    actual = gaussian_copula(cov, scenarios=4, seed=123, chunks=chunks,
                             rng='SOBOL')
    expect = [[ 0.        ,  0.        ,  0.        ],  # noqa
              [-0.67448975, -0.31303751, -1.15262386],
              [ 0.67448975,  0.31303751,  1.15262386],  # noqa
              [ 0.31863936,  0.78820110, -0.53727912]]  # noqa
    assert_allclose(expect, actual, 1e-6, 0)
    if chunks:
        assert actual.chunks == expect_chunks
    else:
        assert isinstance(actual, np.ndarray)


def test_student_t_mersenne_np():
    actual = t_copula(cov, df=3, scenarios=4, seed=123, chunks=None,
                      rng='Mersenne Twister')
    expect = [[-0.99107466, -0.53046951, -1.03952341],
              [-2.08652315, -2.15403950,  0.10015459],  # noqa
              [-1.11240463, -1.09187061, -0.43874990],
              [-0.98508856, -1.17355012, -0.36287494]]
    assert_allclose(expect, actual, 1e-6, 0)
    assert isinstance(actual, np.ndarray)


def test_student_t_mersenne_da():
    actual = t_copula(cov, df=3, scenarios=4, seed=123, chunks=2,
                      rng='Mersenne Twister')
    print(repr(actual.compute()))
    expect = [[ 1.31308802,  1.14171788,  0.78120784],  # noqa
              [ 0.00859970,  0.67931082, -0.58360671],  # noqa
              [-0.48331984, -0.58214528, -0.26145540],
              [-1.44327714, -1.15851122, -1.33815158]]
    assert_allclose(expect, actual, 1e-6, 0)
    assert actual.chunks == ((2, 2), (2, 1))


@pytest.mark.parametrize('chunks,expect_chunks', [
    (None, None),
    (2, ((2, 2), (2, 1))),
    (((3, 1), (2, 1)), ((3, 1), (2, 1))),
])
def test_student_t_sobol(chunks, expect_chunks):
    actual = t_copula(cov, df=3, scenarios=4, seed=123, chunks=chunks,
                      rng='SOBOL')
    expect = [[ 0.        ,  0.        ,  0.        ],  # noqa
              [-0.90292647, -0.44513114, -1.38033019],
              [ 0.51756147,  0.24504617,  0.84650386],  # noqa
              [ 0.59093028,  1.28328308, -0.94456215]]  # noqa
    assert_allclose(expect, actual, 1e-6, 0)
    if chunks:
        assert actual.chunks == expect_chunks
    else:
        assert isinstance(actual, np.ndarray)
