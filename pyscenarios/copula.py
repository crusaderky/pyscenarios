"""High performance copula generators"""

from typing import Literal, TypeAlias, overload

import dask.array as da
import numpy as np
import numpy.linalg
import numpy.typing as npt
from dask.array.core import normalize_chunks

from pyscenarios import duck
from pyscenarios._sobol import sobol
from pyscenarios.typing import Chunks2D, NormalizedChunks2D

RNG: TypeAlias = Literal["Mersenne Twister", "Sobol"]


@overload
def gaussian_copula(
    cov: npt.ArrayLike,
    samples: int,
    *,
    seed: int = 0,
    chunks: None = None,
    rng: RNG = "Mersenne Twister",
) -> npt.NDArray[np.float64]: ...


@overload
def gaussian_copula(
    cov: npt.ArrayLike,
    samples: int,
    *,
    seed: int = 0,
    chunks: Chunks2D,
    rng: RNG = "Mersenne Twister",
) -> da.Array: ...


def gaussian_copula(
    cov: npt.ArrayLike,
    samples: int,
    *,
    seed: int = 0,
    chunks: Chunks2D | None = None,
    rng: RNG = "Mersenne Twister",
) -> npt.NDArray[np.float64] | da.Array:
    """Gaussian Copula scenario generator.

    Simplified algorithm::

        >>> l = numpy.linalg.cholesky(cov)
        >>> y = numpy.random.standard_normal(size=(samples, cov.shape[0]))
        >>> p = (l @ y.T).T

    :param ArrayLike cov:
        covariance matrix, a.k.a. correlation matrix. It must be a
        Hermitian, positive-definite matrix in any square array-like format.
        The width of cov determines the number of dimensions of the output.

    :param int samples:
        Number of random samples to generate

        .. note::
           When using Sobol, to obtain a uniform distribution one must use
           :math:`2^{n} - 1` samples (for any n > 0).

    :param chunks:
        Chunk size for the return array, which has shape (samples, dimensions).
        It can be anything accepted by dask (a positive integer, a tuple of two
        ints, or a tuple of two tuples of ints) for the output shape.

        Set to None to return a numpy array.

        .. warning::
           When using the Mersenne Twister random generator, the chunk size
           changes the random sequence. To guarantee repeatability, it must be
           fixed together with the seed. chunks=None also produces different
           results from using dask.

    :param int seed:
        Random seed.

        With ``rng='Sobol'``, this is the initial dimension; when generating
        multiple copulas with different seeds, one should never use seeds that
        are less than ``cov.shape[0]`` apart from each other.

        The maximum seed when using sobol is::

            pyscenarios.sobol.max_sobol_dimensions() - cov.shape[0] - 1

    :param str rng:
        Either ``Mersenne Twister`` or ``Sobol``

    :returns:
        array of shape (samples, dimensions), with all series
        being normal (0, 1) distributions.

    :rtype:
        If chunks is not None, :class:`dask.array.Array`; else
        :class:`numpy.ndarray`
    """
    return _copula_impl(
        cov=cov, df=None, samples=samples, seed=seed, chunks=chunks, rng=rng
    )


@overload
def t_copula(
    cov: npt.ArrayLike,
    df: npt.ArrayLike,
    samples: int,
    seed: int = 0,
    *,
    chunks: None = None,
    rng: RNG = "Mersenne Twister",
) -> npt.NDArray[np.float64]: ...


@overload
def t_copula(
    cov: npt.ArrayLike,
    df: npt.ArrayLike,
    samples: int,
    seed: int = 0,
    *,
    chunks: Chunks2D,
    rng: RNG = "Mersenne Twister",
) -> da.Array: ...


def t_copula(
    cov: npt.ArrayLike,
    df: npt.ArrayLike,
    samples: int,
    seed: int = 0,
    *,
    chunks: Chunks2D | None = None,
    rng: RNG = "Mersenne Twister",
) -> npt.NDArray[np.float64] | da.Array:
    """Student T Copula / IT Copula scenario generator.

    Simplified algorithm::

        >>> l = numpy.linalg.cholesky(cov)
        >>> y = numpy.random.standard_normal(size=(samples, cov.shape[0]))
        >>> p = (l @ y.T).T  # Gaussian Copula
        >>> r = numpy.random.uniform(size=(samples, 1))
        >>> s = scipy.stats.chi2.ppf(r, df=df)
        >>> z = numpy.sqrt(df / s) * p
        >>> u = scipy.stats.t.cdf(z, df=df)
        >>> t = scipy.stats.norm.ppf(u)

    :param ArrayLike cov:
        covariance matrix, a.k.a. correlation matrix. It must be a
        Hermitian, positive-definite matrix in any square array-like format.
        The width of cov determines the number of dimensions of the output.

    :param ArrayLike df:
        Number of degrees of freedom. Can be either a scalar int for
        Student T Copula, or a one-dimensional NumPy array or array-like with
        one point per dimension for IT Copula.

    :param int samples:
        Number of random samples to generate

        .. note::
           When using Sobol, to obtain a uniform distribution one must use
           :math:`2^{n} - 1` samples (for any n > 0).

    :param chunks:
        Chunk size for the return array, which has shape (samples, dimensions).
        It can be anything accepted by dask (a positive integer, a tuple of two
        ints, or a tuple of two tuples of ints) for the output shape.

        Set to None to return a numpy array.

        .. warning::
           When using the Mersenne Twister random generator, the chunk size
           changes the random sequence. To guarantee repeatability, it must be
           fixed together with the seed. chunks=None also produces different
           results from using dask.

    :param int seed:
        Random seed.

        With ``rng='Sobol'``, this is the initial dimension; when generating
        multiple copulas with different seeds, one should never use seeds that
        are less than ``cov.shape[0] + 1`` apart from each other.

        The maximum seed when using sobol is::

            pyscenarios.sobol.max_sobol_dimensions() - cov.shape[0] - 2

    :param str rng:
        Either ``Mersenne Twister`` or ``Sobol``

    :returns:
        array of shape (samples, dimensions), with all series
        being normal (0, 1) distributions.

    :rtype:
        If chunks is not None, :class:`dask.array.Array`; else
        :class:`numpy.ndarray`
    """
    return _copula_impl(
        cov=cov, df=df, samples=samples, seed=seed, chunks=chunks, rng=rng
    )


def _copula_impl(
    cov: npt.ArrayLike,
    df: npt.ArrayLike | None,
    samples: int,
    seed: int,
    *,
    chunks: Chunks2D | None,
    rng: RNG,
) -> npt.NDArray[np.float64] | da.Array:
    """Implementation of gaussian_copula and t_copula"""
    samples = int(samples)
    if samples <= 0:
        raise ValueError("Number of samples must be positive")
    cov = np.asarray(cov)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be a square matrix")
    dimensions = cov.shape[0]

    L = numpy.linalg.cholesky(cov)
    norm_chunks: NormalizedChunks2D | None = None
    if chunks is not None:
        norm_chunks = normalize_chunks(chunks, shape=(samples, dimensions))
        assert norm_chunks is not None
        L = da.from_array(L, chunks=(norm_chunks[1], norm_chunks[1]))
    del chunks

    rng_low = rng.lower()
    if rng_low == "mersenne twister":
        # When pulling samples from the Mersenne Twister generator, we have
        # the samples on the rows. This guarantees that if we draw more
        # samples, the original samples won't change.
        rnd_state_y = duck.RandomState(seed)
        y = rnd_state_y.standard_normal(size=(samples, dimensions), chunks=norm_chunks)
    elif rng_low == "sobol":
        y = sobol(size=(samples, dimensions), d0=seed, chunks=norm_chunks)
        y = duck.norm_ppf(y)
    else:
        raise ValueError(
            f"Invalid rng: required 'Mersenne Twister' or 'Sobol'; got {rng!r}"
        )
    del rng

    p = (L @ y.T).T  # Gaussian Copula
    if df is None:
        return p

    # Pre-process df into a 1D numpy/dask array
    df = np.asarray(df)
    if (df <= 0).any():
        raise ValueError("df must always be greater than zero")
    if df.shape not in ((), (dimensions,)):
        raise ValueError(
            "df must be either a scalar or a 1D vector with as "
            "many points as the width of the correlation matrix"
        )
    if df.ndim == 1 and norm_chunks is not None:
        df = da.from_array(df, chunks=(norm_chunks[1],))

    # Define chunks for the S chi-square matrix
    chunks_r = (norm_chunks[0], (1,)) if norm_chunks is not None else None

    if rng_low == "mersenne twister":
        # Use two separate random states for the normal and the chi2
        # distributions. This is NOT the same as just extracting two series
        # from the same RandomState, as we must guarantee that, if you extract
        # a different number of samples from the generator, the initial
        # samples must remain the same.
        # Don't just do seed + 1 as that would have unwanted repercussions
        # when one tries to extract different series from different seeds.
        seed_r = (seed + 190823761298456) % 2**32
        rnd_state_r = duck.RandomState(seed_r)
        r = rnd_state_r.uniform(size=(samples, 1), chunks=chunks_r)
    elif rng_low == "sobol":
        seed_r = seed + dimensions
        r = sobol(size=(samples, 1), d0=seed_r, chunks=chunks_r)
    else:
        raise AssertionError("unreachable")  # pragma: nocover

    s = duck.chi2_ppf(r, df)
    z = duck.sqrt(df / s) * p
    # Convert t distribution to normal (0, 1)
    u = duck.t_cdf(z, df)
    return duck.norm_ppf(u)
