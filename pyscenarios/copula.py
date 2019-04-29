"""High performance copula generators
"""
from typing import List, Optional, Union, cast

import numpy as np
import numpy.random
import numpy.linalg
import dask.array as da
from dask.array.core import normalize_chunks

from . import duck
from .sobol import sobol
from .typing import Chunks2D, NormalizedChunks2D


def gaussian_copula(cov: Union[List[List[float]], np.ndarray],
                    samples: int, seed: int = 0,
                    chunks: Chunks2D = None,
                    rng: str = 'Mersenne Twister'
                    ) -> Union[np.ndarray, da.Array]:
    """Gaussian Copula scenario generator.

    Simplified algorithm::

        >>> l = numpy.linalg.cholesky(cov)
        >>> y = numpy.random.standard_normal(size=(samples, cov.shape[0]))
        >>> p = numpy.dot(l, y.T).T

    :param numpy.ndarray cov:
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

            pysamples.sobol.max_dimensions() - cov.shape[0] - 1

    :param str rng:
        Either ``Mersenne Twister`` or ``Sobol``

    :returns:
        array of shape (samples, dimensions), with all series
        being normal (0, 1) distributions.
    :rtype:
        If chunks is not None, :class:`dask.array.Array`; else
        :class:`numpy.ndarray`
    """
    assert samples > 0
    cov = np.asarray(cov)
    assert cov.ndim == 2
    assert cov.shape[0] == cov.shape[1]

    L = numpy.linalg.cholesky(cov)  # type: Union[np.ndarray, da.Array]
    if chunks:
        chunks = cast(NormalizedChunks2D,
                      normalize_chunks(chunks, shape=(samples, cov.shape[0])))
        L = da.from_array(L, chunks=(chunks[1], chunks[1]))

    rng = rng.lower()
    if rng == 'mersenne twister':
        rnd_state = duck.RandomState(seed)
        # When pulling samples from the Mersenne Twister generator, we have
        # the samples on the rows. This guarantees that if we draw more
        # samples, the original samples won't change.
        y = rnd_state.standard_normal(size=(samples, cov.shape[0]),
                                      chunks=chunks)
    elif rng == 'sobol':
        # Generate uniform (0, 1) distributions
        samples = sobol(size=(samples, cov.shape[0]),
                        d0=seed, chunks=chunks)
        # Convert to normal (0, 1)
        y = duck.norm_ppf(samples)
    else:
        raise ValueError("Unknown rng: %s" % rng)

    return duck.dot(L, y.T).T


def t_copula(cov: Union[List[List[float]], np.ndarray],
             df: Union[int, List[int], np.ndarray],
             samples: int, seed: int = 0,
             chunks: Chunks2D = None,
             rng: str = 'Mersenne Twister'
             ) -> Union[np.ndarray, da.Array]:
    """Student T Copula / IT Copula scenario generator.

    Simplified algorithm::

        >>> l = numpy.linalg.cholesky(cov)
        >>> y = numpy.random.standard_normal(size=(samples, cov.shape[0]))
        >>> r = numpy.random.uniform(size=(samples, 1))
        >>> s = scipy.stats.chi2.ppf(r, df=df)
        >>> z = numpy.sqrt(df / s) * numpy.dot(l, y.T).T
        >>> u = scipy.stats.t.cdf(z, df=df)
        >>> p = scipy.stats.norm.ppf(u)

    :param df:
        Number of degrees of freedom. Can be either a scalar int for
        Student T Copula, or a one-dimensional array-like with one point per
        dimension for IT Copula.

    :param int seed:
        Random seed.

        With ``rng='Sobol'``, this is the initial dimension; when generating
        multiple copulas with different seeds, one should never use seeds that
        are less than ``cov.shape[0] + 1`` apart from each other.

        The maximum seed when using sobol is::

            pysamples.sobol.max_dimensions() - cov.shape[0] - 2

    All other parameters and the return value are the same as in
    :func:`gaussian_copula`.
    """
    assert samples > 0
    cov = np.asarray(cov)
    assert cov.ndim == 2
    assert cov.shape[0] == cov.shape[1]
    dimensions = cov.shape[0]

    L = numpy.linalg.cholesky(cov)
    if chunks is not None:
        chunks = cast(NormalizedChunks2D,
                      normalize_chunks(chunks, shape=(samples, dimensions)))
        L = da.from_array(L, chunks=(chunks[1], chunks[1]))

    # Pre-process df into a 1D dask array
    df = np.asarray(df)
    if (df <= 0).any():
        raise ValueError("df must always be greater than zero")
    if df.shape not in ((), (dimensions, )):
        raise ValueError("df must be either a scalar or a 1D vector with as "
                         "many points as the width of the correlation matrix")
    if df.ndim == 1 and chunks is not None:
        df = da.from_array(df, chunks=(chunks[1], ))

    # Define chunks for the S chi-square matrix
    chunks_r = None  # type: Optional[NormalizedChunks2D]
    if chunks is not None:
        chunks_r = (chunks[0], (1, ))

    rng = rng.lower()
    if rng == 'mersenne twister':
        # Use two separate random states for the normal and the chi2
        # distributions. This is NOT the same as just extracting two series
        # from the same RandomState, as we must guarantee that, if you extract
        # a different number of samples from the generator, the initial
        # samples must remain the same.
        # For the same reason, we have the samples on the rows.
        rnd_state_y = duck.RandomState(seed)
        # Don't just do seed + 1 as that would have unwanted repercussions
        # when one tries to extract different series from different seeds.
        seed_r = (seed + 190823761298456) % 2**32
        rnd_state_r = duck.RandomState(seed_r)

        y = rnd_state_y.standard_normal(size=(samples, dimensions),
                                        chunks=chunks)
        r = rnd_state_r.uniform(size=(samples, 1), chunks=chunks_r)

    elif rng == 'sobol':
        seed_r = seed + dimensions

        y = sobol(size=(samples, dimensions), d0=seed, chunks=chunks)
        y = duck.norm_ppf(y)
        r = sobol(size=(samples, 1), d0=seed_r, chunks=chunks_r)

    else:
        raise ValueError("Unknown rng: %s" % rng)

    s = duck.chi2_ppf(r, df)
    z = duck.sqrt(df / s) * duck.dot(L, y.T).T
    # Convert t distribution to normal (0, 1)
    u = duck.t_cdf(z, df)
    p = duck.norm_ppf(u)
    return p
