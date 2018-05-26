"""High performance copula generators
"""
import numpy
import numpy.random
import numpy.linalg
import dask.array
import dask.base
from dask.array.core import normalize_chunks
from .sobol import sobol
from . import duck


def gaussian_copula(cov, samples, seed=0, chunks=None,
                    rng='Mersenne Twister'):
    """Gaussian Copula scenario generator

    :param numpy.ndarray cov:
        covariance matrix, a.k.a. correlation matrix. It must be a
        Hermitian, positive-definite matrix formatted as a
        numpy array with shape (dimensions, dimensions)

    :param int samples:
        Number of random samples to generate

        .. note::
           When using SOBOL, to obtain a uniform distribution one must use
           :math:`2^{n} - 1` samples (for any n > 0).

    :param chunks:
        Chunk size. It can be anything accepted by dask (a positive integer, a
        tuple of two ints, or a tuple of two tuples of ints) for the output
        shape (see result below). It is used to chunk the return array of
        (samples, dimensions).

        If None, use pure numpy.

        .. warning::
           When using the Mersenne Twister random generator, the chunk size
           changes the random sequence. To guarantee repeatability, it must be
           fixed together with the seed. chunks=None also produces different
           results from using dask.

    :param int seed:
        Random seed.

        When invoking this function multiple times with different seeds and
        uses ``rng='SOBOL', this is the initial dimension; one should never
        use seeds that are less than cov.shape[0] apart from each other.

        The maximum seed when using sobol is::

            pysamples.sobol.max_dimensions() - cov.shape[0] - 1

    :param str rng:
        Either ``Mersenne Twister`` or ``SOBOL``

    :returns:
        array of shape (samples, dimensions), with all series
        being normal (0, 1) distributions.
    :rtype:
        If chunks is not None, :class:`dask.array.Array`; else
        :class:`numpy.ndarray`
    """
    assert samples > 0
    cov = numpy.array(cov)
    assert cov.ndim == 2
    assert cov.shape[0] == cov.shape[1]

    L = numpy.linalg.cholesky(cov)

    if chunks:
        chunks = normalize_chunks(chunks, shape=(samples, cov.shape[0]))
        L = dask.array.from_array(L, chunks=(chunks[1], chunks[1]))

    if rng == 'Mersenne Twister':
        rnd_state = duck.RandomState(seed)
        # When pulling samples from the Mersenne Twister generator, we have
        # the samples on the rows. This guarantees that if we draw more
        # samples, the original samples won't change.
        y = rnd_state.standard_normal(size=(samples, cov.shape[0]),
                                      chunks=chunks)
    elif rng == 'SOBOL':
        # Generate uniform (0, 1) distributions
        samples = sobol(size=(samples, cov.shape[0]),
                        d0=seed, chunks=chunks)
        # Convert to normal (0, 1)
        y = duck.norm_ppf(samples)
    else:
        raise ValueError("Unknown rng: %s" % rng)

    return duck.dot(L, y.T).T


def t_copula(cov, df, samples, seed=0, chunks=None, rng='Mersenne Twister'):
    """Student T Copula / IT Copula scenario generator

    Simplified algorithm::
        l = numpy.linalg.cholesky(cov)
        y = numpy.random.normal(size=(samples, dimensions))
        s = numpy.random.chisquare(df=df, size=samples)
        z = numpy.sqrt(df / s) * numpy.tensordot(l, y, ((1, ), (1, )))
        u = scipy.stats.t.cdf(z, df=df)
        p = scipy.stats.norm.ppf(u)

    :param df:
        Number of degrees of freedom. Can be either a scalar int for a
        Student T Copula, or a one-dimensional array-like with one point per
        riskdriver for a IT Copula.

    :param int seed:
        Random seed.

        When invoking this function multiple times with different seeds and
        uses ``rng='SOBOL', this is the initial dimension; one should never
        use seeds that are less than ``cov.shape[0] + df.size`` apart from each
        other, or ``cov.shape[0] + 1`` if df is a scalar.

        The maximum seed when using sobol is::

            pysamples.sobol.max_dimensions() - cov.shape[0] - df.size - 1

    All other parameters and the return type are the same as in
    :func:`gaussian_copula`.
    """
    assert samples > 0
    cov = numpy.array(cov)
    assert cov.ndim == 2
    assert cov.shape[0] == cov.shape[1]

    L = numpy.linalg.cholesky(cov)
    if chunks:
        chunks = normalize_chunks(chunks, shape=(samples, cov.shape[0]))
        L = dask.array.from_array(L, chunks=(chunks[1], chunks[1]))

    # Pre-process df into a 1D dask array
    df = numpy.array(df)
    if (df <= 0).any():
        raise ValueError("df must always be greater than zero")
    if df.shape == ():
        # Convert to 1D
        df = df.ravel()
        if chunks:
            df = dask.array.from_array(df, chunks=(1, ))
    elif df.shape == (cov.shape[0], ):
        if chunks:
            df = dask.array.from_array(df, chunks=(chunks[1], ))
    else:
        raise ValueError("df must be either a scalar or a 1D vector with as "
                         "many points as the width of the correlation matrix")

    # Define chunks for the S chi-square matrix
    if chunks:
        chunks_s = (chunks[0], df.chunks[0])
    else:
        chunks_s = None

    if rng == 'Mersenne Twister':
        # Use two separate random states for the normal and the chi2
        # distributions. This is NOT the same as just extracting two series
        # from the same RandomState, as we must guarantee that, if you extract
        # a different number of samples from the generator, the initial
        # samples must remain the same.
        # For the same reason, we have the samples on the rows.
        rnd_state_y = duck.RandomState(seed)
        # Don't just do seed + 1 as that would have unwanted repercussions
        # when one tries to extract different series from different seeds.
        seed_s = (seed + 190823761298456) % 2**32
        rnd_state_s = duck.RandomState(seed_s)

        y = rnd_state_y.standard_normal(
            size=(samples, cov.shape[0]),
            chunks=chunks)
        s = rnd_state_s.chisquare(
            size=(samples, df.size), df=df,
            chunks=chunks_s)

    elif rng == 'SOBOL':
        seed_s = seed + cov.shape[0]

        y = sobol(
            size=(samples, cov.shape[0]),
            d0=seed, chunks=chunks)
        y = duck.norm_ppf(y)
        s = sobol(
            size=(samples, df.size),
            d0=seed_s, chunks=chunks_s)
        s = duck.chi2_ppf(s, df=df)

    else:
        raise ValueError("Unknown rng: %s" % rng)

    z = duck.sqrt(df / s) * duck.dot(L, y.T).T
    # Convert t distribution to normal (0, 1)
    u = duck.t_cdf(z, df)
    p = duck.norm_ppf(u)
    return p
