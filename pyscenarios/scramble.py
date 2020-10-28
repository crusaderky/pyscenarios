import numpy as np
from scipy import stats


def scramble(samples: np.ndarray) -> np.ndarray:
    """
    Scramble function as in Owen (1997)

    Reference:

    .. [1] Saltelli, A., Chan, K., Scott, E.M., "Sensitivity Analysis"
    """
    assert samples.ndim == 2
    out = samples.copy()
    for c in range(samples.shape[1]):
        _scramble_inplace(out[:, c])
    return out


def _scramble_inplace(samples: np.ndarray) -> None:
    assert samples.ndim == 1
    h = samples.size // 2
    n = h * 2

    idx = samples[:n].argsort()
    iidx = idx.argsort()

    # Generate binomial values and switch position for the second half of the array
    bi = stats.binom(1, 0.5).rvs(size=h).astype(bool)
    pos = stats.uniform.rvs(size=h).argsort()

    # Scramble the indexes
    tmp = idx[:h][bi]
    idx[:h][bi] = idx[h:n][pos[bi]]
    idx[h:n][pos[bi]] = tmp

    # Apply the scrambling
    samples[:n] = samples[:n][idx[iidx]]

    # Apply scrambling to sub intervals
    if n > 2:
        _scramble_inplace(samples[:h])
        _scramble_inplace(samples[h:n])
