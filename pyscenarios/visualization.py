from typing import Tuple, Union

import dask.array as da
import numpy as np


def plot_couples(
    samples: Union[np.ndarray, da.Array],
    *,
    d0: int = 0,
    heat: Union[np.ndarray, da.Array] = None,
    limits: Tuple[float, float] = (0, 1),
    figsize: Tuple[float, float] = (16, 16),
) -> None:
    """Given a series of samples with S samples and D dimensions, pick the dimensions
    two by two and produce scatter plots with matplotlib.
    This function is meant to be executed in a Jupyter Notebook.

    :param samples:
        2D array with shape (samples, dimensions)
    :param int d0:
        Initial dimension, to be added in the labels
    :param heat:
        1D array of heat of each sample, e.g. emitted by
        :func:`~pyscenarios.clusterization`. Points with the least heat will appear in
        blue while those with the most heat will appear in red.
    :param limits:
        Boundaries of each plot
    :param figsize:
        Size of the matplotlib plot

    **Example:**

    .. code-block:: python

        >>> from pyscenarios import *
        >>> S = sobol((8191, 6), d0=10, chunks=8192)
        >>> H = clusterization(S)
        >>> plot_couples(S, d0=10, heat=H)
    """
    from matplotlib import pyplot as plt

    samples, heat = da.compute(samples, heat)
    fig, axs = plt.subplots(samples.shape[1] - 1, samples.shape[1] - 1, figsize=figsize)

    for i in range(samples.shape[1] - 1):
        for j in range(1, samples.shape[1]):
            ax = axs[i, j - 1]
            ax.set_xlim(limits)
            ax.set_ylim(limits)
            if i >= j:
                continue
            ax.scatter(
                samples[:, i], samples[:, j], c=heat, cmap="jet", marker=".", s=1
            )
            ax.set_title(f"({i + d0},{j + d0})")

    for ax in fig.get_axes():
        ax.label_outer()
    fig.tight_layout()
