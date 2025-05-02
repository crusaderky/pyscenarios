"""
Sobol V matrix constant, calculated upon import from original Joe/Kuo's directions.

This module is imported automatically the first time you call :func:`sobol`.

When using Dask, the V matrix is calculated internally by the Dask workers and
neither stored in the Dask graph nor propagated across the cluster from a single
compute node.
"""

import lzma
import pkgutil

import numpy as np
import numpy.typing as npt

DIRECTIONS = "new-joe-kuo-6.21201.txt.xz"


def _load_directions(resource_fname: str) -> npt.NDArray[np.uint32]:
    """Load input file containing direction numbers.
    The file must one of those available on the website of the
    original author, or formatted like one.

    :returns:
        contents of the input file as a 2D numpy array.
        Every row contains the information for the matching dimension.
        Column 0 contains the a values, while columns 1+ contain the m values.
        The m values are padded on the right with zeros.
    """
    data_bin = pkgutil.get_data("pyscenarios", resource_fname)
    assert data_bin
    data_bin = lzma.decompress(data_bin)
    data_txt = data_bin.decode("ascii")
    del data_bin
    rows = [row.split() for row in data_txt.splitlines()]
    del data_txt

    # Add padding at end of rows
    # Drop first 2 columns
    # Replace header with element for d=1
    rowlen = len(rows[-1])
    for row in rows:
        row[:] = row[2:] + ["0"] * (rowlen - len(row))
    rows[0] = ["0"] + ["1"] * (rowlen - 3)
    return np.array(rows, dtype=np.uint32)


def _calc_v(directions: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]:
    """Calculate V matrix from directions

    Note: an earlier version of this function was implemented in Numba,
    but offered zero benefit over plain NumPy.
    """
    ndim, ndir = directions.shape
    V = np.empty((ndim, 32), dtype=np.uint32)

    V[:, : ndir - 1] = np.roll(directions, -1, axis=1)[:, :-1] * 2 ** np.arange(
        31, 32 - ndir, -1
    )

    # Index of the first column where directions == 0, ignoring column 0
    ss = (
        np.where(
            np.all(directions[:, 1:] != 0, axis=1),
            ndir,
            np.argmin(directions[:, 1:], axis=1) + 1,
        )
        - 1
    )

    # Overwrite V[:, ss:]
    for s in range(ndir):
        row_mask = ss == s
        V[row_mask, :] = _fill_v(V[row_mask, :], directions[row_mask, 0], s)
    return V


def _fill_v(
    Vs: npt.NDArray[np.uint32], dir0: npt.NDArray[np.uint32], s: int
) -> npt.NDArray[np.uint32]:
    """Helper of _calc_v to fill the right side of the V matrix for a given s"""
    for t in range(s, 32):
        vts = Vs[:, t - s]
        vt = vts ^ (vts // 2**s)
        for k in range(1, s):
            vt ^= ((dir0 // 2 ** (s - 1 - k)) & 1) * Vs[:, t - k]
        Vs[:, t] = vt
    return Vs


V = _calc_v(_load_directions(DIRECTIONS))
