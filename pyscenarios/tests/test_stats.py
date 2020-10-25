import dask.array as da
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyscenarios import tail_dependence


@pytest.mark.parametrize("chunk", [False, True])
def test_tail_dependence(chunk):
    x = [0.1, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9]
    y = [0.4, 0.1, 0.2, 0.5, 0.9, 0.9, 0.1]
    q = [[0.05, 0.15, 0.35, 0.50, 0.75], [0.35, 0.75, 0.50, 0.70, 0.05]]

    if chunk:
        x = da.from_array(np.array(x), chunks=5)
        y = da.from_array(np.array(y), chunks=5)
        q = da.from_array(np.array(q), chunks=(1, 2))

    d = tail_dependence(x, y, q)
    assert_array_equal(
        d, [[np.nan, 0.0, 0.5, 0.75, 0.5], [0.5, 0.5, 0.75, 0.5, np.nan]]
    )

    if chunk:
        assert d.chunks == q.chunks
    else:
        assert isinstance(d, np.ndarray)

    # Scalar q
    d = tail_dependence(x, y, 0.5)
    assert_array_equal(d, 0.75)

    if chunk:
        assert d.chunks == ()
    else:
        assert isinstance(d, float)
