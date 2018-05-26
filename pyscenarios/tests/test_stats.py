import dask.array as da
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pyscenarios.stats import tail_dependence


@pytest.mark.parametrize('chunk', [False, True])
def test_tail_dependence(chunk):
    x = [.1, .3, .4, .5, .6, .8, .9]
    y = [.4, .1, .2, .5, .9, .9, .1]
    q = [[.05, .15, .35, .50, .75],
         [.35, .75, .50, .70, .05]]

    if chunk:
        x = da.from_array(np.array(x), chunks=5)
        y = da.from_array(np.array(y), chunks=5)
        q = da.from_array(np.array(q), chunks=(1, 2))

    d = tail_dependence(x, y, q)
    assert_array_equal(d, [[np.nan, 0., 0.5, 0.75, 0.5],
                           [.5, .5, .75, .5, np.nan]])

    if chunk:
        assert d.chunks == q.chunks
    else:
        assert isinstance(d, np.ndarray)

    # Scalar q
    d = tail_dependence(x, y, .5)
    assert_array_equal(d, .75)

    if chunk:
        assert d.chunks == ()
    else:
        assert isinstance(d, float)
