import numpy as np
from numpy.testing import assert_array_equal
from pyscenarios.sobol import sobol, max_dimensions


EXPECT = np.array(
    [[0.5   , 0.5   , 0.5   , 0.5   ],
     [0.25  , 0.75  , 0.25  , 0.25  ],
     [0.75  , 0.25  , 0.75  , 0.75  ],
     [0.625 , 0.875 , 0.375 , 0.125 ],
     [0.125 , 0.375 , 0.875 , 0.625 ],
     [0.875 , 0.125 , 0.125 , 0.375 ],
     [0.375 , 0.625 , 0.625 , 0.875 ],
     [0.1875, 0.5625, 0.1875, 0.5625],
     [0.6875, 0.0625, 0.6875, 0.0625],
     [0.4375, 0.3125, 0.4375, 0.8125],
     [0.9375, 0.8125, 0.9375, 0.3125],
     [0.5625, 0.4375, 0.3125, 0.6875],
     [0.0625, 0.9375, 0.8125, 0.1875],
     [0.8125, 0.6875, 0.0625, 0.9375],
     [0.3125, 0.1875, 0.5625, 0.4375]])


def test_max_dimensions():
    assert max_dimensions() == 21201


def test_numpy():
    output = sobol(15, 4, d0=123)
    assert_array_equal(EXPECT, output)


def test_dask():
    output = sobol(15, 4, d0=123, chunks=(10, 3))
    assert output.chunks == ((10, 5), (3, 1))
    assert_array_equal(EXPECT, output.compute())
