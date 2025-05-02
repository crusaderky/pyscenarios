import pickle
import sys
from multiprocessing import get_context

import pytest

from pyscenarios import gaussian_copula, sobol


def check_lazy_imports():  # pragma: nocover
    # This function will run in a subprocess
    gaussian_copula(cov=[[1.0, 0.9], [0.9, 1.0]], samples=4)
    gaussian_copula(cov=[[1.0, 0.9], [0.9, 1.0]], samples=4, chunks=2)
    assert "numba" not in sys.modules
    assert "pyscenarios._sobol._vmatrix" not in sys.modules

    gaussian_copula(cov=[[1.0, 0.9], [0.9, 1.0]], samples=4, rng="Sobol")
    assert "pyscenarios._sobol._vmatrix" in sys.modules
    if "numba" not in sys.modules:
        with pytest.raises(ImportError):
            import numba  # noqa: F401


def test_expensive_imports_are_lazy():
    """Test that, when PyScenarios is used for tasks unrelated to Sobol,
    Numba is not imported and the V matrix is not calculated.
    """
    ctx = get_context("spawn")
    process = ctx.Process(target=check_lazy_imports)
    process.start()
    process.join()
    assert process.exitcode == 0


def test_dask_workers_without_numba():
    """Test that Sobol can run when the Dask client has Numba installed,
    but the workers do not.
    """
    output = sobol((15, 4), chunks=(6, 3))
    pik = pickle.dumps(output)
    assert b"sobol_kernel" in pik
    assert b"numba" not in pik
