import sys
from multiprocessing import get_context

import pytest


def check_lazy_imports():
    # This function will run in a subprocess
    from pyscenarios import gaussian_copula

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
