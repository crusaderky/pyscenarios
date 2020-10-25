import pkg_resources

from .copula import gaussian_copula, t_copula
from .sobol import max_sobol_dimensions, sobol
from .stats import tail_dependence

try:
    __version__ = pkg_resources.get_distribution("pyscenarios").version
except Exception:  # pragma: nocover
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"

# Prevent Intersphinx from pointing to the implementation modules
for obj in gaussian_copula, t_copula, sobol, max_sobol_dimensions, tail_dependence:
    obj.__module__ = "pyscenarios"

del obj
del pkg_resources

__all__ = (
    "__version__",
    "gaussian_copula",
    "t_copula",
    "sobol",
    "max_sobol_dimensions",
    "tail_dependence",
)
