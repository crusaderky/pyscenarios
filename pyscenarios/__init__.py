import importlib.metadata

from pyscenarios._sobol import max_sobol_dimensions, sobol
from pyscenarios.copula import gaussian_copula, t_copula
from pyscenarios.stats import tail_dependence

try:
    __version__ = importlib.metadata.version("pyscenarios")
except importlib.metadata.PackageNotFoundError:  # pragma: nocover
    # Local copy, not installed with pip
    __version__ = "9999"

# Prevent Intersphinx from pointing to the implementation modules
for obj in gaussian_copula, t_copula, max_sobol_dimensions, sobol, tail_dependence:
    obj.__module__ = "pyscenarios"
del obj

__all__ = (
    "__version__",
    "gaussian_copula",
    "max_sobol_dimensions",
    "sobol",
    "t_copula",
    "tail_dependence",
)
