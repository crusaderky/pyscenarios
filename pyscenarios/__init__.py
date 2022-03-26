import importlib.metadata

from pyscenarios.copula import gaussian_copula, t_copula
from pyscenarios.sobol import max_sobol_dimensions, sobol
from pyscenarios.stats import tail_dependence

try:
    __version__ = importlib.metadata.version("pyscenarios")
except importlib.metadata.PackageNotFoundError:  # pragma: nocover
    # Local copy, not installed with pip
    __version__ = "999"

# Prevent Intersphinx from pointing to the implementation modules
for obj in gaussian_copula, t_copula, sobol, max_sobol_dimensions, tail_dependence:
    obj.__module__ = "pyscenarios"
del obj

__all__ = (
    "__version__",
    "gaussian_copula",
    "t_copula",
    "sobol",
    "max_sobol_dimensions",
    "tail_dependence",
)
