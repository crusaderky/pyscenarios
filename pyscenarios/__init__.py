import pkg_resources

try:
    __version__ = pkg_resources.get_distribution("pyscenarios").version
except Exception:  # pragma: nocover
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"

__all__ = ("__version__",)
