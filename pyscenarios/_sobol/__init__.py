"""Sobol sequence generator

This is a reimplementation of a C++ algorithm by
`Stephen Joe and Frances Y. Kuo <http://web.maths.unsw.edu.au/~fkuo/sobol/>`_.
Directions are based on :file:`new-joe-kuo-6.21201` from the URL above.
"""

from ._sobol import max_sobol_dimensions, sobol

__all__ = ["max_sobol_dimensions", "sobol"]
