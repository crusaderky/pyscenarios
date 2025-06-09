Installation
============

Required dependencies
---------------------

- Python 3.10 or later
- `Dask <https://dask.org>`_
- `NumPy <http://www.numpy.org>`_
- `Scipy <https://www.scipy.org>`_

Additionally, if you plan to use :func:`pyscenarios.sobol`, installing
`Numba <http://numba.pydata.org>`_ will greatly speed up the calculation.

You can install the required dependencies using pip::

   pip install pyscenarios
   pip install pyscenarios[numba]
   pip install pyscenarios[all]

or conda::

   conda install -c conda-forge pyscenarios
   conda install -c conda-forge pyscenarios numba


.. _mindeps_policy:

Minimum dependency versions
---------------------------
This project adopts a rolling policy based on `SPEC 0
<https://scientific-python.org/specs/spec-0000/>`_ regarding the minimum
supported version of its dependencies.

You can see the actual minimum tested versions in `pyproject.toml
<https://github.com/crusaderky/pyscenarios/blob/main/pyproject.toml>`_.
