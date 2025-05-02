.. _installing:

Installation
============

Required dependencies
---------------------

- Python 3.10 or later
- `dask <https://dask.org>`_
- `numba <http://numba.pydata.org>`_
- `numpy <http://www.numpy.org>`_
- `scipy <https://www.scipy.org>`_


.. _mindeps_policy:

Minimum dependency versions
---------------------------
pyscenarios adopts a rolling policy based on `SPEC 0
<https://scientific-python.org/specs/spec-0000/>`_ regarding the minimum
supported version of its dependencies.

You can see the actual minimum tested versions in the `anaconda requirements file
<https://github.com/crusaderky/pyscenarios/blob/main/ci/requirements-minimal.yml>`_.


Testing
-------

To run the test suite after installing pyscenarios, first install (via pypi or conda)

- `py.test <https://pytest.org>`_: Simple unit testing library

and run ``py.test``.
