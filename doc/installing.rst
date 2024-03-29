.. _installing:

Installation
============

Required dependencies
---------------------

- Python 3.8 or later
- `dask <https://dask.org>`_
- `numba <http://numba.pydata.org>`_
- `numpy <http://www.numpy.org>`_
- `scipy <https://www.scipy.org>`_


.. _mindeps_policy:

Minimum dependency versions
---------------------------
pyscenarios adopts a rolling policy based on `NEP-29
<https://numpy.org/neps/nep-0029-deprecation_policy.html>`_ regarding the minimum
supported version of its dependencies:

- **Python:** 42 months
  (`NEP-29 <https://numpy.org/neps/nep-0029-deprecation_policy.html>`_)
- **numpy:** 24 months
  (`NEP-29 <https://numpy.org/neps/nep-0029-deprecation_policy.html>`_)
- **all other libraries:** 12 months

The above should be interpreted as *the minor version (X.Y) initially published no more
than N months ago*. Patch versions (x.y.Z) are not pinned, and only the latest available
at the moment of publishing the xarray release is guaranteed to work.

You can see the actual minimum tested versions in the `anaconda requirements file
<https://github.com/crusaderky/pyscenarios/blob/main/ci/requirements-minimal.yml>`_.


Testing
-------

To run the test suite after installing pyscenarios, first install (via pypi or conda)

- `py.test <https://pytest.org>`_: Simple unit testing library

and run ``py.test``.
