.. currentmodule:: pyscenarios

What's New
==========


.. _whats-new.0.3.0:

v0.3.0 (Unreleased)
-------------------

- Added explicit support for Python 3.8
- Now using setuptools-scm for versioning
- This project now adheres to NEP-29; see :ref:`mindeps_policy`.
  Bumped up minimum versions for all dependencies:

  ==========  ====== ====
  Dependency  v0.2   v0.3
  ==========  ====== ====
  python      3.5.0  3.6
  dask        0.17.3 2.2
  numba       0.34   0.44
  numpy       1.13   1.15
  pandas      0.20   0.25
  scipy       1.0    1.3
  ==========  ====== ====


.. _whats-new.0.2.1:

v0.2.1 (2019-05-01)
-------------------

- Make package discoverable by mypy
- A more robust fix for `dask#4739 <https://github.com/dask/dask/issues/4739>`_


.. _whats-new.0.2.0:

v0.2.0 (2019-04-29)
-------------------

- Type annotations
- 'rng' parameter in copula functions is now case insensitive
- Work around regression in IT copula with dask >= 1.1
  (`dask#4739 <https://github.com/dask/dask/issues/4739>`_)
- Smaller binary package; simplified setup
- Explicit CI tests for Windows, Python 3.5.0, and Python 3.7
- Mandatory flake8 and mypy in CI
- Changed license to Apache 2.0


.. _whats-new.0.1.0:

v0.1.0 (2018-05-27)
-------------------

Initial release.