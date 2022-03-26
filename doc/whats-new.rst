.. currentmodule:: pyscenarios

What's New
==========

.. _whats-new.0.4.0:

v0.4.0 (2022-03-26)
-------------------

- Moved and renamed functions:

  ================================== ================================
  v0.3.0                             v0.4.0
  ================================== ================================
  pyscenarios.copula.gaussian_copula pyscenarios.gaussian_copula
  pyscenarios.copula.t_copula        pyscenarios.t_copula
  pyscenarios.sobol.sobol            pyscenarios.sobol
  pyscenarios.sobol.max_dimensions   pyscenarios.max_sobol_dimensions
  pyscenarios.stats.tail_dependence  pyscenarios.tail_dependence
  ================================== ================================

- Added support for Python 3.9 and 3.10
- Bumped up minimum versions for dependencies:

  ==========  ==== ====
  Dependency  v0.3 v0.4
  ==========  ==== ====
  python      3.6  3.8
  numba       0.44 0.47
  numpy       1.15 1.16
  ==========  ==== ====

- Dropped requirement for pandas
- Dropped requirement for setuptools at runtime


.. _whats-new.0.3.0:

v0.3.0 (2020-07-02)
-------------------

- Added explicit support for Python 3.8
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

- Now using setuptools-scm for versioning
- Functions decorated by numba are now covered by coveralls
- Migrated CI from travis + appveyor + coveralls to GitHub actions + codecov.io
- Added black and isort to CI


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