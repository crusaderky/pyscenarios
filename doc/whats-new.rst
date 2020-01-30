.. currentmodule:: pyscenarios

What's New
**********

.. _whats-new.0.2.0:

v0.2.0 (unreleased)
===================

Code changes
------------
- Fixed IT copula for newer versions of dask

Project and CI changes
----------------------
- Added explicit CI for Windows, Python 3.7, and 3.8
- This project now adheres to NEP-29; see :ref:`mindeps_policy`.
  Bumped up minimum versions for all dependencies:

  ==========  ====== ====
  Dependency  v0.1   v0.2
  ==========  ====== ====
  python      3.5    3.6
  dask        0.17.3 2.2
  numba       0.34   0.44
  numpy       1.13   1.15
  pandas      0.20   0.25
  scipy       1.0    1.3
  ==========  ====== ====

- Changed license to Apache 2.0
- Mandatory flake8 in CI
- Now using setuptools-scm for versioning


.. _whats-new.0.1.0:

v0.1.0 (2018-05-27)
===================

Initial release.