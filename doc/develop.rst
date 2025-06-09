Development Guidelines
======================

Install
-------

1. Clone this repository with git:

.. code-block:: bash

     git clone git@github.com:crusaderky/pyscenarios.git
     cd pyscenarios

2. `Install pixi <https://pixi.sh/latest/#installation>`_
3. To keep a fork in sync with the upstream source:

.. code-block:: bash

   cd pyscenarios
   git remote add upstream git@github.com:crusaderky/pyscenarios.git
   git remote -v
   git fetch -a upstream
   git checkout main
   git pull upstream main
   git push origin main

Test
----

Test using pixi:

.. code-block:: bash

   pixi run tests

Test with coverage:

.. code-block:: bash

   pixi run coverage

Test with coverage and open HTML report in your browser:

.. code-block:: bash

   pixi run open-coverage

Code Formatting
---------------

pyscenarios uses several code linters (ruff, black, mypy), which are enforced by CI.
Developers should run them locally before they submit a PR, through the single command

.. code-block:: bash

    pixi run lint

This makes sure that linter versions and options are aligned for all developers.

Optionally, you may wish to setup the `pre-commit hooks <https://pre-commit.com/>`_ to
run automatically when you make a git commit. This can be done by running:

.. code-block:: bash

   pixi run pre-commit-install

Now the code linters will be run each time you commit changes.
You can skip these checks with ``git commit --no-verify`` or with
the short version ``git commit -n``.

Documentation
-------------

Build the documentation in ``build/html`` using pixi:

.. code-block:: bash

    pixi run docs

Build the documentation and open it in your browser:

.. code-block:: bash

    pixi run open-docs
