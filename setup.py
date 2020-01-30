#!/usr/bin/env python
from setuptools import setup
import pyscenarios.sobol

pyscenarios.sobol.calc_v()


setup(use_scm_version=True)
