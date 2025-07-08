"""Test import the library and print essential information"""

import sys

import pyscenarios

print("Python interpreter:", sys.executable)
print("Python version    :", sys.version)
print("Library path      :", pyscenarios.__file__)
print("Library version   :", pyscenarios.__version__)
