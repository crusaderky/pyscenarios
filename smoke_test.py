"""Test import the library and print essential information"""

import platform
import sys

import pyscenarios

print("Python interpreter:", sys.executable)
print("Python version    :", sys.version)
print("Platform          :", platform.platform())
print("Library path      :", pyscenarios.__file__)
print("Library version   :", pyscenarios.__version__)
