import os

import pytest

has_jit = os.getenv("NUMBA_DISABLE_JIT") != "1"
requires_jit = pytest.mark.skipif(condition=not has_jit, reason="Requires numba jit")
