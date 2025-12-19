"""Top-level performance tests (renamed to avoid import-file collisions).

This file was renamed to avoid pytest import-file mismatch warnings that
occur when multiple test files in different directories share the same
basename (`test_performance.py`). Tests remain unchanged; filename only
was updated to give it a unique module name during collection.
"""

from tests.performance._orig_test_performance import *  # pragma: no cover - re-export existing tests
