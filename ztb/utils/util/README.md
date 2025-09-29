# Test Utilities Developer Guide

This guide covers the test isolation utilities available in `ztb/tests/util/` for writing robust, isolated unit tests.

## Overview

The test utilities provide:

- **Temporary Directory Management**: Isolated file system operations
- **Performance Monitoring**: Test execution timing and slow test detection
- **Test Isolation**: Unique resources per test to prevent interference

## Temporary Directory Utilities (`tempdir.py`)

### `get_unique_temp_dir(prefix: str = "test") -> str`

Generates a unique temporary directory path with timestamp and UUID for complete test isolation.

**Parameters:**

- `prefix`: Directory name prefix (default: "test")

**Returns:**

- Absolute path to unique temporary directory

**Example:**

```python
from ztb.tests.util.tempdir import get_unique_temp_dir

def test_file_operations():
    temp_dir = get_unique_temp_dir("file_test")
    # temp_dir: "/tmp/file_test_1234567890_abc12345"
    assert os.path.exists(temp_dir) == False  # Not created yet
```

### `TempDirManager(prefix: str = "test")`

Context manager for automatic creation and cleanup of temporary directories.

**Parameters:**

- `prefix`: Directory name prefix (default: "test")

**Yields:**

- Absolute path to created temporary directory

**Example:**

```python
from ztb.tests.util.tempdir import TempDirManager

def test_with_temp_dir():
    with TempDirManager("my_feature") as temp_dir:
        # Directory is created and available
        assert os.path.exists(temp_dir)

        # Perform file operations
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test data")

        assert os.path.exists(test_file)

    # Directory is automatically cleaned up
    assert not os.path.exists(temp_dir)
```

**Best Practices:**

- Use `TempDirManager` for tests requiring file I/O
- Avoid hardcoded paths; always use generated temp directories
- Test cleanup is automatic - no manual `shutil.rmtree` needed

## Performance Monitoring (`test_utils.py`)

### `@timed_test` Decorator

Monitors test execution time and logs slow or failed tests.

**Features:**

- Automatic timing of test functions
- Slow test detection (>5 seconds)
- Failed test duration logging
- Non-intrusive (doesn't affect test logic)

**Example:**

```python
from ztb.tests.util.test_utils import timed_test

@timed_test
def test_expensive_operation():
    # Simulate expensive operation
    time.sleep(6)  # This will trigger slow test warning

# Output: "SLOW TEST: test_expensive_operation took 6.00s"

@timed_test
def test_failing_operation():
    time.sleep(2)
    raise AssertionError("Test failed")

# Output: "FAILED TEST: test_failing_operation failed after 2.00s"
```

**Best Practices:**

- Apply to integration tests and slow unit tests
- Use for performance regression detection
- Review slow test logs during CI runs

## Integration with pytest

### Using with pytest fixtures

```python
import pytest
from ztb.tests.util.tempdir import TempDirManager
from ztb.tests.util.test_utils import timed_test

class TestFileOperations:

    @pytest.fixture
    def temp_dir(self):
        with TempDirManager("file_ops") as d:
            yield d

    @timed_test
    def test_write_read_cycle(self, temp_dir):
        # Test implementation using temp_dir
        pass

    @timed_test
    def test_concurrent_access(self, temp_dir):
        # Test implementation
        pass
```

### Test Organization

```text
ztb/tests/
├── unit/
│   ├── core/
│   └── utils/
├── integration/
│   └── features/
└── util/              # Test utilities
    ├── tempdir.py     # Directory isolation
    └── test_utils.py  # Performance monitoring
```

## Troubleshooting

### Common Issues

**Temp directory not cleaned up:**

- Ensure `TempDirManager` is used as context manager
- Check for file handles keeping directory locked

**Slow test warnings:**

- Review test logic for optimization opportunities
- Consider splitting large tests
- Use `@timed_test` to identify bottlenecks

**Test interference:**

- Use shared global state
- Clean up resources in `finally` blocks

### Performance Guidelines

- **Fast tests (< 0.1s)**: No decorator needed
- **Medium tests (0.1-5s)**: Optional `@timed_test`
- **Slow tests (> 5s)**: Always use `@timed_test`
- **Integration tests**: Always use both utilities

## Migration from Legacy Tests

**Before:**

```python
def test_legacy():
    # Hardcoded temp dir
    temp_dir = "/tmp/test_data"
    os.makedirs(temp_dir, exist_ok=True)
    # ... test logic ...
    # Manual cleanup (often forgotten)
    shutil.rmtree(temp_dir, ignore_errors=True)
```

**After:**

```python
@timed_test
def test_modern():
    with TempDirManager("legacy_migration") as temp_dir:
        # ... test logic ...
        # Automatic cleanup
```

This ensures reliable, isolated, and maintainable tests.
