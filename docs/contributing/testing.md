# Testing Guide

This guide covers the testing strategy and practices for the Zaif Trade Bot project.

## Testing Overview

The project uses a comprehensive testing strategy with multiple layers:

- **Unit Tests**: Isolated component testing
- **Integration Tests**: End-to-end pipeline validation
- **Performance Tests**: Memory and timing analysis
- **Quality Gates**: Automated validation

## Test Structure

```text
ztb/tests/
├── unit/              # Unit tests
│   ├── core/
│   └── utils/
├── integration/       # Integration tests
│   └── features/
└── util/              # Test utilities
    ├── tempdir.py     # Directory isolation
    └── test_utils.py  # Performance monitoring
```

## Running Tests

### Unit Tests

```bash
# Run all unit tests
npm run test:unit

# Run specific test file
python -m pytest ztb/tests/unit/test_specific.py

# Run with coverage
python -m pytest --cov=ztb ztb/tests/unit/
```

### Integration Tests

```bash
# Run fast integration tests
npm run test:int-fast

# Run all integration tests
npm run test:integration
```

### Performance Tests

```bash
# Run performance benchmarks
python scripts/benchmark_features.py

# Memory profiling
python -c "import ztb; ztb.run_memory_profile()"
```

## Test Utilities

### Temporary Directory Management

Use `TempDirManager` for isolated file system operations:

```python
from ztb.tests.util.tempdir import TempDirManager

def test_file_operations():
    with TempDirManager("file_test") as temp_dir:
        # Isolated directory for testing
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test data")
        assert os.path.exists(test_file)
```

### Performance Monitoring

Use `@timed_test` decorator for slow test detection:

```python
from ztb.tests.util.test_utils import timed_test

@timed_test
def test_expensive_operation():
    # Test that takes >5 seconds will trigger warning
    time.sleep(6)
```

## Writing Tests

### Unit Test Best Practices

1. **Isolation**: Use mocks and fixtures for external dependencies
2. **Descriptive Names**: `test_should_calculate_sharpe_ratio()`
3. **Single Responsibility**: One assertion per test
4. **Fast Execution**: Keep unit tests under 0.1 seconds

### Integration Test Guidelines

1. **Real Dependencies**: Use actual services where safe
2. **Cleanup**: Always clean up resources
3. **Timeouts**: Set reasonable timeouts for async operations
4. **Data Isolation**: Use unique test data

### Test Organization

```python
import pytest
from ztb.tests.util.tempdir import TempDirManager
from ztb.tests.util.test_utils import timed_test

class TestFeatureEngineering:

    @pytest.fixture
    def temp_dir(self):
        with TempDirManager("features") as d:
            yield d

    @timed_test
    def test_rsi_calculation(self, temp_dir):
        # Test implementation
        pass

    def test_bollinger_bands(self, temp_dir):
        # Test implementation
        pass
```

## Quality Gates

### Automated Checks

- **Type Checking**: `mypy ztb/`
- **Linting**: `flake8 ztb/`
- **Formatting**: `black ztb/` and `isort ztb/`
- **Test Coverage**: Minimum 80% coverage required

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Debugging Tests

### Common Issues

**Test Failures**:
- Check environment variables
- Verify test data setup
- Review error messages and stack traces

**Slow Tests**:
- Use `@timed_test` to identify bottlenecks
- Profile with `cProfile`
- Consider test optimization or splitting

**Flaky Tests**:
- Add retries for network-dependent tests
- Use deterministic seeds for random operations
- Check for race conditions

### Test Debugging Tools

```python
# Debug specific test
python -m pytest ztb/tests/unit/test_feature.py::TestClass::test_method -v -s

# Run with debugging
python -m pytest --pdb ztb/tests/unit/test_feature.py

# Profile test performance
python -m pytest --durations=10 ztb/tests/unit/
```

## Multi-Python Testing with Nox

The project supports Python 3.11, 3.12, and 3.13. Use Nox for isolated testing across versions.

### Installation

```bash
pip install nox
```

### Running Tests with Nox

```bash
# Test all supported Python versions
nox -s test

# Test specific version
nox -s test-3.11
nox -s test-3.12
nox -s test-3.13

# Run linting across versions
nox -s lint

# Run type checking
nox -s type_check

# Run integration tests
nox -s integration_test
```

### Nox Sessions

- **test**: Run pytest with coverage across Python versions
- **lint**: Code formatting and style checks
- **type_check**: MyPy type checking
- **benchmark**: Performance benchmarking (Python 3.11 only)
- **safety**: Security vulnerability scanning
- **integration_test**: Full integration test suite

### Configuration

Nox configuration is in `noxfile.py`. Each session:

- Creates isolated virtual environment
- Installs dependencies from `requirements.txt` and `requirements-dev.txt`
- Runs specified commands
- Cleans up automatically

### Constraints File

For reproducible builds, use `constraints.txt`:

```bash
pip install -c constraints.txt -r requirements.txt
```

This ensures exact package versions across environments.

## CI/CD Integration

Tests run automatically on:

- Pull request creation
- Code pushes to main branch
- Scheduled nightly runs

### Test Results

- **Coverage Reports**: Generated in `coverage/`
- **Performance Metrics**: Logged to experiment tracking
- **Quality Gates**: Block merges if requirements not met

## Contributing

When adding new features:

1. Write tests first (TDD approach)
2. Ensure 80%+ test coverage
3. Add integration tests for new pipelines
4. Update this guide if new patterns emerge

## Resources

- [Test Utilities Guide](../util/README.md)
- [Development Setup](setup.md)
- [Architecture Overview](architecture.md)
