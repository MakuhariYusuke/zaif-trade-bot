"""
Nox configuration for multi-Python testing.

Provides isolated testing environments for Python 3.11, 3.12, and 3.13.
"""

import nox


@nox.session(python=["3.11", "3.12", "3.13"])
def test(session):
    """Run tests with specific Python version."""
    session.install("-r", "requirements.txt", "-r", "requirements-dev.txt")
    session.install("-e", ".")

    # Run tests
    session.run("python", "-m", "pytest", "--cov=ztb", "--cov-report=html", "tests/")

    # Type checking
    session.run("python", "-m", "mypy", "ztb/")

    # Import sorting check
    session.run(
        "python", "-m", "isort", "--check-only", "--diff", "ztb/", "ztb/scripts/"
    )


@nox.session(python=["3.11", "3.12", "3.13"])
def lint(session):
    """Run linting checks."""
    session.install("-r", "requirements-dev.txt")

    # Code formatting
    session.run("python", "-m", "black", "--check", "--diff", "ztb/", "ztb/scripts/")
    session.run(
        "python", "-m", "isort", "--check-only", "--diff", "ztb/", "ztb/scripts/"
    )

    # Linting
    session.run("python", "-m", "flake8", "ztb/", "ztb/scripts/")


@nox.session(python=["3.11", "3.12", "3.13"])
def type_check(session):
    """Run type checking."""
    session.install("-r", "requirements-dev.txt")
    session.install("-e", ".")

    session.run("python", "-m", "mypy", "ztb/", "ztb/scripts/")


@nox.session(python="3.11")
def benchmark(session):
    """Run performance benchmarks."""
    session.install("-r", "requirements.txt")
    session.install("-e", ".")

    # Run benchmark scripts
    session.run("python", "benchmark_features.py")
    session.run("python", "benchmark_compression.py")


@nox.session
def docs(session):
    """Build documentation."""
    session.install("-r", "requirements-dev.txt")

    # Build docs (if sphinx is configured)
    # session.run("sphinx-build", "docs/", "docs/_build/")


@nox.session(python="3.11")
def safety(session):
    """Run security checks."""
    session.install("-r", "requirements-dev.txt")

    # Check for known vulnerabilities
    session.run("python", "-m", "safety", "check", "--file", "requirements.txt")

    # Run bandit for security linting
    session.run("python", "-m", "bandit", "-r", "ztb/")


@nox.session(python=["3.11", "3.12", "3.13"])
def integration_test(session):
    """Run integration tests."""
    session.install("-r", "requirements.txt", "-r", "requirements-dev.txt")
    session.install("-e", ".")

    # Run integration tests
    session.run("python", "-m", "pytest", "tests/", "-m", "integration", "--tb=short")
