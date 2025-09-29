"""Test that no Python files remain in ztb/scripts/ directory."""

from pathlib import Path


def test_no_py_files_in_scripts_dir():
    """Assert that no *.py files exist in ztb/scripts/ except the deprecated trading_service shim."""
    scripts_dir = Path(__file__).parent.parent.parent.parent / "ztb" / "scripts"
    py_files = list(scripts_dir.glob("*.py"))
    # Allow the deprecated trading_service.py shim
    allowed_files = {"trading_service.py"}
    py_files = [f for f in py_files if f.name not in allowed_files]
    assert not py_files, f"Found Python files in scripts dir: {py_files}"
