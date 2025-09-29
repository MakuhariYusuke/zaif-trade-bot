"""
Unit Tests for Import Validation
インポート検証の単体テスト

This module tests that all major modules can be imported without errors,
validating the package structure and dependencies.
"""

import sys
import unittest
from pathlib import Path

# Add project root to path for testing
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class TestImports(unittest.TestCase):
    """Test that all major modules can be imported"""

    def test_experiments_imports(self):
        """Test experiments module imports"""
        try:
            from ztb.experiments.base import ExperimentResult
            from ztb.experiments.ml_reinforcement_1k import (
                MLReinforcement100KExperiment,
            )

            self.assertIsNotNone(ExperimentResult)
            self.assertIsNotNone(MLReinforcement100KExperiment)
            print("✅ Experiments imports: OK")
        except ImportError as e:
            self.fail(f"Experiments import failed: {e}")

    def test_utils_imports(self):
        """Test utils module imports"""
        try:
            from ztb.utils.parallel_experiments import run_parallel_experiments

            self.assertIsNotNone(run_parallel_experiments)
            print("✅ Utils imports: OK")
        except ImportError as e:
            self.fail(f"Utils import failed: {e}")

    def test_all_major_modules_importable(self):
        """Test that all major modules are importable without exceptions"""
        modules_to_test = [
            "ztb.experiments.base",
            "ztb.experiments.ml_reinforcement_1k",
            "ztb.utils.parallel_experiments",
        ]

        failed_imports = []

        for module in modules_to_test:
            try:
                __import__(module)
            except ImportError as e:
                failed_imports.append(f"{module}: {e}")
            except Exception as e:
                failed_imports.append(f"{module}: {e}")

        if failed_imports:
            self.fail(f"Failed to import modules: {failed_imports}")

        print(f"✅ All {len(modules_to_test)} major modules imported successfully")

    def test_package_structure(self):
        """Test that package __init__.py files exist"""
        required_init_files = [
            project_root / "ztb" / "__init__.py",
            project_root / "ztb" / "experiments" / "__init__.py",
            project_root / "ztb" / "utils" / "__init__.py",
            project_root / "ztb" / "tests" / "__init__.py",
            project_root / "ztb" / "tests" / "unit" / "__init__.py",
        ]

        missing_files = []
        for init_file in required_init_files:
            if not init_file.exists():
                missing_files.append(str(init_file))

        if missing_files:
            self.fail(f"Missing __init__.py files: {missing_files}")

        print(f"✅ All {len(required_init_files)} __init__.py files present")


if __name__ == "__main__":
    unittest.main()
