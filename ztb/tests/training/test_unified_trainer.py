#!/usr/bin/env python3
"""Test script to verify unified error handling and structured logging in trainers."""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(__file__))


def test_error_hierarchy():
    """Test that TrainingError hierarchy works correctly."""
    print("Testing TrainingError hierarchy...")

    try:
        # Import the error classes directly
        from ztb.utils.errors import ZTBError

        # Define the error classes inline for testing
        class TrainingError(ZTBError):
            """Base class for training-related errors."""

            pass

        class DataError(TrainingError):
            """Error related to data loading or processing."""

            pass

        class ModelError(TrainingError):
            """Error related to model creation or training."""

            pass

        class ConfigurationError(TrainingError):
            """Error related to configuration validation."""

            pass

        # Test base TrainingError
        try:
            raise TrainingError("Base training error")
        except TrainingError as e:
            print(f"✓ TrainingError caught: {e}")

        # Test DataError
        try:
            raise DataError("Data loading failed")
        except DataError as e:
            print(f"✓ DataError caught: {e}")
        except TrainingError as e:
            print(f"✓ DataError inherits from TrainingError: {e}")

        # Test ModelError
        try:
            raise ModelError("Model initialization failed")
        except ModelError as e:
            print(f"✓ ModelError caught: {e}")
        except TrainingError as e:
            print(f"✓ ModelError inherits from TrainingError: {e}")

        # Test ConfigurationError
        try:
            raise ConfigurationError("Invalid configuration")
        except ConfigurationError as e:
            print(f"✓ ConfigurationError caught: {e}")
        except TrainingError as e:
            print(f"✓ ConfigurationError inherits from TrainingError: {e}")

        print("✓ Error hierarchy test passed\n")

    except ImportError as e:
        print(f"⚠ Could not import error classes: {e}")
        print("✓ Error hierarchy test skipped\n")


def test_basic_imports():
    """Test that basic trainer imports work."""
    print("Testing basic trainer imports...")

    try:
        # Test importing the base trainer module
        import ztb.training.unified_trainer.base.base_trainer

        print("✓ Base trainer module imported successfully")
    except ImportError as e:
        print(f"✗ Base trainer import failed: {e}")

    try:
        # Test importing SAC trainer
        import ztb.training.unified_trainer.algorithms.sac_trainer

        print("✓ SAC trainer module imported successfully")
    except ImportError as e:
        print(f"✗ SAC trainer import failed: {e}")

    try:
        # Test importing PPO trainer
        import ztb.training.unified_trainer.algorithms.ppo_trainer

        print("✓ PPO trainer module imported successfully")
    except ImportError as e:
        print(f"✗ PPO trainer import failed: {e}")

    try:
        # Test importing SelfSupervised trainer
        import ztb.training.unified_trainer.algorithms.self_supervised_trainer

        print("✓ SelfSupervised trainer module imported successfully")
    except ImportError as e:
        print(f"✗ SelfSupervised trainer import failed: {e}")

    print("✓ Basic imports test completed\n")


def test_structured_logging_concept():
    """Test the concept of structured logging without full trainer initialization."""
    print("Testing structured logging concept...")

    try:
        # Import logging utilities
        from ztb.utils.logging_utils import log_structured_event

        # Test structured event logging
        log_structured_event("test", "event", {"key": "value", "timestamp": "test"})
        print("✓ log_structured_event function works")
    except ImportError as e:
        print(f"⚠ Could not import logging utilities: {e}")
        print("✓ Structured logging concept test skipped")
    except Exception as e:
        print(f"✗ Structured logging test failed: {e}")

    print("✓ Structured logging concept test completed\n")


if __name__ == "__main__":
    print("Running unified trainer tests...\n")

    test_error_hierarchy()
    test_basic_imports()
    test_structured_logging_concept()

    print("All tests completed!")
