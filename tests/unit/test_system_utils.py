#!/usr/bin/env python3
"""
Unit tests for system_utils.py

Tests for library availability checking and system utilities.
"""

import importlib
from unittest.mock import patch

import pytest

from ztb.utils.system_utils import check_library_availability, create_library_flags, safe_import


class TestLibraryAvailability:
    """Test library availability checking functions."""

    def test_check_library_availability_available(self):
        """Test checking availability of an available library."""
        # Test with a library that should be available (built-in)
        result = check_library_availability('os', 'OS utilities')
        assert result is True

    def test_check_library_availability_unavailable(self):
        """Test checking availability of an unavailable library."""
        # Test with a library that should not exist
        result = check_library_availability('nonexistent_library_xyz123', 'Test feature')
        assert result is False

    def test_safe_import_available(self):
        """Test safe import of an available library."""
        result = safe_import('os', 'OS utilities')
        assert result is not None
        assert hasattr(result, 'path')  # os module has path attribute

    def test_safe_import_unavailable(self):
        """Test safe import of an unavailable library."""
        result = safe_import('nonexistent_library_xyz123', 'Test feature')
        assert result is None

    @patch('ztb.utils.system_utils.check_library_availability')
    def test_create_library_flags(self, mock_check):
        """Test creating library availability flags."""
        # Mock the check function to return alternating True/False
        mock_check.side_effect = [True, False, True, False, True, False, True, False, True]

        flags = create_library_flags()

        expected_keys = [
            'OPTUNA_AVAILABLE',
            'TQDM_AVAILABLE',
            'PSUTIL_AVAILABLE',
            'SCIPY_AVAILABLE',
            'SKLEARN_AVAILABLE',
            'PANDAS_AVAILABLE',
            'NUMPY_AVAILABLE',
            'MATPLOTLIB_AVAILABLE',
            'SEABORN_AVAILABLE'
        ]

        for key in expected_keys:
            assert key in flags
            assert isinstance(flags[key], bool)

    def test_create_library_flags_real(self):
        """Test creating library flags with real libraries."""
        flags = create_library_flags()

        # At minimum, these should be available in our environment
        assert 'NUMPY_AVAILABLE' in flags
        assert isinstance(flags['NUMPY_AVAILABLE'], bool)

        # Test that the function completes without error
        assert len(flags) == 9  # Should have 9 library flags


class TestSystemConfiguration:
    """Test system configuration functions."""

    @patch('ztb.utils.system_utils.os.environ')
    def test_configure_pytorch_environment_cuda_enabled(self, mock_environ):
        """Test PyTorch environment configuration with CUDA enabled."""
        from ztb.utils.system_utils import configure_pytorch_environment

        configure_pytorch_environment(cuda_optimizations=True)

        # Check that CUDA-related environment variables are set
        assert mock_environ.__setitem__.called

        # Verify some key settings
        calls = mock_environ.__setitem__.call_args_list
        env_vars = {call[0][0]: call[0][1] for call in calls}

        assert env_vars.get("PYTORCH_DISABLE_TORCH_DYNAMO") == "1"
        assert env_vars.get("TORCH_USE_CUDA_DSA") == "1"
        assert env_vars.get("CUDA_LAUNCH_BLOCKING") == "1"

    @patch('ztb.utils.system_utils.os.environ')
    def test_configure_pytorch_environment_cuda_disabled(self, mock_environ):
        """Test PyTorch environment configuration with CUDA disabled."""
        from ztb.utils.system_utils import configure_pytorch_environment

        configure_pytorch_environment(cuda_optimizations=False)

        calls = mock_environ.__setitem__.call_args_list
        env_vars = {call[0][0]: call[0][1] for call in calls}

        assert env_vars.get("CUDA_VISIBLE_DEVICES") == ""

    @patch('ztb.utils.system_utils.os.environ')
    def test_get_system_info(self, mock_environ):
        """Test getting system information."""
        from ztb.utils.system_utils import get_system_info

        # Mock environment variables
        def mock_get(key, default=None):
            env_vars = {
                "CUDA_VISIBLE_DEVICES": "",
                "PYTORCH_DISABLE_TORCH_DYNAMO": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"
            }
            return env_vars.get(key, default)

        def mock_contains(key):
            env_vars = {
                "CUDA_VISIBLE_DEVICES": "",
                "PYTORCH_DISABLE_TORCH_DYNAMO": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"
            }
            return key in env_vars

        mock_environ.get.side_effect = mock_get
        mock_environ.__contains__.side_effect = mock_contains

        info = get_system_info()

        assert 'cuda_available' in info
        assert 'pytorch_dynamo_disabled' in info
        assert 'memory_optimized' in info

        assert info['cuda_available'] is False
        assert info['pytorch_dynamo_disabled'] is True
        assert info['memory_optimized'] is True


if __name__ == "__main__":
    pytest.main([__file__])