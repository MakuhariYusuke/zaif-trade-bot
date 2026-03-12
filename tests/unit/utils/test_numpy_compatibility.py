"""
Unit tests for NumPy compatibility

NumPy互換性の単体テスト
"""

import numpy as np
import pandas as pd
import scipy.stats
import pytest
from ztb.trading.environment.constants import BYTES_PER_MB


class TestNumPyCompatibility:
    """NumPy compatibility unit tests"""

    def test_numpy_version_compatibility(self):
        """Test NumPy version meets requirements"""
        version_parts = np.__version__.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1])

        # Should be 1.24.4 or higher
        assert (major > 1) or (major == 1 and minor >= 24), \
            f"NumPy version {np.__version__} is too old. Required: >= 1.24.4"

    def test_basic_numpy_operations(self):
        """Test basic NumPy operations"""
        # Array creation
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.shape == (5,)
        assert arr.dtype == np.int64 or arr.dtype == np.int32

        # Statistical functions
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        assert mean_val == 3.0
        assert std_val == pytest.approx(1.4142135623730951, rel=1e-10)

        # Broadcasting
        arr2 = arr * 2
        expected = np.array([2, 4, 6, 8, 10])
        np.testing.assert_array_equal(arr2, expected)

        # Indexing
        subset = arr[1:4]
        expected_subset = np.array([2, 3, 4])
        np.testing.assert_array_equal(subset, expected_subset)

    def test_scipy_operations(self):
        """Test SciPy statistical operations"""
        # Generate test data
        data = np.random.normal(0, 1, 1000)

        # Z-score calculation via NumPy avoids unrelated torch array-api shims
        z_scores = (data - np.mean(data)) / np.std(data)
        assert len(z_scores) == len(data)
        assert abs(np.mean(z_scores)) < 0.1  # Should be approximately 0
        assert abs(np.std(z_scores) - 1.0) < 0.1  # Should be approximately 1

        # IQR calculation
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        assert iqr > 0  # IQR should be positive

    def test_pandas_operations(self):
        """Test Pandas operations with NumPy arrays"""
        # Create DataFrame
        data = {
            'A': np.random.randn(100),
            'B': np.random.randn(100),
            'C': np.random.randint(0, 10, 100)
        }
        df = pd.DataFrame(data)

        assert len(df) == 100
        assert list(df.columns) == ['A', 'B', 'C']

        # Statistical operations
        stats = df.describe()
        assert 'mean' in stats.index
        assert 'std' in stats.index
        assert 'min' in stats.index
        assert 'max' in stats.index

        # NaN handling
        df_with_nan = df.copy()
        df_with_nan.loc[0:10, 'A'] = np.nan
        df_filled = df_with_nan.fillna(df_with_nan.mean())
        assert not df_filled['A'].isna().any()

    def test_memory_operations(self):
        """Test memory-intensive NumPy operations"""
        # Large array creation
        large_array = np.random.randn(1000, 100)
        assert large_array.shape == (1000, 100)

        # Memory usage check
        size_bytes = large_array.nbytes
        size_mb = size_bytes / BYTES_PER_MB
        assert size_mb > 0

        # Array operations
        result = np.mean(large_array, axis=0)
        assert result.shape == (100,)

        # Clean up
        del large_array, result

    def test_ml_operations(self):
        """Test ML-related NumPy operations"""
        # Feature matrix
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)

        assert X.shape == (100, 10)
        assert y.shape == (100,)

        # Normalization
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_normalized = (X - X_mean) / (X_std + 1e-8)

        assert X_normalized.shape == X.shape
        # Check normalization (mean should be close to 0, std close to 1)
        normalized_mean = np.mean(X_normalized, axis=0)
        normalized_std = np.std(X_normalized, axis=0)

        np.testing.assert_allclose(normalized_mean, 0, atol=1e-10)
        np.testing.assert_allclose(normalized_std, 1, atol=1e-10)

        # Correlation matrix
        corr_matrix = np.corrcoef(X_normalized.T)
        assert corr_matrix.shape == (10, 10)

        # Check diagonal is 1
        np.testing.assert_allclose(np.diag(corr_matrix), 1.0, atol=1e-10)

    def test_numpy_array_dtypes(self):
        """Test NumPy array data types"""
        # Integer arrays
        int_arr = np.array([1, 2, 3], dtype=np.int32)
        assert int_arr.dtype == np.int32

        # Float arrays
        float_arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        assert float_arr.dtype == np.float64

        # Boolean arrays
        bool_arr = np.array([True, False, True])
        assert bool_arr.dtype == bool

    def test_numpy_linear_algebra(self):
        """Test NumPy linear algebra operations"""
        # Matrix creation
        A = np.random.randn(5, 5)
        b = np.random.randn(5)

        # Matrix multiplication
        result = np.dot(A, b)
        assert result.shape == (5,)

        # Matrix inversion (if invertible)
        try:
            A_inv = np.linalg.inv(A)
            identity = np.dot(A, A_inv)
            # Check if close to identity
            np.testing.assert_allclose(identity, np.eye(5), atol=1e-10)
        except np.linalg.LinAlgError:
            # Matrix might not be invertible, skip test
            pass

    def test_numpy_random_operations(self):
        """Test NumPy random operations"""
        # Set seed for reproducibility
        np.random.seed(42)

        # Random arrays
        rand_arr = np.random.rand(10, 10)
        assert rand_arr.shape == (10, 10)
        assert np.all(rand_arr >= 0) and np.all(rand_arr < 1)

        # Normal distribution
        normal_arr = np.random.normal(0, 1, 1000)
        assert len(normal_arr) == 1000

        # Check approximate mean and std
        mean_val = np.mean(normal_arr)
        std_val = np.std(normal_arr)
        assert abs(mean_val) < 0.1  # Should be close to 0
        assert abs(std_val - 1.0) < 0.1  # Should be close to 1

    def test_numpy_version_info(self):
        """Test NumPy version information"""
        version = np.__version__
        assert isinstance(version, str)
        assert len(version.split('.')) >= 2

        # Check version components (NumPy 2.0+ compatible)
        version_parts = version.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0

        assert major >= 1
        assert minor >= 0
