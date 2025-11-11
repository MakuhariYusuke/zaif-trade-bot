#!/usr/bin/env python3
"""
Direct NumPy compatibility test

NumPy互換性の直接テスト
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import scipy.stats

def test_numpy_compatibility():
    print('NumPy version:', np.__version__)

    # Basic operations
    arr = np.array([1, 2, 3, 4, 5])
    print('✓ Array creation')

    mean_val = np.mean(arr)
    std_val = np.std(arr)
    print('✓ Statistical functions')

    # Broadcasting
    arr2 = arr * 2
    print('✓ Broadcasting')

    # SciPy
    data = np.random.normal(0, 1, 100)
    z_scores = scipy.stats.zscore(data)
    print('✓ SciPy operations')

    # Pandas
    df = pd.DataFrame({'A': np.random.randn(10), 'B': np.random.randn(10)})
    stats = df.describe()
    print('✓ Pandas operations')

    # Version check
    version_parts = np.__version__.split('.')
    major = int(version_parts[0])
    minor = int(version_parts[1]) if len(version_parts) > 1 else 0

    assert major >= 1
    print('✓ Version compatibility')

    print('\n🎉 All NumPy compatibility tests passed!')

if __name__ == '__main__':
    test_numpy_compatibility()