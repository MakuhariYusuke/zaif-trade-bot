#!/usr/bin/env python3
"""
NumPy Compatibility Test for SAC v446

NumPyバージョン互換性テスト
"""

import sys
import numpy as np
import pandas as pd
import scipy.stats
import logging
from typing import Dict, Any, List

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_numpy_version():
    """NumPyバージョン確認"""
    logger.info(f"NumPy version: {np.__version__}")
    logger.info(f"Python version: {sys.version}")

    # バージョン要件チェック
    version_parts = np.__version__.split('.')
    major = int(version_parts[0])
    minor = int(version_parts[1])

    # 1.24.4 以上であることを確認
    if (major > 1) or (major == 1 and minor >= 24):
        logger.info(f"NumPy version {np.__version__} is compatible (>= 1.24.4)")
        return True
    else:
        logger.error(f"NumPy version {np.__version__} is too old. Required: >= 1.24.4")
        return False


def test_basic_numpy_operations():
    """基本的なNumPy操作テスト"""
    try:
        # 基本配列操作
        arr = np.array([1, 2, 3, 4, 5])
        logger.info("✓ Basic array creation")

        # 統計関数
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        logger.info(f"✓ Statistical functions: mean={mean_val}, std={std_val}")

        # ブロードキャスティング
        arr2 = arr * 2
        logger.info("✓ Broadcasting operations")

        # インデックス操作
        subset = arr[1:4]
        logger.info("✓ Indexing operations")

        return True
    except Exception as e:
        logger.error(f"✗ Basic NumPy operations failed: {e}")
        return False


def test_scipy_operations():
    """SciPy操作テスト"""
    try:
        # Z-score計算
        data = np.random.normal(0, 1, 1000)
        z_scores = scipy.stats.zscore(data)
        logger.info("✓ SciPy zscore calculation")

        # IQR計算
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        logger.info(f"✓ IQR calculation: IQR={iqr:.4f}")

        return True
    except Exception as e:
        logger.error(f"✗ SciPy operations failed: {e}")
        return False


def test_pandas_operations():
    """Pandas操作テスト"""
    try:
        # DataFrame作成
        df = pd.DataFrame({
            'A': np.random.randn(100),
            'B': np.random.randn(100),
            'C': np.random.randint(0, 10, 100)
        })
        logger.info("✓ DataFrame creation")

        # 統計量計算
        stats = df.describe()
        logger.info("✓ DataFrame statistics")

        # 欠損値処理
        df_with_nan = df.copy()
        df_with_nan.loc[0:10, 'A'] = np.nan
        df_filled = df_with_nan.fillna(df_with_nan.mean())
        logger.info("✓ NaN handling")

        return True
    except Exception as e:
        logger.error(f"✗ Pandas operations failed: {e}")
        return False


def test_memory_operations():
    """メモリ操作テスト"""
    try:
        # 大規模配列作成
        large_array = np.random.randn(10000, 100)
        logger.info(f"✓ Large array creation: shape={large_array.shape}")

        # メモリ使用量確認
        size_bytes = large_array.nbytes
        size_mb = size_bytes / (1024 * 1024)
        logger.info(f"✓ Memory usage: {size_mb:.2f} MB")

        # 配列操作
        result = np.mean(large_array, axis=0)
        logger.info(f"✓ Large array operations: result shape={result.shape}")

        # メモリ解放
        del large_array, result

        return True
    except Exception as e:
        logger.error(f"✗ Memory operations failed: {e}")
        return False


def test_ml_operations():
    """ML関連操作テスト"""
    try:
        # 特徴量行列
        X = np.random.randn(1000, 50)
        y = np.random.randint(0, 2, 1000)

        # 基本的な行列演算
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_normalized = (X - X_mean) / (X_std + 1e-8)
        logger.info("✓ Feature normalization")

        # 相関係数
        corr_matrix = np.corrcoef(X_normalized.T)
        logger.info(f"✓ Correlation matrix: shape={corr_matrix.shape}")

        return True
    except Exception as e:
        logger.error(f"✗ ML operations failed: {e}")
        return False


def run_environment_tests():
    """環境テスト実行"""
    logger.info("Starting NumPy compatibility tests for SAC v446")
    logger.info("=" * 50)

    tests = [
        ("NumPy Version Check", test_numpy_version),
        ("Basic NumPy Operations", test_basic_numpy_operations),
        ("SciPy Operations", test_scipy_operations),
        ("Pandas Operations", test_pandas_operations),
        ("Memory Operations", test_memory_operations),
        ("ML Operations", test_ml_operations),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\nRunning: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "PASS" if result else "FAIL"
            logger.info(f"Result: {status}")
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))

    # 結果サマリー
    logger.info("\n" + "=" * 50)
    logger.info("TEST RESULTS SUMMARY")

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
        if result:
            passed += 1

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! Environment is ready for SAC v446.")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the environment setup.")
        return False


if __name__ == "__main__":
    success = run_environment_tests()
    sys.exit(0 if success else 1)