#!/usr/bin/env python3
"""
Test script for enhanced optimizer features with statistical improvements
"""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.features.optimizer_features import OptimizerFeatureTracker


def test_basic_functionality():
    """基本機能のテスト"""
    print("Testing basic functionality...")

    tracker = OptimizerFeatureTracker(max_history=100)

    # テストデータを追加
    for i in range(10):
        lr = 0.001 * (0.9**i)  # 学習率の減衰
        grad_norm = 1.0 + 0.1 * np.random.randn()  # 勾配ノルム
        step_size = lr * 0.1  # ステップサイズ

        tracker.update_optimizer_features(
            step=i, learning_rate=lr, gradient_norm=grad_norm, step_size=step_size
        )

    # 特徴量を取得
    features = tracker.get_feature_vector(include_debug_info=True)
    print(f"Generated {len(features) - 1} features")  # -1 for debug_info

    # 基本的な特徴量をチェック
    assert "optimizer_learning_rate" in features
    assert "optimizer_gradient_norm_avg" in features
    assert features["optimizer_learning_rate"] > 0

    print("✓ Basic functionality test passed")


def test_statistical_improvements():
    """統計的改善のテスト"""
    print("Testing statistical improvements...")

    tracker = OptimizerFeatureTracker(
        max_history=100,
        enable_normalization=True,
        normalization_method="robust",
        outlier_threshold=1.5,
    )

    # 外れ値を含むテストデータ
    np.random.seed(42)
    for i in range(50):
        # 正常データ
        lr = 0.001 + 0.0001 * np.random.randn()
        grad_norm = 1.0 + 0.2 * np.random.randn()

        # 外れ値を追加
        if i == 25:
            lr = 100.0  # 極端な外れ値
            grad_norm = -10.0  # 負の外れ値

        tracker.update_optimizer_features(
            step=i, learning_rate=lr, gradient_norm=grad_norm, step_size=lr * 0.1
        )

    # 相関分析をテスト
    correlations = tracker.compute_feature_correlations()
    print(f"Computed correlations for {len(correlations)} features")

    # 重要度評価をテスト
    importance = tracker.compute_feature_importance()
    print(f"Computed importance scores for {len(importance)} features")

    # デバッグ情報をテスト
    features_with_debug = tracker.get_feature_vector(include_debug_info=True)
    debug_info = features_with_debug.get("_debug_info", {})
    print(f"Debug info includes {len(debug_info)} metrics")

    assert len(correlations) > 0
    assert len(importance) > 0
    assert "update_count" in debug_info

    print("✓ Statistical improvements test passed")


def test_error_handling():
    """エラーハンドリングのテスト"""
    print("Testing error handling...")

    tracker = OptimizerFeatureTracker()

    # 無効なデータを追加
    tracker.update_optimizer_features(
        step=0, learning_rate=float("nan"), gradient_norm=float("inf"), step_size=-1.0
    )

    # 特徴量を取得（NaN/infが適切に処理されるはず）
    features = tracker.get_feature_vector()

    # すべての特徴量が有限値であることを確認
    for name, value in features.items():
        if name != "_debug_info":
            assert np.isfinite(value), f"Non-finite value for {name}: {value}"

    print("✓ Error handling test passed")


def test_performance_monitoring():
    """パフォーマンス監視のテスト"""
    print("Testing performance monitoring...")

    tracker = OptimizerFeatureTracker()

    import time

    start_time = time.time()

    # 多数の更新を実行
    for i in range(100):
        tracker.update_optimizer_features(
            step=i, learning_rate=0.001, gradient_norm=1.0, step_size=0.0001
        )

    end_time = time.time()

    # デバッグ情報を確認
    features = tracker.get_feature_vector(include_debug_info=True)
    debug_info = features.get("_debug_info", {})

    assert debug_info["update_count"] == 100
    assert debug_info["error_count"] == 0  # エラーがないはず
    assert "last_update_time" in debug_info

    print(f"Processed 100 updates in {end_time - start_time:.4f} seconds")
    print("✓ Performance monitoring test passed")


if __name__ == "__main__":
    print("Running optimizer features enhancement tests...\n")

    try:
        test_basic_functionality()
        test_statistical_improvements()
        test_error_handling()
        test_performance_monitoring()

        print(
            "\n🎉 All tests passed! Optimizer features enhancements are working correctly."
        )

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
