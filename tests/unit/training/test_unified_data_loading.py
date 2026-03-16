"""BaseAlgorithmTrainer.load_data() の統合テスト

Phase 4 Week 1 Day 2-3実装検証:
- CSV/Parquet自動検出
- 事前計算特徴の厳密な検出（42番再レビュー対応）
- SACTrainerでの統合経路確認
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ztb.training.unified_trainer.base.base_trainer import BaseAlgorithmTrainer


class TestTrainer(BaseAlgorithmTrainer):
    """テスト用のトレーナークラス"""

    def train(self):
        pass

    def validate_config(self):
        """設定検証（テスト用ダミー実装）"""
        pass

    def get_training_stats(self):
        """トレーニング統計取得（テスト用ダミー実装）"""
        return {}


def test_load_csv_data():
    """CSV読み込みの基本動作確認"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        # シンプルなCSV
        f.write("timestamp,open,high,low,close,volume\n")
        f.write("2024-01-01,100,101,99,100,1000\n")
        f.write("2024-01-02,100,102,99,101,1100\n")
        csv_path = f.name

    try:
        trainer = TestTrainer(config={})
        df = trainer.load_data(csv_path)

        assert len(df) == 2
        assert "open" in df.columns
        assert "close" in df.columns
    finally:
        Path(csv_path).unlink(missing_ok=True)


def test_load_parquet_with_precomputed_features():
    """Parquet + 事前計算特徴の読み込みと検出"""
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        # 8特徴を含むParquet
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=100, freq="1h"),
                "open": np.random.randn(100) + 100,
                "high": np.random.randn(100) + 101,
                "low": np.random.randn(100) + 99,
                "close": np.random.randn(100) + 100,
                "volume": np.random.randint(1000, 10000, 100),
                # 8特徴
                "rsi": np.random.randn(100),
                "macd": np.random.randn(100),
                "bb_width": np.random.randn(100),
                "volatility": np.random.randn(100),
                "momentum": np.random.randn(100),
                "volume_ma_ratio": np.random.randn(100),
                "atr": np.random.randn(100),
                "obv": np.random.randn(100),
            }
        )
        df.to_parquet(f.name)
        parquet_path = f.name

    try:
        trainer = TestTrainer(config={})
        loaded_df = trainer.load_data(parquet_path)

        # データが正しく読み込まれる
        assert len(loaded_df) == 100
        assert "rsi" in loaded_df.columns
        assert "macd" in loaded_df.columns

        # 事前計算特徴が検出される
        assert trainer._has_precomputed_features(loaded_df) is True

        # feature_set が minimal に設定される
        assert trainer.config.get("training", {}).get("features", {}).get(
            "feature_set"
        ) == "minimal"
    finally:
        Path(parquet_path).unlink(missing_ok=True)


def test_parquet_without_features_not_detected():
    """Parquet（OHLCVのみ）は事前計算特徴として検出されない"""
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        # OHLCVのみ
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=100, freq="1h"),
                "open": np.random.randn(100) + 100,
                "high": np.random.randn(100) + 101,
                "low": np.random.randn(100) + 99,
                "close": np.random.randn(100) + 100,
                "volume": np.random.randint(1000, 10000, 100),
            }
        )
        df.to_parquet(f.name)
        parquet_path = f.name

    try:
        trainer = TestTrainer(config={})
        loaded_df = trainer.load_data(parquet_path)

        # データが正しく読み込まれる
        assert len(loaded_df) == 100

        # 事前計算特徴は検出されない（OHLCVのみ）
        assert trainer._has_precomputed_features(loaded_df) is False

        # feature_set は設定されない
        assert (
            trainer.config.get("training", {}).get("features", {}).get("feature_set")
            is None
        )
    finally:
        Path(parquet_path).unlink(missing_ok=True)


def test_strict_feature_detection():
    """厳密な特徴検出ロジック（42番再レビュー対応）"""
    trainer = TestTrainer(config={})

    # ケース1: 5列以上 + 既知特徴3個以上 → 検出
    df_valid = pd.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "open": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [100, 101, 102],
            "volume": [1000, 1100, 1200],
            "rsi": [50, 55, 60],
            "macd": [0.1, 0.2, 0.3],
            "bb_width": [0.5, 0.6, 0.7],
            "momentum": [1.0, 1.1, 1.2],
            "atr": [0.8, 0.9, 1.0],
        }
    )
    assert trainer._has_precomputed_features(df_valid) is True

    # ケース2: 5列以上だが既知特徴が2個のみ → 検出しない
    df_invalid = pd.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "open": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [100, 101, 102],
            "volume": [1000, 1100, 1200],
            "rsi": [50, 55, 60],
            "unknown1": [0.1, 0.2, 0.3],
            "unknown2": [0.5, 0.6, 0.7],
            "unknown3": [1.0, 1.1, 1.2],
            "unknown4": [0.8, 0.9, 1.0],
        }
    )
    assert trainer._has_precomputed_features(df_invalid) is False

    # ケース3: 既知特徴が4個以上あれば4列でも検出しない（5列以上が必須）
    df_four_cols = pd.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "open": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [100, 101, 102],
            "volume": [1000, 1100, 1200],
            "rsi": [50, 55, 60],
            "macd": [0.1, 0.2, 0.3],
            "bb_width": [0.5, 0.6, 0.7],
            "momentum": [1.0, 1.1, 1.2],
        }
    )
    assert trainer._has_precomputed_features(df_four_cols) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
