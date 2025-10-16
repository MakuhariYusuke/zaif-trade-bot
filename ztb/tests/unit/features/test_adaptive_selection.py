#!/usr/bin/env python3
"""
Unit tests for adaptive feature selection
適応型特徴量選択のユニットテスト
"""

import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ztb.features.adaptive_selection import (
    AdaptiveFeatureSelector,
    MarketRegimeClassifier,
    FeatureAttentionLayer
)


class TestMarketRegimeClassifier(unittest.TestCase):
    """MarketRegimeClassifierのテスト"""

    def setUp(self):
        self.classifier = MarketRegimeClassifier()

    def test_classify_trending(self):
        """トレンド相場判定テスト"""
        df = pd.DataFrame({
            'ADX': [30.0],  # 高ADX = トレンド
            'ATR': [1.0]
        })
        regime = self.classifier.classify_market_regime(df)
        self.assertEqual(regime, "trending")

    def test_classify_ranging(self):
        """レンジ相場判定テスト"""
        df = pd.DataFrame({
            'ADX': [15.0],  # 低ADX = レンジ
            'ATR': [1.0]
        })
        regime = self.classifier.classify_market_regime(df)
        self.assertEqual(regime, "ranging")

    def test_classify_high_volatility(self):
        """高ボラティリティ判定テスト"""
        # まず履歴を蓄積
        for i in range(25):
            df = pd.DataFrame({'ATR': [1.0]})
            self.classifier.classify_market_regime(df)

        # 高ボラティリティ
        df = pd.DataFrame({
            'ADX': [20.0],
            'ATR': [3.0]  # 高いATR
        })
        regime = self.classifier.classify_market_regime(df)
        self.assertEqual(regime, "high_volatility")

    def test_classify_low_volatility(self):
        """低ボラティリティ判定テスト"""
        # まず履歴を蓄積
        for i in range(25):
            df = pd.DataFrame({'ATR': [2.0]})
            self.classifier.classify_market_regime(df)

        # 低ボラティリティ
        df = pd.DataFrame({
            'ADX': [20.0],
            'ATR': [0.5]  # 低いATR
        })
        regime = self.classifier.classify_market_regime(df)
        self.assertEqual(regime, "low_volatility")


class TestAdaptiveFeatureSelector(unittest.TestCase):
    """AdaptiveFeatureSelectorのテスト"""

    def setUp(self):
        self.selector = AdaptiveFeatureSelector()

    def test_get_regime_weights_trending(self):
        """トレンド相場での重み取得テスト"""
        weights = self.selector.get_regime_weights("trending")

        # トレンド関連特徴量が高重み
        self.assertEqual(weights.get("ADX"), 1.0)
        self.assertEqual(weights.get("MACD"), 1.0)

        # レンジ関連特徴量が低重み
        self.assertEqual(weights.get("RSI"), 0.3)

    def test_get_regime_weights_ranging(self):
        """レンジ相場での重み取得テスト"""
        weights = self.selector.get_regime_weights("ranging")

        # レンジ関連特徴量が高重み
        self.assertEqual(weights.get("RSI"), 1.0)
        self.assertEqual(weights.get("STOCH_K"), 1.0)

        # トレンド関連特徴量が低重み
        self.assertEqual(weights.get("ADX"), 0.3)

    def test_select_features_adaptive(self):
        """適応型特徴量選択テスト"""
        # サンプルデータ
        df = pd.DataFrame({
            'ADX': [30.0],
            'ATR': [1.0],
            'RSI': [50.0],
            'MACD': [0.1],
            'VOLUME_RATIO': [1.2]
        })

        all_features = ['ADX', 'ATR', 'RSI', 'MACD', 'VOLUME_RATIO']

        selected_features, weights = self.selector.select_features_adaptive(df, all_features)

        # 特徴量が選択されていることを確認
        self.assertGreater(len(selected_features), 0)
        self.assertEqual(len(selected_features), len(weights))

        # 重みが正規化されていることを確認
        self.assertTrue(all(w >= 0.4 for w in weights))


class TestFeatureAttentionLayer(unittest.TestCase):
    """FeatureAttentionLayerのテスト"""

    def setUp(self):
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")
        self.layer = FeatureAttentionLayer(n_features=10, hidden_dim=8)

    def test_forward(self):
        """順伝播テスト"""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        batch_size = 4
        n_features = 10

        x = torch.randn(batch_size, n_features)
        weights = self.layer(x)

        # 出力形状チェック
        self.assertEqual(weights.shape, (batch_size, n_features))

        # 重みが0-1の範囲であることを確認
        self.assertTrue(torch.all(weights >= 0))
        self.assertTrue(torch.all(weights <= 1))


if __name__ == "__main__":
    unittest.main()