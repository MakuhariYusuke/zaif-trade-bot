#!/usr/bin/env python3
"""
SignalQualityScorer単体テスト
SIGNAL_GUIDANCEの各コンポーネントをテスト
"""

import unittest
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.trading.signal.quality_scorer import SignalQualityScorer
from ztb.trading.signal.timeframe.phase4_manager import Phase4MinuteTradingManager


class TestSignalQualityScorer(unittest.TestCase):
    """SignalQualityScorerのテスト"""

    def setUp(self):
        """テスト前の準備"""
        self.scorer = SignalQualityScorer()
        # テスト用の市場データ作成
        self.test_df = pd.DataFrame({
            'open': [50000, 50100, 49900, 50200, 50300],
            'high': [50100, 50200, 50000, 50300, 50400],
            'low': [49900, 50000, 49800, 50100, 50200],
            'close': [50050, 50080, 49950, 50250, 50350],
            'volume': [100, 110, 90, 120, 130]
        })

        self.portfolio = {
            'position': 0.0,
            'cash': 1000000,
            'value': 1000000
        }

    def test_initialization(self):
        """初期化テスト"""
        self.assertIsInstance(self.scorer, SignalQualityScorer)
        self.assertIsNotNone(self.scorer.weights)

    def test_calculate_signal_quality(self):
        """シグナル品質計算テスト"""
        action, score = self.scorer.calculate_signal_quality(
            self.test_df, 0.0, self.portfolio
        )

        # 戻り値がタプルであることを確認
        self.assertIsInstance(action, int)
        self.assertIsInstance(score, float)

        # アクションが有効範囲内
        self.assertIn(action, [-1, 0, 1])

        # スコアが0-100の範囲内
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

    def test_buy_signal_generation(self):
        """買いシグナル生成テスト"""
        # 強い買いシグナルとなるデータを準備
        buy_df = pd.DataFrame({
            'open': [50000, 49900, 49800, 49700, 49600],
            'high': [50100, 50000, 49900, 49800, 49700],
            'low': [49900, 49800, 49700, 49600, 49500],
            'close': [50080, 49980, 49880, 49780, 49680],  # 上昇トレンド
            'volume': [100, 110, 120, 130, 140]  # 出来高増加
        })

        action, score = self.scorer.calculate_signal_quality(
            buy_df, 0.5, self.portfolio
        )

        # 買いシグナルが生成されることを期待
        self.assertEqual(action, 1)
        self.assertGreater(score, 70)

    def test_sell_signal_generation(self):
        """売りシグナル生成テスト"""
        # 強い売りシグナルとなるデータを準備
        sell_df = pd.DataFrame({
            'open': [50000, 50100, 50200, 50300, 50400],
            'high': [50100, 50200, 50300, 50400, 50500],
            'low': [49900, 50000, 50100, 50200, 50300],
            'close': [49920, 50020, 50120, 50220, 50320],  # 下落トレンド
            'volume': [100, 110, 120, 130, 140]  # 出来高増加
        })

        action, score = self.scorer.calculate_signal_quality(
            sell_df, -0.5, self.portfolio
        )

        # 売りシグナルが生成されることを期待
        self.assertEqual(action, -1)
        self.assertLess(score, 30)


class TestSignalGuidanceBacktestEnv(unittest.TestCase):
    """SignalGuidanceBacktestEnvのテスト"""

    def setUp(self):
        """テスト前の準備"""
        # モックデータ作成
        self.mock_df = pd.DataFrame({
            'close': [50000, 50100, 49900, 50200],
            'high': [50100, 50200, 50000, 50300],
            'low': [49900, 50000, 49800, 50100],
            'volume': [100, 110, 90, 120]
        })

        self.config = {
            'transaction_cost': 0.001,
            'max_position_size': 0.1,
            'feature_names': list(self.mock_df.columns),
            'reward_scaling': 1.0,
            'max_steps': 100
        }

    @patch('ztb.trading.signal.quality_scorer.SignalQualityScorer')
    @patch('ztb.trading.signal.timeframe.phase4_manager.Phase4MinuteTradingManager')
    def test_env_initialization(self, mock_phase4, mock_scorer):
        """環境初期化テスト"""
        from signal_guidance_backtest import SignalGuidanceBacktestEnv

        env = SignalGuidanceBacktestEnv(self.mock_df, self.config)

        self.assertIsNotNone(env.signal_scorer)
        self.assertIsNotNone(env.phase4_manager)
        self.assertEqual(env.initial_balance, 1000000)

    def test_technical_signals_extraction(self):
        """技術指標抽出テスト"""
        from signal_guidance_backtest import SignalGuidanceBacktestEnv

        env = SignalGuidanceBacktestEnv(self.mock_df, self.config)

        # V4FeatureExtractor形式の観測データ（Supertrend, Supertrend_Direction, OBV）
        observation = np.array([1.2, 1.0, 0.7])

        signals = env._extract_technical_signals(observation)

        # 抽出されたシグナルを確認
        self.assertIn('supertrend', signals)
        self.assertIn('supertrend_direction', signals)
        self.assertIn('obv', signals)
        self.assertIn('bb_position', signals)  # Supertrend_Directionから派生

        self.assertEqual(signals['supertrend'], 1.2)
        self.assertEqual(signals['supertrend_direction'], 1.0)
        self.assertEqual(signals['obv'], 0.7)

    def test_signal_guidance_scoring(self):
        """SIGNAL_GUIDANCEスコアリングテスト"""
        from signal_guidance_backtest import SignalGuidanceBacktestEnv

        env = SignalGuidanceBacktestEnv(self.mock_df, self.config)

        # テスト用の観測データ
        observation = np.array([1.0, 1.0, 0.8])  # Supertrend, Direction=上昇, OBV=買い

        action, score = env._get_signal_guidance_score(observation, 0.0)

        # スコアが計算されることを確認
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

        # 上昇トレンド + 買いOBVなので高いスコアが期待される
        self.assertGreater(score, 50)

    def test_action_conversion(self):
        """アクション変換テスト"""
        from signal_guidance_backtest import SignalGuidanceBacktestEnv

        env = SignalGuidanceBacktestEnv(self.mock_df, self.config)

        # 高スコア（売りシグナル）：強い上昇トレンド + 強い買いOBV + 買いBB
        observation = np.array([1.0, 1.0, 0.9])  # Supertrend=1.0, Direction=1.0, OBV=0.9
        # BB_Position = (Direction + 1.0) / 2.0 = (1.0 + 1.0) / 2.0 = 1.0 → 20点
        # Supertrend = 75点, OBV = 75点
        # Total = 20*0.4 + 75*0.4 + 75*0.2 = 8 + 30 + 15 = 53点 (HOLD)
        action, score = env._get_signal_guidance_score(observation, 0.0)
        self.assertTrue(25 < score < 75)  # 中間スコアでHOLD
        self.assertEqual(action, 0)  # HOLD

        # より強い売りシグナル：強い上昇トレンド + 買いOBV + 上限BB
        # スコア53はHOLDになるが、より極端なシグナルでSELLになるはず
        observation = np.array([1.0, 1.0, 0.95])  # より強い買いシグナル
        # BB_Position = 1.0 → 20点, Supertrend = 75点, OBV = 75点
        # Total = 20*0.4 + 75*0.4 + 75*0.2 = 53点 (HOLD)
        action, score = env._get_signal_guidance_score(observation, 0.0)
        self.assertTrue(score < 75)  # 75未満
        self.assertEqual(action, 0)  # HOLD

        # 低スコア（売りシグナル）：下降トレンド + 強い売りOBV
        observation = np.array([1.0, -1.0, 0.1])  # Supertrend=1.0, Direction=-1.0, OBV=0.1
        action, score = env._get_signal_guidance_score(observation, 0.0)
        self.assertLessEqual(score, 25)  # 25以下でSELL
        self.assertEqual(action, -1)  # SELL

        # 中間スコア（ホールド）：中立トレンド + 中立OBV
        observation = np.array([1.0, 0.0, 0.5])  # Supertrend=1.0, Direction=0.0, OBV=0.5
        action, score = env._get_signal_guidance_score(observation, 0.0)
        self.assertTrue(25 < score < 75)  # 25-75の範囲でHOLD
        self.assertEqual(action, 0)  # HOLD


class TestV4FeatureIntegration(unittest.TestCase):
    """V4FeatureExtractor統合テスト"""

    def test_v4_feature_mapping(self):
        """V4特徴量マッピングテスト"""
        from signal_guidance_backtest import SignalGuidanceBacktestEnv

        # V4FeatureExtractorの実際の特徴量を確認
        try:
            from ztb.features.unified_feature import UnifiedFeatureEngineer
            feature_extractor = UnifiedFeatureEngineer()

            # SACモデルの特徴量を取得
            available_features = feature_extractor.get_available_features(model_type="sac")
            print(f"V4 Available features: {available_features}")

            # Supertrend, Supertrend_Direction, OBVが含まれていることを確認
            self.assertIn('Supertrend', available_features)
            self.assertIn('Supertrend_Direction', available_features)
            self.assertIn('OBV', available_features)

        except ImportError:
            self.skipTest("V4FeatureExtractor not available")

    def test_feature_extraction_consistency(self):
        """特徴量抽出の一貫性テスト"""
        # 同じ入力で同じ特徴量が抽出されることを確認
        from signal_guidance_backtest import SignalGuidanceBacktestEnv

        env = SignalGuidanceBacktestEnv(pd.DataFrame(), {})

        obs1 = np.array([1.0, 1.0, 0.7])
        obs2 = np.array([1.0, 1.0, 0.7])

        signals1 = env._extract_technical_signals(obs1)
        signals2 = env._extract_technical_signals(obs2)

        self.assertEqual(signals1, signals2)


if __name__ == '__main__':
    # 詳細なテスト出力
    unittest.main(verbosity=2)