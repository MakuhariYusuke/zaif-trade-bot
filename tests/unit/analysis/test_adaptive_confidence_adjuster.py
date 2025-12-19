"""
適応型信頼度調整の単体テスト

AdaptiveConfidenceAdjusterクラスの機能をテストします。
"""

import pytest
import pandas as pd
import numpy as np

from ztb.analysis.adaptive_confidence_adjuster import (
    AdaptiveConfidenceAdjuster, ConfidenceThresholds, MarketRegime,
    AdaptiveThresholdDecision, MarketRegimeDetector
)


class TestAdaptiveConfidenceAdjuster:
    """AdaptiveConfidenceAdjusterのテスト"""

    @pytest.fixture
    def sample_market_data(self):
        """サンプル市場データ"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)

        # トレンドデータ生成
        trend = np.linspace(0, 20, 100)  # 上昇トレンド
        noise = np.random.randn(100) * 0.5  # ノイズを減らす
        prices = 100 + trend + noise

        data = pd.DataFrame({
            'open': prices,
            'high': prices + np.abs(np.random.randn(100)),
            'low': prices - np.abs(np.random.randn(100)),
            'close': prices
        }, index=dates)

        return data

    @pytest.fixture
    def sample_performance_data(self):
        """サンプルパフォーマンスデータ"""
        return [
            {'pnl': 100, 'confidence': 0.8, 'success': True},
            {'pnl': -50, 'confidence': 0.6, 'success': False},
            {'pnl': 150, 'confidence': 0.9, 'success': True},
            {'pnl': -30, 'confidence': 0.7, 'success': False},
            {'pnl': 200, 'confidence': 0.85, 'success': True},
        ]

    @pytest.fixture
    def confidence_adjuster(self):
        """AdaptiveConfidenceAdjusterインスタンス"""
        return AdaptiveConfidenceAdjuster()

    def test_initialization(self):
        """初期化テスト"""
        adjuster = AdaptiveConfidenceAdjuster()
        assert adjuster.thresholds is not None
        assert adjuster.regime_detector is not None
        assert adjuster.performance_history == []
        assert adjuster.threshold_history == []

    def test_initialization_with_custom_thresholds(self):
        """カスタム閾値での初期化テスト"""
        custom_thresholds = ConfidenceThresholds(
            base_threshold=0.8,
            min_threshold=0.6,
            max_threshold=0.95
        )
        adjuster = AdaptiveConfidenceAdjuster(custom_thresholds)
        assert adjuster.thresholds.base_threshold == 0.8
        assert adjuster.thresholds.min_threshold == 0.6
        assert adjuster.thresholds.max_threshold == 0.95

    def test_calculate_adaptive_threshold(self, confidence_adjuster, sample_market_data, sample_performance_data):
        """適応型閾値計算テスト"""
        decision = confidence_adjuster.calculate_adaptive_threshold(
            data=sample_market_data,
            recent_performance=sample_performance_data
        )

        assert isinstance(decision, AdaptiveThresholdDecision)
        assert hasattr(decision, 'current_threshold')
        assert hasattr(decision, 'market_regime')
        assert hasattr(decision, 'final_threshold')
        assert hasattr(decision, 'confidence_score')

        # 閾値が有効範囲内
        assert 0.5 <= decision.final_threshold <= 0.9

    def test_calculate_adaptive_threshold_no_performance(self, confidence_adjuster, sample_market_data):
        """パフォーマンスデータなしでの閾値計算テスト"""
        decision = confidence_adjuster.calculate_adaptive_threshold(
            data=sample_market_data,
            recent_performance=None
        )

        assert isinstance(decision, AdaptiveThresholdDecision)
        # パフォーマンス調整なしでも動作する
        assert decision.final_threshold > 0

    def test_regime_based_thresholds(self, confidence_adjuster, sample_market_data):
        """レジームベースの閾値テスト"""
        # 強気トレンドデータ
        bull_data = sample_market_data.copy()
        bull_data['close'] = bull_data['close'] * 1.1  # 上昇傾向を強める
        bull_data['high'] = bull_data['high'] * 1.1
        bull_data['low'] = bull_data['low'] * 1.1
        bull_data['open'] = bull_data['open'] * 1.1

        decision = confidence_adjuster.calculate_adaptive_threshold(bull_data)

        # 検出されたレジームに基づく基本閾値が正しく取得されていることを確認
        expected_base_threshold = confidence_adjuster.thresholds.get_threshold_for_regime(decision.market_regime)
        assert decision.base_threshold == expected_base_threshold

    def test_performance_adjustment(self, confidence_adjuster, sample_market_data):
        """パフォーマンス調整テスト"""
        # 良いパフォーマンス
        good_performance = [
            {'pnl': 100, 'confidence': 0.8, 'success': True},
            {'pnl': 150, 'confidence': 0.9, 'success': True},
            {'pnl': 200, 'confidence': 0.85, 'success': True},
        ]

        # 悪いパフォーマンス
        bad_performance = [
            {'pnl': -100, 'confidence': 0.8, 'success': False},
            {'pnl': -150, 'confidence': 0.9, 'success': False},
            {'pnl': -200, 'confidence': 0.85, 'success': False},
        ]

        good_decision = confidence_adjuster.calculate_adaptive_threshold(
            sample_market_data, good_performance
        )

        bad_decision = confidence_adjuster.calculate_adaptive_threshold(
            sample_market_data, bad_performance
        )

        # 良いパフォーマンス時は閾値が低くなるはず（自信を持ってエントリー）
        # 悪いパフォーマンス時は閾値が高くなるはず（慎重にエントリー）
        assert good_decision.final_threshold <= bad_decision.final_threshold

    def test_volatility_adjustment(self, confidence_adjuster):
        """ボラティリティ調整テスト"""
        # 固定シードで再現性確保
        np.random.seed(42)

        # 低ボラティリティデータ
        low_vol_data = pd.DataFrame({
            'open': [100] * 50,
            'high': [100.1] * 50,
            'low': [99.9] * 50,
            'close': [100] * 50
        })

        # 高ボラティリティデータ
        high_vol_data = pd.DataFrame({
            'open': np.random.uniform(95, 105, 50),
            'high': np.random.uniform(105, 115, 50),
            'low': np.random.uniform(85, 95, 50),
            'close': np.random.uniform(95, 105, 50)
        })

        low_vol_decision = confidence_adjuster.calculate_adaptive_threshold(low_vol_data)
        high_vol_decision = confidence_adjuster.calculate_adaptive_threshold(high_vol_data)

        # 高ボラティリティ時は閾値が高くなるはず
        assert high_vol_decision.final_threshold >= low_vol_decision.final_threshold

    def test_threshold_bounds(self, confidence_adjuster, sample_market_data):
        """閾値境界テスト"""
        # 極端なパフォーマンスデータで境界チェック
        extreme_performance = [
            {'pnl': 1000, 'confidence': 0.95, 'success': True},
            {'pnl': 2000, 'confidence': 0.98, 'success': True},
        ] * 10  # 多数の成功データ

        decision = confidence_adjuster.calculate_adaptive_threshold(
            sample_market_data, extreme_performance
        )

        # 最小・最大閾値内に収まる
        assert confidence_adjuster.thresholds.min_threshold <= decision.final_threshold <= confidence_adjuster.thresholds.max_threshold

    def test_confidence_score_calculation(self, confidence_adjuster, sample_market_data, sample_performance_data):
        """信頼度スコア計算テスト"""
        decision = confidence_adjuster.calculate_adaptive_threshold(
            sample_market_data, sample_performance_data
        )

        assert 0 <= decision.confidence_score <= 1
        assert isinstance(decision.reasoning, str)
        assert len(decision.reasoning) > 0

    def test_history_tracking(self, confidence_adjuster, sample_market_data):
        """履歴追跡テスト"""
        initial_history_length = len(confidence_adjuster.performance_history)

        decision = confidence_adjuster.calculate_adaptive_threshold(sample_market_data)

        # 履歴が更新される
        assert len(confidence_adjuster.threshold_history) > 0
        assert confidence_adjuster.threshold_history[-1] == decision.final_threshold


class TestMarketRegimeDetector:
    """MarketRegimeDetectorのテスト"""

    @pytest.fixture
    def regime_detector(self):
        """MarketRegimeDetectorインスタンス"""
        return MarketRegimeDetector()

    @pytest.fixture
    def trend_data(self):
        """トレンドデータ"""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        # 明確な上昇トレンド
        prices = 100 + np.arange(50) * 0.5

        data = pd.DataFrame({
            'open': prices,
            'high': prices + 0.5,
            'low': prices - 0.5,
            'close': prices
        }, index=dates)

        return data

    @pytest.fixture
    def sideways_data(self):
        """レンジデータ"""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        # 完全に横ばい
        prices = np.full(50, 100.0)

        data = pd.DataFrame({
            'open': prices,
            'high': prices + 0.5,
            'low': prices - 0.5,
            'close': prices
        }, index=dates)

        return data

    def test_detect_regime_trend(self, regime_detector, trend_data):
        """トレンド検出テスト"""
        regime = regime_detector.detect_regime(trend_data)

        assert isinstance(regime, MarketRegime)
        # 上昇トレンドなのでSTRONG_BULLまたはMODERATE_BULLが検出されるはず
        assert regime in [MarketRegime.STRONG_BULL, MarketRegime.MODERATE_BULL]

    def test_detect_regime_sideways(self, regime_detector, sideways_data):
        """レンジ検出テスト"""
        regime = regime_detector.detect_regime(sideways_data)

        assert isinstance(regime, MarketRegime)
        # 横ばいなのでTIGHT_RANGEまたはQUIET_RANGEが検出されるはず
        assert regime in [MarketRegime.TIGHT_RANGE, MarketRegime.QUIET_RANGE]

    def test_detect_regime_insufficient_data(self, regime_detector):
        """データ不足時のレジーム検出テスト"""
        short_data = pd.DataFrame({
            'open': [100, 101],
            'high': [101, 102],
            'low': [99, 100],
            'close': [100, 101]
        })

        regime = regime_detector.detect_regime(short_data)

        # データ不足時はQUIET_RANGEが返される
        assert regime == MarketRegime.QUIET_RANGE

    def test_trend_strength_calculation(self, regime_detector, trend_data):
        """トレンド強度計算テスト"""
        strength = regime_detector._calculate_trend_strength(trend_data)

        assert -5 <= strength <= 5
        # 上昇トレンドなので正の値
        assert strength > 0

    def test_volatility_calculation(self, regime_detector, trend_data):
        """ボラティリティ計算テスト"""
        volatility = regime_detector._calculate_volatility(trend_data)

        assert 0 <= volatility <= 1

    def test_breakout_detection(self, regime_detector, trend_data):
        """ブレイクアウト検出テスト"""
        is_breakout = regime_detector._detect_breakout(trend_data)

        assert isinstance(is_breakout, bool)


class TestConfidenceThresholds:
    """ConfidenceThresholdsのテスト"""

    def test_initialization(self):
        """初期化テスト"""
        thresholds = ConfidenceThresholds()
        assert thresholds.base_threshold == 0.7
        assert thresholds.min_threshold == 0.5
        assert thresholds.max_threshold == 0.9

    def test_get_threshold_for_regime(self):
        """レジーム別閾値取得テスト"""
        thresholds = ConfidenceThresholds()

        # 各レジームの閾値を取得
        strong_bull_threshold = thresholds.get_threshold_for_regime(MarketRegime.STRONG_BULL)
        assert strong_bull_threshold == thresholds.strong_bull_threshold

        moderate_bull_threshold = thresholds.get_threshold_for_regime(MarketRegime.MODERATE_BULL)
        assert moderate_bull_threshold == thresholds.moderate_bull_threshold

        strong_bear_threshold = thresholds.get_threshold_for_regime(MarketRegime.STRONG_BEAR)
        assert strong_bear_threshold == thresholds.strong_bear_threshold

        tight_range_threshold = thresholds.get_threshold_for_regime(MarketRegime.TIGHT_RANGE)
        assert tight_range_threshold == thresholds.tight_range_threshold

        # BREAKOUT_UPレジームの閾値を取得
        breakout_up_threshold = thresholds.get_threshold_for_regime(MarketRegime.BREAKOUT_UP)
        assert breakout_up_threshold == thresholds.breakout_up_threshold

    def test_to_dict(self):
        """辞書変換テスト"""
        thresholds = ConfidenceThresholds()
        threshold_dict = thresholds.to_dict()

        assert isinstance(threshold_dict, dict)
        assert 'base_threshold' in threshold_dict
        assert 'min_threshold' in threshold_dict
        assert 'max_threshold' in threshold_dict