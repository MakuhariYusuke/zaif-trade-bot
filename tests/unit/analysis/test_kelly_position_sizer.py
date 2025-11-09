"""
Kelly基準ポジションサイザーの単体テスト

KellyPositionSizerクラスの機能をテストします。
"""

import pytest
import numpy as np
from unittest.mock import Mock

from ztb.analysis.kelly_position_sizer import (
    KellyPositionSizer, KellyParameters
)


class TestKellyPositionSizer:
    """KellyPositionSizerのテスト"""

    @pytest.fixture
    def sample_trades(self):
        """サンプルトレードデータ"""
        return [
            {'pnl': 100, 'confidence': 0.8, 'entry_price': 100},
            {'pnl': -50, 'confidence': 0.6, 'entry_price': 105},
            {'pnl': 150, 'confidence': 0.9, 'entry_price': 102},
            {'pnl': -30, 'confidence': 0.7, 'entry_price': 108},
            {'pnl': 200, 'confidence': 0.85, 'entry_price': 110},
            {'pnl': 80, 'confidence': 0.75, 'entry_price': 115},
            {'pnl': -70, 'confidence': 0.65, 'entry_price': 118},
            {'pnl': 120, 'confidence': 0.82, 'entry_price': 120},
            {'pnl': -40, 'confidence': 0.68, 'entry_price': 122},
            {'pnl': 180, 'confidence': 0.88, 'entry_price': 125},
        ]

    @pytest.fixture
    def kelly_sizer(self):
        """KellyPositionSizerインスタンス"""
        return KellyPositionSizer()

    def test_initialization(self):
        """初期化テスト"""
        sizer = KellyPositionSizer()
        assert sizer is not None

    def test_calculate_kelly_parameters(self, kelly_sizer, sample_trades):
        """Kellyパラメータ計算テスト"""
        params = kelly_sizer.calculate_kelly_parameters(sample_trades, 10000)

        assert hasattr(params, 'kelly_fraction')
        assert hasattr(params, 'win_rate')
        assert hasattr(params, 'win_loss_ratio')
        assert hasattr(params, 'total_trades')

        # 値の妥当性チェック
        assert 0 <= params.kelly_fraction <= 1.0
        assert params.total_trades == len(sample_trades)

    def test_calculate_kelly_parameters_insufficient_trades(self, kelly_sizer):
        """トレード数が不足する場合のテスト"""
        insufficient_trades = [
            {'pnl': 100, 'confidence': 0.8, 'entry_price': 100},
            {'pnl': -50, 'confidence': 0.6, 'entry_price': 105},
        ]

        params = kelly_sizer.calculate_kelly_parameters(insufficient_trades, 10000)

        assert params is None

    def test_calculate_dynamic_position_size(self, kelly_sizer, sample_trades):
        """動的ポジションサイズ計算テスト"""
        decision = kelly_sizer.calculate_dynamic_position_size(
            sample_trades, 10000, "half"
        )

        assert hasattr(decision, 'position_size_fraction')
        assert hasattr(decision, 'risk_amount')
        assert hasattr(decision, 'confidence_score')
        assert hasattr(decision, 'reasoning')

        assert 0 <= decision.position_size_fraction <= 1.0
        assert decision.risk_amount > 0

    def test_calculate_dynamic_position_size_insufficient_trades(self, kelly_sizer):
        """トレード数が不足する場合の動的ポジションサイズ計算テスト"""
        insufficient_trades = [
            {'pnl': 100, 'confidence': 0.8, 'entry_price': 100},
        ]

        decision = kelly_sizer.calculate_dynamic_position_size(
            insufficient_trades, 10000, "half"
        )

        # デフォルト値が返される
        assert decision.position_size_fraction == 0.01  # デフォルト値（max_risk_fractionの半分）

    def test_calculate_dynamic_position_size_zero_volatility(self, kelly_sizer):
        """ボラティリティがゼロの場合のテスト"""
        zero_vol_trades = [
            {'pnl': 100, 'confidence': 0.8, 'entry_price': 100},
            {'pnl': 100, 'confidence': 0.8, 'entry_price': 100},
            {'pnl': 100, 'confidence': 0.8, 'entry_price': 100},
        ]

        decision = kelly_sizer.calculate_dynamic_position_size(
            zero_vol_trades, 10000, "half"
        )

        # ボラティリティがゼロでも動作する
        assert decision.position_size_fraction >= 0

    def test_risk_tolerance_application(self, kelly_sizer, sample_trades):
        """リスク許容度の適用テスト"""
        # フルKelly
        full_decision = kelly_sizer.calculate_dynamic_position_size(
            sample_trades, 10000, "full"
        )

        # ハーフKelly
        half_decision = kelly_sizer.calculate_dynamic_position_size(
            sample_trades, 10000, "half"
        )

        # クォーターKelly
        quarter_decision = kelly_sizer.calculate_dynamic_position_size(
            sample_trades, 10000, "quarter"
        )

        # ハーフはフルより小さいはず
        assert half_decision.position_size_fraction <= full_decision.position_size_fraction
        # クォーターはハーフより小さいはず
        assert quarter_decision.position_size_fraction <= half_decision.position_size_fraction

    def test_confidence_adjustment(self, kelly_sizer, sample_trades):
        """信頼度調整テスト"""
        # 高信頼度のトレードのみ
        high_confidence_trades = [
            trade for trade in sample_trades if trade['confidence'] >= 0.8
        ]

        # 低信頼度のトレードのみ
        low_confidence_trades = [
            trade for trade in sample_trades if trade['confidence'] < 0.8
        ]

        if len(high_confidence_trades) >= 3:
            high_decision = kelly_sizer.calculate_dynamic_position_size(
                high_confidence_trades, 10000, "half"
            )

            low_decision = kelly_sizer.calculate_dynamic_position_size(
                low_confidence_trades, 10000, "half"
            )

            # 高信頼度のトレードの方が大きなポジションサイズになるはず
            assert high_decision.position_size_fraction >= low_decision.position_size_fraction