"""
Phase 3-1: ActionSignalGuideAdapter統合テスト

ActionSignalGuideAdapterのシグナル品質向上統合機能をテストします。
"""


import numpy as np
import pandas as pd
import pytest

from ztb.trading.backtest.adapters import ActionSignalGuideAdapter


class TestActionSignalGuideAdapterIntegration:
    """ActionSignalGuideAdapter統合テスト"""

    @pytest.fixture
    def adapter(self):
        """テスト用のアダプター"""
        # 設定を最小限にして高速化
        config = {
            "debug_short_mode": True,
            "short_mode_recognizer_limit": 3,  # さらに制限
            "enable_candlestick_patterns": True,
            "enable_oscillator_patterns": True,
            "enable_bollinger_patterns": True,
            "enable_fibonacci_patterns": False,
            "enable_gann_patterns": False,
            "enable_wave_patterns": False,
            "enable_harmonic_patterns": False,
            "enable_volume_patterns": False,
            "enable_adx_patterns": False,
            "enable_granville_patterns": False,
            "enable_heikin_ashi_patterns": False,
            "enable_dow_theory_patterns": False,
        }
        return ActionSignalGuideAdapter(config)

    @pytest.fixture
    def sample_market_data(self):
        """テスト用の市場データ"""
        dates = pd.date_range("2023-01-01", periods=100, freq="1h")
        np.random.seed(42)

        # トレンドのあるデータを作成
        trend = np.linspace(100, 120, 100)
        noise = np.random.normal(0, 2, 100)
        close_prices = trend + noise

        return pd.DataFrame(
            {
                "open": close_prices - np.random.uniform(1, 3, 100),
                "high": close_prices + np.random.uniform(1, 3, 100),
                "low": close_prices - np.random.uniform(1, 3, 100),
                "close": close_prices,
                "volume": np.random.uniform(1000, 2000, 100),
            },
            index=dates,
        )

    def test_initialization_with_quality_filter(self, adapter):
        """品質フィルタ付き初期化テスト"""
        assert adapter.integrated_filter is not None
        assert hasattr(adapter, "signal_cache")
        assert hasattr(adapter, "volatility_cache")
        assert hasattr(adapter, "thresholds_cache")

    def test_generate_signal_with_quality_filter(self, adapter, sample_market_data):
        """品質フィルタ付きシグナル生成テスト"""
        # 適度な長さのデータを使用
        test_data = sample_market_data.tail(50)

        signal = adapter.generate_signal(test_data, 0)

        assert isinstance(signal, dict)
        assert "action" in signal

        # 品質フィルタ関連のフィールドが追加されているはず
        quality_fields = ["quality_score", "quality_level", "filter_passed"]
        quality_present = any(field in signal for field in quality_fields)

        if signal["action"] != "hold":
            # シグナルが生成された場合、品質情報が含まれるはず
            assert any(
                field in signal for field in quality_fields
            ), f"Quality fields missing from signal: {signal}"

    def test_quality_filter_integration(self, adapter, sample_market_data):
        """品質フィルタ統合テスト"""
        test_data = sample_market_data.tail(50)

        # 複数のシグナルを生成して品質フィルタが機能するかテスト
        signals_with_quality = []
        signals_without_quality = []

        for i in range(min(10, len(test_data))):
            signal = adapter.generate_signal(test_data, i)

            if signal["action"] != "hold":
                # 品質関連フィールドがあるかチェック
                has_quality_info = any(
                    key.startswith("quality_") or key == "filter_passed"
                    for key in signal.keys()
                )

                if has_quality_info:
                    signals_with_quality.append(signal)
                else:
                    signals_without_quality.append(signal)

        # 少なくとも一部のシグナルに品質情報が含まれているはず
        total_signals = len(signals_with_quality) + len(signals_without_quality)
        if total_signals > 0:
            quality_info_ratio = len(signals_with_quality) / total_signals
            assert (
                quality_info_ratio > 0.5
            ), f"Only {quality_info_ratio:.2%} of signals have quality info"

    def test_risk_management_with_quality_filter(self, adapter, sample_market_data):
        """リスク管理と品質フィルタ統合テスト"""
        test_data = sample_market_data.tail(50)

        # ポジションをシミュレート
        signal = adapter.generate_signal(test_data, 0)

        if signal["action"] != "hold":
            # ポジションオープン
            position_id = adapter.open_position(
                position_type="long" if signal["action"] == "buy" else "short",
                entry_price=test_data["close"].iloc[-1],
                position_size=0.1,
                stop_loss=test_data["close"].iloc[-1] * 0.95,
                take_profit=test_data["close"].iloc[-1] * 1.05,
                current_time=test_data.index[-1],
                signal_data=signal,
            )

            assert position_id is not None
            assert position_id in adapter.active_positions

            # ポジションクローズをテスト
            closed_positions = adapter.update_positions(
                current_price=test_data["close"].iloc[-1] * 1.03,  # 利益確定
                current_time=test_data.index[-1] + pd.Timedelta(hours=1),
            )

            if closed_positions:
                assert len(closed_positions) == 1
                assert closed_positions[0]["position_id"] == position_id

    def test_cache_system_integration(self, adapter, sample_market_data):
        """キャッシュシステム統合テスト"""
        test_data = sample_market_data.tail(50)

        # 同じデータを複数回処理してキャッシュが機能するかテスト
        signal1 = adapter.generate_signal(test_data, 0)
        signal2 = adapter.generate_signal(test_data, 0)  # 同じインデックス

        # キャッシュが効いているはず（同じ結果が返される）
        assert signal1["action"] == signal2["action"]

    def test_error_handling_with_quality_filter(self, adapter):
        """品質フィルタ付きエラーハンドリングテスト"""
        # 不十分なデータでテスト
        insufficient_data = pd.DataFrame({"close": [100, 101, 102]})

        signal = adapter.generate_signal(insufficient_data, 0)

        # 不十分なデータではholdシグナルが返される（エラーハンドリング）
        assert signal["action"] == "hold"
        # エラーが発生しない場合でも正常動作

    def test_dynamic_thresholds_with_quality_filter(self, adapter, sample_market_data):
        """動的閾値と品質フィルタ統合テスト"""
        test_data = sample_market_data.tail(50)

        # 動的閾値を計算
        thresholds = adapter._calculate_dynamic_thresholds(test_data)

        assert "confidence_threshold" in thresholds
        assert "signal_strength_threshold" in thresholds
        assert 0 < thresholds["confidence_threshold"] <= 1
        assert 0 < thresholds["signal_strength_threshold"] <= 1

    def test_volatility_calculation_with_cache(self, adapter, sample_market_data):
        """ボラティリティ計算とキャッシュ統合テスト"""
        test_data = sample_market_data.tail(50)

        # ボラティリティを計算
        volatility1 = adapter._calculate_market_volatility(test_data)
        volatility2 = adapter._calculate_market_volatility(test_data)  # キャッシュから

        assert volatility1 == volatility2  # キャッシュが効いている
        assert 0 <= volatility1 <= 1  # ボラティリティは0-1の範囲

    def test_signal_statistics_tracking(self, adapter, sample_market_data):
        """シグナル統計追跡テスト"""
        test_data = sample_market_data.tail(20)

        initial_stats = adapter.signal_stats.copy()

        # 複数のシグナルを生成
        for i in range(min(5, len(test_data))):
            adapter.generate_signal(test_data, i)

        final_stats = adapter.signal_stats

        # 統計が更新されているはず
        assert final_stats["total_signals"] >= initial_stats["total_signals"]

    @pytest.mark.parametrize("market_regime", ["bull", "bear", "sideways", "volatile"])
    def test_market_regime_adaptation(self, adapter, market_regime):
        """市場レジーム適応テスト"""
        # アダプタはバックテスト用に緩和基準 (min_quality=0.45) で初期化される。
        # update_market_regime は相対的に基準を調整するが、
        # floor/ceiling があるため direction は初期値依存で反転しうる。
        # ここでは「調整後も有効範囲内」であることを検証する。

        # 市場レジームを設定
        adapter.integrated_filter.update_market_regime(market_regime)

        # 基準が適応されているはず
        criteria = adapter.integrated_filter.filter_criteria

        # 全レジームで品質スコアは 0.0〜1.0 の有効範囲内
        assert 0.0 < criteria.min_quality_score <= 1.0
        assert 0.0 < criteria.min_confidence_score <= 1.0

        if market_regime == "bull":
            assert criteria.min_quality_score <= 0.6  # bull は上限 0.6 以下
        elif market_regime == "bear":
            assert criteria.min_quality_score >= 0.5  # bear は下限 0.5 以上
        elif market_regime == "volatile":
            assert criteria.min_quality_score >= 0.5  # volatile は下限 0.5 以上


if __name__ == "__main__":
    pytest.main([__file__])
