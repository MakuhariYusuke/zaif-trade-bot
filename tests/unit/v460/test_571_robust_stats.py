"""571# テスト: RobustStats ユーティリティ + 執行品質比較セクション."""

from __future__ import annotations

import numpy as np
import pytest

from ztb.utils.robust_stats import RobustStats


# ══════════════════════════════════════════════════════════════════════
# 1. clip_outliers_mad
# ══════════════════════════════════════════════════════════════════════


class TestClipOutliersMad:
    """MAD ベース外れ値クリッピングの検証."""

    def test_empty_array(self) -> None:
        result = RobustStats.clip_outliers_mad(np.array([]))
        assert len(result) == 0

    def test_uniform_data_no_clip(self) -> None:
        """全値同一 → MAD=0 → そのまま返す."""
        data = np.array([5.0, 5.0, 5.0, 5.0])
        result = RobustStats.clip_outliers_mad(data)
        np.testing.assert_array_equal(result, data)

    def test_normal_data_no_outlier(self) -> None:
        """外れ値なしデータ → 変化なし."""
        data = np.array([10.0, 11.0, 10.5, 10.2, 10.8])
        result = RobustStats.clip_outliers_mad(data)
        np.testing.assert_array_equal(result, data)

    def test_outlier_clipped(self) -> None:
        """極端な外れ値がクリップされる."""
        data = np.array([8.0, 9.0, 10.0, 11.0, 12.0, 100.0])
        result = RobustStats.clip_outliers_mad(data, threshold=3.0)
        assert result[-1] < 100.0
        assert result[-1] == result.max()

    def test_threshold_controls_sensitivity(self) -> None:
        """threshold が小さいほど積極的にクリップ."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 20.0])
        strict = RobustStats.clip_outliers_mad(data, threshold=1.0)
        loose = RobustStats.clip_outliers_mad(data, threshold=5.0)
        assert strict[-1] <= loose[-1]


# ══════════════════════════════════════════════════════════════════════
# 2. robust_ema
# ══════════════════════════════════════════════════════════════════════


class TestRobustEma:
    """スパイク保護付き EMA の検証."""

    def test_normal_update(self) -> None:
        """通常の EMA 更新."""
        result = RobustStats.robust_ema(10.0, 9.0, alpha=0.3)
        expected = 0.3 * 10.0 + 0.7 * 9.0
        assert result == pytest.approx(expected)

    def test_spike_clipped(self) -> None:
        """sigma_limit 超過スパイクのクリッピング."""
        result = RobustStats.robust_ema(100.0, 10.0, alpha=0.3, sigma_limit=5.0)
        # clipped_val = 10.0 + 5.0 = 15.0
        expected = 0.3 * 15.0 + 0.7 * 10.0
        assert result == pytest.approx(expected)

    def test_negative_spike_clipped(self) -> None:
        """下方スパイクも同等にクリップ."""
        result = RobustStats.robust_ema(-100.0, 10.0, alpha=0.3, sigma_limit=5.0)
        # clipped_val = 10.0 - 5.0 = 5.0
        expected = 0.3 * 5.0 + 0.7 * 10.0
        assert result == pytest.approx(expected)

    def test_no_sigma_limit(self) -> None:
        """sigma_limit=None はクリッピングなし."""
        result = RobustStats.robust_ema(100.0, 10.0, alpha=0.3, sigma_limit=None)
        expected = 0.3 * 100.0 + 0.7 * 10.0
        assert result == pytest.approx(expected)

    def test_within_sigma_limit_no_clip(self) -> None:
        """sigma_limit 内の値はクリップされない."""
        result = RobustStats.robust_ema(12.0, 10.0, alpha=0.3, sigma_limit=5.0)
        expected = 0.3 * 12.0 + 0.7 * 10.0
        assert result == pytest.approx(expected)


# ══════════════════════════════════════════════════════════════════════
# 3. asymmetric_ema
# ══════════════════════════════════════════════════════════════════════


class TestAsymmetricEma:
    """非対称 EMA の検証."""

    def test_up_uses_alpha_up(self) -> None:
        """上昇時は alpha_up を使用."""
        result = RobustStats.asymmetric_ema(15.0, 10.0, alpha_up=0.1, alpha_down=0.5)
        expected = 0.1 * 15.0 + 0.9 * 10.0
        assert result == pytest.approx(expected)

    def test_down_uses_alpha_down(self) -> None:
        """下降時は alpha_down を使用."""
        result = RobustStats.asymmetric_ema(5.0, 10.0, alpha_up=0.1, alpha_down=0.5)
        expected = 0.5 * 5.0 + 0.5 * 10.0
        assert result == pytest.approx(expected)

    def test_equal_uses_alpha_down(self) -> None:
        """等値は alpha_down (<=)."""
        result = RobustStats.asymmetric_ema(10.0, 10.0, alpha_up=0.1, alpha_down=0.5)
        expected = 0.5 * 10.0 + 0.5 * 10.0
        assert result == pytest.approx(10.0)


# ══════════════════════════════════════════════════════════════════════
# 4. median_filter_fast
# ══════════════════════════════════════════════════════════════════════


class TestMedianFilterFast:
    """中央値フィルタの検証."""

    def test_odd_length(self) -> None:
        result = RobustStats.median_filter_fast(np.array([1.0, 3.0, 2.0]))
        assert result == pytest.approx(2.0)

    def test_even_length(self) -> None:
        result = RobustStats.median_filter_fast(np.array([1.0, 3.0, 2.0, 4.0]))
        assert result == pytest.approx(2.5)

    def test_single_element(self) -> None:
        result = RobustStats.median_filter_fast(np.array([42.0]))
        assert result == pytest.approx(42.0)

    def test_returns_float(self) -> None:
        result = RobustStats.median_filter_fast(np.array([1, 2, 3]))
        assert isinstance(result, float)


# ══════════════════════════════════════════════════════════════════════
# 5. section_execution_quality_comparison (analyze_fill_logs)
# ══════════════════════════════════════════════════════════════════════


class TestSectionExecutionQualityComparison:
    """571# 執行品質比較セクションの検証."""

    @pytest.fixture()
    def _import_section(self):
        from scripts.v460.analysis.analyze_fill_logs import (
            section_execution_quality_comparison,
        )
        return section_execution_quality_comparison

    def test_no_fills_returns_no_fills_message(self, _import_section) -> None:
        fn = _import_section
        result = fn([{"filled": False}])
        assert any("no fills" in line for line in result)

    def test_multiplicative_only(self, _import_section) -> None:
        """additive 未使用 → multiplicative のみ表示."""
        fn = _import_section
        records = [
            {
                "filled": True,
                "spread_capture_bps": 1.5,
                "adverse_selection_cost_bps": -0.5,
            },
            {
                "filled": True,
                "spread_capture_bps": 2.0,
                "adverse_selection_cost_bps": -1.0,
            },
        ]
        lines = fn(records)
        text = "\n".join(lines)
        assert "MULTIPLICATIVE" in text
        assert "Spread Capture" in text
        assert "ADDITIVE" not in text

    def test_additive_group(self, _import_section) -> None:
        """execution_additive_enabled=True → ADDITIVE に分類."""
        fn = _import_section
        records = [
            {
                "filled": True,
                "execution_additive_enabled": True,
                "spread_capture_bps": 3.0,
                "adverse_selection_cost_bps": -0.2,
            },
        ]
        lines = fn(records)
        text = "\n".join(lines)
        assert "ADDITIVE" in text

    def test_icr_calculation(self, _import_section) -> None:
        """ICR = spread_capture / |AS cost|."""
        fn = _import_section
        records = [
            {
                "filled": True,
                "spread_capture_bps": 2.0,
                "adverse_selection_cost_bps": -1.0,
            },
        ]
        lines = fn(records)
        text = "\n".join(lines)
        # ICR = 2.0 / 1.0 = 2.000
        assert "2.000" in text


# ══════════════════════════════════════════════════════════════════════
# 6. 574# get_robust_inputs 統合テスト
# ══════════════════════════════════════════════════════════════════════


class TestGetRobustInputs:
    """574# Task B: maker_price.get_robust_inputs() の検証."""

    def _make_maker_price(self):
        """最小限のMakerPriceCalculator もどき."""
        from collections import deque
        from unittest.mock import MagicMock

        from scripts.v460.lib.fill_config import FillTestConfig

        mp = MagicMock()
        mp._last_sigma = 0.0002  # 2 bps
        mp._robust_sigma = 0.0
        mp._ofi_history = deque(maxlen=50)
        mp.get_adverse_ofi = MagicMock(return_value=0.1)

        # bind the real method
        from scripts.v460.lib.maker_price import MakerPriceCalculator
        mp.get_robust_inputs = MakerPriceCalculator.get_robust_inputs.__get__(mp)
        return mp

    def test_sigma_uses_asymmetric_ema(self) -> None:
        """ロバスト σ は asymmetric_ema で平滑化される."""
        mp = self._make_maker_price()
        sigma, ofi = mp.get_robust_inputs("buy")
        # _robust_sigma was 0, _last_sigma is 0.0002
        # asymmetric_ema(0.0002, 0.0, alpha_up=0.20) → 0.20 * 0.0002 = 0.00004
        assert sigma == pytest.approx(0.00004)

    def test_ofi_falls_back_when_insufficient(self) -> None:
        """OFI 履歴不足時は get_adverse_ofi にフォールバック."""
        mp = self._make_maker_price()
        for i in range(5):
            mp._ofi_history.append(0.1)
        _, ofi = mp.get_robust_inputs("buy")
        # < 10 samples → fallback to get_adverse_ofi
        assert ofi == 0.1

    def test_ofi_uses_median_filter_with_enough_data(self) -> None:
        """10+ サイクルのデータがあれば中央値フィルタ."""
        mp = self._make_maker_price()
        # buy 用: adverse = -ofi_med → max(0, adverse)
        # 10サイクル分 positive values → median positive → adverse for buy = -median → 0
        for i in range(15):
            mp._ofi_history.append(0.2)
        _, ofi = mp.get_robust_inputs("buy")
        # median(all 0.2) = 0.2, adverse for buy = -0.2 → max(0, -0.2) = 0.0
        assert ofi == pytest.approx(0.0)

    def test_ofi_sell_adverse(self) -> None:
        """sell 時は正の OFI が adverse."""
        mp = self._make_maker_price()
        for i in range(12):
            mp._ofi_history.append(0.3)
        _, ofi = mp.get_robust_inputs("sell")
        # median = 0.3, adverse for sell = 0.3
        assert ofi == pytest.approx(0.3)
