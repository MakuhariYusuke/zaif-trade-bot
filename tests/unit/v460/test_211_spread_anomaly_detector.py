"""211# P1-C: Spread Anomaly Detector テスト.

流動性枯渇 (spread 急拡大) の自動検知・防御メカニズムのユニットテスト。
"""

from __future__ import annotations

import pytest

from scripts.v460.lib.spread_anomaly_detector import (
    SADConfig,
    SADLevel,
    SADResult,
    SpreadAnomalyDetector,
)


# =====================================================================
# 基本動作テスト
# =====================================================================


class TestSpreadAnomalyDetectorBasic:
    """SAD の基本動作テスト."""

    def test_disabled_returns_normal(self) -> None:
        """enabled=False → 常に NORMAL."""
        sad = SpreadAnomalyDetector(SADConfig(enabled=False))
        sad.update(100.0, 1000.0)
        result = sad.check(1000.0)
        assert result.level == SADLevel.NORMAL

    def test_no_data_returns_normal(self) -> None:
        """データなし → NORMAL."""
        sad = SpreadAnomalyDetector(SADConfig())
        result = sad.check(1000.0)
        assert result.level == SADLevel.NORMAL

    def test_stable_spread_returns_normal(self) -> None:
        """安定した spread → NORMAL."""
        sad = SpreadAnomalyDetector(SADConfig(sample_interval_sec=1.0))
        for i in range(100):
            sad.update(50.0, float(i))  # constant spread
        result = sad.check(100.0)
        assert result.level == SADLevel.NORMAL
        assert result.spread_ratio == pytest.approx(1.0)


# =====================================================================
# 段階判定テスト
# =====================================================================


class TestSpreadAnomalyDetectorLevels:
    """spread_ratio による段階判定テスト."""

    @staticmethod
    def _build_sad_with_baseline(
        baseline_spread: float = 50.0,
        n_samples: int = 100,
    ) -> SpreadAnomalyDetector:
        """baseline を構築した SAD を返す."""
        sad = SpreadAnomalyDetector(SADConfig(
            sample_interval_sec=1.0,
            wide_ratio=2.0,
            dry_ratio=4.0,
            frozen_ratio=8.0,
        ))
        for i in range(n_samples):
            sad.update(baseline_spread, float(i))
        return sad

    def test_wide_level(self) -> None:
        """spread_ratio 2.0~4.0 → WIDE."""
        sad = self._build_sad_with_baseline(50.0)
        sad.update(120.0, 101.0)  # 120/50 = 2.4 → WIDE
        result = sad.check(101.0)
        assert result.level == SADLevel.WIDE
        assert result.spread_ratio >= 2.0

    def test_dry_level(self) -> None:
        """spread_ratio 4.0~8.0 → DRY."""
        sad = self._build_sad_with_baseline(50.0)
        sad.update(250.0, 101.0)  # 250/50 = 5.0 → DRY
        result = sad.check(101.0)
        assert result.level == SADLevel.DRY
        assert result.offset_mult == 3.0
        assert result.interval_mult == 2.0
        assert result.lot_mult == 0.5

    def test_frozen_level(self) -> None:
        """spread_ratio > 8.0 → FROZEN."""
        sad = self._build_sad_with_baseline(50.0)
        sad.update(500.0, 101.0)  # 500/50 = 10.0 → FROZEN
        result = sad.check(101.0)
        assert result.level == SADLevel.FROZEN
        assert result.frozen_remaining_sec > 0

    def test_wide_offset_mult(self) -> None:
        """WIDE 時の offset_mult = ratio * wide_offset_scale."""
        sad = SpreadAnomalyDetector(SADConfig(
            sample_interval_sec=1.0,
            wide_ratio=2.0,
            wide_offset_scale=0.5,
        ))
        for i in range(100):
            sad.update(50.0, float(i))
        sad.update(150.0, 100.0)  # ratio = 3.0
        result = sad.check(100.0)
        assert result.level == SADLevel.WIDE
        assert result.offset_mult == pytest.approx(1.5, abs=0.1)  # 3.0 * 0.5


# =====================================================================
# FROZEN クールダウンテスト
# =====================================================================


class TestSpreadAnomalyDetectorFrozen:
    """FROZEN クールダウンテスト."""

    def test_frozen_reevaluate(self) -> None:
        """FROZEN 後 reevaluate 期間中は FROZEN を返し続ける."""
        sad = SpreadAnomalyDetector(SADConfig(
            sample_interval_sec=1.0,
            frozen_ratio=8.0,
            frozen_reevaluate_sec=30.0,
        ))
        for i in range(100):
            sad.update(50.0, float(i))
        sad.update(500.0, 100.0)  # FROZEN

        r1 = sad.check(100.0)
        assert r1.level == SADLevel.FROZEN

        # 15秒後 — まだ FROZEN
        r2 = sad.check(115.0)
        assert r2.level == SADLevel.FROZEN
        assert r2.frozen_remaining_sec > 0


# =====================================================================
# 中央値計算テスト
# =====================================================================


class TestMedian:
    """_median のテスト."""

    def test_odd_count(self) -> None:
        assert SpreadAnomalyDetector._median([3.0, 1.0, 2.0]) == 2.0

    def test_even_count(self) -> None:
        assert SpreadAnomalyDetector._median([4.0, 1.0, 2.0, 3.0]) == 2.5

    def test_single(self) -> None:
        assert SpreadAnomalyDetector._median([5.0]) == 5.0


# =====================================================================
# export / import テスト
# =====================================================================


class TestSpreadAnomalyDetectorState:
    """状態永続化テスト."""

    def test_export_empty(self) -> None:
        sad = SpreadAnomalyDetector(SADConfig())
        state = sad.export_state()
        assert state["frozen_until"] == 0.0
        assert state["total_frozens"] == 0
        assert state["spread_buffer"] == []

    def test_roundtrip(self) -> None:
        sad = SpreadAnomalyDetector(SADConfig(sample_interval_sec=1.0))
        for i in range(50):
            sad.update(50.0 + i * 0.1, float(i))
        state = sad.export_state()

        sad2 = SpreadAnomalyDetector(SADConfig(sample_interval_sec=1.0))
        sad2.import_state(state)
        state2 = sad2.export_state()

        assert len(state["spread_buffer"]) == len(state2["spread_buffer"])


# =====================================================================
# FillTestConfig / cancel_reasons 統合テスト
# =====================================================================


class TestSADIntegration:
    """FillTestConfig / cancel_reasons との統合テスト."""

    def test_config_fields_exist(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert cfg.sad_enabled is False
        assert cfg.sad_wide_ratio == 2.0
        assert cfg.sad_dry_ratio == 4.0
        assert cfg.sad_frozen_ratio == 8.0

    def test_cancel_reasons_exist(self) -> None:
        from scripts.v460.lib import cancel_reasons as CR
        assert CR.SAD_FROZEN == "sad_frozen"
        assert CR.SAD_DRY == "sad_dry"
        assert CR.SAD_FROZEN in CR.AUDIT_CANCEL_REASONS
        assert CR.SAD_DRY in CR.AUDIT_CANCEL_REASONS


# =====================================================================
# 217# self-review: last_spread_raw 修正の検証テスト
# =====================================================================


class TestLastSpreadRaw217:
    """217# maker_price.last_spread_raw が staleness guard なしで spread を返す."""

    def test_last_spread_raw_bypasses_staleness(self) -> None:
        """last_spread_raw は 60s staleness guard をバイパスする."""
        from unittest.mock import MagicMock, patch
        from scripts.v460.lib.maker_price import MakerPriceCalculator

        # MakerPriceCalculator の最小構築 (依存を mock)
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        mpc = MakerPriceCalculator(cfg, fast_fill_defense=MagicMock(), regime_detector=MagicMock(), base_offset_ratio=0.001)

        # 内部 spread を直接設定 (compute() 経由の代替)
        mpc._last_spread = 500.0
        mpc._last_spread_time = 0.0  # 古い時刻

        # last_spread (staleness guard 付き) → 60s 以上前なので None
        with patch("time.time", return_value=120.0):
            assert mpc.last_spread is None

        # last_spread_raw (guard なし) → 常に値を返す
        assert mpc.last_spread_raw == 500.0

    def test_last_spread_raw_returns_none_when_unset(self) -> None:
        """初期状態では None."""
        from unittest.mock import MagicMock
        from scripts.v460.lib.maker_price import MakerPriceCalculator
        from scripts.v460.lib.fill_config import FillTestConfig
        mpc = MakerPriceCalculator(FillTestConfig(), fast_fill_defense=MagicMock(), regime_detector=MagicMock(), base_offset_ratio=0.001)
        assert mpc.last_spread_raw is None
