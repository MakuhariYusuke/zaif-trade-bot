"""549# EWMA 入力クランプ (Winsorization) テスト.

sell_dynamic_kill EWMA poisoning problem:
  - 単一の極端な AS イベント (-13.5 bps) が EWMA を数十分間支配
  - systematic risk ではなく idiosyncratic shock で kill スパイラル誘発
  - ewma_input_clamp_bps で入力を Winsorize して解決
"""

from __future__ import annotations

import pytest

from ztb.risk.sell_dynamic_kill import DynamicKillConfig, DynamicKillManager


class TestEwmaInputClamp:
    """549#: ewma_input_clamp_bps による Winsorization."""

    def _make_mgr(
        self,
        clamp: float = 5.0,
        alpha: float = 0.05,
        window: int = 50,
    ) -> DynamicKillManager:
        cfg = DynamicKillConfig(
            window=window,
            ewma_alpha=alpha,
            ewma_input_clamp_bps=clamp,
        )
        return DynamicKillManager(cfg)

    def test_clamp_zero_disables(self) -> None:
        """clamp=0.0 ならクランプ無効 (生値がそのまま反映)."""
        mgr = self._make_mgr(clamp=0.0)
        mgr.track(-13.5)
        assert mgr._ewma_value == pytest.approx(-13.5)

    def test_clamp_limits_negative_extreme(self) -> None:
        """-13.5 bps は -5.0 にクランプされる."""
        mgr = self._make_mgr(clamp=5.0)
        mgr.track(-13.5)
        assert mgr._ewma_value == pytest.approx(-5.0)

    def test_clamp_limits_positive_extreme(self) -> None:
        """+10.0 bps は +5.0 にクランプされる."""
        mgr = self._make_mgr(clamp=5.0)
        mgr.track(10.0)
        assert mgr._ewma_value == pytest.approx(5.0)

    def test_within_range_passes_through(self) -> None:
        """クランプ範囲内の値はそのまま通過."""
        mgr = self._make_mgr(clamp=5.0)
        mgr.track(-3.0)
        assert mgr._ewma_value == pytest.approx(-3.0)

    def test_boundary_exact(self) -> None:
        """境界値 ±5.0 はそのまま通過."""
        mgr = self._make_mgr(clamp=5.0)
        mgr.track(-5.0)
        assert mgr._ewma_value == pytest.approx(-5.0)
        mgr2 = self._make_mgr(clamp=5.0)
        mgr2.track(5.0)
        assert mgr2._ewma_value == pytest.approx(5.0)

    def test_ewma_sequence_with_clamp(self) -> None:
        """実際の poisoning シナリオ: [-13.5, 1.12, -6.76, 4.23] with clamp=5."""
        mgr = self._make_mgr(clamp=5.0, alpha=0.05)
        pnl_history = [-13.536, 1.12, -6.755, 4.231]
        for v in pnl_history:
            mgr.track(v)
        # EWMA without clamp: seed=-13.536, very negative
        # EWMA with clamp=5: seed=clamp(-13.536)=-5.0
        #   step2: 0.05*1.12 + 0.95*(-5.0) = 0.056 - 4.75 = -4.694
        #   step3: 0.05*clamp(-6.755=-5.0) + 0.95*(-4.694) = -0.25 - 4.4593 = -4.7093
        #   step4: 0.05*4.231 + 0.95*(-4.7093) = 0.21155 - 4.47384 = -4.26228
        expected = -4.26228
        assert mgr._ewma_value is not None
        assert mgr._ewma_value == pytest.approx(expected, abs=0.01)
        # Without clamp: much worse
        mgr_nocl = self._make_mgr(clamp=0.0, alpha=0.05)
        for v in pnl_history:
            mgr_nocl.track(v)
        assert mgr_nocl._ewma_value is not None
        assert mgr_nocl._ewma_value < mgr._ewma_value  # type: ignore[operator]

    def test_rebuild_ewma_applies_clamp(self) -> None:
        """549#: _rebuild_ewma_from_history() もクランプを適用し track() と一致."""
        cfg = DynamicKillConfig(
            window=50, ewma_alpha=0.05, ewma_input_clamp_bps=5.0,
        )
        # track 経由で EWMA 構築
        mgr_track = DynamicKillManager(cfg)
        for v in [-13.536, 1.12, -6.755, 4.231]:
            mgr_track.track(v)

        # import_state (ewma_value 欠落) → rebuild で再構築
        mgr_rebuild = DynamicKillManager(cfg)
        old_state = {
            "pnl_history": [-13.536, 1.12, -6.755, 4.231],
            "cooldown": 0,
            "total_kills": 0,
            "total_cooldown_cycles": 0,
            "side": "sell",
            "stale_counter": 0,
            "total_probe_cycles": 0,
            "consecutive_probes": 0,
            "force_released": False,
            "kill_activated_at": None,
            # ewma_value 意図的欠落 → rebuild
        }
        mgr_rebuild.import_state(old_state)
        assert mgr_rebuild._ewma_value == pytest.approx(mgr_track._ewma_value)  # type: ignore[arg-type]

    def test_export_import_roundtrip_with_clamp(self) -> None:
        """ewma_input_clamp_bps 付きの export/import が正常にラウンドトリップ."""
        mgr = self._make_mgr(clamp=5.0)
        mgr.track(-20.0)  # clamped to -5.0
        mgr.track(2.0)
        state = mgr.export_state()
        assert "ewma_value" in state

        mgr2 = self._make_mgr(clamp=5.0)
        mgr2.import_state(state)
        assert mgr2._ewma_value == pytest.approx(state["ewma_value"])

    def test_pnl_history_stores_raw(self) -> None:
        """pnl_history は生値を保持し、クランプは EWMA 計算時のみ適用."""
        mgr = self._make_mgr(clamp=5.0)
        mgr.track(-13.5)
        assert mgr._pnl_history[-1] == -13.5  # raw value preserved


class TestEwmaInputClampConfig:
    """DynamicKillConfig.ewma_input_clamp_bps のバリデーション."""

    def test_default_zero(self) -> None:
        """デフォルト値は 0.0 (無効)."""
        cfg = DynamicKillConfig()
        assert cfg.ewma_input_clamp_bps == 0.0

    def test_positive_value_accepted(self) -> None:
        cfg = DynamicKillConfig(ewma_input_clamp_bps=5.0)
        assert cfg.ewma_input_clamp_bps == 5.0
