"""349#/352# DynamicKillManager EWMA 修正テスト.

P0: EWMA 状態永続化
P1: EWMA シード安定化 (352#: history 平均→直接値に撤回)
P2: TIME LIMIT 後の EWMA decay (352#: effective_threshold 使用)
352#: exact rebuild 検証, effective_threshold テスト追加
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ztb.risk.sell_dynamic_kill import DynamicKillConfig, DynamicKillManager


# =====================================================================
# P0: EWMA 状態永続化
# =====================================================================


class TestEwmaStatePersistence:
    """349# P0: export_state / import_state に ewma_value が含まれる."""

    def test_export_includes_ewma_value(self) -> None:
        mgr = DynamicKillManager(DynamicKillConfig(window=5, ewma_alpha=0.05))
        mgr.track(1.0)
        state = mgr.export_state()
        assert "ewma_value" in state
        assert state["ewma_value"] == 1.0

    def test_roundtrip_ewma(self) -> None:
        cfg = DynamicKillConfig(window=3, ewma_alpha=0.1)
        mgr = DynamicKillManager(cfg)
        for v in [1.0, -2.0, 0.5]:
            mgr.track(v)
        state = mgr.export_state()

        mgr2 = DynamicKillManager(cfg)
        mgr2.import_state(state)
        assert mgr2._ewma_value == pytest.approx(state["ewma_value"])  # type: ignore[attr-defined]

    def test_import_missing_ewma_rebuilds(self) -> None:
        """ewma_value が欠落した旧フォーマットの state → history から再構築."""
        cfg = DynamicKillConfig(window=3, ewma_alpha=0.1)
        mgr = DynamicKillManager(cfg)
        # まず track() 経由で正確な EWMA を算出
        ref = DynamicKillManager(cfg)
        for v in [1.0, -1.0, 0.5]:
            ref.track(v)
        expected_ewma = ref._ewma_value  # type: ignore[attr-defined]
        # 旧フォーマットの state を構築
        old_state = {
            "pnl_history": [1.0, -1.0, 0.5],
            "cooldown": 0,
            "total_kills": 0,
            "total_cooldown_cycles": 0,
            "side": "sell",
            "stale_counter": 0,
            "total_probe_cycles": 0,
            "consecutive_probes": 0,
            "force_released": False,
            "kill_activated_at": None,
            # ewma_value は意図的に欠落
        }
        mgr.import_state(old_state)
        # 352#: 再構築値が track() 経由と一致することを検証
        assert mgr._ewma_value == pytest.approx(expected_ewma)  # type: ignore[attr-defined]

    def test_import_missing_ewma_empty_history(self) -> None:
        """ewma_value 欠落 + 空 history → None のまま."""
        cfg = DynamicKillConfig(window=3, ewma_alpha=0.1)
        mgr = DynamicKillManager(cfg)
        mgr.import_state({"pnl_history": []})
        assert mgr._ewma_value is None  # type: ignore[attr-defined]

    def test_import_ewma_disabled(self) -> None:
        """ewma_alpha=0 のとき import_state 後も None."""
        cfg = DynamicKillConfig(window=3, ewma_alpha=0.0)
        mgr = DynamicKillManager(cfg)
        mgr.import_state({"pnl_history": [1.0, 2.0, 3.0]})
        assert mgr._ewma_value is None  # type: ignore[attr-defined]


# =====================================================================
# P1: EWMA シード安定化
# =====================================================================


class TestEwmaSeedStability:
    """352#: 初回シードは当該値を直接使用 (349# P1 の history 平均は撤回)."""

    def test_first_track_seeds_with_value(self) -> None:
        """初回 track → その値がシード."""
        mgr = DynamicKillManager(DynamicKillConfig(window=5, ewma_alpha=0.1))
        mgr.track(-5.0)
        assert mgr._ewma_value == -5.0  # type: ignore[attr-defined]

    def test_second_track_applies_ewma_update(self) -> None:
        """352#: 2件目の track で EWMA 更新式が適用される."""
        cfg = DynamicKillConfig(window=5, ewma_alpha=0.1)
        mgr = DynamicKillManager(cfg)
        mgr.track(1.0)  # seed = 1.0
        mgr.track(-10.0)  # ewma = 0.1 * (-10) + 0.9 * 1.0 = -0.1
        assert mgr._ewma_value == pytest.approx(0.1 * (-10.0) + 0.9 * 1.0)  # type: ignore[attr-defined]

    def test_ewma_not_poisoned_by_single_outlier(self) -> None:
        """再起動後の単一外れ値が EWMA を支配しない."""
        cfg = DynamicKillConfig(
            window=50, threshold_bps=-0.5, ewma_alpha=0.05,
        )
        mgr = DynamicKillManager(cfg)
        # 50件の正常データで warmup
        for _ in range(50):
            mgr.track(0.1)
        # export/import (EWMA 付き)
        state = mgr.export_state()
        mgr2 = DynamicKillManager(cfg)
        mgr2.import_state(state)
        # 1件の大きな外れ値
        mgr2.track(-20.0)
        # EWMA は 0.05 * (-20) + 0.95 * (前の EWMA ≈ 0.1) = -1.0 + 0.095 ≈ -0.905
        # kill 閾値 -0.5 を下回るが、-20.0 そのものにはならない
        assert mgr2._ewma_value > -2.0  # type: ignore[attr-defined]


# =====================================================================
# P2: TIME LIMIT 後の EWMA decay
# =====================================================================


class TestTimeLimitEwmaDecay:
    """349# P2: TIME LIMIT 解除時に EWMA を threshold 付近にリセット."""

    def test_time_limit_resets_ewma(self) -> None:
        """TIME LIMIT で EWMA が effective_threshold * 0.8 にリセットされる."""
        cfg = DynamicKillConfig(
            window=3, threshold_bps=-0.5, resume_window=2,
            ewma_alpha=0.1, max_kill_duration_sec=60,
        )
        mgr = DynamicKillManager(cfg)
        for _ in range(3):
            mgr.track(-2.0)
        killed, _ = mgr.check_kill()
        assert killed is True

        # TIME LIMIT を超過させる
        with patch("ztb.risk.sell_dynamic_kill.time") as mock_time:
            mock_time.time.return_value = mgr._kill_activated_at + 61  # type: ignore[operator]
            killed2, _ = mgr.check_kill()
        assert killed2 is False
        # EWMA は threshold * 0.8 = -0.5 * 0.8 = -0.4 にリセット
        assert mgr._ewma_value == pytest.approx(-0.5 * 0.8)  # type: ignore[attr-defined]

    def test_time_limit_uses_effective_threshold_with_offset(self) -> None:
        """352#: TIME LIMIT で threshold_offset_bps が反映される."""
        cfg = DynamicKillConfig(
            window=3, threshold_bps=-0.5, resume_window=2,
            ewma_alpha=0.1, max_kill_duration_sec=60,
        )
        mgr = DynamicKillManager(cfg)
        for _ in range(3):
            mgr.track(-2.0)
        killed, _ = mgr.check_kill()
        assert killed is True

        # TIME LIMIT + threshold_offset_bps=0.3 (緩和)
        # effective = -0.5 - 0.3 = -0.8, reset = -0.8 * 0.8 = -0.64
        with patch("ztb.risk.sell_dynamic_kill.time") as mock_time:
            mock_time.time.return_value = mgr._kill_activated_at + 61  # type: ignore[operator]
            killed2, _ = mgr.check_kill(threshold_offset_bps=0.3)
        assert killed2 is False
        assert mgr._ewma_value == pytest.approx((-0.5 - 0.3) * 0.8)  # type: ignore[attr-defined]

    def test_time_limit_with_regime_threshold(self) -> None:
        """TIME LIMIT でレジーム別閾値が使用される."""
        cfg = DynamicKillConfig(
            window=3, threshold_bps=-0.5, resume_window=2,
            ewma_alpha=0.1, max_kill_duration_sec=60,
            regime_thresholds={"trending_down": -1.0},
        )
        mgr = DynamicKillManager(cfg)
        for _ in range(3):
            mgr.track(-2.0)
        killed, _ = mgr.check_kill(regime="trending_down")
        assert killed is True

        with patch("ztb.risk.sell_dynamic_kill.time") as mock_time:
            mock_time.time.return_value = mgr._kill_activated_at + 61  # type: ignore[operator]
            killed2, _ = mgr.check_kill(regime="trending_down")
        assert killed2 is False
        # trending_down threshold = -1.0, reset = -1.0 * 0.8 = -0.8
        assert mgr._ewma_value == pytest.approx(-1.0 * 0.8)  # type: ignore[attr-defined]

    def test_time_limit_regime_plus_offset(self) -> None:
        """352#: TIME LIMIT で regime_threshold + offset が両方反映."""
        cfg = DynamicKillConfig(
            window=3, threshold_bps=-0.5, resume_window=2,
            ewma_alpha=0.1, max_kill_duration_sec=60,
            regime_thresholds={"trending_down": -1.0},
        )
        mgr = DynamicKillManager(cfg)
        for _ in range(3):
            mgr.track(-2.0)
        killed, _ = mgr.check_kill(regime="trending_down")
        assert killed is True

        # effective = -1.0 - 0.2 = -1.2, reset = -1.2 * 0.8 = -0.96
        with patch("ztb.risk.sell_dynamic_kill.time") as mock_time:
            mock_time.time.return_value = mgr._kill_activated_at + 61  # type: ignore[operator]
            killed2, _ = mgr.check_kill(regime="trending_down", threshold_offset_bps=0.2)
        assert killed2 is False
        assert mgr._ewma_value == pytest.approx((-1.0 - 0.2) * 0.8)  # type: ignore[attr-defined]

    def test_time_limit_no_decay_when_ewma_disabled(self) -> None:
        """ewma_alpha=0 のとき TIME LIMIT で decay は行わない."""
        cfg = DynamicKillConfig(
            window=3, threshold_bps=-0.5, resume_window=2,
            ewma_alpha=0.0, max_kill_duration_sec=60,
        )
        mgr = DynamicKillManager(cfg)
        for _ in range(3):
            mgr.track(-2.0)
        killed, _ = mgr.check_kill()
        assert killed is True

        with patch("ztb.risk.sell_dynamic_kill.time") as mock_time:
            mock_time.time.return_value = mgr._kill_activated_at + 61  # type: ignore[operator]
            killed2, _ = mgr.check_kill()
        assert killed2 is False
        assert mgr._ewma_value is None  # type: ignore[attr-defined]

    def test_after_time_limit_next_fill_can_prevent_rekill(self) -> None:
        """TIME LIMIT+decay 後、良い fill が入れば再 kill を回避できる."""
        cfg = DynamicKillConfig(
            window=3, threshold_bps=-0.5, resume_window=2,
            ewma_alpha=0.1, max_kill_duration_sec=60,
        )
        mgr = DynamicKillManager(cfg)
        for _ in range(3):
            mgr.track(-2.0)
        mgr.check_kill()  # kill activated

        # TIME LIMIT
        with patch("ztb.risk.sell_dynamic_kill.time") as mock_time:
            mock_time.time.return_value = mgr._kill_activated_at + 61  # type: ignore[operator]
            mgr.check_kill()
        # EWMA = -0.4 (threshold * 0.8)
        # 良い fill
        mgr.track(2.0)
        # EWMA = 0.1 * 2.0 + 0.9 * (-0.4) = 0.2 - 0.36 = -0.16 > -0.5
        killed, _ = mgr.check_kill()
        assert killed is False


# =====================================================================
# reset() テスト
# =====================================================================


class TestResetIncludesEwma:
    """349#: reset() が EWMA も初期化する."""

    def test_reset_clears_ewma(self) -> None:
        mgr = DynamicKillManager(DynamicKillConfig(window=3, ewma_alpha=0.1))
        mgr.track(1.0)
        assert mgr._ewma_value is not None  # type: ignore[attr-defined]
        mgr.reset()
        assert mgr._ewma_value is None  # type: ignore[attr-defined]


# =====================================================================
# 352#: exact rebuild 検証
# =====================================================================


class TestExactRebuild:
    """352# P0-3: _rebuild_ewma_from_history が track() と同一の EWMA を返す."""

    def test_rebuild_matches_track_sequence(self) -> None:
        """history replay が track() の逐次更新と一致する."""
        cfg = DynamicKillConfig(window=10, ewma_alpha=0.1)
        data = [1.0, -1.0, 0.5, -0.5, 0.0, -10.0, 3.0, -2.5, 1.5, -0.3]

        # track() で正解 EWMA を計算
        mgr_ref = DynamicKillManager(cfg)
        for v in data:
            mgr_ref.track(v)
        expected = mgr_ref._ewma_value  # type: ignore[attr-defined]

        # export → ewma_value 除去 → import (rebuild 発動)
        state = mgr_ref.export_state()
        del state["ewma_value"]
        mgr_test = DynamicKillManager(cfg)
        mgr_test.import_state(state)

        assert mgr_test._ewma_value == pytest.approx(expected)  # type: ignore[attr-defined]

    def test_rebuild_single_element(self) -> None:
        """1 要素の history で rebuild → その値がシード."""
        cfg = DynamicKillConfig(window=5, ewma_alpha=0.05)
        mgr = DynamicKillManager(cfg)
        mgr.import_state({"pnl_history": [-7.0]})
        assert mgr._ewma_value == pytest.approx(-7.0)  # type: ignore[attr-defined]

    def test_rebuild_vs_track_long_sequence(self) -> None:
        """150 要素の長い系列で rebuild と track() が一致."""
        cfg = DynamicKillConfig(window=50, ewma_alpha=0.05)
        import random
        rng = random.Random(42)
        data = [rng.gauss(0.0, 3.0) for _ in range(150)]

        mgr_ref = DynamicKillManager(cfg)
        for v in data:
            mgr_ref.track(v)
        expected = mgr_ref._ewma_value  # type: ignore[attr-defined]

        state = mgr_ref.export_state()
        del state["ewma_value"]
        mgr_test = DynamicKillManager(cfg)
        mgr_test.import_state(state)
        # window * 3 = 150 なので全履歴が保持される
        assert mgr_test._ewma_value == pytest.approx(expected, abs=1e-10)  # type: ignore[attr-defined]
