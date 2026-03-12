"""219# DynamicKillManager progressive probe + force release tests."""
from __future__ import annotations

import pytest

from ztb.risk.sell_dynamic_kill import (
    BuyDynamicKillManager,
    DynamicKillConfig,
    DynamicKillManager,
)


class TestProgressiveProbe219:
    """219# progressive probe interval + force release."""

    def _make_mgr(
        self,
        *,
        window: int = 5,
        threshold_bps: float = -0.5,
        resume_window: int = 2,
        max_stale: int = 10,
        min_probe_interval: int = 2,
        max_force_release_probes: int = 5,
    ) -> DynamicKillManager:
        return DynamicKillManager(
            DynamicKillConfig(
                enabled=True,
                window=window,
                threshold_bps=threshold_bps,
                resume_window=resume_window,
                max_stale_kill_cycles=max_stale,
                min_probe_interval=min_probe_interval,
                max_force_release_probes=max_force_release_probes,
            ),
            side="buy",
        )

    def _fill_bad_data(self, mgr: DynamicKillManager, n: int = 5) -> None:
        """window 分のネガティブ PnL を投入."""
        for _ in range(n):
            mgr.track(-1.0)

    def test_progressive_interval_halves(self) -> None:
        """連続 probe で interval が半減する (10→5→3→2→2)."""
        mgr = self._make_mgr(max_stale=10, resume_window=1, min_probe_interval=2)
        self._fill_bad_data(mgr)

        probe_gaps: list[int] = []
        kills_since_probe = 0

        for _ in range(100):
            killed, _ = mgr.check_kill()
            if killed:
                kills_since_probe += 1
            else:
                probe_gaps.append(kills_since_probe)
                kills_since_probe = 0

        # 1st probe at ~10 cycles, 2nd at ~5, 3rd at ~3, 4th at ~2
        assert len(probe_gaps) >= 4, f"expected >=4 probes, got {len(probe_gaps)}: {probe_gaps}"
        # 最初の interval が最も長く、progressive に短縮
        assert probe_gaps[0] > probe_gaps[-2], (
            f"first gap {probe_gaps[0]} should be > later gaps {probe_gaps[-2]}"
        )

    def test_effective_probe_interval_calculation(self) -> None:
        """_effective_probe_interval が正しく計算される."""
        mgr = self._make_mgr(max_stale=10, min_probe_interval=2)
        self._fill_bad_data(mgr)

        # consecutive_probes=0 → base=10
        assert mgr._effective_probe_interval() == 10

        # simulate consecutive probes
        mgr._consecutive_probes = 1
        assert mgr._effective_probe_interval() == 5  # 10 // 2 = 5

        mgr._consecutive_probes = 2
        assert mgr._effective_probe_interval() == 3  # 5 // 2 = 2.5 → (5+1)//2=3

        mgr._consecutive_probes = 3
        assert mgr._effective_probe_interval() == 2  # 3 // 2 = 1.5 → (3+1)//2=2

        mgr._consecutive_probes = 4
        assert mgr._effective_probe_interval() == 2  # min_probe_interval floor

    def test_force_release_after_n_probes(self) -> None:
        """max_force_release_probes 回の連続 probe で kill 強制解除."""
        mgr = self._make_mgr(
            max_stale=4, resume_window=1, max_force_release_probes=3,
            min_probe_interval=2,
        )
        self._fill_bad_data(mgr)

        # run until force release
        force_released = False
        total_cycles = 0
        for _ in range(100):
            killed, _ = mgr.check_kill()
            total_cycles += 1
            if not killed and mgr._force_released:
                force_released = True
                break

        assert force_released, "force release should trigger"
        assert mgr._consecutive_probes == 3

        # After force release, all subsequent calls return not killed
        for _ in range(10):
            killed, _ = mgr.check_kill()
            assert not killed, "should stay unblocked during force release"

    def test_force_release_ends_on_track(self) -> None:
        """track() で force_released がリセットされる."""
        mgr = self._make_mgr(
            max_stale=3, resume_window=1, max_force_release_probes=2,
            min_probe_interval=2,
        )
        self._fill_bad_data(mgr)

        # run until force release
        for _ in range(50):
            killed, _ = mgr.check_kill()
            if mgr._force_released:
                break

        assert mgr._force_released

        # track() should end force release
        mgr.track(-1.0)
        assert not mgr._force_released
        assert mgr._consecutive_probes == 0

        # kill should re-activate (mean still bad)
        killed, _ = mgr.check_kill()
        assert killed

    def test_force_release_zero_disables(self) -> None:
        """max_force_release_probes=0 で force release 無効."""
        mgr = self._make_mgr(
            max_stale=3, resume_window=1, max_force_release_probes=0,
            min_probe_interval=2,
        )
        self._fill_bad_data(mgr)

        # probe は発生するが force release はしない
        for _ in range(50):
            mgr.check_kill()
        assert not mgr._force_released

    def test_track_resets_consecutive_probes(self) -> None:
        """track() で consecutive_probes がリセットされ、次は full interval."""
        mgr = self._make_mgr(max_stale=6, resume_window=1, min_probe_interval=2)
        self._fill_bad_data(mgr)

        # probe 2回発生させる
        probes_seen = 0
        for _ in range(30):
            killed, _ = mgr.check_kill()
            if not killed:
                probes_seen += 1
                if probes_seen >= 2:
                    break

        assert mgr._consecutive_probes >= 2

        # track() で連続数リセット
        mgr.track(-1.0)
        assert mgr._consecutive_probes == 0

        # 次の probe まで full interval (6) が必要
        kills = 0
        for _ in range(10):
            killed, _ = mgr.check_kill()
            if killed:
                kills += 1
            else:
                break
        # full interval 6 に戻っているので 5〜6 回 kill
        assert kills >= 5, f"after track(), first probe should take full interval, got {kills}"

    def test_export_import_preserves_219_fields(self) -> None:
        """export/import で consecutive_probes, force_released が保持される."""
        mgr = self._make_mgr(max_stale=4, resume_window=1, min_probe_interval=2)
        self._fill_bad_data(mgr)

        # probe 1回
        for _ in range(20):
            killed, _ = mgr.check_kill()
            if not killed:
                break

        state = mgr.export_state()
        assert "consecutive_probes" in state
        assert "force_released" in state
        assert state["consecutive_probes"] >= 1

        mgr2 = self._make_mgr(max_stale=4, resume_window=1, min_probe_interval=2)
        mgr2.import_state(state)
        assert mgr2._consecutive_probes == state["consecutive_probes"]
        assert mgr2._force_released == state["force_released"]

    def test_buy_manager_progressive_works(self) -> None:
        """BuyDynamicKillManager でも progressive probe が機能."""
        mgr = BuyDynamicKillManager(
            DynamicKillConfig(
                enabled=True,
                window=5,
                threshold_bps=-0.5,
                resume_window=1,
                max_stale_kill_cycles=6,
                min_probe_interval=2,
                max_force_release_probes=3,
            )
        )
        for _ in range(5):
            mgr.track(-1.0)

        probes = 0
        force_released = False
        for _ in range(100):
            killed, _ = mgr.check_kill()
            if not killed:
                probes += 1
            if mgr._force_released:
                force_released = True
                break

        assert probes >= 3, f"expected >= 3 probes, got {probes}"
        assert force_released

    def test_default_config_values(self) -> None:
        """219# デフォルト設定値の確認."""
        cfg = DynamicKillConfig()
        assert cfg.max_stale_kill_cycles == 10  # 219# 30→10
        assert cfg.min_probe_interval == 2
        assert cfg.max_force_release_probes == 5
