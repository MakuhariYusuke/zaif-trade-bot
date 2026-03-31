"""669# restore パスの max_lot クランプテスト.

_try_lot_restore / restore_lot_on_success が、hot-reload で max_lot が
縮小された場合に旧値へ戻らないことを検証する。
"""

from __future__ import annotations

import logging

import pytest

from scripts.v460.lib.balance_checker import BalanceChecker
from scripts.v460.lib.fill_config import FillTestConfig


def _make_config(**overrides: object) -> FillTestConfig:
    defaults = dict(
        order_quantity=0.001,
        min_order_btc=0.001,
        max_lot=0.001,
        balance_margin_ratio=1.1,
        dust_sweep_enabled=False,
    )
    defaults.update(overrides)
    return FillTestConfig(**defaults)  # type: ignore[arg-type]


# ==================================================================
# _try_lot_restore: max_lot clamp
# ==================================================================


class TestTryLotRestoreMaxLotClamp:
    """669# _try_lot_restore が max_lot を超えて復元しないことの検証."""

    def test_restore_clamped_by_max_lot(self) -> None:
        """pre_shrink=0.002 だが max_lot=0.001 → 0.001 にクランプ."""
        cfg = _make_config(max_lot=0.001)
        bc = BalanceChecker(cfg)
        # 476# で 0.002 に拡大された後に縮小されたシナリオを模擬
        bc.pre_shrink_lot = 0.002
        bc._current_lot = 0.0005  # 残高不足で縮小中
        bc._try_lot_restore(True, "JPY")
        assert bc.current_lot == 0.001  # max_lot でクランプ

    def test_restore_not_clamped_when_within_max_lot(self) -> None:
        """pre_shrink=0.001 で max_lot=0.005 → そのまま 0.001."""
        cfg = _make_config(max_lot=0.005)
        bc = BalanceChecker(cfg)
        bc._apply_lot_shrink(0.0005, "shrink")
        bc._try_lot_restore(True, "BTC")
        assert bc.current_lot == 0.001  # pre_shrink_lot そのまま

    def test_restore_no_clamp_when_max_lot_zero(self) -> None:
        """max_lot=0 (無効) → クランプなしで pre_shrink に復元."""
        cfg = _make_config(max_lot=0)
        bc = BalanceChecker(cfg)
        bc.pre_shrink_lot = 0.003
        bc._current_lot = 0.001
        bc._try_lot_restore(True, "JPY")
        assert bc.current_lot == 0.003

    def test_restore_clamp_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        """復元がクランプされた場合もログに正しい値が出力される."""
        cfg = _make_config(max_lot=0.001)
        bc = BalanceChecker(cfg)
        bc.pre_shrink_lot = 0.002
        bc._current_lot = 0.0005
        with caplog.at_level(logging.INFO):
            bc._try_lot_restore(True, "JPY")
        assert "0.0005" in caplog.text  # old_lot
        assert "0.0010" in caplog.text  # restored (clamped)

    def test_hot_reload_scenario(self) -> None:
        """実運用シナリオ: max_lot が 0.005→0.001 に hot-reload された場合."""
        cfg = _make_config(max_lot=0.005)
        bc = BalanceChecker(cfg)
        # Step 1: 476# で 0.002 に拡大
        bc._current_lot = 0.002
        # Step 2: 残高不足で 0.0005 に縮小
        bc._apply_lot_shrink(0.0005, "shrink")
        assert bc.pre_shrink_lot == 0.002
        # Step 3: max_lot が hot-reload で 0.001 に変更
        cfg.max_lot = 0.001
        # Step 4: 残高回復 → 復元時にクランプ
        bc._try_lot_restore(True, "JPY")
        assert bc.current_lot == 0.001  # 0.002 ではなく 0.001


# ==================================================================
# restore_lot_on_success: max_lot clamp
# ==================================================================


class TestRestoreLotOnSuccessMaxLotClamp:
    """669# restore_lot_on_success が max_lot を超えて復元しないことの検証."""

    def test_success_restore_clamped_by_max_lot(self) -> None:
        """balance_shrink 解除時も max_lot でクランプ."""
        cfg = _make_config(max_lot=0.001)
        bc = BalanceChecker(cfg)
        bc.pre_shrink_lot = 0.002
        bc._current_lot = 0.0005
        bc.balance_shrink_active = True
        bc.restore_lot_on_success()
        assert bc.current_lot == 0.001  # max_lot でクランプ
        assert bc.balance_shrink_active is False

    def test_success_restore_not_clamped_within_limit(self) -> None:
        """max_lot 以内なら通常通り復元."""
        cfg = _make_config(max_lot=0.005)
        bc = BalanceChecker(cfg)
        bc.pre_shrink_lot = 0.002
        bc._current_lot = 0.001
        bc.balance_shrink_active = True
        bc.restore_lot_on_success()
        assert bc.current_lot == 0.002

    def test_success_restore_no_clamp_when_max_lot_zero(self) -> None:
        """max_lot=0 (無効) → クランプなし."""
        cfg = _make_config(max_lot=0)
        bc = BalanceChecker(cfg)
        bc.pre_shrink_lot = 0.003
        bc._current_lot = 0.001
        bc.balance_shrink_active = True
        bc.restore_lot_on_success()
        assert bc.current_lot == 0.003

    def test_success_restore_hot_reload_scenario(self) -> None:
        """実運用: balance_shrink 中に max_lot 変更 → 成功時のクランプ."""
        cfg = _make_config(max_lot=0.005)
        bc = BalanceChecker(cfg)
        bc.pre_shrink_lot = 0.003
        bc._current_lot = 0.001
        bc.balance_shrink_active = True
        # hot-reload
        cfg.max_lot = 0.001
        bc.restore_lot_on_success()
        assert bc.current_lot == 0.001  # 0.003 ではなく 0.001
        assert bc.balance_shrink_active is False

    def test_no_restore_when_not_active(self) -> None:
        """balance_shrink_active=False なら何もしない."""
        cfg = _make_config(max_lot=0.001)
        bc = BalanceChecker(cfg)
        bc.pre_shrink_lot = 0.002
        bc._current_lot = 0.0005
        bc.restore_lot_on_success()
        assert bc.current_lot == 0.0005  # 変化なし
