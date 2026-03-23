"""120# PnlMeasurer — 約定後 PnL 計測モジュール.

run_fill_test.py FillTestRunner からの God Object 分割:
- _measure_post_fill_pnl (109L) → measure()
- 30s/60s/120s multi-timeframe 計測 (047# E3)
- Early Exit 監視 (054# S3)
- 049# サンプリング制御
- 168# §4.1 #1: sell 保持期間延長 (side 別 post_fill_wait_sec)

型安全: Optional チェーン明示化、Final 定数。
メモリ: __slots__ 適用。
"""

from __future__ import annotations

import asyncio
import logging
import random as _rng
import time
from typing import Awaitable, Callable

from scripts.v460.lib.fill_config import FillTestConfig, PnlMeasurement

logger = logging.getLogger(__name__)

from scripts.v460.lib.constants import BPS_FACTOR as _BPS_FACTOR


class PnlMeasurer:
    """約定後 PnL 計測 — FillTestRunner から分割.

    __slots__ でメモリフットプリントを制御。
    """

    __slots__ = ("_config",)

    def __init__(self, config: FillTestConfig) -> None:
        self._config = config

    @staticmethod
    def _side_pnl_bps(side: str, mid_at_fill: float, mid_after: float) -> float:
        """304# DRY: side 別 PnL bps 計算 (buy: mid上昇が利益, sell: mid下落が利益)."""
        if side == "buy":
            return (mid_after - mid_at_fill) / mid_at_fill * _BPS_FACTOR
        return (mid_at_fill - mid_after) / mid_at_fill * _BPS_FACTOR

    async def measure(
        self,
        filled: bool,
        fill_price: float | None,
        side: str,
        *,
        get_mid_price: Callable[[], Awaitable[float]],
        wait_sec_override: float | None = None,  # 179# D: regime 別 post-fill wait
    ) -> PnlMeasurement:
        """約定後 PnL 計測 — 30s/60s/120s.

        113# R1: run_single_cycle Phase 5.
        047# E3: multi-timeframe 計測.
        054# S3: Early Exit 監視.
        305# Execution Quality 分解: spread_capture + adverse_selection_cost.

        Returns:
            PnlMeasurement dataclass.
        """
        m = PnlMeasurement()
        cfg = self._config

        if not filled or fill_price is None:
            return m

        # 168# §4.1 #1: sell 保持期間延長 — side 別 post_fill_wait_sec
        # 179# D: regime 別 post-fill wait override
        if wait_sec_override is not None:
            wait_sec = wait_sec_override
            logger.debug(
                f"[179# D] Using regime-overridden wait: {wait_sec}s "
                f"(base config={cfg.post_fill_wait_sec}s)"
            )
        else:
            wait_sec = cfg.post_fill_wait_sec
            if side == "sell" and cfg.post_fill_wait_sec_sell is not None:
                wait_sec = cfg.post_fill_wait_sec_sell
                logger.debug(
                    f"[168# sell_hold] Using sell-specific wait: {wait_sec}s "
                    f"(default={cfg.post_fill_wait_sec}s)"
                )

        try:
            m.mid_at_fill = await get_mid_price()
        except Exception as exc:
            logger.debug("mid_at_fill fetch failed: %s", exc)

        # 054# S3: Early Exit 監視付き待機
        early_exit_triggered = False
        t_post_fill_start = time.time()

        if cfg.early_exit_enabled and m.mid_at_fill is not None:
            monitor_sec = cfg.early_exit_monitor_interval_sec
            ticks = max(1, int(wait_sec / monitor_sec))
            tick = 0
            for tick in range(ticks):
                await asyncio.sleep(monitor_sec)
                try:
                    mid_now = await get_mid_price()
                    interim_pnl = self._side_pnl_bps(side, m.mid_at_fill, mid_now)
                    if interim_pnl < -cfg.early_exit_threshold_bps:
                        logger.warning(
                            f"[early_exit] Loss threshold hit at {(tick+1)*monitor_sec:.0f}s: "
                            f"{interim_pnl:+.2f} bps < -{cfg.early_exit_threshold_bps}"
                        )
                        early_exit_triggered = True
                        # 120# A4-2: 中断時点の PnL を保存
                        m.pnl_at_exit_bps = interim_pnl
                        break
                except Exception as e:
                    # 255# bare except → debug log (interim PnL calc 例外可観測化)
                    logger.debug("interim PnL calc failed at tick %d: %s", tick, e, exc_info=True)
                    continue
            elapsed_monitor = (tick + 1) * monitor_sec if early_exit_triggered else ticks * monitor_sec
            remaining = wait_sec - elapsed_monitor
            # 120# A4-2: EE 発動でも固定待機時間まで待って真の post_fill_pnl を取得
            if remaining > 0:
                await asyncio.sleep(remaining)
        else:
            logger.info(f"Waiting {wait_sec}s for PnL measurement...")
            await asyncio.sleep(wait_sec)

        m.actual_measurement_sec = time.time() - t_post_fill_start

        try:
            m.mid_30s_after = await get_mid_price()
        except Exception as exc:
            logger.debug("mid_30s_after fetch failed: %s", exc)

        if m.mid_at_fill is not None and m.mid_30s_after is not None:
            m.post_fill_pnl = self._side_pnl_bps(side, m.mid_at_fill, m.mid_30s_after)
            m.adverse_selected_raw = (
                m.mid_30s_after < m.mid_at_fill if side == "buy"
                else m.mid_30s_after > m.mid_at_fill
            )
            m.adverse_selected = m.post_fill_pnl < -cfg.as_deadzone_bps

            # 305# Execution Quality 分解 (Kissell & Glantz 2003):
            #   PnL = spread_capture + adverse_selection_cost
            # spread_capture: fill_price が mid よりも有利な分 (MM の付加価値)
            # adverse_selection_cost: mid が約定後に不利方向に動いた分
            if fill_price is not None and m.mid_at_fill > 0:
                m.spread_capture_bps = self._side_pnl_bps(
                    side, fill_price, m.mid_at_fill,
                )
                m.adverse_selection_cost_bps = self._side_pnl_bps(
                    side, m.mid_at_fill, m.mid_30s_after,
                )

        # early_exit_triggered → 呼び出し側で rapid_exit フラグを設定
        m.early_exit_triggered = early_exit_triggered

        # 047# E3: +30s (=60s) 計測 — 049# サンプリング制御
        # 565# I1: ベースを cfg.post_fill_wait_sec → 実際の wait_sec に修正
        #   旧: sell(90s)で e3_target_60s = 30*2.0=60s → 既に経過 → "60s PnL" が "30s PnL" と同一地点に崩壊
        #   新: sell(90s)で e3_target_60s = 90*2.0=180s → 正しい2倍窓で計測
        do_e3 = m.mid_at_fill is not None and _rng.random() < cfg.e3_sampling_ratio
        if do_e3:
            e3_target_60s = wait_sec * cfg.e3_60s_multiplier
            e3_elapsed = time.time() - t_post_fill_start
            e3_wait_60 = max(0.0, e3_target_60s - e3_elapsed)
            if e3_wait_60 > 0:
                await asyncio.sleep(e3_wait_60)
            try:
                m.mid_60s_after = await get_mid_price()
                m.post_fill_60s_pnl = self._side_pnl_bps(side, m.mid_at_fill, m.mid_60s_after)
            except Exception as exc:
                logger.debug("mid_60s_after PnL failed: %s", exc)

            e3_target_120s = wait_sec * cfg.e3_120s_multiplier
            e3_elapsed = time.time() - t_post_fill_start
            e3_wait_120 = max(0.0, e3_target_120s - e3_elapsed)
            if e3_wait_120 > 0:
                await asyncio.sleep(e3_wait_120)
            try:
                m.mid_120s_after = await get_mid_price()
                m.post_fill_120s_pnl = self._side_pnl_bps(side, m.mid_at_fill, m.mid_120s_after)
            except Exception as exc:
                logger.debug("mid_120s_after PnL failed: %s", exc)

        # 137# P1-11: fee 控除統一 — PnL bps から maker fee を一律控除
        # 139# §9-#4: 仕様確定 — maker fee のみ控除 (現行は maker 注文のみ)。
        # taker_fee_bps は将来の taker/IOC 注文導入時に使用する予約フィールド。
        # slippage 控除は queue_wait 等のデータ蓄積後に別 P1 課題として対応予定。
        if cfg.pnl_fee_deduction_enabled and cfg.maker_fee_bps > 0:
            fee = cfg.maker_fee_bps
            if m.post_fill_pnl is not None:
                m.post_fill_pnl -= fee
            if m.post_fill_60s_pnl is not None:
                m.post_fill_60s_pnl -= fee
            if m.post_fill_120s_pnl is not None:
                m.post_fill_120s_pnl -= fee
            if m.pnl_at_exit_bps is not None:
                m.pnl_at_exit_bps -= fee

        return m
