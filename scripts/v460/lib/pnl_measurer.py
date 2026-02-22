"""120# PnlMeasurer — 約定後 PnL 計測モジュール.

run_fill_test.py FillTestRunner からの God Object 分割:
- _measure_post_fill_pnl (109L) → measure()
- 30s/60s/120s multi-timeframe 計測 (047# E3)
- Early Exit 監視 (054# S3)
- 049# サンプリング制御

型安全: Optional チェーン明示化、Final 定数。
メモリ: __slots__ 適用。
"""

from __future__ import annotations

import asyncio
import logging
import random as _rng
import time
from typing import Awaitable, Callable, Final, Optional

from scripts.v460.lib.fill_config import FillTestConfig, PnlMeasurement

logger = logging.getLogger(__name__)

# 定数
_BPS_FACTOR: Final[int] = 10_000


class PnlMeasurer:
    """約定後 PnL 計測 — FillTestRunner から分割.

    __slots__ でメモリフットプリントを制御。
    """

    __slots__ = ("_config",)

    def __init__(self, config: FillTestConfig) -> None:
        self._config = config

    async def measure(
        self,
        filled: bool,
        fill_price: Optional[float],
        side: str,
        *,
        get_mid_price: Callable[[], Awaitable[float]],
    ) -> PnlMeasurement:
        """約定後 PnL 計測 — 30s/60s/120s.

        113# R1: run_single_cycle Phase 5.
        047# E3: multi-timeframe 計測.
        054# S3: Early Exit 監視.

        Returns:
            PnlMeasurement dataclass.
        """
        m = PnlMeasurement()
        cfg = self._config

        if not filled or fill_price is None:
            return m

        try:
            m.mid_at_fill = await get_mid_price()
        except Exception:
            pass

        # 054# S3: Early Exit 監視付き 30s 待機
        early_exit_triggered = False
        t_post_fill_start = time.time()

        if cfg.early_exit_enabled and m.mid_at_fill is not None:
            monitor_sec = cfg.early_exit_monitor_interval_sec
            ticks = max(1, int(cfg.post_fill_wait_sec / monitor_sec))
            tick = 0
            for tick in range(ticks):
                await asyncio.sleep(monitor_sec)
                try:
                    mid_now = await get_mid_price()
                    if side == "buy":
                        interim_pnl = (mid_now - m.mid_at_fill) / m.mid_at_fill * _BPS_FACTOR
                    else:
                        interim_pnl = (m.mid_at_fill - mid_now) / m.mid_at_fill * _BPS_FACTOR
                    if interim_pnl < -cfg.early_exit_threshold_bps:
                        logger.warning(
                            f"[early_exit] Loss threshold hit at {(tick+1)*monitor_sec:.0f}s: "
                            f"{interim_pnl:+.2f} bps < -{cfg.early_exit_threshold_bps}"
                        )
                        early_exit_triggered = True
                        # 120# A4-2: 中断時点の PnL を保存
                        m.pnl_at_exit_bps = interim_pnl
                        break
                except Exception:
                    continue
            elapsed_monitor = (tick + 1) * monitor_sec if early_exit_triggered else ticks * monitor_sec
            remaining = cfg.post_fill_wait_sec - elapsed_monitor
            # 120# A4-2: EE 発動でも固定30s まで待機して真の post_fill_pnl を取得
            if remaining > 0:
                await asyncio.sleep(remaining)
        else:
            logger.info(f"Waiting {cfg.post_fill_wait_sec}s for PnL measurement...")
            await asyncio.sleep(cfg.post_fill_wait_sec)

        m.actual_measurement_sec = time.time() - t_post_fill_start

        try:
            m.mid_30s_after = await get_mid_price()
        except Exception:
            pass

        if m.mid_at_fill is not None and m.mid_30s_after is not None:
            if side == "buy":
                m.post_fill_pnl = (m.mid_30s_after - m.mid_at_fill) / m.mid_at_fill * _BPS_FACTOR
                m.adverse_selected_raw = m.mid_30s_after < m.mid_at_fill
                m.adverse_selected = m.post_fill_pnl < -cfg.as_deadzone_bps
            else:
                m.post_fill_pnl = (m.mid_at_fill - m.mid_30s_after) / m.mid_at_fill * _BPS_FACTOR
                m.adverse_selected_raw = m.mid_30s_after > m.mid_at_fill
                m.adverse_selected = m.post_fill_pnl < -cfg.as_deadzone_bps

        # early_exit_triggered → 呼び出し側で rapid_exit フラグを設定
        m.early_exit_triggered = early_exit_triggered

        # 047# E3: +30s (=60s) 計測 — 049# サンプリング制御
        do_e3 = m.mid_at_fill is not None and _rng.random() < cfg.e3_sampling_ratio
        if do_e3:
            e3_target_60s = cfg.post_fill_wait_sec * cfg.e3_60s_multiplier
            e3_elapsed = time.time() - t_post_fill_start
            e3_wait_60 = max(0.0, e3_target_60s - e3_elapsed)
            if e3_wait_60 > 0:
                await asyncio.sleep(e3_wait_60)
            try:
                m.mid_60s_after = await get_mid_price()
                if side == "buy":
                    m.post_fill_60s_pnl = (m.mid_60s_after - m.mid_at_fill) / m.mid_at_fill * _BPS_FACTOR
                else:
                    m.post_fill_60s_pnl = (m.mid_at_fill - m.mid_60s_after) / m.mid_at_fill * _BPS_FACTOR
            except Exception:
                pass

            e3_target_120s = cfg.post_fill_wait_sec * cfg.e3_120s_multiplier
            e3_elapsed = time.time() - t_post_fill_start
            e3_wait_120 = max(0.0, e3_target_120s - e3_elapsed)
            if e3_wait_120 > 0:
                await asyncio.sleep(e3_wait_120)
            try:
                m.mid_120s_after = await get_mid_price()
                if side == "buy":
                    m.post_fill_120s_pnl = (m.mid_120s_after - m.mid_at_fill) / m.mid_at_fill * _BPS_FACTOR
                else:
                    m.post_fill_120s_pnl = (m.mid_at_fill - m.mid_120s_after) / m.mid_at_fill * _BPS_FACTOR
            except Exception:
                pass

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
