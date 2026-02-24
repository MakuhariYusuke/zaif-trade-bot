#!/usr/bin/env python3
"""
G1.1-exec Fill Test Runner — 009# §4.2 準拠.

maker limit 注文を発注し、fill rate / queue wait / adverse selection を実測する。

Usage:
  python scripts/v460/run_fill_test.py --hours 24 --dry-run
  python scripts/v460/run_fill_test.py --hours 168              # .env から自動読込
  python scripts/v460/run_fill_test.py --hours 168 --api-key KEY --api-secret SECRET
  python scripts/v460/run_fill_test.py --results-only --results-dir results/v460/fill_test
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ztb.metrics.fill_quality import (
    FillRecord,
    compute_fill_metrics,
    filter_clean_records,
    g1_1_judgment,
    g1_1_quick_judgment,
    g1_2_full_judgment,
    load_fill_records_glob,
    save_fill_records,
)
from ztb.risk.circuit_breakers import KillSwitch
from ztb.trading.live.exchanges.base.broker_interfaces import IBroker
from ztb.trading.live.registry.broker_registry import get_broker_registry
from scripts.v460.lib.adaptation_engine import AdaptationEngine
from scripts.v460.lib.balance_checker import BalanceChecker
from scripts.v460.lib import cancel_reasons as CR  # 145# §9-#6
from scripts.v460.lib.batch_persistence import BatchPersistence
from scripts.v460.lib.fast_fill_defense import FastFillDefense, FastFillDefenseConfig
from scripts.v460.lib.fill_config import (
    FillTestConfig,
    SkipGateResult as _SkipGateResult,
    FillMonitorResult as _FillMonitorResult,
    PnlMeasurement as _PnlMeasurement,
)
from scripts.v460.lib.lot_manager import (
    compute_confidence_lot_factor,
    compute_effective_order_lot,
    resolve_regime_lot_multiplier,
    scale_lot_by_regime,
)
from scripts.v460.lib.maker_price import MakerPriceCalculator
from scripts.v460.lib.ob_recorder import OBRecorder
from scripts.v460.lib.order_monitor import OrderMonitor
from ztb.data.trades_recorder import TradesRecorder
from ztb.data.trades_health import check_trades_health
from ztb.utils.git_utils import get_git_sha as _get_shared_git_sha
from scripts.v460.lib.pnl_measurer import PnlMeasurer
from scripts.v460.lib.resilience import (
    CircuitBreaker,
    CircuitState,
    FillTestHealthMonitor,
    FillTestStatePersistence,
    FillTestState,
    HealthThresholds,
    create_api_circuit_breaker,
)
from scripts.v460.lib.results_analyzer import (
    run_results_only,
    save_judgment,
)
from scripts.v460.lib.side_selector import SideSelector
from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator
from scripts.v460.lib.time_filter import TimeFilter
from scripts.v460.lib.abstract_cycle_runner import AbstractCycleRunner
from scripts.v460.lib.event_logger import log_event as _log_event, TeeWriter as _TeeWriter, setup_stderr_mirror as _setup_stderr_mirror
from scripts.v460.lib.lock_manager import LockManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ======================================================================
# 148# Event Logger — event_logger.py に移動 (158# P2-4)
# _log_event, _TeeWriter, _setup_stderr_mirror は
# scripts.v460.lib.event_logger からインポート
# ======================================================================


# ======================================================================
# Configuration — 119# fill_config.py に移動
# FillTestConfig, SkipGateResult, FillMonitorResult, PnlMeasurement は
# scripts.v460.lib.fill_config からインポート
# ======================================================================


# ======================================================================
# 113# R1: run_single_cycle 分割用 内部データクラス
# 119# fill_config.py に移動 → _SkipGateResult etc. としてインポート
# ======================================================================


# ======================================================================
# Fill Test Runner
# ======================================================================

class FillTestRunner(AbstractCycleRunner):
    """Maker 注文の fill quality を実測する.

    009# §4.2 の設計に準拠.
    145# §10.1-#2: AbstractCycleRunner を継承 (共通インタフェース定義).
    """

    def __init__(
        self,
        adapter: IBroker,
        config: FillTestConfig,
        yaml_cfg: Optional[dict] = None,
    ) -> None:
        self.adapter = adapter
        self.config = config
        self._yaml_cfg = yaml_cfg or {}  # YAML 生の設定 (サブモジュールに渡す用)
        self._results_dir = Path(config.results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._cycle_count = 0
        # 121# SideSelector に side 決定ロジックを委譲
        self._side_selector = SideSelector(config)
        self._preflight_skip_count = 0  # 044# 連続 preflight スキップ計
        # 120# KillSwitch 統合: _shutdown_requested bool → KillSwitch (ztb.risk)
        self._kill_switch = KillSwitch("fill_test")
        self._pending_order_id: Optional[str] = None

        # 020# O4: データバージョン管理
        self._run_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        self._git_sha = self._get_git_sha()

        # 044# 単一起動ロック → 158# P2-4: LockManager に委譲
        self._lock_manager = LockManager(
            self._results_dir,
            self._run_id,
            lock_stale_heartbeat_sec=config.lock_stale_heartbeat_sec,
            lock_acquire_retries=config.lock_acquire_retries,
            lock_heartbeat_period_sec=config.lock_heartbeat_period_sec,
        )

        # 033# 方策 B: 動的ロットの実行時数量 (config.order_quantity を初期値とする)
        # 121# BalanceChecker に残高チェック + ロット管理を委譲
        self._balance_checker = BalanceChecker(config)
        # 046# soft loss_cap 発動済みフラグ (重複半減を防止)
        self._soft_loss_cap_triggered: bool = False
        # 101# §4: soft_cap スナップショット (起動時残高ベース、動的 loss_cap 連動しない)
        self._soft_cap_jpy_snapshot: float | None = None
        # 121# TimeFilter に時間帯フィルターを委譲
        self._time_filter = TimeFilter(config)
        # 100# God Object 分割: FastFillDefense クラスに委譲 (side-aware)
        self._fast_fill_defense = FastFillDefense(
            config=FastFillDefenseConfig(
                enabled=config.fast_fill_defense_enabled,
                threshold_sec=config.fast_fill_threshold_sec,
                threshold_sec_buy=config.fast_fill_threshold_sec_buy,
                threshold_sec_sell=config.fast_fill_threshold_sec_sell,
                offset_boost=config.fast_fill_offset_boost,
                offset_boost_buy=config.fast_fill_offset_boost_buy,
                offset_boost_sell=config.fast_fill_offset_boost_sell,
                max_offset_ratio=config.max_offset_ratio,
                min_offset_ratio=config.min_offset_ratio,
            ),
            base_offset_ratio=config.spread_offset_ratio,
            base_offset_ratio_buy=config.spread_offset_ratio_buy,
            base_offset_ratio_sell=config.spread_offset_ratio_sell,
        )

        # 120# God Object 分割: MakerPriceCalculator に価格算出ロジックを委譲
        self._maker_price = MakerPriceCalculator(
            config=config,
            fast_fill_defense=self._fast_fill_defense,
            regime_detector=None,  # _regime_detector 初期化後に設定
            base_offset_ratio=config.spread_offset_ratio,
            base_offset_ratio_buy=config.spread_offset_ratio_buy,
            base_offset_ratio_sell=config.spread_offset_ratio_sell,
        )

        # 120# God Object 分割: OrderMonitor に約定ポーリングを委譲
        self._order_monitor = OrderMonitor(config)

        # 120# God Object 分割: PnlMeasurer に PnL 計測を委譲
        self._pnl_measurer = PnlMeasurer(config)

        # 037# レジーム検知 (035# §4)
        self._regime_detector: Optional["FillTestRegimeDetector"] = None
        if config.enable_regime:
            from scripts.v460.lib.regime_detector import (
                FillTestRegime,
                FillTestRegimeDetector,
                RegimeConfig,
            )

            regime_cfg = RegimeConfig(
                window=config.regime_window,
                trend_threshold_pct=config.regime_trend_threshold_pct,
                high_vol_multiplier=config.regime_high_vol_multiplier,
                hysteresis_count=config.regime_hysteresis_count,
                min_confidence=config.regime_min_confidence,
            )
            self._regime_detector = FillTestRegimeDetector(regime_cfg)
            # MakerPriceCalculator に regime detector を設定
            self._maker_price._regime_detector = self._regime_detector
            logger.info(
                f"[Regime] detector enabled: window={regime_cfg.window}, "
                f"hysteresis={regime_cfg.hysteresis_count}"
            )

        # 119# BatchPersistence: 保存失敗トラッキング・リトライ・緊急ダンプ
        self._batch_persistence = BatchPersistence(
            results_dir=self._results_dir,
            max_retries=config.max_save_retries,
            save_fail_threshold=config.save_fail_threshold,
            retry_backoff_sec=config.save_retry_backoff_sec,
            flush_interval_sec=config.batch_flush_interval_sec,
        )

        # 120# God Object 分割: AdaptationEngine に適応ロジックを委譲
        self._adaptation_engine = AdaptationEngine(
            config=config,
            yaml_cfg=self._yaml_cfg,
            results_dir=self._results_dir,
        )

        # 121# God Object 分割: SkipGateEvaluator に ML 判定を委譲
        self._skip_gate_evaluator = SkipGateEvaluator(config, _PROJECT_ROOT)
        self._skip_gate = self._skip_gate_evaluator.skip_gate  # OrderMonitor 等互換用

        # 113# resilience: CircuitBreaker / HealthMonitor / StatePersistence
        # 158# YAML 外部化: config から閾値を取得
        self._circuit_breaker = create_api_circuit_breaker(
            failure_threshold=config.cb_failure_threshold,
            recovery_timeout=config.cb_recovery_timeout,
            success_threshold=config.cb_success_threshold,
            timeout=config.cb_timeout,
        )
        self._health_monitor = FillTestHealthMonitor(
            thresholds=HealthThresholds(
                rss_warn_mb=config.hm_rss_warn_mb,
                rss_critical_mb=config.hm_rss_critical_mb,
                disk_free_warn_gb=config.hm_disk_free_warn_gb,
                gc_interval_cycles=config.hm_gc_interval_cycles,
                check_interval_sec=config.hm_check_interval_sec,
            )
        )
        self._state_persistence = FillTestStatePersistence(self._results_dir)

        # 129# OB recorder: 板スナップショットを raw JSONL.gz に蓄積 (retrain_scheduler 用)
        self._ob_recorder = OBRecorder(enabled=True)

        # 135# P0-04: trades recorder — 約定データを raw JSONL.gz に蓄積
        self._trades_recorder = TradesRecorder(enabled=True)

        # 136# P1-03: sell 動的 kill — 独立マネージャに委譲
        from ztb.risk.sell_dynamic_kill import SellDynamicKillManager, SellKillConfig, BuyDynamicKillManager, DynamicKillConfig
        self._sell_kill_mgr = SellDynamicKillManager(SellKillConfig(
            enabled=config.sell_dynamic_kill_enabled,
            window=config.sell_dynamic_kill_window,
            threshold_bps=config.sell_dynamic_kill_threshold_bps,
            resume_window=config.sell_dynamic_kill_resume_window,
            regime_thresholds=config.sell_dynamic_kill_regime_thresholds,  # 139# §9-#2
        ))

        # 157# §19: buy 動的 kill — sell との対称性確保
        self._buy_kill_mgr = BuyDynamicKillManager(DynamicKillConfig(
            enabled=config.buy_dynamic_kill_enabled,
            window=config.buy_dynamic_kill_window,
            threshold_bps=config.buy_dynamic_kill_threshold_bps,
            resume_window=config.buy_dynamic_kill_resume_window,
            regime_thresholds=config.buy_dynamic_kill_regime_thresholds,
        ))

        # 安全設計: atexit + signal で残存注文キャンセル + 未保存データ退避 + ロック解放
        atexit.register(self._cleanup_sync)

        # 151# §13 #2: confidence_lot 起動時ガード — 有効化告知 + effectivity check 注意喚起
        if config.enable_confidence_lot:
            logger.warning(
                "[confidence_lot] ENABLED: scale=%.2f, floor=%.2f, mode=%s. "
                "Ensure effectivity check (no-op ratio < 80%%) was performed "
                "before production use. See 151# §11.3.",
                config.confidence_lot_scale,
                config.confidence_lot_floor,
                config.confidence_lot_mode,
            )
            # 152# §9 P1-6: no-op 検知ガード — order_quantity ≈ min_order_btc なら実質無意味
            if config.order_quantity <= config.min_order_btc * 1.01:
                logger.warning(
                    "[confidence_lot] NO-OP DETECTED: order_quantity (%.4f) ≈ "
                    "min_order_btc (%.4f). Shrink-only design cannot reduce below "
                    "min_order_btc → confidence_lot has zero effect. "
                    "Consider increasing order_quantity or setting enabled=false. "
                    "See 152# §3.5.",
                    config.order_quantity,
                    config.min_order_btc,
                )

        # 137# P1-08: narrow spread pause 連続カウンタ
        self._narrow_spread_consecutive: int = 0

        # 138# P1-10: preflight pause カウンタ (run 内の累積 pause 回数)
        self._preflight_pause_count: int = 0

        # 154# C-2: balance_forced_skip 連続カウンタ (deadlock 防止)
        self._balance_forced_skip_count: int = 0

        # 158# §20-B: trending_sell_skip 連続カウンタ (安全弁)
        self._trending_sell_skip_count: int = 0

        # 044# A-7: loss_cap 更新カウンタ
        self._loss_cap_update_interval = config.loss_cap_update_interval

    # 121# _current_lot プロパティ: BalanceChecker に委譲 (後方互換)
    @property
    def _current_lot(self) -> float:
        return self._balance_checker.current_lot

    @_current_lot.setter
    def _current_lot(self, value: float) -> None:
        self._balance_checker.current_lot = value

    # ──────────────────────────────────────────────────────────────────
    # 145# §9-#5/7: skip_record / cycle_id ヘルパ (DRY)
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _new_cycle_id(prefix: str | None = None) -> str:
        """145# §9-#7: cycle_id 一元生成."""
        base = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        return f"{prefix}_{base}" if prefix else base

    def _make_skip_record(
        self,
        *,
        side: str,
        cancel_reason: str,
        cycle_id: str | None = None,
        order_quantity: float | None = None,
        order_price: float = 0.0,
        spread_at_order: float | None = None,
        spread_offset_ratio: float | None = None,
        regime: str | None = None,
        balance_forced_switch: bool = False,
        **extra: object,
    ) -> FillRecord:
        """145# §9-#5: skip/監査系 FillRecord 一元生成.

        run_id, git_sha, timestamp は自動設定。
        cancel_reason 文字列は cancel_reasons モジュールの定数を使うこと。
        """
        return FillRecord(
            cycle_id=cycle_id or self._new_cycle_id(),
            timestamp=time.time(),
            side=side,
            order_price=order_price,
            order_quantity=order_quantity if order_quantity is not None else self._current_lot,
            cancelled=True,
            cancel_reason=cancel_reason,
            run_id=self._run_id,
            git_sha=self._git_sha,
            spread_at_order=spread_at_order,
            spread_offset_ratio=spread_offset_ratio,
            regime=regime,
            balance_forced_switch=balance_forced_switch,
            ab_test_variant=self.config.ab_test_variant or None,  # 158# P1-5
            **extra,  # type: ignore[arg-type]
        )

    def _current_regime_value(self) -> str | None:
        """160# skip record 用: 現在の確定レジーム文字列.

        _make_skip_record の regime 引数に渡し、cancel 系レコードにも
        レジーム情報を伝搬する。regime=None → 43.9% だった問題を解消。
        """
        if self._regime_detector is None:
            return None
        return self._regime_detector.current_regime.value

    def _get_regime_state_fields(self) -> dict:
        """121# A4: regime state persistence — FillTestState に渡す regime 関連フィールド."""
        if self._regime_detector is None:
            return {}
        st = self._regime_detector.get_state()
        return {
            "regime_confirmed": st["confirmed"],
            "regime_stability": st["stability"],
            "regime_prices": st["prices"],
            "regime_raw_history": st["raw_history"],
        }

    def _regime_lot_multiplier(self) -> float:
        """145# §8-#1: 現在のレジームに対応するロット倍率を返す.

        倍率辞書が空 / レジーム未検出 / 該当なしの場合は 1.0 を返す.
        """
        multipliers = self.config.regime_lot_multipliers
        regime_value: str | None = None
        if self._regime_detector is not None:
            regime = self._regime_detector.current_regime
            regime_value = regime.value if regime is not None else None
        return resolve_regime_lot_multiplier(multipliers, regime_value=regime_value)

    def _regime_adjusted_lot(self) -> float:
        """143# R-1b: レジーム別ロット倍率を適用した注文ロットを返す.

        regime_lot_multipliers が空の場合は _current_lot をそのまま返す.
        倍率適用後も config.min_order_btc 以上を保証.
        """
        base_lot = self._current_lot
        mult = self._regime_lot_multiplier()
        if mult == 1.0:
            return base_lot
        # 倍率適用 + 安全クランプ (144# #2: config.min_order_btc に統一)
        min_lot = self.config.min_order_btc
        adjusted = scale_lot_by_regime(
            base_lot,
            multiplier=mult,
            min_lot=min_lot,
            max_lot=self.config.max_lot,
        )
        if adjusted != base_lot:
            logger.debug(
                f"[regime_lot] mult={mult:.2f} → lot adjusted: "
                f"{base_lot:.4f} × {mult:.2f} = {adjusted:.4f}"
            )
        return adjusted

    # ------------------------------------------------------------------
    # 151# P3-03: AS 確率連動ロットサイジング (confidence_lot)
    # ------------------------------------------------------------------

    def _confidence_lot_factor(
        self,
        as_prob: float | None,
        *,
        dust_sweep_active: bool = False,
    ) -> float:
        """151# P3-03: AS 確率に基づくロット倍率.

        §10 #5: dust_sweep_active 時は factor=1.0 (端数掃除を妨げない).
        §10 #3: sg.as_prob のみ使用 (mode=pnl は凍結).
        §10 #2: 戻り値は [0, 1] にクランプ (縮小専用の不変条件).

        Returns:
            1.0 (無効時 / 確率なし / dust_sweep) or [floor, 1.0] の倍率.
        """
        mode = self.config.confidence_lot_mode
        if self.config.enable_confidence_lot and mode == "pnl":
            # §10 #3/#6 + §13 #1: mode=pnl は凍結。__post_init__ で弾くが防御的二重ガード
            logger.warning("[confidence_lot] mode='pnl' is frozen; treating as 1.0")
        return compute_confidence_lot_factor(
            enabled=self.config.enable_confidence_lot,
            mode=mode,
            as_prob=as_prob,
            scale=self.config.confidence_lot_scale,
            floor=self.config.confidence_lot_floor,
            dust_sweep_active=dust_sweep_active,
        )

    def _effective_order_lot(
        self,
        regime_lot: float,
        as_prob: float | None = None,
        *,
        dust_sweep_active: bool = False,
    ) -> tuple[float, float]:
        """151# 統合ロット算出: regime_lot × confidence_factor.

        §10 #4: regime_lot は呼び出し元から1回算出して渡す (二重経路回避).

        Args:
            regime_lot: _regime_adjusted_lot() の結果.
            as_prob: SkipGate の AS 確率.
            dust_sweep_active: dust sweep アクティブフラグ.

        Returns:
            (effective_lot, confidence_factor) のタプル.
        """
        conf_factor = self._confidence_lot_factor(
            as_prob, dust_sweep_active=dust_sweep_active,
        )
        lot = compute_effective_order_lot(
            regime_lot=regime_lot,
            confidence_factor=conf_factor,
            min_lot=self.config.min_order_btc,
            max_lot=self.config.max_lot,
        )

        if conf_factor < 1.0:
            logger.debug(
                f"[confidence_lot] as_prob={as_prob}, factor={conf_factor:.2f} "
                f"→ lot={lot:.4f} (regime={regime_lot:.4f})"
            )
        return lot, conf_factor

    # 121# _last_side プロパティ: SideSelector に委譲 (後方互換)
    @property
    def _last_side(self) -> str | None:
        return self._side_selector.last_side

    @_last_side.setter
    def _last_side(self, value: str | None) -> None:
        self._side_selector.last_side = value

    @staticmethod
    def _get_git_sha() -> Optional[str]:
        """現在の git commit short hash を取得."""
        sha = _get_shared_git_sha(cwd=_PROJECT_ROOT)
        if sha == "unknown":
            return None
        return sha[:12]

    def resume_from_existing(self) -> list[FillRecord]:
        """既存 fill_records から状態を復元する (レジューム対応).

        中断→再開時に:
          - _cycle_count を復元
          - _last_side を復元 (片側蓄積防止)
          - 既存レコードを返す (結果集計用)
        """
        existing = load_fill_records_glob(str(self._results_dir))
        if not existing:
            return []

        self._cycle_count = len(existing)
        # 最後のレコードの side を復元
        last_record = existing[-1]
        self._last_side = last_record.side
        logger.info(
            f"Resumed from existing records: n={len(existing)}, "
            f"last_side={self._last_side}, cycle_count={self._cycle_count}"
        )
        return existing

    def _next_side(self) -> str:
        """buy/sell を決定 — 121# SideSelector に委譲."""
        return self._side_selector.next(
            imbalance=self._maker_price._last_imbalance,
        )

    def _is_sell_killed(self) -> bool:
        """133# P0-10 / 136# P1-03: sell 動的 kill 判定 — SellDynamicKillManager に委譲.

        §9 #3: 現在レジームを check_kill() に渡し regime_thresholds を有効化。
        """
        regime: str | None = None
        if self._regime_detector is not None:
            regime = self._regime_detector.current_regime.value
        killed, telemetry = self._sell_kill_mgr.check_kill(regime=regime)
        if killed:
            logger.info(
                f"[136# §9] sell kill: regime={regime or 'default'}, "
                f"threshold_used={telemetry.threshold_used}, "
                f"cooldown_remaining={telemetry.cooldown_remaining}"
            )
        return killed

    def _track_sell_pnl(self, record: "FillRecord") -> None:
        """133# P0-10 / 136# P1-03: sell fill の PnL を追跡 — SellDynamicKillManager に委譲."""
        if (
            record.filled
            and record.side == "sell"
            and record.post_fill_30s_pnl is not None
        ):
            self._sell_kill_mgr.track(record.post_fill_30s_pnl)

    def _is_buy_killed(self) -> bool:
        """157# §19: buy 動的 kill 判定 — BuyDynamicKillManager に委譲.

        sell_dynamic_kill の buy 側対称版。
        """
        regime: str | None = None
        if self._regime_detector is not None:
            regime = self._regime_detector.current_regime.value
        killed, telemetry = self._buy_kill_mgr.check_kill(regime=regime)
        if killed:
            logger.info(
                f"[157# §19] buy kill: regime={regime or 'default'}, "
                f"threshold_used={telemetry.threshold_used}, "
                f"cooldown_remaining={telemetry.cooldown_remaining}"
            )
        return killed

    def _track_buy_pnl(self, record: "FillRecord") -> None:
        """157# §19: buy fill の PnL を追跡 — BuyDynamicKillManager に委譲."""
        if (
            record.filled
            and record.side == "buy"
            and record.post_fill_30s_pnl is not None
        ):
            self._buy_kill_mgr.track(record.post_fill_30s_pnl)

    async def _compute_orderbook_imbalance(self, depth: int = 5) -> tuple[float, float, float]:
        """054# S1: 板不均衡を計算 — 120# MakerPriceCalculator に委譲."""
        r = await self._maker_price.compute_imbalance(self.adapter, self.config.symbol, depth=depth)
        return r.imbalance, r.bid_total, r.ask_total

    async def _get_mid_price(self) -> float:
        """板の best bid/ask から mid price を算出 — 120# MakerPriceCalculator に委譲."""
        return await self._maker_price.get_mid_price(self.adapter, self.config.symbol)

    async def _compute_maker_price(self, side: str) -> tuple[float, float, float]:
        """maker limit 価格を算出 — 120# MakerPriceCalculator に委譲."""
        r = await self._maker_price.compute(side, self.adapter, self.config.symbol)
        return r.price, r.spread, r.effective_offset_ratio

    def _is_time_filtered(self, side: str | None = None) -> bool:
        """時間帯フィルター — 121# TimeFilter に委譲."""
        return self._time_filter.is_filtered(side=side)

    # 106# R2: bps 換算定数 (1 bps = 1e-4)
    _BPS_FACTOR: int = 10_000

    async def _check_balance_for_side(
        self, side: str, *, regime_mult: float = 1.0,
    ) -> bool:
        """残高 pre-flight check — 121# BalanceChecker に委譲.

        145# §8-#1: regime_mult を渡してレジーム倍率込みで残高判定.
        """
        return await self._balance_checker.check(
            side, self.adapter, self.config.symbol,
            regime_mult=regime_mult,
        )

    # ------------------------------------------------------------------
    # 158# P2-4: Lock 管理 — LockManager に委譲
    # ------------------------------------------------------------------
    def _acquire_lock(self) -> None:
        """044# 単一起動ロック — LockManager に委譲."""
        self._lock_manager.acquire()

    def _release_lock(self) -> None:
        """044# ロックファイル解放 — LockManager に委譲."""
        self._lock_manager.release()

    def _update_lock_heartbeat(self) -> None:
        """129# heartbeat 更新 — LockManager に委譲."""
        self._lock_manager.update_heartbeat()

    async def _cancel_stale_orders(self) -> int:
        """042# 起動時の滞留注文自動クリア.

        前回プロセスが異常終了した際に残った未約定注文をキャンセルする。
        これにより、303s のポーリング浪費を回避。

        Returns:
            キャンセルした注文数。
        """
        cancelled_count = 0
        try:
            open_orders = await self.adapter.get_open_orders(self.config.symbol)
            if not open_orders:
                logger.info("[startup] No stale orders found.")
                return 0
            for order in open_orders:
                try:
                    await self.adapter.cancel_order(order.order_id)
                    cancelled_count += 1
                    logger.warning(
                        f"[startup] Cancelled stale order: "
                        f"id={order.order_id}, side={order.side}, "
                        f"price={order.price}, qty={order.quantity}"
                    )
                except Exception as e:
                    logger.error(
                        f"[startup] Failed to cancel stale order "
                        f"{order.order_id}: {e}"
                    )
            logger.info(
                f"[startup] Stale order cleanup complete: "
                f"{cancelled_count}/{len(open_orders)} cancelled."
            )
        except Exception as e:
            logger.warning(f"[startup] Stale order check failed (non-fatal): {e}")
        return cancelled_count

    # ==================================================================
    # 113# R1: run_single_cycle から抽出したサブメソッド
    # ==================================================================

    async def _evaluate_skip_gate(
        self,
        side: str,
        cycle_id: str,
        order_price: float,
        spread_at_order: Optional[float],
        effective_offset_ratio: float,
        *,
        order_lot: float | None = None,
    ) -> _SkipGateResult:
        """SkipGate ML 判定 — 121# SkipGateEvaluator に委譲.

        145# §9-#4: order_lot を渡してレジーム倍率適用後のロットで記録.
        """
        regime_value = (
            self._regime_detector.current_regime.value
            if self._regime_detector is not None
            else None
        )
        _lot = order_lot if order_lot is not None else self._current_lot
        return await self._skip_gate_evaluator.evaluate(
            side=side,
            cycle_id=cycle_id,
            order_price=order_price,
            spread_at_order=spread_at_order,
            effective_offset_ratio=effective_offset_ratio,
            adapter=self.adapter,
            symbol=self.config.symbol,
            current_lot=_lot,
            run_id=self._run_id,
            git_sha=self._git_sha,
            regime_value=regime_value,
            last_imbalance=self._maker_price._last_imbalance,
            last_bid_depth=self._maker_price._last_bid_depth,
            last_ask_depth=self._maker_price._last_ask_depth,
            imbalance_enabled=self.config.imbalance_enabled,
            maker_price_vpin_setter=lambda v: setattr(self._maker_price, '_last_vpin', v),
        )

    async def _monitor_fill_polling(
        self,
        order: object,
        order_price: float,
        side: str,
        t_submit: float,
        spread_at_order: Optional[float],
        effective_offset_ratio: float,
        *,
        order_lot: float | None = None,
    ) -> _FillMonitorResult:
        """約定ポーリング監視 — 120# OrderMonitor に委譲.

        120# 型安全: order: Any → object (OrderLike Protocol 準拠)。
        145# fix §9-#1: current_lot → order_lot (regime 調整済みの正しいロット)。
        """
        _lot = order_lot if order_lot is not None else self._current_lot

        def _set_pending(oid: str | None) -> None:
            self._pending_order_id = oid

        return await self._order_monitor.monitor(
            adapter=self.adapter,
            order=order,  # type: ignore[arg-type]
            order_price=order_price,
            side=side,
            t_submit=t_submit,
            spread_at_order=spread_at_order,
            effective_offset_ratio=effective_offset_ratio,
            shutdown_check=self._kill_switch,
            pending_order_setter=_set_pending,
            get_mid_price=self._get_mid_price,
            compute_maker_price=self._compute_maker_price,
            skip_gate=self._skip_gate,
            regime_detector=self._regime_detector,
            current_lot=_lot,
        )

    async def _measure_post_fill_pnl(
        self,
        filled: bool,
        fill_price: Optional[float],
        side: str,
    ) -> _PnlMeasurement:
        """約定後 PnL 計測 — 120# PnlMeasurer に委譲."""
        pnl = await self._pnl_measurer.measure(
            filled=filled,
            fill_price=fill_price,
            side=side,
            get_mid_price=self._get_mid_price,
        )
        # 054# S3: early exit → rapid exit フラグ
        if pnl.early_exit_triggered:
            self._side_selector.set_rapid_exit(side)
        return pnl

    async def run_single_cycle(
        self,
        side_override: str | None = None,
        balance_forced_switch: bool = False,
        balance_forced_rescue: bool = False,
    ) -> FillRecord:
        """1 サイクル: 発注 → 監視 → 結果記録.

        009# §4.2 の流れに準拠.
        041# 時間帯フィルター・残高チェック追加.
        055# Fix: side 決定前に最新 imbalance を取得.
        075# Fix: side_override で run_continuous() が決定した side を強制適用.
        129# D.2: balance_forced_switch フラグを FillRecord に記録.
        158# P1-1: balance_forced_rescue — offset 倍増で安全にポジション解消.
        """
        self._cycle_count += 1
        cycle_id = self._new_cycle_id()

        # 113# resilience: CircuitBreaker ガード — OPEN 中は API 呼出しを回避
        if self._circuit_breaker.state == CircuitState.OPEN:
            if not self._circuit_breaker.should_attempt_reset():
                logger.warning(
                    f"[circuit_breaker] OPEN — skipping cycle {self._cycle_count} "
                    f"(recovery in {self._circuit_breaker.config.recovery_timeout}s)"
                )
                return self._make_skip_record(
                    side=side_override or "buy",
                    cancel_reason=CR.CIRCUIT_BREAKER_OPEN,
                    cycle_id=cycle_id,
                    regime=self._current_regime_value(),  # 160#
                )

        # 055# Fix #2: Smart Side 判定用に最新板 imbalance を事前取得
        # (_compute_maker_price 内での取得では side 決定後 → 1サイクル遅延)
        # 122# §7.3 方法 2: OB データ記録のため常時計算 (smart_side 無効時もデータ蓄積)
        try:
            imb, bid_d, ask_d = await self._compute_orderbook_imbalance(
                depth=self.config.imbalance_depth,
            )
            self._maker_price._last_imbalance = imb
            self._maker_price._last_bid_depth = bid_d
            self._maker_price._last_ask_depth = ask_d
            # 129# OB recorder: サイクルごとに板スナップショットを記録
            ob = self._maker_price._last_ob_snapshot
            if ob is not None:
                self._ob_recorder.record(ob.bids, ob.asks, ob.timestamp)
        except Exception as e:
            logger.warning(f"[ob_prefetch] Pre-fetch imbalance failed, using last: {e}")
            # フォールバック: 前回値を維持

        # 135# P0-04: trades recorder — OB とは独立した try で障害分離 (§9.1 #3)
        try:
            recent = await self.adapter.get_recent_trades(
                self.config.symbol, limit=self.config.trades_recorder_fetch_limit,
            )
            self._trades_recorder.record_from_adapter(recent)
        except Exception as te:
            logger.debug(f"Trades fetch for recording skipped: {te}")

        # 075# Fix: side_override があればそれを使い、_next_side() 二重呼出を防止
        if side_override is not None:
            side = side_override
        else:
            side = self._next_side()
        # 054# S2: 連続同 side カウンタ更新 — 121# SideSelector に委譲
        self._side_selector.update_after_decision(side)

        logger.info(f"=== Cycle {self._cycle_count} ({side}) ===")

        # 1. maker limit 価格算出
        spread_at_order: Optional[float] = None
        effective_offset_ratio: float = self.config.spread_offset_ratio
        try:
            order_price, spread_at_order, effective_offset_ratio = await self._compute_maker_price(side)
        except Exception as e:
            # 130# orderbook_error 細分化
            err_msg = str(e).lower()
            if "timeout" in err_msg or "timed out" in err_msg:
                ob_cancel_reason = "orderbook_timeout"
                logger.error(f"Failed to compute maker price: {e}")
            elif "rate" in err_msg or "limit" in err_msg or "too many" in err_msg:
                ob_cancel_reason = "orderbook_rate_limit"
                logger.error(f"Failed to compute maker price: {e}")
            elif "empty" in err_msg or "no bid" in err_msg or "no ask" in err_msg:
                ob_cancel_reason = "orderbook_empty"
                logger.error(f"Failed to compute maker price: {e}")
            elif "sell_guard" in err_msg:
                ob_cancel_reason = "sell_guard_reject"
                logger.warning(f"Maker price rejected: {e}")
            elif "spread too narrow" in err_msg:
                # 158# §20-D: spread_too_narrow を専用分類 — ERROR→INFO 降格
                ob_cancel_reason = "spread_too_narrow"
                logger.info(f"[158# §20-D] {e}")
            else:
                ob_cancel_reason = "orderbook_error"
                logger.error(f"Failed to compute maker price: {e}")
            # 155# §9.5 #3: orderbook_error 時に前回 mid_price をフォールバック
            # 156# §10 #5: 鮮度判定 — 閾値超は stale とみなす
            # 156# §16 review: _prev_mid_* 直接アクセス → 公開メソッド化
            _fb_price, _fb_time = self._maker_price.get_fallback_price()
            _fallback_price = _fb_price or 0.0
            _fallback_age: float | None = None
            _fallback_stale = False
            _stale_sec = self.config.fallback_stale_sec
            if _fallback_price > 0 and _fb_time is not None:
                _fallback_age = time.time() - _fb_time
                _fallback_stale = _fallback_age > _stale_sec
                logger.info(
                    f"[155# ob_fallback] Using last mid_price={_fallback_price:.0f} "
                    f"age={_fallback_age:.1f}s stale={_fallback_stale} "
                    f"as reference for skip record"
                )
            elif _fallback_price > 0:
                # no timestamp → stale 扱い (安全策)
                _fallback_stale = True
                logger.info(
                    f"[155# ob_fallback] Using last mid_price={_fallback_price:.0f} "
                    f"(no timestamp, treated as stale) as reference for skip record"
                )
            return self._make_skip_record(
                side=side,
                cancel_reason=ob_cancel_reason,
                cycle_id=cycle_id,
                order_price=_fallback_price if not _fallback_stale else 0.0,
                spread_offset_ratio=self._maker_price.base_offset_ratio,
                error_message=(
                    f"{e} [fallback_age={_fallback_age:.1f}s stale={_fallback_stale}]"
                    if _fallback_age is not None else str(e)
                ),
                regime=self._current_regime_value(),  # 160#
            )

        # 113# R1: SkipGate 判定を _evaluate_skip_gate() に委譲
        # 158# P1-1: balance_forced rescue → offset 倍増で安全にポジション解消
        if balance_forced_rescue and effective_offset_ratio > 0:
            _rescue_mult = self.config.balance_forced_rescue_offset_mult
            _pre_rescue_offset = effective_offset_ratio
            effective_offset_ratio = min(
                effective_offset_ratio * _rescue_mult,
                self.config.max_offset_ratio,
            )
            # 価格を rescue offset で再計算
            if spread_at_order is not None and spread_at_order > 0:
                mid_est = order_price + (spread_at_order * _pre_rescue_offset / 2 if side == "buy"
                                         else -spread_at_order * _pre_rescue_offset / 2)
                if side == "buy":
                    order_price = mid_est - spread_at_order * effective_offset_ratio / 2
                else:
                    order_price = mid_est + spread_at_order * effective_offset_ratio / 2
                order_price = round(order_price)
            logger.info(
                f"[158# P1-1] balance_forced_rescue: offset "
                f"{_pre_rescue_offset:.4f}→{effective_offset_ratio:.4f} "
                f"(×{_rescue_mult:.1f}), price={order_price:.0f}"
            )

        # 137# P1-08: spread 狭小時の「休む」判定
        if (
            self.config.narrow_spread_pause_enabled
            and spread_at_order is not None
            and order_price > 0
        ):
            mid_est = order_price  # 近似: maker price ≈ mid
            spread_bps_val = spread_at_order / mid_est * self._BPS_FACTOR if mid_est > 0 else 0.0
            if spread_bps_val < self.config.narrow_spread_pause_bps:
                self._narrow_spread_consecutive += 1
                if self._narrow_spread_consecutive <= self.config.narrow_spread_pause_max_consecutive:
                    pause_sec = self.config.narrow_spread_pause_sec
                    logger.info(
                        f"[137# P1-08] Spread too narrow ({spread_bps_val:.1f}bps "
                        f"< {self.config.narrow_spread_pause_bps}bps). "
                        f"Pausing {pause_sec}s "
                        f"({self._narrow_spread_consecutive}/{self.config.narrow_spread_pause_max_consecutive})"
                    )
                    # 139# §9-#3: 実際に待機してから FillRecord を返す
                    await asyncio.sleep(pause_sec)
                    return self._make_skip_record(
                        side=side,
                        cancel_reason=CR.NARROW_SPREAD_PAUSE,
                        cycle_id=cycle_id,
                        order_price=order_price,
                        spread_at_order=spread_at_order,
                        spread_offset_ratio=effective_offset_ratio,
                        regime=self._current_regime_value(),  # 160#
                    )
            else:
                self._narrow_spread_consecutive = 0

        # 151# §10 #4: regime_lot を1回だけ算出し、SkipGate/発注/記録へ共通引き回し
        _regime_lot = self._regime_adjusted_lot()

        sg = await self._evaluate_skip_gate(
            side, cycle_id, order_price, spread_at_order, effective_offset_ratio,
            order_lot=_regime_lot,
        )
        skip_gate_skipped = sg.skipped
        skip_gate_score = sg.score
        skip_gate_reason = sg.reason
        skip_gate_model_used = sg.model_used
        skip_gate_as_prob = sg.as_prob
        skip_gate_threshold_used = sg.threshold_used
        skip_gate_hour_offset = sg.hour_offset if sg.hour_offset != 0.0 else None
        if sg.early_return_record is not None:
            return sg.early_return_record

        # 2. 発注 (CM-2: リトライ付き)
        t_submit = time.time()
        order = None
        last_error: Optional[str] = None
        cancel_reason: str = "unknown"  # 032# #6: ループ未実行時の NameError 防止

        # 105#: lot floor guard — 121# BalanceChecker に委譲
        self._balance_checker.apply_lot_floor()

        # 143# R-1b + 151# P3-03: regime × confidence (per-cycle, _current_lot には永続化しない)
        # 145# fix: §8-#2 乗法的複利と §8-#3 片側更新を修正
        # §10 #5: dust_sweep 時は confidence_factor=1.0
        _order_lot, _confidence_factor = self._effective_order_lot(
            _regime_lot,
            as_prob=skip_gate_as_prob,
            dust_sweep_active=self._balance_checker.dust_sweep_active,
        )

        for attempt in range(1 + self.config.max_order_retries):
            try:
                # 130# E1: postonly 二重確認 — 発注直前に mid price を再取得し
                # テイカー側になっていないか確認 (postonly_reject 低減)
                try:
                    _pre_ob = await self.adapter.get_orderbook(self.config.symbol, depth=1)
                    if _pre_ob and _pre_ob.bids and _pre_ob.asks:
                        from scripts.v460.lib.ob_utils import best_bid_ask
                        _pre_best_bid, _pre_best_ask = best_bid_ask(_pre_ob)
                        # buy の指値が best_ask 以上 → テイカー側
                        if side == "buy" and order_price >= _pre_best_ask:
                            _safe_price = _pre_best_bid
                            logger.info(
                                f"[postonly_guard] 130# buy price {order_price:.0f} >= best_ask "
                                f"{_pre_best_ask:.0f}, adjusted to best_bid {_safe_price:.0f}"
                            )
                            order_price = _safe_price
                        # sell の指値が best_bid 以下 → テイカー側
                        elif side == "sell" and order_price <= _pre_best_bid:
                            _safe_price = _pre_best_ask
                            logger.info(
                                f"[postonly_guard] 130# sell price {order_price:.0f} <= best_bid "
                                f"{_pre_best_bid:.0f}, adjusted to best_ask {_safe_price:.0f}"
                            )
                            order_price = _safe_price
                except Exception as _pre_e:
                    logger.debug(f"[postonly_guard] Pre-check failed (non-fatal): {_pre_e}")

                order = await self.adapter.place_order(
                    symbol=self.config.symbol,
                    side=side,
                    quantity=_order_lot,
                    price=order_price,
                    order_type="limit",
                )
                self._pending_order_id = order.order_id
                logger.info(
                    f"Placed {side} limit @ {order_price:.0f} JPY, "
                    f"qty={_order_lot}, id={order.order_id}"
                    + (f" (retry {attempt})" if attempt > 0 else "")
                )
                break
            except Exception as e:
                last_error = str(e)
                # CM-2: エラー分類
                err_lower = last_error.lower()
                if "post_only" in err_lower or "taker" in err_lower:
                    cancel_reason = "post_only_reject"
                elif (
                    "insufficient" in err_lower
                    or "balance" in err_lower
                    # 042# Coincheck の日本語エラーメッセージ対応 — 121# YAML 外部化
                    or any(p in last_error for p in self.config.insufficient_funds_patterns)
                ):
                    cancel_reason = "insufficient_funds"
                elif "minimum" in err_lower or "size" in err_lower:
                    cancel_reason = "minimum_size"
                else:
                    cancel_reason = "api_error"

                logger.warning(
                    f"Order attempt {attempt + 1} failed ({cancel_reason}): {e}"
                )

                # 046# Bug10: 残高不足はリトライ不要 (2s 待っても残高は回復しない)
                # 084# post_only_reject もリトライ不要 (価格がスプレッド交差済み)
                _non_retriable = {"insufficient_funds", "post_only_reject", "minimum_size"}
                if cancel_reason in _non_retriable:
                    logger.info(
                        f"[Bug10] Skipping retry — {cancel_reason} is not retriable"
                    )
                    break

                if attempt < self.config.max_order_retries:
                    # 084# 指数バックオフ: 2s → 4s → 8s (rate-limit 緩和) — 121# YAML 外部化
                    _backoff = self.config.retry_delay_sec * (self.config.retry_backoff_base ** attempt)
                    # rate-limit 検出時はさらに延長
                    if "rate" in err_lower or "limit" in err_lower or "too many" in err_lower:
                        _backoff = max(_backoff, self.config.rate_limit_min_backoff_sec)
                        logger.warning(f"Rate-limit detected, extended backoff: {_backoff:.1f}s")
                    else:
                        logger.info(f"Retry backoff: {_backoff:.1f}s")
                    await asyncio.sleep(_backoff)
                    try:
                        ob = await self.adapter.get_orderbook(self.config.symbol, depth=1)
                        if ob.bids and ob.asks:
                            # 保守的価格: best_bid/best_ask そのまま (確実に maker)
                            order_price = ob.bids[0][0] if side == "buy" else ob.asks[0][0]
                            logger.info(f"Retry with conservative price: {order_price:.0f}")
                    except Exception:
                        pass  # 板取得失敗時は前回価格でリトライ

        if order is None:
            logger.error(f"All order attempts failed: {last_error}")
            # 113# resilience: API 失敗を CircuitBreaker に記録
            await self._circuit_breaker.async_on_failure()
            return FillRecord(
                cycle_id=cycle_id,
                timestamp=t_submit,
                side=side,
                order_price=order_price,
                order_quantity=_order_lot,
                cancelled=True,
                cancel_reason=cancel_reason,
                error_message=last_error,  # 031# エラー詳細を記録
                spread_at_order=spread_at_order,
                spread_offset_ratio=effective_offset_ratio,  # 096# 計算済み実効値
                run_id=self._run_id,       # 088# データ品質: エラー時も必須
                git_sha=self._git_sha,     # 088# quarantine 防止
                regime=self._current_regime_value(),  # 160#
            )

        # 113# R1: ポーリング監視 + 未約定キャンセルを _monitor_fill_polling() に委譲
        monitor = await self._monitor_fill_polling(
            order, order_price, side, t_submit, spread_at_order, effective_offset_ratio,
            order_lot=_order_lot,
        )
        filled = monitor.filled
        fill_price = monitor.fill_price
        queue_wait = monitor.queue_wait
        cancel_reason_poll = monitor.cancel_reason
        reprice_count = monitor.reprice_count
        reprice_drift_bps = monitor.reprice_drift_bps  # 158# P1-3
        order_price = monitor.final_order_price  # stale reprice で変更される場合
        _effective_timeout = monitor.effective_timeout  # 145# §9-#2

        # 113# R1: PnL 計測を _measure_post_fill_pnl() に委譲
        pnl = await self._measure_post_fill_pnl(filled, fill_price, side)
        mid_at_fill = pnl.mid_at_fill
        mid_30s_after = pnl.mid_30s_after
        mid_60s_after = pnl.mid_60s_after
        mid_120s_after = pnl.mid_120s_after
        post_fill_pnl = pnl.post_fill_pnl
        post_fill_60s_pnl = pnl.post_fill_60s_pnl
        post_fill_120s_pnl = pnl.post_fill_120s_pnl
        adverse_selected = pnl.adverse_selected
        adverse_selected_raw = pnl.adverse_selected_raw
        actual_measurement_sec = pnl.actual_measurement_sec

        # 037# レジーム検知更新 (035# §7 Week 1)
        regime_str: Optional[str] = None
        regime_conf: Optional[float] = None
        regime_stab: Optional[int] = None
        # 156# §18: データシンク解消 — trend_pct/volatility_ratio を FillRecord へ
        regime_trend_pct: Optional[float] = None
        regime_vol_ratio: Optional[float] = None
        if self._regime_detector is not None:
            # 100# P1-6 fix: unfilled 時は order_price (offset 込み) ではなく
            # 直近の真の mid price を使用。order_price は offset を含むため
            # regime 検知のノイズ源となる。
            if mid_at_fill is not None:
                regime_price = mid_at_fill
            else:
                _fb_price, _ = self._maker_price.get_fallback_price()
                regime_price = _fb_price  # None if unavailable

            if regime_price is not None:
                regime_result = self._regime_detector.update(t_submit, regime_price)
                regime_str = regime_result.regime.value
                regime_conf = regime_result.confidence
                regime_stab = regime_result.stability
                regime_trend_pct = regime_result.trend_pct
                regime_vol_ratio = regime_result.volatility_ratio

        record = FillRecord(
            cycle_id=cycle_id,
            timestamp=t_submit,
            side=side,
            order_price=order_price,
            order_quantity=_order_lot,
            fill_price=fill_price,
            filled=filled,
            cancelled=not filled,
            queue_wait_sec=queue_wait,
            mid_at_fill=mid_at_fill,
            mid_30s_after=mid_30s_after,
            mid_60s_after=mid_60s_after,
            mid_120s_after=mid_120s_after,
            post_fill_30s_pnl=post_fill_pnl,
            post_fill_60s_pnl=post_fill_60s_pnl,
            post_fill_120s_pnl=post_fill_120s_pnl,
            adverse_selected=adverse_selected,
            adverse_selected_raw=adverse_selected_raw,
            cancel_reason=(
                cancel_reason_poll
                if cancel_reason_poll
                else (
                    "timeout"
                    # 145# §9-#2: regime 調整済みの effective_timeout を使用
                    if (not filled and queue_wait >= (_effective_timeout or self.config.order_timeout_sec))
                    else ("unknown" if not filled else None)  # 117# C-fix: None 防止
                )
            ),
            run_id=self._run_id,
            git_sha=self._git_sha,
            # 031# 追加フィールド
            spread_at_order=spread_at_order,
            spread_offset_ratio=effective_offset_ratio,  # 050# Bug#3 fix: 実効値を記録
            # 037# レジーム情報
            regime=regime_str,
            regime_confidence=regime_conf,
            regime_stability=regime_stab,
            # 156# §18: データシンク解消
            regime_trend_pct=regime_trend_pct,
            regime_volatility_ratio=regime_vol_ratio,
            # 054# S5: AS 予測データ基盤
            # 122# R5/§7.3 方法 2: OB 記録を imbalance_enabled と独立させ常時記録
            orderbook_imbalance=self._maker_price._last_imbalance,
            bid_depth_total=self._maker_price._last_bid_depth,
            ask_depth_total=self._maker_price._last_ask_depth,
            mid_price_trend_5s=self._maker_price._last_mid_trend_bps,
            spread_bps=(
                (spread_at_order / mid_at_fill * self._BPS_FACTOR)
                if spread_at_order is not None and mid_at_fill is not None and mid_at_fill > 0
                else None
            ),
            effective_offset_used=effective_offset_ratio,
            # 062# SkipGate 判定情報 (PASS 時も記録 → 後続分析用)
            skip_gate_skipped=skip_gate_skipped,
            skip_gate_score=skip_gate_score,
            skip_gate_reason=skip_gate_reason,
            skip_gate_model_used=skip_gate_model_used,
            # 084# P(AS) 可観測性改善
            skip_gate_as_prob=skip_gate_as_prob,
            skip_gate_threshold_used=skip_gate_threshold_used,
            # 158# P1-6: 時間帯別 skip_gate 閾値調整のオフセット
            skip_gate_hour_offset=skip_gate_hour_offset,
            # 094# stale order cancel-replace 追跡
            reprice_count=reprice_count,
            # 158# P1-3: reprice 累積 drift (bps)
            reprice_drift_bps=reprice_drift_bps if reprice_count > 0 else None,
            # 100# P1-4: 実際の PnL 計測経過秒数
            actual_measurement_sec=actual_measurement_sec if filled else None,
            # 120# A4: Early Exit 明示フラグ
            early_exit_triggered=pnl.early_exit_triggered if filled else None,
            # 120# A4-2: EE 中断時点 PnL (計測バイアス分離)
            pnl_at_exit_bps=pnl.pnl_at_exit_bps if filled else None,
            # 120# P2-1: 寄与分解基盤 — FFD/VG イベントフラグ
            ffd_boost_active=self._fast_fill_defense.is_boost_active(side),
            vg_triggered=self._maker_price.last_vg_triggered,
            # 158# P2-6: VG 詳細ログ (ヒンドサイト分析用)
            vg_velocity_bps=self._maker_price.last_vg_velocity_bps,
            vg_vpin=self._maker_price.last_vg_vpin,
            vg_boost_factor=self._maker_price.last_vg_boost_factor,
            # 129# D.2: 残高制約による side 強制切替フラグ
            balance_forced_switch=balance_forced_switch or None,
            # 151# P3-03: confidence lot 可観測性 (§10 #7)
            confidence_lot_factor=_confidence_factor if self.config.enable_confidence_lot else None,
            order_lot_regime=_regime_lot,
            order_lot_effective=_order_lot,
            confidence_lot_mode=self.config.confidence_lot_mode if self.config.enable_confidence_lot else None,
            # 158# P1-5: A/B テスト variant 識別子
            ab_test_variant=self.config.ab_test_variant or None,
        )

        logger.info(
            f"Cycle {self._cycle_count} result: "
            f"filled={filled}, wait={queue_wait:.1f}s, "
            f"pnl={post_fill_pnl:.2f}bps" if post_fill_pnl is not None
            else f"Cycle {self._cycle_count} result: filled={filled}, wait={queue_wait:.1f}s"
        )

        # 113# resilience: API 成功を CircuitBreaker に記録
        await self._circuit_breaker.async_on_success()

        return record

    async def run_continuous(self, hours: float) -> list[FillRecord]:
        """指定時間、連続してサイクルを実行.

        009# §4.4: 7 日間 (168h) の実測想定.
        中断→再開時は既存 fill_records を自動復元 (レジューム対応).

        024# R1-R4: 保存失敗耐性・例外分離・メモリ制御を強化.
        032# P0: 方策 A パラメータ適応統合.
        033# 方策 B: 動的ロットサイジング統合.
        033# F4: 累積 PnL 安全キャップ (000# §3.9).
        """
        end_time = time.time() + hours * 3600

        # 044# 単一起動ロック取得
        self._acquire_lock()

        # 135# P2-09→P1: trades データ健全性チェック
        try:
            th = check_trades_health(lookback_days=3, stale_threshold_hours=36.0)
            if not th.healthy:
                logger.warning(f"[trades_health] {th.message}")
                if th.missing_days:
                    logger.warning(
                        "[trades_health] retrain 品質が低下する可能性あり。"
                        "fill_test 内蔵 TradesRecorder の動作状態を確認してください"
                    )
                # 148# P1: trades stale 自動イベント記録
                # 159# §2.1 fix: latest_ts/age_hours → available_days[-1]/stale_hours
                _log_event(
                    "trades_health_alert",
                    self._results_dir,
                    run_id=self._run_id,
                    git_sha=self._git_sha,
                    reason=f"trades unhealthy: {th.message}",
                    details={
                        "healthy": th.healthy,
                        "latest_day": th.available_days[-1] if th.available_days else None,
                        "missing_days": th.missing_days,
                        "stale_hours": round(th.stale_hours, 1),
                    },
                )
            else:
                logger.info(f"[trades_health] {th.message}")
        except Exception as e:
            logger.warning(f"[trades_health] check failed: {e}")

        # 041# 動的 loss_cap: API 残高から算出
        if self.config.loss_cap_auto:
            await self._update_dynamic_loss_cap()

        # 101# §4: soft_cap スナップショット — 起動時の残高ベースで固定
        # 動的 loss_cap_jpy が変動しても soft_cap は連動しない
        self._soft_cap_jpy_snapshot = (
            self.config.loss_cap_jpy
            * self.config.soft_loss_cap_ratio
            / self.config.loss_cap_ratio
        )

        # 042# 起動時の滞留注文クリア (前回プロセスの残注文防止)
        await self._cancel_stale_orders()

        # レジューム: 既存レコードから状態復元
        existing_records = self.resume_from_existing()
        # 046# clean/quarantine 分離: ゾンビプロセス由来レコードを除外して集計
        clean_records, quarantine_records = filter_clean_records(existing_records)
        if quarantine_records:
            logger.warning(
                f"[quarantine] {len(quarantine_records)} records excluded from "
                f"PnL computation (blank git_sha)"
            )

        # 088# schema health check: run_id / git_sha の自己検証
        if not self._run_id or not self._run_id.strip():
            logger.error("[schema_health] CRITICAL: run_id is empty — data quality at risk")
        if not self._git_sha or not self._git_sha.strip():
            logger.error("[schema_health] CRITICAL: git_sha is empty — records will be quarantined")
        else:
            logger.info(
                f"[schema_health] OK: run_id={self._run_id}, git_sha={self._git_sha}, "
                f"clean={len(clean_records)}, quarantine={len(quarantine_records)}"
            )
        # 024# O4: メモリ制御 — 全レコード保持ではなくカウンタのみ
        total_count = len(existing_records)  # 全件カウント (quarantine 含む)
        filled_count = sum(1 for r in existing_records if r.filled)

        # 033# F4: レジューム時の累積 PnL 計算 (クリーンレコードのみ)
        cumulative_pnl_jpy = 0.0
        for r in clean_records:
            if r.filled and r.post_fill_30s_pnl is not None and r.fill_price:
                cumulative_pnl_jpy += (
                    r.post_fill_30s_pnl / self._BPS_FACTOR * r.fill_price * r.order_quantity
                )

        # 101# §2: soft_loss_cap_triggered をレジューム復元
        # 前回 run 中に soft cap 発動していた場合、再起動で False に戻ると
        # 二重ロット半減が発生する。cumulative_pnl_jpy から論理的に判定。
        if existing_records and self.config.loss_cap_auto:
            soft_cap_jpy = (
                self.config.loss_cap_jpy
                * self.config.soft_loss_cap_ratio
                / self.config.loss_cap_ratio
            )
            if cumulative_pnl_jpy <= -soft_cap_jpy:
                self._soft_loss_cap_triggered = True
                logger.info(
                    f"[resume] soft_loss_cap already triggered: "
                    f"cumPnL={cumulative_pnl_jpy:.0f} JPY <= -{soft_cap_jpy:.0f} JPY"
                )
        # 101# P1-5: regime detector warm-up — 既存レコードの mid price で初期化
        # window=20 に対して再起動後 20 サイクルは判定不安定になるため、
        # レジューム時の既存レコード (直近 window*3 件) で事前投入する。
        # 121# A4: StatePersistence から regime state を優先復元 (warm-up より正確)
        regime_restored = False
        if self._regime_detector is not None:
            saved_state = self._state_persistence.load()
            if saved_state is not None and saved_state.regime_prices:
                regime_restored = self._regime_detector.restore_state({
                    "confirmed": saved_state.regime_confirmed,
                    "stability": saved_state.regime_stability,
                    "prices": saved_state.regime_prices,
                    "raw_history": saved_state.regime_raw_history or [],
                })

        if self._regime_detector is not None and existing_records and not regime_restored:
            # fallback: 旧方式の warm-up (state 復元失敗時)
            filled_with_mid = [
                r for r in existing_records
                if r.filled and r.mid_at_fill is not None
            ]
            # window*multiplier (バッファ上限に合わせる) の直近分だけ投入
            warmup_window = self._regime_detector.config.window * self.config.regime_warmup_multiplier
            warmup_records = filled_with_mid[-warmup_window:]
            for r in warmup_records:
                assert r.mid_at_fill is not None  # filtered above
                self._regime_detector.update(r.timestamp, r.mid_at_fill)
            if warmup_records:
                logger.info(
                    f"[regime] warm-up (fallback): fed {len(warmup_records)} records, "
                    f"regime={self._regime_detector.current_regime.value}"
                )

        del existing_records, clean_records, quarantine_records  # メモリ解放

        batch: list[FillRecord] = self._batch_persistence.take_unsaved()  # 前回未保存分を引き継ぐ
        batch_size = self.config.batch_size  # 032# #18: 設定化

        # 148# P0: heartbeat 更新タスク — stale 誤判定防止
        async def _heartbeat_loop() -> None:
            """lock heartbeat を周期的に更新."""
            while not self._kill_switch.is_killed():
                self._update_lock_heartbeat()
                await asyncio.sleep(self.config.lock_heartbeat_period_sec)

        heartbeat_task = asyncio.create_task(_heartbeat_loop())

        logger.info(f"Starting fill test: {hours}h, interval={self.config.cycle_interval_sec}s")

        while time.time() < end_time and not self._kill_switch.is_killed():
            # 129# D.2: 残高制約による side 強制切替追跡
            _balance_forced = False
            _is_rescue = False  # 158# P1-1: balance_forced rescue フラグ
            # 073# side 別時間帯フィルター: side 決定後にフィルタリング
            # side 別リスト未設定時はグローバルリスト (041# 互換)
            next_side = self._next_side()

            # side 別チェック (073#): side固有リストがあれば side 別判定
            side_filtered = self._is_time_filtered(side=next_side)
            if side_filtered:
                # 反対 side でもフィルタされるか確認
                alt_side = "sell" if next_side == "buy" else "buy"
                alt_filtered = self._is_time_filtered(side=alt_side)
                if alt_filtered:
                    # 両 side ともフィルタ → スリープ
                    # 140# §8.1-#2: skip record を生成し可観測性確保 (132# F4)
                    if not self._time_filter.in_filter:
                        self._time_filter.on_enter()
                        batch.append(self._make_skip_record(
                            side=next_side,
                            cancel_reason=CR.TIME_FILTER_BOTH_SIDES,
                            order_quantity=0.0,
                            regime=self._current_regime_value(),  # 160#
                        ))
                    else:
                        # 079# heartbeat: 長時間抑制中にプロセス生存を定期ログ
                        now_ts = time.time()
                        if now_ts - self._time_filter.last_heartbeat_time >= self.config.heartbeat_interval_sec:
                            utc_h = datetime.now(timezone.utc).hour
                            try:
                                import psutil  # lazy import
                                proc = psutil.Process()
                                mem_mb = proc.memory_info().rss / (1024 * 1024)
                                mem_info = f"mem={mem_mb:.1f}MB, "
                            except Exception:
                                mem_info = ""
                            logger.info(
                                f"[heartbeat] Still in time_filter zone "
                                f"(UTC {utc_h}h), "
                                f"{mem_info}"
                                f"unsaved_batch={len(batch)}, "
                                f"cycles={self._cycle_count}"
                            )
                            self._time_filter.last_heartbeat_time = now_ts
                            # 129# lock heartbeat 更新
                            self._update_lock_heartbeat()
                        # 107# R1: 重複 flush → _maybe_flush_batch 統合
                        batch = self._batch_persistence.maybe_flush(batch, "time_filter")
                    await asyncio.sleep(self.config.cycle_interval_sec)
                    continue
                else:
                    # 反対 side は通過 → side 切り替え
                    # 086# Bug: alt_side が _last_side と同じ場合、片側蓄積が発生する
                    # (例: _last_side=buy, next=sell がブロック, alt=buy → double buy)
                    # この場合は両方ブロックと同じ扱いにして待機する
                    if alt_side == self._last_side:
                        self._time_filter.consecutive_086_wait += 1
                        max_wait = self.config.max_086_consecutive_wait
                        utc_h = datetime.now(timezone.utc).hour
                        # 110# デッドロック解除: 連続待機が上限を超えたら alt_side を許可
                        if max_wait > 0 and self._time_filter.consecutive_086_wait > max_wait:
                            logger.info(
                                f"[time_filter] 086# deadlock break: "
                                f"{self._time_filter.consecutive_086_wait} consecutive waits "
                                f"exceeded max={max_wait}, allowing {alt_side} "
                                f"(110# デッドロック解除)"
                            )
                            self._time_filter.consecutive_086_wait = 0
                            next_side = alt_side
                            # ↓ alt_side 許可 → 通常フローに合流
                        else:
                            logger.info(
                                f"[time_filter] {next_side} filtered at UTC {utc_h}h, "
                                f"alt={alt_side} would repeat last side → "
                                f"treating as both-filtered "
                                f"(086# 片側蓄積防止, wait={self._time_filter.consecutive_086_wait}/{max_wait})"
                            )
                            if not self._time_filter.in_filter:
                                self._time_filter.on_enter()
                                # 140# §8.1-#2: 086 deadlock 進入時も record 生成
                                batch.append(self._make_skip_record(
                                    side=next_side,
                                    cancel_reason=CR.TIME_FILTER_086_DEADLOCK,
                                    order_quantity=0.0,
                                    regime=self._current_regime_value(),  # 160#
                                ))
                            # 107# R1: 重複 flush → _maybe_flush_batch 統合
                            batch = self._batch_persistence.maybe_flush(batch, "alt_side==last_side wait")
                            await asyncio.sleep(self.config.cycle_interval_sec)
                            continue
                    else:
                        # 086# ではない通常の side 切り替え → カウンタリセット
                        self._time_filter.consecutive_086_wait = 0
                    utc_h = datetime.now(timezone.utc).hour
                    logger.debug(
                        f"[time_filter] {next_side} filtered at UTC {utc_h}h, "
                        f"switching to {alt_side}"
                    )
                    next_side = alt_side

            # 047# Issue12: 離脱時のみログ出力
            self._time_filter.on_exit()

            # 158# §20-A: メインループ毎にレジーム更新 — skip パスでも遷移を保証
            # Root cause fix: trending_sell_skip 等のスキップパスでは
            # run_single_cycle が呼ばれず regime_detector.update() が滞留→デッドロック
            # fallback price (直近 OB mid) を毎ループ投入して遷移機会を確保
            if self._regime_detector is not None:
                _fb_price, _fb_time = self._maker_price.get_fallback_price()
                if _fb_price is not None:
                    _pre_regime = self._regime_detector.current_regime
                    _regime_result = self._regime_detector.update(
                        time.time(), _fb_price
                    )
                    if _regime_result.regime != _pre_regime:
                        logger.info(
                            f"[158# §20-A] Regime transition in main loop: "
                            f"{_pre_regime.value} → {_regime_result.regime.value} "
                            f"(stability={_regime_result.stability}, "
                            f"trend_pct={_regime_result.trend_pct:.4f})"
                        )

            # 041# 残高 pre-flight check: 不足サイドはスキップ
            # 145# §8-#1: レジーム倍率込みで残高判定 (preflight-lot alignment)
            _regime_mult = self._regime_lot_multiplier()
            if await self._check_balance_for_side(next_side, regime_mult=_regime_mult):
                # 091# 即座に反対 side を試す: time_filter との組合せで停滞するのを防止
                opposite = "sell" if next_side == "buy" else "buy"
                tried_opposite = False
                if not await self._check_balance_for_side(opposite, regime_mult=_regime_mult):
                    # 反対 side は残高 OK → 即座に切替
                    logger.info(
                        f"[balance] {next_side} insufficient, "
                        f"switching to {opposite} immediately (091#)"
                    )
                    # 120# A5: 不足 side を N サイクル凍結 (API 呼出し節約)
                    # 158# YAML 外部化: balance_freeze_cycles
                    self._side_selector.freeze_side(
                        next_side, cycles=self.config.balance_freeze_cycles,
                    )
                    next_side = opposite
                    self._last_side = opposite  # 次回は再び元の side
                    self._preflight_skip_count = 0
                    tried_opposite = True
                    _balance_forced = True  # 129# D.2

                if not tried_opposite:
                    # 両 side とも残高不足 → 従来通りの処理
                    self._last_side = next_side  # → 次の _next_side() が反対を返す
                    self._preflight_skip_count += 1

                    # 140# §8.1-#2: preflight skip record 生成 (132# F4)
                    # 145# §9-#5: _make_skip_record DRY 化
                    batch.append(self._make_skip_record(
                        side=next_side,
                        cancel_reason=CR.PREFLIGHT_INSUFFICIENT,
                        order_quantity=self._current_lot,
                        regime=self._current_regime_value(),  # 160#
                    ))
                    # 107# R1: 重複 flush → _maybe_flush_batch 統合
                    batch = self._batch_persistence.maybe_flush(batch, "preflight skip")

                    # 051# P2-3: Balance auto-shrink — 連続失敗でロット縮小を試行
                    # 052#: 最低ロットを min_order_btc に統一 (Coincheck 0.001 BTC)
                    min_lot = max(self.config.order_quantity, self.config.min_order_btc)
                    if (
                        self._preflight_skip_count >= self.config.balance_shrink_consecutive
                        and not self._balance_checker.balance_shrink_active
                        and self._current_lot > min_lot
                    ):
                        old_lot = self._current_lot
                        # 105#: 0.001 BTC 単位に切り捨て (浮動小数点丸め誤差 → API 400 防止)
                        raw_shrunk = self._current_lot / self.config.balance_shrink_divisor
                        _mob = self.config.min_order_btc
                        self._current_lot = max(
                            min_lot,
                            int(raw_shrunk / _mob) * _mob,
                        )
                        self._balance_checker.balance_shrink_active = True
                        logger.warning(
                            f"[balance_shrink] 連続 preflight 失敗 {self._preflight_skip_count} 回. "
                            f"ロット縮小: {old_lot:.4f} → {self._current_lot:.4f} BTC"
                        )
                        # カウンタリセットして縮小ロットで再試行
                        self._preflight_skip_count = 0
                        await asyncio.sleep(self.config.cycle_interval_sec)
                        continue

                    # 138# P1-10: preflight pause — SAFE_STOP 前に一時停止で回復を待つ
                    if (
                        self.config.preflight_pause_enabled
                        and self._preflight_skip_count >= self.config.preflight_pause_threshold
                        and self._preflight_pause_count < self.config.preflight_max_pauses
                    ):
                        self._preflight_pause_count += 1
                        pause_sec = self.config.preflight_pause_sec
                        logger.warning(
                            f"[preflight_pause] 連続 preflight 失敗 {self._preflight_skip_count} 回 "
                            f"(閾値 {self.config.preflight_pause_threshold}). "
                            f"pause #{self._preflight_pause_count}/{self.config.preflight_max_pauses} "
                            f"→ {pause_sec:.0f}s 待機後に再開"
                        )
                        # 140# §8.1-#1: batch.append 導線に統一 (undefined _append_fill_record 修正)
                        # 143# 140§7 #2: cycle_id に timestamp 付与で一意化
                        # 145# §9-#5: _make_skip_record DRY 化
                        batch.append(self._make_skip_record(
                            side="none",
                            cancel_reason=CR.PREFLIGHT_PAUSE,
                            cycle_id=f"preflight_pause_{self._preflight_pause_count}_{int(time.time())}",
                            order_quantity=0.0,
                            regime=self._current_regime_value(),  # 160#
                        ))
                        batch = self._batch_persistence.maybe_flush(batch, "preflight_pause")
                        self._preflight_skip_count = 0
                        await asyncio.sleep(pause_sec)
                        continue

                    # 044# F8: 連続 preflight 失敗上限 → SAFE_STOP
                    if self._preflight_skip_count >= self.config.max_preflight_skip:
                        logger.error(
                            f"SAFE_STOP: 連続 preflight スキップ {self._preflight_skip_count} 回 "
                            f"(上限 {self.config.max_preflight_skip}). "
                            f"buy/sell 両方で残高不足の可能性. 停止します."
                        )
                        self._kill_switch.kill("preflight_skip_exceeded")
                        break
                    await asyncio.sleep(self.config.cycle_interval_sec)
                    continue

            # preflight 成功 → カウンタリセット
            self._preflight_skip_count = 0
            # 051# P2-3: 成功時に balance_shrink を解除し、ロットを原値に復元
            self._balance_checker.restore_lot_on_success()
            # 120# A5: 残高回復 → freeze 解除
            self._side_selector.unfreeze_side()

            # --- サイクル実行 ---
            # 133# P0-08: balance_forced_switch 時のハードスキップ
            # 154# C-1/C-2: deadlock 防止 — 片側残高のみの場合は forced でも実行許可
            if _balance_forced and self.config.skip_balance_forced:
                # 154# C-1: 元の side (forced 前) も残高不足かチェック
                # → 両方不足 = preflight で弾かれるはずなので到達しない
                # → 元 side だけ不足 = forced side が唯一の選択肢 → 実行すべき
                original_side = "buy" if next_side == "sell" else "sell"
                original_also_insufficient = await self._check_balance_for_side(
                    original_side, regime_mult=_regime_mult
                )
                # 154# C-2: 連続 forced skip カウンタによるフォールバック
                _deadlock_limit = self.config.balance_forced_deadlock_limit
                _over_deadlock_limit = (
                    _deadlock_limit > 0
                    and self._balance_forced_skip_count >= _deadlock_limit
                )

                if original_also_insufficient or _over_deadlock_limit:
                    # 片側しか取引できない or デッドロック上限超過 → 実行許可
                    _reason = (
                        "one_sided_balance" if original_also_insufficient
                        else f"deadlock_limit({self._balance_forced_skip_count})"
                    )
                    logger.info(
                        f"[154# C-1] balance_forced but {_reason} — "
                        f"proceeding with {next_side} (original_side={original_side} "
                        f"insufficient={original_also_insufficient})"
                    )
                    self._balance_forced_skip_count = 0
                    # → continue しない: run_single_cycle へ進む
                elif self.config.balance_forced_rescue_enabled:
                    # 158# P1-1: rescue モード — skip せず offset 倍増で安全実行
                    self._balance_forced_skip_count = 0
                    _is_rescue = True  # run_single_cycle に渡すフラグ
                    logger.info(
                        f"[158# P1-1] balance_forced rescue mode: "
                        f"executing {next_side} with offset ×"
                        f"{self.config.balance_forced_rescue_offset_mult:.1f} "
                        f"(was consecutive skip={self._balance_forced_skip_count})"
                    )
                    # → continue しない: run_single_cycle へ進む (rescue=True)
                else:
                    # 両方残高 OK → 従来通りスキップ (forced switch は損失回避のため)
                    self._balance_forced_skip_count += 1
                    logger.info(
                        f"[133# P0-08] Skipping cycle — balance_forced_switch=True "
                        f"(avg -1.98bps loss). side={next_side}, "
                        f"consecutive={self._balance_forced_skip_count}"
                    )
                    # 145# §9-#5: _make_skip_record DRY 化
                    _skip_record = self._make_skip_record(
                        side=next_side,
                        cancel_reason=CR.BALANCE_FORCED_SKIP,
                        order_quantity=self._current_lot,
                        balance_forced_switch=True,
                        balance_forced_consecutive=self._balance_forced_skip_count,
                        regime=self._current_regime_value(),  # 160#
                    )
                    batch.append(_skip_record)
                    total_count += 1
                    batch = self._batch_persistence.maybe_flush(batch, "balance_forced_skip")
                    await asyncio.sleep(self.config.cycle_interval_sec)
                    continue

            # 133# P0-09: unknown regime での buy スキップ
            # 156# §12: balance_forced 時はバイパス (片側残高で buy するしかない局面)
            if (
                self.config.skip_buy_unknown_regime
                and next_side == "buy"
                and not _balance_forced
                and self._regime_detector is not None
                and self._regime_detector.current_regime.value == "unknown"
            ):
                logger.info(
                    f"[133# P0-09] Skipping buy — unknown regime "
                    f"(avg -1.384bps loss)"
                )
                # 145# §9-#5: _make_skip_record DRY 化
                _skip_record = self._make_skip_record(
                    side="buy",
                    cancel_reason=CR.UNKNOWN_REGIME_BUY_SKIP,
                    order_quantity=self._current_lot,
                    regime="unknown",
                )
                batch.append(_skip_record)
                total_count += 1
                batch = self._batch_persistence.maybe_flush(batch, "unknown_buy_skip")
                await asyncio.sleep(self.config.cycle_interval_sec)
                continue

            # 155# §9: trending レジーム時の sell 抑制
            # trending sell avg -0.687 bps — 逆行トレード削減
            # 156# §10 #1: balance_forced 時はバイパス (片側残高で sell するしかない局面)
            # 156# D-4: trending 方向別分解 — trending_down sell は開放可能
            # 158# §20-B: 連続 skip 安全弁 — max_consecutive 超過で強制許可
            if (
                self.config.skip_sell_trending
                and next_side == "sell"
                and not _balance_forced
                and self._regime_detector is not None
                and self._regime_detector.current_regime.is_trending
            ):
                # D-4: trending_up_only モードならば trending_down は通過させる
                _current_regime = self._regime_detector.current_regime
                _should_skip = True
                if (
                    self.config.skip_sell_trending_up_only
                    and _current_regime.value == "trending_down"
                ):
                    logger.debug(
                        "[156# D-4] Allowing sell in trending_down regime"
                    )
                    _should_skip = False

                # 158# §20-B: 連続 skip 安全弁
                _max_consec = self.config.max_consecutive_trending_sell_skip
                if (
                    _should_skip
                    and _max_consec > 0
                    and self._trending_sell_skip_count >= _max_consec
                ):
                    logger.warning(
                        f"[158# §20-B] Trending sell skip safety valve: "
                        f"{self._trending_sell_skip_count} consecutive skips "
                        f">= limit {_max_consec} — forcing sell execution"
                    )
                    self._trending_sell_skip_count = 0
                    _should_skip = False

                if _should_skip:
                    self._trending_sell_skip_count += 1
                    logger.info(
                        f"[155# §9] Skipping sell — {_current_regime.value} regime "
                        f"(avg -0.687bps loss) "
                        f"[consecutive={self._trending_sell_skip_count}"
                        f"/{_max_consec if _max_consec > 0 else '∞'}]"
                    )
                    _skip_record = self._make_skip_record(
                        side="sell",
                        cancel_reason=CR.TRENDING_SELL_SKIP,
                        order_quantity=self._current_lot,
                        regime=_current_regime.value,
                    )
                    batch.append(_skip_record)
                    total_count += 1
                    batch = self._batch_persistence.maybe_flush(batch, "trending_sell_skip")
                    await asyncio.sleep(self.config.cycle_interval_sec)
                    continue
                else:
                    # skip しない → カウンタリセット
                    self._trending_sell_skip_count = 0

            # 157# §19: buy 動的 kill — rolling PnL が閾値以下なら buy 停止
            # balance_forced 時はバイパス (sell_dynamic_kill と対称)
            if (
                self.config.buy_dynamic_kill_enabled
                and next_side == "buy"
                and not _balance_forced
                and self._is_buy_killed()
            ):
                logger.info(
                    f"[157# §19] Skipping buy — rolling PnL below "
                    f"{self.config.buy_dynamic_kill_threshold_bps}bps"
                )
                _skip_record = self._make_skip_record(
                    side="buy",
                    cancel_reason=CR.BUY_DYNAMIC_KILL,
                    order_quantity=self._current_lot,
                    regime=self._current_regime_value(),  # 160#
                )
                batch.append(_skip_record)
                total_count += 1
                batch = self._batch_persistence.maybe_flush(batch, "buy_dynamic_kill")
                await asyncio.sleep(self.config.cycle_interval_sec)
                continue

            # 133# P0-10: sell 動的 kill — rolling PnL が閾値以下なら sell 停止
            # 156# §12: balance_forced 時はバイパス (片側残高で sell するしかない局面)
            if (
                self.config.sell_dynamic_kill_enabled
                and next_side == "sell"
                and not _balance_forced
                and self._is_sell_killed()
            ):
                logger.info(
                    f"[133# P0-10] Skipping sell — rolling PnL below "
                    f"{self.config.sell_dynamic_kill_threshold_bps}bps"
                )
                # 145# §9-#5: _make_skip_record DRY 化
                _skip_record = self._make_skip_record(
                    side="sell",
                    cancel_reason=CR.SELL_DYNAMIC_KILL,
                    order_quantity=self._current_lot,
                    regime=self._current_regime_value(),  # 160#
                )
                batch.append(_skip_record)
                total_count += 1
                batch = self._batch_persistence.maybe_flush(batch, "sell_dynamic_kill")
                await asyncio.sleep(self.config.cycle_interval_sec)
                continue

            try:
                record = await self.run_single_cycle(
                    side_override=next_side,
                    balance_forced_switch=_balance_forced,
                    balance_forced_rescue=_is_rescue,
                )
                # 154# C-2: 実サイクル実行 → forced skip カウンタリセット
                self._balance_forced_skip_count = 0
                # 158# §20-B: 実サイクル実行 → trending sell skip カウンタリセット
                self._trending_sell_skip_count = 0
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt — stopping gracefully")
                self._kill_switch.kill("keyboard_interrupt")
                break
            except Exception as e:
                # 024# R2: 例外分類 — サイクル実行エラーは継続可能
                logger.error(f"Cycle execution error: {e}", exc_info=True)
                # 128# 例外時も dust sweep ロットを復元
                self._balance_checker.restore_lot_after_dust_sweep()
                await asyncio.sleep(self.config.cycle_interval_sec)
                continue

            # 128# dust sweep 後のロット復元 (サイクル完了ごとに確実に実行)
            self._balance_checker.restore_lot_after_dust_sweep()

            total_count += 1
            if record.filled:
                filled_count += 1
                # 133# P0-10: sell PnL 追跡 (動的 kill 判定用)
                self._track_sell_pnl(record)
                # 157# §19: buy PnL 追跡 (動的 kill 判定用)
                self._track_buy_pnl(record)
                # 033# F4: 累積 PnL インクリメンタル追跡
                if record.post_fill_30s_pnl is not None and record.fill_price:
                    cumulative_pnl_jpy += (
                        record.post_fill_30s_pnl / self._BPS_FACTOR
                        * record.fill_price * record.order_quantity
                    )
            batch.append(record)

            # --- 046# soft/hard 二段 loss_cap ---
            # soft cap: ロット半減 (一度だけ)
            # 101# §4: _soft_cap_jpy_snapshot を使用 (動的 loss_cap_jpy に連動させない)
            if self.config.loss_cap_auto and not self._soft_loss_cap_triggered:
                if self._soft_cap_jpy_snapshot is not None:
                    soft_cap_jpy = self._soft_cap_jpy_snapshot
                else:
                    soft_cap_jpy = (
                        self.config.loss_cap_jpy
                        * self.config.soft_loss_cap_ratio
                        / self.config.loss_cap_ratio
                    )
                if cumulative_pnl_jpy <= -soft_cap_jpy:
                    old_lot = self._current_lot
                    self._current_lot = max(
                        self.config.order_quantity,  # 最小ロットは下回らない
                        self._current_lot / self.config.soft_loss_cap_lot_divisor,
                    )
                    self._soft_loss_cap_triggered = True
                    # 051# P2-3: shrink 復元先も更新
                    self._balance_checker.pre_shrink_lot = self._current_lot
                    logger.warning(
                        f"[loss_cap] SOFT CAP: cumPnL={cumulative_pnl_jpy:.0f} JPY "
                        f"<= -{soft_cap_jpy:.0f} JPY "
                        f"({self.config.soft_loss_cap_ratio:.0%}). "
                        f"ロット半減: {old_lot:.4f} → {self._current_lot:.4f} BTC"
                    )

            # hard cap: SAFE_STOP (既存 033# F4)
            if cumulative_pnl_jpy <= -self.config.loss_cap_jpy:
                logger.error(
                    f"LOSS CAP REACHED (HARD): cumulative PnL = {cumulative_pnl_jpy:.0f} JPY "
                    f"(cap = -{self.config.loss_cap_jpy:.0f} JPY). Stopping fill test."
                )
                self._kill_switch.kill("hard_loss_cap")

            # --- 100# 即約定防御: FastFillDefense クラスに委譲 ---
            # P0-5: side-aware (sell boost が buy に伝播しない)
            # P0-3: two-layer neg_edge detection (即時 proxy + post-fill PnL)
            # P1-2: side 別 base_offset_ratio による cap
            if record.filled:
                self._fast_fill_defense.evaluate_fill(
                    side=record.side,
                    queue_wait_sec=record.queue_wait_sec,
                    fill_price=record.fill_price,
                    mid_at_fill=record.mid_at_fill,
                    post_fill_pnl_bps=record.post_fill_30s_pnl,
                )
            elif not record.filled:
                self._fast_fill_defense.reset_on_unfilled(record.side)

            # --- バッチ保存 (024# R1: 独立 try/except) ---
            if len(batch) >= batch_size:
                if self._batch_persistence.try_save_batch(batch):
                    batch = []
                    self._batch_persistence.reset_flush_timer()
                    self._adaptation_engine.invalidate_cache()  # 120# TTL キャッシュ無効化
                # 失敗時: batch は保持 → 次回再試行
            # 079# 時間ベース定期flush: batch_size 未満でも一定時間経過で保存
            else:
                batch = self._batch_persistence.maybe_flush(batch, "run_loop")

            # 進捗ログ
            if self._cycle_count % self.config.progress_log_interval == 0:
                regime_tag = (
                    self._regime_detector.current_regime.value
                    if self._regime_detector else "n/a"
                )
                logger.info(
                    f"Progress: {self._cycle_count} cycles, "
                    f"fill rate={filled_count}/{total_count} "
                    f"({filled_count/total_count*100:.1f}%), "
                    f"cumPnL={cumulative_pnl_jpy:.1f}JPY, "
                    f"lot={self._current_lot:.4f}BTC, "
                    f"regime={regime_tag}, "
                    f"unsaved_batch={len(batch)}"
                )

            # 113# resilience: HealthMonitor 定期チェック + GC
            health_status = self._health_monitor.maybe_check(self._cycle_count)
            if health_status and health_status.get("level") == "critical":
                logger.error(
                    f"[resilience] Health CRITICAL at cycle {self._cycle_count}: "
                    f"{health_status}"
                )
            self._health_monitor.maybe_gc()

            # 113# resilience: 状態永続化 (progress_log_interval ごと)
            if self._cycle_count % self.config.progress_log_interval == 0:
                # 129# lock heartbeat 更新 (state 保存と同期)
                self._update_lock_heartbeat()
                self._state_persistence.save(FillTestState(
                    run_id=self._run_id,
                    cycle_count=self._cycle_count,
                    total_count=total_count,
                    filled_count=filled_count,
                    cumulative_pnl_jpy=cumulative_pnl_jpy,
                    current_lot=self._current_lot,
                    soft_loss_cap_triggered=self._soft_loss_cap_triggered,
                    base_offset_ratio=self._maker_price.base_offset_ratio,
                    base_offset_ratio_buy=self._maker_price.base_offset_ratio_buy,
                    base_offset_ratio_sell=self._maker_price.base_offset_ratio_sell,
                    **self._get_regime_state_fields(),
                ))

            # --- 044# A-7: loss_cap 定期更新 (残高変動を反映) ---
            if (
                self.config.loss_cap_auto
                and self._cycle_count % self._loss_cap_update_interval == 0
                and self._cycle_count > 0
            ):
                await self._update_dynamic_loss_cap()

            # --- 032# P0: 方策 A パラメータ適応 ---
            if (
                self.config.enable_auto_adapt
                and self._cycle_count % self.config.adapt_interval_cycles == 0
                and total_count >= self.config.min_adapt_samples
            ):
                self._try_auto_adapt(total_count, filled_count)

            # --- 033# 方策 B: 動的ロットサイジング ---
            if (
                self.config.enable_dynamic_lot
                and self._cycle_count % self.config.lot_adapt_interval_cycles == 0
                and total_count >= self.config.min_adapt_samples
            ):
                self._try_auto_lot_size()

            # 次サイクルまで待機
            # 054# S3: rapid exit 時は interval を短縮
            if time.time() < end_time and not self._kill_switch.is_killed():
                if self._side_selector.rapid_exit_side is not None:
                    interval = self.config.early_exit_rapid_interval_sec
                    logger.info(
                        f"[early_exit] Rapid exit: interval shortened to "
                        f"{interval:.0f}s (next side={self._side_selector.rapid_exit_side})"
                    )
                else:
                    interval = self.config.cycle_interval_sec
                await asyncio.sleep(interval)

        # 残りバッチを保存
        if batch:
            if not self._batch_persistence.try_save_batch(batch):
                # 最終手段: 緊急ダンプ
                self._batch_persistence.emergency_dump(batch, "final")

        # 113# resilience: 最終状態保存
        self._state_persistence.save(FillTestState(
            run_id=self._run_id,
            cycle_count=self._cycle_count,
            total_count=total_count,
            filled_count=filled_count,
            cumulative_pnl_jpy=cumulative_pnl_jpy,
            current_lot=self._current_lot,
            soft_loss_cap_triggered=self._soft_loss_cap_triggered,
            base_offset_ratio=self._maker_price.base_offset_ratio,
            base_offset_ratio_buy=self._maker_price.base_offset_ratio_buy,
            base_offset_ratio_sell=self._maker_price.base_offset_ratio_sell,
            **self._get_regime_state_fields(),
        ))

        # 148# heartbeat タスク終了
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

        logger.info(
            f"Fill test completed: {total_count} cycles, "
            f"{filled_count} filled"
        )
        # 024# O4: 集計用に全レコードをリロード
        return load_fill_records_glob(str(self._results_dir))

    def _build_adapt_kwargs(self) -> dict:
        """120# AdaptationEngine に委譲."""
        return self._adaptation_engine._build_adapt_kwargs()

    def _build_lot_kwargs(self) -> dict:
        """120# AdaptationEngine に委譲."""
        return self._adaptation_engine._build_lot_kwargs()

    async def _update_dynamic_loss_cap(self) -> None:
        """041# 動的 loss_cap — 120# AdaptationEngine に委譲."""
        await self._adaptation_engine.update_dynamic_loss_cap(
            self.adapter, self.config.symbol,
        )

    def _try_auto_adapt(self, total_count: int, filled_count: int) -> None:
        """032# P0: 方策 A — 120# AdaptationEngine に委譲."""
        result = self._adaptation_engine.try_auto_adapt(
            total_count=total_count,
            filled_count=filled_count,
            base_offset_ratio=self._maker_price.base_offset_ratio,
            base_offset_ratio_buy=self._maker_price.base_offset_ratio_buy,
            base_offset_ratio_sell=self._maker_price.base_offset_ratio_sell,
            regime_detector=self._regime_detector,
            fast_fill_defense=self._fast_fill_defense,
        )
        # offset 変更を MakerPriceCalculator に反映
        if result.base_offset_changed or result.buy_offset_changed or result.sell_offset_changed:
            self._maker_price.update_base_offsets(
                result.new_base_offset,
                result.new_buy_offset,
                result.new_sell_offset,
            )

    def _try_auto_lot_size(self) -> None:
        """033# 方策 B — 120# AdaptationEngine に委譲."""
        changed, new_lot = self._adaptation_engine.try_auto_lot_size(
            self._current_lot,
            regime_detector=self._regime_detector,
        )
        if changed:
            self._current_lot = new_lot

    def _cleanup_sync(self) -> None:
        """atexit: 残存注文キャンセル + 未保存データ退避 + ロック解放 (同期 wrapper).

        024# R1: 未保存バッチを緊急ダンプに退避.
        044# A-4: 残存注文キャンセルを確実に実行.
        044# Bug7: ロックファイルを解放.
        129# OB recorder: 最終 flush.
        """
        # 129# OB recorder: バッファ残を書き出し
        try:
            n = self._ob_recorder.flush()
            if n:
                logger.info(f"OB recorder: flushed {n} snapshots on exit")
        except Exception as e:
            logger.error(f"OB recorder final flush failed: {e}")

        # 135# P0-04: trades recorder 最終 flush
        try:
            n_tr = self._trades_recorder.flush()
            if n_tr:
                logger.info(f"Trades recorder: flushed {n_tr} trades on exit")
        except Exception as e:
            logger.error(f"Trades recorder final flush failed: {e}")

        # 未保存バッチの退避
        unsaved = self._batch_persistence.unsaved_batch
        if unsaved:
            logger.warning(
                f"Saving {len(unsaved)} unsaved records on exit"
            )
            self._batch_persistence.emergency_dump(unsaved, "atexit")
            self._batch_persistence.take_unsaved()  # クリア

        # 044# A-4: 残存注文のキャンセル (確実に await する)
        if self._pending_order_id:
            logger.warning(f"Cleaning up pending order: {self._pending_order_id}")
            try:
                # 新しいイベントループで確実に実行
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(
                        self.adapter.cancel_order(self._pending_order_id)
                    )
                    logger.info(f"Cancelled pending order: {self._pending_order_id}")
                finally:
                    loop.close()
            except Exception as e:
                logger.error(f"Cleanup failed: {e}")

        # 044# Bug7: ロックファイル解放
        self._release_lock()


# 119# run_results_only / save_judgment は results_analyzer.py に移動済み

# ======================================================================
# CLI — 158# P2-4: fill_test_cli.py に移動
# ======================================================================

def main() -> None:
    """Fill Test CLI エントリポイント — fill_test_cli.py に委譲."""
    from scripts.v460.lib.fill_test_cli import fill_test_main
    fill_test_main()


if __name__ == "__main__":
    main()

