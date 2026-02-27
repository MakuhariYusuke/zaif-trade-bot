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
from scripts.v460.lib.fill_record_helpers import FillRecordHelpersMixin  # 163#
from scripts.v460.lib.fill_cycle_executor import FillCycleExecutorMixin  # 163#
from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin  # 163#

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

class FillTestRunner(
    FillRecordHelpersMixin,
    FillCycleExecutorMixin,
    FillLoopOrchestratorMixin,
    AbstractCycleRunner,
):
    """Maker 注文の fill quality を実測する.

    009# §4.2 の設計に準拠.
    145# §10.1-#2: AbstractCycleRunner を継承 (共通インタフェース定義).
    163# Mixin 分割: メソッドは責務別に3つの Mixin に分散。
      - FillRecordHelpersMixin: skip record / lot / regime ヘルパー
      - FillCycleExecutorMixin: run_single_cycle + OB/SkipGate/PnL
      - FillLoopOrchestratorMixin: run_continuous + kill/filter/adapt
    WARNING: __init__ と property のみこのファイルに残す。
    新メソッド追加は適切な Mixin ファイルに行うこと。

    ╔══════════════════════════════════════════════════════════════╗
    ║  ⚠ GOD OBJECT 化 禁止 — AI コーディングエージェント向け警告  ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  このクラスは 163# で 2231→378 行に分割済み。               ║
    ║  新しいメソッドをここに追加してはならない。                  ║
    ║  責務に応じて以下の Mixin にメソッドを追加すること:         ║
    ║    • fill_record_helpers.py  (pure helpers, MAX 300 行)     ║
    ║    • fill_cycle_executor.py  (単一サイクル, MAX 700 行)     ║
    ║    • fill_loop_orchestrator.py (連続ループ, MAX 1100 行)   ║
    ║  このファイルは __init__ + property のみ。                  ║
    ║  行数上限: 400 行。超過時は新 Mixin を作成すること。        ║
    ╚══════════════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        adapter: IBroker,
        config: FillTestConfig,
        yaml_cfg: Optional[dict] = None,
        config_yaml_path: Optional[str] = None,  # 169# config hot-reload
    ) -> None:
        self.adapter = adapter
        self.config = config
        self._yaml_cfg = yaml_cfg or {}  # YAML 生の設定 (サブモジュールに渡す用)
        self._config_yaml_path = config_yaml_path  # 169# hot-reload 用 YAML パス
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

        # 168# §4.1 #3: 日次ドローダウンガード
        from scripts.v460.lib.daily_drawdown_guard import DailyDrawdownGuard
        self._daily_drawdown_guard = DailyDrawdownGuard(
            enabled=config.daily_drawdown_enabled,
            hard_limit_bps=config.daily_drawdown_hard_limit_bps,
            soft_limit_bps=config.daily_drawdown_soft_limit_bps,
        )

        # 044# A-7: loss_cap 更新カウンタ
        self._loss_cap_update_interval = config.loss_cap_update_interval

        # 169# Config Hot-Reload: YAML 変更のライブ反映
        from scripts.v460.lib.config_hot_reload import ConfigHotReloader
        self._config_reloader = ConfigHotReloader(
            config=config,
            yaml_path=self._config_yaml_path,
            yaml_cfg=self._yaml_cfg,
            check_interval_sec=config.hot_reload_check_interval_sec,
        )

    # 121# _current_lot プロパティ: BalanceChecker に委譲 (後方互換)
    @property
    def _current_lot(self) -> float:
        return self._balance_checker.current_lot

    @_current_lot.setter
    def _current_lot(self, value: float) -> None:
        self._balance_checker.current_lot = value


    # ==================================================================
    # 169# Config Hot-Reload: コンポーネント再構築コールバック
    # ConfigHotReloader がフィールド変更検出時に呼び出す
    # ==================================================================

    def _rebuild_sell_kill_mgr(self) -> None:
        """sell_dynamic_kill 設定変更時にマネージャを再構築."""
        from ztb.risk.sell_dynamic_kill import SellDynamicKillManager, SellKillConfig
        self._sell_kill_mgr = SellDynamicKillManager(SellKillConfig(
            enabled=self.config.sell_dynamic_kill_enabled,
            window=self.config.sell_dynamic_kill_window,
            threshold_bps=self.config.sell_dynamic_kill_threshold_bps,
            resume_window=self.config.sell_dynamic_kill_resume_window,
            regime_thresholds=self.config.sell_dynamic_kill_regime_thresholds,
        ))

    def _rebuild_buy_kill_mgr(self) -> None:
        """buy_dynamic_kill 設定変更時にマネージャを再構築."""
        from ztb.risk.sell_dynamic_kill import BuyDynamicKillManager, DynamicKillConfig
        self._buy_kill_mgr = BuyDynamicKillManager(DynamicKillConfig(
            enabled=self.config.buy_dynamic_kill_enabled,
            window=self.config.buy_dynamic_kill_window,
            threshold_bps=self.config.buy_dynamic_kill_threshold_bps,
            resume_window=self.config.buy_dynamic_kill_resume_window,
            regime_thresholds=self.config.buy_dynamic_kill_regime_thresholds,
        ))

    def _rebuild_daily_drawdown_guard(self) -> None:
        """daily_drawdown 設定変更時にガードを再構築 (状態継承)."""
        from scripts.v460.lib.daily_drawdown_guard import DailyDrawdownGuard
        old_state = self._daily_drawdown_guard.export_state()
        self._daily_drawdown_guard = DailyDrawdownGuard(
            enabled=self.config.daily_drawdown_enabled,
            hard_limit_bps=self.config.daily_drawdown_hard_limit_bps,
            soft_limit_bps=self.config.daily_drawdown_soft_limit_bps,
        )
        if old_state:
            self._daily_drawdown_guard.import_state(old_state)

    def _rebuild_fast_fill_defense(self) -> None:
        """fast_fill_defense 設定変更時に再構築."""
        from scripts.v460.lib.fast_fill_defense import FastFillDefense, FastFillDefenseConfig
        self._fast_fill_defense = FastFillDefense(
            config=FastFillDefenseConfig(
                enabled=self.config.fast_fill_defense_enabled,
                threshold_sec=self.config.fast_fill_threshold_sec,
                threshold_sec_buy=self.config.fast_fill_threshold_sec_buy,
                threshold_sec_sell=self.config.fast_fill_threshold_sec_sell,
                offset_boost=self.config.fast_fill_offset_boost,
                offset_boost_buy=self.config.fast_fill_offset_boost_buy,
                offset_boost_sell=self.config.fast_fill_offset_boost_sell,
                max_offset_ratio=self.config.max_offset_ratio,
                min_offset_ratio=self.config.min_offset_ratio,
            ),
            base_offset_ratio=self.config.spread_offset_ratio,
            base_offset_ratio_buy=self.config.spread_offset_ratio_buy,
            base_offset_ratio_sell=self.config.spread_offset_ratio_sell,
        )

    # ==================================================================
    # 163# Mixin 分割: 以下のメソッドは個別ファイルに抽出済み
    # - fill_record_helpers.py: _make_skip_record, regime/lot/side helpers
    # - fill_cycle_executor.py: run_single_cycle + OB/SkipGate/PnL
    # - fill_loop_orchestrator.py: run_continuous + kill/filter/adapt
    # God Object 化防止: 新メソッドは適切な mixin に追加すること
    # ==================================================================



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
