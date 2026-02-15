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

import argparse
import asyncio
import atexit
import json
import logging
import logging.handlers
import os
import platform
import signal
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
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
    load_fill_records_glob,
    save_fill_records,
)
from ztb.trading.live.exchanges.coincheck.adapter import CoincheckAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class FillTestConfig:
    """Fill test runner の設定.

    優先順位: CLI引数 > YAML > dataclass defaults.
    """

    symbol: str = "btc_jpy"
    order_quantity: float = 0.001  # 初期ロット (Coincheck BTC 最小)
    cycle_interval_sec: float = 120.0  # サイクル間隔
    order_timeout_sec: float = 300.0  # 注文タイムアウト
    poll_interval_sec: float = 5.0  # ポーリング間隔
    post_fill_wait_sec: float = 30.0  # 約定後 PnL 計測待ち
    results_dir: str = "results/v460/fill_test"
    # 044# 連続 preflight 失敗上限 (buy/sell 両方不足で無限スキップ防止)
    max_preflight_skip: int = 10
    # 開始サイド: JPY 残高不足時は "sell" で開始すると自己資金循環できる
    start_side: str = "buy"
    # CM-1: スプレッド比例オフセット (post_only リジェクト防止)
    spread_offset_ratio: float = 0.05  # 031# 0.2→0.05: AS低減のため保守化
    min_offset_jpy: float = 1.0  # 最小オフセット (JPY)
    # CM-2: 注文失敗リトライ
    max_order_retries: int = 1  # 失敗時のリトライ回数
    retry_delay_sec: float = 2.0  # リトライ間隔
    # CM-3: AS 判定デッドゾーン (bps)
    as_deadzone_bps: float = 0.5  # ±0.5 bps 以内の逆行は AS と判定しない
    # 031# スプレッドフィルター
    min_spread_jpy: float = 0.0  # 0 = フィルタなし
    # 保存
    batch_size: int = 10  # バッチ保存のサイクル数
    max_save_retries: int = 3  # 保存リトライ上限
    save_fail_threshold: int = 3  # 緊急ダンプ発動の連続失敗回数
    # ログ
    progress_log_interval: int = 50  # 進捗ログの出力間隔 (サイクル数)
    log_max_bytes: int = 10 * 1024 * 1024  # ログファイル上限 (10 MB)
    log_backup_count: int = 5  # ログバックアップ世代数
    # 方策 A: パラメータ適応
    enable_auto_adapt: bool = False
    adapt_interval_cycles: int = 50
    min_adapt_samples: int = 50  # 方策 A/B 発動の最小サンプル数
    # 方策 B: 動的ロットサイジング
    enable_dynamic_lot: bool = False
    max_lot: float = 0.005
    lot_adapt_interval_cycles: int = 50
    recent_pnl_window: int = 50  # 方策 B 直近 PnL 計算ウィンドウ
    # レジーム検知 (035# §4)
    enable_regime: bool = True
    regime_window: int = 20
    regime_trend_threshold_pct: float = 0.5
    regime_high_vol_multiplier: float = 2.0
    regime_hysteresis_count: int = 3
    regime_min_confidence: float = 0.4
    # 052#: トレンディング時のオフセットブースト (PnL -1.2bps)
    regime_trending_offset_boost: float = 1.5  # トレンディング検出時に offset × 1.5
    # 041# 時間帯フィルター (AS 高リスク時間帯のスキップ)
    enable_time_filter: bool = False
    skip_utc_hours: list[int] | None = None
    # 安全設計 (000# §3.9)
    loss_cap_jpy: float = 10_000.0
    loss_cap_warning_ratio: float = 0.7
    # 041# 動的 loss_cap: API 残高から算出
    loss_cap_auto: bool = False
    loss_cap_ratio: float = 0.05  # 残高の 5% をキャップ (hard)
    # 046# soft/hard 二段 loss_cap: soft 超過でロット半減、hard 超過で SAFE_STOP
    soft_loss_cap_ratio: float = 0.02  # 残高の 2% で soft cap (ロット半減)
    # 049# E3 サンプリング: 全約定ではなくサンプリングで multi-timeframe 計測
    e3_sampling_ratio: float = 1.0  # 0.0-1.0, 1.0=全約定, 0.33=1/3 のみ
    # 049# side 別 offset: buy/sell で独立に offset を設定
    spread_offset_ratio_buy: float | None = None   # None = 共通 offset を使用
    spread_offset_ratio_sell: float | None = None   # None = 共通 offset を使用
    # 049# 即約定防御: queue_wait が閾値以下で負エッジの場合に保守化
    fast_fill_defense_enabled: bool = False
    fast_fill_threshold_sec: float = 5.0   # この秒数以下で「速い約定」と判定
    fast_fill_offset_boost: float = 2.0    # 防御時の offset 倍率
    # 054# S1: Orderbook Imbalance ベース AS 予測フィルター
    imbalance_enabled: bool = False
    imbalance_depth: int = 5               # 板深さ (上位 N 段)
    imbalance_threshold: float = 0.3       # 偏り判定閾値 (|imbalance| > threshold)
    imbalance_offset_boost: float = 1.5    # AS リスク時の offset 倍率
    imbalance_skip_threshold: float = 0.7  # この閾値以上なら注文スキップ
    # 054# S2: Smart Side Selection (機械的交互 → 条件付き)
    smart_side_enabled: bool = False
    smart_side_mode: str = "suppress"      # suppress / follow
    smart_side_max_consecutive: int = 2    # 片側蓄積防止 (000# §3.3)
    # 054# S3: テール損失カット (post-fill早期監視)
    early_exit_enabled: bool = False
    early_exit_threshold_bps: float = 5.0  # 損失閾値 (bps)
    early_exit_monitor_interval_sec: float = 5.0  # 監視刻み (秒)
    early_exit_rapid_interval_sec: float = 10.0   # rapid exit 時 cycle interval
    # 054# S4: Spread-Responsive Offset (スプレッド適応型オフセット)
    spread_adaptive_enabled: bool = False
    narrow_spread_bps: float = 10.0        # 狭スプレッド閾値 (bps)
    narrow_spread_boost: float = 2.0       # 狭い時の offset 倍率
    wide_spread_bps: float = 25.0          # 広スプレッド閾値 (bps)
    wide_spread_ratio: float = 0.5         # 広い時の offset 割引
    # 062# S5: SkipGate ML フィルター (AS 分類器ベースの注文スキップ)
    skip_gate_enabled: bool = False
    skip_gate_mode: str = "as"             # "pnl" or "as" (061# AS 分類器推奨)
    skip_gate_model_path: str = "models/v460/skip_gate_as.pkl"  # モデルファイル
    skip_gate_as_threshold: float = 0.6    # AS 確率スキップ閾値 (mode=as)
    skip_gate_pnl_threshold: float = 0.0   # PnL 予測スキップ閾値 (mode=pnl)
    skip_gate_max_skip_rate: float = 0.3   # 連続スキップ率上限 (安全弁)

    @classmethod
    def from_yaml(cls, yaml_cfg: dict) -> "FillTestConfig":
        """YAML dict から FillTestConfig を構築.

        YAML のフラットキー + ネスト (adaptation / lot_sizing / safety) を
        dataclass フィールドにマッピングする.
        """
        kwargs: dict = {}

        # フラットキー (YAML キー == dataclass フィールド名)
        flat_keys = {
            "symbol", "order_quantity", "cycle_interval_sec", "order_timeout_sec",
            "poll_interval_sec", "post_fill_wait_sec", "results_dir",
            "max_preflight_skip", "start_side",
            "spread_offset_ratio", "min_offset_jpy",
            "max_order_retries", "retry_delay_sec",
            "as_deadzone_bps", "min_spread_jpy",
            "batch_size", "max_save_retries", "save_fail_threshold",
            "progress_log_interval", "log_max_bytes", "log_backup_count",
            "min_adapt_samples",
        }
        for key in flat_keys:
            if key in yaml_cfg:
                kwargs[key] = yaml_cfg[key]

        # adaptation セクション → 方策 A
        adapt = yaml_cfg.get("adaptation", {})
        if adapt.get("enabled") is not None:
            kwargs["enable_auto_adapt"] = adapt["enabled"]
        if "interval_cycles" in adapt:
            kwargs["adapt_interval_cycles"] = adapt["interval_cycles"]

        # lot_sizing セクション → 方策 B
        lot = yaml_cfg.get("lot_sizing", {})
        if lot.get("enabled") is not None:
            kwargs["enable_dynamic_lot"] = lot["enabled"]
        if "interval_cycles" in lot:
            kwargs["lot_adapt_interval_cycles"] = lot["interval_cycles"]
        if "max_lot" in lot:
            kwargs["max_lot"] = lot["max_lot"]
        if "recent_pnl_window" in lot:
            kwargs["recent_pnl_window"] = lot["recent_pnl_window"]

        # regime セクション → レジーム検知 (035# §4)
        regime = yaml_cfg.get("regime", {})
        if regime.get("enabled") is not None:
            kwargs["enable_regime"] = regime["enabled"]
        regime_map = {
            "window": "regime_window",
            "trend_threshold_pct": "regime_trend_threshold_pct",
            "high_vol_multiplier": "regime_high_vol_multiplier",
            "hysteresis_count": "regime_hysteresis_count",
            "min_confidence": "regime_min_confidence",
            "trending_offset_boost": "regime_trending_offset_boost",
        }
        for yaml_key, config_key in regime_map.items():
            if yaml_key in regime:
                kwargs[config_key] = regime[yaml_key]

        # safety セクション → 損失キャップ
        safety = yaml_cfg.get("safety", {})
        if "loss_cap_jpy" in safety:
            kwargs["loss_cap_jpy"] = safety["loss_cap_jpy"]
        if "loss_cap_warning_ratio" in safety:
            kwargs["loss_cap_warning_ratio"] = safety["loss_cap_warning_ratio"]
        # 041# 動的 loss_cap
        if safety.get("loss_cap_auto") is not None:
            kwargs["loss_cap_auto"] = safety["loss_cap_auto"]
        if "loss_cap_ratio" in safety:
            kwargs["loss_cap_ratio"] = safety["loss_cap_ratio"]
        # 046# soft/hard 二段 loss_cap
        if "soft_loss_cap_ratio" in safety:
            kwargs["soft_loss_cap_ratio"] = safety["soft_loss_cap_ratio"]

        # 041# 時間帯フィルター
        tf = yaml_cfg.get("time_filter", {})
        if tf.get("enabled") is not None:
            kwargs["enable_time_filter"] = tf["enabled"]
        if "skip_utc_hours" in tf:
            kwargs["skip_utc_hours"] = tf["skip_utc_hours"]

        # 049# E3 サンプリング
        e3 = yaml_cfg.get("e3", {})
        if "sampling_ratio" in e3:
            kwargs["e3_sampling_ratio"] = e3["sampling_ratio"]

        # 049# side 別 offset
        side_offset = yaml_cfg.get("side_offset", {})
        if "buy" in side_offset:
            kwargs["spread_offset_ratio_buy"] = side_offset["buy"]
        if "sell" in side_offset:
            kwargs["spread_offset_ratio_sell"] = side_offset["sell"]

        # 049# 即約定防御
        ffd = yaml_cfg.get("fast_fill_defense", {})
        if ffd.get("enabled") is not None:
            kwargs["fast_fill_defense_enabled"] = ffd["enabled"]
        if "threshold_sec" in ffd:
            kwargs["fast_fill_threshold_sec"] = ffd["threshold_sec"]
        if "offset_boost" in ffd:
            kwargs["fast_fill_offset_boost"] = ffd["offset_boost"]

        # 054# S1: Orderbook Imbalance
        imb = yaml_cfg.get("imbalance", {})
        if imb.get("enabled") is not None:
            kwargs["imbalance_enabled"] = imb["enabled"]
        imb_map = {
            "depth": "imbalance_depth",
            "threshold": "imbalance_threshold",
            "offset_boost": "imbalance_offset_boost",
            "skip_threshold": "imbalance_skip_threshold",
        }
        for yaml_key, config_key in imb_map.items():
            if yaml_key in imb:
                kwargs[config_key] = imb[yaml_key]

        # 054# S2: Smart Side
        ss = yaml_cfg.get("smart_side", {})
        if ss.get("enabled") is not None:
            kwargs["smart_side_enabled"] = ss["enabled"]
        if "mode" in ss:
            kwargs["smart_side_mode"] = ss["mode"]
        if "max_consecutive_same" in ss:
            kwargs["smart_side_max_consecutive"] = ss["max_consecutive_same"]

        # 054# S3: Early Exit (テール損失カット)
        ee = yaml_cfg.get("early_exit", {})
        if ee.get("enabled") is not None:
            kwargs["early_exit_enabled"] = ee["enabled"]
        ee_map = {
            "threshold_bps": "early_exit_threshold_bps",
            "monitoring_interval_sec": "early_exit_monitor_interval_sec",
            "rapid_exit_interval_sec": "early_exit_rapid_interval_sec",
        }
        for yaml_key, config_key in ee_map.items():
            if yaml_key in ee:
                kwargs[config_key] = ee[yaml_key]

        # 054# S4: Spread Adaptive Offset
        sa = yaml_cfg.get("spread_adaptive", {})
        if sa.get("enabled") is not None:
            kwargs["spread_adaptive_enabled"] = sa["enabled"]
        sa_map = {
            "narrow_spread_bps": "narrow_spread_bps",
            "narrow_spread_boost": "narrow_spread_boost",
            "wide_spread_bps": "wide_spread_bps",
            "wide_spread_ratio": "wide_spread_ratio",
        }
        for yaml_key, config_key in sa_map.items():
            if yaml_key in sa:
                kwargs[config_key] = sa[yaml_key]

        # 062# S5: SkipGate ML フィルター
        sg = yaml_cfg.get("skip_gate", {})
        if sg.get("enabled") is not None:
            kwargs["skip_gate_enabled"] = sg["enabled"]
        sg_map = {
            "mode": "skip_gate_mode",
            "model_path": "skip_gate_model_path",
            "as_threshold": "skip_gate_as_threshold",
            "pnl_threshold": "skip_gate_pnl_threshold",
            "max_skip_rate": "skip_gate_max_skip_rate",
        }
        for yaml_key, config_key in sg_map.items():
            if yaml_key in sg:
                kwargs[config_key] = sg[yaml_key]

        return cls(**kwargs)


# ======================================================================
# Fill Test Runner
# ======================================================================

class FillTestRunner:
    """Maker 注文の fill quality を実測する.

    009# §4.2 の設計に準拠.
    """

    def __init__(
        self,
        adapter: CoincheckAdapter,
        config: FillTestConfig,
        yaml_cfg: Optional[dict] = None,
    ) -> None:
        self.adapter = adapter
        self.config = config
        self._yaml_cfg = yaml_cfg or {}  # YAML 生の設定 (サブモジュールに渡す用)
        self._results_dir = Path(config.results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._cycle_count = 0
        # start_side に応じて _last_side を設定 (交互ロジック用)
        if config.start_side == "sell":
            self._last_side = "buy"  # → _next_side() が "sell" を返す
        else:
            self._last_side = None  # → _next_side() が "buy" を返す
        self._preflight_skip_count = 0  # 044# 連続 preflight スキップ計
        self._shutdown_requested = False
        self._pending_order_id: Optional[str] = None
        self._lockfile_path: Optional[Path] = None  # 044# 単一起動ロック

        # 020# O4: データバージョン管理
        self._run_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        self._git_sha = self._get_git_sha()

        # 033# 方策 B: 動的ロットの実行時数量 (config.order_quantity を初期値とする)
        self._current_lot: float = config.order_quantity
        # 046# soft loss_cap 発動済みフラグ (重複半減を防止)
        self._soft_loss_cap_triggered: bool = False
        # 047# Issue12: time_filter ログ throttle (突入/離脱のみ出力)
        self._in_time_filter: bool = False
        # 049# 即約定防御: 次サイクルの offset を一時的に増加
        self._fast_fill_boost_active: bool = False
        # 050# Bug#1 fix: boost 発動前の offset を保存 (解除時に復元)
        self._pre_boost_offset: float | None = None
        self._pre_boost_offset_sell: float | None = None
        # 051# P2-3: Balance auto-shrink (残高不足時のロット一時縮小)
        self._balance_shrink_active: bool = False
        self._pre_shrink_lot: float = config.order_quantity
        # 054# S1: Orderbook Imbalance — 直前計測値を保持 (S2 Smart Side で参照)
        self._last_imbalance: float = 0.0
        self._last_bid_depth: float = 0.0
        self._last_ask_depth: float = 0.0
        # 054# S2: Smart Side — 同一 side 連続カウンタ
        self._consecutive_same_side: int = 0
        # 054# S3: Early Exit — rapid exit フラグ
        self._rapid_exit_pending: bool = False
        self._rapid_exit_side: str | None = None
        # 054# S1/S4: mid price 追跡 (trend 計算用)
        self._prev_mid_price: float | None = None
        self._prev_mid_time: float | None = None
        self._last_mid_trend_bps: float | None = None

        # 037# レジーム検知 (035# §4)
        self._regime_detector: Optional["FillTestRegimeDetector"] = None
        if config.enable_regime:
            from scripts.v460.lib.regime_detector import (
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
            logger.info(
                f"[Regime] detector enabled: window={regime_cfg.window}, "
                f"hysteresis={regime_cfg.hysteresis_count}"
            )

        # 024# R1: 保存失敗トラッキング
        self._unsaved_batch: list[FillRecord] = []
        self._save_fail_count: int = 0
        self._max_save_retries: int = config.max_save_retries

        # 062# S5: SkipGate ML フィルター
        self._skip_gate: Optional["SkipGate"] = None
        if config.skip_gate_enabled:
            try:
                from scripts.v460.ml.skip_gate import SkipGate, SkipGateConfig

                gate_path = Path(config.skip_gate_model_path)
                if not gate_path.is_absolute():
                    gate_path = _PROJECT_ROOT / gate_path
                if not gate_path.exists():
                    logger.warning(
                        f"[skip_gate] Model not found: {gate_path}. "
                        f"SkipGate disabled."
                    )
                else:
                    self._skip_gate = SkipGate.load(gate_path)
                    # ランタイム設定でオーバーライド
                    self._skip_gate.config.mode = config.skip_gate_mode
                    self._skip_gate.config.as_threshold = config.skip_gate_as_threshold
                    self._skip_gate.config.threshold_bps = config.skip_gate_pnl_threshold
                    self._skip_gate.config.max_skip_rate = config.skip_gate_max_skip_rate
                    logger.info(
                        f"[skip_gate] Loaded: mode={config.skip_gate_mode}, "
                        f"as_threshold={config.skip_gate_as_threshold}, "
                        f"features={len(self._skip_gate.feature_cols)}, "
                        f"path={gate_path}"
                    )
            except Exception as e:
                logger.error(f"[skip_gate] Failed to load: {e}. SkipGate disabled.")
                self._skip_gate = None

        # 安全設計: atexit + signal で残存注文キャンセル + 未保存データ退避 + ロック解放
        atexit.register(self._cleanup_sync)

        # 044# A-7: loss_cap 更新カウンタ (50サイクル毎に残高再取得)
        self._loss_cap_update_interval = 50  # サイクル数

    @staticmethod
    def _get_git_sha() -> Optional[str]:
        """現在の git commit short hash を取得."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=str(_PROJECT_ROOT),
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

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
        """buy/sell を決定: 交互 or 054# S2 Smart Side.

        009# §4.2: 片側ポジション蓄積禁止.
        054# S2: imbalance による side 抑制/追従.
        055# Fix: rapid_exit_side 優先返却.
        """
        # 055# Fix #1: S3 rapid exit で決定された side を最優先で返却
        if self._rapid_exit_side is not None:
            forced_side = self._rapid_exit_side
            self._rapid_exit_side = None  # 使用済みクリア
            logger.info(f"[early_exit] Rapid exit forcing side={forced_side}")
            return forced_side

        base_side = "buy" if (self._last_side is None or self._last_side == "sell") else "sell"

        if not self.config.smart_side_enabled:
            return base_side

        imbalance = self._last_imbalance

        if self.config.smart_side_mode == "suppress":
            # 不利な side を抑制: buy を出そうとしているが売り圧力が強い → buy スキップ
            should_suppress = False
            if base_side == "buy" and imbalance < -self.config.imbalance_threshold:
                should_suppress = True
            elif base_side == "sell" and imbalance > self.config.imbalance_threshold:
                should_suppress = True

            if should_suppress:
                # 連続同 side 制限チェック (000# §3.3: 片側蓄積防止)
                if self._consecutive_same_side >= self.config.smart_side_max_consecutive:
                    logger.debug(
                        f"[smart_side] Max consecutive ({self._consecutive_same_side}) reached, "
                        f"forcing {base_side}"
                    )
                    return base_side
                alt_side = self._last_side or ("sell" if base_side == "buy" else "buy")
                logger.info(
                    f"[smart_side] Suppressing {base_side} (imb={imbalance:+.3f}), "
                    f"continuing {alt_side}"
                )
                return alt_side

        elif self.config.smart_side_mode == "follow":
            # imbalance 方向に追従
            if abs(imbalance) > self.config.imbalance_threshold:
                follow_side = "buy" if imbalance > 0 else "sell"
                # 連続同 side 制限チェック
                if (
                    follow_side == self._last_side
                    and self._consecutive_same_side >= self.config.smart_side_max_consecutive
                ):
                    return base_side
                return follow_side

        return base_side

    async def _compute_orderbook_imbalance(self, depth: int = 5) -> tuple[float, float, float]:
        """054# S1: 板不均衡を計算.

        Returns:
            (imbalance, bid_total, ask_total) タプル.
            imbalance ∈ [-1, +1].
            +1 = bid 側が圧倒的 (買い圧力 = 価格上昇示唆).
            -1 = ask 側が圧倒的 (売り圧力 = 価格下落示唆).
        """
        ob = await self.adapter.get_orderbook(self.config.symbol, depth=depth)
        bid_volume = sum(qty for _, qty in ob.bids[:depth]) if ob.bids else 0.0
        ask_volume = sum(qty for _, qty in ob.asks[:depth]) if ob.asks else 0.0
        total = bid_volume + ask_volume
        if total == 0:
            return 0.0, 0.0, 0.0
        imbalance = (bid_volume - ask_volume) / total
        return imbalance, bid_volume, ask_volume

    async def _get_mid_price(self) -> float:
        """板の best bid/ask から mid price を算出."""
        ob = await self.adapter.get_orderbook(self.config.symbol, depth=1)
        if not ob.bids or not ob.asks:
            raise ValueError("Empty orderbook — cannot compute mid price")
        best_bid = ob.bids[0][0]
        best_ask = ob.asks[0][0]
        return (best_bid + best_ask) / 2.0

    async def _compute_maker_price(self, side: str) -> tuple[float, float, float]:
        """maker limit 価格を算出: スプレッド比例オフセット + post_only 安全策.

        009# §4.2: スプレッド内側に配置して maker 約定を狙う.
        CM-1: 固定 1 JPY → スプレッド比例 + post_only リジェクト防止.
        054# S1: Imbalance ベース AS リスク補正.
        054# S4: Spread 適応型 offset.

        Returns:
            (price, spread_at_order, effective_offset_ratio) タプル.
        """
        # 054# S1: imbalance 計算 (depth>1 で板の深さを参照)
        if self.config.imbalance_enabled:
            imb, bid_d, ask_d = await self._compute_orderbook_imbalance(
                depth=self.config.imbalance_depth,
            )
            self._last_imbalance = imb
            self._last_bid_depth = bid_d
            self._last_ask_depth = ask_d
        else:
            imb = 0.0

        ob = await self.adapter.get_orderbook(self.config.symbol, depth=1)
        if not ob.bids or not ob.asks:
            raise ValueError("Empty orderbook")
        best_bid = ob.bids[0][0]
        best_ask = ob.asks[0][0]
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2.0

        # 054# mid price trend 追跡
        mid_trend_bps: float | None = None
        now = time.time()
        if self._prev_mid_price is not None and self._prev_mid_time is not None:
            dt = now - self._prev_mid_time
            if 0 < dt < 300:  # 5分以内のデータのみ有効
                mid_trend_bps = (mid_price - self._prev_mid_price) / self._prev_mid_price * 10000
        self._prev_mid_price = mid_price
        self._prev_mid_time = now
        self._last_mid_trend_bps = mid_trend_bps

        # 031# スプレッドフィルター: 狭すぎる場合はスキップ
        if spread < self.config.min_spread_jpy:
            raise ValueError(
                f"Spread too narrow: {spread:.0f} JPY < min {self.config.min_spread_jpy:.0f}"
            )

        # スプレッド比例オフセット (最小保証付き)
        # 049# side 別 offset: buy/sell 独立設定がある場合はそちらを優先
        effective_offset_ratio = self.config.spread_offset_ratio
        if side == "buy" and self.config.spread_offset_ratio_buy is not None:
            effective_offset_ratio = self.config.spread_offset_ratio_buy
        elif side == "sell" and self.config.spread_offset_ratio_sell is not None:
            effective_offset_ratio = self.config.spread_offset_ratio_sell

        # 052#: トレンディング時にオフセットをブースト (PnL -1.2bps 対策)
        if (
            self._regime_detector is not None
            and self._regime_detector.current_regime.value == "trending"
            and self.config.regime_trending_offset_boost > 1.0
        ):
            effective_offset_ratio *= self.config.regime_trending_offset_boost
            logger.debug(
                f"[regime] trending → offset boosted: "
                f"{effective_offset_ratio / self.config.regime_trending_offset_boost:.4f} "
                f"→ {effective_offset_ratio:.4f}"
            )

        # 054# S4: Spread 適応型 offset
        if self.config.spread_adaptive_enabled:
            spread_bps = spread / mid_price * 10000
            if spread_bps < self.config.narrow_spread_bps:
                effective_offset_ratio = min(
                    effective_offset_ratio * self.config.narrow_spread_boost, 0.30,
                )
                logger.debug(
                    f"[spread_adaptive] Narrow spread {spread_bps:.1f}bps "
                    f"→ offset boosted to {effective_offset_ratio:.4f}"
                )
            elif spread_bps > self.config.wide_spread_bps:
                effective_offset_ratio = max(
                    effective_offset_ratio * self.config.wide_spread_ratio, 0.01,
                )
                logger.debug(
                    f"[spread_adaptive] Wide spread {spread_bps:.1f}bps "
                    f"→ offset reduced to {effective_offset_ratio:.4f}"
                )

        # 054# S1: Imbalance ベース AS リスク補正
        imbalance_skipped = False
        if self.config.imbalance_enabled and abs(imb) > self.config.imbalance_threshold:
            # AS リスクが高い side かチェック
            as_risk = (
                (side == "buy" and imb < -self.config.imbalance_threshold)
                or (side == "sell" and imb > self.config.imbalance_threshold)
            )
            if as_risk:
                if abs(imb) >= self.config.imbalance_skip_threshold:
                    # 極端な偏り → 注文自体をスキップ
                    imbalance_skipped = True
                    logger.info(
                        f"[imbalance] Extreme AS risk: {side} imb={imb:+.3f} "
                        f">= skip_threshold {self.config.imbalance_skip_threshold}. "
                        f"Skipping order."
                    )
                    raise ValueError(
                        f"Imbalance skip: {side} order suppressed (imb={imb:+.3f})"
                    )
                else:
                    # 中程度の偏り → offset 拡大
                    effective_offset_ratio *= self.config.imbalance_offset_boost
                    effective_offset_ratio = min(effective_offset_ratio, 0.30)
                    logger.info(
                        f"[imbalance] {side} AS risk: imb={imb:+.3f}, "
                        f"offset boosted to {effective_offset_ratio:.4f}"
                    )

        offset = max(self.config.min_offset_jpy, spread * effective_offset_ratio)

        if side == "buy":
            price = best_bid + offset
            # CM-1: post_only ガード — best_ask 以上にならないよう保護
            if price >= best_ask:
                price = best_bid  # best bid に退避 (確実に maker)
                logger.info(
                    f"Spread guard: buy price {best_bid + offset:.0f} >= ask {best_ask:.0f}, "
                    f"fallback to best_bid {best_bid:.0f} (spread={spread:.0f})"
                )
            return price, spread, effective_offset_ratio
        else:
            price = best_ask - offset
            # CM-1: post_only ガード — best_bid 以下にならないよう保護
            if price <= best_bid:
                price = best_ask  # best ask に退避 (確実に maker)
                logger.info(
                    f"Spread guard: sell price {best_ask - offset:.0f} <= bid {best_bid:.0f}, "
                    f"fallback to best_ask {best_ask:.0f} (spread={spread:.0f})"
                )
            return price, spread, effective_offset_ratio

    def _is_time_filtered(self) -> bool:
        """041# 時間帯フィルター: 高 AS 時間帯かどうかを判定.

        Returns True の場合、呼び出し元は FillRecord を生成せずスリープする。
        レコード不生成により fill_rate メトリクスの汚染を防止。
        """
        if not self.config.enable_time_filter or not self.config.skip_utc_hours:
            return False
        current_utc_hour = datetime.now(timezone.utc).hour
        return current_utc_hour in self.config.skip_utc_hours

    # 052#: Coincheck 取引所 BTC 最小注文数量 (板取引)
    _MIN_ORDER_BTC: float = 0.001

    async def _check_balance_for_side(self, side: str) -> bool:
        """041# 残高 pre-flight check: 発注前に残高が十分か確認.

        不足時は True を返す (スキップすべき)。
        052#: 残高に基づくロット自動縮小 — 残高が現ロットに不足するが
        最小ロット (0.001 BTC) 以上なら自動的にロットを縮小して継続。
        """
        try:
            if side == "sell":
                # sell には BTC 残高が必要
                btc_balances = await self.adapter.get_balance("BTC")
                btc_free = sum(b.free for b in btc_balances) if btc_balances else 0.0
                if btc_free < self._current_lot:
                    # 052#: 最小ロット以上の残高があれば縮小して継続
                    if btc_free >= self._MIN_ORDER_BTC:
                        # 0.001 BTC 単位に切り捨て
                        new_lot = int(btc_free / self._MIN_ORDER_BTC) * self._MIN_ORDER_BTC
                        if new_lot >= self._MIN_ORDER_BTC:
                            old_lot = self._current_lot
                            self._current_lot = new_lot
                            logger.info(
                                f"[balance] BTC {btc_free:.6f} < {old_lot:.4f}. "
                                f"ロット自動縮小: {old_lot:.4f} → {new_lot:.4f} BTC"
                            )
                            return False  # 縮小ロットで発注 OK
                    logger.warning(
                        f"[balance] Insufficient BTC for sell: "
                        f"{btc_free:.6f} < {self._MIN_ORDER_BTC:.4f}. "
                        f"Skipping sell → will retry buy next."
                    )
                    return True
            else:
                # buy には JPY 残高が必要
                price = await self.adapter.get_current_price(self.config.symbol)
                if price:
                    jpy_needed = self._current_lot * price * 1.01  # 1% margin
                    jpy_balances = await self.adapter.get_balance("JPY")
                    jpy_free = sum(b.free for b in jpy_balances) if jpy_balances else 0.0
                    if jpy_free < jpy_needed:
                        # 052#: JPY 残高から発注可能なロットを逆算
                        affordable_lot = jpy_free / (price * 1.01)
                        affordable_lot = int(affordable_lot / self._MIN_ORDER_BTC) * self._MIN_ORDER_BTC
                        if affordable_lot >= self._MIN_ORDER_BTC:
                            old_lot = self._current_lot
                            self._current_lot = affordable_lot
                            logger.info(
                                f"[balance] JPY {jpy_free:.0f} < {jpy_needed:.0f}. "
                                f"ロット自動縮小: {old_lot:.4f} → {affordable_lot:.4f} BTC"
                            )
                            return False  # 縮小ロットで発注 OK
                        logger.warning(
                            f"[balance] Insufficient JPY for buy: "
                            f"{jpy_free:.0f} < min {self._MIN_ORDER_BTC * price * 1.01:.0f}. "
                            f"Skipping buy → will retry sell next."
                        )
                        return True
        except Exception as e:
            logger.debug(f"[balance] Pre-flight check failed (non-fatal): {e}")
        return False

    def _acquire_lock(self) -> None:
        """044# Bug7: 単一起動ロック (lockfile + PID + stale 回収).

        047# A4: TOCTOU race 対策 — open(path, 'x') で排他的作成。
        同一 results_dir に対して複数プロセスが並行動作することを防止。
        ロックファイルに PID を記録し、起動時に既存ロックの生死を検証する。
        """
        lock_path = self._results_dir / "fill_test.lock"
        self._lockfile_path = lock_path
        lock_content = f"{os.getpid()}|{int(time.time())}|{self._run_id}"

        def _check_stale_and_reclaim() -> bool:
            """既存ロックが stale なら削除して True を返す."""
            try:
                content = lock_path.read_text(encoding="utf-8").strip()
                parts = content.split("|")
                existing_pid = int(parts[0])
                import psutil  # type: ignore[import-untyped]
                if psutil.pid_exists(existing_pid):
                    try:
                        proc = psutil.Process(existing_pid)
                        cmdline = " ".join(proc.cmdline())
                        if "fill_test" in cmdline or "run_fill_test" in cmdline:
                            raise RuntimeError(
                                f"別の fill_test プロセスが実行中です "
                                f"(PID={existing_pid}). "
                                f"強制起動するにはロックファイルを削除: {lock_path}"
                            )
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except (ValueError, ImportError, OSError):
                pass
            # stale lock — 削除して再取得を試みる
            logger.warning(f"[lock] Stale lockfile detected, reclaiming: {lock_path}")
            try:
                lock_path.unlink()
            except OSError:
                pass
            return True

        # 047# A4: open(path, 'x') で排他的にファイル作成 (atomic)
        # FileExistsError なら既存ロックを検証 → stale ならリトライ
        for _attempt in range(2):
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, lock_content.encode("utf-8"))
                finally:
                    os.close(fd)
                logger.info(
                    f"[lock] Acquired lockfile: PID={os.getpid()}, run_id={self._run_id}"
                )
                return
            except FileExistsError:
                _check_stale_and_reclaim()
        # 2 回リトライ後もダメな場合
        raise RuntimeError(f"ロックファイルの取得に失敗しました: {lock_path}")

    def _release_lock(self) -> None:
        """044# ロックファイル解放."""
        if self._lockfile_path and self._lockfile_path.exists():
            try:
                # 自プロセスのロックのみ解放
                content = self._lockfile_path.read_text(encoding="utf-8").strip()
                if content.startswith(f"{os.getpid()}|"):
                    self._lockfile_path.unlink()
                    logger.info("[lock] Released lockfile")
            except Exception as e:
                logger.warning(f"[lock] Failed to release lockfile: {e}")

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

    async def run_single_cycle(self) -> FillRecord:
        """1 サイクル: 発注 → 監視 → 結果記録.

        009# §4.2 の流れに準拠.
        041# 時間帯フィルター・残高チェック追加.
        055# Fix: side 決定前に最新 imbalance を取得.
        """
        self._cycle_count += 1
        cycle_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"

        # 055# Fix #2: Smart Side 判定用に最新板 imbalance を事前取得
        # (_compute_maker_price 内での取得では side 決定後 → 1サイクル遅延)
        if self.config.imbalance_enabled and self.config.smart_side_enabled:
            try:
                imb, bid_d, ask_d = await self._compute_orderbook_imbalance(
                    depth=self.config.imbalance_depth,
                )
                self._last_imbalance = imb
                self._last_bid_depth = bid_d
                self._last_ask_depth = ask_d
            except Exception as e:
                logger.warning(f"[smart_side] Pre-fetch imbalance failed, using last: {e}")
                # フォールバック: 前回値を維持

        side = self._next_side()
        # 054# S2: 連続同 side カウンタ更新
        if side == self._last_side:
            self._consecutive_same_side += 1
        else:
            self._consecutive_same_side = 0
        self._last_side = side

        logger.info(f"=== Cycle {self._cycle_count} ({side}) ===")

        # 1. maker limit 価格算出
        spread_at_order: Optional[float] = None
        effective_offset_ratio: float = self.config.spread_offset_ratio
        try:
            order_price, spread_at_order, effective_offset_ratio = await self._compute_maker_price(side)
        except Exception as e:
            logger.error(f"Failed to compute maker price: {e}")
            return FillRecord(
                cycle_id=cycle_id,
                timestamp=time.time(),
                side=side,
                order_price=0.0,
                order_quantity=self._current_lot,
                cancelled=True,
                cancel_reason="orderbook_error",
                error_message=str(e),
                spread_offset_ratio=self.config.spread_offset_ratio,
            )

        # 062# 1.5: SkipGate ML 判定 (注文前にスキップ判断)
        skip_gate_skipped: Optional[bool] = None
        skip_gate_score: Optional[float] = None
        skip_gate_reason: Optional[str] = None
        if self._skip_gate is not None:
            try:
                from scripts.v460.ml.skip_gate import build_features_from_market_state

                # OB データはすでに _compute_maker_price 内で取得済み
                # → _last_imbalance, _last_bid_depth, _last_ask_depth を再利用
                ob = await self.adapter.get_orderbook(
                    self.config.symbol, depth=self.config.imbalance_depth,
                )
                best_bid = ob.bids[0][0] if ob.bids else 0.0
                best_ask = ob.asks[0][0] if ob.asks else 0.0
                bid_vol = sum(qty for _, qty in ob.bids[:5]) if ob.bids else 0.0
                ask_vol = sum(qty for _, qty in ob.asks[:5]) if ob.asks else 0.0

                # レジーム情報
                sg_regime = "unknown"
                if self._regime_detector is not None:
                    sg_regime = self._regime_detector.current_regime.value

                # 直近約定データ取得 (利用可能な場合)
                recent_trades_data: list[dict] | None = None
                try:
                    trades = await self.adapter.get_recent_trades(
                        self.config.symbol, limit=50,
                    )
                    if trades:
                        recent_trades_data = [
                            {
                                "ts": getattr(t, "timestamp", time.time()),
                                "price": getattr(t, "price", 0.0),
                                "amount": getattr(t, "amount", getattr(t, "quantity", 0.0)),
                                "side": getattr(t, "side", "buy"),
                            }
                            for t in trades
                        ]
                except Exception:
                    pass  # 約定データ取得失敗は非致命的

                gate_features = build_features_from_market_state(
                    side=side,
                    spread_jpy=spread_at_order or 0.0,
                    offset_ratio=effective_offset_ratio,
                    regime=sg_regime,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    bid_vol_5=bid_vol,
                    ask_vol_5=ask_vol,
                    recent_trades=recent_trades_data,
                    market_timestamp=time.time(),
                )

                decision = self._skip_gate.evaluate(gate_features)
                skip_gate_skipped = decision.should_skip
                skip_gate_score = decision.predicted_pnl_bps
                skip_gate_reason = decision.reason

                if decision.should_skip:
                    logger.info(
                        f"[skip_gate] SKIP: {side} order skipped "
                        f"(score={skip_gate_score:.3f}, reason={skip_gate_reason}, "
                        f"features={decision.features_used})"
                    )
                    return FillRecord(
                        cycle_id=cycle_id,
                        timestamp=time.time(),
                        side=side,
                        order_price=order_price,
                        order_quantity=self._current_lot,
                        cancelled=True,
                        cancel_reason="skip_gate",
                        spread_at_order=spread_at_order,
                        spread_offset_ratio=effective_offset_ratio,
                        skip_gate_skipped=True,
                        skip_gate_score=skip_gate_score,
                        skip_gate_reason=skip_gate_reason,
                        orderbook_imbalance=self._last_imbalance if self.config.imbalance_enabled else None,
                        bid_depth_total=self._last_bid_depth if self.config.imbalance_enabled else None,
                        ask_depth_total=self._last_ask_depth if self.config.imbalance_enabled else None,
                    )
                else:
                    logger.debug(
                        f"[skip_gate] PASS: {side} order allowed "
                        f"(score={skip_gate_score:.3f}, reason={skip_gate_reason})"
                    )
            except Exception as e:
                logger.warning(f"[skip_gate] Evaluation failed (non-fatal): {e}")
                skip_gate_reason = f"error:{e}"

        # 2. 発注 (CM-2: リトライ付き)
        t_submit = time.time()
        order = None
        last_error: Optional[str] = None
        cancel_reason: str = "unknown"  # 032# #6: ループ未実行時の NameError 防止
        for attempt in range(1 + self.config.max_order_retries):
            try:
                order = await self.adapter.place_order(
                    symbol=self.config.symbol,
                    side=side,
                    quantity=self._current_lot,
                    price=order_price,
                    order_type="limit",
                )
                self._pending_order_id = order.order_id
                logger.info(
                    f"Placed {side} limit @ {order_price:.0f} JPY, "
                    f"qty={self._current_lot}, id={order.order_id}"
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
                    # 042# Coincheck の日本語エラーメッセージ対応
                    or "所持金額" in last_error
                    or "足りません" in last_error
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
                if cancel_reason == "insufficient_funds":
                    logger.info(
                        f"[Bug10] Skipping retry — {cancel_reason} is not retriable"
                    )
                    break

                if attempt < self.config.max_order_retries:
                    # リトライ: 板を再取得してより保守的な価格で再発注
                    await asyncio.sleep(self.config.retry_delay_sec)
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
            return FillRecord(
                cycle_id=cycle_id,
                timestamp=t_submit,
                side=side,
                order_price=order_price,
                order_quantity=self._current_lot,
                cancelled=True,
                cancel_reason=cancel_reason,
                error_message=last_error,  # 031# エラー詳細を記録
                spread_at_order=spread_at_order,
                spread_offset_ratio=self.config.spread_offset_ratio,
            )

        # 3. ポーリング監視
        filled = False
        fill_price: Optional[float] = None
        t_fill: Optional[float] = None
        cancel_reason_poll: Optional[str] = None  # 025# F6: poll 中の cancel 理由
        elapsed = 0.0

        while elapsed < self.config.order_timeout_sec and not self._shutdown_requested:
            await asyncio.sleep(self.config.poll_interval_sec)
            elapsed = time.time() - t_submit

            try:
                status_order = await self.adapter.get_order_status(order.order_id)
                if status_order is None:
                    # 025# F6: open orders にも transactions にもない
                    # → API 一時障害の可能性があるため 1 回リトライ
                    logger.warning(
                        f"Order {order.order_id} not found — retrying after 2s"
                    )
                    await asyncio.sleep(2.0)
                    status_order = await self.adapter.get_order_status(
                        order.order_id,
                    )
                    if status_order is not None and status_order.status == "filled":
                        filled = True
                        fill_price = (
                            status_order.price
                            if status_order.price
                            else order_price
                        )
                        t_fill = time.time()
                        logger.info(
                            f"Order confirmed filled on retry @ "
                            f"{fill_price:.0f} JPY"
                        )
                        break
                    # リトライ後も不明 → 保守的に cancelled 扱い
                    logger.warning(
                        f"Order {order.order_id} status unknown after retry "
                        f"— treating as cancelled (status_unknown)"
                    )
                    cancel_reason_poll = "status_unknown"
                    break
                elif status_order.status == "filled":
                    filled = True
                    fill_price = (
                        status_order.price if status_order.price else order_price
                    )
                    t_fill = time.time()
                    logger.info(
                        f"Order filled @ {fill_price:.0f} JPY, "
                        f"wait={elapsed:.1f}s"
                    )
                    break
                elif status_order.status in ("cancelled", "rejected"):
                    # 031# 取引所キャンセル/リジェクトの理由を明示的に記録
                    cancel_reason_poll = f"exchange_{status_order.status}"
                    logger.info(f"Order {status_order.status}: {order.order_id}")
                    break
            except Exception as e:
                logger.warning(f"Poll error: {e}")

        # 4. 未約定 → キャンセル
        if not filled:
            try:
                await self.adapter.cancel_order(order.order_id)
                logger.info(f"Cancelled unfilled order after {elapsed:.1f}s")
            except Exception as e:
                logger.warning(f"Cancel failed: {e}")
                # 047# Bug11: cancel 失敗 = 既に約定済みの可能性
                # "Failed to cancel" は Coincheck がキャンセル不可を返す場合
                # → get_order_status で再確認し、約定済みなら filled に修正
                if "Failed to cancel" in str(e) or "not found" in str(e).lower():
                    try:
                        recheck = await self.adapter.get_order_status(order.order_id)
                        if recheck is not None and recheck.status == "filled":
                            filled = True
                            fill_price = (
                                recheck.price if recheck.price else order_price
                            )
                            t_fill = time.time()
                            cancel_reason_poll = None  # status_unknown を取り消し
                            logger.info(
                                f"[Bug11] Order was actually filled @ "
                                f"{fill_price:.0f} JPY (detected on cancel failure)"
                            )
                        else:
                            logger.info(
                                f"[Bug11] Recheck: order not found in transactions either"
                            )
                    except Exception as recheck_err:
                        logger.warning(f"[Bug11] Recheck failed: {recheck_err}")

        self._pending_order_id = None
        queue_wait = elapsed

        # 5. 約定後 mid price 計測 (047# E3: 30/60/120s multi-timeframe)
        mid_at_fill: Optional[float] = None
        mid_30s_after: Optional[float] = None
        mid_60s_after: Optional[float] = None
        mid_120s_after: Optional[float] = None
        post_fill_pnl: Optional[float] = None
        post_fill_60s_pnl: Optional[float] = None
        post_fill_120s_pnl: Optional[float] = None
        adverse_selected: Optional[bool] = None
        adverse_selected_raw: Optional[bool] = None

        if filled and fill_price is not None:
            try:
                mid_at_fill = await self._get_mid_price()
            except Exception:
                pass

            # 054# S3: Early Exit 監視付き 30s 待機
            early_exit_triggered = False
            if self.config.early_exit_enabled and mid_at_fill is not None:
                # 5s 刻みで 30s まで mid を監視
                monitor_sec = self.config.early_exit_monitor_interval_sec
                ticks = max(1, int(self.config.post_fill_wait_sec / monitor_sec))
                for tick in range(ticks):
                    await asyncio.sleep(monitor_sec)
                    try:
                        mid_now = await self._get_mid_price()
                        if side == "buy":
                            interim_pnl = (mid_now - mid_at_fill) / mid_at_fill * 10000
                        else:
                            interim_pnl = (mid_at_fill - mid_now) / mid_at_fill * 10000
                        if interim_pnl < -self.config.early_exit_threshold_bps:
                            logger.warning(
                                f"[early_exit] Loss threshold hit at {(tick+1)*monitor_sec:.0f}s: "
                                f"{interim_pnl:+.2f} bps < -{self.config.early_exit_threshold_bps}"
                            )
                            early_exit_triggered = True
                            break
                    except Exception:
                        continue
                # 残り時間を消化 (e.g. 15s で early exit → 残り 15s は不要)
                elapsed_monitor = (tick + 1) * monitor_sec if early_exit_triggered else ticks * monitor_sec
                remaining = self.config.post_fill_wait_sec - elapsed_monitor
                if remaining > 0 and not early_exit_triggered:
                    await asyncio.sleep(remaining)
            else:
                # 通常の 30s 待機
                logger.info(f"Waiting {self.config.post_fill_wait_sec}s for PnL measurement...")
                await asyncio.sleep(self.config.post_fill_wait_sec)

            try:
                mid_30s_after = await self._get_mid_price()
            except Exception:
                pass

            if mid_at_fill is not None and mid_30s_after is not None:
                # PnL in bps (basis points)
                if side == "buy":
                    # buy: 価格上昇が有利
                    post_fill_pnl = (mid_30s_after - mid_at_fill) / mid_at_fill * 10000
                    # 020# O5: raw AS 判定 (deadzone 非適用)
                    adverse_selected_raw = mid_30s_after < mid_at_fill
                    # CM-3: AS デッドゾーン — ノイズ幅以内の逆行は AS と判定しない
                    adverse_selected = post_fill_pnl < -self.config.as_deadzone_bps
                else:
                    # sell: 価格下落が有利
                    post_fill_pnl = (mid_at_fill - mid_30s_after) / mid_at_fill * 10000
                    # 020# O5: raw AS 判定 (deadzone 非適用)
                    adverse_selected_raw = mid_30s_after > mid_at_fill
                    # CM-3: AS デッドゾーン
                    adverse_selected = post_fill_pnl < -self.config.as_deadzone_bps

            # 054# S3: early exit → rapid exit フラグ (次サイクルの interval 短縮)
            if early_exit_triggered:
                self._rapid_exit_pending = True
                self._rapid_exit_side = "sell" if side == "buy" else "buy"

            # 047# E3: +30s (=60s) 計測 — 049# サンプリング制御
            # e3_sampling_ratio < 1.0 の場合、確率的にスキップしてサイクル効率を回復
            import random as _rng
            do_e3 = mid_at_fill is not None and _rng.random() < self.config.e3_sampling_ratio
            if do_e3:
                await asyncio.sleep(self.config.post_fill_wait_sec)  # +30s
                try:
                    mid_60s_after = await self._get_mid_price()
                    if side == "buy":
                        post_fill_60s_pnl = (mid_60s_after - mid_at_fill) / mid_at_fill * 10000
                    else:
                        post_fill_60s_pnl = (mid_at_fill - mid_60s_after) / mid_at_fill * 10000
                except Exception:
                    pass

                # 047# E3: +60s (=120s) 計測
                await asyncio.sleep(self.config.post_fill_wait_sec * 2)  # +60s
                try:
                    mid_120s_after = await self._get_mid_price()
                    if side == "buy":
                        post_fill_120s_pnl = (mid_120s_after - mid_at_fill) / mid_at_fill * 10000
                    else:
                        post_fill_120s_pnl = (mid_at_fill - mid_120s_after) / mid_at_fill * 10000
                except Exception:
                    pass

        # 037# レジーム検知更新 (035# §7 Week 1)
        regime_str: Optional[str] = None
        regime_conf: Optional[float] = None
        regime_stab: Optional[int] = None
        if self._regime_detector is not None:
            # mid_at_fill または order_price をレジーム検知の入力に使用
            regime_price = mid_at_fill if mid_at_fill is not None else order_price
            regime_result = self._regime_detector.update(t_submit, regime_price)
            regime_str = regime_result.regime.value
            regime_conf = regime_result.confidence
            regime_stab = regime_result.stability

        record = FillRecord(
            cycle_id=cycle_id,
            timestamp=t_submit,
            side=side,
            order_price=order_price,
            order_quantity=self._current_lot,
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
                else ("timeout" if (not filled and queue_wait >= self.config.order_timeout_sec) else None)
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
            # 054# S5: AS 予測データ基盤
            orderbook_imbalance=self._last_imbalance if self.config.imbalance_enabled else None,
            bid_depth_total=self._last_bid_depth if self.config.imbalance_enabled else None,
            ask_depth_total=self._last_ask_depth if self.config.imbalance_enabled else None,
            mid_price_trend_5s=getattr(self, '_last_mid_trend_bps', None),
            spread_bps=(
                (spread_at_order / mid_at_fill * 10000)
                if spread_at_order is not None and mid_at_fill is not None and mid_at_fill > 0
                else None
            ),
            effective_offset_used=effective_offset_ratio,
            # 062# SkipGate 判定情報 (PASS 時も記録 → 後続分析用)
            skip_gate_skipped=skip_gate_skipped,
            skip_gate_score=skip_gate_score,
            skip_gate_reason=skip_gate_reason,
        )

        logger.info(
            f"Cycle {self._cycle_count} result: "
            f"filled={filled}, wait={queue_wait:.1f}s, "
            f"pnl={post_fill_pnl:.2f}bps" if post_fill_pnl is not None
            else f"Cycle {self._cycle_count} result: filled={filled}, wait={queue_wait:.1f}s"
        )

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

        # 041# 動的 loss_cap: API 残高から算出
        if self.config.loss_cap_auto:
            await self._update_dynamic_loss_cap()

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
        # 024# O4: メモリ制御 — 全レコード保持ではなくカウンタのみ
        total_count = len(existing_records)  # 全件カウント (quarantine 含む)
        filled_count = sum(1 for r in existing_records if r.filled)

        # 033# F4: レジューム時の累積 PnL 計算 (クリーンレコードのみ)
        cumulative_pnl_jpy = 0.0
        for r in clean_records:
            if r.filled and r.post_fill_30s_pnl is not None and r.fill_price:
                cumulative_pnl_jpy += (
                    r.post_fill_30s_pnl * 1e-4 * r.fill_price * r.order_quantity
                )
        del existing_records, clean_records, quarantine_records  # メモリ解放

        batch: list[FillRecord] = list(self._unsaved_batch)  # 前回未保存分を引き継ぐ
        self._unsaved_batch = []
        batch_size = self.config.batch_size  # 032# #18: 設定化

        logger.info(f"Starting fill test: {hours}h, interval={self.config.cycle_interval_sec}s")

        while time.time() < end_time and not self._shutdown_requested:
            # 041# 時間帯フィルター: レコード不生成でスリープ (メトリクス汚染防止)
            if self._is_time_filtered():
                # 047# Issue12: 突入時のみログ出力 (2分毎のノイズ防止)
                if not self._in_time_filter:
                    self._in_time_filter = True
                    utc_h = datetime.now(timezone.utc).hour
                    logger.info(
                        f"[time_filter] Entering High-AS zone (UTC {utc_h}h) "
                        f"— suppressing cycle logs until exit"
                    )
                await asyncio.sleep(self.config.cycle_interval_sec)
                continue
            # 047# Issue12: 離脱時のみログ出力
            if self._in_time_filter:
                self._in_time_filter = False
                logger.info("[time_filter] Exiting High-AS zone — resuming cycles")

            # 041# 残高 pre-flight check: 不足サイドはスキップ
            next_side = self._next_side()
            if await self._check_balance_for_side(next_side):
                # 反対サイドを試す: _last_side を反転して次は反対サイド
                self._last_side = next_side  # → 次の _next_side() が反対を返す
                self._preflight_skip_count += 1

                # 051# P2-3: Balance auto-shrink — 連続3回失敗でロット半減を試行
                # 052#: 最低ロットを _MIN_ORDER_BTC に統一 (Coincheck 0.001 BTC)
                min_lot = max(self.config.order_quantity, self._MIN_ORDER_BTC)
                if (
                    self._preflight_skip_count >= 3
                    and not self._balance_shrink_active
                    and self._current_lot > min_lot
                ):
                    old_lot = self._current_lot
                    self._current_lot = max(
                        min_lot,
                        self._current_lot / 2,
                    )
                    self._balance_shrink_active = True
                    logger.warning(
                        f"[balance_shrink] 連続 preflight 失敗 {self._preflight_skip_count} 回. "
                        f"ロット縮小: {old_lot:.4f} → {self._current_lot:.4f} BTC"
                    )
                    # カウンタリセットして縮小ロットで再試行
                    self._preflight_skip_count = 0
                    await asyncio.sleep(self.config.cycle_interval_sec)
                    continue

                # 044# F8: 連続 preflight 失敗上限 → SAFE_STOP
                if self._preflight_skip_count >= self.config.max_preflight_skip:
                    logger.error(
                        f"SAFE_STOP: 連続 preflight スキップ {self._preflight_skip_count} 回 "
                        f"(上限 {self.config.max_preflight_skip}). "
                        f"buy/sell 両方で残高不足の可能性. 停止します."
                    )
                    self._shutdown_requested = True
                    break
                await asyncio.sleep(self.config.cycle_interval_sec)
                continue

            # preflight 成功 → カウンタリセット
            self._preflight_skip_count = 0
            # 051# P2-3: 成功時に balance_shrink を解除し、ロットを原値に復元
            if self._balance_shrink_active:
                old_lot = self._current_lot
                self._current_lot = self._pre_shrink_lot
                self._balance_shrink_active = False
                logger.info(
                    f"[balance_shrink] 解除: ロット復元 {old_lot:.4f} → {self._current_lot:.4f} BTC"
                )

            # --- サイクル実行 ---
            try:
                record = await self.run_single_cycle()
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt — stopping gracefully")
                self._shutdown_requested = True
                break
            except Exception as e:
                # 024# R2: 例外分類 — サイクル実行エラーは継続可能
                logger.error(f"Cycle execution error: {e}", exc_info=True)
                await asyncio.sleep(self.config.cycle_interval_sec)
                continue

            total_count += 1
            if record.filled:
                filled_count += 1
                # 033# F4: 累積 PnL インクリメンタル追跡
                if record.post_fill_30s_pnl is not None and record.fill_price:
                    cumulative_pnl_jpy += (
                        record.post_fill_30s_pnl * 1e-4
                        * record.fill_price * record.order_quantity
                    )
            batch.append(record)

            # --- 046# soft/hard 二段 loss_cap ---
            # soft cap: ロット半減 (一度だけ)
            if self.config.loss_cap_auto and not self._soft_loss_cap_triggered:
                soft_cap_jpy = (
                    self.config.loss_cap_jpy
                    * self.config.soft_loss_cap_ratio
                    / self.config.loss_cap_ratio
                )
                if cumulative_pnl_jpy <= -soft_cap_jpy:
                    old_lot = self._current_lot
                    self._current_lot = max(
                        self.config.order_quantity,  # 最小ロットは下回らない
                        self._current_lot / 2,
                    )
                    self._soft_loss_cap_triggered = True
                    # 051# P2-3: shrink 復元先も更新
                    self._pre_shrink_lot = self._current_lot
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
                self._shutdown_requested = True

            # --- 049# 即約定防御: queue_wait が閾値以下 + 負エッジのとき次サイクルを保守化 ---
            if self.config.fast_fill_defense_enabled and record.filled:
                is_fast = record.queue_wait_sec <= self.config.fast_fill_threshold_sec
                has_negative_edge = (
                    record.mid_at_fill is not None
                    and record.fill_price is not None
                    and (
                        (record.side == "buy" and record.fill_price > record.mid_at_fill)
                        or (record.side == "sell" and record.fill_price < record.mid_at_fill)
                    )
                )
                if is_fast and has_negative_edge:
                    if not self._fast_fill_boost_active:
                        self._fast_fill_boost_active = True
                        boost = self.config.fast_fill_offset_boost
                        # 050# Bug#1 fix: boost 前の値を保存
                        self._pre_boost_offset = self.config.spread_offset_ratio
                        self._pre_boost_offset_sell = self.config.spread_offset_ratio_sell
                        old_common = self.config.spread_offset_ratio
                        self.config.spread_offset_ratio = min(
                            old_common * boost,
                            0.30,  # max_offset_ratio ハードリミット
                        )
                        # 050# Bug#2 fix: side-specific offset も boost
                        if self.config.spread_offset_ratio_sell is not None:
                            old_sell = self.config.spread_offset_ratio_sell
                            self.config.spread_offset_ratio_sell = min(
                                old_sell * boost, 0.30,
                            )
                            logger.info(
                                f"[fast_fill_defense] Activated: wait={record.queue_wait_sec:.1f}s "
                                f"(< {self.config.fast_fill_threshold_sec}s), "
                                f"negative edge detected. "
                                f"common {old_common:.4f}→{self.config.spread_offset_ratio:.4f}, "
                                f"sell {old_sell:.4f}→{self.config.spread_offset_ratio_sell:.4f}"
                            )
                        else:
                            logger.info(
                                f"[fast_fill_defense] Activated: wait={record.queue_wait_sec:.1f}s "
                                f"(< {self.config.fast_fill_threshold_sec}s), "
                                f"negative edge detected. "
                                f"offset {old_common:.4f}→{self.config.spread_offset_ratio:.4f}"
                            )
                elif self._fast_fill_boost_active:
                    # 正常約定に戻った → boost 解除 + offset 復元
                    old_val = self.config.spread_offset_ratio
                    self.config.spread_offset_ratio = (
                        self._pre_boost_offset
                        if self._pre_boost_offset is not None
                        else self.config.spread_offset_ratio
                    )
                    self.config.spread_offset_ratio_sell = self._pre_boost_offset_sell
                    self._fast_fill_boost_active = False
                    self._pre_boost_offset = None
                    self._pre_boost_offset_sell = None
                    logger.info(
                        "[fast_fill_defense] Deactivated: normal fill detected, "
                        f"offset {old_val:.4f}→{self.config.spread_offset_ratio:.4f}"
                    )
            # --- バッチ保存 (024# R1: 独立 try/except) ---
            if len(batch) >= batch_size:
                if self._try_save_batch(batch):
                    batch = []
                # 失敗時: batch は保持 → 次回再試行

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
            # 055# Fix: _rapid_exit_side は _next_side() で消費するため、ここではクリアしない
            if time.time() < end_time and not self._shutdown_requested:
                if self._rapid_exit_pending:
                    interval = self.config.early_exit_rapid_interval_sec
                    logger.info(
                        f"[early_exit] Rapid exit: interval shortened to "
                        f"{interval:.0f}s (next side={self._rapid_exit_side})"
                    )
                    self._rapid_exit_pending = False
                    # _rapid_exit_side は _next_side() が消費するので保持
                else:
                    interval = self.config.cycle_interval_sec
                await asyncio.sleep(interval)

        # 残りバッチを保存
        if batch:
            if not self._try_save_batch(batch):
                # 最終手段: 緊急ダンプ
                self._emergency_dump(batch, "final")

        logger.info(
            f"Fill test completed: {total_count} cycles, "
            f"{filled_count} filled"
        )
        # 024# O4: 集計用に全レコードをリロード
        return load_fill_records_glob(str(self._results_dir))

    def _try_save_batch(self, batch: list[FillRecord]) -> bool:
        """バッチ保存を試行。失敗時はリトライ + フォールバック.

        024# R1: 保存専用 try/except を分離し、失敗を握り潰さない.
        024# R4: record.timestamp 由来の日付でファイル分割.

        Returns:
            True if save succeeded, False otherwise.
        """
        last_error: Optional[Exception] = None
        for attempt in range(self._max_save_retries):
            try:
                self._save_batch_by_date(batch)
                self._save_fail_count = 0
                return True
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Batch save attempt {attempt + 1}/{self._max_save_retries} "
                    f"failed: {e}",
                    exc_info=True,
                )
                time.sleep(0.5 * (2 ** attempt))  # 指数バックオフ

        # 全リトライ失敗
        self._save_fail_count += 1
        logger.error(
            f"Batch save FAILED after {self._max_save_retries} retries "
            f"(consecutive failures: {self._save_fail_count}): {last_error}"
        )

        # 024# R1: 連続失敗時は緊急ダンプ
        if self._save_fail_count >= self.config.save_fail_threshold:
            self._emergency_dump(batch, "save_fail")
            self._save_fail_count = 0
            return True  # ダンプ成功ならバッチクリア

        # batch は呼び出し元で保持 → 次回再試行
        self._unsaved_batch = list(batch)
        return False

    def _save_batch_by_date(self, batch: list[FillRecord]) -> None:
        """024# R4: record.timestamp 由来の日付でファイル分割保存."""
        # レコードを UTC 日付ごとにグルーピング
        by_date: dict[str, list[FillRecord]] = {}
        for record in batch:
            day_str = datetime.fromtimestamp(
                record.timestamp, tz=timezone.utc
            ).strftime("%Y%m%d")
            by_date.setdefault(day_str, []).append(record)

        for day_str, day_records in by_date.items():
            path = self._results_dir / f"fill_records_{day_str}.jsonl"
            save_fill_records(day_records, path)

    def _emergency_dump(self, batch: list[FillRecord], reason: str) -> None:
        """024# R1: 緊急ダンプ — 通常保存が不可能な場合のフォールバック."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        dump_dir = self._results_dir / "emergency"
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_path = dump_dir / f"emergency_{reason}_{ts}.jsonl"

        try:
            save_fill_records(batch, dump_path)
            logger.warning(
                f"Emergency dump: {len(batch)} records saved to {dump_path}"
            )
        except Exception as e:
            # 最終手段: stderr に直接出力
            import sys
            print(
                f"CRITICAL: Emergency dump also failed: {e}\n"
                f"Unsaved records: {len(batch)}",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)

    def _build_adapt_kwargs(self) -> dict:
        """YAML adaptation セクションから AdaptationConfig 用 kwargs を構築."""
        adapt_yaml = self._yaml_cfg.get("adaptation", {})
        kwargs: dict = {}
        key_map = {
            "min_fill_rate": "min_fill_rate",
            "max_as_ratio": "max_as_ratio",
            "step_ratio": "step_ratio",
            "min_offset_ratio": "min_offset_ratio",
            "max_offset_ratio": "max_offset_ratio",
            "min_samples": "min_samples",
        }
        for yaml_key, config_key in key_map.items():
            if yaml_key in adapt_yaml:
                kwargs[config_key] = adapt_yaml[yaml_key]
        return kwargs

    def _build_lot_kwargs(self) -> dict:
        """YAML lot_sizing セクションから LotSizingConfig 用 kwargs を構築."""
        lot_yaml = self._yaml_cfg.get("lot_sizing", {})
        safety_yaml = self._yaml_cfg.get("safety", {})
        kwargs: dict = {}
        key_map = {
            "min_lot": "min_lot",
            "lot_step": "lot_step",
            "min_fill_rate": "min_fill_rate",
            "max_as_ratio": "max_as_ratio",
            "min_recent_pnl_bps": "min_recent_pnl_bps",
            "min_samples": "min_samples",
        }
        for yaml_key, config_key in key_map.items():
            if yaml_key in lot_yaml:
                kwargs[config_key] = lot_yaml[yaml_key]
        # safety セクションから損失キャップ
        if "loss_cap_jpy" in safety_yaml:
            kwargs["loss_cap_jpy"] = safety_yaml["loss_cap_jpy"]
        if "loss_cap_warning_ratio" in safety_yaml:
            kwargs["loss_cap_warning_ratio"] = safety_yaml["loss_cap_warning_ratio"]
        return kwargs

    async def _update_dynamic_loss_cap(self) -> None:
        """041# 動的 loss_cap: API から口座残高を取得し、残高×比率でキャップを算出.

        失敗時はフォールバック値 (YAML/デフォルト) を維持.
        """
        try:
            btc_price = await self.adapter.get_current_price(self.config.symbol)
            if btc_price is None:
                logger.warning(
                    "[loss_cap] BTC価格取得失敗 — フォールバック値を維持: "
                    f"{self.config.loss_cap_jpy:.0f} JPY"
                )
                return

            balances = await self.adapter.get_balance()
            total_jpy = 0.0
            for b in balances:
                currency = b.currency.upper()
                # 046# E-4 修正後: free+locked=total が正しく解析されるため
                # reserved は total に含まれている (個別チェック不要)
                if currency == "JPY":
                    total_jpy += b.total
                elif currency == "BTC":
                    total_jpy += b.total * btc_price

            if total_jpy <= 0:
                logger.warning(
                    "[loss_cap] 残高ゼロまたは取得不可 — フォールバック値を維持: "
                    f"{self.config.loss_cap_jpy:.0f} JPY"
                )
                return

            new_cap = total_jpy * self.config.loss_cap_ratio
            # 最低50円は保証 (極端に小さいキャップは運用不能)
            new_cap = max(50.0, new_cap)
            old_cap = self.config.loss_cap_jpy
            self.config.loss_cap_jpy = new_cap
            logger.info(
                f"[loss_cap] 動的キャップ算出: 残高={total_jpy:.0f} JPY "
                f"× {self.config.loss_cap_ratio:.0%} = {new_cap:.0f} JPY "
                f"(旧: {old_cap:.0f} JPY)"
            )
        except Exception as e:
            logger.warning(
                f"[loss_cap] 残高取得失敗 — フォールバック値を維持: "
                f"{self.config.loss_cap_jpy:.0f} JPY. error={e}"
            )

    def _try_auto_adapt(self, total_count: int, filled_count: int) -> None:
        """032# P0: 方策 A — fill メトリクスに基づく spread_offset_ratio 自動適応.

        run_continuous のサイクルループ内から呼ばれ、
        fill_rate / AS_ratio に応じて offset を段階調整する。
        """
        try:
            from scripts.v460.lib.param_adapter import (
                AdaptationConfig,
                compute_adaptation,
            )

            # 直近のレコードからメトリクスを算出
            # 047# A1: quarantine レコードを除外し clean のみで適応判断
            all_records = load_fill_records_glob(str(self._results_dir))
            records, _q = filter_clean_records(all_records)
            del all_records
            if len(records) < self.config.min_adapt_samples:
                return

            metrics = compute_fill_metrics(records)
            del records  # メモリ解放

            adapt_config = AdaptationConfig(
                current_offset_ratio=self.config.spread_offset_ratio,
                **self._build_adapt_kwargs(),
            )
            result = compute_adaptation(
                fill_rate=metrics.fill_rate_p90,
                as_ratio=metrics.adverse_selection_ratio,
                sample_count=metrics.total_orders,
                config=adapt_config,
            )

            if result.changed:
                old = self.config.spread_offset_ratio
                self.config.spread_offset_ratio = result.new_offset
                # 052#: side-specific offset も比例調整 (sell offset が独立設定されている場合)
                if self.config.spread_offset_ratio_sell is not None and old > 0:
                    ratio = result.new_offset / old
                    old_sell = self.config.spread_offset_ratio_sell
                    self.config.spread_offset_ratio_sell = min(
                        old_sell * ratio, 0.30,
                    )
                regime_tag = (
                    self._regime_detector.current_regime.value
                    if self._regime_detector else "n/a"
                )
                sell_info = (
                    f", sell {old_sell:.4f}→{self.config.spread_offset_ratio_sell:.4f}"
                    if self.config.spread_offset_ratio_sell is not None else ""
                )
                logger.info(
                    f"[方策A] offset adapted: {old:.4f} → {result.new_offset:.4f} "
                    f"({result.action}: {result.reason}){sell_info} [regime={regime_tag}]"
                )
            else:
                logger.debug(
                    f"[方策A] offset unchanged: {result.reason}"
                )
        except Exception as e:
            logger.warning(f"[方策A] Auto-adapt failed (non-fatal): {e}")

    def _try_auto_lot_size(self) -> None:
        """033# 方策 B — fill メトリクスに基づくロットサイズ自動適応.

        run_continuous のサイクルループ内から呼ばれ、
        fill_rate / AS_ratio / PnL に応じてロットサイズを段階調整する。
        """
        try:
            from scripts.v460.lib.lot_sizer import (
                LotSizingConfig,
                compute_cumulative_pnl_jpy,
                compute_lot_size,
                compute_recent_pnl_bps,
            )

            # 047# A1: quarantine レコードを除外し clean のみでロット判断
            all_records = load_fill_records_glob(str(self._results_dir))
            records, _q = filter_clean_records(all_records)
            del all_records
            if len(records) < self.config.min_adapt_samples:
                return

            metrics = compute_fill_metrics(records)
            cum_pnl = compute_cumulative_pnl_jpy(records)
            recent_pnl = compute_recent_pnl_bps(
                records, window=self.config.recent_pnl_window
            )
            del records  # メモリ解放

            lot_config = LotSizingConfig(
                current_lot=self._current_lot,
                max_lot=self.config.max_lot,
                **self._build_lot_kwargs(),
            )
            result = compute_lot_size(
                fill_rate=metrics.fill_rate_p90,
                as_ratio=metrics.adverse_selection_ratio,
                recent_pnl_bps=recent_pnl,
                cumulative_pnl_jpy=cum_pnl,
                sample_count=metrics.total_orders,
                config=lot_config,
            )

            if result.changed:
                old = self._current_lot
                self._current_lot = result.new_lot
                regime_tag = (
                    self._regime_detector.current_regime.value
                    if self._regime_detector else "n/a"
                )
                logger.info(
                    f"[方策B] lot adapted: {old:.4f} → {result.new_lot:.4f} BTC "
                    f"({result.action}: {result.reason}) [regime={regime_tag}]"
                )
            else:
                logger.debug(
                    f"[方策B] lot unchanged: {result.reason}"
                )
        except Exception as e:
            logger.warning(f"[方策B] Auto lot-size failed (non-fatal): {e}")

    def _cleanup_sync(self) -> None:
        """atexit: 残存注文キャンセル + 未保存データ退避 + ロック解放 (同期 wrapper).

        024# R1: 未保存バッチを緊急ダンプに退避.
        044# A-4: 残存注文キャンセルを確実に実行.
        044# Bug7: ロックファイルを解放.
        """
        # 未保存バッチの退避
        if self._unsaved_batch:
            logger.warning(
                f"Saving {len(self._unsaved_batch)} unsaved records on exit"
            )
            self._emergency_dump(self._unsaved_batch, "atexit")
            self._unsaved_batch = []

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


# ======================================================================
# Results-only mode: 既存データからメトリクス算出
# ======================================================================

def run_results_only(results_dir: str, thresholds_path: str | None = None) -> dict:
    """既存の fill_records JSONL から G1.1 判定を実施."""
    from scripts.v460.lib.config_loader import load_gate_thresholds

    all_records = load_fill_records_glob(results_dir)
    if not all_records:
        logger.error(f"No fill records found in {results_dir}")
        return {"gate": "G1.1-exec", "gate_result": "NO_DATA", "error": "No records found"}

    # 047# A2: quarantine レコードを除外し clean のみで Gate 判定
    records, quarantine = filter_clean_records(all_records)
    if quarantine:
        logger.warning(
            f"[results-only] {len(quarantine)} records quarantined, "
            f"using {len(records)} clean records for judgment"
        )
    del all_records
    if not records:
        logger.error("All records are quarantined")
        return {"gate": "G1.1-exec", "gate_result": "NO_DATA", "error": "All records quarantined"}

    metrics = compute_fill_metrics(records)
    thresholds = load_gate_thresholds().get("g1_1_exec", {})
    judgment = g1_1_judgment(metrics, thresholds)

    logger.info(f"G1.1 Result: {judgment['gate_result']}")
    for check_name, check_data in judgment["checks"].items():
        status = "✓" if check_data["pass"] else "✗"
        logger.info(f"  {status} {check_name}: {check_data['value']:.4f} (threshold: {check_data['threshold']})")

    return judgment


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="G1.1-exec Fill Test Runner (009# §4.2)",
    )
    parser.add_argument("--hours", type=float, default=24.0,
                        help="実測時間 (時間). デフォルト: 24h")
    parser.add_argument("--dry-run", action="store_true",
                        help="Dry-run モード (実際に発注しない)")
    parser.add_argument("--config", default=None,
                        help="設定 YAML パス (デフォルト: configs/v460/fill_test.yaml)")
    # 032# #2: CLI 認証情報は .env からのみ推奨 (後方互換のため残すが非推奨警告)
    parser.add_argument("--api-key", default=None,
                        help="[DEPRECATED] .env から読込を推奨")
    parser.add_argument("--api-secret", default=None,
                        help="[DEPRECATED] .env から読込を推奨")
    parser.add_argument("--results-dir", default=None,
                        help="結果保存ディレクトリ (CLI > YAML)")
    parser.add_argument("--results-only", action="store_true",
                        help="既存データからメトリクスのみ算出")
    parser.add_argument("--cycle-interval", type=float, default=None,
                        help="サイクル間隔 (秒) (CLI > YAML)")
    parser.add_argument("--output", default=None,
                        help="判定結果の JSON 出力先")
    parser.add_argument("--start-side", choices=["buy", "sell"], default=None,
                        help="開始サイド (CLI > YAML)")
    parser.add_argument("--spread-offset-ratio", type=float, default=None,
                        help="スプレッド比例オフセット率 (CLI > YAML)")
    parser.add_argument("--min-spread-jpy", type=float, default=None,
                        help="最小スプレッドフィルター (JPY) (CLI > YAML)")
    parser.add_argument("--enable-auto-adapt", action="store_true", default=False,
                        help="方策A: 自動パラメータ適応を有効化 (CLI > YAML)")
    parser.add_argument("--enable-dynamic-lot", action="store_true", default=False,
                        help="方策B: 動的ロットサイジングを有効化 (CLI > YAML)")
    parser.add_argument("--max-lot", type=float, default=None,
                        help="方策B: ロット上限 (BTC) (CLI > YAML)")
    args = parser.parse_args()

    if args.results_only:
        rd = args.results_dir or "results/v460/fill_test"
        result = run_results_only(rd)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved judgment to {args.output}")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        # 047# A3: FINAL PASS のみ exit 0。INTERIM/PROVISIONAL PASS は exit 2。
        jtype = result.get("judgment_type", "PROVISIONAL")
        gate = result.get("gate_result")
        if gate == "PASS" and jtype == "FINAL":
            sys.exit(0)
        elif gate == "PASS":
            # INTERIM / PROVISIONAL PASS — まだ確定判定ではない
            logger.info(f"Gate PASS but judgment_type={jtype} (not FINAL), exit 2")
            sys.exit(2)
        else:
            sys.exit(1)

    # Adapter setup
    # .env ファイルから API 認証情報を自動読込 (CLI 引数が未指定の場合)
    from dotenv import load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env")
    api_key = os.environ.get("COINCHECK_API_KEY")
    api_secret = os.environ.get("COINCHECK_API_SECRET")

    # 032# #2: CLI引数からの認証情報は非推奨警告付きで後方互換維持
    if args.api_key or args.api_secret:
        logger.warning(
            "WARNING: --api-key/--api-secret はプロセスリストや履歴に平文で残ります。"
            ".env ファイルからの読込を推奨します。"
        )
        api_key = args.api_key or api_key
        api_secret = args.api_secret or api_secret

    if not args.dry_run and not (api_key and api_secret):
        logger.error(
            "API credentials required for live mode. "
            "Set COINCHECK_API_KEY/COINCHECK_API_SECRET in .env"
        )
        sys.exit(1)

    adapter = CoincheckAdapter(
        api_key=api_key,
        api_secret=api_secret,
        dry_run=args.dry_run,
    )

    # --- 設定構築: YAML → CLI override ---
    from scripts.v460.lib.config_loader import load_fill_test_config

    yaml_cfg = load_fill_test_config(args.config)
    config = FillTestConfig.from_yaml(yaml_cfg)

    # CLI 引数が明示指定された場合のみ上書き (None / False はスキップ)
    if args.cycle_interval is not None:
        config.cycle_interval_sec = args.cycle_interval
    if args.results_dir is not None:
        config.results_dir = args.results_dir
    if args.start_side is not None:
        config.start_side = args.start_side
    if args.spread_offset_ratio is not None:
        config.spread_offset_ratio = args.spread_offset_ratio
    if args.min_spread_jpy is not None:
        config.min_spread_jpy = args.min_spread_jpy
    if args.enable_auto_adapt:
        config.enable_auto_adapt = True
    if args.enable_dynamic_lot:
        config.enable_dynamic_lot = True
    if args.max_lot is not None:
        config.max_lot = args.max_lot

    logger.info(
        f"Config loaded: YAML={args.config or 'default'}, "
        f"offset={config.spread_offset_ratio}, lot={config.order_quantity}, "
        f"adapt={config.enable_auto_adapt}, dynamic_lot={config.enable_dynamic_lot}, "
        f"regime={config.enable_regime}, "
        f"time_filter={config.enable_time_filter}, "
        f"loss_cap_auto={config.loss_cap_auto}"
    )

    runner = FillTestRunner(adapter, config, yaml_cfg=yaml_cfg)

    # 024# O3: ログファイル出力 (ローテーション付き)
    log_dir = Path(config.results_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "fill_test.log",
        maxBytes=config.log_max_bytes,
        backupCount=config.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Log file: {log_dir / 'fill_test.log'}")

    # Signal handler for graceful shutdown
    def _signal_handler(signum: int, frame: object) -> None:
        logger.info(f"Signal {signum} received — requesting shutdown")
        runner._shutdown_requested = True

    signal.signal(signal.SIGINT, _signal_handler)
    # 044# A-1: Windows では SIGTERM が未サポート → SIGBREAK を使用
    if platform.system() == "Windows":
        try:
            signal.signal(signal.SIGBREAK, _signal_handler)  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            logger.debug("SIGBREAK not available on this platform")
    else:
        signal.signal(signal.SIGTERM, _signal_handler)

    # Run
    records = asyncio.run(runner.run_continuous(args.hours))

    # Compute metrics & judgment
    if records:
        from scripts.v460.lib.config_loader import load_gate_thresholds

        # 049# §4-#2: clean のみで集計 (quarantine 混在による誤判定防止)
        clean_records, quarantine_records = filter_clean_records(records)
        if quarantine_records:
            logger.info(
                f"[main] quarantine {len(quarantine_records)}/{len(records)} "
                f"records excluded from final metrics"
            )
        metrics = compute_fill_metrics(clean_records)
        thresholds = load_gate_thresholds().get("g1_1_exec", {})
        judgment = g1_1_judgment(metrics, thresholds)

        # 049# §6.1-#4: clean/quarantine/coverage を judgment に追加
        judgment["data_quality"] = {
            "total_records": len(records),
            "clean_records": len(clean_records),
            "quarantine_records": len(quarantine_records),
            "clean_rate": len(clean_records) / len(records) if records else 0.0,
            "quarantine_rate": len(quarantine_records) / len(records) if records else 0.0,
            "as_coverage": metrics.as_coverage,
            "as_raw_coverage": metrics.as_raw_coverage,
        }

        out_str = json.dumps(judgment, indent=2, ensure_ascii=False)
        print(out_str)

        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(out_str)
            logger.info(f"Saved judgment to {args.output}")

        # 049# §4-#1: exit code を results-only と統一
        # FINAL+PASS → 0, INTERIM/PROVISIONAL+PASS → 2, FAIL → 1
        jtype = judgment.get("judgment_type", "PROVISIONAL")
        gate = judgment.get("gate_result")
        if gate == "PASS" and jtype == "FINAL":
            sys.exit(0)
        elif gate == "PASS":
            logger.info(f"Gate PASS but judgment_type={jtype} (not FINAL), exit 2")
            sys.exit(2)
        else:
            sys.exit(1)
    else:
        logger.warning("No records collected")
        sys.exit(1)


if __name__ == "__main__":
    main()
