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
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ztb.metrics.fill_quality import (
    FillRecord,
    compute_fill_metrics,
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
    # 安全設計: 片側蓄積禁止 — buy/sell 交互
    max_consecutive_same_side: int = 2
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
    # 方策 A: パラメータ適応
    enable_auto_adapt: bool = False
    adapt_interval_cycles: int = 50
    # 方策 B: 動的ロットサイジング
    enable_dynamic_lot: bool = False
    max_lot: float = 0.005
    lot_adapt_interval_cycles: int = 50
    # 安全設計 (000# §3.9)
    loss_cap_jpy: float = 10_000.0
    loss_cap_warning_ratio: float = 0.7

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
            "max_consecutive_same_side", "start_side",
            "spread_offset_ratio", "min_offset_jpy",
            "max_order_retries", "retry_delay_sec",
            "as_deadzone_bps", "min_spread_jpy",
            "batch_size", "max_save_retries",
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

        # safety セクション → 損失キャップ
        safety = yaml_cfg.get("safety", {})
        if "loss_cap_jpy" in safety:
            kwargs["loss_cap_jpy"] = safety["loss_cap_jpy"]
        if "loss_cap_warning_ratio" in safety:
            kwargs["loss_cap_warning_ratio"] = safety["loss_cap_warning_ratio"]

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
        self._same_side_count = 0
        self._shutdown_requested = False
        self._pending_order_id: Optional[str] = None

        # 020# O4: データバージョン管理
        self._run_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        self._git_sha = self._get_git_sha()

        # 033# 方策 B: 動的ロットの実行時数量 (config.order_quantity を初期値とする)
        self._current_lot: float = config.order_quantity

        # 024# R1: 保存失敗トラッキング
        self._unsaved_batch: list[FillRecord] = []
        self._save_fail_count: int = 0
        self._max_save_retries: int = config.max_save_retries

        # 安全設計: atexit + signal で残存注文キャンセル + 未保存データ退避
        atexit.register(self._cleanup_sync)

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
        """buy/sell を交互に返す.

        009# §4.2: 片側ポジション蓄積禁止.
        """
        if self._last_side is None or self._last_side == "sell":
            return "buy"
        return "sell"

    async def _get_mid_price(self) -> float:
        """板の best bid/ask から mid price を算出."""
        ob = await self.adapter.get_orderbook(self.config.symbol, depth=1)
        if not ob.bids or not ob.asks:
            raise ValueError("Empty orderbook — cannot compute mid price")
        best_bid = ob.bids[0][0]
        best_ask = ob.asks[0][0]
        return (best_bid + best_ask) / 2.0

    async def _compute_maker_price(self, side: str) -> tuple[float, float]:
        """maker limit 価格を算出: スプレッド比例オフセット + post_only 安全策.

        009# §4.2: スプレッド内側に配置して maker 約定を狙う.
        CM-1: 固定 1 JPY → スプレッド比例 + post_only リジェクト防止.

        Returns:
            (price, spread_at_order) タプル.
        """
        ob = await self.adapter.get_orderbook(self.config.symbol, depth=1)
        if not ob.bids or not ob.asks:
            raise ValueError("Empty orderbook")
        best_bid = ob.bids[0][0]
        best_ask = ob.asks[0][0]
        spread = best_ask - best_bid

        # 031# スプレッドフィルター: 狭すぎる場合はスキップ
        if spread < self.config.min_spread_jpy:
            raise ValueError(
                f"Spread too narrow: {spread:.0f} JPY < min {self.config.min_spread_jpy:.0f}"
            )

        # スプレッド比例オフセット (最小保証付き)
        offset = max(self.config.min_offset_jpy, spread * self.config.spread_offset_ratio)

        if side == "buy":
            price = best_bid + offset
            # CM-1: post_only ガード — best_ask 以上にならないよう保護
            if price >= best_ask:
                price = best_bid  # best bid に退避 (確実に maker)
                logger.info(
                    f"Spread guard: buy price {best_bid + offset:.0f} >= ask {best_ask:.0f}, "
                    f"fallback to best_bid {best_bid:.0f} (spread={spread:.0f})"
                )
            return price, spread
        else:
            price = best_ask - offset
            # CM-1: post_only ガード — best_bid 以下にならないよう保護
            if price <= best_bid:
                price = best_ask  # best ask に退避 (確実に maker)
                logger.info(
                    f"Spread guard: sell price {best_ask - offset:.0f} <= bid {best_bid:.0f}, "
                    f"fallback to best_ask {best_ask:.0f} (spread={spread:.0f})"
                )
            return price, spread

    async def run_single_cycle(self) -> FillRecord:
        """1 サイクル: 発注 → 監視 → 結果記録.

        009# §4.2 の流れに準拠.
        """
        self._cycle_count += 1
        cycle_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        side = self._next_side()
        self._last_side = side

        logger.info(f"=== Cycle {self._cycle_count} ({side}) ===")

        # 1. maker limit 価格算出
        spread_at_order: Optional[float] = None
        try:
            order_price, spread_at_order = await self._compute_maker_price(side)
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
                elif "insufficient" in err_lower or "balance" in err_lower:
                    cancel_reason = "insufficient_funds"
                elif "minimum" in err_lower or "size" in err_lower:
                    cancel_reason = "minimum_size"
                else:
                    cancel_reason = "api_error"

                logger.warning(
                    f"Order attempt {attempt + 1} failed ({cancel_reason}): {e}"
                )

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

        self._pending_order_id = None
        queue_wait = elapsed

        # 5. 約定後 30 秒の mid price 計測
        mid_at_fill: Optional[float] = None
        mid_30s_after: Optional[float] = None
        post_fill_pnl: Optional[float] = None
        adverse_selected: Optional[bool] = None
        adverse_selected_raw: Optional[bool] = None

        if filled and fill_price is not None:
            try:
                mid_at_fill = await self._get_mid_price()
            except Exception:
                pass

            # 30 秒待機
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
            post_fill_30s_pnl=post_fill_pnl,
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
            spread_offset_ratio=self.config.spread_offset_ratio,
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

        # レジューム: 既存レコードから状態復元
        existing_records = self.resume_from_existing()
        # 024# O4: メモリ制御 — 全レコード保持ではなくカウンタのみ
        total_count = len(existing_records)
        filled_count = sum(1 for r in existing_records if r.filled)

        # 033# F4: レジューム時の累積 PnL 計算
        cumulative_pnl_jpy = 0.0
        for r in existing_records:
            if r.filled and r.post_fill_30s_pnl is not None and r.fill_price:
                cumulative_pnl_jpy += (
                    r.post_fill_30s_pnl * 1e-4 * r.fill_price * r.order_quantity
                )
        del existing_records  # メモリ解放

        batch: list[FillRecord] = list(self._unsaved_batch)  # 前回未保存分を引き継ぐ
        self._unsaved_batch = []
        batch_size = self.config.batch_size  # 032# #18: 設定化

        logger.info(f"Starting fill test: {hours}h, interval={self.config.cycle_interval_sec}s")

        while time.time() < end_time and not self._shutdown_requested:
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

            # --- 033# F4: 累積 PnL 安全キャップ (000# §3.9) ---
            if cumulative_pnl_jpy <= -self.config.loss_cap_jpy:
                logger.error(
                    f"LOSS CAP REACHED: cumulative PnL = {cumulative_pnl_jpy:.0f} JPY "
                    f"(cap = -{self.config.loss_cap_jpy:.0f} JPY). Stopping fill test."
                )
                self._shutdown_requested = True

            # --- バッチ保存 (024# R1: 独立 try/except) ---
            if len(batch) >= batch_size:
                if self._try_save_batch(batch):
                    batch = []
                # 失敗時: batch は保持 → 次回再試行

            # 進捗ログ
            if self._cycle_count % 50 == 0:
                logger.info(
                    f"Progress: {self._cycle_count} cycles, "
                    f"fill rate={filled_count}/{total_count} "
                    f"({filled_count/total_count*100:.1f}%), "
                    f"cumPnL={cumulative_pnl_jpy:.1f}JPY, "
                    f"lot={self._current_lot:.4f}BTC, "
                    f"unsaved_batch={len(batch)}"
                )

            # --- 032# P0: 方策 A パラメータ適応 ---
            if (
                self.config.enable_auto_adapt
                and self._cycle_count % self.config.adapt_interval_cycles == 0
                and total_count >= 50
            ):
                self._try_auto_adapt(total_count, filled_count)

            # --- 033# 方策 B: 動的ロットサイジング ---
            if (
                self.config.enable_dynamic_lot
                and self._cycle_count % self.config.lot_adapt_interval_cycles == 0
                and total_count >= 50
            ):
                self._try_auto_lot_size()

            # 次サイクルまで待機
            if time.time() < end_time and not self._shutdown_requested:
                await asyncio.sleep(self.config.cycle_interval_sec)

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
        from datetime import datetime, timezone

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
        if self._save_fail_count >= 3:
            self._emergency_dump(batch, "save_fail")
            self._save_fail_count = 0
            return True  # ダンプ成功ならバッチクリア

        # batch は呼び出し元で保持 → 次回再試行
        self._unsaved_batch = list(batch)
        return False

    def _save_batch_by_date(self, batch: list[FillRecord]) -> None:
        """024# R4: record.timestamp 由来の日付でファイル分割保存."""
        from datetime import datetime, timezone

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
        import traceback
        from datetime import datetime, timezone

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
            records = load_fill_records_glob(str(self._results_dir))
            if len(records) < 50:
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
                logger.info(
                    f"[方策A] offset adapted: {old:.4f} → {result.new_offset:.4f} "
                    f"({result.action}: {result.reason})"
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

            records = load_fill_records_glob(str(self._results_dir))
            if len(records) < 50:
                return

            metrics = compute_fill_metrics(records)
            cum_pnl = compute_cumulative_pnl_jpy(records)
            recent_pnl = compute_recent_pnl_bps(records, window=50)
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
                logger.info(
                    f"[方策B] lot adapted: {old:.4f} → {result.new_lot:.4f} BTC "
                    f"({result.action}: {result.reason})"
                )
            else:
                logger.debug(
                    f"[方策B] lot unchanged: {result.reason}"
                )
        except Exception as e:
            logger.warning(f"[方策B] Auto lot-size failed (non-fatal): {e}")

    def _cleanup_sync(self) -> None:
        """atexit: 残存注文キャンセル + 未保存データ退避 (同期 wrapper).

        024# R1: 未保存バッチを緊急ダンプに退避.
        """
        # 未保存バッチの退避
        if self._unsaved_batch:
            logger.warning(
                f"Saving {len(self._unsaved_batch)} unsaved records on exit"
            )
            self._emergency_dump(self._unsaved_batch, "atexit")
            self._unsaved_batch = []

        # 残存注文のキャンセル
        if self._pending_order_id:
            logger.warning(f"Cleaning up pending order: {self._pending_order_id}")
            try:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    loop.create_task(
                        self.adapter.cancel_order(self._pending_order_id)
                    )
                else:
                    asyncio.run(
                        self.adapter.cancel_order(self._pending_order_id)
                    )
            except Exception as e:
                logger.error(f"Cleanup failed: {e}")


# ======================================================================
# Results-only mode: 既存データからメトリクス算出
# ======================================================================

def run_results_only(results_dir: str, thresholds_path: str | None = None) -> dict:
    """既存の fill_records JSONL から G1.1 判定を実施."""
    from scripts.v460.lib.config_loader import load_gate_thresholds

    records = load_fill_records_glob(results_dir)
    if not records:
        logger.error(f"No fill records found in {results_dir}")
        return {"gate": "G1.1-exec", "gate_result": "NO_DATA", "error": "No records found"}

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
        sys.exit(0 if result.get("gate_result") == "PASS" else 1)

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
        f"adapt={config.enable_auto_adapt}, dynamic_lot={config.enable_dynamic_lot}"
    )

    runner = FillTestRunner(adapter, config, yaml_cfg=yaml_cfg)

    # 024# O3: ログファイル出力 (ローテーション付き)
    log_dir = Path(args.results_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "fill_test.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
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
    signal.signal(signal.SIGTERM, _signal_handler)

    # Run
    records = asyncio.run(runner.run_continuous(args.hours))

    # Compute metrics & judgment
    if records:
        from scripts.v460.lib.config_loader import load_gate_thresholds

        metrics = compute_fill_metrics(records)
        thresholds = load_gate_thresholds().get("g1_1_exec", {})
        judgment = g1_1_judgment(metrics, thresholds)

        out_str = json.dumps(judgment, indent=2, ensure_ascii=False)
        print(out_str)

        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(out_str)
            logger.info(f"Saved judgment to {args.output}")

        sys.exit(0 if judgment["gate_result"] == "PASS" else 1)
    else:
        logger.warning("No records collected")
        sys.exit(1)


if __name__ == "__main__":
    main()
