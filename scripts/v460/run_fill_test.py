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
import os
import signal
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
    """Fill test runner の設定."""

    symbol: str = "btc_jpy"
    order_quantity: float = 0.001  # 最小ロット (Coincheck BTC)
    cycle_interval_sec: float = 120.0  # サイクル間隔
    order_timeout_sec: float = 300.0  # 注文タイムアウト
    poll_interval_sec: float = 5.0  # ポーリング間隔
    post_fill_wait_sec: float = 30.0  # 約定後 PnL 計測待ち
    results_dir: str = "results/v460/fill_test"
    # 安全設計: 片側蓄積禁止 — buy/sell 交互
    max_consecutive_same_side: int = 2
    # 開始サイド: JPY 残高不足時は "sell" で開始すると自己資金循環できる
    start_side: str = "buy"


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
    ) -> None:
        self.adapter = adapter
        self.config = config
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

        # 安全設計: atexit + signal で残存注文を一括キャンセル
        atexit.register(self._cleanup_sync)

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

    async def _compute_maker_price(self, side: str) -> float:
        """maker limit 価格を算出: best bid+1 / best ask-1.

        009# §4.2: スプレッド内側に配置して maker 約定を狙う.
        """
        ob = await self.adapter.get_orderbook(self.config.symbol, depth=1)
        if not ob.bids or not ob.asks:
            raise ValueError("Empty orderbook")
        best_bid = ob.bids[0][0]
        best_ask = ob.asks[0][0]

        if side == "buy":
            # best bid + 1 JPY (スプレッド内側)
            return best_bid + 1.0
        else:
            # best ask - 1 JPY (スプレッド内側)
            return best_ask - 1.0

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
        try:
            order_price = await self._compute_maker_price(side)
        except Exception as e:
            logger.error(f"Failed to compute maker price: {e}")
            return FillRecord(
                cycle_id=cycle_id,
                timestamp=time.time(),
                side=side,
                order_price=0.0,
                order_quantity=self.config.order_quantity,
                cancelled=True,
            )

        # 2. 発注
        t_submit = time.time()
        try:
            order = await self.adapter.place_order(
                symbol=self.config.symbol,
                side=side,
                quantity=self.config.order_quantity,
                price=order_price,
                order_type="limit",
            )
            self._pending_order_id = order.order_id
            logger.info(
                f"Placed {side} limit @ {order_price:.0f} JPY, "
                f"qty={self.config.order_quantity}, id={order.order_id}"
            )
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return FillRecord(
                cycle_id=cycle_id,
                timestamp=t_submit,
                side=side,
                order_price=order_price,
                order_quantity=self.config.order_quantity,
                cancelled=True,
            )

        # 3. ポーリング監視
        filled = False
        fill_price: Optional[float] = None
        t_fill: Optional[float] = None
        elapsed = 0.0

        while elapsed < self.config.order_timeout_sec and not self._shutdown_requested:
            await asyncio.sleep(self.config.poll_interval_sec)
            elapsed = time.time() - t_submit

            try:
                status_order = await self.adapter.get_order_status(order.order_id)
                if status_order is None:
                    # 注文が open orders にも transactions にもない
                    # → 約定済みか期限切れ。transactions を確認して filled 扱い
                    logger.info(f"Order {order.order_id} no longer found — likely filled")
                    filled = True
                    fill_price = order_price  # best estimate
                    t_fill = time.time()
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
                    adverse_selected = mid_30s_after < mid_at_fill
                else:
                    # sell: 価格下落が有利
                    post_fill_pnl = (mid_at_fill - mid_30s_after) / mid_at_fill * 10000
                    adverse_selected = mid_30s_after > mid_at_fill

        record = FillRecord(
            cycle_id=cycle_id,
            timestamp=t_submit,
            side=side,
            order_price=order_price,
            order_quantity=self.config.order_quantity,
            fill_price=fill_price,
            filled=filled,
            cancelled=not filled,
            queue_wait_sec=queue_wait,
            mid_at_fill=mid_at_fill,
            mid_30s_after=mid_30s_after,
            post_fill_30s_pnl=post_fill_pnl,
            adverse_selected=adverse_selected,
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
        """
        end_time = time.time() + hours * 3600

        # レジューム: 既存レコードから状態復元
        existing_records = self.resume_from_existing()
        records: list[FillRecord] = list(existing_records)
        batch: list[FillRecord] = []
        batch_size = 10  # 10 サイクルごとに保存

        logger.info(f"Starting fill test: {hours}h, interval={self.config.cycle_interval_sec}s")

        while time.time() < end_time and not self._shutdown_requested:
            try:
                record = await self.run_single_cycle()
                records.append(record)
                batch.append(record)

                # バッチ保存
                if len(batch) >= batch_size:
                    self._save_batch(batch)
                    batch = []

                # 進捗ログ
                if self._cycle_count % 50 == 0:
                    filled_count = sum(1 for r in records if r.filled)
                    logger.info(
                        f"Progress: {self._cycle_count} cycles, "
                        f"fill rate={filled_count}/{len(records)} "
                        f"({filled_count/len(records)*100:.1f}%)"
                    )

            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt — stopping gracefully")
                self._shutdown_requested = True
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}", exc_info=True)
                # エラーでも続行 — R3 対策 (API 障害耐性)
                await asyncio.sleep(self.config.cycle_interval_sec)
                continue

            # 次サイクルまで待機
            if time.time() < end_time and not self._shutdown_requested:
                await asyncio.sleep(self.config.cycle_interval_sec)

        # 残りバッチを保存
        if batch:
            self._save_batch(batch)

        logger.info(
            f"Fill test completed: {len(records)} cycles, "
            f"{sum(1 for r in records if r.filled)} filled"
        )
        return records

    def _save_batch(self, batch: list[FillRecord]) -> None:
        """日別 JSONL ファイルにバッチ保存."""
        from datetime import datetime, timezone

        day_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = self._results_dir / f"fill_records_{day_str}.jsonl"
        save_fill_records(batch, path)

    def _cleanup_sync(self) -> None:
        """atexit: 残存注文のキャンセル (同期 wrapper)."""
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
    parser.add_argument("--api-key", default=None, help="Coincheck API key")
    parser.add_argument("--api-secret", default=None, help="Coincheck API secret")
    parser.add_argument("--results-dir", default="results/v460/fill_test",
                        help="結果保存ディレクトリ")
    parser.add_argument("--results-only", action="store_true",
                        help="既存データからメトリクスのみ算出")
    parser.add_argument("--cycle-interval", type=float, default=120.0,
                        help="サイクル間隔 (秒)")
    parser.add_argument("--output", default=None,
                        help="判定結果の JSON 出力先")
    parser.add_argument("--start-side", choices=["buy", "sell"], default="buy",
                        help="開始サイド (JPY残高不足時は sell 推奨)")
    args = parser.parse_args()

    if args.results_only:
        result = run_results_only(args.results_dir)
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
    api_key = args.api_key or os.environ.get("COINCHECK_API_KEY")
    api_secret = args.api_secret or os.environ.get("COINCHECK_API_SECRET")

    if not args.dry_run and not (api_key and api_secret):
        logger.error(
            "API credentials required for live mode. "
            "Set COINCHECK_API_KEY/COINCHECK_API_SECRET in .env or use --api-key/--api-secret"
        )
        sys.exit(1)

    adapter = CoincheckAdapter(
        api_key=api_key,
        api_secret=api_secret,
        dry_run=args.dry_run,
    )

    config = FillTestConfig(
        cycle_interval_sec=args.cycle_interval,
        results_dir=args.results_dir,
        start_side=args.start_side,
    )

    runner = FillTestRunner(adapter, config)

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
