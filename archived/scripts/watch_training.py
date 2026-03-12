#!/usr/bin/env python3
"""
学習進捗リアルタイム監視ツール

TensorBoardなしでコマンドラインから学習進捗を確認できます。
1Mロングラン設計の早期停止条件を監視します。

Usage:
    python scripts/watch_training.py --log-dir logs/ensemble_B_100k_test
    python scripts/watch_training.py --log-dir logs/ensemble_B_100k_test --interval 10
    python scripts/watch_training.py --log-dir logs/ensemble_B_100k_test --metrics train/legal_sell_rate eval/sharpe_proxy
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Import Long-Run monitoring constants and functions
from ztb.training.ppo_config import (
    DEFAULT_TOTAL_TIMESTEPS,
    GRAD_NORM_SELL_MIN,
    KL_CRITICAL_THRESHOLD,
    KL_VIOLATION_THRESHOLD,
    MIN_LEGAL_SELL_RATE,
    SHARPE_PATIENCE_EVALS,
    SHARPE_PROXY_THRESHOLD,
    get_sell_rate_patience,
)

try:
    from tensorboard.backend.event_processing import event_accumulator

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠️  TensorBoard not installed. Install with: pip install tensorboard")


def get_latest_value(
    ea: event_accumulator.EventAccumulator, metric: str
) -> Optional[tuple]:
    """指定メトリックの最新値を取得"""
    try:
        scalars = ea.Tags().get("scalars", [])
        if metric in scalars:
            values = ea.Scalars(metric)
            if values:
                latest = values[-1]
                return (latest.step, latest.value, latest.wall_time)
    except Exception:
        pass
    return None


def format_value(value: float, metric: str) -> str:
    """値を適切なフォーマットで表示"""
    # パーセント表示
    if any(keyword in metric.lower() for keyword in ["rate", "ratio", "distribution"]):
        return f"{value:7.2%}"

    # 小数点以下4桁
    if abs(value) < 1000:
        return f"{value:10.4f}"

    # 大きな数値は指数表記
    return f"{value:10.2e}"


def watch_training(
    log_dir: Path,
    interval: int = 10,
    metrics: Optional[List[str]] = None,
    compact: bool = False,
):
    """学習進捗をリアルタイム監視（1M早期停止条件も監視）"""
    if not TENSORBOARD_AVAILABLE:
        print("❌ TensorBoard is required for this tool")
        sys.exit(1)

    if not log_dir.exists():
        print(f"❌ Log directory not found: {log_dir}")
        sys.exit(1)

    # デフォルトの重要指標（1M設計対応）
    if metrics is None:
        metrics = [
            "train/legal_sell_rate",  # Early stop condition 1
            "train/grad_norm(SELL)",  # Early stop condition 2
            "eval/sharpe_proxy",  # Early stop condition 3
            "train/entropy",  # Target entropy monitoring
            "train/kl_divergence",  # KL violation monitoring
            "rollout/ep_rew_mean",
            "train/pan_total_samples",
            "train/loss",
        ]

    print(f"🔍 Monitoring: {log_dir}")
    print(f"📊 Refresh interval: {interval}s")
    print(f"📈 Metrics: {len(metrics)}")
    print("=" * 100)
    print("\n🚨 Early Stop Conditions (1M Long-Run Design):")
    print(
        f"  1️⃣  legal_sell_rate < {MIN_LEGAL_SELL_RATE:.2f} for {SELL_RATE_PATIENCE_STEPS:,} steps"
    )
    print(f"  2️⃣  grad_norm(SELL) ≈ 0 (< {GRAD_NORM_SELL_MIN:.1e})")
    print(
        f"  3️⃣  Sharpe_proxy ≤ {SHARPE_PROXY_THRESHOLD} for {SHARPE_PATIENCE_EVALS} consecutive evals"
    )
    print(
        f"  ⚠️  KL violation: {KL_VIOLATION_THRESHOLD:.1f} (warning), {KL_CRITICAL_THRESHOLD:.1f} (critical)"
    )
    print()
    print("Press Ctrl+C to stop")
    print()

    ea = event_accumulator.EventAccumulator(str(log_dir))
    last_update = None

    # Early stop monitoring state
    # Calculate patience for 1M default (can be overridden if total_timesteps known)
    sell_rate_patience = get_sell_rate_patience(DEFAULT_TOTAL_TIMESTEPS)
    low_sell_rate_streak = 0
    low_sharpe_streak = 0

    try:
        iteration = 0
        while True:
            iteration += 1
            ea.Reload()

            # 利用可能なメトリック
            available_scalars = ea.Tags().get("scalars", [])

            if not compact or iteration % 6 == 1:
                print("\n" + "=" * 100)
                print(
                    f"⏱️  {datetime.now().strftime('%H:%M:%S')} | Iteration {iteration}"
                )
                print("=" * 100)

            metrics_found = 0
            warnings = []

            for metric in metrics:
                result = get_latest_value(ea, metric)
                if result:
                    step, value, wall_time = result
                    formatted_value = format_value(value, metric)

                    # コンパクトモード
                    if compact:
                        status_icon = ""

                        # Check early stop conditions
                        if (
                            metric == "train/legal_sell_rate"
                            and value < MIN_LEGAL_SELL_RATE
                        ):
                            low_sell_rate_streak += 1
                            status_icon = "⚠️ "
                            if low_sell_rate_streak * interval >= sell_rate_patience:
                                warnings.append(
                                    f"🚨 EARLY STOP CONDITION 1: Low sell rate for {low_sell_rate_streak * interval}s"
                                )
                        else:
                            low_sell_rate_streak = 0

                        if (
                            metric == "train/grad_norm(SELL)"
                            and value < GRAD_NORM_SELL_MIN
                        ):
                            status_icon = "🚨 "
                            warnings.append(
                                f"🚨 EARLY STOP CONDITION 2: Gradient collapse (grad_norm={value:.2e})"
                            )

                        if (
                            metric == "eval/sharpe_proxy"
                            and value <= SHARPE_PROXY_THRESHOLD
                        ):
                            low_sharpe_streak += 1
                            status_icon = "⚠️ "
                            if low_sharpe_streak >= SHARPE_PATIENCE_EVALS:
                                warnings.append(
                                    f"🚨 EARLY STOP CONDITION 3: Low Sharpe for {low_sharpe_streak} consecutive evals"
                                )
                        else:
                            low_sharpe_streak = 0

                        if metric == "train/kl_divergence":
                            if value > KL_CRITICAL_THRESHOLD:
                                status_icon = "🔴 "
                                warnings.append(
                                    f"🔴 CRITICAL: KL divergence = {value:.2f} > {KL_CRITICAL_THRESHOLD:.1f}"
                                )
                            elif value > KL_VIOLATION_THRESHOLD:
                                status_icon = "⚠️ "
                                warnings.append(
                                    f"⚠️  WARNING: KL divergence = {value:.2f} > {KL_VIOLATION_THRESHOLD:.1f}"
                                )

                        print(
                            f"{status_icon}{metric:35s}: {formatted_value} (step {step:7d})"
                        )
                    else:
                        # 時刻も表示
                        timestamp = datetime.fromtimestamp(wall_time).strftime(
                            "%H:%M:%S"
                        )
                        print(
                            f"{metric:35s}: {formatted_value} | step {step:7d} | {timestamp}"
                        )

                    metrics_found += 1
                    last_update = wall_time

            # Display warnings
            if warnings:
                print("\n" + "🚨" * 50)
                for warning in warnings:
                    print(warning)
                print("🚨" * 50)

            if metrics_found == 0:
                print("⚠️  No metrics found yet. Waiting for training to start...")
                print(f"   Available metrics: {len(available_scalars)}")
            else:
                if not compact:
                    print(f"\n📊 {metrics_found}/{len(metrics)} metrics displayed")
                    if last_update:
                        age = time.time() - last_update
                        print(f"⏱️  Last update: {age:.1f}s ago")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped monitoring")
        sys.exit(0)


def list_available_metrics(log_dir: Path):
    """利用可能なメトリックを一覧表示"""
    if not TENSORBOARD_AVAILABLE:
        print("❌ TensorBoard is required")
        sys.exit(1)

    if not log_dir.exists():
        print(f"❌ Log directory not found: {log_dir}")
        sys.exit(1)

    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()

    scalars = ea.Tags().get("scalars", [])

    print(f"\n📊 Available metrics in {log_dir}:")
    print("=" * 80)

    categories = {}
    for metric in sorted(scalars):
        category = metric.split("/")[0] if "/" in metric else "other"
        if category not in categories:
            categories[category] = []
        categories[category].append(metric)

    for category, metrics in sorted(categories.items()):
        print(f"\n{category.upper()}:")
        for metric in metrics:
            result = get_latest_value(ea, metric)
            if result:
                step, value, _ = result
                formatted = format_value(value, metric)
                print(f"  {metric:40s}: {formatted} (step {step})")

    print(f"\n📊 Total: {len(scalars)} metrics")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Watch training progress in real-time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor with default metrics
  python scripts/watch_training.py --log-dir logs/ensemble_B_100k_test

  # Custom refresh interval
  python scripts/watch_training.py --log-dir logs/ensemble_B_100k_test --interval 5

  # Custom metrics
  python scripts/watch_training.py --log-dir logs/ensemble_B_100k_test --metrics train/legal_sell_rate eval/sharpe_proxy

  # Compact mode (less verbose)
  python scripts/watch_training.py --log-dir logs/ensemble_B_100k_test --compact

  # List available metrics
  python scripts/watch_training.py --log-dir logs/ensemble_B_100k_test --list
        """,
    )

    parser.add_argument(
        "--log-dir", required=True, help="Path to TensorBoard log directory"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Refresh interval in seconds (default: 10)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="Custom metrics to monitor (default: predefined important metrics)",
    )
    parser.add_argument(
        "--compact", action="store_true", help="Compact output mode (less verbose)"
    )
    parser.add_argument(
        "--list", action="store_true", help="List available metrics and exit"
    )

    args = parser.parse_args()

    log_dir = Path(args.log_dir)

    if args.list:
        list_available_metrics(log_dir)
    else:
        watch_training(
            log_dir=log_dir,
            interval=args.interval,
            metrics=args.metrics,
            compact=args.compact,
        )


if __name__ == "__main__":
    main()
