#!/usr/bin/env python3
"""
チェックポイント性能比較ツール

複数のチェックポイントの性能を一覧表示し、最良のモデルを特定します。

Usage:
    python scripts/compare_checkpoints.py --checkpoint-dir checkpoints/ensemble_B_100k_test
    python scripts/compare_checkpoints.py --checkpoint-dir checkpoints/ensemble_B_100k_test --metrics train/legal_sell_rate eval/sharpe_proxy
    python scripts/compare_checkpoints.py --checkpoint-dir checkpoints/ensemble_B_100k_test --top 5
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

try:
    from tensorboard.backend.event_processing import event_accumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠️  TensorBoard not installed. Install with: pip install tensorboard")


@dataclass
class CheckpointMetrics:
    """チェックポイントのメトリック情報"""
    checkpoint_name: str
    step: int
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def get_score(self, primary_metric: str = "eval/sharpe_proxy") -> float:
        """スコア取得（ソート用）"""
        return self.metrics.get(primary_metric, float('-inf'))


def get_metric_at_step(
    ea: event_accumulator.EventAccumulator,
    metric: str,
    target_step: int,
    tolerance: int = 5000
) -> Optional[float]:
    """指定ステップ付近のメトリック値を取得"""
    try:
        values = ea.Scalars(metric)
        if not values:
            return None
        
        # target_stepに最も近い値を探す
        closest = min(values, key=lambda x: abs(x.step - target_step))
        
        # 許容範囲内かチェック
        if abs(closest.step - target_step) <= tolerance:
            return closest.value
        
    except Exception:
        pass
    
    return None


def compare_checkpoints(
    checkpoint_dir: Path,
    metrics: Optional[List[str]] = None,
    top_n: Optional[int] = None,
    primary_metric: str = "eval/sharpe_proxy",
) -> List[CheckpointMetrics]:
    """チェックポイントの性能を比較"""
    if not TENSORBOARD_AVAILABLE:
        print("❌ TensorBoard is required for this tool")
        sys.exit(1)
    
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)
    
    # デフォルトメトリック
    if metrics is None:
        metrics = [
            "train/legal_sell_rate",
            "eval/sharpe_proxy",
            "train/entropy",
            "rollout/ep_rew_mean",
            "train/grad_norm(SELL)",
        ]
    
    # チェックポイントディレクトリを探す
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*"))
    
    if not checkpoints:
        print(f"❌ No checkpoints found in {checkpoint_dir}")
        sys.exit(1)
    
    # 対応するログディレクトリを探す
    log_dir = checkpoint_dir.parent.parent / "logs" / checkpoint_dir.name
    
    if not log_dir.exists():
        print(f"⚠️  Log directory not found: {log_dir}")
        print(f"   Trying parent directory...")
        # logsディレクトリを探す
        possible_log_dirs = [
            checkpoint_dir.parent / "logs" / checkpoint_dir.name,
            checkpoint_dir.parent / "logs",
            Path("logs") / checkpoint_dir.name,
        ]
        
        log_dir = None
        for possible_dir in possible_log_dirs:
            if possible_dir.exists():
                log_dir = possible_dir
                print(f"   Found: {log_dir}")
                break
        
        if log_dir is None:
            print(f"❌ Could not find log directory for {checkpoint_dir}")
            sys.exit(1)
    
    # EventAccumulator初期化
    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()
    
    # 各チェックポイントのメトリックを取得
    checkpoint_metrics_list = []
    
    for ckpt in checkpoints:
        step = int(ckpt.name.split("_")[-1])
        
        ckpt_metrics = CheckpointMetrics(
            checkpoint_name=ckpt.name,
            step=step,
        )
        
        for metric in metrics:
            value = get_metric_at_step(ea, metric, step)
            if value is not None:
                ckpt_metrics.metrics[metric] = value
        
        checkpoint_metrics_list.append(ckpt_metrics)
    
    # primary_metricでソート（降順）
    checkpoint_metrics_list.sort(key=lambda x: x.get_score(primary_metric), reverse=True)
    
    # top_nに制限
    if top_n:
        checkpoint_metrics_list = checkpoint_metrics_list[:top_n]
    
    return checkpoint_metrics_list


def print_comparison_table(
    checkpoint_metrics_list: List[CheckpointMetrics],
    primary_metric: str = "eval/sharpe_proxy",
):
    """比較表を出力"""
    if not checkpoint_metrics_list:
        print("❌ No checkpoint metrics to display")
        return
    
    # ヘッダー
    print("=" * 120)
    print(f"📊 Checkpoint Comparison (sorted by {primary_metric})")
    print("=" * 120)
    print()
    
    # メトリック名を取得
    all_metrics = set()
    for cm in checkpoint_metrics_list:
        all_metrics.update(cm.metrics.keys())
    
    metrics = sorted(all_metrics)
    
    # テーブルヘッダー
    header = f"{'Rank':5s} {'Checkpoint':25s} {'Step':10s}"
    for metric in metrics:
        # メトリック名を短縮
        short_name = metric.split('/')[-1] if '/' in metric else metric
        header += f" {short_name:12s}"
    
    print(header)
    print("-" * 120)
    
    # 各チェックポイント
    for rank, cm in enumerate(checkpoint_metrics_list, 1):
        # 最良モデルにマーク
        marker = "⭐" if rank == 1 else f"{rank:2d}"
        
        row = f"{marker:5s} {cm.checkpoint_name:25s} {cm.step:10d}"
        
        for metric in metrics:
            value = cm.metrics.get(metric)
            if value is not None:
                # フォーマット
                if any(kw in metric.lower() for kw in ['rate', 'ratio']):
                    formatted = f"{value:11.2%}"
                elif abs(value) < 1000:
                    formatted = f"{value:12.4f}"
                else:
                    formatted = f"{value:12.2e}"
                
                row += f" {formatted}"
            else:
                row += f" {'N/A':12s}"
        
        print(row)
    
    print()
    print("=" * 120)
    
    # ベストモデル情報
    best = checkpoint_metrics_list[0]
    print(f"\n⭐ Best Model: {best.checkpoint_name} (step {best.step})")
    print(f"   {primary_metric}: {best.metrics.get(primary_metric, 'N/A')}")
    print()


def export_to_csv(
    checkpoint_metrics_list: List[CheckpointMetrics],
    output_path: Path,
):
    """CSV形式でエクスポート"""
    import csv
    
    if not checkpoint_metrics_list:
        print("❌ No data to export")
        return
    
    # メトリック名を取得
    all_metrics = set()
    for cm in checkpoint_metrics_list:
        all_metrics.update(cm.metrics.keys())
    
    metrics = sorted(all_metrics)
    
    # CSVヘッダー
    fieldnames = ['rank', 'checkpoint', 'step'] + metrics
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for rank, cm in enumerate(checkpoint_metrics_list, 1):
            row = {
                'rank': rank,
                'checkpoint': cm.checkpoint_name,
                'step': cm.step,
            }
            row.update(cm.metrics)
            writer.writerow(row)
    
    print(f"✅ Exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare checkpoint performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all checkpoints
  python scripts/compare_checkpoints.py --checkpoint-dir checkpoints/ensemble_B_100k_test
  
  # Top 5 checkpoints only
  python scripts/compare_checkpoints.py --checkpoint-dir checkpoints/ensemble_B_100k_test --top 5
  
  # Custom metrics
  python scripts/compare_checkpoints.py --checkpoint-dir checkpoints/ensemble_B_100k_test --metrics train/legal_sell_rate eval/sharpe_proxy
  
  # Export to CSV
  python scripts/compare_checkpoints.py --checkpoint-dir checkpoints/ensemble_B_100k_test --export results.csv
        """
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--metrics",
        nargs='+',
        help="Metrics to compare (default: predefined metrics)"
    )
    parser.add_argument(
        "--top",
        type=int,
        help="Show only top N checkpoints"
    )
    parser.add_argument(
        "--primary-metric",
        default="eval/sharpe_proxy",
        help="Primary metric for sorting (default: eval/sharpe_proxy)"
    )
    parser.add_argument(
        "--export",
        help="Export results to CSV file"
    )
    
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    
    # 比較実行
    checkpoint_metrics_list = compare_checkpoints(
        checkpoint_dir=checkpoint_dir,
        metrics=args.metrics,
        top_n=args.top,
        primary_metric=args.primary_metric,
    )
    
    # 結果表示
    print_comparison_table(checkpoint_metrics_list, args.primary_metric)
    
    # CSV出力
    if args.export:
        export_to_csv(checkpoint_metrics_list, Path(args.export))


if __name__ == "__main__":
    main()
