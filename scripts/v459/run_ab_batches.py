#!/usr/bin/env python3
"""
Phase 3 Day 4-5: AB Reward Experiments - Batch Execution Plan
フル48実験をバッチ実行するスクリプト

メモリ制約を考慮し、4バッチに分割して実行
各バッチ後にPythonプロセスを再起動してメモリをクリア
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import subprocess
import time
import json
from datetime import datetime
from typing import List, Dict, Any

# =============================================================================
# 実験計画: 4 Seeds × 3 Stages = 12実験 (各実験でWalk-Forward 4窓 = 48サンプル)
# =============================================================================

SEEDS = [42, 123, 456, 789]
STAGES = ["stage1_basic", "stage2_extended", "stage3_advanced"]

# バッチ分割（メモリ制約対策）
BATCHES = [
    # Batch 1: Seed 42 (全3ステージ)
    {"seeds": [42], "description": "Batch 1: Seed 42 (Baseline)"},
    
    # Batch 2: Seed 123 (全3ステージ)
    {"seeds": [123], "description": "Batch 2: Seed 123"},
    
    # Batch 3: Seed 456 (全3ステージ)
    {"seeds": [456], "description": "Batch 3: Seed 456"},
    
    # Batch 4: Seed 789 (全3ステージ)
    {"seeds": [789], "description": "Batch 4: Seed 789"},
]

RESULTS_DIR = project_root / "results" / "ab_rewards"
BATCH_LOG_FILE = RESULTS_DIR / "batch_execution_log.json"


class BatchExecutionManager:
    """バッチ実行マネージャー"""
    
    def __init__(self):
        self.batch_log = []
        self.start_time = None
        
        # 既存ログを読み込み
        if BATCH_LOG_FILE.exists():
            with open(BATCH_LOG_FILE, "r", encoding="utf-8") as f:
                self.batch_log = json.load(f)
    
    def run_batch(self, batch_num: int, seeds: List[int], description: str) -> Dict[str, Any]:
        """単一バッチの実行"""
        print("=" * 80)
        print(f"🚀 {description}")
        print(f"   Seeds: {seeds}")
        print(f"   Expected Experiments: {len(seeds) * len(STAGES)}")
        print("=" * 80)
        
        batch_start = time.time()
        
        # run_ab_reward_experiments.py を呼び出し
        cmd = [
            sys.executable,
            str(project_root / "scripts" / "v459" / "run_ab_reward_experiments.py"),
            "--seeds", *[str(s) for s in seeds],
        ]
        
        print(f"\n📝 Command: {' '.join(cmd)}\n")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace"
            )
            
            batch_elapsed = time.time() - batch_start
            
            success = result.returncode == 0
            
            batch_result = {
                "batch_num": batch_num,
                "description": description,
                "seeds": seeds,
                "timestamp": datetime.now().isoformat(),
                "elapsed_seconds": batch_elapsed,
                "success": success,
                "return_code": result.returncode,
                "stdout_lines": len(result.stdout.splitlines()) if result.stdout else 0,
                "stderr_lines": len(result.stderr.splitlines()) if result.stderr else 0,
            }
            
            # エラーがあれば記録
            if not success and result.stderr:
                batch_result["error_sample"] = result.stderr[-500:]  # 最後500文字
            
            self.batch_log.append(batch_result)
            self._save_log()
            
            if success:
                print(f"\n✅ Batch {batch_num} completed successfully ({batch_elapsed:.1f}s)")
            else:
                print(f"\n❌ Batch {batch_num} failed (return code: {result.returncode})")
                if result.stderr:
                    print(f"   Error sample: {result.stderr[-200:]}")
            
            return batch_result
            
        except Exception as e:
            print(f"\n❌ Batch {batch_num} exception: {e}")
            batch_result = {
                "batch_num": batch_num,
                "description": description,
                "seeds": seeds,
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "exception": str(e)
            }
            self.batch_log.append(batch_result)
            self._save_log()
            return batch_result
    
    def run_all_batches(self) -> Dict[str, Any]:
        """全バッチの実行"""
        self.start_time = time.time()
        
        print("=" * 80)
        print("🎯 Phase 3 AB Reward Experiments - Full Batch Execution")
        print("=" * 80)
        print(f"Total Batches: {len(BATCHES)}")
        print(f"Total Experiments: {len(SEEDS) * len(STAGES)}")
        print(f"Total Samples (4 windows): {len(SEEDS) * len(STAGES) * 4}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print()
        
        successful_batches = 0
        failed_batches = 0
        
        for i, batch_config in enumerate(BATCHES, start=1):
            batch_result = self.run_batch(
                batch_num=i,
                seeds=batch_config["seeds"],
                description=batch_config["description"]
            )
            
            if batch_result["success"]:
                successful_batches += 1
            else:
                failed_batches += 1
            
            # バッチ間でメモリクリア待機（最後のバッチ以外）
            if i < len(BATCHES):
                print(f"\n⏳ Waiting 30 seconds for memory cleanup...\n")
                time.sleep(30)
        
        total_elapsed = time.time() - self.start_time
        
        summary = {
            "total_batches": len(BATCHES),
            "successful_batches": successful_batches,
            "failed_batches": failed_batches,
            "total_elapsed_seconds": total_elapsed,
            "elapsed_hours": total_elapsed / 3600,
            "completion_time": datetime.now().isoformat()
        }
        
        print("\n" + "=" * 80)
        print("📊 BATCH EXECUTION SUMMARY")
        print("=" * 80)
        print(f"Total Batches: {summary['total_batches']}")
        print(f"Successful: {summary['successful_batches']}")
        print(f"Failed: {summary['failed_batches']}")
        print(f"Total Time: {summary['elapsed_hours']:.2f} hours")
        print(f"Completion: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        self._save_summary(summary)
        
        return summary
    
    def _save_log(self):
        """ログを保存"""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(BATCH_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(self.batch_log, f, indent=2, ensure_ascii=False)
    
    def _save_summary(self, summary: Dict[str, Any]):
        """サマリーを保存"""
        summary_file = RESULTS_DIR / "batch_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Summary saved: {summary_file}")


def main():
    """メイン実行"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Execute AB Reward Experiments in batches"
    )
    parser.add_argument(
        "--batch",
        type=int,
        help="Execute specific batch only (1-4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show execution plan without running"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("=" * 80)
        print("DRY RUN: Execution Plan")
        print("=" * 80)
        for i, batch in enumerate(BATCHES, start=1):
            print(f"\nBatch {i}: {batch['description']}")
            print(f"  Seeds: {batch['seeds']}")
            print(f"  Experiments: {len(batch['seeds']) * len(STAGES)}")
        print("\n" + "=" * 80)
        return 0
    
    manager = BatchExecutionManager()
    
    if args.batch:
        # 特定バッチのみ実行
        if not (1 <= args.batch <= len(BATCHES)):
            print(f"❌ Invalid batch number: {args.batch} (must be 1-{len(BATCHES)})")
            return 1
        
        batch_config = BATCHES[args.batch - 1]
        result = manager.run_batch(
            batch_num=args.batch,
            seeds=batch_config["seeds"],
            description=batch_config["description"]
        )
        
        return 0 if result["success"] else 1
    
    else:
        # 全バッチ実行
        summary = manager.run_all_batches()
        return 0 if summary["failed_batches"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
