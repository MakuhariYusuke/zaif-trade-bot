#!/usr/bin/env python3
"""
Phase 4.5 Gate C1-C3 統合実行ランナー

Gate C0/C1修正後のSAC再実験とベースラインを一括実行。
メモリ分離のため各実験をサブプロセスで実行。

実行順序:
1. ベースライン3種 × 4seed（軽量、先に完了させる）
2. P1-1修正版（use_simple_reward=True, hold_penalty_multiplier=1.0）× 4seed
3. P1-3現行設定 × 4seed
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
RESULTS_DIR = project_root / "results" / "gatec1c3_results"
PYTHON = sys.executable


def run_baselines():
    """ベースライン3種を実行"""
    print("\n" + "=" * 70)
    print("Phase 1/3: ベースライン実行（Gate C3）")
    print("=" * 70)

    script = str(project_root / "scripts" / "v459" / "run_baselines.py")
    proc = subprocess.run(
        [PYTHON, script],
        text=True,
        timeout=7200,  # 2時間
    )
    return proc.returncode == 0


def run_sac_experiments():
    """SAC P1-1/P1-3をサブプロセスで実行"""
    print("\n" + "=" * 70)
    print("Phase 2/3: SAC再実験（Gate C1修正版）")
    print("=" * 70)

    script = str(project_root / "scripts" / "v459" / "run_phase45_p1_subprocess.py")
    proc = subprocess.run(
        [PYTHON, script],
        text=True,
        timeout=28800,  # 8時間
    )
    return proc.returncode == 0


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()

    print("=" * 70)
    print("Phase 4.5 Gate C1-C3 統合実行")
    print(f"開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Phase 3: ベースライン（軽量なので先に実行）
    baseline_ok = run_baselines()
    print(f"\nベースライン: {'✅ 成功' if baseline_ok else '❌ 失敗'}")

    # Phase 2: SAC再実験
    sac_ok = run_sac_experiments()
    print(f"\nSAC再実験: {'✅ 成功' if sac_ok else '❌ 失敗'}")

    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"全実験完了: {elapsed/3600:.1f}時間")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
