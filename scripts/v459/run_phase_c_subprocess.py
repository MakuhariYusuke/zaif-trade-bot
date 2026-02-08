#!/usr/bin/env python3
"""
Phase C サブプロセスランナー — メモリ隔離版

各実験をサブプロセスで実行し、メモリリークを防止。
run_phase_c.py の --single-run モードを呼び出す。
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

PYTHON = sys.executable
SCRIPT_PATH = Path(__file__).parent / "run_phase_c.py"
OUTPUT_DIR = project_root / "results" / "phase_c"
LOG_DIR = OUTPUT_DIR / "logs"


def run_experiment_subprocess(
    experiment: str,
    seed: int,
    timeout: int = 3600,
) -> Dict[str, Any]:
    """1実験をサブプロセスで実行"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    stdout_log = LOG_DIR / f"{experiment}_seed{seed}.stdout.log"
    stderr_log = LOG_DIR / f"{experiment}_seed{seed}.stderr.log"
    
    cmd = [
        PYTHON, str(SCRIPT_PATH),
        "--single-run",
        "--experiment", experiment,
        "--seed", str(seed),
    ]
    
    print(f"\n[subprocess] {experiment} seed={seed} ...", flush=True)
    start_time = time.time()
    
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(project_root),
        )
        
        elapsed = time.time() - start_time
        
        # ログ保存
        stdout_log.write_text(proc.stdout, encoding="utf-8")
        stderr_log.write_text(proc.stderr, encoding="utf-8")
        
        if proc.returncode != 0:
            print(f"  FAILED (exit={proc.returncode}, {elapsed:.0f}s)")
            # stderr末尾を表示
            err_lines = proc.stderr.strip().split("\n")[-5:]
            for line in err_lines:
                print(f"    {line}")
            return {
                "experiment": experiment,
                "seed": seed,
                "success": False,
                "error": f"exit_code={proc.returncode}",
                "stderr_tail": "\n".join(err_lines),
            }
        
        # JSON結果をstdout末尾から抽出
        result = None
        for line in reversed(proc.stdout.strip().split("\n")):
            line = line.strip()
            if line.startswith("{"):
                try:
                    result = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        
        if result is None:
            print(f"  WARNING: JSON parse failed ({elapsed:.0f}s)")
            return {
                "experiment": experiment,
                "seed": seed,
                "success": False,
                "error": "JSON parse failed",
            }
        
        # サマリ表示
        g2 = result.get("gate2", {})
        roi = result.get("net_roi", "?")
        trades = result.get("total_trades", "?")
        pf = g2.get("profit_factor", "?")
        sharpe = g2.get("sharpe", "?")
        gate = "PASS" if g2.get("gate2_pass") else "FAIL"
        
        print(
            f"  OK ({elapsed:.0f}s): ROI={roi}% Trades={trades} "
            f"PF={pf} Sharpe={sharpe} [{gate}]"
        )
        
        return result
        
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"  TIMEOUT ({elapsed:.0f}s)")
        return {
            "experiment": experiment,
            "seed": seed,
            "success": False,
            "error": f"timeout after {timeout}s",
        }


def run_batch(
    experiments: List[str],
    seeds: List[int],
    batch_name: str = "phase_c",
) -> List[Dict[str, Any]]:
    """バッチ実行"""
    total = len(experiments) * len(seeds)
    all_results: List[Dict[str, Any]] = []
    
    print(f"\n{'='*70}")
    print(f"Phase C Batch: {batch_name}")
    print(f"  {len(experiments)} experiments × {len(seeds)} seeds = {total} runs")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    for i, exp in enumerate(experiments, 1):
        for seed in seeds:
            print(f"\n--- [{i}/{len(experiments)}] {exp} seed={seed} ---")
            result = run_experiment_subprocess(exp, seed)
            all_results.append(result)
    
    total_time = time.time() - start_time
    
    # サマリ
    success = sum(1 for r in all_results if r.get("success"))
    print(f"\n{'='*70}")
    print(f"Batch Complete: {success}/{total} succeeded ({total_time:.0f}s total)")
    print(f"{'='*70}")
    
    # 結果テーブル
    print(f"\n{'Experiment':<35} {'γ':>5} {'Thr':>5} {'ROI%':>8} "
          f"{'GrossPnL':>10} {'Fees':>8} {'Trades':>7} "
          f"{'PF':>6} {'Sharpe':>7} {'MaxDD%':>7} {'WinR%':>6} {'G2':>4}")
    print("-" * 120)
    
    for r in all_results:
        if not r.get("success"):
            print(f"{r['experiment']:<35} FAILED: {r.get('error', '?')[:50]}")
            continue
        g2 = r.get("gate2", {})
        cfg = r.get("config", {})
        print(
            f"{r['experiment']:<35} "
            f"{cfg.get('gamma', '?'):>5} "
            f"{cfg.get('threshold', 0.33):>5.2f} "
            f"{r.get('net_roi', 0):>7.2f}% "
            f"{r.get('gross_pnl', 0):>+10.0f} "
            f"{r.get('total_fees', 0):>8.0f} "
            f"{r.get('total_trades', 0):>7} "
            f"{g2.get('profit_factor', 0):>6.3f} "
            f"{g2.get('sharpe', 0):>7.3f} "
            f"{g2.get('max_drawdown', 0):>6.2f}% "
            f"{g2.get('win_rate', 0)*100:>5.1f}% "
            f"{'OK' if g2.get('gate2_pass') else 'NG':>4}"
        )
    
    print(f"\nGate 2 基準 (0番§5.2): ROI>5% | PF>1.20 | Sharpe>1.0 | MaxDD<15% | WinRate>35%")
    
    # 結果保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = OUTPUT_DIR / f"{batch_name}_{timestamp}.json"
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({
            "batch": batch_name,
            "timestamp": timestamp,
            "total_time_seconds": round(total_time, 1),
            "results": all_results,
        }, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"Results saved: {filepath}")
    return all_results


# ============================================================================
# C0+C1 統合バッチ定義
# ============================================================================

# Phase C ステップ1: seed=42スクリーニング (14実験 ≈ 8-10時間)
C0_C1_SCREENING = [
    "c0_baseline_p1",           # ベースライン (P1-1再現)
    "c1_gamma_080",             # 91# H1 最優先
    "c1_gamma_090",             # γ感度
    "c1_gamma_095",             # γ感度
    "c1_threshold_50",          # H2 取引削減
    "c1_threshold_60",          # H2 取引削減
    "c1_threshold_70",          # H2 取引削減
    "c1_gamma080_threshold_50", # H1+H2 組合せ
    "c1_gamma080_threshold_60", # H1+H2 組合せ
    "c1_gamma080_threshold_70", # H1+H2 組合せ
    "c1_holding_5",             # 保持期間
    "c1_holding_10",            # 保持期間
    "c1_holding_15",            # 保持期間
    "c1_v451_golden",           # v451 Golden Era復元
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase C サブプロセスランナー")
    parser.add_argument("--batch", type=str, default="c0_c1",
                       choices=["c0_c1", "custom"],
                       help="バッチ名")
    parser.add_argument("--experiments", type=str, default=None,
                       help="カンマ区切り実験名 (--batch custom 時)")
    parser.add_argument("--seeds", type=str, default="42",
                       help="カンマ区切りseed (default: 42)")
    args = parser.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(",")]
    
    if args.batch == "c0_c1":
        experiments = C0_C1_SCREENING
    elif args.batch == "custom" and args.experiments:
        experiments = [e.strip() for e in args.experiments.split(",")]
    else:
        parser.print_help()
        return
    
    run_batch(experiments, seeds, batch_name=args.batch)


if __name__ == "__main__":
    main()
