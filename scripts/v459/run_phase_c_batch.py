#!/usr/bin/env python3
"""
Phase C バッチ実行: サブプロセス分離版（メモリリーク完全防止）

各実験を独立プロセスで実行し、プロセス終了時にメモリを完全解放。
中間結果を毎回保存 → クラッシュ耐性あり。
既完了実験は自動スキップ → 再開可能。

Usage:
  python scripts/v459/run_phase_c_batch.py
  python scripts/v459/run_phase_c_batch.py --resume          # クラッシュ後の再開
  python scripts/v459/run_phase_c_batch.py --only c1_gamma_080,c1_gamma_090
"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

project_root = Path(__file__).parent.parent.parent
RESULTS_DIR = project_root / "results" / "phase_c"
SCRIPT_PATH = project_root / "scripts" / "v459" / "run_phase_c.py"
PYTHON = sys.executable

# 14 experiments — same order as run_phase_c.py BATCHES["c0_c1"]
C0_C1_EXPERIMENTS = [
    "c0_baseline_p1",
    "c1_gamma_080", "c1_gamma_090", "c1_gamma_095",
    "c1_threshold_50", "c1_threshold_60", "c1_threshold_70",
    "c1_gamma080_threshold_50", "c1_gamma080_threshold_60", "c1_gamma080_threshold_70",
    "c1_holding_5", "c1_holding_10", "c1_holding_15",
    "c1_v451_golden",
]


def load_completed(results_dir: Path) -> Dict[str, dict]:
    """完了済み実験を中間JSONから復元"""
    completed: Dict[str, dict] = {}
    partial_files = sorted(results_dir.glob("c0_c1_*_partial.json"))
    for f in partial_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            for r in data.get("results", []):
                if r.get("success"):
                    key = f"{r['experiment']}_seed{r['seed']}"
                    completed[key] = r
        except Exception:
            continue
    return completed


def run_one_experiment(experiment: str, seed: int, idx: int, total: int) -> dict:
    """1実験を独立subprocessで実行 → JSON結果を返す"""
    print(f"\n{'='*70}")
    print(f"  [{idx}/{total}] {experiment} seed={seed}")
    print(f"{'='*70}")

    cmd = [
        PYTHON, str(SCRIPT_PATH),
        "--single-run",
        "--experiment", experiment,
        "--seed", str(seed),
    ]

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30分タイムアウト（50K steps ≈ 7-8分）
        )
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"  ❌ タイムアウト (30分超過)")
        return {
            "experiment": experiment, "seed": seed,
            "success": False, "error": "timeout_30min",
            "elapsed_seconds": round(elapsed, 1),
        }

    elapsed = time.time() - start

    # ログ保存
    log_dir = RESULTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_base = f"{experiment}_seed{seed}"
    if proc.stdout:
        (log_dir / f"{log_base}.stdout.log").write_text(
            proc.stdout[-50000:], encoding="utf-8"  # 最後50KB
        )
    if proc.stderr:
        (log_dir / f"{log_base}.stderr.log").write_text(
            proc.stderr[-50000:], encoding="utf-8"
        )

    # JSON結果をstdout末尾から取得
    result = None
    if proc.returncode == 0 and proc.stdout:
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    result = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue

    if result is None:
        print(f"  ❌ 失敗 (exit={proc.returncode}, {elapsed:.0f}秒)")
        if proc.stderr:
            for line in proc.stderr.strip().splitlines()[-10:]:
                print(f"    STDERR: {line}")
        return {
            "experiment": experiment, "seed": seed,
            "success": False, "error": f"exit_code={proc.returncode}",
            "elapsed_seconds": round(elapsed, 1),
        }

    # 成功時のサマリ出力
    roi = result.get("net_roi", "N/A")
    trades = result.get("total_trades", "N/A")
    gross = result.get("gross_pnl", "N/A")
    fees = result.get("total_fees", "N/A")
    g2 = result.get("gate2", {})
    pf = g2.get("profit_factor", "N/A")
    sharpe = g2.get("sharpe", "N/A")
    g2_pass = "PASS" if g2.get("gate2_pass") else "FAIL"

    print(f"  ✅ 完了 ({elapsed:.0f}秒)")
    print(f"     ROI={roi}% | Trades={trades} | Gross={gross} | Fees={fees}")
    print(f"     Gate2: PF={pf} Sharpe={sharpe} → {g2_pass}")

    return result


def print_summary(results: List[dict]) -> None:
    """最終サマリテーブル"""
    print(f"\n{'='*130}")
    print("Phase C0+C1 RESULTS SUMMARY")
    print(f"{'='*130}")

    header = (
        f"{'Experiment':<35} {'γ':>5} {'Thr':>5} {'Net ROI%':>9} "
        f"{'GrossPnL':>10} {'Fees':>8} {'Trades':>7} "
        f"{'PF':>6} {'Sharpe':>7} {'MaxDD%':>7} {'WinR%':>6} {'G2':>4}"
    )
    print(header)
    print("-" * 130)

    for r in results:
        if not r.get("success"):
            print(f"{r['experiment']:<35} FAILED: {r.get('error', '?')[:50]}")
            continue

        g2 = r.get("gate2", {})
        cfg = r.get("config", {})

        print(
            f"{r['experiment']:<35} "
            f"{cfg.get('gamma', '?'):>5} "
            f"{cfg.get('threshold', 0.33):>5.2f} "
            f"{r.get('net_roi', 0):>8.2f}% "
            f"{r.get('gross_pnl', 0):>+10.0f} "
            f"{r.get('total_fees', 0):>8.0f} "
            f"{r.get('total_trades', 0):>7} "
            f"{g2.get('profit_factor', 0):>6.3f} "
            f"{g2.get('sharpe', 0):>7.3f} "
            f"{g2.get('max_drawdown', 0):>6.2f}% "
            f"{g2.get('win_rate', 0)*100:>5.1f}% "
            f"{'OK' if g2.get('gate2_pass') else 'NG':>4}"
        )

    print(f"\n{'─'*70}")
    print("Gate 2 基準 (0番§5.2): ROI>5% | PF>1.20 | Sharpe>1.0 | MaxDD<15% | WinRate>35%")
    print(f"{'─'*70}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="完了済みをスキップして再開")
    parser.add_argument("--only", type=str, default=None,
                        help="実行する実験名のカンマ区切り")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed = args.seed

    # 実行対象の決定
    if args.only:
        experiments = [e.strip() for e in args.only.split(",")]
    else:
        experiments = C0_C1_EXPERIMENTS

    # 完了済みのスキップ（--resume）
    completed = {}
    if args.resume:
        completed = load_completed(RESULTS_DIR)
        skip_count = sum(1 for e in experiments if f"{e}_seed{seed}" in completed)
        print(f"再開モード: {skip_count}/{len(experiments)} 完了済み → スキップ")

    print("=" * 70)
    print(f"Phase C0+C1 バッチ (subprocess分離, メモリリーク防止)")
    print(f"  実験数: {len(experiments)} | seed={seed}")
    print(f"  推定時間: ~{len(experiments) * 8}分 ({len(experiments) * 8 / 60:.1f}時間)")
    print(f"  結果: {RESULTS_DIR}")
    print("=" * 70)

    all_results = list(completed.values())
    total_start = time.time()

    for i, exp_name in enumerate(experiments, 1):
        key = f"{exp_name}_seed{seed}"

        # スキップチェック
        if key in completed:
            print(f"\n  [{i}/{len(experiments)}] {exp_name} → スキップ (完了済み)")
            continue

        # 実行
        result = run_one_experiment(exp_name, seed, i, len(experiments))
        all_results.append(result)

        # 中間保存（クラッシュ耐性）
        partial_file = RESULTS_DIR / f"c0_c1_{timestamp}_partial.json"
        with open(partial_file, "w", encoding="utf-8") as f:
            json.dump({
                "batch": "c0_c1",
                "timestamp": timestamp,
                "seed": seed,
                "results": all_results,
            }, f, indent=2, ensure_ascii=False, default=str)

    total_elapsed = time.time() - total_start

    # 最終保存
    final_file = RESULTS_DIR / f"c0_c1_{timestamp}_final.json"
    with open(final_file, "w", encoding="utf-8") as f:
        json.dump({
            "batch": "c0_c1",
            "timestamp": timestamp,
            "seed": seed,
            "total_elapsed_seconds": round(total_elapsed, 1),
            "results": all_results,
            "gate2_criteria": {
                "roi": "> 5%", "profit_factor": "> 1.20",
                "sharpe": "> 1.0", "max_drawdown": "< 15%",
                "win_rate": "> 35%",
            },
        }, f, indent=2, ensure_ascii=False, default=str)

    # サマリ
    print_summary(all_results)
    print(f"\n合計時間: {total_elapsed:.0f}秒 ({total_elapsed/60:.1f}分)")
    print(f"結果: {final_file}")


if __name__ == "__main__":
    main()
