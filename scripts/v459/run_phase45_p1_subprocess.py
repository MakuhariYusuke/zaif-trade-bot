#!/usr/bin/env python3
"""
Phase 4.5 P1: サブプロセス実行ランナー（メモリリーク防止版）

各実験を別プロセスで実行し、プロセス終了時にメモリを完全解放。
16GB RAMでも安全に4seeds × 2条件 = 8実験を実行可能。
"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
RESULTS_DIR = project_root / "results" / "phase45_p1_baseline"
SCRIPT_PATH = project_root / "scripts" / "v459" / "run_phase45_p1.py"
PYTHON = sys.executable


def run_single_seed(category: str, seed: int) -> dict:
    """1 seed を別プロセスで実行"""
    print(f"\n{'='*60}")
    print(f"  {category} seed={seed} を別プロセスで実行中...")
    print(f"{'='*60}")

    # run_phase45_p1.py を --single-run モードで呼び出し
    cmd = [
        PYTHON, str(SCRIPT_PATH),
        "--single-run",
        "--category", category,
        "--seed", str(seed),
    ]

    start = time.time()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3600,  # 1時間タイムアウト
    )
    elapsed = time.time() - start

    # 結果JSONを stdout 末尾から取得
    result = None
    if proc.returncode == 0:
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    result = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue

    # seed別ログを保存（Gate C0: 観測性確保）
    log_dir = RESULTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_base = f"{category}_seed{seed}"
    if proc.stdout:
        (log_dir / f"{log_base}.stdout.log").write_text(proc.stdout, encoding="utf-8")
    if proc.stderr:
        (log_dir / f"{log_base}.stderr.log").write_text(proc.stderr, encoding="utf-8")

    if result is None:
        print(f"  ❌ 結果取得失敗 (exit={proc.returncode})")
        if proc.stderr:
            # 最後の20行だけ表示
            err_lines = proc.stderr.strip().splitlines()[-20:]
            for line in err_lines:
                print(f"  STDERR: {line}")
        result = {
            "experiment_name": f"{category}_seed{seed}",
            "seed": seed,
            "success": False,
            "error": f"exit_code={proc.returncode}",
            "total_time_seconds": elapsed,
        }
    else:
        roi = result.get("balance_roi", result.get("final_roi", "N/A"))
        gross = result.get("gross_pnl", "N/A")
        trades = result.get("total_trades", "N/A")
        print(f"  ✅ 完了 ({elapsed:.0f}秒)")
        print(f"     ROI: {roi}, Gross PnL: {gross}, Trades: {trades}")

    result["experiment_category"] = category
    return result


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("Phase 4.5 P1: サブプロセス実行（メモリリーク防止版）")
    print(f"各実験は独立プロセスで実行 → 終了時にメモリ完全解放")
    print("=" * 70)

    seeds = [42, 123, 456, 789]
    experiments = [
        ("P1-1", seeds),  # PnLのみ
        ("P1-3", seeds),  # デフォルト
    ]

    all_results = []
    total_start = time.time()

    for category, seed_list in experiments:
        print(f"\n{'='*60}")
        print(f"カテゴリ: {category}")
        print(f"{'='*60}")
        for seed in seed_list:
            result = run_single_seed(category, seed)
            all_results.append(result)

            # 中間保存
            intermediate_file = RESULTS_DIR / f"p1_results_{timestamp}_partial.json"
            with open(intermediate_file, "w", encoding="utf-8") as f:
                json.dump({"partial_results": all_results}, f, indent=2, ensure_ascii=False)

    total_elapsed = time.time() - total_start

    # 最終集計
    import numpy as np

    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)

    for cat in ["P1-1", "P1-3"]:
        cat_results = [r for r in all_results if r.get("experiment_category") == cat and r.get("success")]
        rois = [r.get("balance_roi", r.get("final_roi")) for r in cat_results if r.get("balance_roi") is not None or r.get("final_roi") is not None]
        gross_pnls = [r.get("gross_pnl") for r in cat_results if r.get("gross_pnl") is not None]
        fees = [r.get("total_fees") for r in cat_results if r.get("total_fees") is not None]
        trades = [r.get("total_trades") for r in cat_results if r.get("total_trades") is not None]

        print(f"\n{cat}:")
        print(f"  成功: {len(cat_results)}/{len([r for r in all_results if r.get('experiment_category') == cat])}")
        if rois:
            print(f"  ROI: {np.mean(rois):.2f}% ± {np.std(rois):.2f}%")
        if gross_pnls:
            print(f"  Gross PnL: {np.mean(gross_pnls):+,.0f} ± {np.std(gross_pnls):,.0f}")
        if fees:
            print(f"  Total Fees: {np.mean(fees):,.0f}")
        if trades:
            print(f"  Trades: {np.mean(trades):.0f}")

    # 結果保存
    results_file = RESULTS_DIR / f"p1_results_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "total_elapsed_seconds": total_elapsed,
            "all_results": all_results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved: {results_file}")
    print(f"Total time: {total_elapsed/60:.1f} min")

    # 中間ファイル削除
    partial = RESULTS_DIR / f"p1_results_{timestamp}_partial.json"
    if partial.exists():
        partial.unlink()


if __name__ == "__main__":
    main()
