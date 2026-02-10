#!/usr/bin/env python3
"""Gate2 deterministic eval の段階診断スクリプト"""
import sys
import json
import time
import logging
import os
import warnings
import traceback

sys.path.insert(0, ".")
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

# ログ抑制（但しERROR以上は表示）
logging.basicConfig(level=logging.ERROR, format="%(levelname)s:%(message)s")

QUICK_STEPS = 10000  # 統合テスト（約5分）

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

try:
    log("=== run_single_experiment統合テスト (10K steps) ===")
    import scripts.v459.run_phase_c as rpc

    # TOTAL_TIMESTEPSを一時的にオーバーライド
    original_ts = rpc.TOTAL_TIMESTEPS
    rpc.TOTAL_TIMESTEPS = QUICK_STEPS

    exp_def = rpc.get_experiment_configs()["c0_baseline_p1"]
    t0 = time.time()
    result = rpc.run_single_experiment("integration_test", 42, exp_def)
    elapsed = time.time() - t0

    # 復元
    rpc.TOTAL_TIMESTEPS = original_ts

    # 結果出力
    out_path = "temp_scripts/gate2_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    log(f"完了: {elapsed:.0f}秒")
    log(f"success={result.get('success')}")
    
    g2 = result.get("gate2", {})
    log(f"gate2_available={g2.get('gate2_available')}")
    if g2.get("gate2_available"):
        log(f"  eval_trades={g2.get('eval_trades')}")
        log(f"  eval_net_roi={g2.get('eval_net_roi', 'N/A')}")
        log(f"  sharpe={g2.get('sharpe')}")
        log(f"  max_drawdown={g2.get('max_drawdown')}")
        log(f"  profit_factor={g2.get('profit_factor')}")
        log(f"  win_rate={g2.get('win_rate')}")
        log(f"  gate2_pass={g2.get('gate2_pass')}")
    else:
        log(f"  gate2_error={g2.get('gate2_error')}")

    log(f"結果: {out_path}")

except Exception as e:
    log(f"EXCEPTION: {e}")
    traceback.print_exc()
    sys.exit(1)
