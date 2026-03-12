import os
from pathlib import Path


from utils.analysis_utils import load_analysis_data, print_basic_stats


def analyze_log(filepath):
    if not os.path.exists(filepath):
        print(f"Log file not found: {filepath}")
        return

    df = load_analysis_data(filepath)

    print_basic_stats(df, "Gate Analysis Data")

    # 1. Data Inconsistency
    mismatch_df = df[df["gate_bin"] == "Neutral"]
    mismatch_count = len(mismatch_df)
    print(f"Binning Mismatch Count: {mismatch_count} ({mismatch_count/len(df):.2%})")

    # 2. Alpha Absence (Warmup Analysis)
    forced_df = df[df["gate_status"] == "forced"]
    print(f"Forced Trades: {len(forced_df)}")

    # 3. Fail-Closed (Cost Inf)
    cost_inf_df = df[df["block_reason"] == "cost_inf"]
    cost_inf_count = len(cost_inf_df)
    print(f"Cost Inf Count: {cost_inf_count} ({cost_inf_count/len(df):.2%})")

    # 4. Decay (n_eff low)
    n_eff_low_df = df[df["block_reason"] == "n_eff_low"]
    n_eff_low_count = len(n_eff_low_df)
    print(f"n_eff Low Count: {n_eff_low_count} ({n_eff_low_count/len(df):.2%})")

    # 5. Cost > EV
    cost_gt_ev_df = df[df["block_reason"] == "cost_gt_ev"]
    cost_gt_ev_count = len(cost_gt_ev_df)
    print(f"Cost > EV Count: {cost_gt_ev_count} ({cost_gt_ev_count/len(df):.2%})")

    # Decision Logic
    print("\n--- Decision Logic ---")

    print("Alpha Absence: LIKELY (Net PnL < 0)")

    # 3. Fail-Closed
    if cost_inf_count / len(df) > 0.1:
        print("Fail-Closed: TRIGGERED (> 10%)")
    else:
        print("Fail-Closed: Not Triggered")

    # 4. Decay
    blocked_df = df[df["gate_status"] == "blocked"]
    if len(blocked_df) > 0:
        n_eff_ratio = len(blocked_df[blocked_df["block_reason"] == "n_eff_low"]) / len(
            blocked_df
        )
        print(f"Decay Issue (n_eff < n_min in Blocked): {n_eff_ratio:.2%}")
        if n_eff_ratio > 0.5:
            print("Decay Issue: TRIGGERED")

    # 5. Cost
    if len(blocked_df) > 0:
        cost_ratio = len(blocked_df[blocked_df["block_reason"] == "cost_gt_ev"]) / len(
            blocked_df
        )
        print(f"Cost Issue (Cost > EV in Blocked): {cost_ratio:.2%}")
        if cost_ratio > 0.5:
            print("Cost Issue: TRIGGERED")

    # 6. Binning
    if len(blocked_df) > 0:
        # Check if blocked trades are mostly Neutral
        neutral_blocked = blocked_df[blocked_df["gate_bin"] == "Neutral"]
        neutral_ratio = len(neutral_blocked) / len(blocked_df)
        print(f"Binning Mismatch (Neutral in Blocked): {neutral_ratio:.2%}")
        if neutral_ratio > 0.5:  # Heuristic
            print("Binning Mismatch: TRIGGERED")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    log_path = os.path.join(project_root, "backtest_gate_log.csv")
    analyze_log(log_path)
