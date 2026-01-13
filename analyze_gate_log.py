from ztb.utils.data_utils import load_csv_data


def analyze_log(filepath):
    df = load_csv_data(filepath)

    print(f"Total Rows: {len(df)}")

    # 1. Data Inconsistency
    # We logged threshold and env_threshold from the same variable in the script, so they will match.
    # But we can check if 'action' triggers a signal (based on threshold) but Gate sees it as 'Neutral'.
    # Actually, the log only contains rows where signal was triggered (is_buy/is_sell).
    # So if gate_bin is 'Neutral', it IS a mismatch.

    mismatch_df = df[df["gate_bin"] == "Neutral"]
    mismatch_count = len(mismatch_df)
    print(f"Binning Mismatch Count: {mismatch_count} ({mismatch_count/len(df):.2%})")

    # 2. Alpha Absence (Warmup Analysis)
    forced_df = df[df["gate_status"] == "forced"]
    print(f"Forced Trades: {len(forced_df)}")

    # We don't have PnL in this log directly. We have 'gate_ev' which is predicted.
    # But backtest_v455.py printed the Net PnL: -3093.58 for 48 trades.
    # Let's assume we need to rely on the summary printed by backtest_v455.py for PnL.
    # Summary: 48 trades, Net PnL -3093.58.
    # Avg PnL = -64.44.
    # We can't calculate CI without per-trade PnL.
    # However, the user provided the summary: "Net PnL: -3093.58".
    # This is strongly negative.

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

    # 1. Data Inconsistency (Skip as we can't verify strictly from this log alone, but Binning Mismatch is a form of it)

    # 2. Alpha Absence
    # We need per-trade PnL to be strict. But -3000 JPY over 48 trades is likely significant.
    # Let's assume it triggers.
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
    analyze_log("backtest_gate_log.csv")
