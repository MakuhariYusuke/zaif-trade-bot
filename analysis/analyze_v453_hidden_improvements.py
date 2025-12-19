import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if "portfolio_value" in df.columns:
        df["pnl_delta"] = df["portfolio_value"].diff().fillna(0.0)
    else:
        df["pnl_delta"] = 0.0

    if "hour" not in df.columns:
        df["hour"] = df["timestamp"].dt.hour

    if "atr" not in df.columns:
        df["atr"] = np.nan

    # Volatility bin: use ATR directly if present; fallback to rolling std of price returns
    vol_series = df["atr"]
    if vol_series.isna().all() and "price" in df.columns:
        returns = pd.Series(df["price"]).pct_change()
        vol_series = returns.rolling(window=60).std()

    df["vol_bin"] = pd.qcut(vol_series, q=10, labels=False, duplicates="drop")

    return df


def _regime_hour_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        df.groupby(["regime", "hour"])["pnl_delta"]
        .sum()
        .reset_index()
        .pivot(index="regime", columns="hour", values="pnl_delta")
        .fillna(0.0)
    )
    return pivot


def _worst_regime_hour(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    g = (
        df.groupby(["regime", "hour"])["pnl_delta"]
        .agg(["sum", "count", "mean"])
        .sort_values("sum")
        .head(top_n)
        .reset_index()
    )
    return g


def _regime_volbin(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["regime", "vol_bin"])["pnl_delta"]
        .agg(["sum", "count", "mean"])
        .sort_values("sum")
        .reset_index()
    )
    return g


def _entry_exit_analysis(df: pd.DataFrame) -> dict[str, pd.DataFrame] | None:
    if "position" not in df.columns:
        return None

    pos = df["position"].fillna(0).astype(int)
    prev_pos = pos.shift(1).fillna(0).astype(int)

    is_entry = (prev_pos == 0) & (pos != 0)
    is_exit = (prev_pos != 0) & (pos == 0)
    is_reversal = (prev_pos != 0) & (pos != 0) & (prev_pos != pos)

    work = df.copy()
    work["is_entry"] = is_entry
    work["is_exit"] = is_exit
    work["is_reversal"] = is_reversal

    # Forward expectancy (60 min)
    if "portfolio_value" in work.columns:
        work["fwd_pnl_60"] = work["portfolio_value"].shift(-60) - work["portfolio_value"]
    else:
        work["fwd_pnl_60"] = np.nan

    entries = work[work["is_entry"]].copy()
    exits = work[work["is_exit"]].copy()
    reversals = work[work["is_reversal"]].copy()

    entry_by_regime = entries.groupby("regime")["fwd_pnl_60"].agg(["mean", "count"]).sort_values("mean")
    entry_by_regime_hour = (
        entries.groupby(["regime", "hour"])["fwd_pnl_60"]
        .agg(["mean", "count"])
        .sort_values("mean")
        .head(20)
        .reset_index()
    )

    # Blocked entry diagnostics if present
    blocked = None
    if "blocked_entry" in work.columns:
        blocked = work[work["blocked_entry"]].groupby("regime").size().sort_values(ascending=False).to_frame("blocked_entries")

    return {
        "entries": entries,
        "exits": exits,
        "reversals": reversals,
        "entry_by_regime": entry_by_regime,
        "entry_by_regime_hour": entry_by_regime_hour,
        "blocked_by_regime": blocked,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, help="e.g. backtest_results/v453_hybrid_v3")
    parser.add_argument("--out", default=None, help="output markdown path (default: <results-dir>/hidden_improvements.md)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    csv_path = results_dir / "backtest_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = _load_csv(csv_path)

    total_pnl = float(df["pnl_delta"].sum())
    regime_pnl = df.groupby("regime")["pnl_delta"].sum().sort_values()

    worst_rh = _worst_regime_hour(df)
    regime_vol = _regime_volbin(df)

    event = _entry_exit_analysis(df)

    out_path = Path(args.out) if args.out else (results_dir / "hidden_improvements.md")

    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Hidden Improvements Report\n\n")
        f.write(f"Results: `{results_dir.as_posix()}`\n\n")
        f.write(f"## Total PnL (portfolio_value diff)\n{total_pnl:.2f}\n\n")

        f.write("## Regime PnL (worst first)\n")
        f.write(regime_pnl.to_string())
        f.write("\n\n")

        f.write("## Worst Regime×Hour (Top 15)\n")
        f.write(worst_rh.to_string(index=False))
        f.write("\n\n")

        f.write("## Regime×VolatilityBin (sorted by total PnL)\n")
        f.write(regime_vol.head(30).to_string(index=False))
        f.write("\n\n")

        if event is None:
            f.write("## Entry/Exit Analysis\n")
            f.write("(skipped: `position` column not available in CSV)\n")
        else:
            f.write("## Entry Expectancy (fwd 60m) by Regime (worst first)\n")
            f.write(event["entry_by_regime"].to_string())
            f.write("\n\n")

            f.write("## Worst Entry Regime×Hour by fwd 60m (Top 20)\n")
            f.write(event["entry_by_regime_hour"].to_string(index=False))
            f.write("\n\n")

            if event.get("blocked_by_regime") is not None:
                f.write("## Blocked Entries by Regime\n")
                f.write(event["blocked_by_regime"].to_string())
                f.write("\n\n")

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
