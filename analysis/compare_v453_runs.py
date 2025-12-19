import argparse
from pathlib import Path

import pandas as pd


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize booleans that might be stored as strings
    for col in ("filter_active", "blocked_entry"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(("true", "1", "yes"))
    if "effective_action" in df.columns:
        df["effective_action"] = pd.to_numeric(df["effective_action"], errors="coerce").fillna(0).astype(int)
    if "attempted_discrete_action" in df.columns:
        df["attempted_discrete_action"] = pd.to_numeric(df["attempted_discrete_action"], errors="coerce").fillna(0).astype(int)
    return df


def _summarize(df: pd.DataFrame) -> dict:
    out: dict = {}
    out["rows"] = len(df)

    if "regime" in df.columns:
        out["regime_counts_top"] = df["regime"].value_counts().head(12)

    if "filter_active" in df.columns:
        out["filter_active_true"] = int(df["filter_active"].sum())
        if "filter_reasons" in df.columns:
            active = df[df["filter_active"]].copy()
            # split reasons (comma-separated) and explode
            active["_reason"] = active["filter_reasons"].fillna("")
            reasons = (
                active.assign(_reason=active["_reason"].str.split(","))
                .explode("_reason")
                .query("_reason != ''")
            )
            out["filter_reason_counts"] = reasons["_reason"].value_counts()

    if "blocked_entry" in df.columns:
        out["blocked_entry_true"] = int(df["blocked_entry"].sum())
        if "regime" in df.columns:
            out["blocked_by_regime_top"] = df[df["blocked_entry"]]["regime"].value_counts().head(12)

    if "effective_action" in df.columns:
        out["effective_nonhold"] = int((df["effective_action"] != 0).sum())
        if "regime" in df.columns:
            out["effective_by_regime_top"] = df[df["effective_action"] != 0]["regime"].value_counts().head(12)

    return out


def _print_report(name: str, rep: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"rows: {rep.get('rows')}")

    if "regime_counts_top" in rep:
        print("\nregime_counts_top:")
        print(rep["regime_counts_top"].to_string())

    if "filter_active_true" in rep:
        print(f"\nfilter_active_true: {rep['filter_active_true']}")
    if "filter_reason_counts" in rep:
        print("filter_reason_counts:")
        print(rep["filter_reason_counts"].to_string())

    if "blocked_entry_true" in rep:
        print(f"\nblocked_entry_true: {rep['blocked_entry_true']}")
    if "blocked_by_regime_top" in rep:
        print("blocked_by_regime_top:")
        print(rep["blocked_by_regime_top"].to_string())

    if "effective_nonhold" in rep:
        print(f"\neffective_nonhold: {rep['effective_nonhold']}")
    if "effective_by_regime_top" in rep:
        print("effective_by_regime_top:")
        print(rep["effective_by_regime_top"].to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="results dir A")
    ap.add_argument("--b", required=True, help="results dir B")
    args = ap.parse_args()

    a_dir = Path(args.a)
    b_dir = Path(args.b)

    a_csv = a_dir / "backtest_results.csv"
    b_csv = b_dir / "backtest_results.csv"

    df_a = _load(a_csv)
    df_b = _load(b_csv)

    rep_a = _summarize(df_a)
    rep_b = _summarize(df_b)

    _print_report(a_dir.name, rep_a)
    _print_report(b_dir.name, rep_b)


if __name__ == "__main__":
    main()
