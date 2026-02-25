"""Fill Test ログ統合分析スクリプト (162# P0 再現可能分析).

Usage:
    python tools/analysis/analyze_fill_logs.py
    python tools/analysis/analyze_fill_logs.py --date-from 2026-02-20 --date-to 2026-02-24
    python tools/analysis/analyze_fill_logs.py --git-sha d9874bbee12a
    python tools/analysis/analyze_fill_logs.py --run-id 1771932882_97af3a30
    python tools/analysis/analyze_fill_logs.py --date-from 2026-02-22 --git-sha 5c65ef925 --output report.txt

Replaces: temp/analyze_logs.py + temp/analyze_logs2.py
Purpose: 因果混在排除のためフィルタ条件を明示して再現可能な分析を行う
Reference: 162# §7.3 P0 / 000# 運用方針追補 §3
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fill Test ログ統合分析 (再現可能フィルタ付き)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data-dir",
        default="results/v460/fill_test",
        help="fill_records_*.jsonl の格納ディレクトリ (default: results/v460/fill_test)",
    )
    p.add_argument("--run-id", help="run_id 完全一致フィルタ")
    p.add_argument(
        "--git-sha", help="git_sha 前方一致フィルタ (短縮 SHA 可)"
    )
    p.add_argument(
        "--date-from",
        help="開始日 inclusive (YYYY-MM-DD UTC, ファイル名 + timestamp 両方でフィルタ)",
    )
    p.add_argument(
        "--date-to",
        help="終了日 inclusive (YYYY-MM-DD UTC)",
    )
    p.add_argument(
        "--side",
        choices=["buy", "sell"],
        help="side フィルタ (省略時: 全 side)",
    )
    p.add_argument(
        "--regime",
        help="regime 完全一致フィルタ (trending/ranging/volatile)",
    )
    p.add_argument(
        "--output", "-o", help="結果をファイルに書き出す (省略時: stdout)",
    )
    p.add_argument(
        "--json", action="store_true", help="JSON 形式で出力",
    )
    return p


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_records(
    data_dir: str,
    date_from: str | None,
    date_to: str | None,
) -> list[dict[str, Any]]:
    """JSONL ファイルを読み込み、日付範囲でファイルをプリフィルタ."""
    base = pathlib.Path(data_dir)
    if not base.exists():
        print(f"ERROR: data directory not found: {base}", file=sys.stderr)
        sys.exit(1)

    files = sorted(base.glob("fill_records_*.jsonl"))
    if not files:
        print(f"ERROR: no fill_records_*.jsonl found in {base}", file=sys.stderr)
        sys.exit(1)

    # ファイル名日付でプリフィルタ (高速化)
    if date_from or date_to:
        df = date_from.replace("-", "") if date_from else "00000000"
        dt = date_to.replace("-", "") if date_to else "99999999"
        filtered: list[pathlib.Path] = []
        for f in files:
            # fill_records_20260225.jsonl -> 20260225
            stem = f.stem  # fill_records_20260225
            file_date = stem.split("_")[-1]  # 20260225
            if len(file_date) == 8 and file_date.isdigit():
                if df <= file_date <= dt:
                    filtered.append(f)
            else:
                filtered.append(f)  # 日付形式でなければ含める
        files = filtered

    records: list[dict[str, Any]] = []
    for f in files:
        for line in f.read_text("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def apply_filters(
    records: list[dict[str, Any]],
    *,
    run_id: str | None = None,
    git_sha: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    side: str | None = None,
    regime: str | None = None,
) -> list[dict[str, Any]]:
    """レコードレベルのフィルタリングを適用."""
    out = records

    if run_id:
        out = [r for r in out if r.get("run_id") == run_id]

    if git_sha:
        out = [r for r in out if str(r.get("git_sha", "")).startswith(git_sha)]

    if date_from:
        ts_from = datetime.strptime(date_from, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        ).timestamp()
        out = [r for r in out if (r.get("timestamp") or 0) >= ts_from]

    if date_to:
        ts_to = (
            datetime.strptime(date_to, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
            + 86400  # 終了日 inclusive (翌日0:00まで)
        )
        out = [r for r in out if (r.get("timestamp") or 0) < ts_to]

    if side:
        out = [r for r in out if r.get("side") == side]

    if regime:
        out = [r for r in out if r.get("regime") == regime]

    return out


# ---------------------------------------------------------------------------
# Analysis Sections
# ---------------------------------------------------------------------------

def _np(values: list[float]) -> np.ndarray:
    return np.array(values, dtype=np.float64) if values else np.array([], dtype=np.float64)


def _pnls(records: list[dict[str, Any]], key: str = "post_fill_30s_pnl") -> np.ndarray:
    return _np([float(r[key]) for r in records if r.get(key) is not None])


def section_header(records: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    """再現性ヘッダー."""
    lines = [
        "=" * 70,
        "Fill Test ログ分析レポート",
        "=" * 70,
        "",
        "## フィルタ条件 (再現用)",
        f"  data_dir   : {args.data_dir}",
        f"  run_id     : {args.run_id or '(all)'}",
        f"  git_sha    : {args.git_sha or '(all)'}",
        f"  date_from  : {args.date_from or '(all)'}",
        f"  date_to    : {args.date_to or '(all)'}",
        f"  side       : {args.side or '(all)'}",
        f"  regime     : {args.regime or '(all)'}",
        f"  generated  : {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S UTC}",
        "",
    ]
    timestamps = [r["timestamp"] for r in records if r.get("timestamp")]
    if timestamps:
        t_min = datetime.fromtimestamp(min(timestamps), tz=timezone.utc)
        t_max = datetime.fromtimestamp(max(timestamps), tz=timezone.utc)
        lines.append(f"  data_range : {t_min:%Y-%m-%d %H:%M} ~ {t_max:%Y-%m-%d %H:%M} UTC")

    # git_sha / run_id 分布 (上位3)
    sha_dist = collections.Counter(r.get("git_sha", "?") for r in records)
    run_dist = collections.Counter(r.get("run_id", "?") for r in records)
    lines.append(f"  git_sha_unique : {len(sha_dist)} ({', '.join(f'{s}:{c}' for s, c in sha_dist.most_common(3))})")
    lines.append(f"  run_id_unique  : {len(run_dist)} ({', '.join(f'{s[:16]}:{c}' for s, c in run_dist.most_common(3))})")
    lines.append("")
    return lines


def section_basic(records: list[dict[str, Any]]) -> list[str]:
    """基本統計."""
    n = len(records)
    filled = [r for r in records if r.get("filled")]
    nf = len(filled)
    lines = [
        "## 基本統計",
        f"  Total: {n}, Filled: {nf}, Fill rate: {nf/n*100:.1f}%" if n else "  (no records)",
        "",
    ]
    return lines


def section_side(records: list[dict[str, Any]]) -> list[str]:
    """Side 別."""
    lines = ["## Side 別"]
    n = len(records)
    for side in ["buy", "sell"]:
        s_all = [r for r in records if r.get("side") == side]
        s_filled = [r for r in s_all if r.get("filled")]
        pnl_arr = _pnls(s_filled)
        fill_rate = len(s_filled) / len(s_all) * 100 if s_all else 0
        avg_pnl = float(np.mean(pnl_arr)) if len(pnl_arr) else 0
        p10 = float(np.percentile(pnl_arr, 10)) if len(pnl_arr) else 0
        p05 = float(np.percentile(pnl_arr, 5)) if len(pnl_arr) else 0
        profitable = float(np.sum(pnl_arr > 0) / len(pnl_arr) * 100) if len(pnl_arr) else 0
        as_cnt = sum(1 for r in s_filled if r.get("adverse_selected"))
        as_rate = as_cnt / len(s_filled) * 100 if s_filled else 0
        lines.append(
            f"  {side}: {len(s_all)} total, {len(s_filled)} filled ({fill_rate:.1f}%), "
            f"avg_pnl30={avg_pnl:.2f}bps, p10={p10:.2f}, p05={p05:.2f}, "
            f"profitable={profitable:.1f}%, AS率={as_rate:.1f}%"
        )
    lines.append("")
    return lines


def section_regime(records: list[dict[str, Any]]) -> list[str]:
    """Regime 別."""
    lines = ["## Regime 別"]
    regime_counter = collections.Counter(r.get("regime", "null") for r in records)
    for regime, cnt in regime_counter.most_common():
        rf = [r for r in records if r.get("regime") == regime and r.get("filled")]
        pnl_arr = _pnls(rf)
        avg_pnl = float(np.mean(pnl_arr)) if len(pnl_arr) else float("nan")
        fill_r = len(rf) / cnt * 100 if cnt else 0
        lines.append(f"  {regime}: {cnt} total, {len(rf)} filled ({fill_r:.1f}%), avg_pnl30={avg_pnl:.2f}bps")
    lines.append("")
    return lines


def section_cancel(records: list[dict[str, Any]]) -> list[str]:
    """Cancel Reason."""
    cancels = [r for r in records if not r.get("filled")]
    lines = ["## Cancel Reason (top 15)"]
    if not cancels:
        lines.append("  (no cancels)")
    else:
        reasons = collections.Counter(r.get("cancel_reason", "unknown") for r in cancels)
        for reason, cnt in reasons.most_common(15):
            lines.append(f"  {reason}: {cnt} ({cnt/len(cancels)*100:.1f}%)")
    lines.append("")
    return lines


def section_skip_gate(records: list[dict[str, Any]]) -> list[str]:
    """SkipGate 統計."""
    n = len(records)
    skipped = [r for r in records if r.get("skip_gate_skipped")]
    lines = [
        "## SkipGate",
        f"  Skip total: {len(skipped)}/{n} ({len(skipped)/n*100:.1f}%)" if n else "  (no data)",
    ]
    skip_reasons = collections.Counter(r.get("skip_gate_reason", "unknown") for r in skipped)
    for reason, cnt in skip_reasons.most_common(10):
        lines.append(f"  {reason}: {cnt}")

    # Model usage
    sg_models = collections.Counter(str(r.get("skip_gate_model_used", "?")) for r in skipped)
    if sg_models:
        lines.append("  --- Model Usage ---")
        for m, cnt in sg_models.most_common(5):
            lines.append(f"    {m}: {cnt}")
    lines.append("")
    return lines


def section_adverse_selection(records: list[dict[str, Any]]) -> list[str]:
    """Adverse Selection 詳細."""
    filled = [r for r in records if r.get("filled")]
    nf = len(filled)
    as_records = [r for r in filled if r.get("adverse_selected")]
    lines = [
        "## Adverse Selection",
        f"  AS count: {len(as_records)}/{nf} ({len(as_records)/nf*100:.1f}%)" if nf else "  (no fills)",
    ]
    if as_records:
        as_pnls = _pnls(as_records)
        non_as = [r for r in filled if not r.get("adverse_selected")]
        non_as_pnls = _pnls(non_as)
        if len(as_pnls):
            lines.append(f"  AS avg_pnl30: {np.mean(as_pnls):.2f}bps (n={len(as_pnls)})")
        if len(non_as_pnls):
            lines.append(f"  Non-AS avg_pnl30: {np.mean(non_as_pnls):.2f}bps (n={len(non_as_pnls)})")
    lines.append("")
    return lines


def section_hourly(records: list[dict[str, Any]]) -> list[str]:
    """時間帯別 (UTC)."""
    filled = [r for r in records if r.get("filled")]
    hour_pnls: dict[int, list[float]] = collections.defaultdict(list)
    for r in filled:
        ts = r.get("timestamp")
        pnl = r.get("post_fill_30s_pnl")
        if ts and pnl is not None:
            h = datetime.fromtimestamp(ts, tz=timezone.utc).hour
            hour_pnls[h].append(float(pnl))

    lines = ["## 時間帯別 (UTC)"]
    for h in sorted(hour_pnls.keys()):
        arr = _np(hour_pnls[h])
        n_h = len(arr)
        avg_h = float(np.mean(arr))
        p_rate = float(np.sum(arr > 0) / n_h * 100) if n_h else 0
        lines.append(f"  {h:02d}h: n={n_h:3d}, avg_pnl30={avg_h:+.2f}bps, profitable={p_rate:.0f}%")
    lines.append("")
    return lines


def section_daily(records: list[dict[str, Any]]) -> list[str]:
    """日別サマリ."""
    day_records: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in records:
        ts = r.get("timestamp")
        if ts:
            d = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            day_records[d].append(r)

    lines = ["## 日別サマリ"]
    for d in sorted(day_records.keys()):
        day_r = day_records[d]
        day_f = [r for r in day_r if r.get("filled")]
        pnl_arr = _pnls(day_f)
        avg_pnl = float(np.mean(pnl_arr)) if len(pnl_arr) else float("nan")
        total_pnl = float(np.sum(pnl_arr)) if len(pnl_arr) else 0
        fill_rate = len(day_f) / len(day_r) * 100 if day_r else 0
        lines.append(
            f"  {d}: {len(day_r)} orders, {len(day_f)} filled ({fill_rate:.0f}%), "
            f"avg_pnl30={avg_pnl:+.2f}bps, sum_pnl30={total_pnl:+.1f}bps"
        )
    lines.append("")
    return lines


def section_git_sha(records: list[dict[str, Any]]) -> list[str]:
    """git_sha 別 (因果混在の可視化)."""
    lines = ["## git_sha 別"]
    sha_counter = collections.Counter(r.get("git_sha", "?") for r in records)
    for sha, cnt in sha_counter.most_common(10):
        sha_filled = sum(1 for r in records if r.get("git_sha") == sha and r.get("filled"))
        pnls = _pnls([r for r in records if r.get("git_sha") == sha and r.get("filled")])
        avg_p = float(np.mean(pnls)) if len(pnls) else float("nan")
        lines.append(f"  {sha}: {cnt} orders, {sha_filled} filled, avg_pnl30={avg_p:.2f}bps")
    lines.append("")
    return lines


def section_reprice(records: list[dict[str, Any]]) -> list[str]:
    """Reprice 統計."""
    filled = [r for r in records if r.get("filled")]
    nf = len(filled)
    repriced = [r for r in filled if (r.get("reprice_count") or 0) > 0]
    lines = [
        "## Reprice",
        f"  Repriced: {len(repriced)}/{nf} ({len(repriced)/nf*100:.1f}%)" if nf else "  (no fills)",
    ]
    if repriced:
        drift_vals = [float(r["reprice_drift_bps"]) for r in repriced if r.get("reprice_drift_bps") is not None]
        reprice_counts = [r.get("reprice_count", 0) for r in repriced]
        rpnls = _pnls(repriced)
        lines.append(f"  Avg reprice_count: {np.mean(reprice_counts):.1f}")
        if drift_vals:
            lines.append(f"  Avg drift_bps: {np.mean(drift_vals):.2f}")
        if len(rpnls):
            lines.append(f"  Repriced avg_pnl30: {np.mean(rpnls):.2f}bps")
    lines.append("")
    return lines


def section_queue_wait(records: list[dict[str, Any]]) -> list[str]:
    """Queue Wait 統計."""
    filled = [r for r in records if r.get("filled")]
    wait_vals = [float(r["queue_wait_sec"]) for r in filled if r.get("queue_wait_sec") is not None]
    lines = ["## Queue Wait"]
    if wait_vals:
        arr = _np(wait_vals)
        lines.append(
            f"  Avg: {np.mean(arr):.1f}s, Median: {np.median(arr):.1f}s, "
            f"p90: {np.percentile(arr, 90):.1f}s, Max: {np.max(arr):.1f}s"
        )
    else:
        lines.append("  (no data)")
    lines.append("")
    return lines


def section_spread(records: list[dict[str, Any]]) -> list[str]:
    """Spread."""
    spreads = [float(r["spread_bps"]) for r in records if r.get("spread_bps") is not None]
    lines = ["## Spread (at order)"]
    if spreads:
        arr = _np(spreads)
        lines.append(
            f"  Mean: {np.mean(arr):.2f}bps, Median: {np.median(arr):.2f}bps, "
            f"p90: {np.percentile(arr, 90):.2f}bps"
        )
    else:
        lines.append("  (no data)")
    lines.append("")
    return lines


def section_balance_forced(records: list[dict[str, Any]]) -> list[str]:
    """Balance Forced."""
    bf = [r for r in records if r.get("cancel_reason") == "balance_forced_skip"]
    lines = ["## Balance Forced"]
    bf_consec = [r.get("balance_forced_consecutive", 0) for r in bf if r.get("balance_forced_consecutive") is not None]
    if bf_consec:
        lines.append(f"  Total: {len(bf)}, consecutive mean={np.mean(bf_consec):.1f}, max={np.max(bf_consec)}")
    else:
        lines.append(f"  Total: {len(bf)}")

    bf_by_day: dict[str, int] = collections.defaultdict(int)
    for r in bf:
        ts = r.get("timestamp")
        if ts:
            d = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%m/%d")
            bf_by_day[d] += 1
    for d in sorted(bf_by_day):
        lines.append(f"  {d}: {bf_by_day[d]} balance_forced")
    lines.append("")
    return lines


def section_sell_guard(records: list[dict[str, Any]]) -> list[str]:
    """Sell Guard / Dynamic Kill."""
    sdk = [r for r in records if r.get("cancel_reason") == "sell_dynamic_kill"]
    tss = [r for r in records if r.get("cancel_reason") == "trending_sell_skip"]
    sgr = [r for r in records if r.get("cancel_reason") == "sell_guard_reject"]
    lines = [
        "## Sell Guard / Dynamic Kill",
        f"  sell_dynamic_kill: {len(sdk)}",
        f"  trending_sell_skip: {len(tss)}",
        f"  sell_guard_reject: {len(sgr)}",
        "",
    ]
    return lines


def section_confidence_lot(records: list[dict[str, Any]]) -> list[str]:
    """Confidence Lot."""
    filled = [r for r in records if r.get("filled")]
    lines = ["## Confidence Lot"]
    lot_modes = collections.Counter(str(r.get("confidence_lot_mode", "?")) for r in filled)
    for m, cnt in lot_modes.most_common(5):
        lines.append(f"  {m}: {cnt}")
    lot_factors = [float(r["confidence_lot_factor"]) for r in filled if r.get("confidence_lot_factor") is not None]
    if lot_factors:
        arr = _np(lot_factors)
        lines.append(f"  lot factor: mean={np.mean(arr):.3f}, min={np.min(arr):.3f}, max={np.max(arr):.3f}")
    lines.append("")
    return lines


def section_early_exit(records: list[dict[str, Any]]) -> list[str]:
    """Early Exit."""
    filled = [r for r in records if r.get("filled")]
    ee = [r for r in filled if r.get("early_exit_triggered")]
    lines = [
        "## Early Exit",
        f"  Early exit: {len(ee)}/{len(filled)} filled",
    ]
    if ee:
        ee_pnls = [float(r["pnl_at_exit_bps"]) for r in ee if r.get("pnl_at_exit_bps") is not None]
        if ee_pnls:
            lines.append(f"  avg_exit_pnl: {np.mean(ee_pnls):.2f}bps")
    lines.append("")
    return lines


def section_volatility_guard(records: list[dict[str, Any]]) -> list[str]:
    """Volatility Guard."""
    n = len(records)
    vg = [r for r in records if r.get("vg_triggered")]
    lines = [
        "## Volatility Guard",
        f"  VG triggered: {len(vg)}/{n} ({len(vg)/n*100:.1f}%)" if n else "  (no data)",
        "",
    ]
    return lines


def section_ffd_boost(records: list[dict[str, Any]]) -> list[str]:
    """FFD Boost."""
    filled = [r for r in records if r.get("filled")]
    nf = len(filled)
    ffd = [r for r in filled if r.get("ffd_boost_active")]
    lines = [
        "## FFD Boost",
        f"  FFD boost active: {len(ffd)}/{nf} ({len(ffd)/nf*100:.1f}%)" if nf else "  (no fills)",
        "",
    ]
    return lines


def section_ab_test(records: list[dict[str, Any]]) -> list[str]:
    """A/B Test Variant."""
    lines = ["## A/B Test Variant"]
    variants = collections.Counter(r.get("ab_test_variant", "none") for r in records)
    for v, cnt in variants.most_common():
        v_filled = [r for r in records if r.get("ab_test_variant") == v and r.get("filled")]
        pnl_arr = _pnls(v_filled)
        avg = float(np.mean(pnl_arr)) if len(pnl_arr) else float("nan")
        lines.append(f"  {v}: {cnt} total, {len(v_filled)} filled, avg_pnl30={avg:.2f}bps")
    lines.append("")
    return lines


def section_ob_age(records: list[dict[str, Any]]) -> list[str]:
    """OB Age."""
    filled = [r for r in records if r.get("filled")]
    ob_ages = [float(r["ob_age_ms"]) for r in filled if r.get("ob_age_ms") is not None]
    lines = ["## OB Age (ms)"]
    if ob_ages:
        arr = _np(ob_ages)
        lines.append(
            f"  Mean: {np.mean(arr):.0f}ms, Median: {np.median(arr):.0f}ms, "
            f"p90: {np.percentile(arr, 90):.0f}ms, Max: {np.max(arr):.0f}ms"
        )
    else:
        lines.append("  (no data)")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def build_json_summary(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """JSON 形式のサマリーを構築."""
    n = len(records)
    filled = [r for r in records if r.get("filled")]
    nf = len(filled)
    pnl_arr = _pnls(filled)

    summary: dict[str, Any] = {
        "filters": {
            "data_dir": args.data_dir,
            "run_id": args.run_id,
            "git_sha": args.git_sha,
            "date_from": args.date_from,
            "date_to": args.date_to,
            "side": args.side,
            "regime": args.regime,
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "total_records": n,
        "filled": nf,
        "fill_rate": round(nf / n * 100, 2) if n else 0,
        "avg_pnl30_bps": round(float(np.mean(pnl_arr)), 4) if len(pnl_arr) else None,
        "sum_pnl30_bps": round(float(np.sum(pnl_arr)), 4) if len(pnl_arr) else None,
    }

    # Side breakdown
    sides: dict[str, Any] = {}
    for side in ["buy", "sell"]:
        s_all = [r for r in records if r.get("side") == side]
        s_filled = [r for r in s_all if r.get("filled")]
        s_pnl = _pnls(s_filled)
        sides[side] = {
            "total": len(s_all),
            "filled": len(s_filled),
            "fill_rate": round(len(s_filled) / len(s_all) * 100, 2) if s_all else 0,
            "avg_pnl30_bps": round(float(np.mean(s_pnl)), 4) if len(s_pnl) else None,
        }
    summary["sides"] = sides

    # git_sha distribution
    sha_dist = collections.Counter(r.get("git_sha", "?") for r in records)
    summary["git_sha_distribution"] = {sha: cnt for sha, cnt in sha_dist.most_common(10)}

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Load
    records = load_records(args.data_dir, args.date_from, args.date_to)

    # Filter
    records = apply_filters(
        records,
        run_id=args.run_id,
        git_sha=args.git_sha,
        date_from=args.date_from,
        date_to=args.date_to,
        side=args.side,
        regime=args.regime,
    )

    if not records:
        print("ERROR: no records after filtering", file=sys.stderr)
        sys.exit(1)

    # JSON mode
    if args.json:
        result = build_json_summary(records, args)
        output = json.dumps(result, indent=2, ensure_ascii=False)
        if args.output:
            pathlib.Path(args.output).write_text(output, encoding="utf-8")
            print(f"JSON written to {args.output}", file=sys.stderr)
        else:
            print(output)
        return

    # Text mode: assemble all sections
    all_lines: list[str] = []
    all_lines.extend(section_header(records, args))
    all_lines.extend(section_basic(records))
    all_lines.extend(section_side(records))
    all_lines.extend(section_regime(records))
    all_lines.extend(section_daily(records))
    all_lines.extend(section_hourly(records))
    all_lines.extend(section_git_sha(records))
    all_lines.extend(section_cancel(records))
    all_lines.extend(section_skip_gate(records))
    all_lines.extend(section_adverse_selection(records))
    all_lines.extend(section_reprice(records))
    all_lines.extend(section_queue_wait(records))
    all_lines.extend(section_spread(records))
    all_lines.extend(section_balance_forced(records))
    all_lines.extend(section_sell_guard(records))
    all_lines.extend(section_confidence_lot(records))
    all_lines.extend(section_early_exit(records))
    all_lines.extend(section_volatility_guard(records))
    all_lines.extend(section_ffd_boost(records))
    all_lines.extend(section_ab_test(records))
    all_lines.extend(section_ob_age(records))

    output_text = "\n".join(all_lines)

    if args.output:
        pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.output).write_text(output_text, encoding="utf-8")
        print(f"Report written to {args.output} ({len(records)} records)", file=sys.stderr)
    else:
        print(output_text)


if __name__ == "__main__":
    main()
