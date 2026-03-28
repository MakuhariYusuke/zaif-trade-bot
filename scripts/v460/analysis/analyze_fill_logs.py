"""Fill Test ログ統合分析スクリプト (162# P0 再現可能分析).

Usage:
    python -m scripts.v460.analysis.analyze_fill_logs
    python -m scripts.v460.analysis.analyze_fill_logs --date-from 2026-02-20 --date-to 2026-02-24
    python -m scripts.v460.analysis.analyze_fill_logs --git-sha d9874bbee12a
    python -m scripts.v460.analysis.analyze_fill_logs --run-id 1771932882_97af3a30
    python -m scripts.v460.analysis.analyze_fill_logs --date-from 2026-02-22 --git-sha 5c65ef925 --output report.txt

Replaces: temp/analyze_logs.py + temp/analyze_logs2.py
Purpose: 因果混在排除のためフィルタ条件を明示して再現可能な分析を行う
Reference: 162# §7.3 P0 / 000# 運用方針追補 §3
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import sys
from datetime import datetime, timezone
from typing import cast

import numpy as np
from numpy.typing import NDArray

from scripts.v460.analysis.analysis_common import (
    DEFAULT_RESULTS_DIR,
    Record,
    FloatArray,
    extract_pnl_array,
    load_and_filter_records,
    write_output,
)

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
        default=DEFAULT_RESULTS_DIR,
        help=f"fill_records_*.jsonl の格納ディレクトリ (default: {DEFAULT_RESULTS_DIR})",
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
# Data Loading — delegates to analysis_common (ztb.metrics.fill_quality 経由)
# ---------------------------------------------------------------------------

# load_records / apply_filters → analysis_common.load_and_filter_records に統合


# ---------------------------------------------------------------------------
# Analysis Sections
# ---------------------------------------------------------------------------

def _np(values: list[float]) -> FloatArray:
    return cast(
        FloatArray,
        np.array(values, dtype=np.float64)
        if values
        else np.array([], dtype=np.float64),
    )


def _pnls(records: list[Record], key: str = "post_fill_30s_pnl") -> FloatArray:
    return extract_pnl_array(records, key=key)


def section_header(records: list[Record], args: argparse.Namespace) -> list[str]:
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


def section_basic(records: list[Record]) -> list[str]:
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


def section_side(records: list[Record]) -> list[str]:
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


def section_regime(records: list[Record]) -> list[str]:
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


def section_cancel(records: list[Record]) -> list[str]:
    """Cancel Reason."""
    cancels = [r for r in records if not r.get("filled")]
    lines = ["## Cancel Reason (top 15)"]
    if not cancels:
        lines.append("  (no cancels)")
    else:
        reasons = collections.Counter(r.get("cancel_reason", "unknown") for r in cancels)
        for reason, cnt in reasons.most_common(15):
            lines.append(f"  {reason}: {cnt} ({cnt/len(cancels)*100:.1f}%)")
            
        # Top 3 reasons breakdown by side and regime
        lines.append("\n  [Breakdown of Top 3 Reasons]")
        for reason, _ in reasons.most_common(3):
            sub_records = [r for r in cancels if r.get("cancel_reason", "unknown") == reason]
            sub_counter = collections.Counter(f"{r.get('side', 'unknown')} / {r.get('regime', 'unknown')}" for r in sub_records)
            lines.append(f"  * {reason}")
            for k, v in sub_counter.most_common(5):
                lines.append(f"      - {k}: {v}")

    lines.append("")
    return lines


def section_skip_gate(records: list[Record]) -> list[str]:
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

    # composite_risk_exceeded などの詳細
    cr_skipped = [r for r in skipped if "composite_risk" in str(r.get("skip_gate_reason", ""))]
    if cr_skipped:
        lines.append("  --- Composite Risk Breakdown ---")
        cr_details = collections.Counter(str(r.get("composite_risk_details", "none")) for r in cr_skipped)
        for detail, dcnt in cr_details.most_common(5):
            lines.append(f"    {detail}: {dcnt}")

    # Model usage
    sg_models = collections.Counter(str(r.get("skip_gate_model_used", "?")) for r in skipped)
    if sg_models:
        lines.append("  --- Model Usage ---")
        for m, cnt in sg_models.most_common(5):
            lines.append(f"    {m}: {cnt}")

    # 622# threshold_used 分布分析 — adaptive bypass 検出
    passed = [r for r in records if not r.get("skip_gate_skipped")]
    if passed:
        th_vals = [r.get("skip_gate_threshold_used") for r in passed
                   if r.get("skip_gate_threshold_used") is not None]
        score_vals = [r.get("skip_gate_score") for r in passed
                      if r.get("skip_gate_score") is not None]
        if th_vals:
            lines.append("  --- Threshold Used (passed fills) ---")
            th_arr = sorted(th_vals)
            lines.append(
                f"    min={th_arr[0]:.3f}  p50={th_arr[len(th_arr)//2]:.3f}  "
                f"max={th_arr[-1]:.3f}  n={len(th_arr)}"
            )
        if score_vals and th_vals and len(score_vals) == len(th_vals):
            margin = [s - t for s, t in zip(score_vals, th_vals)]
            margin.sort()
            lines.append(
                f"    margin(score-th): min={margin[0]:.3f}  "
                f"p50={margin[len(margin)//2]:.3f}  max={margin[-1]:.3f}"
            )
        # Regime 別 threshold 分布
        regime_ths: dict[str, list[float]] = {}
        for r in passed:
            reg = r.get("regime", "unknown")
            th = r.get("skip_gate_threshold_used")
            if th is not None:
                regime_ths.setdefault(reg, []).append(th)
        if len(regime_ths) > 1:
            lines.append("  --- Threshold by Regime ---")
            for reg, vals in sorted(regime_ths.items()):
                vals_s = sorted(vals)
                lines.append(
                    f"    {reg}: min={vals_s[0]:.3f} p50={vals_s[len(vals_s)//2]:.3f} "
                    f"max={vals_s[-1]:.3f} n={len(vals_s)}"
                )
    lines.append("")
    return lines


def section_adverse_selection(records: list[Record]) -> list[str]:
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


def section_hourly(records: list[Record]) -> list[str]:
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


def section_daily(records: list[Record]) -> list[str]:
    """日別サマリ."""
    day_records: dict[str, list[Record]] = collections.defaultdict(list)
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


def section_git_sha(records: list[Record]) -> list[str]:
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


def section_reprice(records: list[Record]) -> list[str]:
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


def section_queue_wait(records: list[Record]) -> list[str]:
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


def section_spread(records: list[Record]) -> list[str]:
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


def section_balance_forced(records: list[Record]) -> list[str]:
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


def section_sell_guard(records: list[Record]) -> list[str]:
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


def section_execution_quality(records: list[Record]) -> list[str]:
    """565# I2: Execution Quality 分解 (Kissell & Glantz 2003).

    PnL = spread_capture + adverse_selection_cost を side×regime で分解し
    offset 戦略の質と市場毒性を独立評価する。
    """
    filled = [r for r in records if r.get("filled")]
    lines = ["## Execution Quality (565# I2 / 305#)"]
    if not filled:
        lines.append("  (no fills)")
        lines.append("")
        return lines

    # --- 全体 ---
    sc_all = _np([float(r["spread_capture_bps"]) for r in filled if r.get("spread_capture_bps") is not None])
    ac_all = _np([float(r["adverse_selection_cost_bps"]) for r in filled if r.get("adverse_selection_cost_bps") is not None])
    if len(sc_all) == 0:
        lines.append(f"  spread_capture_bps 未記録 (0/{len(filled)} fills)")
        lines.append("  → fill_recorder で spread_capture_bps を記録する実装が必要")
        lines.append("")
        return lines
    lines.append(
        f"  Overall: spread_capture={float(np.mean(sc_all)):+.2f}bps, "
        f"AS_cost={float(np.mean(ac_all)):+.2f}bps (n={len(sc_all)})"
    )

    # --- Side × Regime ---
    regimes = sorted({r.get("regime", "unknown") for r in filled})
    for side in ["buy", "sell"]:
        for regime in regimes:
            subset = [
                r for r in filled
                if r.get("requested_side") == side
                and r.get("regime") == regime
                and r.get("spread_capture_bps") is not None
            ]
            if len(subset) < 3:
                continue
            sc = _np([float(r["spread_capture_bps"]) for r in subset])
            ac = _np([float(r["adverse_selection_cost_bps"]) for r in subset
                     if r.get("adverse_selection_cost_bps") is not None])
            pnls = _pnls(subset)
            lines.append(
                f"  {side:4s}/{regime}: "
                f"sc={float(np.mean(sc)):+.2f}, as_cost={float(np.mean(ac)):+.2f}, "
                f"pnl={float(np.mean(pnls)):+.2f}bps (n={len(subset)})"
            )

    # --- AS/Non-AS 別 ---
    for label, predicate in [("AS", True), ("Non-AS", False)]:
        sub = [
            r for r in filled
            if r.get("adverse_selected") == predicate
            and r.get("spread_capture_bps") is not None
        ]
        if len(sub) < 3:
            continue
        sc = _np([float(r["spread_capture_bps"]) for r in sub])
        ac = _np([float(r["adverse_selection_cost_bps"]) for r in sub
                 if r.get("adverse_selection_cost_bps") is not None])
        lines.append(
            f"  {label}: spread_capture={float(np.mean(sc)):+.2f}, "
            f"AS_cost={float(np.mean(ac)):+.2f} (n={len(sub)})"
        )
    lines.append("")
    return lines


def section_clamp_saturation(records: list[Record]) -> list[str]:
    """531# Offset Clamp Saturation — pipeline 出力の ceiling 飽和率.

    565# I3: pre_clamp offset の分布（p50/p75/p90/p99）を追加。
    ceiling 引上げ幅の根拠データとして使用する。
    """
    filled = [r for r in records if r.get("filled")]
    lines = ["## Clamp Saturation (531# / 565# I3)"]
    if not filled:
        lines.append("  (no fills)")
        lines.append("")
        return lines

    for side in ["buy", "sell"]:
        sf = [r for r in filled if r.get("requested_side") == side]
        with_data = [
            r for r in sf
            if r.get("execution_pre_clamp_offset") is not None
            and r.get("effective_offset_used") is not None
        ]
        clamped = [
            r for r in with_data
            if r["execution_pre_clamp_offset"] > r["effective_offset_used"] + 0.001
        ]
        n_wd = len(with_data)
        n_c = len(clamped)
        if n_wd:
            pre_vals = _np([r["execution_pre_clamp_offset"] for r in with_data])
            eff_vals = _np([r["effective_offset_used"] for r in with_data])
            clamp_pnl = _pnls(clamped)
            unclamp_pnl = _pnls([r for r in with_data if r not in clamped])
            lines.append(
                f"  {side}: clamped {n_c}/{n_wd} ({n_c/n_wd*100:.0f}%), "
                f"pre_clamp avg={float(np.mean(pre_vals)):.4f}, "
                f"effective avg={float(np.mean(eff_vals)):.4f}"
            )
            if len(clamp_pnl):
                lines.append(f"    clamped PnL avg={float(np.mean(clamp_pnl)):.2f}bps")
            if len(unclamp_pnl):
                lines.append(f"    unclamped PnL avg={float(np.mean(unclamp_pnl)):.2f}bps")
            # 565# I3: pre_clamp offset 分布
            p50 = float(np.percentile(pre_vals, 50))
            p75 = float(np.percentile(pre_vals, 75))
            p90 = float(np.percentile(pre_vals, 90))
            p99 = float(np.percentile(pre_vals, 99))
            lines.append(
                f"    pre_clamp distribution: "
                f"p50={p50:.4f}, p75={p75:.4f}, p90={p90:.4f}, p99={p99:.4f}"
            )
        else:
            lines.append(f"  {side}: (no offset data)")
    lines.append("")
    return lines


def section_information_loss(records: list[Record]) -> list[str]:
    """614# Phase 1 Attribution: Information Loss from Clamping."""
    filled = [r for r in records if r.get("filled")]
    lines = ["--- Pipeline Attribution (Phase 1) ---", "## Information Loss"]
    if not filled:
        lines.append("  (no fills)")
        lines.append("")
        return lines

    for side in ["buy", "sell"]:
        sf = [r for r in filled if r.get("requested_side") == side]
        with_data = [
            r for r in sf
            if r.get("execution_pre_clamp_offset") is not None
            and r.get("effective_offset_used") is not None
        ]
        clamped = [
            r for r in with_data
            if r["execution_pre_clamp_offset"] > r["effective_offset_used"] + 0.0001
        ]
        if not with_data:
            lines.append(f"  {side}: (no data)")
            continue
        
        # Calculate loss in bps (ratios to bps)
        loss_bps = [(r["execution_pre_clamp_offset"] - r["effective_offset_used"]) * 10000 for r in clamped]
        avg_loss = float(np.mean(loss_bps)) if loss_bps else 0.0
        total_loss = float(np.sum(loss_bps)) if loss_bps else 0.0
        lines.append(f"  {side}: Loss applied in {len(clamped)}/{len(with_data)} fills")
        if clamped:
            lines.append(f"    Avg Loss: {avg_loss:.2f} bps, Total Loss: {total_loss:.2f} bps")
    lines.append("")
    return lines


def section_stage_saturation(records: list[Record]) -> list[str]:
    """614# Phase 1 Attribution: Stage Saturation (>= 1.99)."""
    import json
    filled = [r for r in records if r.get("filled")]
    lines = ["## Stage Saturation"]
    if not filled:
        lines.append("  (no fills)")
        lines.append("")
        return lines

    saturations: dict[str, collections.Counter[str]] = {
        "buy": collections.Counter(),
        "sell": collections.Counter(),
    }
    valid_records = {"buy": 0, "sell": 0}

    for r in filled:
        side = r.get("requested_side")
        if side not in ["buy", "sell"]:
            continue
        raw_stages = r.get("executor_offset_stages")
        if not raw_stages or not isinstance(raw_stages, str):
            continue
        try:
            stages = json.loads(raw_stages)
            valid_records[side] += 1
            for k, v in stages.items():
                if v is None:
                    continue
                if abs(float(v)) >= 1.99:
                    saturations[side][k] += 1
        except json.JSONDecodeError:
            pass

    for side in ["buy", "sell"]:
        if valid_records[side] == 0:
            lines.append(f"  {side}: (no stage data)")
            continue
        lines.append(f"  {side}: valid parses={valid_records[side]}")
        if not saturations[side]:
            lines.append("    No saturated stages (>= 1.99)")
        else:
            for k, v in saturations[side].most_common():
                lines.append(f"    {k}: {v}/{valid_records[side]} ({v/valid_records[side]*100:.1f}%)")
    lines.append("")
    return lines


def section_cross_venue_engagement(records: list[Record]) -> list[str]:
    """531# Cross-Venue Engagement — CV適用率・方向別PnL."""
    filled = [r for r in records if r.get("filled")]
    lines = ["## Cross-Venue Engagement (531#)"]
    if not filled:
        lines.append("  (no fills)")
        lines.append("")
        return lines

    cv_applied = [r for r in filled if r.get("cross_venue_lead_lag_applied")]
    cv_vetoed = [r for r in records if r.get("cross_venue_lead_lag_vetoed")]
    lines.append(
        f"  CV applied: {len(cv_applied)}/{len(filled)} fills "
        f"({len(cv_applied)/len(filled)*100:.1f}%), vetoed: {len(cv_vetoed)} cycles"
    )

    for side in ["buy", "sell"]:
        sf_cv = [r for r in cv_applied if r.get("requested_side") == side]
        if not sf_cv:
            continue
        # Classify: tighten (post < pre) vs widen (post > pre) vs neutral
        tighten = [
            r for r in sf_cv
            if r.get("cross_venue_lead_lag_pre_offset") is not None
            and r.get("cross_venue_lead_lag_post_offset") is not None
            and r["cross_venue_lead_lag_post_offset"] < r["cross_venue_lead_lag_pre_offset"] - 0.001
        ]
        widen = [
            r for r in sf_cv
            if r.get("cross_venue_lead_lag_pre_offset") is not None
            and r.get("cross_venue_lead_lag_post_offset") is not None
            and r["cross_venue_lead_lag_post_offset"] > r["cross_venue_lead_lag_pre_offset"] + 0.001
        ]
        cap_hit = sum(1 for r in sf_cv if r.get("cross_venue_lead_lag_cap_hit"))
        t_pnl = _pnls(tighten)
        w_pnl = _pnls(widen)
        lines.append(f"  {side}: {len(sf_cv)} fills (tighten={len(tighten)}, widen={len(widen)}, cap_hit={cap_hit})")
        if len(t_pnl):
            lines.append(f"    tighten PnL avg={float(np.mean(t_pnl)):+.2f}bps")
        if len(w_pnl):
            lines.append(f"    widen PnL avg={float(np.mean(w_pnl)):+.2f}bps")
    lines.append("")
    return lines


def section_tail_risk(records: list[Record]) -> list[str]:
    """531# Tail Risk Concentration — 損失のテール集中度."""
    filled = [r for r in records if r.get("filled")]
    lines = ["## Tail Risk Concentration (531#)"]
    if len(filled) < 5:
        lines.append("  (insufficient fills)")
        lines.append("")
        return lines

    pnls = [(r.get("requested_side", "?"), float(r.get("post_fill_30s_pnl", 0))) for r in filled]
    pnls_sorted = sorted(pnls, key=lambda x: x[1])
    total_pnl = sum(p for _, p in pnls)
    total_loss = sum(p for _, p in pnls if p < 0)

    # Worst 5 fills
    worst5 = pnls_sorted[:5]
    worst5_sum = sum(p for _, p in worst5)
    lines.append(f"  Total PnL: {total_pnl:+.2f}bps, Total Loss: {total_loss:.2f}bps")
    lines.append(f"  Worst 5 fills: {worst5_sum:.2f}bps", )
    if total_loss < 0:
        lines.append(f"    = {abs(worst5_sum / total_loss) * 100:.0f}% of total loss")
    lines.append("  Worst fills:")
    for side, p in worst5:
        lines.append(f"    {side}: {p:+.2f}bps")
    lines.append("")
    return lines


def section_confidence_lot(records: list[Record]) -> list[str]:
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


def section_early_exit(records: list[Record]) -> list[str]:
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


def section_volatility_guard(records: list[Record]) -> list[str]:
    """Volatility Guard."""
    n = len(records)
    vg = [r for r in records if r.get("vg_triggered")]
    lines = [
        "## Volatility Guard",
        f"  VG triggered: {len(vg)}/{n} ({len(vg)/n*100:.1f}%)" if n else "  (no data)",
        "",
    ]
    return lines


def section_ffd_boost(records: list[Record]) -> list[str]:
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


def section_ab_test(records: list[Record]) -> list[str]:
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


def section_ob_age(records: list[Record]) -> list[str]:
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


def section_model_used(records: list[Record]) -> list[str]:
    """165# model_used 経路別 AS/PnL 分析."""
    filled = [r for r in records if r.get("filled")]
    nf = len(filled)
    lines = ["## Model Used 経路別 (165# 7.3)"]
    if not nf:
        lines.append("  (no fills)")
        lines.append("")
        return lines

    groups: dict[str, list[Record]] = collections.defaultdict(list)
    for r in filled:
        model = str(r.get("skip_gate_model_used") or "none")
        groups[model].append(r)

    lines.append(f"  {'Model':>25s} {'N':>5s} {'AS#':>4s} {'AS%':>6s} {'PnL30':>8s} {'AS_Loss':>8s}")
    for model in sorted(groups.keys()):
        recs = groups[model]
        as_recs = [r for r in recs if r.get("adverse_selected")]
        as_rate = len(as_recs) / len(recs) * 100
        pnl_arr = _pnls(recs)
        avg_pnl = float(np.mean(pnl_arr)) if len(pnl_arr) else float("nan")
        as_pnl_arr = _pnls(as_recs)
        avg_as = float(np.mean(as_pnl_arr)) if len(as_pnl_arr) else 0.0
        lines.append(
            f"  {model:>25s} {len(recs):>5d} {len(as_recs):>4d} "
            f"{as_rate:>5.1f}% {avg_pnl:>+8.2f} {avg_as:>+8.2f}"
        )
    lines.append("")
    return lines


def build_json_summary(records: list[Record], args: argparse.Namespace) -> dict[str, object]:
    """JSON 形式のサマリーを構築."""
    n = len(records)
    filled = [r for r in records if r.get("filled")]
    nf = len(filled)
    pnl_arr = _pnls(filled)

    summary: dict[str, object] = {
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
    sides: dict[str, object] = {}
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


def section_microstructure_correlation(records: list[Record]) -> list[str]:
    """561# Sell AS とマイクロ構造指標の相関深掘り."""
    filled_sell = [r for r in records if r.get("side") == "sell" and r.get("filled")]
    if not filled_sell:
        return ["## Microstructure Correlation (Sell)", "  (no filled sell orders)", ""]

    lines = ["## Microstructure Correlation (Sell Side AS Deep Dive)"]

    # 1. Spread vs AS
    spreads = [float(r["spread_bps"]) for r in filled_sell if r.get("spread_bps") is not None]
    if spreads:
        as_flags = [1 if r.get("adverse_selected") else 0 for r in filled_sell if r.get("spread_bps") is not None]
        avg_spread = float(np.mean(spreads))
        # スプレッド帯別のAS率
        low_spread_as = [a for s, a in zip(spreads, as_flags) if s < avg_spread]
        high_spread_as = [a for s, a in zip(spreads, as_flags) if s >= avg_spread]
        if low_spread_as and high_spread_as:
            lines.append(f"  AS Rate by Spread: Low (<{avg_spread:.2f}bps): {np.mean(low_spread_as)*100:.1f}% (n={len(low_spread_as)}), High: {np.mean(high_spread_as)*100:.1f}% (n={len(high_spread_as)})")

    # 2. Orderbook Imbalance vs AS
    imbalances = [
        float(r["orderbook_imbalance"])
        for r in filled_sell
        if r.get("orderbook_imbalance") is not None
    ]
    if imbalances:
        as_flags = [1 if r.get("adverse_selected") else 0 for r in filled_sell if r.get("orderbook_imbalance") is not None]
        # Imbalance > 0.3 (買い圧が強い) 時の Sell AS 率
        toxic_imbalance = [a for i, a in zip(imbalances, as_flags) if i > 0.3]
        normal_imbalance = [a for i, a in zip(imbalances, as_flags) if i <= 0.3]
        if toxic_imbalance:
            lines.append(f"  AS Rate by Imbalance: Toxic (>0.3): {np.mean(toxic_imbalance)*100:.1f}% (n={len(toxic_imbalance)}), Normal: {np.mean(normal_imbalance)*100:.1f}% (n={len(normal_imbalance)})")

    # 3. VPIN (Liquidity Risk) vs AS
    vpins = [float(r["vg_vpin"]) for r in filled_sell if r.get("vg_vpin") is not None]
    if vpins:
        as_flags = [1 if r.get("adverse_selected") else 0 for r in filled_sell if r.get("vg_vpin") is not None]
        avg_vpin = float(np.mean(vpins))
        high_vpin_as = [a for v, a in zip(vpins, as_flags) if v > avg_vpin]
        low_vpin_as = [a for v, a in zip(vpins, as_flags) if v <= avg_vpin]
        if high_vpin_as and low_vpin_as:
            lines.append(f"  AS Rate by VPIN: High (>{avg_vpin:.2f}): {np.mean(high_vpin_as)*100:.1f}% (n={len(high_vpin_as)}), Low: {np.mean(low_vpin_as)*100:.1f}% (n={len(low_vpin_as)})")

    lines.append("")
    return lines


def section_pre_clamp_distribution(records: list[Record]) -> list[str]:
    """565# I3: pre_clamp offset 分布の取得."""
    filled = [r for r in records if r.get("filled")]
    if not filled:
        return ["## Pre-clamp Offset Distribution", "  (no fills)", ""]

    lines = ["## Pre-clamp Offset Distribution (I3)"]
    for side in ["buy", "sell"]:
        vals = [
            float(r["execution_pre_clamp_offset"]) 
            for r in filled 
            if r.get("side") == side and r.get("execution_pre_clamp_offset") is not None
        ]
        if vals:
            arr = np.array(vals)
            lines.append(
                f"  {side:4s}: Median={np.median(arr):.4f}, P90={np.percentile(arr, 90):.4f}, "
                f"P99={np.percentile(arr, 99):.4f}, Max={np.max(arr):.4f} (n={len(arr)})"
            )
    lines.append("")
    return lines


def section_execution_quality_comparison(records: list[Record]) -> list[str]:
    """582# 加法 vs 乗法パイプラインの執行品質比較分析.

    execution_additive_enabled を優先し、未記録の旧データでは
    executor_offset_stages JSON に tox_buffer が含まれていれば additive とみなす。
    それ以外は multiplicative として分類し Kissell & Glantz 指標を比較する。
    """
    import json as _json

    filled = [r for r in records if r.get("filled")]
    if not filled:
        return ["## Execution Quality Comparison", "  (no fills)", ""]

    def _load_executor_offset_stages(record: Record) -> dict[str, object] | None:
        _raw = record.get("executor_offset_stages")
        if not _raw or not isinstance(_raw, str):
            return None
        try:
            _parsed = _json.loads(_raw)
        except (ValueError, TypeError):
            return None
        return _parsed if isinstance(_parsed, dict) else None

    def _is_additive_execution(record: Record) -> bool:
        _explicit = record.get("execution_additive_enabled")
        if _explicit is not None:
            return bool(_explicit)
        _stages = _load_executor_offset_stages(record)
        return bool(_stages and "tox_buffer" in _stages)

    groups: dict[str, list[Record]] = {"multiplicative": [], "additive": []}
    for r in filled:
        _is_additive = _is_additive_execution(r)
        mode = "additive" if _is_additive else "multiplicative"
        groups[mode].append(r)

    lines = ["## Execution Quality Comparison (571# Additive vs Multiplicative)"]

    for mode, recs in groups.items():
        if not recs:
            continue

        captures = [
            float(r["spread_capture_bps"])
            for r in recs
            if r.get("spread_capture_bps") is not None
        ]
        costs = [
            float(r["adverse_selection_cost_bps"])
            for r in recs
            if r.get("adverse_selection_cost_bps") is not None
        ]

        if not captures or not costs:
            lines.append(f"  --- {mode.upper()} (n={len(recs)}) ---")
            lines.append("    (spread_capture_bps / adverse_selection_cost_bps not recorded)")
            continue

        avg_cap = float(np.mean(captures))
        avg_cost = float(np.mean(costs))
        net_pnl = avg_cap + avg_cost

        icr = avg_cap / abs(avg_cost) if avg_cost != 0 else 0.0

        lines.append(f"  --- {mode.upper()} (n={len(recs)}) ---")
        lines.append(f"    Spread Capture: {avg_cap:+.2f} bps")
        lines.append(f"    AS Cost (90s):  {avg_cost:+.2f} bps")
        lines.append(f"    Net Spread PnL: {net_pnl:+.2f} bps")
        lines.append(f"    Info Capture Ratio: {icr:.3f}")

    if not any(groups.values()):
        lines.append("  (no pipeline data)")

    lines.append("")
    return lines


def section_spread_decomposition(records: list[Record]) -> list[str]:
    """565# I2: Spread Capture / Adverse Selection Cost 分解."""
    filled = [r for r in records if r.get("filled")]
    if not filled:
        return ["## Spread Decomposition", "  (no fills)", ""]

    lines = ["## Spread Decomposition (I2: Kissell & Glantz)"]
    regimes = sorted(list(set(str(r.get("regime", "null")) for r in records)))
    
    for side in ["buy", "sell"]:
        lines.append(f"  --- {side.upper()} ---")
        for reg in regimes:
            rf = [
                r for r in filled 
                if r.get("side") == side and str(r.get("regime", "null")) == reg
            ]
            if not rf:
                continue
            
            captures = [float(r["spread_capture_bps"]) for r in rf if r.get("spread_capture_bps") is not None]
            costs = [float(r["adverse_selection_cost_bps"]) for r in rf if r.get("adverse_selection_cost_bps") is not None]
            
            if captures and costs:
                avg_cap = np.mean(captures)
                avg_cost = np.mean(costs)
                lines.append(
                    f"    {reg:12s}: Capture={avg_cap:+.2f}bps, AS_Cost={avg_cost:+.2f}bps, Net={avg_cap+avg_cost:+.2f}bps (n={len(rf)})"
                )
    lines.append("")
    return lines


def section_buffer_decomposition(records: list[Record]) -> list[str]:
    """582# Toxicity / Liquidity バッファ分解分析.

    executor_offset_stages JSON から tox_buffer / liq_buffer を抽出し、
    side × regime 別にバッファ寄与度を分析する。
    """
    import json as _json

    filled = [r for r in records if r.get("filled")]
    if not filled:
        return ["## Buffer Decomposition (582#)", "  (no fills)", ""]

    # Parse stages JSON
    parsed: list[tuple[Record, dict[str, object]]] = []
    for r in filled:
        _raw = r.get("executor_offset_stages")
        if not _raw or not isinstance(_raw, str):
            continue
        try:
            stages = _json.loads(_raw)
        except (ValueError, TypeError):
            continue
        if "tox_buffer" in stages:
            parsed.append((r, stages))

    if not parsed:
        return [
            "## Buffer Decomposition (582#)",
            "  (no additive pipeline data — tox_buffer not found in stages)",
            "",
        ]

    lines = ["## Buffer Decomposition (582# Toxicity vs Liquidity)"]
    lines.append(f"  Additive fills: {len(parsed)} / {len(filled)} total fills")
    lines.append("")

    tox_vals = [
        float(cast(float | int | str, s["tox_buffer"]))
        for _, s in parsed
        if s.get("tox_buffer") is not None
    ]
    liq_vals = [
        float(cast(float | int | str, s["liq_buffer"]))
        for _, s in parsed
        if s.get("liq_buffer") is not None
    ]
    lines.append("  --- OVERALL ---")
    lines.append(f"    Tox Buffer: mean={np.mean(tox_vals):.4f}, p50={np.median(tox_vals):.4f}, p90={np.percentile(tox_vals, 90):.4f}")
    lines.append(f"    Liq Buffer: mean={np.mean(liq_vals):.4f}, p50={np.median(liq_vals):.4f}, p90={np.percentile(liq_vals, 90):.4f}")
    tox_dominant = sum(1 for t, l in zip(tox_vals, liq_vals) if bool(t > l))
    lines.append(
        f"    Tox Dominant: {tox_dominant}/{len(parsed)} "
        f"({100 * tox_dominant / len(parsed):.1f}%)"
    )
    lines.append("")

    # Per side
    for side in ["buy", "sell"]:
        side_data = [(r, s) for r, s in parsed if r.get("side") == side]
        if not side_data:
            continue
        t = [s["tox_buffer"] for _, s in side_data]
        l = [s["liq_buffer"] for _, s in side_data]
        lines.append(f"  --- {side.upper()} (n={len(side_data)}) ---")
        lines.append(f"    Tox: mean={np.mean(t):.4f}, Liq: mean={np.mean(l):.4f}")

        # Per-stage contribution breakdown
        stage_names = ["velocity", "trending", "toxicity", "vg_supp", "alert", "ev", "macro"]
        for sn in stage_names:
            vals = [s.get(sn) for _, s in side_data if s.get(sn) is not None]
            if vals:
                lines.append(f"      {sn:12s}: active={len(vals)}/{len(side_data)} ({100*len(vals)/len(side_data):.0f}%), mean_mult={np.mean(vals):.3f}")

    lines.append("")
    return lines


def section_attribution_phase2(records: list[Record]) -> list[str]:
    """616# §1: Euler RMS 分解 + Ceiling Occupancy.

    加法パイプラインの tox/liq RMS を Euler 定理で個別ステージに分解し、
    Ceiling に対する消費率 (Occupancy) を算出する。
    """
    import json as _json

    filled = [r for r in records if r.get("filled")]
    lines = ["## Attribution Phase 2 (616# §1 Euler RMS Decomposition)"]
    if not filled:
        lines.append("  (no fills)")
        lines.append("")
        return lines

    # Tox ステージ名 / Liq ステージ名
    tox_stage_names = ["velocity", "trending", "toxicity", "vg_supp", "alert"]
    liq_stage_names = ["ev", "macro"]

    # Ceiling 値は fill_record から取得できないので、
    # execution_pre_clamp_offset と effective_offset_used の差で間接推定
    # → 直接 base_ratio を使えないため、base = effective - tox_rms - liq_rms で逆算

    for side in ["buy", "sell"]:
        side_records = [r for r in filled if r.get("requested_side") == side]
        parsed: list[tuple[dict[str, object], dict[str, float]]] = []
        for r in side_records:
            raw = r.get("executor_offset_stages")
            if not raw or not isinstance(raw, str):
                continue
            try:
                stages = _json.loads(raw)
            except (ValueError, TypeError):
                continue
            if "tox_buffer" not in stages:
                continue
            parsed.append((r, stages))

        if not parsed:
            lines.append(f"  {side}: (no additive pipeline data)")
            continue

        lines.append(f"  --- {side.upper()} (n={len(parsed)}) ---")

        # Euler 寄与度を集計
        euler_contrib: dict[str, list[float]] = {s: [] for s in tox_stage_names + liq_stage_names}
        occupancy_vals: list[float] = []

        for r, stages in parsed:
            tox_rms = float(stages.get("tox_buffer", 0.0))
            liq_rms = float(stages.get("liq_buffer", 0.0))

            # 各 tox ステージの delta を再構築:
            # stages[name] は multiplier 値。delta_i = base * (mult - 1.0)
            # ただし base_ratio は不明なので、C_i の比率のみで按分
            # Euler: C_i = delta_i^2 / rms
            if tox_rms > 1e-8:
                tox_sq_sum = 0.0
                stage_sq: dict[str, float] = {}
                for sn in tox_stage_names:
                    mult = stages.get(sn)
                    if mult is None or float(mult) <= 1.0:
                        stage_sq[sn] = 0.0
                        continue
                    # delta は未知 base を含むが、C_i/tox_rms = delta_i^2/Σdelta_j^2
                    # → tox_rms^2 = Σdelta_j^2 なので C_i = delta_i^2 / tox_rms
                    # delta_i^2 は比例項として (mult-1)^2 を使用
                    sq = (float(mult) - 1.0) ** 2
                    stage_sq[sn] = sq
                    tox_sq_sum += sq

                for sn in tox_stage_names:
                    if tox_sq_sum > 0:
                        # C_i = tox_rms * (sq_i / tox_sq_sum)
                        euler_contrib[sn].append(tox_rms * stage_sq[sn] / tox_sq_sum)
                    else:
                        euler_contrib[sn].append(0.0)

            if liq_rms > 1e-8:
                liq_sq_sum = 0.0
                liq_stage_sq: dict[str, float] = {}
                for sn in liq_stage_names:
                    mult = stages.get(sn)
                    if mult is None or float(mult) <= 1.0:
                        liq_stage_sq[sn] = 0.0
                        continue
                    sq = (float(mult) - 1.0) ** 2
                    liq_stage_sq[sn] = sq
                    liq_sq_sum += sq

                for sn in liq_stage_names:
                    if liq_sq_sum > 0:
                        euler_contrib[sn].append(liq_rms * liq_stage_sq[sn] / liq_sq_sum)
                    else:
                        euler_contrib[sn].append(0.0)

            # Occupancy: tox_rms / (ceiling - base_ratio) * 100
            eff_obj = r.get("effective_offset_used")
            eff = float(eff_obj) if isinstance(eff_obj, (int, float)) else None
            if eff is not None and eff > 1e-8:
                base_est = eff - tox_rms - liq_rms
                if base_est > 0:
                    headroom = eff - base_est  # = tox_rms + liq_rms
                    # pre_clamp があれば ceiling を推定可能
                    pre_clamp_obj = r.get("execution_pre_clamp_offset")
                    pre_clamp = (
                        float(pre_clamp_obj)
                        if isinstance(pre_clamp_obj, (int, float))
                        else None
                    )
                    if pre_clamp is not None:
                        ceiling_est = max(pre_clamp, eff)
                        denom = ceiling_est - base_est
                        if denom > 1e-8:
                            occupancy_vals.append(headroom / denom * 100.0)

        # Euler 寄与度の出力
        lines.append("    Euler Contribution (avg bps):")
        for sn in tox_stage_names + liq_stage_names:
            vals = euler_contrib[sn]
            if vals:
                avg_c = float(np.mean(vals)) * 10000  # ratio → bps
                max_c = float(np.max(vals)) * 10000
                nonzero = sum(1 for v in vals if v > 1e-8)
                if nonzero > 0:
                    lines.append(
                        f"      {sn:12s}: avg={avg_c:+.2f}bps, max={max_c:.2f}bps, "
                        f"active={nonzero}/{len(vals)}"
                    )

        # Occupancy の出力
        if occupancy_vals:
            occ_arr = _np(occupancy_vals)
            lines.append(
                f"    Occupancy: mean={float(np.mean(occ_arr)):.1f}%, "
                f"p90={float(np.percentile(occ_arr, 90)):.1f}%, "
                f"saturated(>100%)={sum(1 for o in occupancy_vals if o > 100)}/"
                f"{len(occupancy_vals)}"
            )
        else:
            lines.append("    Occupancy: (insufficient data)")

    lines.append("")
    return lines


def section_sidecar_signal(records: list[Record]) -> list[str]:
    """589# Sidecar signal status 分布分析.

    sidecar_signal_status (fresh/stale/missing/error) の分布と
    ステータス別 PnL を集計し、SAC sidecar の有効性を可視化する。
    """
    filled = [r for r in records if r.get("filled")]
    if not filled:
        return ["## Sidecar Signal Distribution (589#)", "  (no fills)", ""]

    statuses: dict[str, list[Record]] = {}
    for r in filled:
        status = r.get("sidecar_signal_status") or "missing"
        statuses.setdefault(status, []).append(r)

    lines = ["## Sidecar Signal Distribution (589#)"]
    total = len(filled)

    for status in ["fresh", "stale", "missing", "error"]:
        recs = statuses.get(status, [])
        n = len(recs)
        pct = 100.0 * n / total if total else 0.0
        if not recs:
            lines.append(f"  {status:8s}: {n:4d} ({pct:5.1f}%)")
            continue

        pnls = [float(r["post_fill_30s_pnl"]) for r in recs if r.get("post_fill_30s_pnl") is not None]
        as_vals = [float(r["adverse_selection_cost_bps"]) for r in recs if r.get("adverse_selection_cost_bps") is not None]

        pnl_str = f"PnL30s mean={np.mean(pnls):+.1f}" if pnls else "PnL30s N/A"
        as_str = f"AS mean={np.mean(as_vals):+.2f}bps" if as_vals else "AS N/A"
        lines.append(f"  {status:8s}: {n:4d} ({pct:5.1f}%)  {pnl_str}  {as_str}")

    # 未知ステータスがあれば表示
    for status, recs in statuses.items():
        if status not in ("fresh", "stale", "missing", "error"):
            lines.append(f"  {status:8s}: {len(recs):4d} ({100.0 * len(recs) / total:5.1f}%)  (unknown)")

    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Load + Filter (analysis_common に委譲)
    records = load_and_filter_records(
        args.data_dir,
        date_from=args.date_from,
        date_to=args.date_to,
        git_sha=args.git_sha,
        run_id=args.run_id,
        side=args.side,
        regime=args.regime,
    )

    # JSON mode
    if args.json:
        from scripts.v460.analysis.analysis_common import write_json_output
        result = build_json_summary(records, args)
        write_json_output(result, args.output)
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
    all_lines.extend(section_execution_quality(records))
    all_lines.extend(section_clamp_saturation(records))
    all_lines.extend(section_information_loss(records))
    all_lines.extend(section_stage_saturation(records))
    all_lines.extend(section_cross_venue_engagement(records))
    all_lines.extend(section_tail_risk(records))
    all_lines.extend(section_confidence_lot(records))
    all_lines.extend(section_early_exit(records))
    all_lines.extend(section_volatility_guard(records))
    all_lines.extend(section_ffd_boost(records))
    all_lines.extend(section_ab_test(records))
    all_lines.extend(section_ob_age(records))
    all_lines.extend(section_model_used(records))
    all_lines.extend(section_microstructure_correlation(records))
    all_lines.extend(section_pre_clamp_distribution(records))
    all_lines.extend(section_spread_decomposition(records))
    all_lines.extend(section_execution_quality_comparison(records))
    all_lines.extend(section_buffer_decomposition(records))
    all_lines.extend(section_attribution_phase2(records))
    all_lines.extend(section_sidecar_signal(records))

    write_output("\n".join(all_lines), args.output)


if __name__ == "__main__":
    main()
