"""333# SHA-isolated comprehensive analysis of fill records.

334# §7.P1-5 promotion: 再現可能な SHA 固定分析スクリプト。
temp/sha_analysis.py + temp/sha_combined_analysis.py を統合し、
プロジェクトの既存ユーティリティを活用した正式版。

Usage:
    python analysis/333_sha_isolated_analysis.py                    # デフォルト SHA
    python analysis/333_sha_isolated_analysis.py --sha dcc3064 4e67014
    python analysis/333_sha_isolated_analysis.py --sha 114a0f0 --json

Outputs:
    - stdout: 人間可読レポート
    - --json: analysis_results/333_sha_isolated_<sha1>[_<sha2>...].json
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import TypeAlias, TypedDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.v460.analysis.analysis_common import (
    AS_THRESHOLD_BPS,
    Record,
    SEVERE_AS_THRESHOLD_BPS,
    get_pnl,
    write_json_output,
)
from scripts.v460.lib.ab_judgment import (
    ABJudgmentCriteria,
    evaluate_ab_variant,
)
from ztb.metrics.fill_quality import load_fill_record_objects_glob

# ======================================================================
# Configuration
# ======================================================================

RESULTS_DIR = _PROJECT_ROOT / "results" / "v460" / "fill_test"
OUTPUT_DIR = _PROJECT_ROOT / "analysis_results"

# 333# デフォルト: dcc3064 + functionally equivalent 4e67014
DEFAULT_SHAS = ["dcc3064", "4e67014"]

# AS_THRESHOLD_BPS / SEVERE_AS_THRESHOLD_BPS → analysis_common からインポート

FillRecord: TypeAlias = Record


class RegimeBuckets(TypedDict):
    """レジーム別の fill/all バケット."""

    filled: list[FillRecord]
    all: list[FillRecord]


class HourlyBuckets(TypedDict):
    """時間帯別 PnL バケット."""

    pnls: list[float]
    sell_pnls: list[float]
    buy_pnls: list[float]


class DailyBuckets(TypedDict):
    """日次集計の可変バケット."""

    total: int
    filled: int
    pnls: list[float]
    bf: int


# ======================================================================
# Helper dataclasses for JSON output
# ======================================================================


@dataclass
class SideMetrics:
    """片サイドの集計結果."""

    side: str
    n_total: int
    n_filled: int
    fill_rate: float
    avg_pnl_bps: float
    sum_pnl_bps: float
    win_rate: float
    p10_bps: float
    p25_bps: float
    p50_bps: float
    p75_bps: float
    p90_bps: float
    as_rate: float
    severe_as_rate: float


@dataclass
class RegimeMetrics:
    """レジーム別集計."""

    regime: str
    n_total: int
    n_filled: int
    fill_rate: float
    avg_pnl_bps: float | None = None
    by_side: list[SideMetrics] = field(default_factory=list)


@dataclass
class DailyMetrics:
    """日次集計."""

    day: str
    n_total: int
    n_filled: int
    fill_rate: float
    avg_pnl_bps: float | None = None
    sum_pnl_bps: float | None = None
    win_rate: float | None = None
    p10_bps: float | None = None


@dataclass
class ABSideJudgment:
    """AB判定 (片サイド)."""

    side: str
    fill_rate: float
    fill_rate_verdict: str
    avg_pnl_bps: float
    avg_pnl_verdict: str
    p10_bps: float
    p10_verdict: str
    overall_verdict: str


@dataclass
class AnalysisResult:
    """全体結果."""

    shas: list[str]
    n_records: int
    n_filled: int
    n_skipped: int
    fill_rate: float
    period_start_jst: str | None = None
    period_end_jst: str | None = None
    duration_hours: float = 0.0
    overall: SideMetrics | None = None
    by_side: list[SideMetrics] = field(default_factory=list)
    by_regime: list[RegimeMetrics] = field(default_factory=list)
    daily: list[DailyMetrics] = field(default_factory=list)
    ab_judgments: list[ABSideJudgment] = field(default_factory=list)
    ab_overall: str = ""
    cancel_reasons: dict[str, int] = field(default_factory=dict)


# ======================================================================
# Utilities
# ======================================================================


# _get_pnl → analysis_common.get_pnl


def _ts_to_jst(ts: float) -> str:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc) + timedelta(hours=9)
    return dt.strftime("%Y-%m-%d %H:%M")


def _ts_to_jst_day(ts: float) -> str:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc) + timedelta(hours=9)
    return dt.strftime("%m/%d")


def _percentile(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def _is_target(r: FillRecord, sha_prefixes: list[str]) -> bool:
    sha = (str(r.get("git_sha", "")) or "")[:7]
    return any(sha == prefix[:7] for prefix in sha_prefixes)


def _compute_side_metrics(
    filled: list[FillRecord], all_records: list[FillRecord], side: str,
) -> SideMetrics | None:
    """指定サイドの集計."""
    sf = [r for r in filled if r.get("side") == side]
    sa = [r for r in all_records if r.get("side") == side]
    pnls = [p for p in (get_pnl(r) for r in sf) if p is not None]
    if not pnls:
        return None
    pos = [p for p in pnls if p > 0]
    n_as = sum(1 for p in pnls if p < AS_THRESHOLD_BPS)
    n_severe = sum(1 for p in pnls if p < SEVERE_AS_THRESHOLD_BPS)
    return SideMetrics(
        side=side,
        n_total=len(sa),
        n_filled=len(sf),
        fill_rate=len(sf) / max(len(sa), 1),
        avg_pnl_bps=sum(pnls) / len(pnls),
        sum_pnl_bps=sum(pnls),
        win_rate=len(pos) / len(pnls),
        p10_bps=_percentile(pnls, 10),
        p25_bps=_percentile(pnls, 25),
        p50_bps=_percentile(pnls, 50),
        p75_bps=_percentile(pnls, 75),
        p90_bps=_percentile(pnls, 90),
        as_rate=n_as / len(pnls),
        severe_as_rate=n_severe / len(pnls),
    )


# ======================================================================
# Core analysis
# ======================================================================


def run_analysis(sha_prefixes: list[str]) -> AnalysisResult:
    """SHA 絞り込み分析を実行."""
    # Load all records
    all_records = load_fill_record_objects_glob(str(RESULTS_DIR))

    # SHA distribution (for stdout)
    sha_counts: Counter[str] = Counter()
    for r in all_records:
        sha = (str(r.get("git_sha", "")) or "unknown")[:7]
        sha_counts[sha] += 1

    print("=" * 72)
    print("  SHA 分布 (全レコード)")
    print("=" * 72)
    for sha, cnt in sha_counts.most_common(15):
        marker = " <<<" if any(sha == p[:7] for p in sha_prefixes) else ""
        print(f"  {sha}: {cnt} records{marker}")
    print(f"  Total: {len(all_records)}")

    # Filter to target SHAs
    dcc = [r for r in all_records if _is_target(r, sha_prefixes)]

    if not dcc:
        print(f"\n  ERROR: No records found for SHAs {sha_prefixes}!")
        sys.exit(1)

    # Per-SHA breakdown
    print(f"\n{'=' * 72}")
    print(f"  Target SHA(s): {', '.join(sha_prefixes)}")
    print(f"{'=' * 72}")
    for prefix in sha_prefixes:
        subset = [r for r in dcc if (str(r.get("git_sha", "")) or "")[:7] == prefix[:7]]
        sf = [r for r in subset if r.get("filled")]
        ts_list = [r["timestamp"] for r in subset
                   if isinstance(r.get("timestamp"), (int, float))]
        if ts_list:
            t0 = _ts_to_jst(min(ts_list))
            t1 = _ts_to_jst(max(ts_list))
            hours = (max(ts_list) - min(ts_list)) / 3600
            print(f"  {prefix[:7]}: n={len(subset)}, filled={len(sf)}, "
                  f"{t0} ~ {t1} ({hours:.1f}h)")
        else:
            print(f"  {prefix[:7]}: n={len(subset)}, filled={len(sf)}")

    # Time range
    timestamps = [r["timestamp"] for r in dcc
                  if isinstance(r.get("timestamp"), (int, float))]
    period_start = None
    period_end = None
    duration_h = 0.0
    if timestamps:
        period_start = _ts_to_jst(min(timestamps))
        period_end = _ts_to_jst(max(timestamps))
        duration_h = (max(timestamps) - min(timestamps)) / 3600
        print(f"\n  Combined: {period_start} → {period_end} JST ({duration_h:.1f}h)")

    filled = [r for r in dcc if r.get("filled")]
    skipped = [r for r in dcc if not r.get("filled")]

    print(f"  Total: {len(dcc)} | Filled: {len(filled)} "
          f"({100 * len(filled) / max(len(dcc), 1):.1f}%) | Skip: {len(skipped)}")

    # Overall PnL
    all_pnls = [p for p in (get_pnl(r) for r in filled) if p is not None]
    if all_pnls:
        pos = [p for p in all_pnls if p > 0]
        print(f"\n  Overall PnL(bps):")
        print(f"    mean={sum(all_pnls) / len(all_pnls):+.3f} | "
              f"sum={sum(all_pnls):+.2f} | win={100 * len(pos) / len(all_pnls):.1f}%")
        print(f"    p10={_percentile(all_pnls, 10):+.3f} | "
              f"p25={_percentile(all_pnls, 25):+.3f} | "
              f"p50={_percentile(all_pnls, 50):+.3f} | "
              f"p75={_percentile(all_pnls, 75):+.3f} | "
              f"p90={_percentile(all_pnls, 90):+.3f}")

    # Side breakdown
    print(f"\n  Side Breakdown:")
    side_metrics: list[SideMetrics] = []
    for side in ["sell", "buy"]:
        sm = _compute_side_metrics(filled, dcc, side)
        if sm:
            side_metrics.append(sm)
            print(f"    {side:4s}: n={sm.n_filled:3d}/{sm.n_total:3d} "
                  f"(fill {100 * sm.fill_rate:.1f}%), "
                  f"mean={sm.avg_pnl_bps:+.3f}, sum={sm.sum_pnl_bps:+.2f}, "
                  f"win={100 * sm.win_rate:.1f}%, "
                  f"p10={sm.p10_bps:+.3f}, p50={sm.p50_bps:+.3f}")

    # Balance forced
    bf = [r for r in dcc if r.get("balance_forced_switch")]
    print(f"\n  Balance forced: {len(bf)} ({100 * len(bf) / max(len(dcc), 1):.1f}%)")

    # Cancel reasons
    cr_counts: Counter[str] = Counter(
        str(r.get("cancel_reason", "")) for r in skipped
    )
    print(f"\n  Top cancel reasons:")
    for cr, cnt in cr_counts.most_common(10):
        pct = 100 * cnt / len(skipped) if skipped else 0
        print(f"    {cr or '(none)'}: {cnt} ({pct:.1f}%)")

    # ============================================================
    # Regime breakdown
    # ============================================================
    print(f"\n{'=' * 72}")
    print(f"  Regime Analysis")
    print(f"{'=' * 72}")

    regime_data: dict[str, RegimeBuckets] = defaultdict(
        lambda: {"filled": [], "all": []},
    )
    for r in dcc:
        regime = str(r.get("regime", r.get("regime_at_order", "unknown")) or "unknown")
        regime_data[regime]["all"].append(r)
        if r.get("filled"):
            regime_data[regime]["filled"].append(r)

    regime_metrics_list: list[RegimeMetrics] = []
    for regime in sorted(regime_data.keys()):
        regime_bucket = regime_data[regime]
        n_all = len(regime_bucket["all"])
        n_filled = len(regime_bucket["filled"])
        fill_rate = n_filled / n_all if n_all else 0
        pnls_r = [p for p in (get_pnl(r) for r in regime_bucket["filled"]) if p is not None]

        rm = RegimeMetrics(
            regime=regime,
            n_total=n_all,
            n_filled=n_filled,
            fill_rate=fill_rate,
        )

        print(f"\n  [{regime}] records={n_all}, filled={n_filled} ({100 * fill_rate:.1f}%)")
        if pnls_r:
            pw = [p for p in pnls_r if p > 0]
            rm.avg_pnl_bps = sum(pnls_r) / len(pnls_r)
            print(f"    PnL: mean={rm.avg_pnl_bps:+.3f}, "
                  f"sum={sum(pnls_r):+.2f}, "
                  f"p10={_percentile(pnls_r, 10):+.3f}, "
                  f"win={100 * len(pw) / len(pnls_r):.1f}%")

            for side in ["sell", "buy"]:
                sm = _compute_side_metrics(regime_bucket["filled"], regime_bucket["all"], side)
                if sm:
                    rm.by_side.append(sm)
                    print(f"      {side}: n={sm.n_filled}, "
                          f"mean={sm.avg_pnl_bps:+.3f}, "
                          f"p10={sm.p10_bps:+.3f}, "
                          f"win={100 * sm.win_rate:.1f}%")

        regime_metrics_list.append(rm)

    # ============================================================
    # AB Judgment
    # ============================================================
    print(f"\n{'=' * 72}")
    print(f"  AB Judgment (311#/317# criteria)")
    print(f"  Thresholds: fill_rate≥30% | avg_pnl≥-1.00bps | p10≥-5.00bps")
    print(f"{'=' * 72}")

    ab_judgments: list[ABSideJudgment] = []
    overall_pass = True
    for side in ["sell", "buy"]:
        sf = [r for r in filled if r.get("side") == side]
        sa = [r for r in dcc if r.get("side") == side]
        sp = [p for p in (get_pnl(r) for r in sf) if p is not None]
        if sp:
            fr = len(sf) / max(len(sa), 1)
            avg = sum(sp) / len(sp)
            dp10 = _percentile(sp, 10)
            fr_j = "PASS" if fr >= 0.30 else "FAIL"
            avg_j = "PASS" if avg >= -1.0 else "FAIL"
            p10_j = "PASS" if dp10 >= -5.0 else "FAIL"
            any_fail = any(j == "FAIL" for j in [fr_j, avg_j, p10_j])
            if any_fail:
                overall_pass = False
            verdict = ">>> FAIL <<<" if any_fail else "PASS"
            print(f"  {side}: fill={100 * fr:.1f}% [{fr_j}] | "
                  f"avg={avg:+.3f} [{avg_j}] | "
                  f"p10={dp10:+.3f} [{p10_j}]  {verdict}")
            ab_judgments.append(ABSideJudgment(
                side=side,
                fill_rate=fr,
                fill_rate_verdict=fr_j,
                avg_pnl_bps=avg,
                avg_pnl_verdict=avg_j,
                p10_bps=dp10,
                p10_verdict=p10_j,
                overall_verdict=verdict.replace(">>> ", "").replace(" <<<", ""),
            ))
        else:
            overall_pass = False
            print(f"  {side}: NO DATA")

    ab_overall = "PASS" if overall_pass else "FAIL"
    print(f"\n  Overall AB Verdict: {'PASS' if overall_pass else '>>> FAIL <<<'}")

    # ============================================================
    # Adverse Selection
    # ============================================================
    print(f"\n{'=' * 72}")
    print(f"  Adverse Selection Analysis (threshold: {AS_THRESHOLD_BPS} bps)")
    print(f"{'=' * 72}")

    for side in ["sell", "buy"]:
        sf = [r for r in filled if r.get("side") == side]
        sp = [p for p in (get_pnl(r) for r in sf) if p is not None]
        if sp:
            as_count = sum(1 for p in sp if p < AS_THRESHOLD_BPS)
            big_as = sum(1 for p in sp if p < SEVERE_AS_THRESHOLD_BPS)
            print(f"  {side}: AS rate={100 * as_count / len(sp):.1f}% "
                  f"({as_count}/{len(sp)}) | "
                  f"severe(<{SEVERE_AS_THRESHOLD_BPS}bps)="
                  f"{100 * big_as / len(sp):.1f}% ({big_as}/{len(sp)})")

    # ============================================================
    # Time-of-Day
    # ============================================================
    print(f"\n{'=' * 72}")
    print(f"  Time-of-Day (UTC hour)")
    print(f"{'=' * 72}")

    hourly: dict[int, HourlyBuckets] = defaultdict(
        lambda: {"pnls": [], "sell_pnls": [], "buy_pnls": []},
    )
    for r in filled:
        ts = r.get("timestamp")
        if isinstance(ts, (int, float)):
            utc_hour = datetime.fromtimestamp(ts, tz=timezone.utc).hour
            p = get_pnl(r)
            if p is not None:
                hour_bucket = hourly[utc_hour]
                hour_bucket["pnls"].append(p)
                side = r.get("side")
                if side == "sell":
                    hour_bucket["sell_pnls"].append(p)
                elif side == "buy":
                    hour_bucket["buy_pnls"].append(p)

    print(f"  {'UTC':>3s}  {'n':>4s}  {'mean':>8s}  {'p10':>7s}  "
          f"{'win%':>5s}  {'sell_n':>6s}  {'sell_avg':>9s}  "
          f"{'buy_n':>5s}  {'buy_avg':>9s}")
    for h in sorted(hourly.keys()):
        hour_bucket = hourly[h]
        n = len(hour_bucket["pnls"])
        if n == 0:
            continue
        mean_p = sum(hour_bucket["pnls"]) / n
        hp10 = _percentile(hour_bucket["pnls"], 10) if n >= 3 else float("nan")
        wn = sum(1 for p in hour_bucket["pnls"] if p > 0)
        wr = 100 * wn / n
        sn = len(hour_bucket["sell_pnls"])
        sa_v = sum(hour_bucket["sell_pnls"]) / sn if sn else float("nan")
        bn = len(hour_bucket["buy_pnls"])
        ba_v = sum(hour_bucket["buy_pnls"]) / bn if bn else float("nan")
        print(f"  {h:3d}  {n:4d}  {mean_p:+8.3f}  {hp10:+7.3f}  "
              f"{wr:5.1f}  {sn:6d}  {sa_v:+9.3f}  {bn:5d}  {ba_v:+9.3f}")

    # ============================================================
    # Daily Breakdown
    # ============================================================
    print(f"\n{'=' * 72}")
    print(f"  Daily Breakdown (JST)")
    print(f"{'=' * 72}")

    daily_data: dict[str, DailyBuckets] = defaultdict(
        lambda: {"total": 0, "filled": 0, "pnls": [], "bf": 0},
    )
    for r in dcc:
        ts = r.get("timestamp")
        if isinstance(ts, (int, float)):
            day = _ts_to_jst_day(ts)
            daily_data[day]["total"] += 1
            if r.get("filled"):
                daily_data[day]["filled"] += 1
                p = get_pnl(r)
                if p is not None:
                    daily_data[day]["pnls"].append(p)
            if r.get("balance_forced_switch"):
                daily_data[day]["bf"] += 1

    daily_metrics_list: list[DailyMetrics] = []
    print(f"  {'Day':>5s}  {'All':>4s}  {'Fill':>4s}  {'Rate%':>5s}  "
          f"{'mean':>8s}  {'sum':>8s}  {'win%':>5s}  {'p10':>7s}  {'BF%':>4s}")
    for day in sorted(daily_data.keys()):
        dd: DailyBuckets = daily_data[day]
        fr = dd["filled"] / dd["total"] if dd["total"] else 0
        bf_pct = 100 * dd["bf"] / dd["total"] if dd["total"] else 0
        dm = DailyMetrics(
            day=day,
            n_total=dd["total"],
            n_filled=dd["filled"],
            fill_rate=fr,
        )
        if dd["pnls"]:
            dm.avg_pnl_bps = sum(dd["pnls"]) / len(dd["pnls"])
            dm.sum_pnl_bps = sum(dd["pnls"])
            dm.win_rate = sum(1 for p in dd["pnls"] if p > 0) / len(dd["pnls"])
            dm.p10_bps = _percentile(dd["pnls"], 10) if len(dd["pnls"]) >= 3 else None
            print(f"  {day:>5s}  {dd['total']:4d}  {dd['filled']:4d}  "
                  f"{100 * fr:5.1f}  {dm.avg_pnl_bps:+8.3f}  "
                  f"{dm.sum_pnl_bps:+8.2f}  "
                  f"{100 * dm.win_rate:5.1f}  "
                  f"{dm.p10_bps:+7.3f}  " if dm.p10_bps is not None else
                  f"  {day:>5s}  {dd['total']:4d}  {dd['filled']:4d}  "
                  f"{100 * fr:5.1f}  {dm.avg_pnl_bps:+8.3f}  "
                  f"{dm.sum_pnl_bps:+8.2f}  "
                  f"{100 * dm.win_rate:5.1f}       —  "
                  f"{bf_pct:4.1f}")
        else:
            print(f"  {day:>5s}  {dd['total']:4d}  {dd['filled']:4d}  "
                  f"{100 * fr:5.1f}       —        —      —       —  "
                  f"{bf_pct:4.1f}")
        daily_metrics_list.append(dm)

    # ============================================================
    # evaluate_ab_variant (正式 AB 判定)
    # ============================================================
    non_target = [r for r in all_records if not _is_target(r, sha_prefixes)]
    if len(non_target) >= 30 and len(filled) >= 10:
        print(f"\n{'=' * 72}")
        print(f"  Formal AB Judgment (evaluate_ab_variant)")
        print(f"{'=' * 72}")
        criteria = ABJudgmentCriteria(
            min_filled_records=10,  # SHA 単位は小規模
            min_control_filled_records=10,
            min_calendar_days=1,
        )
        result_ab = evaluate_ab_variant(
            variant_records=dcc,
            control_records=non_target,
            criteria=criteria,
            variant_label="_".join(p[:7] for p in sha_prefixes),
            control_label="baseline",
        )
        print(f"  Overall: {result_ab.overall.value}")
        for cr in result_ab.criteria:
            print(f"    {cr.name}: {cr.verdict.value} "
                  f"(val={cr.value:.4f}, thr={cr.threshold:.4f}) "
                  f"— {cr.detail}")
        if result_ab.pnl30_p_value is not None:
            print(f"  Welch t p-value: {result_ab.pnl30_p_value:.6f}")
        if result_ab.mann_whitney_p_value is not None:
            print(f"  Mann-Whitney p: {result_ab.mann_whitney_p_value:.6f}")
        if result_ab.bootstrap_mean_diff is not None:
            print(f"  Bootstrap Δmean: {result_ab.bootstrap_mean_diff:+.4f} "
                  f"[{result_ab.bootstrap_ci_lower:+.4f}, "
                  f"{result_ab.bootstrap_ci_upper:+.4f}]")

    # ============================================================
    # Summary
    # ============================================================
    n_as = sum(1 for p in all_pnls if p < AS_THRESHOLD_BPS) if all_pnls else 0
    n_severe = sum(1 for p in all_pnls if p < SEVERE_AS_THRESHOLD_BPS) if all_pnls else 0
    overall_sm = SideMetrics(
        side="all",
        n_total=len(dcc),
        n_filled=len(filled),
        fill_rate=len(filled) / max(len(dcc), 1),
        avg_pnl_bps=sum(all_pnls) / len(all_pnls) if all_pnls else 0.0,
        sum_pnl_bps=sum(all_pnls) if all_pnls else 0.0,
        win_rate=sum(1 for p in all_pnls if p > 0) / len(all_pnls) if all_pnls else 0.0,
        p10_bps=_percentile(all_pnls, 10),
        p25_bps=_percentile(all_pnls, 25),
        p50_bps=_percentile(all_pnls, 50),
        p75_bps=_percentile(all_pnls, 75),
        p90_bps=_percentile(all_pnls, 90),
        as_rate=n_as / len(all_pnls) if all_pnls else 0.0,
        severe_as_rate=n_severe / len(all_pnls) if all_pnls else 0.0,
    ) if all_pnls else None

    print(f"\n{'=' * 72}")
    print(f"  SUMMARY")
    print(f"{'=' * 72}")
    print(f"  SHAs: {', '.join(sha_prefixes)}")
    print(f"  Sample: {len(filled)} fills / {len(dcc)} records ({duration_h:.1f}h)")
    if len(filled) < 100:
        print(f"  ⚠ n={len(filled)} — 統計的判定には限界あり (目安 n≥100)")
    if all_pnls:
        print(f"  Overall: mean={sum(all_pnls) / len(all_pnls):+.3f}bps, "
              f"sum={sum(all_pnls):+.2f}bps, "
              f"p10={_percentile(all_pnls, 10):+.3f}")
        print(f"  Win rate: {100 * sum(1 for p in all_pnls if p > 0) / len(all_pnls):.1f}%")
    print(f"  AB: {ab_overall}")

    return AnalysisResult(
        shas=sha_prefixes,
        n_records=len(dcc),
        n_filled=len(filled),
        n_skipped=len(skipped),
        fill_rate=len(filled) / max(len(dcc), 1),
        period_start_jst=period_start,
        period_end_jst=period_end,
        duration_hours=duration_h,
        overall=overall_sm,
        by_side=side_metrics,
        by_regime=regime_metrics_list,
        daily=daily_metrics_list,
        ab_judgments=ab_judgments,
        ab_overall=ab_overall,
        cancel_reasons=dict(cr_counts.most_common(20)),
    )


# ======================================================================
# Entry point
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SHA-isolated fill record analysis (333#)",
    )
    parser.add_argument(
        "--sha",
        nargs="+",
        default=DEFAULT_SHAS,
        help="SHA prefix(es) to analyze (default: %(default)s)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON to analysis_results/",
    )
    args = parser.parse_args()

    result = run_analysis(args.sha)

    if args.json:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        sha_tag = "_".join(s[:7] for s in args.sha)
        out_path = OUTPUT_DIR / f"333_sha_isolated_{sha_tag}.json"
        write_json_output(asdict(result), out_path)
        print(f"\n  JSON → {out_path}")


if __name__ == "__main__":
    main()
