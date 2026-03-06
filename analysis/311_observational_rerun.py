"""311# 観察比較再実行 + 310# 理論検証.

310# で修正した理論 (sell AS time-of-day boost, L2 guardrails, param_adapter path split,
none regime observability, spread/AS decomposition) が正しく機能しているかを検証する。

306# と同一手法で sell vs buy の観察比較を再実行し、改善点を探る。
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.v460.lib.ab_judgment import (
    ABJudgmentCriteria,
    evaluate_ab_variant,
    _block_bootstrap_mean_diff,
    _matched_temporal_comparison,
    _mann_whitney_u,
    _cliffs_delta,
)
from ztb.metrics.fill_quality import (
    iter_fill_record_objects_glob,
    apply_fill_record_filters,
)
from ztb.utils.safety import safe_to_finite


# ======================================================================
# Data Loading
# ======================================================================

def load_records(
    results_dir: str = "results/v460/fill_test",
    *,
    git_sha: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict]:
    """fill record をロード. 314# T0-3: SHA/date フィルタ対応."""
    raw = list(iter_fill_record_objects_glob(
        Path(results_dir), include_emergency=False,
    ))
    if git_sha or date_from or date_to:
        filtered, meta = apply_fill_record_filters(
            raw, git_sha=git_sha, date_from=date_from, date_to=date_to,
        )
        print(f"  [filter] git_sha={git_sha}, date_from={date_from}, "
              f"date_to={date_to} → {len(raw)} → {len(filtered)} records")
        return filtered
    return raw


def extract_filled(records: list[dict], side: str | None = None,
                   regime: str | None = None) -> list[dict]:
    out = []
    for r in records:
        if not r.get("filled"):
            continue
        if side and r.get("side") != side:
            continue
        if regime and str(r.get("regime") or "none") != regime:
            continue
        out.append(r)
    return out


def pnl_array(records: list[dict]) -> np.ndarray:
    vals = []
    for r in records:
        v = safe_to_finite(r.get("post_fill_30s_pnl"))
        if v is not None:
            vals.append(v)
    return np.array(vals, dtype=float) if vals else np.array([], dtype=float)


def as_rate(records: list[dict]) -> float:
    if not records:
        return 0.0
    n_as = sum(1 for r in records if r.get("adverse_selected"))
    return n_as / len(records)


# ======================================================================
# §1: 299# 観察比較再実行 (sell vs buy)
# ======================================================================

def run_ab_comparison(records: list[dict]) -> dict:
    """evaluate_ab_variant で sell vs buy 比較."""
    sell_recs = [r for r in records if r.get("side") == "sell"]
    buy_recs = [r for r in records if r.get("side") == "buy"]

    criteria = ABJudgmentCriteria()

    # none 除外
    result_excl = evaluate_ab_variant(
        sell_recs, buy_recs,
        criteria=criteria,
        variant_label="sell",
        control_label="buy",
    )

    # none 含有
    criteria_incl = ABJudgmentCriteria(exclude_regimes=[])
    result_incl = evaluate_ab_variant(
        sell_recs, buy_recs,
        criteria=criteria_incl,
        variant_label="sell",
        control_label="buy",
    )

    return {
        "ab_judgment": _result_to_dict(result_excl),
        "ab_judgment_incl_none": _result_to_dict(result_incl),
    }


def run_per_regime_comparison(records: list[dict]) -> list[dict]:
    """Regime 別の売買比較."""
    regimes = sorted({str(r.get("regime") or "none") for r in records if r.get("filled")})
    results = []
    for regime in regimes:
        sell_recs = [r for r in records if r.get("side") == "sell"
                     and str(r.get("regime") or "none") == regime]
        buy_recs = [r for r in records if r.get("side") == "buy"
                     and str(r.get("regime") or "none") == regime]
        criteria = ABJudgmentCriteria(exclude_regimes=[])
        result = evaluate_ab_variant(
            sell_recs, buy_recs,
            criteria=criteria,
            variant_label=f"sell[{regime}]",
            control_label=f"buy[{regime}]",
        )
        results.append({
            "regime": regime,
            **_result_to_dict(result),
        })
    return results


def _result_to_dict(result: object) -> dict:
    """ABJudgmentResult を dict 化."""
    d = asdict(result)
    d["overall"] = d["overall"].value if hasattr(d["overall"], "value") else str(d["overall"])
    for c in d.get("criteria", []):
        if hasattr(c.get("verdict"), "value"):
            c["verdict"] = c["verdict"].value
    return d


# ======================================================================
# §2: 310# 新フィールド検証
# ======================================================================

def decision_path_analysis(records: list[dict]) -> dict:
    """310# B: decision_path 分布分析."""
    result = {}
    for side in ["sell", "buy"]:
        filled = extract_filled(records, side=side)
        by_path: dict[str, list[float]] = defaultdict(list)
        for r in filled:
            path = str(r.get("decision_path") or "unknown")
            pnl = safe_to_finite(r.get("post_fill_30s_pnl"))
            if pnl is not None:
                by_path[path].append(pnl)
        path_stats = {}
        for path, pnls in sorted(by_path.items()):
            arr = np.array(pnls)
            path_stats[path] = {
                "n": len(pnls),
                "mean_pnl": round(float(np.mean(arr)), 4),
                "p10": round(float(np.percentile(arr, 10)), 4) if len(pnls) >= 5 else None,
                "as_rate": round(as_rate([r for r in filled if str(r.get("decision_path") or "unknown") == path]), 4),
            }
        result[side] = path_stats
    return result


def sell_hour_boost_analysis(records: list[dict]) -> dict:
    """310# A: sell_hour_offset_boost 効果検証.

    314# T0-4 redesign: 312# F3 指摘に対応.
    - 時間帯別比較: boost 対象 vs 非対象 (構造差の確認)
    - 同一時間帯内 pre/post SHA 比較 (介入効果の測定)
    """
    from datetime import datetime, timezone
    boost_hours = {8, 13, 14, 16}
    # 310# SHA prefix for pre/post split
    post_310_sha = "dcc3064"

    filled_sell = extract_filled(records, side="sell")
    boosted: list[dict] = []
    non_boosted: list[dict] = []
    # 同一時間帯内の pre/post 分離
    boost_hour_pre: list[dict] = []
    boost_hour_post: list[dict] = []

    for r in filled_sell:
        ts = r.get("timestamp")
        if ts is None:
            continue
        try:
            if isinstance(ts, str):
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            elif isinstance(ts, (int, float)):
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            else:
                continue
            hour = dt.hour
        except Exception:
            continue

        sha = str(r.get("git_sha", ""))
        is_post = sha.startswith(post_310_sha)

        if hour in boost_hours:
            boosted.append(r)
            if is_post:
                boost_hour_post.append(r)
            else:
                boost_hour_pre.append(r)
        else:
            non_boosted.append(r)

    def _stats(recs: list[dict]) -> dict:
        arr = pnl_array(recs)
        n = len(arr)
        return {
            "n": len(recs),
            "n_with_pnl": n,
            "mean_pnl": round(float(np.mean(arr)), 4) if n > 0 else None,
            "p10": round(float(np.percentile(arr, 10)), 4) if n >= 5 else None,
            "p50": round(float(np.median(arr)), 4) if n > 0 else None,
            "as_rate": round(as_rate(recs), 4),
            "mean_offset": round(float(np.mean([
                safe_to_finite(r.get("effective_offset_used")) or 0.0
                for r in recs
            ])), 6) if recs else None,
        }

    return {
        "boosted_hours": sorted(boost_hours),
        "boosted": _stats(boosted),
        "non_boosted": _stats(non_boosted),
        # 314# F3: 同一時間帯内の pre/post 310# 比較
        "boost_hour_pre_310": _stats(boost_hour_pre),
        "boost_hour_post_310": _stats(boost_hour_post),
    }


def spread_as_decomposition(records: list[dict]) -> dict:
    """310# E / 314# T0-1: Spread Capture / AS Cost 分解.

    314# 修正: fill_price と mid_at_fill から直接 spread capture を計算.
    旧式 (spread_bps * ratio) は ratio のセマンティクスが
    maker_price.py と fill_cycle_executor.py で異なるため不正確.

    spread_capture_bps = (fill_price - mid_at_fill) / mid_at_fill × 10000  (sell)
                       = (mid_at_fill - fill_price) / mid_at_fill × 10000  (buy)
    as_cost = spread_capture - realized_pnl
    """
    BPS = 10000.0
    result = {}
    for side in ["sell", "buy"]:
        filled = extract_filled(records, side=side)
        sc_list: list[float] = []
        pnl_list: list[float] = []
        as_cost_list: list[float] = []
        for r in filled:
            fill_price = safe_to_finite(r.get("fill_price"))
            mid = safe_to_finite(r.get("mid_at_fill"))
            pnl = safe_to_finite(r.get("post_fill_30s_pnl"))
            if fill_price is None or mid is None or mid <= 0 or pnl is None:
                continue
            # spread capture: fill price の mid からの有利乖離
            if side == "sell":
                sc_bps = (fill_price - mid) / mid * BPS
            else:
                sc_bps = (mid - fill_price) / mid * BPS
            as_cost = sc_bps - pnl
            sc_list.append(sc_bps)
            pnl_list.append(pnl)
            as_cost_list.append(as_cost)
        n = len(sc_list)
        sc_arr = np.array(sc_list) if sc_list else np.array([])
        pnl_arr = np.array(pnl_list) if pnl_list else np.array([])
        as_arr = np.array(as_cost_list) if as_cost_list else np.array([])
        result[side] = {
            "n": n,
            "spread_capture_bps": {
                "mean": round(float(np.mean(sc_arr)), 4) if n > 0 else None,
                "p50": round(float(np.median(sc_arr)), 4) if n > 0 else None,
            },
            "realized_pnl_bps": {
                "mean": round(float(np.mean(pnl_arr)), 4) if n > 0 else None,
                "p50": round(float(np.median(pnl_arr)), 4) if n > 0 else None,
            },
            "as_cost_bps": {
                "mean": round(float(np.mean(as_arr)), 4) if n > 0 else None,
                "p50": round(float(np.median(as_arr)), 4) if n > 0 else None,
                "p90": round(float(np.percentile(as_arr, 90)), 4) if n >= 5 else None,
            },
            "efficiency": round(float(np.mean(pnl_arr) / np.mean(sc_arr)), 4)
            if n > 0 and np.mean(sc_arr) != 0 else None,
        }
    return result


def none_regime_analysis(records: list[dict]) -> dict:
    """310# D: None regime 影響分析."""
    filled = [r for r in records if r.get("filled")]
    none_recs = [r for r in filled if str(r.get("regime") or "none") == "none"]
    non_none = [r for r in filled if str(r.get("regime") or "none") != "none"]

    def _stats(recs: list[dict]) -> dict:
        arr = pnl_array(recs)
        return {
            "n": len(recs),
            "mean_pnl": round(float(np.mean(arr)), 4) if len(arr) > 0 else None,
            "as_rate": round(as_rate(recs), 4),
        }

    # None regime の sell/buy 分解
    none_sell = [r for r in none_recs if r.get("side") == "sell"]
    none_buy = [r for r in none_recs if r.get("side") == "buy"]

    return {
        "total_filled": len(filled),
        "none_count": len(none_recs),
        "none_rate": round(len(none_recs) / len(filled), 4) if filled else 0,
        "none_overall": _stats(none_recs),
        "none_sell": _stats(none_sell),
        "none_buy": _stats(none_buy),
        "non_none_overall": _stats(non_none),
    }


# ======================================================================
# §3: 時間帯別 + offset quintile 深堀り
# ======================================================================

def hourly_pnl_as(records: list[dict]) -> dict:
    """UTC 時間帯別の PnL + AS率 (sell/buy 分離)."""
    from datetime import datetime, timezone
    result = {}
    for side in ["sell", "buy"]:
        filled = extract_filled(records, side=side)
        by_hour: dict[int, list[dict]] = defaultdict(list)
        for r in filled:
            ts = r.get("timestamp")
            if ts is None:
                continue
            try:
                if isinstance(ts, str):
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                elif isinstance(ts, (int, float)):
                    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                else:
                    continue
                by_hour[dt.hour].append(r)
            except Exception:
                continue
        hourly = []
        for h in range(24):
            recs = by_hour.get(h, [])
            arr = pnl_array(recs)
            hourly.append({
                "hour": h,
                "n": len(recs),
                "mean_pnl": round(float(np.mean(arr)), 4) if len(arr) > 0 else None,
                "p10": round(float(np.percentile(arr, 10)), 4) if len(arr) >= 5 else None,
                "as_rate": round(as_rate(recs), 4),
            })
        result[side] = hourly
    return result


def offset_quintile_analysis(records: list[dict]) -> dict:
    """Offset quintile 別の PnL + AS率."""
    result = {}
    for side in ["sell", "buy"]:
        filled = extract_filled(records, side=side)
        offsets_pnls = []
        for r in filled:
            o = safe_to_finite(r.get("effective_offset_used"))
            p = safe_to_finite(r.get("post_fill_30s_pnl"))
            if o is not None and p is not None:
                offsets_pnls.append((o, p, r.get("adverse_selected", False)))

        if len(offsets_pnls) < 10:
            result[side] = {"n": len(offsets_pnls), "quintiles": []}
            continue

        offsets_pnls.sort(key=lambda x: x[0])
        n = len(offsets_pnls)
        quintiles = []
        for qi in range(5):
            start = qi * n // 5
            end = (qi + 1) * n // 5
            chunk = offsets_pnls[start:end]
            offs = [c[0] for c in chunk]
            pnls = np.array([c[1] for c in chunk])
            n_as = sum(1 for c in chunk if c[2])
            quintiles.append({
                "quintile": qi + 1,
                "n": len(chunk),
                "offset_range": [round(min(offs), 6), round(max(offs), 6)],
                "mean_pnl": round(float(np.mean(pnls)), 4),
                "as_rate": round(n_as / len(chunk), 4) if chunk else 0,
            })
        result[side] = {"n": n, "quintiles": quintiles}
    return result


# ======================================================================
# §4: 改善提案導出
# ======================================================================

def derive_improvement_proposals(
    ab_result: dict,
    regime_results: list[dict],
    decomp: dict,
    hourly: dict,
    hour_boost: dict,
    decision_path: dict,
    none_regime: dict,
) -> list[dict]:
    """分析結果から改善提案を導出."""
    proposals = []

    # 1. Downside tail check
    for crit in ab_result.get("ab_judgment", {}).get("criteria", []):
        if crit.get("name") == "downside_p10" and crit.get("verdict") == "fail":
            proposals.append({
                "id": "P1",
                "priority": "P0",
                "title": "Sell downside tail (p10) の改善",
                "detail": f"sell p10 = {crit['value']:.2f} bps (閾値 {crit['threshold']})",
                "source": "AB judgment downside_p10",
            })

    # 2. Regime-specific issues
    for rr in regime_results:
        regime = rr.get("regime", "")
        for crit in rr.get("criteria", []):
            if crit.get("verdict") == "fail":
                proposals.append({
                    "id": f"R-{regime}-{crit['name']}",
                    "priority": "P1",
                    "title": f"{regime} regime: {crit['name']} FAIL",
                    "detail": crit.get("detail", ""),
                    "source": f"per_regime[{regime}]",
                })

    # 3. AS cost > spread capture
    # 314# T0-2: efficiency ベースの P0 判定を廃止 (312# F2).
    # spread_capture 式が修正されたため、P1 としてのみ報告.
    for side in ["sell", "buy"]:
        d = decomp.get(side, {})
        sc_mean = d.get("spread_capture_bps", {}).get("mean")
        as_mean = d.get("as_cost_bps", {}).get("mean")
        if sc_mean is not None and as_mean is not None and as_mean > sc_mean:
            proposals.append({
                "id": f"D-{side}",
                "priority": "P1",
                "title": f"{side} AS cost ({as_mean:.2f}) > spread capture ({sc_mean:.2f})",
                "detail": f"AS exceeds spread capture by {as_mean - sc_mean:.2f} bps",
                "source": "spread_as_decomposition",
            })

    # 4. None regime degradation
    none_pnl = none_regime.get("none_overall", {}).get("mean_pnl")
    nn_pnl = none_regime.get("non_none_overall", {}).get("mean_pnl")
    if none_pnl is not None and nn_pnl is not None and none_pnl < nn_pnl:
        proposals.append({
            "id": "N1",
            "priority": "P1",
            "title": f"None regime PnL 劣後 ({none_pnl:.2f} vs {nn_pnl:.2f} bps)",
            "detail": f"None rate = {none_regime.get('none_rate', 0):.1%}",
            "source": "none_regime_analysis",
        })

    # 5. Hour boost effectiveness
    boosted = hour_boost.get("boosted", {})
    non_boosted = hour_boost.get("non_boosted", {})
    if boosted.get("n", 0) > 0 and non_boosted.get("n", 0) > 0:
        b_pnl = boosted.get("mean_pnl")
        nb_pnl = non_boosted.get("mean_pnl")
        b_as = boosted.get("as_rate", 0)
        nb_as = non_boosted.get("as_rate", 0)
        if b_pnl is not None and nb_pnl is not None:
            if b_pnl < nb_pnl:
                proposals.append({
                    "id": "H1",
                    "priority": "P1",
                    "title": f"Boost時間帯 PnL ({b_pnl:.2f}) < 非Boost ({nb_pnl:.2f}) — AS率 {b_as:.1%} vs {nb_as:.1%}",
                    "detail": "sell_hour_offset_boost がAS改善に寄与しているか要検証",
                    "source": "sell_hour_boost_analysis",
                })

    return proposals


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--git-sha", default=None, help="git SHA prefix filter")
    parser.add_argument("--date-from", default=None, help="date filter (YYYY-MM-DD)")
    parser.add_argument("--date-to", default=None, help="date filter (YYYY-MM-DD)")
    args = parser.parse_args()

    print("=" * 70)
    print("  311# 観察比較再実行 + 310# 理論検証")
    print("  314# T0: spread capture / SHA filter / hour boost redesign 修正済")
    print("=" * 70)

    records = load_records(
        git_sha=args.git_sha,
        date_from=args.date_from,
        date_to=args.date_to,
    )
    filled = [r for r in records if r.get("filled")]
    print(f"\n  Total: {len(records)} records, Filled: {len(filled)}")

    sell_filled = extract_filled(records, side="sell")
    buy_filled = extract_filled(records, side="buy")
    print(f"  Sell filled: {len(sell_filled)}, Buy filled: {len(buy_filled)}")

    # --- §1: AB Comparison ---
    print("\n" + "=" * 70)
    print("  §1 Sell vs Buy 観察比較 (299# 手法再実行)")
    print("=" * 70)
    ab = run_ab_comparison(records)
    for label, key in [("None除外", "ab_judgment"), ("None含有", "ab_judgment_incl_none")]:
        d = ab[key]
        print(f"\n  [{label}] overall={d['overall']}")
        print(f"    sell(n={d['n_variant']}) vs buy(n={d['n_control']})")
        for c in d.get("criteria", []):
            mark = "✅" if c.get("verdict") == "pass" else "❌"
            print(f"    {mark} {c['name']}: {c.get('detail', '')}")
        if d.get("bootstrap_mean_diff") is not None:
            print(f"    Bootstrap: diff={d['bootstrap_mean_diff']:+.4f}, "
                  f"CI=[{d['bootstrap_ci_lower']:+.4f}, {d['bootstrap_ci_upper']:+.4f}], "
                  f"p={d['bootstrap_p_value']:.4f}")
        if d.get("matched_n_pairs") is not None:
            print(f"    Matched(n={d['matched_n_pairs']}): diff={d['matched_mean_diff']:+.4f}, "
                  f"CI=[{d['matched_ci_lower']:+.4f}, {d['matched_ci_upper']:+.4f}], "
                  f"p={d['matched_p_value']:.4f}")

    # --- §2: Per-Regime Comparison ---
    print("\n" + "=" * 70)
    print("  §2 Regime 別比較")
    print("=" * 70)
    per_regime = run_per_regime_comparison(records)
    for rr in per_regime:
        print(f"\n  [{rr['regime']}] overall={rr['overall']}")
        print(f"    sell(n={rr['n_variant']}) vs buy(n={rr['n_control']})")
        for c in rr.get("criteria", []):
            mark = "✅" if c.get("verdict") == "pass" else "❌"
            print(f"    {mark} {c['name']}: {c.get('detail', '')}")

    # --- §3: Spread/AS Decomposition ---
    print("\n" + "=" * 70)
    print("  §3 Spread Capture / AS Cost 分解")
    print("=" * 70)
    decomp = spread_as_decomposition(records)
    for side in ["sell", "buy"]:
        d = decomp[side]
        sc = d["spread_capture_bps"]
        rp = d["realized_pnl_bps"]
        ac = d["as_cost_bps"]
        print(f"\n  [{side.upper()}] n={d['n']}")
        print(f"    spread_capture: mean={sc['mean']} bps")
        print(f"    realized_pnl:   mean={rp['mean']} bps")
        print(f"    AS cost:        mean={ac['mean']} bps, p90={ac.get('p90')} bps")
        print(f"    efficiency:     {d['efficiency']}")

    # --- §4: Decision Path ---
    print("\n" + "=" * 70)
    print("  §4 310# B: Decision Path 分析")
    print("=" * 70)
    dp = decision_path_analysis(records)
    for side in ["sell", "buy"]:
        print(f"\n  [{side.upper()}]")
        for path, stats in dp[side].items():
            print(f"    {path}: n={stats['n']}, pnl={stats['mean_pnl']:+.4f}, AS={stats['as_rate']}")

    # --- §5: Sell Hour Boost ---
    print("\n" + "=" * 70)
    print("  §5 310# A: Sell Hour Boost 効果検証")
    print("=" * 70)
    hb = sell_hour_boost_analysis(records)
    b = hb["boosted"]
    nb = hb["non_boosted"]
    print(f"\n  Boost対象 (UTC {hb['boosted_hours']}): n={b['n']}")
    if b.get("mean_pnl") is not None:
        print(f"    PnL: mean={b['mean_pnl']:+.4f}, p10={b.get('p10')}, AS={b['as_rate']:.1%}")
        print(f"    Offset: mean={b['mean_offset']:.6f}")
    print(f"\n  非Boost: n={nb['n']}")
    if nb.get("mean_pnl") is not None:
        print(f"    PnL: mean={nb['mean_pnl']:+.4f}, p10={nb.get('p10')}, AS={nb['as_rate']:.1%}")
        print(f"    Offset: mean={nb['mean_offset']:.6f}")

    # 314# F3: 同一時間帯内の pre/post 310# 比較
    pre_h = hb.get("boost_hour_pre_310", {})
    post_h = hb.get("boost_hour_post_310", {})
    print(f"\n  [314# F3] Boost時間帯 pre-310# (n={pre_h.get('n', 0)}):")
    if pre_h.get("mean_pnl") is not None:
        print(f"    PnL: mean={pre_h['mean_pnl']:+.4f}, AS={pre_h['as_rate']:.1%}")
    print(f"  [314# F3] Boost時間帯 post-310# (n={post_h.get('n', 0)}):")
    if post_h.get("mean_pnl") is not None:
        print(f"    PnL: mean={post_h['mean_pnl']:+.4f}, AS={post_h['as_rate']:.1%}")
    else:
        print(f"    データ不足 — post-310# 蓄積待ち")

    # --- §6: None Regime ---
    print("\n" + "=" * 70)
    print("  §6 310# D: None Regime 分析")
    print("=" * 70)
    nr = none_regime_analysis(records)
    print(f"  Total: {nr['total_filled']}, None: {nr['none_count']} ({nr['none_rate']:.1%})")
    no = nr["none_overall"]
    print(f"  None overall: pnl={no['mean_pnl']}, AS={no['as_rate']:.1%}")
    ns = nr["none_sell"]
    nb_ = nr["none_buy"]
    print(f"  None sell: n={ns['n']}, pnl={ns['mean_pnl']}, AS={ns['as_rate']:.1%}")
    print(f"  None buy:  n={nb_['n']}, pnl={nb_['mean_pnl']}, AS={nb_['as_rate']:.1%}")
    nno = nr["non_none_overall"]
    print(f"  Non-none: n={nno['n']}, pnl={nno['mean_pnl']}, AS={nno['as_rate']:.1%}")

    # --- §7: Hourly PnL ---
    print("\n" + "=" * 70)
    print("  §7 UTC 時間帯別 PnL + AS率")
    print("=" * 70)
    hourly = hourly_pnl_as(records)
    for side in ["sell", "buy"]:
        print(f"\n  [{side.upper()}]")
        for h in hourly[side]:
            avg = f"{h['mean_pnl']:+.4f}" if h['mean_pnl'] is not None else "N/A"
            p10 = f"{h['p10']:+.4f}" if h.get('p10') is not None else "N/A"
            print(f"    {h['hour']:02d}h: n={h['n']:3d}, pnl={avg}, p10={p10}, AS={h['as_rate']:.1%}")

    # --- §8: Offset Quintile ---
    print("\n" + "=" * 70)
    print("  §8 Offset Quintile 分析")
    print("=" * 70)
    oq = offset_quintile_analysis(records)
    for side in ["sell", "buy"]:
        d = oq[side]
        print(f"\n  [{side.upper()}] n={d['n']}")
        for q in d.get("quintiles", []):
            print(f"    Q{q['quintile']}(n={q['n']}): offset=[{q['offset_range'][0]:.4f}, "
                  f"{q['offset_range'][1]:.4f}], pnl={q['mean_pnl']:+.4f}, AS={q['as_rate']:.1%}")

    # --- §9: 306# との比較 ---
    print("\n" + "=" * 70)
    print("  §9 306# 結果との比較")
    print("=" * 70)
    prev_path = Path("analysis_results/306_observational_comparison_rerun.json")
    if prev_path.exists():
        with open(prev_path, encoding="utf-8-sig") as f:
            prev = json.load(f)
        prev_ab = prev.get("ab_judgment", {})
        curr_ab = ab["ab_judgment"]
        print(f"\n  306# sell: n={prev_ab.get('n_variant')}, pnl_diff={prev_ab.get('bootstrap_mean_diff', 'N/A')}")
        print(f"    Bootstrap p={prev_ab.get('bootstrap_p_value', 'N/A')}")
        print(f"  311# sell: n={curr_ab.get('n_variant')}, pnl_diff={curr_ab.get('bootstrap_mean_diff', 'N/A')}")
        print(f"    Bootstrap p={curr_ab.get('bootstrap_p_value', 'N/A')}")

        # Downside tail 比較
        for label, d in [("306#", prev_ab), ("311#", curr_ab)]:
            for c in d.get("criteria", []):
                if c.get("name") == "downside_p10":
                    print(f"  {label} sell p10: {c['value']}")
    else:
        print("  306# 結果ファイルが見つかりません")

    # --- §10: 改善提案 ---
    print("\n" + "=" * 70)
    print("  §10 改善提案")
    print("=" * 70)
    proposals = derive_improvement_proposals(
        ab, per_regime, decomp, hourly, hb, dp, nr,
    )
    for i, p in enumerate(proposals, 1):
        print(f"\n  [{p['priority']}] {p['id']}: {p['title']}")
        print(f"    {p['detail']}")
        print(f"    Source: {p['source']}")

    if not proposals:
        print("  改善提案なし — 全指標合格")

    # --- JSON 出力 ---
    full_result = {
        "ab_judgment": ab["ab_judgment"],
        "ab_judgment_incl_none": ab["ab_judgment_incl_none"],
        "per_regime_judgment": per_regime,
        "spread_as_decomposition": decomp,
        "decision_path": dp,
        "sell_hour_boost": hb,
        "none_regime": nr,
        "hourly_pnl_as": hourly,
        "offset_quintiles": oq,
        "improvement_proposals": proposals,
    }
    out_path = Path("analysis_results/311_observational_rerun.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  JSON: {out_path}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
