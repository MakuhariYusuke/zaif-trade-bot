"""462# Fill Test Schema-Aware Analysis v3.

Fixes from 462# review:
- Schema drift: only compare fields that exist in both populations
- run_id based grouping (not just git_sha)
- balance_switch uses balance_forced_switch OR resolved_side_reason
- Population-aware: raw/processed/filled EV bins
- ceiling_rate only computed on records with execution_pre_clamp_offset present
"""
import json
import collections
import statistics
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

BASE = Path("results/v460/fill_test")

# === Schema presence map ===
LATE_FIELDS = {
    "execution_pre_clamp_offset": "421#",
    "cross_venue_lead_lag_applied": "443#",
    "resolved_side_reason": "420#",
    "start_git_sha": "420#",
}


def load_records(date_str: str) -> list[dict]:
    fp = BASE / f"fill_records_{date_str}.jsonl"
    if not fp.exists():
        return []
    records = []
    for line in open(fp):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def has_field(records: list[dict], field: str) -> bool:
    """Check if ANY record in the set has this field (not just None)."""
    return any(field in r for r in records[:50])


def is_filled(r: dict) -> bool:
    return r.get("filled", False)


def get_pnl(r: dict, horizon: int = 30) -> Optional[float]:
    key = f"post_fill_{horizon}s_pnl"
    return r.get(key)


def get_sha(r: dict) -> str:
    return (r.get("git_sha") or "?")[:7]


def get_run_id(r: dict) -> str:
    return r.get("run_id") or "unknown"


def is_balance_switch(r: dict) -> bool:
    """Schema-aware balance switch detection."""
    if r.get("balance_forced_switch") is True:
        return True
    if r.get("resolved_side_reason") == "balance_switch":
        return True
    return False


def is_balance_switch_determinable(r: dict) -> bool:
    """Can we determine balance switch status for this record?"""
    return "balance_forced_switch" in r or "resolved_side_reason" in r


def ceiling_clamped(r: dict) -> Optional[bool]:
    """Returns True/False if clamp is determinable, None otherwise."""
    pre = r.get("execution_pre_clamp_offset")
    post = r.get("effective_offset_used")
    if pre is None or post is None:
        return None
    return pre > post + 0.001


def pnl_summary(records: list[dict], label: str = "") -> dict:
    filled = [r for r in records if is_filled(r)]
    if not filled:
        return {"n": 0}
    pnl30 = [v for r in filled if (v := get_pnl(r, 30)) is not None]
    as_cnt = sum(1 for r in filled if r.get("adverse_selected"))
    result = {
        "n": len(filled),
        "pnl_30": round(statistics.mean(pnl30), 2) if pnl30 else 0,
        "pnl_30_med": round(statistics.median(pnl30), 2) if pnl30 else 0,
        "as%": round(as_cnt / len(filled) * 100, 1),
        "win%": round(sum(1 for v in pnl30 if v > 0) / len(pnl30) * 100, 1) if pnl30 else 0,
    }
    if pnl30 and len(pnl30) >= 10:
        s = sorted(pnl30)
        n_tail = max(1, len(s) // 10)
        result["tail10"] = round(statistics.mean(s[:n_tail]), 2)
    return result


def run_analysis() -> None:
    dates = [f"2026030{d}" if d < 10 else f"202603{d}" for d in range(8, 18)]
    all_data: dict[str, list[dict]] = {}
    for date in dates:
        recs = load_records(date)
        if recs:
            all_data[date] = recs

    all_records: list[dict] = []
    for recs in all_data.values():
        all_records.extend(recs)

    print("=" * 80)
    print("PART 1: Schema Presence Audit")
    print("=" * 80)
    for date in sorted(all_data):
        recs = all_data[date]
        presence = {}
        for field in LATE_FIELDS:
            present = has_field(recs, field)
            presence[field] = "YES" if present else "NO"
        print(f"  {date}: {presence}")

    # ==========================================
    print()
    print("=" * 80)
    print("PART 2: Run-based Analysis (run_id + git_sha mapping)")
    print("=" * 80)

    # Group by run_id
    runs: dict[str, list[dict]] = collections.defaultdict(list)
    for r in all_records:
        runs[get_run_id(r)].append(r)

    # Top runs by fill count
    run_shas: dict[str, set] = {}
    run_meta: dict[str, dict] = {}
    for rid, recs in sorted(runs.items()):
        shas = set(get_sha(r) for r in recs)
        run_shas[rid] = shas
        filled = [r for r in recs if is_filled(r)]
        ts_list = [r.get("timestamp", 0) for r in recs if r.get("timestamp")]
        run_meta[rid] = {
            "n": len(recs),
            "fills": len(filled),
            "shas": sorted(shas),
            "t0": min(ts_list) if ts_list else 0,
            "t1": max(ts_list) if ts_list else 0,
        }

    print(f"\n  Total runs: {len(runs)}")
    # Sort by fill count descending
    top_runs = sorted(run_meta.items(), key=lambda x: -x[1]["fills"])[:15]
    for rid, meta in top_runs:
        t0 = datetime.utcfromtimestamp(meta["t0"]).strftime("%m/%d %H:%M") if meta["t0"] else "?"
        t1 = datetime.utcfromtimestamp(meta["t1"]).strftime("%m/%d %H:%M") if meta["t1"] else "?"
        fr = round(meta["fills"] / meta["n"] * 100, 1) if meta["n"] else 0
        shas_str = ",".join(meta["shas"])
        print(f"  {rid}: n={meta['n']} fills={meta['fills']}({fr}%) SHAs=[{shas_str}] {t0}-{t1}")

    # ==========================================
    print()
    print("=" * 80)
    print("PART 3: Schema-Corrected Per-SHA Analysis (top 8)")
    print("=" * 80)

    sha_recs: dict[str, list[dict]] = collections.defaultdict(list)
    for r in all_records:
        sha_recs[get_sha(r)].append(r)

    top_shas = sorted(sha_recs.items(), key=lambda x: -len(x[1]))[:8]

    for sha, recs in top_shas:
        filled = [r for r in recs if is_filled(r)]
        cancelled = [r for r in recs if r.get("cancelled")]
        n_total = len(recs)
        n_fill = len(filled)

        # Schema awareness flags
        has_pre_clamp = has_field(recs, "execution_pre_clamp_offset")
        has_cv = has_field(recs, "cross_venue_lead_lag_applied")
        has_resolved = has_field(recs, "resolved_side_reason")

        print(f"\n--- SHA={sha} (n={n_total}, fill={n_fill}({round(n_fill/n_total*100,1)}%)) ---")
        print(f"  Schema: pre_clamp={'Y' if has_pre_clamp else 'N'}, cv={'Y' if has_cv else 'N'}, resolved={'Y' if has_resolved else 'N'}")

        # PnL summary
        all_stats = pnl_summary(recs)
        buy_stats = pnl_summary([r for r in recs if r.get("side") == "buy"])
        sell_stats = pnl_summary([r for r in recs if r.get("side") == "sell"])
        print(f"  [ALL] n={all_stats['n']}, pnl_30={all_stats['pnl_30']}bps, AS={all_stats['as%']}%, win={all_stats['win%']}%")
        print(f"  [buy] n={buy_stats.get('n',0)}, pnl_30={buy_stats.get('pnl_30','-')}bps")
        print(f"  [sell] n={sell_stats.get('n',0)}, pnl_30={sell_stats.get('pnl_30','-')}bps")

        # Offset & ceiling - SCHEMA AWARE
        for side in ["buy", "sell"]:
            side_filled = [r for r in filled if r.get("side") == side]
            offsets = [r.get("effective_offset_used") for r in side_filled if r.get("effective_offset_used") is not None]
            if not offsets:
                continue
            off_mean = round(statistics.mean(offsets), 3)

            if has_pre_clamp:
                clamp_determinable = [r for r in side_filled if r.get("execution_pre_clamp_offset") is not None]
                clamped = sum(1 for r in clamp_determinable if ceiling_clamped(r))
                pre_clamps = [r["execution_pre_clamp_offset"] for r in clamp_determinable]
                pre_mean = round(statistics.mean(pre_clamps), 3) if pre_clamps else "N/A"
                print(f"  {side}: offset={off_mean}, pre_clamp={pre_mean}, ceiling={clamped}/{len(clamp_determinable)} ({round(clamped/max(len(clamp_determinable),1)*100,1)}%)")
            else:
                print(f"  {side}: offset={off_mean}, ceiling=N/A (pre_clamp field absent)")

        # Balance switch - SCHEMA AWARE
        bal_determinable = [r for r in filled if is_balance_switch_determinable(r)]
        if bal_determinable:
            bal_on = [r for r in bal_determinable if is_balance_switch(r)]
            bal_off = [r for r in bal_determinable if not is_balance_switch(r)]
            on_stats = pnl_summary(bal_on) if bal_on else {"n": 0}
            off_stats = pnl_summary(bal_off) if bal_off else {"n": 0}
            method = "balance_forced_switch" if not has_resolved else "resolved_side_reason+balance_forced_switch"
            print(f"  BalSwitch (method={method}): on={on_stats.get('n', 0)}, pnl={on_stats.get('pnl_30', '-')}, AS={on_stats.get('as%', '-')}% | off={off_stats.get('n', 0)}, pnl={off_stats.get('pnl_30', '-')}, AS={off_stats.get('as%', '-')}%")
        else:
            print(f"  BalSwitch: N/A (no determinable records)")

        # Cross-venue - SCHEMA AWARE
        if has_cv:
            cv_on = [r for r in filled if r.get("cross_venue_lead_lag_applied") is True]
            cv_off = [r for r in filled if r.get("cross_venue_lead_lag_applied") is False]
            cv_on_stats = pnl_summary(cv_on) if cv_on else {"n": 0}
            cv_off_stats = pnl_summary(cv_off) if cv_off else {"n": 0}
            print(f"  CrossVenue: on={cv_on_stats.get('n',0)}, pnl={cv_on_stats.get('pnl_30','-')}, AS={cv_on_stats.get('as%','-')}% | off={cv_off_stats.get('n',0)}, pnl={cv_off_stats.get('pnl_30','-')}, AS={cv_off_stats.get('as%','-')}%")
        else:
            print(f"  CrossVenue: N/A (field absent)")

        # Cancel reasons
        reasons = collections.Counter()
        for r in cancelled:
            reasons[r.get("cancel_reason") or "unknown"] += 1
        top3 = reasons.most_common(5)
        print(f"  Cancel: {dict(top3)}")

    # ==========================================
    print()
    print("=" * 80)
    print("PART 4: EV Score Full-Population Analysis (f840d0e)")
    print("=" * 80)

    f840 = sha_recs.get("f840d0e", [])
    ev_bins_all: dict[str, dict] = {
        "<0.5": {"filled": [], "cancel": []},
        "0.5-1.0": {"filled": [], "cancel": []},
        "1.0-2.0": {"filled": [], "cancel": []},
        ">2.0": {"filled": [], "cancel": []},
        "None": {"filled": [], "cancel": []},
    }
    for r in f840:
        ev = r.get("ev_score_pretrade")
        if ev is None:
            b = "None"
        elif ev < 0.5:
            b = "<0.5"
        elif ev < 1.0:
            b = "0.5-1.0"
        elif ev < 2.0:
            b = "1.0-2.0"
        else:
            b = ">2.0"
        if is_filled(r):
            ev_bins_all[b]["filled"].append(r)
        else:
            ev_bins_all[b]["cancel"].append(r)

    print(f"  f840d0e total: {len(f840)}")
    print(f"  {'bin':10s} {'filled':>6s} {'cancel':>6s} {'total':>6s} {'fill_pnl':>10s} {'fill_AS':>8s}")
    for b in ["<0.5", "0.5-1.0", "1.0-2.0", ">2.0", "None"]:
        fi = ev_bins_all[b]["filled"]
        ca = ev_bins_all[b]["cancel"]
        pnl_str = "-"
        as_str = "-"
        if fi:
            pnls = [v for r in fi if (v := get_pnl(r, 30)) is not None]
            pnl_str = f"{statistics.mean(pnls):.2f}" if pnls else "-"
            as_str = f"{sum(1 for r in fi if r.get('adverse_selected'))/len(fi)*100:.1f}%"
        print(f"  {b:10s} {len(fi):6d} {len(ca):6d} {len(fi)+len(ca):6d} {pnl_str:>10s} {as_str:>8s}")

    # Also check OTHER SHAs' EV distributions
    print()
    print("  EV 0.5-1.0 presence across all SHAs (filled+cancel):")
    for sha, recs in top_shas:
        mid_ev = sum(1 for r in recs if r.get("ev_score_pretrade") is not None and 0.5 <= r["ev_score_pretrade"] < 1.0)
        ev_total = sum(1 for r in recs if r.get("ev_score_pretrade") is not None)
        print(f"    {sha}: ev_0.5-1.0={mid_ev}/{ev_total} ({round(mid_ev/max(ev_total,1)*100,1)}%)")

    # ==========================================
    print()
    print("=" * 80)
    print("PART 5: Schema-Corrected Balance Switch Comparison")
    print("=" * 80)

    print("  Using balance_forced_switch (available in ALL records)")
    for sha, recs in top_shas:
        filled = [r for r in recs if is_filled(r)]
        if not filled:
            continue
        # Use balance_forced_switch only (available everywhere)
        bal_true = [r for r in filled if r.get("balance_forced_switch") is True]
        bal_none = [r for r in filled if r.get("balance_forced_switch") is None]
        bal_false = [r for r in filled if r.get("balance_forced_switch") is False]

        on_pnl = pnl_summary(bal_true) if bal_true else {"n": 0}
        off_pnl = pnl_summary(bal_none + bal_false) if (bal_none + bal_false) else {"n": 0}

        print(f"  {sha}: fills={len(filled)}, bal_T={len(bal_true)}, bal_N={len(bal_none)}, bal_F={len(bal_false)}")
        if on_pnl["n"]:
            print(f"    switch_on: n={on_pnl['n']}, pnl={on_pnl['pnl_30']}, AS={on_pnl['as%']}%")
        if off_pnl["n"]:
            print(f"    switch_off: n={off_pnl['n']}, pnl={off_pnl['pnl_30']}, AS={off_pnl['as%']}%")

    # ==========================================
    print()
    print("=" * 80)
    print("PART 6: Cross-Venue Selection Bias Check (f840d0e)")
    print("=" * 80)

    f840_filled = [r for r in f840 if is_filled(r)]
    if f840_filled:
        # By side
        for side in ["buy", "sell"]:
            side_recs = [r for r in f840_filled if r.get("side") == side]
            cv_on = [r for r in side_recs if r.get("cross_venue_lead_lag_applied") is True]
            cv_off = [r for r in side_recs if r.get("cross_venue_lead_lag_applied") is not True]
            on_stats = pnl_summary(cv_on) if cv_on else {"n": 0}
            off_stats = pnl_summary(cv_off) if cv_off else {"n": 0}
            print(f"  {side}: cv_on n={on_stats.get('n',0)}, pnl={on_stats.get('pnl_30','-')}, AS={on_stats.get('as%','-')}% | cv_off n={off_stats.get('n',0)}, pnl={off_stats.get('pnl_30','-')}, AS={off_stats.get('as%','-')}%")

        # By regime
        for regime in ["ranging", "trending_up", "trending_down"]:
            rg_recs = [r for r in f840_filled if r.get("regime") == regime]
            if not rg_recs:
                continue
            cv_on = [r for r in rg_recs if r.get("cross_venue_lead_lag_applied") is True]
            cv_off = [r for r in rg_recs if r.get("cross_venue_lead_lag_applied") is not True]
            on_stats = pnl_summary(cv_on) if cv_on else {"n": 0}
            off_stats = pnl_summary(cv_off) if cv_off else {"n": 0}
            print(f"  {regime}: cv_on n={on_stats.get('n',0)}, pnl={on_stats.get('pnl_30','-')}, AS={on_stats.get('as%','-')}% | cv_off n={off_stats.get('n',0)}, pnl={off_stats.get('pnl_30','-')}, AS={off_stats.get('as%','-')}%")

    # ==========================================
    print()
    print("=" * 80)
    print("PART 7: sell_dynamic_kill Timeline (SHA-correlated)")
    print("=" * 80)

    for date in sorted(all_data):
        recs = all_data[date]
        cancelled = [r for r in recs if r.get("cancelled")]
        sdk_by_sha: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
        for r in cancelled:
            sha = get_sha(r)
            sdk_by_sha[sha][1] += 1
            if r.get("cancel_reason") == "sell_dynamic_kill":
                sdk_by_sha[sha][0] += 1
        vals = []
        for sha, (sdk, total) in sorted(sdk_by_sha.items(), key=lambda x: -x[1][1]):
            if total >= 10:
                vals.append(f"{sha}:{sdk}/{total}({round(sdk/total*100)}%)")
        print(f"  {date}: {', '.join(vals[:5])}")


if __name__ == "__main__":
    run_analysis()
