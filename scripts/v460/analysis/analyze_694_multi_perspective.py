"""694# Multi-perspective log analysis for AI agent review."""
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

FILL_DIR = Path("results/v460/fill_test")

def load_records(*dates: str) -> list[dict]:
    records = []
    for d in dates:
        p = FILL_DIR / f"fill_records_{d}.jsonl"
        if p.exists():
            with open(p) as f:
                for line in f:
                    records.append(json.loads(line))
    return records

def analyze_period(records: list[dict], label: str) -> dict:
    """Analyze a set of fill records from multiple perspectives."""
    result: dict = {"label": label, "total": len(records)}
    
    filled = [r for r in records if r.get("filled", False)]
    cancelled = [r for r in records if r.get("cancelled", False)]
    result["filled_count"] = len(filled)
    result["cancelled_count"] = len(cancelled)
    result["fill_rate"] = len(filled) / len(records) * 100 if records else 0
    
    # --- Perspective 1: PnL Distribution ---
    pnls = [r["post_fill_30s_pnl"] for r in filled if r.get("post_fill_30s_pnl") is not None]
    if pnls:
        result["pnl"] = {
            "n": len(pnls),
            "mean": round(statistics.mean(pnls), 3),
            "median": round(statistics.median(pnls), 3),
            "stdev": round(statistics.stdev(pnls), 3) if len(pnls) > 1 else 0,
            "p10": round(sorted(pnls)[max(0, len(pnls)//10)], 3),
            "p90": round(sorted(pnls)[min(len(pnls)-1, len(pnls)*9//10)], 3),
            "min": round(min(pnls), 3),
            "max": round(max(pnls), 3),
            "positive_rate": round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1),
        }
        # By side
        for side in ["buy", "sell"]:
            sp = [r["post_fill_30s_pnl"] for r in filled if r.get("side") == side and r.get("post_fill_30s_pnl") is not None]
            if sp:
                result["pnl"][f"{side}_mean"] = round(statistics.mean(sp), 3)
                result["pnl"][f"{side}_n"] = len(sp)
                result["pnl"][f"{side}_p10"] = round(sorted(sp)[max(0, len(sp)//10)], 3)
    
    # --- Perspective 2: Adverse Selection ---
    if filled:
        as_total = sum(1 for r in filled if r.get("adverse_selected_raw", False))
        result["adverse_selection"] = {
            "total_rate": round(as_total / len(filled) * 100, 1),
        }
        for side in ["buy", "sell"]:
            sf = [r for r in filled if r.get("side") == side]
            sa = sum(1 for r in sf if r.get("adverse_selected_raw", False))
            if sf:
                result["adverse_selection"][f"{side}_rate"] = round(sa / len(sf) * 100, 1)
                result["adverse_selection"][f"{side}_n"] = len(sf)
    
    # --- Perspective 3: Cancel Reason Breakdown ---
    reasons = Counter(r.get("cancel_reason", "unknown") for r in cancelled)
    result["cancel_reasons"] = {r: {"count": c, "pct": round(c / len(cancelled) * 100, 1)} for r, c in reasons.most_common()} if cancelled else {}
    
    # --- Perspective 4: Spread Bucket Analysis ---
    spread_buckets = [(0, 1500), (1500, 2500), (2500, 3500), (3500, float("inf"))]
    result["spread_analysis"] = {}
    for lo, hi in spread_buckets:
        label_b = f"{lo}-{hi if hi != float('inf') else '+'}"
        bucket = [r for r in filled if lo <= (r.get("spread_at_order", 0) or 0) < hi]
        if bucket:
            bp = [r["post_fill_30s_pnl"] for r in bucket if r.get("post_fill_30s_pnl") is not None]
            ba = sum(1 for r in bucket if r.get("adverse_selected_raw", False))
            result["spread_analysis"][label_b] = {
                "n": len(bucket),
                "mean_pnl": round(statistics.mean(bp), 3) if bp else None,
                "as_rate": round(ba / len(bucket) * 100, 1),
            }
    
    # --- Perspective 5: UTC Hour Analysis ---
    result["hourly"] = {}
    from datetime import datetime, timezone
    for r in filled:
        ts = r.get("timestamp", 0)
        h = datetime.fromtimestamp(ts, tz=timezone.utc).hour if ts else -1
        if h not in result["hourly"]:
            result["hourly"][h] = {"n": 0, "pnl_sum": 0.0, "as_count": 0}
        result["hourly"][h]["n"] += 1
        result["hourly"][h]["pnl_sum"] += r.get("post_fill_30s_pnl", 0) or 0
        if r.get("adverse_selected_raw", False):
            result["hourly"][h]["as_count"] += 1
    for h, v in result["hourly"].items():
        v["mean_pnl"] = round(v["pnl_sum"] / v["n"], 3) if v["n"] > 0 else 0
        v["as_rate"] = round(v["as_count"] / v["n"] * 100, 1) if v["n"] > 0 else 0
        del v["pnl_sum"]
    
    # --- Perspective 6: Regime Analysis ---
    result["regime"] = {}
    for r in filled:
        reg = r.get("regime", "unknown")
        if reg not in result["regime"]:
            result["regime"][reg] = {"n": 0, "pnl_list": [], "as_count": 0}
        result["regime"][reg]["n"] += 1
        if r.get("post_fill_30s_pnl") is not None:
            result["regime"][reg]["pnl_list"].append(r["post_fill_30s_pnl"])
        if r.get("adverse_selected_raw", False):
            result["regime"][reg]["as_count"] += 1
    for reg, v in result["regime"].items():
        v["mean_pnl"] = round(statistics.mean(v["pnl_list"]), 3) if v["pnl_list"] else 0
        v["as_rate"] = round(v["as_count"] / v["n"] * 100, 1) if v["n"] > 0 else 0
        del v["pnl_list"]
    
    # --- Perspective 7: Entry Gate Guard Status ---
    gate_blocked_reasons = ["entry_gate_ev_negative", "entry_gate_stale", "entry_gate_rate_limit"]
    gate_events = [r for r in cancelled if r.get("cancel_reason", "") in gate_blocked_reasons or r.get("cancel_reason", "").startswith("entry_gate")]
    gate_suppressed = [r for r in filled if r.get("skip_gate_bypassed", False)]
    result["entry_gate"] = {
        "blocked_count": len(gate_events),
        "suppressed_count": len(gate_suppressed),
    }
    
    # --- Perspective 8: Skip Gate Analysis ---
    skip_gate_cancelled = [r for r in cancelled if r.get("cancel_reason") == "skip_gate"]
    sg_scores = [r.get("skip_gate_score", 0) for r in skip_gate_cancelled if r.get("skip_gate_score") is not None]
    result["skip_gate"] = {
        "cancelled_count": len(skip_gate_cancelled),
        "score_mean": round(statistics.mean(sg_scores), 3) if sg_scores else None,
    }

    # --- Perspective 9: sell_hour_boost Effect (4/2 specific) ---
    # UTC2 and UTC4 should have sell_hour_offset_boost=5.0 (was 2.5)
    boost_hours = [2, 4]
    result["sell_hour_boost"] = {}
    for bh in boost_hours:
        h_fills = [r for r in filled if r.get("side") == "sell" and datetime.fromtimestamp(r.get("timestamp", 0), tz=timezone.utc).hour == bh]
        if h_fills:
            h_pnls = [r["post_fill_30s_pnl"] for r in h_fills if r.get("post_fill_30s_pnl") is not None]
            h_as = sum(1 for r in h_fills if r.get("adverse_selected_raw", False))
            result["sell_hour_boost"][f"UTC{bh}"] = {
                "sell_fills": len(h_fills),
                "mean_pnl": round(statistics.mean(h_pnls), 3) if h_pnls else None,
                "as_rate": round(h_as / len(h_fills) * 100, 1),
            }
    
    return result

def main():
    # Load 4/1 (baseline) and 4/2 (sell_hour_boost applied)
    recs_0401 = load_records("20260401")
    recs_0402 = load_records("20260402")
    
    # Also load 3/29-3/31 for longer baseline
    recs_baseline = load_records("20260329", "20260330", "20260331")
    
    output = {
        "analysis_date": "2026-04-02",
        "analysis_type": "694# multi-perspective fill test analysis",
        "description": "sell_hour_offset_boost 2.5→5.0 効果検証 + entry_gate observe分析",
        "note": "4/2 ボット起動 07:54 JST (UTC -2:06). sell_hour_boost=5.0適用済み.",
        "baseline_3day": analyze_period(recs_baseline, "baseline_3/29-3/31"),
        "day_0401": analyze_period(recs_0401, "2026-04-01"),
        "day_0402": analyze_period(recs_0402, "2026-04-02 (boost=5.0)"),
    }
    
    # Add comparison summary
    b = output["day_0401"]
    t = output["day_0402"]
    if b.get("pnl") and t.get("pnl"):
        output["comparison"] = {
            "fill_rate_delta": round(t["fill_rate"] - b["fill_rate"], 1),
            "pnl_mean_delta": round(t["pnl"]["mean"] - b["pnl"]["mean"], 3),
            "as_rate_delta": round(
                t.get("adverse_selection", {}).get("total_rate", 0) - 
                b.get("adverse_selection", {}).get("total_rate", 0), 1
            ),
            "sell_pnl_delta": round(
                (t["pnl"].get("sell_mean", 0) or 0) - (b["pnl"].get("sell_mean", 0) or 0), 3
            ),
        }
    
    out_path = Path("analysis_results/694_multi_perspective_analysis.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    # Print summary
    print(json.dumps(output, indent=2, ensure_ascii=False, default=str))

if __name__ == "__main__":
    main()
