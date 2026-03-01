"""204# Part 2: 勝ちトレード深掘り + 特徴量相関分析."""

import json
import glob
import os
from collections import defaultdict
from datetime import datetime, timezone


def load_filled():
    files = sorted(glob.glob("results/v460/fill_test/fill_records_*.jsonl"))
    all_recs = []
    for f in files:
        date_str = os.path.basename(f).replace("fill_records_", "").replace(".jsonl", "")
        for line in open(f):
            r = json.loads(line)
            r["_file_date"] = date_str
            all_recs.append(r)
    return [r for r in all_recs if r.get("filled")]


def winning_trade_deep_dive(filled):
    big_winners = sorted(
        [r for r in filled if (r.get("post_fill_30s_pnl") or 0) > 5],
        key=lambda r: -(r.get("post_fill_30s_pnl") or 0),
    )
    print(f"=== WHY TRADES WIN: Big Winners (>5bps) = {len(big_winners)} ===")

    # VG/non-VG
    vg_wins = [r for r in big_winners if r.get("vg_triggered")]
    novg_wins = [r for r in big_winners if not r.get("vg_triggered")]
    print(f"  VG triggered: {len(vg_wins)} ({100*len(vg_wins)/max(1,len(big_winners)):.0f}%)")
    print(f"  No VG:        {len(novg_wins)} ({100*len(novg_wins)/max(1,len(big_winners)):.0f}%)")

    # Side
    for side in ["buy", "sell"]:
        sw = [r for r in big_winners if r["side"] == side]
        if sw:
            avg = sum(r["post_fill_30s_pnl"] for r in sw) / len(sw)
            print(f"  {side}: {len(sw)} wins, avg={avg:+.1f}bps")

    # Spread
    sps = [r.get("spread_bps") for r in big_winners if r.get("spread_bps") is not None]
    if sps:
        sps.sort()
        print(f"  Spread: avg={sum(sps)/len(sps):.2f}, med={sps[len(sps)//2]:.2f}")

    # 120s persistence
    both = [r for r in big_winners if r.get("post_fill_120s_pnl") is not None]
    if both:
        still_pos = sum(1 for r in both if r["post_fill_120s_pnl"] > 0)
        avg_120 = sum(r["post_fill_120s_pnl"] for r in both) / len(both)
        print(f"  120s retention: {still_pos}/{len(both)} ({100*still_pos/len(both):.0f}%) still positive")
        print(f"  120s avg PnL: {avg_120:+.1f}bps")

    # Top 10 winners detail
    print("\n  TOP 10 WINNERS:")
    for r in big_winners[:10]:
        t = datetime.fromtimestamp(r["timestamp"], tz=timezone.utc)
        p30 = r["post_fill_30s_pnl"]
        p120 = r.get("post_fill_120s_pnl")
        sp = r.get("spread_bps")
        vg = "VG" if r.get("vg_triggered") else "  "
        p120s = f"{p120:+.1f}" if p120 is not None else " n/a"
        sps = f"{sp:.1f}" if sp is not None else "n/a"
        print(f"    {t.strftime('%Y-%m-%d %H:%M')} {r['side']:4s} 30s={p30:+6.1f} "
              f"120s={p120s:>6s} sp={sps:>5s} {vg} d={r['_file_date']}")


def hour_ranking(filled):
    print("\n=== HOUR NET PnL RANKING (JST) ===")
    hourly = defaultdict(lambda: {"pnl": 0.0, "n": 0, "wins": 0})
    for r in filled:
        jst_h = (datetime.fromtimestamp(r["timestamp"], tz=timezone.utc).hour + 9) % 24
        pnl = r.get("post_fill_30s_pnl", 0) or 0
        hourly[jst_h]["pnl"] += pnl
        hourly[jst_h]["n"] += 1
        if pnl > 0:
            hourly[jst_h]["wins"] += 1

    ranked = sorted(hourly.items(), key=lambda x: x[1]["pnl"])
    print(" Worst:")
    for h, v in ranked[:5]:
        wr = 100 * v["wins"] / v["n"]
        avg = v["pnl"] / v["n"]
        print(f"  JST {h:02d}:00  net={v['pnl']:+7.1f}  n={v['n']:3d}  avg={avg:+5.2f}  WR={wr:.0f}%")
    print(" Best:")
    for h, v in ranked[-5:]:
        wr = 100 * v["wins"] / v["n"]
        avg = v["pnl"] / v["n"]
        print(f"  JST {h:02d}:00  net={v['pnl']:+7.1f}  n={v['n']:3d}  avg={avg:+5.2f}  WR={wr:.0f}%")


def daily_analysis(filled):
    print("\n=== DAILY WIN/LOSS PATTERN ===")
    daily = defaultdict(lambda: {"pnl": 0.0, "n": 0, "buy_pnl": 0.0, "sell_pnl": 0.0})
    for r in filled:
        d = r["_file_date"]
        pnl = r.get("post_fill_30s_pnl", 0) or 0
        daily[d]["pnl"] += pnl
        daily[d]["n"] += 1
        daily[d][r["side"] + "_pnl"] += pnl

    win_days = [(d, v) for d, v in sorted(daily.items()) if v["pnl"] > 0]
    lose_days = [(d, v) for d, v in sorted(daily.items()) if v["pnl"] <= 0]
    print(f"Win days: {len(win_days)}, Lose days: {len(lose_days)}")
    
    if win_days:
        print("\n  Winning Days:")
        for d, v in win_days:
            print(f"    {d}: {v['pnl']:+7.1f}bps  n={v['n']:3d}  "
                  f"buy={v['buy_pnl']:+7.1f}  sell={v['sell_pnl']:+7.1f}")
    
    if lose_days:
        print("\n  Losing Days:")
        for d, v in lose_days:
            print(f"    {d}: {v['pnl']:+7.1f}bps  n={v['n']:3d}  "
                  f"buy={v['buy_pnl']:+7.1f}  sell={v['sell_pnl']:+7.1f}")


def spread_pnl_correlation(filled):
    """Spread幅とPnLの相関: MM理論の核心."""
    print("\n=== SPREAD-PnL CORRELATION (MM Core) ===")
    pairs = [(r.get("spread_bps", 0), r.get("post_fill_30s_pnl", 0) or 0)
             for r in filled if r.get("spread_bps") is not None]
    if not pairs:
        return
    
    sps, pnls = zip(*pairs)
    n = len(pairs)
    avg_sp = sum(sps) / n
    avg_pnl = sum(pnls) / n
    cov = sum((s - avg_sp) * (p - avg_pnl) for s, p in pairs) / n
    var_sp = sum((s - avg_sp) ** 2 for s in sps) / n
    var_pnl = sum((p - avg_pnl) ** 2 for p in pnls) / n
    corr = cov / max(1e-10, (var_sp * var_pnl) ** 0.5)
    print(f"  Pearson r(spread, pnl30) = {corr:.4f}")
    print(f"  avg_spread={avg_sp:.2f}bps, avg_pnl={avg_pnl:+.2f}bps")
    
    # スプレッド半値 vs PnL
    half_sp = avg_sp / 2
    print(f"  Half-spread (theoretical edge) = {half_sp:.2f}bps")
    print(f"  Actual avg PnL = {avg_pnl:+.2f}bps → edge capture ratio = {avg_pnl/max(0.01,half_sp)*100:.0f}%")
    
    # 「spread > 3bps の時だけトレードした場合」のシミュレーション
    wide_sp = [(s, p) for s, p in pairs if s > 3.0]
    if wide_sp:
        ws, wp = zip(*wide_sp)
        print(f"  If only spread>3bps: n={len(wide_sp)}, avg_pnl={sum(wp)/len(wp):+.2f}, "
              f"sum={sum(wp):+.1f}, WR={100*sum(1 for p in wp if p>0)/len(wp):.0f}%")
    narrow = [(s, p) for s, p in pairs if s <= 2.0]
    if narrow:
        ns, np_ = zip(*narrow)
        print(f"  If only spread<=2bps: n={len(narrow)}, avg_pnl={sum(np_)/len(np_):+.2f}, "
              f"sum={sum(np_):+.1f}, WR={100*sum(1 for p in np_ if p>0)/len(np_):.0f}%")


def big_loss_pattern(filled):
    """大損パターンの特徴抽出."""
    print("\n=== BIG LOSS PATTERN ANALYSIS (>10bps) ===")
    big_losses = [r for r in filled if (r.get("post_fill_30s_pnl") or 0) < -10]
    print(f"Total big losses: {len(big_losses)}")
    
    if not big_losses:
        return
    
    # Side
    buy_bl = [r for r in big_losses if r["side"] == "buy"]
    sell_bl = [r for r in big_losses if r["side"] == "sell"]
    print(f"  Buy: {len(buy_bl)}, Sell: {len(sell_bl)}")
    
    # VG
    vg_bl = [r for r in big_losses if r.get("vg_triggered")]
    print(f"  VG triggered: {len(vg_bl)} ({100*len(vg_bl)/len(big_losses):.0f}%)")
    
    # Spread at big loss
    sps = [r.get("spread_bps") for r in big_losses if r.get("spread_bps") is not None]
    if sps:
        print(f"  Spread: avg={sum(sps)/len(sps):.2f}, med={sorted(sps)[len(sps)//2]:.2f}")
    
    # 120s PnL
    both120 = [r for r in big_losses if r.get("post_fill_120s_pnl") is not None]
    if both120:
        recovered = sum(1 for r in both120 if r["post_fill_120s_pnl"] > r["post_fill_30s_pnl"])
        print(f"  120s recovery: {recovered}/{len(both120)} improved at 120s")
        still_bad = [r for r in both120 if r["post_fill_120s_pnl"] < -10]
        print(f"  Still >10bps loss at 120s: {len(still_bad)}/{len(both120)}")
    
    # Day of week
    dow_dist = defaultdict(int)
    for r in big_losses:
        dt = datetime.fromtimestamp(r["timestamp"], tz=timezone.utc)
        dow_dist[dt.strftime("%a")] += 1
    print(f"  Day of week: {dict(dow_dist)}")
    
    # Hour concentration
    hour_dist = defaultdict(int)
    for r in big_losses:
        jst_h = (datetime.fromtimestamp(r["timestamp"], tz=timezone.utc).hour + 9) % 24
        hour_dist[jst_h] += 1
    top3 = sorted(hour_dist.items(), key=lambda x: -x[1])[:5]
    print(f"  Top JST hours: {', '.join(f'{h}h={c}' for h, c in top3)}")
    
    # Total damage
    total_damage = sum(r.get("post_fill_30s_pnl", 0) or 0 for r in big_losses)
    print(f"  Total damage: {total_damage:+.1f}bps (of {sum(r.get('post_fill_30s_pnl',0) or 0 for r in filled):+.1f} total)")


def what_if_analysis(filled):
    """What-if シミュレーション."""
    print("\n=== WHAT-IF SIMULATIONS ===")
    total_pnl = sum(r.get("post_fill_30s_pnl", 0) or 0 for r in filled)
    print(f"  Actual: {total_pnl:+.1f}bps over {len(filled)} fills")
    
    # 1. >10bps の損失を -10bps にキャップした場合
    capped = sum(max(r.get("post_fill_30s_pnl", 0) or 0, -10.0) for r in filled)
    print(f"  If cap losses at -10bps: {capped:+.1f}bps (saving {capped - total_pnl:+.1f})")
    
    # 2. 最悪の3時間帯を除外
    hourly_pnl = defaultdict(float)
    hourly_fills = defaultdict(list)
    for r in filled:
        jst_h = (datetime.fromtimestamp(r["timestamp"], tz=timezone.utc).hour + 9) % 24
        hourly_pnl[jst_h] += r.get("post_fill_30s_pnl", 0) or 0
        hourly_fills[jst_h].append(r)
    worst3 = sorted(hourly_pnl.items(), key=lambda x: x[1])[:3]
    excluded = sum(v for _, v in worst3)
    excluded_n = sum(len(hourly_fills[h]) for h, _ in worst3)
    print(f"  If skip worst 3 hours ({[h for h,_ in worst3]}): "
          f"{total_pnl - excluded:+.1f}bps (skip {excluded_n} fills)")
    
    # 3. VGが正しく発火した勝ちトレードのみ
    vg_winners = [r for r in filled if r.get("vg_triggered") and (r.get("post_fill_30s_pnl") or 0) > 0]
    vg_pnl = sum(r.get("post_fill_30s_pnl", 0) or 0 for r in vg_winners)
    print(f"  VG + win only: {vg_pnl:+.1f}bps over {len(vg_winners)} fills")
    
    # 4. spread > 2.5bps のみ
    wide = [r for r in filled if (r.get("spread_bps") or 0) > 2.5]
    wide_pnl = sum(r.get("post_fill_30s_pnl", 0) or 0 for r in wide)
    print(f"  If only spread>2.5bps: {wide_pnl:+.1f}bps over {len(wide)} fills")
    
    # 5. sell 側 offset を 0.25 にした場合 (概算: 現在 0.18)
    sell_recs = [r for r in filled if r["side"] == "sell"]
    # 近似: offset増加分 = (0.25-0.18) * BTC価格 ≈ 7bps追加マージン → skip率増加で fills削減見込み
    print(f"  [estimate] If sell offset 0.18→0.25: ~{len(sell_recs)*0.3:.0f} fewer fills, "
          f"surviving fills gain ~+7bps/fill edge")


def main():
    filled = load_filled()
    winning_trade_deep_dive(filled)
    hour_ranking(filled)
    daily_analysis(filled)
    spread_pnl_correlation(filled)
    big_loss_pattern(filled)
    what_if_analysis(filled)


if __name__ == "__main__":
    main()
