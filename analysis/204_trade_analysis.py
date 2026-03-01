"""204# 包括的トレード分析 — MM理論・一目均衡表・市場理論に基づく診断."""

import json
import glob
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILL_DIR = ROOT / "results" / "v460" / "fill_test"


def load_all_records():
    """全fill recordsを読み込み."""
    files = sorted(glob.glob(str(FILL_DIR / "fill_records_*.jsonl")))
    all_recs = []
    for f in files:
        date_str = os.path.basename(f).replace("fill_records_", "").replace(".jsonl", "")
        for line in open(f):
            r = json.loads(line)
            r["_file_date"] = date_str
            all_recs.append(r)
    return all_recs


def daily_summary(filled):
    """日次サマリ."""
    daily = defaultdict(lambda: {
        "buy": 0, "sell": 0, "buy_pnl": 0.0, "sell_pnl": 0.0,
        "wins": 0, "losses": 0, "big_wins": 0, "big_losses": 0,
    })
    for r in filled:
        d = r["_file_date"]
        s = r["side"]
        pnl = r.get("post_fill_30s_pnl", 0) or 0
        daily[d][s] += 1
        daily[d][s + "_pnl"] += pnl
        if pnl > 0:
            daily[d]["wins"] += 1
            if pnl > 5.0:
                daily[d]["big_wins"] += 1
        else:
            daily[d]["losses"] += 1
            if pnl < -5.0:
                daily[d]["big_losses"] += 1
    return daily


def win_loss_analysis(filled):
    """勝ち/負けトレードの特徴分析."""
    winners = [r for r in filled if (r.get("post_fill_30s_pnl") or 0) > 0]
    losers = [r for r in filled if (r.get("post_fill_30s_pnl") or 0) < 0]
    breakeven = [r for r in filled if (r.get("post_fill_30s_pnl") or 0) == 0]

    def stats(recs, label):
        if not recs:
            return
        pnls = [r.get("post_fill_30s_pnl", 0) or 0 for r in recs]
        spreads = [r.get("spread_bps") for r in recs if r.get("spread_bps") is not None]
        offsets = [r.get("offset_applied_bps") for r in recs if r.get("offset_applied_bps") is not None]
        velocities = [r.get("velocity_60s") for r in recs if r.get("velocity_60s") is not None]
        ob_imbs = [r.get("ob_imbalance") for r in recs if r.get("ob_imbalance") is not None]
        vg_flags = [r for r in recs if r.get("vg_triggered")]
        skip_probs = [r.get("skip_gate_prob") for r in recs if r.get("skip_gate_prob") is not None]
        # side breakdown
        buy_count = sum(1 for r in recs if r["side"] == "buy")
        sell_count = sum(1 for r in recs if r["side"] == "sell")

        print(f"\n{'='*60}")
        print(f"  {label}: {len(recs)} trades (buy={buy_count}, sell={sell_count})")
        print(f"{'='*60}")
        print(f"  PnL: avg={sum(pnls)/len(pnls):+.2f}, med={sorted(pnls)[len(pnls)//2]:+.2f}, "
              f"min={min(pnls):+.2f}, max={max(pnls):+.2f}, sum={sum(pnls):+.2f}")
        if spreads:
            print(f"  Spread: avg={sum(spreads)/len(spreads):.2f}, "
                  f"min={min(spreads):.2f}, max={max(spreads):.2f}")
        if offsets:
            print(f"  Offset: avg={sum(offsets)/len(offsets):.2f}, "
                  f"min={min(offsets):.2f}, max={max(offsets):.2f}")
        if velocities:
            print(f"  Velocity60s: avg={sum(velocities)/len(velocities):.2f}, "
                  f"min={min(velocities):.2f}, max={max(velocities):.2f}")
        if ob_imbs:
            print(f"  OB Imbalance: avg={sum(ob_imbs)/len(ob_imbs):.3f}, "
                  f"min={min(ob_imbs):.3f}, max={max(ob_imbs):.3f}")
        if skip_probs:
            print(f"  SkipGate Prob: avg={sum(skip_probs)/len(skip_probs):.3f}, "
                  f"min={min(skip_probs):.3f}, max={max(skip_probs):.3f}")
        print(f"  VG triggered: {len(vg_flags)} ({100*len(vg_flags)/len(recs):.1f}%)")

        # 時間帯分布 (UTC)
        hour_dist = defaultdict(int)
        for r in recs:
            h = datetime.fromtimestamp(r["timestamp"], tz=timezone.utc).hour
            hour_dist[h] += 1
        # 上位5時間帯
        top_hours = sorted(hour_dist.items(), key=lambda x: -x[1])[:5]
        print(f"  Top hours (UTC): {', '.join(f'{h}h={c}' for h, c in top_hours)}")

    stats(winners, "WINNERS")
    stats(losers, "LOSERS")
    print(f"\nBreakeven: {len(breakeven)}")
    return winners, losers


def consecutive_loss_analysis(filled):
    """連敗パターン分析."""
    streaks = []
    current_streak = []
    for r in filled:
        pnl = r.get("post_fill_30s_pnl", 0) or 0
        if pnl < 0:
            current_streak.append(r)
        else:
            if len(current_streak) >= 3:
                streaks.append(current_streak)
            current_streak = []
    if len(current_streak) >= 3:
        streaks.append(current_streak)

    print(f"\n{'='*60}")
    print(f"  CONSECUTIVE LOSSES (3+)")
    print(f"{'='*60}")
    print(f"  Total streaks of 3+: {len(streaks)}")
    if streaks:
        max_streak = max(streaks, key=len)
        print(f"  Longest streak: {len(max_streak)} trades")
        total_damage = sum(sum(r.get("post_fill_30s_pnl", 0) or 0 for r in s) for s in streaks)
        print(f"  Cumulative damage from streaks: {total_damage:+.2f} bps")
        # 最悪の連敗
        worst = max(streaks, key=lambda s: abs(sum(r.get("post_fill_30s_pnl", 0) or 0 for r in s)))
        worst_pnl = sum(r.get("post_fill_30s_pnl", 0) or 0 for r in worst)
        t0 = datetime.fromtimestamp(worst[0]["timestamp"], tz=timezone.utc)
        print(f"  Worst streak: {len(worst)} trades, {worst_pnl:+.2f}bps, start={t0.isoformat()}")
    return streaks


def spread_vs_pnl_analysis(filled):
    """スプレッド vs PnL の相関分析 (MM理論的視点)."""
    print(f"\n{'='*60}")
    print(f"  MM THEORY: SPREAD vs PnL ANALYSIS")
    print(f"{'='*60}")

    with_spread = [(r.get("spread_bps", 0) or 0, r.get("post_fill_30s_pnl", 0) or 0) for r in filled
                   if r.get("spread_bps") is not None]
    if not with_spread:
        print("  No spread data available")
        return

    # スプレッド帯別 PnL
    buckets = {"<3bps": [], "3-5bps": [], "5-8bps": [], "8-12bps": [], ">12bps": []}
    for sp, pnl in with_spread:
        if sp < 3:
            buckets["<3bps"].append(pnl)
        elif sp < 5:
            buckets["3-5bps"].append(pnl)
        elif sp < 8:
            buckets["5-8bps"].append(pnl)
        elif sp < 12:
            buckets["8-12bps"].append(pnl)
        else:
            buckets[">12bps"].append(pnl)

    print("  Spread Band | Count | Avg PnL  | Win Rate | Sum PnL")
    print("  " + "-" * 60)
    for band, pnls in buckets.items():
        if pnls:
            avg = sum(pnls) / len(pnls)
            wr = 100 * sum(1 for p in pnls if p > 0) / len(pnls)
            print(f"  {band:>9s}   | {len(pnls):5d} | {avg:+7.2f} | {wr:5.1f}%  | {sum(pnls):+8.1f}")


def velocity_regime_analysis(filled):
    """ボラティリティ・velocity regime 分析 (一目均衡表的視点)."""
    print(f"\n{'='*60}")
    print(f"  REGIME & VELOCITY ANALYSIS (Ichimoku perspective)")
    print(f"{'='*60}")

    # velocity_60s 帯別分析
    with_vel = [(r.get("velocity_60s", 0) or 0, r.get("post_fill_30s_pnl", 0) or 0, r["side"]) for r in filled
                if r.get("velocity_60s") is not None]
    if not with_vel:
        print("  No velocity data available")
        return

    vel_buckets = {
        "strong_down(<-15)": [], "down(-15~-5)": [], "neutral(-5~5)": [],
        "up(5~15)": [], "strong_up(>15)": [],
    }
    for vel, pnl, side in with_vel:
        if vel < -15:
            vel_buckets["strong_down(<-15)"].append((pnl, side))
        elif vel < -5:
            vel_buckets["down(-15~-5)"].append((pnl, side))
        elif vel < 5:
            vel_buckets["neutral(-5~5)"].append((pnl, side))
        elif vel < 15:
            vel_buckets["up(5~15)"].append((pnl, side))
        else:
            vel_buckets["strong_up(>15)"].append((pnl, side))

    print("  Velocity Band     | Count | Avg PnL  | WR    | Buy WR   | Sell WR")
    print("  " + "-" * 72)
    for band, entries in vel_buckets.items():
        if entries:
            pnls = [e[0] for e in entries]
            buy_pnls = [e[0] for e in entries if e[1] == "buy"]
            sell_pnls = [e[0] for e in entries if e[1] == "sell"]
            avg = sum(pnls) / len(pnls)
            wr = 100 * sum(1 for p in pnls if p > 0) / len(pnls)
            buy_wr = (100 * sum(1 for p in buy_pnls if p > 0) / len(buy_pnls)) if buy_pnls else 0
            sell_wr = (100 * sum(1 for p in sell_pnls if p > 0) / len(sell_pnls)) if sell_pnls else 0
            print(f"  {band:>19s} | {len(entries):5d} | {avg:+7.2f} | {wr:5.1f}% | {buy_wr:5.1f}%   | {sell_wr:5.1f}%")


def time_of_day_analysis(filled):
    """時間帯別分析."""
    print(f"\n{'='*60}")
    print(f"  TIME-OF-DAY ANALYSIS (JST = UTC+9)")
    print(f"{'='*60}")

    hourly = defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0})
    for r in filled:
        jst_h = (datetime.fromtimestamp(r["timestamp"], tz=timezone.utc).hour + 9) % 24
        pnl = r.get("post_fill_30s_pnl", 0) or 0
        hourly[jst_h]["count"] += 1
        hourly[jst_h]["pnl"] += pnl
        if pnl > 0:
            hourly[jst_h]["wins"] += 1

    print("  JST Hour | Count | Avg PnL  | Win Rate | Total PnL")
    print("  " + "-" * 55)
    for h in range(24):
        v = hourly[h]
        if v["count"] > 0:
            avg = v["pnl"] / v["count"]
            wr = 100 * v["wins"] / v["count"]
            print(f"  {h:02d}:00    | {v['count']:5d} | {avg:+7.2f} | {wr:5.1f}%  | {v['pnl']:+8.1f}")


def offset_effectiveness_analysis(filled):
    """オフセット有効性分析 — MM理論におけるスプレッド設定の妥当性."""
    print(f"\n{'='*60}")
    print(f"  OFFSET EFFECTIVENESS (Maker Price Strategy)")
    print(f"{'='*60}")

    for side in ["buy", "sell"]:
        side_recs = [r for r in filled if r["side"] == side]
        if not side_recs:
            continue
        pnls = [r.get("post_fill_30s_pnl", 0) or 0 for r in side_recs]
        offsets = [r.get("offset_applied_bps") for r in side_recs if r.get("offset_applied_bps") is not None]
        wins = sum(1 for p in pnls if p > 0)
        print(f"\n  [{side.upper()}] {len(side_recs)} fills, WR={100*wins/len(side_recs):.1f}%, "
              f"avg_pnl={sum(pnls)/len(pnls):+.2f}")
        if offsets:
            print(f"    Offset applied: avg={sum(offsets)/len(offsets):.2f}, "
                  f"min={min(offsets):.2f}, max={max(offsets):.2f}")

        # VG triggered vs not
        vg_on = [r for r in side_recs if r.get("vg_triggered")]
        vg_off = [r for r in side_recs if not r.get("vg_triggered")]
        if vg_on:
            vg_pnls = [r.get("post_fill_30s_pnl", 0) or 0 for r in vg_on]
            print(f"    VG ON:  {len(vg_on)} fills, avg_pnl={sum(vg_pnls)/len(vg_pnls):+.2f}")
        if vg_off:
            vg_pnls = [r.get("post_fill_30s_pnl", 0) or 0 for r in vg_off]
            print(f"    VG OFF: {len(vg_off)} fills, avg_pnl={sum(vg_pnls)/len(vg_pnls):+.2f}")


def adverse_selection_analysis(filled):
    """逆選択 (Adverse Selection) 分析 — MM理論の核心."""
    print(f"\n{'='*60}")
    print(f"  ADVERSE SELECTION ANALYSIS")
    print(f"{'='*60}")

    # post_fill_30s vs post_fill_120s の比較
    both = [r for r in filled
            if r.get("post_fill_30s_pnl") is not None and r.get("post_fill_120s_pnl") is not None]
    if both:
        pnl30 = [r["post_fill_30s_pnl"] for r in both]
        pnl120 = [r["post_fill_120s_pnl"] for r in both]
        print(f"  Records with both 30s/120s PnL: {len(both)}")
        print(f"  30s PnL:  avg={sum(pnl30)/len(pnl30):+.2f}, sum={sum(pnl30):+.1f}")
        print(f"  120s PnL: avg={sum(pnl120)/len(pnl120):+.2f}, sum={sum(pnl120):+.1f}")

        # 30sで勝っているが120sで負けている (= 一時的な利益)
        temp_win = [r for r in both if r["post_fill_30s_pnl"] > 0 and r["post_fill_120s_pnl"] < 0]
        # 30sで負けているが120sで勝っている (= 一時的な損失)
        temp_loss = [r for r in both if r["post_fill_30s_pnl"] < 0 and r["post_fill_120s_pnl"] > 0]
        # 30sも120sも負け (= persistent adverse selection)
        persistent = [r for r in both if r["post_fill_30s_pnl"] < 0 and r["post_fill_120s_pnl"] < 0]
        print(f"  Temp win (30s>0, 120s<0): {len(temp_win)} — short-term noise profit")
        print(f"  Temp loss (30s<0, 120s>0): {len(temp_loss)} — recoverable")
        print(f"  Persistent loss (both<0): {len(persistent)} — TRUE adverse selection")

        if persistent:
            p_avg30 = sum(r["post_fill_30s_pnl"] for r in persistent) / len(persistent)
            p_avg120 = sum(r["post_fill_120s_pnl"] for r in persistent) / len(persistent)
            print(f"    Persistent: avg30={p_avg30:+.2f}, avg120={p_avg120:+.2f}")

    # SkipGate が通したのに負けたケース
    skipped_loss = [r for r in filled
                    if r.get("skip_gate_prob") is not None
                    and (r.get("post_fill_30s_pnl") or 0) < -5.0]
    if skipped_loss:
        probs = [r["skip_gate_prob"] for r in skipped_loss]
        print(f"\n  SkipGate passed but lost >5bps: {len(skipped_loss)}")
        print(f"    SkipGate prob: avg={sum(probs)/len(probs):.3f}, max={max(probs):.3f}")


def recent_window_analysis(filled, days=3):
    """直近N日の詳細分析."""
    from datetime import datetime, timezone

    print(f"\n{'='*60}")
    print(f"  RECENT {days}-DAY DETAILED ANALYSIS")
    print(f"{'='*60}")

    dates = sorted(set(r["_file_date"] for r in filled))[-days:]
    recent = [r for r in filled if r["_file_date"] in dates]

    for date in dates:
        day_recs = [r for r in recent if r["_file_date"] == date]
        print(f"\n  --- {date} ({len(day_recs)} fills) ---")

        for r in sorted(day_recs, key=lambda x: x["timestamp"]):
            t = datetime.fromtimestamp(r["timestamp"], tz=timezone.utc)
            pnl = r.get("post_fill_30s_pnl", 0) or 0
            pnl120 = r.get("post_fill_120s_pnl")
            sp = r.get("spread_bps")
            vel = r.get("velocity_60s")
            vg = "VG" if r.get("vg_triggered") else "  "
            mark = "***" if abs(pnl) > 10 else "   "
            pnl120_str = f"{pnl120:+.1f}" if pnl120 is not None else "  n/a"
            sp_str = f"{sp:.1f}" if sp is not None else "n/a"
            vel_str = f"{vel:+.1f}" if vel is not None else "n/a"
            print(f"    {t.strftime('%H:%M')}UTC {r['side']:4s} pnl30={pnl:+6.1f} "
                  f"pnl120={pnl120_str:>6s} sp={sp_str:>5s} vel={vel_str:>6s} {vg} {mark}")


def market_making_theory_diagnosis(filled):
    """MM理論に基づく総合診断."""
    print(f"\n{'='*60}")
    print(f"  MARKET MAKING THEORY — COMPREHENSIVE DIAGNOSIS")
    print(f"{'='*60}")

    total = len(filled)
    pnls = [r.get("post_fill_30s_pnl", 0) or 0 for r in filled]
    wins = sum(1 for p in pnls if p > 0)
    avg_win = sum(p for p in pnls if p > 0) / max(1, wins)
    avg_loss = sum(p for p in pnls if p < 0) / max(1, total - wins)

    print(f"""
  1. INVENTORY RISK (在庫リスク):
     - Win Rate: {100*wins/total:.1f}% (MM typically needs >50%)
     - Avg Win: {avg_win:+.2f}bps, Avg Loss: {avg_loss:+.2f}bps
     - Risk/Reward Ratio: {abs(avg_loss/avg_win):.2f}:1
     - 診断: {'CRITICAL' if abs(avg_loss) > 2*avg_win else 'WARNING' if abs(avg_loss) > avg_win else 'OK'}

  2. SPREAD CAPTURE (スプレッド獲得):
     - MMの本質は bid-ask spread の獲得
     - 30秒後にPnLが確定 → spread halfwidth を上回る逆行が多ければ
       offsetが不十分か、逆選択に晒されている

  3. ADVERSE SELECTION DEFENSE:
     - 情報トレーダーに対するフィルタリング (SkipGate)
     - Velocity/OB Imbalance ベースの回避
     - VG (Velocity Guard) による価格シフト

  4. ICHIMOKU KINKO HYO PERSPECTIVE (一目均衡表の視点):
     - 「三役好転/逆転」 = trend confirmation → trend-following MM
     - 「雲の厚さ」 = support/resistance → spread widening signal
     - 「遅行線」 = confirmation lag → adverse selection timing
     - 現状: regime_detector (ranging/trending) が部分的に代替
     - 不足: 転換期の検知が甘い (regime stability が低い時に損失集中の可能性)

  5. AVELLANEDA-STOIKOV MODEL:
     - 最適スプレッド = γσ²T + (2/γ)ln(1 + γ/κ)
     - σ (ボラティリティ) が上昇時にスプレッドを拡大すべき
     - κ (注文到着率) は coincheck の流動性に依存
     - 現状: base_offset_ratio で固定的 → σ連動が不十分の可能性
""")


def generate_improvement_candidates(filled):
    """改善候補の生成."""
    print(f"\n{'='*60}")
    print(f"  IMPROVEMENT CANDIDATES (改善候補)")
    print(f"{'='*60}")

    pnls = [r.get("post_fill_30s_pnl", 0) or 0 for r in filled]
    total = len(filled)
    wins = sum(1 for p in pnls if p > 0)

    # 1. 大損トレード分析
    big_losses = [r for r in filled if (r.get("post_fill_30s_pnl") or 0) < -10]
    print(f"\n  A. BIG LOSS PREVENTION (>10bps losses: {len(big_losses)})")
    if big_losses:
        for r in sorted(big_losses, key=lambda x: x.get("post_fill_30s_pnl", 0))[:5]:
            t = datetime.fromtimestamp(r["timestamp"], tz=timezone.utc)
            pnl = r["post_fill_30s_pnl"]
            vel = r.get("velocity_60s")
            sp = r.get("spread_bps")
            print(f"    {t.isoformat()} {r['side']} pnl={pnl:+.1f} vel={vel} sp={sp}")

    # 2. Side asymmetry
    buy_recs = [r for r in filled if r["side"] == "buy"]
    sell_recs = [r for r in filled if r["side"] == "sell"]
    buy_pnl = sum(r.get("post_fill_30s_pnl", 0) or 0 for r in buy_recs)
    sell_pnl = sum(r.get("post_fill_30s_pnl", 0) or 0 for r in sell_recs)
    buy_wr = 100 * sum(1 for r in buy_recs if (r.get("post_fill_30s_pnl") or 0) > 0) / max(1, len(buy_recs))
    sell_wr = 100 * sum(1 for r in sell_recs if (r.get("post_fill_30s_pnl") or 0) > 0) / max(1, len(sell_recs))
    print(f"\n  B. SIDE ASYMMETRY:")
    print(f"    Buy:  {len(buy_recs)} fills, WR={buy_wr:.1f}%, sum={buy_pnl:+.1f}bps")
    print(f"    Sell: {len(sell_recs)} fills, WR={sell_wr:.1f}%, sum={sell_pnl:+.1f}bps")
    print(f"    Imbalance: sell PnL is {sell_pnl/max(1,abs(buy_pnl)):.1f}x buy PnL")

    # 3. 勝ちトレードの共通特徴
    winners = [r for r in filled if (r.get("post_fill_30s_pnl") or 0) > 2.0]
    print(f"\n  C. WINNING TRADE FEATURES (>2bps, n={len(winners)}):")
    if winners:
        win_spreads = [r.get("spread_bps", 0) or 0 for r in winners if r.get("spread_bps")]
        win_vels = [r.get("velocity_60s", 0) or 0 for r in winners if r.get("velocity_60s")]
        if win_spreads:
            print(f"    Spread: avg={sum(win_spreads)/len(win_spreads):.2f}")
        if win_vels:
            print(f"    Velocity: avg={sum(win_vels)/len(win_vels):.2f}")

    # 4. VG有効性
    vg_on = [r for r in filled if r.get("vg_triggered")]
    vg_off = [r for r in filled if not r.get("vg_triggered")]
    if vg_on:
        vg_pnl = sum(r.get("post_fill_30s_pnl", 0) or 0 for r in vg_on) / len(vg_on)
        novg_pnl = sum(r.get("post_fill_30s_pnl", 0) or 0 for r in vg_off) / max(1, len(vg_off))
        print(f"\n  D. VG EFFECTIVENESS:")
        print(f"    VG ON:  avg_pnl={vg_pnl:+.2f} ({len(vg_on)} fills)")
        print(f"    VG OFF: avg_pnl={novg_pnl:+.2f} ({len(vg_off)} fills)")


def main():
    all_recs = load_all_records()
    filled = [r for r in all_recs if r.get("filled")]
    print(f"Total records: {len(all_recs)}, Filled: {len(filled)}")
    print(f"Date range: {all_recs[0]['_file_date']} - {all_recs[-1]['_file_date']}")

    # 日次サマリ
    daily = daily_summary(filled)
    print(f"\nDate       | Buy  Sell | BuyPnL   SellPnL  | Total    | W/L  | BigW/BigL")
    print("-" * 85)
    cum = 0.0
    for d in sorted(daily.keys()):
        v = daily[d]
        total = v["buy_pnl"] + v["sell_pnl"]
        cum += total
        t = v["buy"] + v["sell"]
        print(f"{d} | {v['buy']:4d} {v['sell']:4d} | {v['buy_pnl']:+8.1f} {v['sell_pnl']:+8.1f} "
              f"| {total:+8.1f} | {v['wins']:3d}/{v['losses']:3d} | {v['big_wins']:3d}/{v['big_losses']:3d}")
    print(f"{'TOTAL':10s} |           |                    | {cum:+8.1f} |")

    win_loss_analysis(filled)
    consecutive_loss_analysis(filled)
    spread_vs_pnl_analysis(filled)
    velocity_regime_analysis(filled)
    time_of_day_analysis(filled)
    offset_effectiveness_analysis(filled)
    adverse_selection_analysis(filled)
    recent_window_analysis(filled, days=3)
    market_making_theory_diagnosis(filled)
    generate_improvement_candidates(filled)


if __name__ == "__main__":
    main()
