#!/usr/bin/env python3
"""122# A4+B4: Volatility Guard 効果測定 + AS 日別トレンド分析.

118# §5.4: VG 実装はされたが効果測定が未実施。
118# §8.6: 日別 KPI トレンドの可視化が未実施。

■ A4: Volatility Guard Effectiveness
  - fill_test.log から [volatility_guard] メッセージを解析
  - VG 発動サイクル vs 非発動サイクルの PnL / AS rate を比較
  - offset boost 倍率 (2.0x) の妥当性を評価

■ B4: AS Rate Daily Trend
  - 日別 / 8h別の KPI (AS率, PnL, fill_rate) をブレークダウン
  - A3 (sell SkipGate 無効化) 前後の変化を可視化
  - regime_detector の実効性評価にも活用

Usage:
  python scripts/v460/analysis/vg_and_trend.py
  python scripts/v460/analysis/vg_and_trend.py --results-dir results/v460/fill_test
  python scripts/v460/analysis/vg_and_trend.py --json --output reports/vg_analysis.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ztb.metrics.fill_quality import (
    FillRecord,
    compute_fill_metrics,
    filter_clean_records,
    load_fill_records_glob,
)
from ztb.io.json_io import write_json


# ======================================================================
# A4: Volatility Guard effectiveness
# ======================================================================

# TODO(123# Gemini review): プレーンテキストログの regex パースは脆い。
#   VG 発動等の重要イベントは JSONL 構造化ログとして出力・保存する設計に
#   変更することを推奨。→ 118# E12 として追跡。

# Log format: 2026-02-18 12:18:47,674 INFO [...] [volatility_guard] 107# sell offset boosted: 0.2800→0.3000 (vpin=0.98)
_VG_PATTERN = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \w+ \[.*?\] "
    r"\[volatility_guard\] 107# (\w+) offset boosted: "
    r"([\d.]+)→([\d.]+) \((.+)\)"
)

# Log format: === Cycle N (side) ===
_CYCLE_PATTERN = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \w+ \[.*?\] "
    r"=== Cycle (\d+) \((\w+)\) ==="
)


def _parse_vg_activations(log_path: Path) -> list[dict]:
    """fill_test.log から VG 発動イベントを抽出.

    ログのタイムスタンプはローカルタイム (JST=UTC+9) で記録されるため、
    epoch 変換時にローカルタイムとして解釈する。

    Returns:
        list[dict]: 各要素は {timestamp, side, pre_offset, post_offset, reason}
    """
    activations: list[dict] = []
    if not log_path.exists():
        return activations

    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _VG_PATTERN.match(line.strip())
            if m:
                ts_str, side, pre, post, reason = m.groups()
                # ログはローカルタイム → mktime() でローカル epoch に変換
                import time as _time
                st = _time.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                epoch = _time.mktime(st)
                activations.append({
                    "timestamp": epoch,
                    "datetime_str": ts_str,
                    "side": side,
                    "pre_offset": float(pre),
                    "post_offset": float(post),
                    "reason": reason,
                })
    return activations


def _match_vg_to_records(
    activations: list[dict],
    records: list[FillRecord],
    tolerance_sec: float = 30.0,
) -> set[str]:
    """VG 発動タイムスタンプを FillRecord に紐付け、VG 発動 cycle_id を返す.

    VG はサイクル内の _compute_maker_price() 中に発動するため、
    FillRecord.timestamp (t_submit) と VG ログの時刻は数秒以内。
    """
    vg_cycle_ids: set[str] = set()
    # タイムスタンプでソート済み records のインデックス
    sorted_records = sorted(records, key=lambda r: r.timestamp)

    for act in activations:
        vg_ts = act["timestamp"]
        # 最近接の record を探す (二分探索相当)
        best_match: Optional[FillRecord] = None
        best_diff = float("inf")
        for rec in sorted_records:
            diff = abs(rec.timestamp - vg_ts)
            if diff < best_diff:
                best_diff = diff
                best_match = rec
            elif diff > best_diff + tolerance_sec:
                break  # ソート済みなのでこれ以上離れる

        if best_match is not None and best_diff <= tolerance_sec:
            # side も一致確認
            if best_match.side == act["side"]:
                vg_cycle_ids.add(best_match.cycle_id)

    return vg_cycle_ids


def analyze_vg_effectiveness(
    records: list[FillRecord],
    vg_cycle_ids: set[str],
) -> dict:
    """VG 発動群 vs 非発動群の KPI 比較."""
    filled = [r for r in records if r.filled]

    vg_filled = [r for r in filled if r.cycle_id in vg_cycle_ids]
    non_vg_filled = [r for r in filled if r.cycle_id not in vg_cycle_ids]

    # VG 発動サイクル全体 (filled + cancelled)
    vg_all = [r for r in records if r.cycle_id in vg_cycle_ids]
    non_vg_all = [r for r in records if r.cycle_id not in vg_cycle_ids]

    def _group_stats(group: list[FillRecord], label: str) -> dict:
        if not group:
            return {"label": label, "n": 0}

        pnl_30 = [r.post_fill_30s_pnl for r in group if r.post_fill_30s_pnl is not None]
        pnl_60 = [r.post_fill_60s_pnl for r in group if r.post_fill_60s_pnl is not None]
        pnl_120 = [r.post_fill_120s_pnl for r in group if r.post_fill_120s_pnl is not None]
        as_count = sum(1 for r in group if r.adverse_selected is True)

        import numpy as np

        result: dict = {
            "label": label,
            "n": len(group),
            "pnl_30s_mean": float(np.mean(pnl_30)) if pnl_30 else None,
            "pnl_60s_mean": float(np.mean(pnl_60)) if pnl_60 else None,
            "pnl_120s_mean": float(np.mean(pnl_120)) if pnl_120 else None,
            "as_count": as_count,
            "as_rate": as_count / len(group) if group else 0.0,
        }
        return result

    def _fill_rate(all_recs: list[FillRecord]) -> float:
        if not all_recs:
            return 0.0
        return sum(1 for r in all_recs if r.filled) / len(all_recs)

    vg_stats = _group_stats(vg_filled, "VG-active (filled)")
    non_vg_stats = _group_stats(non_vg_filled, "Non-VG (filled)")

    # VG offset boost 分析
    offsets_vg = [
        r.effective_offset_used for r in vg_filled
        if r.effective_offset_used is not None
    ]
    offsets_non = [
        r.effective_offset_used for r in non_vg_filled
        if r.effective_offset_used is not None
    ]
    import numpy as np

    return {
        "vg_total_cycles": len(vg_all),
        "non_vg_total_cycles": len(non_vg_all),
        "vg_fill_rate": _fill_rate(vg_all),
        "non_vg_fill_rate": _fill_rate(non_vg_all),
        "vg_filled": vg_stats,
        "non_vg_filled": non_vg_stats,
        "vg_mean_offset": float(np.mean(offsets_vg)) if offsets_vg else None,
        "non_vg_mean_offset": float(np.mean(offsets_non)) if offsets_non else None,
        "interpretation": _interpret_vg(vg_stats, non_vg_stats),
    }


def _interpret_vg(vg: dict, non_vg: dict) -> str:
    """VG 効果の自動解釈."""
    if vg["n"] == 0:
        return "VG 発動サイクルが 0 件: 測定不能"
    if non_vg["n"] == 0:
        return "非 VG サイクルが 0 件: 測定不能"

    lines = []
    # AS rate 比較
    vg_as = vg["as_rate"]
    non_vg_as = non_vg["as_rate"]
    diff_as = non_vg_as - vg_as
    if diff_as > 0.02:
        lines.append(f"VG により AS rate が {diff_as:.1%} pt 改善 (有効)")
    elif diff_as < -0.02:
        lines.append(f"VG で AS rate が {abs(diff_as):.1%} pt 悪化 (要検討)")
    else:
        lines.append("AS rate に有意な差なし")

    # PnL 比較
    if vg["pnl_30s_mean"] is not None and non_vg["pnl_30s_mean"] is not None:
        pnl_diff = vg["pnl_30s_mean"] - non_vg["pnl_30s_mean"]
        if pnl_diff > 0.1:
            lines.append(f"VG 群の PnL30 が +{pnl_diff:.3f}bps 良好")
        elif pnl_diff < -0.1:
            lines.append(f"VG 群の PnL30 が {pnl_diff:.3f}bps 悪化 (offset 過大?)")
        else:
            lines.append("PnL30 に有意な差なし")

    return " / ".join(lines)


# ======================================================================
# B4: Daily KPI trend
# ======================================================================


def _date_key(ts: float) -> str:
    """epoch → 'YYYY-MM-DD'."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _period_key(ts: float) -> str:
    """epoch → 'YYYY-MM-DD_8h-period' (0-8, 8-16, 16-24)."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    period = dt.hour // 8
    labels = ["00-08", "08-16", "16-24"]
    return f"{dt.strftime('%Y-%m-%d')}_{labels[period]}"


def analyze_daily_trend(records: list[FillRecord]) -> list[dict]:
    """日別 KPI ブレークダウン."""
    by_day: dict[str, list[FillRecord]] = defaultdict(list)
    for r in records:
        by_day[_date_key(r.timestamp)].append(r)

    results = []
    for day in sorted(by_day.keys()):
        recs = by_day[day]
        filled = [r for r in recs if r.filled]
        n_total = len(recs)
        n_filled = len(filled)
        n_as = sum(1 for r in filled if r.adverse_selected is True)

        pnl_30 = [r.post_fill_30s_pnl for r in filled if r.post_fill_30s_pnl is not None]
        pnl_120 = [r.post_fill_120s_pnl for r in filled if r.post_fill_120s_pnl is not None]
        import numpy as np

        # side 別 AS
        buy_filled = [r for r in filled if r.side == "buy"]
        sell_filled = [r for r in filled if r.side == "sell"]
        buy_as = sum(1 for r in buy_filled if r.adverse_selected is True)
        sell_as = sum(1 for r in sell_filled if r.adverse_selected is True)

        results.append({
            "date": day,
            "total_cycles": n_total,
            "filled": n_filled,
            "fill_rate": n_filled / n_total if n_total > 0 else 0.0,
            "as_count": n_as,
            "as_rate": n_as / n_filled if n_filled > 0 else 0.0,
            "pnl_30s_mean": float(np.mean(pnl_30)) if pnl_30 else None,
            "pnl_120s_mean": float(np.mean(pnl_120)) if pnl_120 else None,
            "buy_filled": len(buy_filled),
            "buy_as_rate": buy_as / len(buy_filled) if buy_filled else 0.0,
            "sell_filled": len(sell_filled),
            "sell_as_rate": sell_as / len(sell_filled) if sell_filled else 0.0,
        })
    return results


def analyze_8h_trend(records: list[FillRecord]) -> list[dict]:
    """8 時間帯別 KPI ブレークダウン."""
    by_period: dict[str, list[FillRecord]] = defaultdict(list)
    for r in records:
        by_period[_period_key(r.timestamp)].append(r)

    results = []
    for period in sorted(by_period.keys()):
        recs = by_period[period]
        filled = [r for r in recs if r.filled]
        n_total = len(recs)
        n_filled = len(filled)
        n_as = sum(1 for r in filled if r.adverse_selected is True)

        pnl_30 = [r.post_fill_30s_pnl for r in filled if r.post_fill_30s_pnl is not None]
        import numpy as np

        results.append({
            "period": period,
            "total_cycles": n_total,
            "filled": n_filled,
            "fill_rate": n_filled / n_total if n_total > 0 else 0.0,
            "as_count": n_as,
            "as_rate": n_as / n_filled if n_filled > 0 else 0.0,
            "pnl_30s_mean": float(np.mean(pnl_30)) if pnl_30 else None,
        })
    return results


# ======================================================================
# Reporting
# ======================================================================


def _format_pnl(v: Optional[float]) -> str:
    """PnL を ±X.XXXbps 形式にフォーマット."""
    if v is None:
        return "N/A"
    return f"{v:+.3f}bps"


def _format_pct(v: float) -> str:
    return f"{v:.1%}"


def print_vg_report(result: dict) -> None:
    """A4: VG 効果の人間可読レポート."""
    print("=" * 60)
    print("A4: Volatility Guard Effectiveness Report")
    print("=" * 60)

    print(f"\n■ サイクル数:")
    print(f"  VG 発動    : {result['vg_total_cycles']:>5} cycles "
          f"(fill_rate: {_format_pct(result['vg_fill_rate'])})")
    print(f"  非 VG      : {result['non_vg_total_cycles']:>5} cycles "
          f"(fill_rate: {_format_pct(result['non_vg_fill_rate'])})")

    print(f"\n■ Filled サイクル比較:")
    vg = result["vg_filled"]
    non_vg = result["non_vg_filled"]
    header = f"{'':15} {'VG-active':>12} {'Non-VG':>12} {'Diff':>12}"
    print(header)
    print("-" * len(header))

    # PnL 30s
    vp = vg.get("pnl_30s_mean")
    nvp = non_vg.get("pnl_30s_mean")
    diff_str = _format_pnl(vp - nvp) if vp is not None and nvp is not None else "N/A"
    print(f"{'PnL 30s':15} {_format_pnl(vp):>12} {_format_pnl(nvp):>12} {diff_str:>12}")

    # PnL 120s
    vp120 = vg.get("pnl_120s_mean")
    nvp120 = non_vg.get("pnl_120s_mean")
    diff120 = _format_pnl(vp120 - nvp120) if vp120 is not None and nvp120 is not None else "N/A"
    print(f"{'PnL 120s':15} {_format_pnl(vp120):>12} {_format_pnl(nvp120):>12} {diff120:>12}")

    # AS rate
    va = vg.get("as_rate", 0)
    nva = non_vg.get("as_rate", 0)
    print(f"{'AS rate':15} {_format_pct(va):>12} {_format_pct(nva):>12} "
          f"{(va - nva):>+.1%} pt".rjust(12))

    # N
    print(f"{'N (filled)':15} {vg['n']:>12} {non_vg['n']:>12}")

    # Offset
    vg_off = result.get("vg_mean_offset")
    nvg_off = result.get("non_vg_mean_offset")
    if vg_off is not None and nvg_off is not None:
        print(f"{'Mean offset':15} {vg_off:>12.4f} {nvg_off:>12.4f} "
              f"{vg_off / nvg_off:>11.1f}x")

    print(f"\n■ 解釈: {result['interpretation']}")


def print_daily_report(daily: list[dict]) -> None:
    """B4: 日別トレンドの人間可読レポート."""
    print("\n" + "=" * 60)
    print("B4: Daily KPI Trend Report")
    print("=" * 60)

    header = (
        f"{'Date':12} {'Fill':>5} {'FillR':>6} {'AS':>4} {'AS%':>6} "
        f"{'PnL30':>10} {'PnL120':>10} "
        f"{'BuyAS%':>7} {'SellAS%':>8}"
    )
    print(header)
    print("-" * len(header))

    for d in daily:
        print(
            f"{d['date']:12} {d['filled']:>5} {_format_pct(d['fill_rate']):>6} "
            f"{d['as_count']:>4} {_format_pct(d['as_rate']):>6} "
            f"{_format_pnl(d['pnl_30s_mean']):>10} {_format_pnl(d['pnl_120s_mean']):>10} "
            f"{_format_pct(d['buy_as_rate']):>7} {_format_pct(d['sell_as_rate']):>8}"
        )

    # トレンド分析
    if len(daily) >= 3:
        as_rates = [d["as_rate"] for d in daily if d["filled"] > 0]
        if len(as_rates) >= 3:
            first_half = as_rates[:len(as_rates) // 2]
            second_half = as_rates[len(as_rates) // 2:]
            import numpy as np
            avg_first = np.mean(first_half)
            avg_second = np.mean(second_half)
            trend = "改善↓" if avg_second < avg_first else "悪化↑" if avg_second > avg_first else "横ばい"
            print(f"\n■ AS rate トレンド: 前半平均 {avg_first:.1%} → 後半平均 {avg_second:.1%} ({trend})")


def print_8h_report(periods: list[dict]) -> None:
    """B4 補足: 8h帯別トレンド."""
    print("\n" + "=" * 60)
    print("B4 Supplementary: 8-Hour Period KPI Breakdown")
    print("=" * 60)

    header = f"{'Period':20} {'Fill':>5} {'FillR':>6} {'AS':>4} {'AS%':>6} {'PnL30':>10}"
    print(header)
    print("-" * len(header))

    for p in periods:
        print(
            f"{p['period']:20} {p['filled']:>5} {_format_pct(p['fill_rate']):>6} "
            f"{p['as_count']:>4} {_format_pct(p['as_rate']):>6} "
            f"{_format_pnl(p['pnl_30s_mean']):>10}"
        )


# ======================================================================
# Main
# ======================================================================


def _load_all_records(results_dir: Path) -> list[FillRecord]:
    """JSONL ファイルから全レコードを読み込み."""
    all_records = load_fill_records_glob(results_dir, include_emergency=False)
    if not all_records:
        raise FileNotFoundError(f"No fill_records_*.jsonl in {results_dir}")
    return all_records


def main() -> None:
    parser = argparse.ArgumentParser(description="122# A4+B4 VG effectiveness and daily trend analysis")
    parser.add_argument(
        "--results-dir",
        default="results/v460/fill_test",
        help="fill_records JSONL ディレクトリ",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="fill_test.log パス (デフォルト: results-dir/logs/fill_test.log)",
    )
    parser.add_argument("--json", action="store_true", help="JSON 出力")
    parser.add_argument("--output", default=None, help="出力ファイルパス")
    parser.add_argument("--skip-vg", action="store_true", help="A4 VG 分析をスキップ")
    parser.add_argument("--skip-trend", action="store_true", help="B4 トレンド分析をスキップ")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    log_file = Path(args.log_file) if args.log_file else results_dir / "logs" / "fill_test.log"

    # Load records
    all_records = _load_all_records(results_dir)
    clean_records, quarantine = filter_clean_records(all_records)
    if not clean_records:
        # git_sha チェックが厳しい場合はバイパス
        clean_records, _ = filter_clean_records(all_records, require_git_sha=False)
        print(f"Loaded: {len(all_records)} total, {len(clean_records)} clean (git_sha bypass)")
    else:
        print(f"Loaded: {len(all_records)} total, {len(clean_records)} clean, "
              f"{len(quarantine)} quarantined")

    output_data: dict = {}

    # A4: VG Effectiveness
    if not args.skip_vg:
        activations = _parse_vg_activations(log_file)
        print(f"VG activations in log: {len(activations)}")

        vg_cycle_ids = _match_vg_to_records(activations, all_records)
        print(f"VG-matched cycles: {len(vg_cycle_ids)}")

        vg_result = analyze_vg_effectiveness(clean_records, vg_cycle_ids)
        output_data["vg_effectiveness"] = vg_result

        if not args.json:
            print_vg_report(vg_result)

    # B4: Daily Trend
    if not args.skip_trend:
        daily = analyze_daily_trend(clean_records)
        periods_8h = analyze_8h_trend(clean_records)
        output_data["daily_trend"] = daily
        output_data["8h_trend"] = periods_8h

        if not args.json:
            print_daily_report(daily)
            print_8h_report(periods_8h)

    # Output
    if args.json:
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(output_path, output_data, indent=2, ensure_ascii=False, default=str)
            print(f"Written to {args.output}")
        else:
            print(json.dumps(output_data, indent=2, ensure_ascii=False, default=str))
    elif args.output:
        # テキストレポートをファイルに保存
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            if not args.skip_vg and "vg_effectiveness" in output_data:
                print_vg_report(output_data["vg_effectiveness"])
            if not args.skip_trend and "daily_trend" in output_data:
                print_daily_report(output_data["daily_trend"])
                print_8h_report(output_data["8h_trend"])
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(buf.getvalue(), encoding="utf-8")
        print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
