"""306# 299# 観察比較の深堀り分析.

Block Bootstrap + Matched Pairs の全体結論は「差なし」だが、
構造的な改善点を見つけるためにレジーム別・時間帯別・offset段階別に分解する。
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ztb.metrics.fill_quality import format_utc_day, iter_fill_record_objects_glob
from ztb.utils.safety import safe_to_finite
from scripts.v460.lib.ab_judgment import (
    _block_bootstrap_mean_diff,
    _matched_temporal_comparison,
    _mann_whitney_u,
    _cliffs_delta,
    _wilcoxon_signed_rank,
    _benjamini_hochberg,
)


def load_records(results_dir: str = "results/v460/fill_test") -> list[dict]:
    return list(iter_fill_record_objects_glob(
        Path(results_dir), include_emergency=False,
    ))


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


def offset_stats(records: list[dict]) -> dict:
    offsets = []
    for r in records:
        v = safe_to_finite(r.get("effective_offset_used"))
        if v is not None:
            offsets.append(v)
    if not offsets:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0, "n": 0}
    arr = np.array(offsets)
    return {
        "mean": round(float(np.mean(arr)), 6),
        "std": round(float(np.std(arr)), 6),
        "p50": round(float(np.percentile(arr, 50)), 6),
        "p90": round(float(np.percentile(arr, 90)), 6),
        "n": len(offsets),
    }


def ev_stats(records: list[dict]) -> dict:
    evs = []
    for r in records:
        v = safe_to_finite(r.get("ev_score_pretrade"))
        if v is not None:
            evs.append(v)
    if not evs:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    arr = np.array(evs)
    return {
        "mean": round(float(np.mean(arr)), 6),
        "std": round(float(np.std(arr)), 6),
        "n": len(evs),
    }


def hourly_analysis(records: list[dict], side: str) -> list[dict]:
    """UTC 時間帯別分析."""
    from datetime import datetime, timezone
    hourly: dict[int, list[dict]] = defaultdict(list)
    for r in extract_filled(records, side=side):
        ts = safe_to_finite(r.get("timestamp"))
        if ts is None:
            continue
        hour = datetime.fromtimestamp(ts, tz=timezone.utc).hour
        hourly[hour].append(r)

    results = []
    for hour in sorted(hourly.keys()):
        recs = hourly[hour]
        pnl = pnl_array(recs)
        results.append({
            "hour_utc": hour,
            "n_filled": len(recs),
            "avg_pnl30": round(float(np.mean(pnl)), 4) if len(pnl) > 0 else None,
            "std_pnl30": round(float(np.std(pnl)), 4) if len(pnl) > 0 else None,
            "p10": round(float(np.percentile(pnl, 10)), 4) if len(pnl) >= 5 else None,
            "as_rate": round(as_rate(recs), 4),
            "profitable_rate": round(float(np.mean(pnl > 0)), 4) if len(pnl) > 0 else None,
        })
    return results


def matched_regime_analysis(records: list[dict]) -> list[dict]:
    """レジーム別 matched temporal comparison."""
    regimes = set()
    for r in records:
        reg = str(r.get("regime") or "none")
        if reg != "none":
            regimes.add(reg)

    results = []
    for regime in sorted(regimes):
        sell_recs = [r for r in records if r.get("side") == "sell"
                     and str(r.get("regime") or "none") == regime]
        buy_recs = [r for r in records if r.get("side") == "buy"
                    and str(r.get("regime") or "none") == regime]

        sell_filled = extract_filled(sell_recs, side="sell")
        buy_filled = extract_filled(buy_recs, side="buy")

        sell_pnl = pnl_array(sell_filled)
        buy_pnl = pnl_array(buy_filled)

        entry: dict = {
            "regime": regime,
            "sell_n": len(sell_filled),
            "buy_n": len(buy_filled),
            "sell_avg_pnl30": round(float(np.mean(sell_pnl)), 4) if len(sell_pnl) > 0 else None,
            "buy_avg_pnl30": round(float(np.mean(buy_pnl)), 4) if len(buy_pnl) > 0 else None,
            "sell_as_rate": round(as_rate(sell_filled), 4),
            "buy_as_rate": round(as_rate(buy_filled), 4),
            "sell_offset": offset_stats(sell_filled),
            "buy_offset": offset_stats(buy_filled),
        }

        # matched comparison per regime
        n_pairs, diff, ci_lo, ci_hi, p = _matched_temporal_comparison(
            sell_recs, buy_recs, max_gap_sec=600.0,
        )
        entry["matched_n_pairs"] = n_pairs
        entry["matched_diff"] = round(diff, 4) if diff is not None else None
        entry["matched_ci"] = ([round(ci_lo, 4), round(ci_hi, 4)]
                               if ci_lo is not None else None)
        entry["matched_p"] = round(p, 4) if p is not None else None

        # block bootstrap per regime
        if len(sell_pnl) >= 20 and len(buy_pnl) >= 20:
            diff_b, ci_lo_b, ci_hi_b, p_b = _block_bootstrap_mean_diff(
                sell_pnl, buy_pnl, n_bootstrap=2000,
            )
            entry["bootstrap_diff"] = round(diff_b, 4)
            entry["bootstrap_ci"] = [round(ci_lo_b, 4), round(ci_hi_b, 4)]
            entry["bootstrap_p"] = round(p_b, 4)

        # Mann-Whitney per regime
        if len(sell_pnl) >= 10 and len(buy_pnl) >= 10:
            _, mw_p = _mann_whitney_u(buy_pnl, sell_pnl)
            cd_val, cd_interp = _cliffs_delta(buy_pnl, sell_pnl)
            entry["mw_p"] = round(mw_p, 4) if math.isfinite(mw_p) else None
            entry["cliffs_delta"] = round(cd_val, 4) if math.isfinite(cd_val) else None
            entry["cliffs_interp"] = cd_interp

        results.append(entry)

    # BH FDR across regime p-values
    p_values = []
    regime_names = []
    for r in results:
        if r.get("matched_p") is not None:
            p_values.append(r["matched_p"])
            regime_names.append(r["regime"])
    if p_values:
        bh_sig = _benjamini_hochberg(p_values)
        for i, name in enumerate(regime_names):
            for r in results:
                if r["regime"] == name:
                    r["bh_fdr_significant"] = bh_sig[i]

    return results


def as_deep_dive(records: list[dict]) -> dict:
    """Adverse Selection 深堀り: AS 発生時の offset, EV, pnl."""
    result = {}
    for side in ["sell", "buy"]:
        filled = extract_filled(records, side=side)
        as_recs = [r for r in filled if r.get("adverse_selected")]
        non_as_recs = [r for r in filled if not r.get("adverse_selected")]

        as_pnl = pnl_array(as_recs)
        non_as_pnl = pnl_array(non_as_recs)

        result[side] = {
            "n_as": len(as_recs),
            "n_non_as": len(non_as_recs),
            "as_avg_pnl30": round(float(np.mean(as_pnl)), 4) if len(as_pnl) > 0 else None,
            "non_as_avg_pnl30": round(float(np.mean(non_as_pnl)), 4) if len(non_as_pnl) > 0 else None,
            "as_offset": offset_stats(as_recs),
            "non_as_offset": offset_stats(non_as_recs),
            "as_ev": ev_stats(as_recs),
            "non_as_ev": ev_stats(non_as_recs),
        }

        # AS vs non-AS bootstrap
        if len(as_pnl) >= 10 and len(non_as_pnl) >= 10:
            diff, ci_lo, ci_hi, p = _block_bootstrap_mean_diff(as_pnl, non_as_pnl)
            result[side]["as_vs_nonas_diff"] = round(diff, 4)
            result[side]["as_vs_nonas_ci"] = [round(ci_lo, 4), round(ci_hi, 4)]
            result[side]["as_vs_nonas_p"] = round(p, 4)

    return result


def offset_pnl_correlation(records: list[dict]) -> dict:
    """゛ｆset と pnl30 の相関分析.

    309# 修正: effective_offset_ratio → effective_offset_used (307# F1)。
    """
    result = {}
    for side in ["sell", "buy"]:
        filled = extract_filled(records, side=side)
        offsets = []
        pnls = []
        for r in filled:
            o = safe_to_finite(r.get("effective_offset_used"))
            p = safe_to_finite(r.get("post_fill_30s_pnl"))
            if o is not None and p is not None:
                offsets.append(o)
                pnls.append(p)
        if len(offsets) < 10:
            result[side] = {"n": len(offsets), "corr": None}
            continue
        o_arr = np.array(offsets)
        p_arr = np.array(pnls)
        # Pearson correlation
        corr = float(np.corrcoef(o_arr, p_arr)[0, 1])
        # offset quintile analysis
        quintiles = np.percentile(o_arr, [20, 40, 60, 80])
        bins = np.digitize(o_arr, quintiles)
        quintile_stats = []
        for q in range(5):
            mask = bins == q
            q_pnl = p_arr[mask]
            if len(q_pnl) > 0:
                quintile_stats.append({
                    "quintile": q + 1,
                    "n": int(np.sum(mask)),
                    "offset_range": [
                        round(float(o_arr[mask].min()), 6),
                        round(float(o_arr[mask].max()), 6),
                    ],
                    "avg_pnl30": round(float(np.mean(q_pnl)), 4),
                    "as_rate": round(float(np.mean(
                        [r.get("adverse_selected", False)
                         for r, b in zip(filled, bins) if b == q]
                    )), 4),
                })
        result[side] = {
            "n": len(offsets),
            "pearson_corr": round(corr, 4) if math.isfinite(corr) else None,
            "quintile_analysis": quintile_stats,
        }
    return result


def weekly_trend(records: list[dict]) -> list[dict]:
    """週次トレンド分析 (sell vs buy)."""
    from datetime import datetime, timezone
    weekly: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if not r.get("filled"):
            continue
        ts = safe_to_finite(r.get("timestamp"))
        if ts is None:
            continue
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        week = dt.strftime("%Y-W%W")
        side = r.get("side", "unknown")
        weekly[week][side].append(r)

    results = []
    for week in sorted(weekly.keys()):
        entry: dict = {"week": week}
        for side in ["sell", "buy"]:
            recs = weekly[week].get(side, [])
            pnl = pnl_array(recs)
            entry[f"{side}_n"] = len(recs)
            entry[f"{side}_avg_pnl30"] = round(float(np.mean(pnl)), 4) if len(pnl) > 0 else None
            entry[f"{side}_as_rate"] = round(as_rate(recs), 4) if recs else None
            entry[f"{side}_profitable"] = round(float(np.mean(pnl > 0)), 4) if len(pnl) > 0 else None
        results.append(entry)
    return results


def fill_speed_analysis(records: list[dict]) -> dict:
    """約定速度分析: queue_wait_sec ベース.

    309# 修正: fill_timestamp → queue_wait_sec (307# F1)。
    FillRecord の queue_wait_sec が約定時の待機秒数。
    """
    result = {}
    for side in ["sell", "buy"]:
        filled = extract_filled(records, side=side)
        durations = []
        for r in filled:
            wait = safe_to_finite(r.get("queue_wait_sec"))
            if wait is not None and wait > 0:
                durations.append(wait)
        if not durations:
            result[side] = {"n": 0}
            continue
        arr = np.array(durations)
        result[side] = {
            "n": len(durations),
            "mean_sec": round(float(np.mean(arr)), 2),
            "median_sec": round(float(np.median(arr)), 2),
            "p90_sec": round(float(np.percentile(arr, 90)), 2),
            "fast_fill_rate": round(float(np.mean(arr < 30.0)), 4),
        }
    return result


def pnl_distribution_shape(records: list[dict]) -> dict:
    """PnL30 分布の形状分析 (skewness, kurtosis)."""
    result = {}
    for side in ["sell", "buy"]:
        pnl = pnl_array(extract_filled(records, side=side))
        if len(pnl) < 10:
            result[side] = {"n": len(pnl)}
            continue
        mean = float(np.mean(pnl))
        std = float(np.std(pnl))
        if std <= 0:
            result[side] = {"n": len(pnl), "mean": mean, "std": 0.0}
            continue
        # Skewness (歪度)
        skew = float(np.mean(((pnl - mean) / std) ** 3))
        # Kurtosis (尖度, excess)
        kurt = float(np.mean(((pnl - mean) / std) ** 4)) - 3.0
        result[side] = {
            "n": len(pnl),
            "mean": round(mean, 4),
            "std": round(std, 4),
            "skewness": round(skew, 4),
            "kurtosis_excess": round(kurt, 4),
            "p05": round(float(np.percentile(pnl, 5)), 4),
            "p10": round(float(np.percentile(pnl, 10)), 4),
            "p25": round(float(np.percentile(pnl, 25)), 4),
            "p50": round(float(np.percentile(pnl, 50)), 4),
            "p75": round(float(np.percentile(pnl, 75)), 4),
            "p90": round(float(np.percentile(pnl, 90)), 4),
            "p95": round(float(np.percentile(pnl, 95)), 4),
        }
    return result


def decision_path_analysis(records: list[dict]) -> dict:
    """309# 新規: decision_path 別分析 (307# F2: 交絡分離).

    alpha trade と repair trade を分離して PnL / AS 率を比較。
    balance_forced_switch も分離。
    """
    result = {}
    for side in ["sell", "buy"]:
        filled = extract_filled(records, side=side)
        # decision_path 別
        path_groups: dict[str, list[dict]] = defaultdict(list)
        for r in filled:
            path = r.get("decision_path") or "unknown"
            path_groups[path].append(r)
        path_stats = {}
        for path, recs in sorted(path_groups.items()):
            pnl = pnl_array(recs)
            path_stats[path] = {
                "n": len(recs),
                "avg_pnl30": round(float(np.mean(pnl)), 4) if len(pnl) > 0 else None,
                "as_rate": round(as_rate(recs), 4),
            }
        # balance_forced_switch 分離
        forced = [r for r in filled if r.get("balance_forced_switch")]
        normal = [r for r in filled if not r.get("balance_forced_switch")]
        f_pnl = pnl_array(forced)
        n_pnl = pnl_array(normal)
        result[side] = {
            "by_decision_path": path_stats,
            "forced_switch": {
                "n": len(forced),
                "avg_pnl30": round(float(np.mean(f_pnl)), 4) if len(f_pnl) > 0 else None,
                "as_rate": round(as_rate(forced), 4),
            },
            "normal": {
                "n": len(normal),
                "avg_pnl30": round(float(np.mean(n_pnl)), 4) if len(n_pnl) > 0 else None,
                "as_rate": round(as_rate(normal), 4),
            },
        }
    return result


def new_observability_analysis(records: list[dict]) -> dict:
    """309# 新規: 306# で追加した可観測性フィールドの分析.

    offset_stages, queue_depth_ahead, queue_fill_prob_est,
    microprice_bias_bps を活用 (307# F1/F7 対応)。
    """
    result = {}
    for side in ["sell", "buy"]:
        filled = extract_filled(records, side=side)

        # offset_stages 分析 (JSON 展開)
        stage_contributions: dict[str, list[float]] = defaultdict(list)
        n_with_stages = 0
        for r in filled:
            raw = r.get("offset_stages")
            if raw and isinstance(raw, str):
                try:
                    stages = json.loads(raw)
                    n_with_stages += 1
                    for k, v in stages.items():
                        val = safe_to_finite(v)
                        if val is not None:
                            stage_contributions[k].append(val)
                except (json.JSONDecodeError, TypeError):
                    pass
        stage_stats = {}
        for stage, vals in sorted(stage_contributions.items()):
            arr = np.array(vals)
            stage_stats[stage] = {
                "n": len(vals),
                "mean": round(float(np.mean(arr)), 6),
                "std": round(float(np.std(arr)), 6),
                "p50": round(float(np.median(arr)), 6),
            }

        # queue_depth_ahead 分析
        queue_depths = []
        for r in filled:
            v = safe_to_finite(r.get("queue_depth_ahead"))
            if v is not None:
                queue_depths.append(v)
        q_arr = np.array(queue_depths) if queue_depths else np.array([])

        # queue_fill_prob_est 分析
        fill_probs = []
        for r in filled:
            v = safe_to_finite(r.get("queue_fill_prob_est"))
            if v is not None:
                fill_probs.append(v)
        fp_arr = np.array(fill_probs) if fill_probs else np.array([])

        # microprice_bias_bps 分析
        mp_biases = []
        for r in filled:
            v = safe_to_finite(r.get("microprice_bias_bps"))
            if v is not None:
                mp_biases.append(v)
        mp_arr = np.array(mp_biases) if mp_biases else np.array([])

        result[side] = {
            "offset_stages": {
                "n_records_with_stages": n_with_stages,
                "per_stage": stage_stats,
            },
            "queue_depth_ahead": {
                "n": len(queue_depths),
                "mean": round(float(np.mean(q_arr)), 6) if len(q_arr) > 0 else None,
                "p50": round(float(np.median(q_arr)), 6) if len(q_arr) > 0 else None,
                "p90": round(float(np.percentile(q_arr, 90)), 6) if len(q_arr) >= 5 else None,
            },
            "queue_fill_prob_est": {
                "n": len(fill_probs),
                "mean": round(float(np.mean(fp_arr)), 6) if len(fp_arr) > 0 else None,
                "p50": round(float(np.median(fp_arr)), 6) if len(fp_arr) > 0 else None,
            },
            "microprice_bias_bps": {
                "n": len(mp_biases),
                "mean": round(float(np.mean(mp_arr)), 4) if len(mp_arr) > 0 else None,
                "std": round(float(np.std(mp_arr)), 4) if len(mp_arr) > 0 else None,
                "p10": round(float(np.percentile(mp_arr, 10)), 4) if len(mp_arr) >= 5 else None,
                "p90": round(float(np.percentile(mp_arr, 90)), 4) if len(mp_arr) >= 5 else None,
            },
        }
    return result


def spread_as_cost_decomposition(records: list[dict]) -> dict:
    """310# E: Spread Capture / AS Cost 分解 (307# F7).

    Maker の理論的 PnL 分解:
      spread_capture_bps = spread_bps × effective_offset_used
        → offset 分だけ板の内側に入った理論的収益 (maker edge)
      realized_pnl = post_fill_30s_pnl (bps)
      as_cost = spread_capture - realized_pnl
        → spread_capture を食い潰した逆選択コスト
    """
    result: dict = {}
    for side in ["sell", "buy"]:
        filled = extract_filled(records, side=side)
        sc_list: list[float] = []
        pnl_list: list[float] = []
        as_cost_list: list[float] = []
        for r in filled:
            offset = safe_to_finite(r.get("effective_offset_used"))
            pnl = safe_to_finite(r.get("post_fill_30s_pnl"))
            spread_bps = safe_to_finite(r.get("spread_bps"))
            if offset is not None and pnl is not None and spread_bps is not None and spread_bps > 0:
                # spread_capture = how much of the spread we captured by placing offset inside
                sc_bps = spread_bps * offset
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
            "efficiency": round(
                float(np.mean(pnl_arr) / np.mean(sc_arr)), 4
            ) if n > 0 and np.mean(sc_arr) != 0 else None,
        }
    return result


def none_regime_analysis(records: list[dict]) -> dict:
    """310# D: None regime 観測性分析 (307# F5).

    regime="none" のレコードを分類:
      - warmup: 初期レコード (timestamp 順で先頭 N レコード)
      - true_none: regime detector が実際に none を返している
    """
    filled = [r for r in records if r.get("filled")]
    none_records = [r for r in filled if str(r.get("regime") or "none") == "none"]
    non_none = [r for r in filled if str(r.get("regime") or "none") != "none"]

    # PnL 比較
    none_pnls = [
        safe_to_finite(r.get("post_fill_30s_pnl"))
        for r in none_records
        if safe_to_finite(r.get("post_fill_30s_pnl")) is not None
    ]
    non_none_pnls = [
        safe_to_finite(r.get("post_fill_30s_pnl"))
        for r in non_none
        if safe_to_finite(r.get("post_fill_30s_pnl")) is not None
    ]
    none_as_count = sum(
        1 for r in none_records
        if r.get("post_fill_30s_pnl") is not None
        and float(r["post_fill_30s_pnl"]) < 0
    )
    none_pnl_arr = np.array(none_pnls) if none_pnls else np.array([])
    non_none_pnl_arr = np.array(non_none_pnls) if non_none_pnls else np.array([])

    # 時間帯分布
    hour_dist: dict[int, int] = defaultdict(int)
    for r in none_records:
        ts = r.get("timestamp")
        if ts:
            from datetime import datetime, timezone
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                hour_dist[dt.hour] += 1
            except (ValueError, TypeError):
                pass

    return {
        "total_filled": len(filled),
        "none_regime_count": len(none_records),
        "none_regime_rate": round(len(none_records) / len(filled), 4) if filled else 0,
        "none_pnl": {
            "n": len(none_pnls),
            "mean": round(float(np.mean(none_pnl_arr)), 4) if none_pnls else None,
            "as_rate": round(none_as_count / len(none_pnls), 4) if none_pnls else None,
        },
        "non_none_pnl": {
            "n": len(non_none_pnls),
            "mean": round(float(np.mean(non_none_pnl_arr)), 4) if non_none_pnls else None,
        },
        "none_hour_distribution": dict(sorted(hour_dist.items())),
    }


def main() -> None:
    print("=" * 70)
    print("  310# 306# 深堀り分析 (310# 設計改修版)")
    print("=" * 70)

    records = load_records()
    print(f"\n  Total records: {len(records)}")
    filled = [r for r in records if r.get("filled")]
    print(f"  Filled records: {len(filled)}")

    # 1. レジーム別 matched + bootstrap 比較
    print("\n" + "=" * 70)
    print("  §1 レジーム別 Matched & Bootstrap 比較 (BH FDR 補正付)")
    print("=" * 70)
    regime_results = matched_regime_analysis(records)
    for r in regime_results:
        print(f"\n  [{r['regime']}]  sell(n={r['sell_n']}) vs buy(n={r['buy_n']})")
        if r.get("sell_avg_pnl30") is not None:
            print(f"    PnL30: sell={r['sell_avg_pnl30']:+.4f}  buy={r['buy_avg_pnl30']:+.4f}")
        print(f"    AS rate: sell={r['sell_as_rate']:.4f}  buy={r['buy_as_rate']:.4f}")
        if r.get("matched_diff") is not None:
            ci = r['matched_ci']
            bh = " (BH-sig)" if r.get("bh_fdr_significant") else ""
            print(f"    Matched(n={r['matched_n_pairs']}): diff={r['matched_diff']:+.4f}, "
                  f"CI=[{ci[0]:+.4f},{ci[1]:+.4f}], p={r['matched_p']:.4f}{bh}")
        if r.get("bootstrap_diff") is not None:
            bci = r["bootstrap_ci"]
            print(f"    Bootstrap: diff={r['bootstrap_diff']:+.4f}, "
                  f"CI=[{bci[0]:+.4f},{bci[1]:+.4f}], p={r['bootstrap_p']:.4f}")
        if r.get("cliffs_delta") is not None:
            print(f"    MannWhitney: p={r['mw_p']:.4f}, "
                  f"Cliff's δ={r['cliffs_delta']:+.4f} ({r['cliffs_interp']})")
        o_s = r.get("sell_offset", {})
        o_b = r.get("buy_offset", {})
        if o_s.get("n", 0) > 0:
            print(f"    Offset: sell(mean={o_s['mean']:.6f}, p90={o_s['p90']:.6f}) "
                  f"buy(mean={o_b['mean']:.6f}, p90={o_b['p90']:.6f})")

    # 2. AS 深堀り
    print("\n" + "=" * 70)
    print("  §2 Adverse Selection 深堀り (AS vs non-AS)")
    print("=" * 70)
    as_result = as_deep_dive(records)
    for side in ["sell", "buy"]:
        d = as_result[side]
        print(f"\n  [{side.upper()}]  AS={d['n_as']} / non-AS={d['n_non_as']}")
        print(f"    PnL30: AS={d['as_avg_pnl30']}  non-AS={d['non_as_avg_pnl30']}")
        print(f"    Offset: AS(mean={d['as_offset']['mean']:.6f}) "
              f"non-AS(mean={d['non_as_offset']['mean']:.6f})")
        print(f"    EV: AS(mean={d['as_ev']['mean']:.6f}) "
              f"non-AS(mean={d['non_as_ev']['mean']:.6f})")
        if "as_vs_nonas_diff" in d:
            print(f"    AS-nonAS Bootstrap: diff={d['as_vs_nonas_diff']:+.4f}, "
                  f"CI={d['as_vs_nonas_ci']}, p={d['as_vs_nonas_p']:.4f}")

    # 3. Offset vs PnL 相関
    print("\n" + "=" * 70)
    print("  §3 Offset-PnL 相関 + Quintile 分析")
    print("=" * 70)
    corr_result = offset_pnl_correlation(records)
    for side in ["sell", "buy"]:
        d = corr_result[side]
        print(f"\n  [{side.upper()}]  n={d['n']}, Pearson r={d.get('pearson_corr')}")
        for q in d.get("quintile_analysis", []):
            print(f"    Q{q['quintile']}(n={q['n']}): offset=[{q['offset_range'][0]:.6f},"
                  f"{q['offset_range'][1]:.6f}], "
                  f"pnl30={q['avg_pnl30']:+.4f}, AS={q['as_rate']:.4f}")

    # 4. PnL 分布形状
    print("\n" + "=" * 70)
    print("  §4 PnL30 分布形状 (skewness / kurtosis)")
    print("=" * 70)
    shape = pnl_distribution_shape(records)
    for side in ["sell", "buy"]:
        d = shape[side]
        if "skewness" in d:
            print(f"\n  [{side.upper()}]  n={d['n']}")
            print(f"    mean={d['mean']:+.4f}, std={d['std']:.4f}")
            print(f"    skewness={d['skewness']:+.4f} (>0=右裾重, <0=左裾重)")
            print(f"    kurtosis_excess={d['kurtosis_excess']:+.4f} (>0=尖った, <0=平坦)")
            print(f"    percentiles: p05={d['p05']:+.4f} p10={d['p10']:+.4f} "
                  f"p25={d['p25']:+.4f} p50={d['p50']:+.4f} "
                  f"p75={d['p75']:+.4f} p90={d['p90']:+.4f} p95={d['p95']:+.4f}")

    # 5. 時間帯別分析
    print("\n" + "=" * 70)
    print("  §5 UTC 時間帯別 PnL30 + AS率")
    print("=" * 70)
    for side in ["sell", "buy"]:
        print(f"\n  [{side.upper()}]")
        hourly = hourly_analysis(records, side)
        for h in hourly:
            avg = f"{h['avg_pnl30']:+.4f}" if h['avg_pnl30'] is not None else "N/A"
            p10 = f"{h['p10']:+.4f}" if h['p10'] is not None else "N/A"
            print(f"    {h['hour_utc']:02d}h: n={h['n_filled']:3d}, "
                  f"pnl30={avg}, p10={p10}, AS={h['as_rate']:.4f}")

    # 6. 週次トレンド
    print("\n" + "=" * 70)
    print("  §6 週次トレンド")
    print("=" * 70)
    weekly = weekly_trend(records)
    for w in weekly:
        s_avg = f"{w['sell_avg_pnl30']:+.4f}" if w.get('sell_avg_pnl30') is not None else "N/A"
        b_avg = f"{w['buy_avg_pnl30']:+.4f}" if w.get('buy_avg_pnl30') is not None else "N/A"
        print(f"  {w['week']}: sell(n={w['sell_n']}, pnl={s_avg}, AS={w.get('sell_as_rate',0):.3f}) "
              f"buy(n={w['buy_n']}, pnl={b_avg}, AS={w.get('buy_as_rate',0):.3f})")

    # 7. 約定速度
    print("\n" + "=" * 70)
    print("  §7 約定速度分析 (queue_wait_sec)")
    print("=" * 70)
    speed = fill_speed_analysis(records)
    for side in ["sell", "buy"]:
        d = speed[side]
        if d.get("n", 0) > 0:
            print(f"  [{side.upper()}] n={d['n']}, mean={d['mean_sec']:.1f}s, "
                  f"median={d['median_sec']:.1f}s, p90={d['p90_sec']:.1f}s, "
                  f"fast_fill(<30s)={d['fast_fill_rate']:.4f}")
        else:
            print(f"  [{side.upper()}] n=0")

    # 8. 交絡分離: decision_path / balance_forced_switch (307# F2)
    print("\n" + "=" * 70)
    print("  §8 交絡分離: decision_path / balance_forced_switch (307# F2)")
    print("=" * 70)
    dp_result = decision_path_analysis(records)
    for side in ["sell", "buy"]:
        d = dp_result[side]
        print(f"\n  [{side.upper()}]")
        print(f"    forced_switch: n={d['forced_switch']['n']}, "
              f"pnl30={d['forced_switch']['avg_pnl30']}, AS={d['forced_switch']['as_rate']:.4f}")
        print(f"    normal: n={d['normal']['n']}, "
              f"pnl30={d['normal']['avg_pnl30']}, AS={d['normal']['as_rate']:.4f}")
        for path, stats in d["by_decision_path"].items():
            print(f"    path={path}: n={stats['n']}, "
                  f"pnl30={stats['avg_pnl30']}, AS={stats['as_rate']:.4f}")

    # 9. 新可観測性フィールド (309# / 307# F1)
    print("\n" + "=" * 70)
    print("  §9 306# 新可観測性 (offset_stages / queue / microprice)")
    print("=" * 70)
    obs_result = new_observability_analysis(records)
    for side in ["sell", "buy"]:
        d = obs_result[side]
        print(f"\n  [{side.upper()}]")
        stages = d["offset_stages"]
        print(f"    offset_stages: {stages['n_records_with_stages']} records")
        for stage, st in stages["per_stage"].items():
            print(f"      {stage}: n={st['n']}, mean={st['mean']:.6f}, p50={st['p50']:.6f}")
        q = d["queue_depth_ahead"]
        print(f"    queue_depth_ahead: n={q['n']}, mean={q.get('mean')}, p90={q.get('p90')}")
        fp = d["queue_fill_prob_est"]
        print(f"    queue_fill_prob_est: n={fp['n']}, mean={fp.get('mean')}, p50={fp.get('p50')}")
        mp = d["microprice_bias_bps"]
        print(f"    microprice_bias: n={mp['n']}, mean={mp.get('mean')}, "
              f"std={mp.get('std')}, p10={mp.get('p10')}, p90={mp.get('p90')}")

    # 10. Spread Capture / AS Cost 分解 (307# F7)
    print("\n" + "=" * 70)
    print("  §10 Spread Capture / AS Cost 分解 (307# F7)")
    print("=" * 70)
    decomp = spread_as_cost_decomposition(records)
    for side in ["sell", "buy"]:
        d = decomp[side]
        print(f"\n  [{side.upper()}] n={d['n']}")
        sc = d["spread_capture_bps"]
        print(f"    spread_capture: mean={sc['mean']} bps, p50={sc['p50']} bps")
        rp = d["realized_pnl_bps"]
        print(f"    realized_pnl:   mean={rp['mean']} bps, p50={rp['p50']} bps")
        ac = d["as_cost_bps"]
        print(f"    AS cost:        mean={ac['mean']} bps, p50={ac['p50']} bps, p90={ac.get('p90')} bps")
        print(f"    efficiency:     {d['efficiency']}")

    # 11. None regime 分析 (307# F5)
    print("\n" + "=" * 70)
    print("  §11 None Regime 分析 (307# F5)")
    print("=" * 70)
    none_result = none_regime_analysis(records)
    print(f"  Total filled: {none_result['total_filled']}")
    print(f"  None regime: {none_result['none_regime_count']} "
          f"({none_result['none_regime_rate']:.2%})")
    np_stats = none_result["none_pnl"]
    print(f"  None PnL: n={np_stats['n']}, mean={np_stats['mean']}, AS={np_stats['as_rate']}")
    nnp = none_result["non_none_pnl"]
    print(f"  Non-none PnL: n={nnp['n']}, mean={nnp['mean']}")
    if none_result["none_hour_distribution"]:
        print(f"  Hour distribution: {none_result['none_hour_distribution']}")

    # 12. 全結果 JSON 出力
    full_result = {
        "regime_matched": regime_results,
        "as_deep_dive": as_result,
        "offset_pnl_corr": corr_result,
        "pnl_shape": shape,
        "weekly_trend": weekly,
        "fill_speed": speed,
        "decision_path": dp_result,
        "new_observability": obs_result,
        "spread_as_decomposition": decomp,
        "none_regime": none_result,
    }
    out_path = Path("analysis_results/306_deep_dive.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  JSON 出力: {out_path}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
