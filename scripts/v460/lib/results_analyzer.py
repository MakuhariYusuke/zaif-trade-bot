"""
ResultsAnalyzer — 既存 fill_records からの Gate 判定 (--results-only).

119# God Object 分割: run_fill_test.py から結果分析ロジックを分離.
120# A2: run 別二系統分析 (all-run / current-run / trailing-N) 追加.
ztb/io/json_io の write_json (atomic write) を活用.
"""

from __future__ import annotations

import heapq
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ztb.io.json_io import write_json
from ztb.metrics.fill_quality import (
    FillMetrics,
    FillRecord,
    PnlAccumulator,
    RegimeMetrics,
    compute_fill_metrics,
    compute_regime_metrics,
    iter_fill_records_glob,
    g1_1_judgment,
    g1_1_quick_judgment,
    g1_2_full_judgment,
    partition_clean_records,
)

logger = logging.getLogger(__name__)


def run_results_only(results_dir: str, thresholds_path: str | None = None) -> dict:
    """既存の fill_records JSONL から G1.1 判定を実施.

    116# 拡張: 旧 g1_1_judgment に加え、quick/full 二段階判定も返す。
    """
    from scripts.v460.lib.config_loader import load_gate_thresholds

    records, quarantine = partition_clean_records(
        iter_fill_records_glob(results_dir),
    )
    if not records and not quarantine:
        logger.error(f"No fill records found in {results_dir}")
        return {"gate": "G1.1-exec", "gate_result": "NO_DATA", "error": "No records found"}

    # 047# A2: quarantine レコードを除外し clean のみで Gate 判定
    if quarantine:
        logger.warning(
            f"[results-only] {len(quarantine)} records quarantined, "
            f"using {len(records)} clean records for judgment"
        )
    if not records:
        logger.error("All records are quarantined")
        return {"gate": "G1.1-exec", "gate_result": "NO_DATA", "error": "All records quarantined"}

    metrics = compute_fill_metrics(records)
    gate_cfg = load_gate_thresholds()
    thresholds = gate_cfg.get("g1_1_exec", {})
    judgment = g1_1_judgment(metrics, thresholds)

    # 116# 二段階判定 (115# レビュー反映)
    quick_thresholds = gate_cfg.get("g1_1_quick_exec", {})
    full_thresholds = gate_cfg.get("g1_2_full_exec", {})
    quick_judgment = g1_1_quick_judgment(metrics, quick_thresholds)
    full_judgment = g1_2_full_judgment(metrics, full_thresholds)

    # 結果統合
    judgment["two_stage"] = {
        "g1_1_quick": quick_judgment,
        "g1_2_full": full_judgment,
    }

    logger.info(f"G1.1 (legacy) Result: {judgment['gate_result']}")
    logger.info(f"G1.1-quick  Result: {quick_judgment['gate_result']}")
    logger.info(f"G1.2-full   Result: {full_judgment['gate_result']}")

    for check_name, check_data in judgment["checks"].items():
        status = "✓" if check_data["pass"] else "✗"
        logger.info(f"  {status} {check_name}: {check_data['value']:.4f} (threshold: {check_data['threshold']})")

    # 116# quick/full の各チェックも表示
    logger.info("--- G1.1-quick checks ---")
    for check_name, check_data in quick_judgment["checks"].items():
        status = "✓" if check_data["pass"] else "✗"
        val = check_data.get("value", "N/A")
        thr = check_data.get("threshold", check_data.get("threshold_mean", "N/A"))
        val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
        logger.info(f"  {status} {check_name}: {val_str} (threshold: {thr})")

    logger.info("--- G1.2-full checks ---")
    for check_name, check_data in full_judgment["checks"].items():
        status = "✓" if check_data["pass"] else "✗"
        val = check_data.get("value", "N/A")
        thr = check_data.get("threshold", check_data.get("alpha", "N/A"))
        val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
        logger.info(f"  {status} {check_name}: {val_str} (threshold: {thr})")

    # 117# 3 系列同時出力 (115# Q10.6: 分母定義の不一致解消)
    logger.info("--- 3-series fill rate summary (117#) ---")
    logger.info(
        f"  overall  (raw):      {metrics.overall_fill_rate:.4f} "
        f"({metrics.filled_orders}/{metrics.total_orders})"
    )
    logger.info(
        f"  clean    (quarantine removed): "
        f"{metrics.filled_orders}/{len(records)} "
        f"= {metrics.filled_orders / len(records):.4f}"
    )
    logger.info(
        f"  attempted (skip_gate removed): {metrics.attempted_fill_rate:.4f} "
        f"({metrics.filled_orders}/{metrics.attempted_orders})"
    )

    # 117# cancel reason breakdown (115# Q10.6)
    if metrics.cancel_reason_breakdown:
        logger.info("--- Cancel reason breakdown (117#) ---")
        for reason, count in sorted(
            metrics.cancel_reason_breakdown.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            pct = count / metrics.cancelled_orders * 100 if metrics.cancelled_orders > 0 else 0
            logger.info(f"  {reason}: {count} ({pct:.1f}%)")

    # 120# A2: run 別二系統分析 (Simpson 逆転リスク対策)
    multi_track = compute_multi_track_analysis(records)
    log_multi_track_summary(multi_track)
    judgment["multi_track"] = multi_track

    # 169# B0: R2 gate metric 3-series 構造化 (分母定義の不一致解消)
    # raw = all_records (quarantine 含む), clean = quarantine 除外, attempted = skip_gate 除外
    # judgment JSON に明示的に 3 系列を格納し downstream の分母混在を防止
    _clean_fill_rate = metrics.filled_orders / len(records) if records else 0.0
    judgment["three_series"] = {
        "raw": {
            "fill_rate": round(metrics.overall_fill_rate, 6),
            "n_total": metrics.total_orders,
            "n_filled": metrics.filled_orders,
            "note": "全レコード (quarantine 含む) ベース",
        },
        "clean": {
            "fill_rate": round(_clean_fill_rate, 6),
            "n_total": len(records),
            "n_filled": metrics.filled_orders,
            "note": "quarantine 除外後ベース (Gate 判定はこの系列)",
        },
        "attempted": {
            "fill_rate": round(metrics.attempted_fill_rate, 6),
            "n_total": metrics.attempted_orders,
            "n_filled": metrics.filled_orders,
            "skip_gate_count": metrics.skip_gate_count,
            "skip_gate_ratio": round(metrics.skip_gate_ratio, 6),
            "note": "skip_gate 除外後ベース (純粋な発注判断の精度)",
        },
        "gate_basis": "clean",  # Gate 判定に使用する系列を明示
    }

    # 120# P2-1: FFD/VG/SG 寄与分解
    event_contrib = compute_event_contribution(records)
    log_event_contribution(event_contrib)
    judgment["event_contribution"] = event_contrib

    # 120# P2-2: regime 別比較基盤
    regime_breakdown = compute_regime_breakdown(records)
    log_regime_breakdown(regime_breakdown)
    judgment["regime_breakdown"] = regime_breakdown

    return judgment


def save_judgment(judgment: dict, output_path: str) -> None:
    """判定結果を JSON に保存 (ztb/io atomic write 活用).

    119# 新規: run_fill_test.py 内の手動 json.dump を write_json に統一.
    """
    write_json(output_path, judgment, indent=2, ensure_ascii=False)
    logger.info(f"Saved judgment to {output_path}")


# ======================================================================
# 120# A2: Run-level two-track analysis
# ======================================================================

_TRAILING_N: int = 200  # trailing window size for Gate 補助指標


def _metrics_summary(metrics: FillMetrics, n_records: int) -> dict:
    """FillMetrics → 簡潔な summary dict (ログ出力用)."""
    return {
        "n_total": metrics.total_orders,
        "n_filled": metrics.filled_orders,
        "fill_rate": round(metrics.overall_fill_rate, 4),
        "attempted_fill_rate": round(metrics.attempted_fill_rate, 4),
        "pnl_30s_mean": round(metrics.post_fill_30s_pnl_mean, 3),
        "pnl_60s_mean": round(metrics.post_fill_60s_pnl_mean, 3),
        "pnl_120s_mean": round(metrics.post_fill_120s_pnl_mean, 3),
        "as_ratio": round(metrics.adverse_selection_ratio, 4),
        "skip_gate_ratio": round(metrics.skip_gate_ratio, 4),
        "queue_wait_median": round(metrics.queue_wait_median_sec, 1),
    }


def compute_run_level_breakdown(
    records: list[FillRecord],
) -> dict[str, list[FillRecord]]:
    """レコードを run_id 別にグルーピング.

    run_id が空 / None のレコードは "legacy" にまとめる。
    """
    groups, _latest_run_id = _group_runs_with_latest_id(records)
    return groups


def _group_runs_with_latest_id(
    records: list[FillRecord],
) -> tuple[dict[str, list[FillRecord]], str | None]:
    """run_id 別グルーピングと最新 run_id 特定を単一パスで行う."""
    groups: dict[str, list[FillRecord]] = defaultdict(list)
    latest_run_id: str | None = None
    latest_ts = float("-inf")

    for record in records:
        key = record.run_id if (record.run_id and record.run_id.strip()) else "legacy"
        groups[key].append(record)
        if record.timestamp > latest_ts:
            latest_ts = record.timestamp
            latest_run_id = key

    return dict(groups), latest_run_id


def compute_multi_track_analysis(
    records: list[FillRecord],
    trailing_n: int = _TRAILING_N,
) -> dict:
    """120# A2: all-run / current-run / trailing-N の三系統で指標算出.

    Returns:
        {
            "all_run": { metrics_summary },
            "current_run": { "run_id": ..., metrics_summary },
            "trailing_N": { metrics_summary },
            "per_run": [ { "run_id": ..., "n": ..., "pnl_mean": ... }, ... ],
        }
    """
    result: dict = {}

    # --- all-run ---
    all_metrics = compute_fill_metrics(records)
    result["all_run"] = _metrics_summary(all_metrics, len(records))

    # --- per-run breakdown ---
    run_groups, latest_run_id = _group_runs_with_latest_id(records)
    per_run_list: list[dict] = []
    latest_run_entry: dict | None = None

    for run_id, run_records in run_groups.items():
        run_metrics = compute_fill_metrics(run_records)
        entry = {
            "run_id": run_id,
            **_metrics_summary(run_metrics, len(run_records)),
        }
        per_run_list.append(entry)
        if run_id == latest_run_id:
            latest_run_entry = dict(entry)

    # timestamp 降順ソート
    per_run_list.sort(
        key=lambda x: x.get("run_id", ""),
        reverse=True,
    )
    result["per_run"] = per_run_list

    # --- current-run ---
    if latest_run_entry is not None:
        result["current_run"] = latest_run_entry
    else:
        result["current_run"] = {"run_id": None, "n_total": 0}

    # --- trailing-N ---
    trailing_records = heapq.nlargest(
        trailing_n,
        records,
        key=lambda r: r.timestamp,
    )
    if trailing_records:
        trailing_metrics = compute_fill_metrics(trailing_records)
        result[f"trailing_{trailing_n}"] = _metrics_summary(
            trailing_metrics, len(trailing_records)
        )
    else:
        result[f"trailing_{trailing_n}"] = {"n_total": 0}

    return result


def log_multi_track_summary(analysis: dict) -> None:
    """120# A2: 三系統の分析結果をログに出力."""
    logger.info("=== 120# Multi-track Analysis ===")

    for track_name in ["all_run", "current_run", f"trailing_{_TRAILING_N}"]:
        track = analysis.get(track_name, {})
        run_id = track.get("run_id", "")
        label = f"{track_name}" + (f" ({run_id})" if run_id else "")
        n = track.get("n_total", 0)
        n_filled = track.get("n_filled", 0)
        pnl = track.get("pnl_30s_mean", 0)
        as_r = track.get("as_ratio", 0)
        fr = track.get("fill_rate", 0)
        logger.info(
            f"  {label}: n={n} filled={n_filled} "
            f"FR={fr:.3f} PnL={pnl:+.3f}bps AS={as_r:.3f}"
        )

    # per_run サマリ
    per_run = analysis.get("per_run", [])
    if per_run:
        logger.info("  --- Per-run breakdown ---")
        for entry in per_run:
            rid = entry.get("run_id", "?")
            n = entry.get("n_total", 0)
            pnl = entry.get("pnl_30s_mean", 0)
            as_r = entry.get("as_ratio", 0)
            logger.info(f"    {rid}: n={n} PnL={pnl:+.3f}bps AS={as_r:.3f}")


# ======================================================================
# 120# P2-1: VG/FFD/SG 寄与分解基盤
# ======================================================================


@dataclass
class _BinaryPnlSplit:
    """bool フラグで 2 群に分けた PnL 集計."""

    positive: PnlAccumulator = field(default_factory=PnlAccumulator)
    negative: PnlAccumulator = field(default_factory=PnlAccumulator)

    def add(self, flag: bool | None, pnl_bps: float) -> None:
        if flag is True:
            self.positive.add(pnl_bps)
        elif flag is False:
            self.negative.add(pnl_bps)

    def to_payload(
        self,
        *,
        positive_label: str,
        negative_label: str,
        require_both_for_delta: bool,
    ) -> dict[str, object]:
        delta: float | None
        if require_both_for_delta and (self.positive.count == 0 or self.negative.count == 0):
            delta = None
        else:
            delta = round(self.positive.mean_bps - self.negative.mean_bps, 3)
        return {
            positive_label: {
                "n": self.positive.count,
                "pnl_mean": round(self.positive.mean_bps, 3),
            },
            negative_label: {
                "n": self.negative.count,
                "pnl_mean": round(self.negative.mean_bps, 3),
            },
            "delta": delta,
        }


def compute_event_contribution(
    records: list[FillRecord],
) -> dict:
    """イベント (FFD/VG/SG) 発火時 vs 非発火時の PnL 差分を算出.

    120# P2-1: 「イベント発火後 N サイクル差分で因果確認」の基盤。
    各イベントの発火有無で filled レコードをグルーピングし、
    PnL mean の差分を返す。

    Returns:
        {
            "ffd": { "active": { "n", "pnl_mean" }, "inactive": { ... }, "delta": ... },
            "vg":  { "triggered": { ... }, "not_triggered": { ... }, "delta": ... },
            "sg":  { "high_prob": { ... }, "low_prob": { ... }, "delta": ... },
        }
    """
    result: dict = {}
    ffd_split = _BinaryPnlSplit()
    vg_split = _BinaryPnlSplit()
    sg_probs: list[float] = []
    sg_pnls: list[float] = []

    for record in records:
        if not record.filled or record.post_fill_30s_pnl is None:
            continue

        pnl_value = float(record.post_fill_30s_pnl)
        if not math.isfinite(pnl_value):
            continue

        ffd_split.add(record.ffd_boost_active, pnl_value)
        vg_split.add(record.vg_triggered, pnl_value)

        as_prob = record.skip_gate_as_prob
        if as_prob is None:
            continue
        prob_value = float(as_prob)
        if math.isfinite(prob_value):
            sg_probs.append(prob_value)
            sg_pnls.append(pnl_value)

    # --- FFD 寄与 ---
    result["ffd"] = ffd_split.to_payload(
        positive_label="active",
        negative_label="inactive",
        require_both_for_delta=True,
    )

    # --- VG 寄与 ---
    result["vg"] = vg_split.to_payload(
        positive_label="triggered",
        negative_label="not_triggered",
        require_both_for_delta=True,
    )

    # --- SG 寄与 (高 P(AS) skip vs 低 P(AS) pass) ---
    # skip された = 負の寄与を避けた、と仮定して
    # pass された中で as_prob 閾値前後の PnL 差を比較
    if sg_probs:
        sorted_probs = sorted(sg_probs)
        median_prob = sorted_probs[len(sorted_probs) // 2]

        sg_high = PnlAccumulator()
        sg_low = PnlAccumulator()
        for prob_value, pnl_value in zip(sg_probs, sg_pnls, strict=False):
            if prob_value >= median_prob:
                sg_high.add(pnl_value)
            else:
                sg_low.add(pnl_value)

        result["sg"] = {
            "high_prob": {
                "n": sg_high.count,
                "pnl_mean": round(sg_high.mean_bps, 3),
                "median_threshold": round(median_prob, 3),
            },
            "low_prob": {
                "n": sg_low.count,
                "pnl_mean": round(sg_low.mean_bps, 3),
            },
            "delta": round(sg_high.mean_bps - sg_low.mean_bps, 3),
        }
    else:
        result["sg"] = {"high_prob": {"n": 0}, "low_prob": {"n": 0}, "delta": None}

    return result


def log_event_contribution(contrib: dict) -> None:
    """120# P2-1: イベント寄与分解結果をログに出力."""
    logger.info("=== 120# Event Contribution Analysis ===")
    for event_name, data in contrib.items():
        if data.get("delta") is not None:
            delta = data["delta"]
            # 各イベントの群とN/PnLを表示
            groups = [k for k in data if k != "delta"]
            parts = []
            for g in groups:
                gd = data[g]
                if isinstance(gd, dict) and "n" in gd:
                    parts.append(f"{g}: n={gd['n']} PnL={gd.get('pnl_mean', 0):+.3f}")
            logger.info(f"  {event_name.upper()}: {' | '.join(parts)} | Δ={delta:+.3f}bps")
        else:
            logger.info(f"  {event_name.upper()}: insufficient data")


# ======================================================================
# 120# P2-2: Regime 条件別比較基盤
# ======================================================================


def compute_regime_breakdown(
    records: list[FillRecord],
) -> dict:
    """regime 別の PnL / AS / fill_rate を算出.

    120# P2-2: 「同一 regime 条件下の比較基盤」。
    run 混在でなくレジーム条件を揃えた比較を可能にする。
    ztb/metrics/fill_quality.compute_regime_metrics を活用し重複排除。

    Returns:
        {
            "trending": { "n_total", "n_filled", "pnl_mean", "as_ratio", "fill_rate" },
            "ranging":  { ... },
            ...
        }
    """
    regime_list: list[RegimeMetrics] = compute_regime_metrics(records)
    result: dict = {}
    for rm in regime_list:
        result[rm.regime] = {
            "n_total": rm.count,
            "n_filled": rm.filled,
            "fill_rate": round(rm.fill_rate, 4),
            "pnl_mean": round(rm.pnl_mean_bps, 3),
            "as_ratio": round(rm.as_ratio, 4),
        }
    return result


def log_regime_breakdown(breakdown: dict) -> None:
    """120# P2-2: レジーム別集計結果をログに出力."""
    logger.info("=== 120# Regime Breakdown ===")
    for regime, data in breakdown.items():
        n = data.get("n_total", 0)
        n_f = data.get("n_filled", 0)
        fr = data.get("fill_rate", 0)
        pnl = data.get("pnl_mean", 0)
        as_r = data.get("as_ratio", 0)
        logger.info(
            f"  {regime:12s}: n={n:4d} filled={n_f:4d} "
            f"FR={fr:.3f} PnL={pnl:+.3f}bps AS={as_r:.3f}"
        )
