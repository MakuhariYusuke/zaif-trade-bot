"""
ResultsAnalyzer — 既存 fill_records からの Gate 判定 (--results-only).

119# God Object 分割: run_fill_test.py から結果分析ロジックを分離.
ztb/io/json_io の write_json (atomic write) を活用.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ztb.io.json_io import write_json
from ztb.metrics.fill_quality import (
    compute_fill_metrics,
    filter_clean_records,
    g1_1_judgment,
    g1_1_quick_judgment,
    g1_2_full_judgment,
    load_fill_records_glob,
)

logger = logging.getLogger(__name__)


def run_results_only(results_dir: str, thresholds_path: str | None = None) -> dict:
    """既存の fill_records JSONL から G1.1 判定を実施.

    116# 拡張: 旧 g1_1_judgment に加え、quick/full 二段階判定も返す。
    """
    from scripts.v460.lib.config_loader import load_gate_thresholds

    all_records = load_fill_records_glob(results_dir)
    if not all_records:
        logger.error(f"No fill records found in {results_dir}")
        return {"gate": "G1.1-exec", "gate_result": "NO_DATA", "error": "No records found"}

    # 047# A2: quarantine レコードを除外し clean のみで Gate 判定
    records, quarantine = filter_clean_records(all_records)
    if quarantine:
        logger.warning(
            f"[results-only] {len(quarantine)} records quarantined, "
            f"using {len(records)} clean records for judgment"
        )
    del all_records
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

    return judgment


def save_judgment(judgment: dict, output_path: str) -> None:
    """判定結果を JSON に保存 (ztb/io atomic write 活用).

    119# 新規: run_fill_test.py 内の手動 json.dump を write_json に統一.
    """
    write_json(output_path, judgment, indent=2, ensure_ascii=False)
    logger.info(f"Saved judgment to {output_path}")
