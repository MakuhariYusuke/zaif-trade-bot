"""554# CalibrationMap offline batch builder.

fill_records JSONL から CalibrationMap を構築し、JSON にエクスポートする。
fill_test 起動時に load_state() で読み込むことで cold start 問題を回避。

546# §B (b) recommended approach:
  Offline batch → JSON/YAML export → startup load

Usage:
  # 全期間で構築
  python scripts/v460/ml/calibration_batch.py

  # 直近 N 日のみ
  python scripts/v460/ml/calibration_batch.py --days 14

  # ライブラリ
  from scripts.v460.ml.calibration_batch import build_calibration_map
  state = build_calibration_map(results_dir, days=14)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_RESULTS_DIR = _PROJECT_ROOT / "results" / "v460" / "fill_test"
_OUTPUT_PATH = _PROJECT_ROOT / "models" / "v460" / "entry_gate_calibration.json"

# CalibrationMap のデフォルト設定
_DEFAULT_CONFIG: dict[str, Any] = {
    "ewma_tau": 100.0,
    "n_min": 30.0,
    "prior_avg_win": 0.0,
    "prior_avg_loss": 0.0,
}


def _side_to_action(side: str) -> float:
    """fill_test の side → CalibrationMap の action に変換.

    559# fix: runtime (orchestrator_mid_cycle / post_cycle) が ±0.3 を使用し
    Buy/Sell bin にマッピングするため、offline も同じ値を使用して
    L1 キー整合性を保つ。
    buy → +0.3 (Buy bin), sell → -0.3 (Sell bin).
    """
    return 0.3 if side == "buy" else -0.3


def build_calibration_map(
    results_dir: Path | str | None = None,
    output_path: Path | str | None = None,
    days: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """fill_records から CalibrationMap を構築して JSON 保存.

    Args:
        results_dir: fill_records のディレクトリ。
        output_path: JSON 出力先。
        days: 直近 N 日のみ使用 (None=全期間)。
        config: CalibrationMap 設定。

    Returns:
        CalibrationMap の state dict。
    """
    from ztb.metrics.fill_quality import iter_fill_records_glob
    from ztb.trading.signal.calibration_map import CalibrationMap

    r_dir = Path(results_dir) if results_dir else _RESULTS_DIR
    o_path = Path(output_path) if output_path else _OUTPUT_PATH
    cfg = config or _DEFAULT_CONFIG

    cal_map = CalibrationMap(cfg)

    # 日数フィルタ
    cutoff_ts: float | None = None
    end_ts: float | None = None
    if days is not None:
        cutoff_ts = (
            datetime.now(timezone.utc).timestamp() - days * 86_400
        )
    if date_from is not None:
        cutoff_ts = max(
            cutoff_ts or float("-inf"),
            datetime.fromisoformat(date_from).replace(tzinfo=timezone.utc).timestamp(),
        )
    if date_to is not None:
        end_dt = datetime.fromisoformat(date_to).replace(tzinfo=timezone.utc) + timedelta(days=1)
        end_ts = end_dt.timestamp()

    n_total = 0
    n_filled = 0
    n_used = 0
    regime_counts: dict[str, int] = {}
    step_counter = 0

    for record in iter_fill_records_glob(r_dir):
        n_total += 1

        # 日数フィルタ
        if cutoff_ts is not None and record.timestamp < cutoff_ts:
            continue
        if end_ts is not None and record.timestamp >= end_ts:
            continue

        # filled レコードのみ (PnL が存在するもの)
        if not record.filled:
            continue
        n_filled += 1

        # gross_pnl: post_fill_30s_pnl を primary に使用
        gross_pnl = record.post_fill_30s_pnl
        if gross_pnl is None:
            # ev_weighted_pnl をフォールバック
            gross_pnl = getattr(record, "ev_weighted_pnl", None)
        if gross_pnl is None:
            continue

        regime = record.regime or "unknown"
        action = _side_to_action(record.side)

        cal_map.update(regime, action, float(gross_pnl), step_counter)
        step_counter += 1
        n_used += 1

        regime_counts[regime] = regime_counts.get(regime, 0) + 1

    # 結果ログ
    logger.info(
        f"[554#] CalibrationMap built: "
        f"{n_total} total, {n_filled} filled, {n_used} used"
    )
    for regime, count in sorted(regime_counts.items()):
        stats = cal_map.get_stats(regime, 1.0)
        l1 = stats["l1"]
        fb = stats["fallback"]
        logger.info(
            f"  {regime}: n={count}, "
            f"p_win_lcb={l1['p_win_lcb']:.3f}, "
            f"n_eff={l1['n_eff']:.1f}, "
            f"fallback_p_win={fb['p_win_lcb']:.3f}"
        )

    # JSON エクスポート
    state = cal_map.get_state()
    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "n_records_total": n_total,
        "n_records_filled": n_filled,
        "n_records_used": n_used,
        "days_filter": days,
        "date_from": date_from,
        "date_to": date_to,
        "regime_counts": regime_counts,
        "config": cfg,
    }
    export = {"meta": meta, **state}

    o_path.parent.mkdir(parents=True, exist_ok=True)
    with open(o_path, "w") as f:
        json.dump(export, f, indent=2, default=str)

    logger.info(f"[554#] Exported to {o_path}")
    return export


def load_calibration_state(
    path: Path | str | None = None,
    config: dict[str, Any] | None = None,
) -> Any:
    """JSON から CalibrationMap を復元.

    fill_test 起動時に呼ぶ想定。

    Returns:
        CalibrationMap instance (or None if file not found).
    """
    from ztb.trading.signal.calibration_map import CalibrationMap

    p = Path(path) if path else _OUTPUT_PATH
    cfg = config or _DEFAULT_CONFIG

    if not p.exists():
        logger.warning(f"[554#] Calibration file not found: {p}")
        return None

    with open(p) as f:
        data = json.load(f)

    cal_map = CalibrationMap(cfg)
    cal_map.load_state(data)

    meta = data.get("meta", {})
    logger.info(
        f"[554#] Loaded CalibrationMap from {p}: "
        f"{meta.get('n_records_used', '?')} records, "
        f"built_at={meta.get('built_at', '?')}"
    )
    return cal_map


def save_calibration_state(
    cal_map: Any,
    path: Path | str | None = None,
) -> None:
    """CalibrationMap の現在状態を JSON に保存 (559# online 学習永続化).

    Args:
        cal_map: CalibrationMap インスタンス。
        path: 保存先パス。None の場合はデフォルトパス。
    """
    p = Path(path) if path else _OUTPUT_PATH
    state = cal_map.get_state()
    export = {
        "meta": {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "source": "online_update",
        },
        **state,
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(export, f, indent=2, default=str)
    logger.debug("[559#] CalibrationMap saved to %s", p)


# ════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════

def main() -> None:
    """CLI メイン."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    parser = argparse.ArgumentParser(
        description="554# CalibrationMap offline batch builder",
    )
    parser.add_argument(
        "--results-dir", type=str, default=str(_RESULTS_DIR),
        help="Fill records directory",
    )
    parser.add_argument(
        "--output", type=str, default=str(_OUTPUT_PATH),
        help="Output JSON path",
    )
    parser.add_argument(
        "--days", type=int, default=None,
        help="Only use last N days of data (default: all)",
    )
    parser.add_argument("--date-from", type=str, default=None, help="開始日 YYYY-MM-DD")
    parser.add_argument("--date-to", type=str, default=None, help="終了日 YYYY-MM-DD")
    args = parser.parse_args()

    export = build_calibration_map(
        results_dir=args.results_dir,
        output_path=args.output,
        days=args.days,
        date_from=args.date_from,
        date_to=args.date_to,
    )

    # サマリ表示
    meta = export.get("meta", {})
    print(f"\nCalibrationMap built:")
    print(f"  Records: {meta.get('n_records_used', 0)} used")
    print(f"  Regimes: {meta.get('regime_counts', {})}")

    global_stats = export.get("stats", {}).get("global", {})
    if global_stats:
        n_eff = (global_stats.get("sum_w", 0) ** 2) / (
            global_stats.get("sum_w_sq", 1e-9)
        )
        print(f"  Global n_eff: {n_eff:.1f}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
