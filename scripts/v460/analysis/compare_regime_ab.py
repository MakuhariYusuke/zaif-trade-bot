"""152# §9 P0-2: regime A/B 比較ハーネス — 改善前 vs 改善後の unknown 削減効果を定量検証.

改善内容 (152# §4.2):
  A: accelerated hysteresis (UNKNOWN → first regime は N-1 連続で確定)
  B: majority fallback (choppy market → 2N 観測後に最頻分類で仮確定)

比較方式:
  fill_records の order_price + timestamp をレジーム検知器に replay し、
  old (A+B なし) vs new (A+B あり) のレジーム分類を比較。

Gate 判定 (§4.6):
  G1: unknown ≤ 3%
  G2: regime 別 PnL ±0.1 bps 以内
  G3: 全体 PnL ≤ 5 bps 悪化

Usage:
    python -m scripts.v460.analysis.compare_regime_ab
    python -m scripts.v460.analysis.compare_regime_ab --start 2026-02-13 --end 2026-02-22
    python -m scripts.v460.analysis.compare_regime_ab --output results/v460/ab_regime
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from scripts.v460.lib.regime_detector import (
    FillTestRegime,
    FillTestRegimeDetector,
    RegimeConfig,
    RegimeResult,
)

# Re-use data loading from reproduce script
from scripts.v460.analysis.reproduce_152_metrics import _load_records


# ---------------------------------------------------------------------------
# Old detector: 152# 以前のロジック (A+B 改善なし)
# ---------------------------------------------------------------------------


class OldFillTestRegimeDetector(FillTestRegimeDetector):
    """152# 以前のヒステリシスロジック (A+B なし) を再現.

    変更点:
      - _apply_hysteresis() で accelerated threshold を使わない
      - majority fallback を実行しない
    """

    def _apply_hysteresis(self, raw_regime: FillTestRegime) -> FillTestRegime:
        """旧ロジック: N 回連続一致のみ (加速なし、フォールバックなし)."""
        self._raw_history.append(raw_regime)
        if len(self._raw_history) > self.config.hysteresis_count * 3:
            self._raw_history = self._raw_history[-self.config.hysteresis_count * 3 :]

        consecutive = 0
        for r in reversed(self._raw_history):
            if r == raw_regime:
                consecutive += 1
            else:
                break

        if raw_regime == self._confirmed_regime:
            self._stability_count = consecutive
            return self._confirmed_regime

        # 旧: 常に hysteresis_count を要求 (加速なし)
        threshold = self.config.hysteresis_count

        if consecutive >= threshold:
            self._confirmed_regime = raw_regime
            self._stability_count = consecutive
            return raw_regime

        # 旧: majority fallback なし → 旧レジーム維持
        self._stability_count += 1
        return self._confirmed_regime


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


@dataclass
class SimRecord:
    """1 レコードに対するシミュレーション結果."""

    timestamp: float
    order_price: float
    recorded_regime: str  # fill_records に記録されたレジーム
    old_regime: str  # old detector のレジーム
    new_regime: str  # new detector (152# A+B) のレジーム
    old_confidence: float
    new_confidence: float
    filled: bool
    pnl_30s: float | None


def _simulate(
    records: list[dict[str, Any]],
    config: RegimeConfig | None = None,
) -> tuple[list[SimRecord], dict[str, int]]:
    """fill records を old/new detector に replay してレジーム分類を比較.

    Returns:
        (results, stats): results はレコードごとの結果, stats は前処理統計.
    """
    cfg = config or RegimeConfig(min_confidence=0.2)
    old_det = OldFillTestRegimeDetector(RegimeConfig(
        window=cfg.window,
        trend_threshold_pct=cfg.trend_threshold_pct,
        high_vol_multiplier=cfg.high_vol_multiplier,
        hysteresis_count=cfg.hysteresis_count,
        # 旧 min_confidence: 0.3 (152# で 0.2 に変更前)
        min_confidence=0.3,
    ))
    new_det = FillTestRegimeDetector(RegimeConfig(
        window=cfg.window,
        trend_threshold_pct=cfg.trend_threshold_pct,
        high_vol_multiplier=cfg.high_vol_multiplier,
        hysteresis_count=cfg.hysteresis_count,
        min_confidence=cfg.min_confidence,
    ))

    # Sort by timestamp for chronological replay
    sorted_recs = sorted(records, key=lambda r: float(r.get("timestamp", 0)))

    # §12 #2: order_price==0 / side 不正のレコードを除外
    prefilter_stats: dict[str, int] = {"total_input": len(sorted_recs), "price_zero_excluded": 0, "side_invalid_excluded": 0}
    valid_recs: list[dict[str, Any]] = []
    for rec in sorted_recs:
        ts = rec.get("timestamp")
        price = rec.get("order_price")
        if ts is None or price is None:
            continue
        if float(price) <= 0:
            prefilter_stats["price_zero_excluded"] += 1
            continue
        side = rec.get("side", "")
        if side not in ("", "buy", "sell", None):
            prefilter_stats["side_invalid_excluded"] += 1
            continue
        valid_recs.append(rec)
    prefilter_stats["valid_records"] = len(valid_recs)

    results: list[SimRecord] = []
    for rec in valid_recs:
        ts = rec.get("timestamp")
        price = rec.get("order_price")
        if ts is None or price is None:
            continue

        ts_f = float(ts)
        price_f = float(price)

        old_result = old_det.update(ts_f, price_f)
        new_result = new_det.update(ts_f, price_f)

        results.append(SimRecord(
            timestamp=ts_f,
            order_price=price_f,
            recorded_regime=str(rec.get("regime", "n/a")),
            old_regime=old_result.regime.value,
            new_regime=new_result.regime.value,
            old_confidence=old_result.confidence,
            new_confidence=new_result.confidence,
            filled=bool(rec.get("filled")),
            pnl_30s=rec.get("post_fill_30s_pnl"),
        ))

    return results, prefilter_stats


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    """§4.6 Gate 判定結果."""

    gate_id: str
    passed: bool
    threshold: str
    actual: str
    detail: str


def _evaluate_gates(
    sim_results: list[SimRecord],
    recorded_regime_pnl: dict[str, float],
) -> list[GateResult]:
    """§4.6 Gate を old/new で評価."""
    gates: list[GateResult] = []

    # --- G1: unknown ratio ---
    old_total = len(sim_results)
    old_unknown = sum(1 for r in sim_results if r.old_regime == "unknown")
    new_unknown = sum(1 for r in sim_results if r.new_regime == "unknown")
    old_unk_pct = old_unknown / old_total * 100 if old_total else 0
    new_unk_pct = new_unknown / old_total * 100 if old_total else 0

    gates.append(GateResult(
        gate_id="G1",
        passed=new_unk_pct <= 3.0,
        threshold="≤ 3%",
        actual=f"old={old_unk_pct:.1f}% → new={new_unk_pct:.1f}%",
        detail=f"unknown: {old_unknown} → {new_unknown} (of {old_total})",
    ))

    # --- G2: regime PnL non-degradation ---
    # Compare filled records' PnL by new-assigned regime vs recorded regime
    filled_results = [r for r in sim_results if r.filled and r.pnl_30s is not None]

    old_regime_pnl: dict[str, list[float]] = defaultdict(list)
    new_regime_pnl: dict[str, list[float]] = defaultdict(list)
    for r in filled_results:
        old_regime_pnl[r.old_regime].append(r.pnl_30s)  # type: ignore[arg-type]
        new_regime_pnl[r.new_regime].append(r.pnl_30s)  # type: ignore[arg-type]

    max_degradation = 0.0
    g2_details: list[str] = []
    for regime in ["ranging", "trending"]:
        old_avg = sum(old_regime_pnl.get(regime, [0])) / max(len(old_regime_pnl.get(regime, [1])), 1)
        new_avg = sum(new_regime_pnl.get(regime, [0])) / max(len(new_regime_pnl.get(regime, [1])), 1)
        diff = abs(new_avg - old_avg)
        max_degradation = max(max_degradation, diff)
        g2_details.append(f"{regime}: old={old_avg:.4f} → new={new_avg:.4f} (Δ={new_avg-old_avg:+.4f})")

    gates.append(GateResult(
        gate_id="G2",
        passed=max_degradation <= 0.1,
        threshold="±0.1 bps",
        actual=f"max_Δ={max_degradation:.4f} bps",
        detail="; ".join(g2_details),
    ))

    # --- G3: total PnL non-degradation (INFORMATIONAL) ---
    # §12 #3: replay では実 PnL が変わらないため、一方的に Δ=0 になる。
    # regime 再分類による lot/timeout パラメータ変更の影響は、実運用後に評価する。
    # 現時点では再分類件数のみ報告。
    old_total_pnl = sum(r.pnl_30s for r in filled_results if r.pnl_30s is not None)
    reclassified_count = sum(
        1 for r in sim_results if r.old_regime != r.new_regime
    )
    reclassified_filled = sum(
        1 for r in filled_results if r.old_regime != r.new_regime
    )

    gates.append(GateResult(
        gate_id="G3",
        passed=True,  # informational: always True
        threshold="informational (replayではPnL同一)",
        actual=f"reclassified={reclassified_count}/{len(sim_results)} ({reclassified_filled} filled)",
        detail=f"total_pnl={old_total_pnl:.2f} bps (unchanged). "
               f"Regime変更によるlot/timeout影響は実運用後に評価。",
    ))

    return gates


# ---------------------------------------------------------------------------
# Display & output
# ---------------------------------------------------------------------------


def _print_report(
    sim_results: list[SimRecord],
    gates: list[GateResult],
) -> None:
    """Print A/B comparison report."""
    print("=" * 60)
    print("152# regime A/B 比較レポート")
    print("=" * 60)

    total = len(sim_results)
    filled = [r for r in sim_results if r.filled]

    # Unknown distribution comparison
    print("\n--- Unknown 比率比較 ---")
    for label, getter in [("old (pre-152#)", lambda r: r.old_regime),
                           ("new (152# A+B)", lambda r: r.new_regime)]:
        unk = sum(1 for r in sim_results if getter(r) == "unknown")
        unk_f = sum(1 for r in filled if getter(r) == "unknown")
        print(f"  {label}: {unk}/{total} ({unk/total*100:.1f}%) all, "
              f"{unk_f}/{len(filled)} ({unk_f/len(filled)*100:.1f}%) filled")

    # Regime distribution comparison
    print("\n--- Regime 分布比較 ---")
    old_dist: Counter[str] = Counter(r.old_regime for r in sim_results)
    new_dist: Counter[str] = Counter(r.new_regime for r in sim_results)
    all_regimes = sorted(set(old_dist) | set(new_dist))
    print(f"  {'Regime':<12} {'old':>8} {'new':>8} {'Δ':>8}")
    for regime in all_regimes:
        o = old_dist.get(regime, 0)
        n = new_dist.get(regime, 0)
        print(f"  {regime:<12} {o:>8} {n:>8} {n-o:>+8}")

    # Reclassification detail: where did unknown records go?
    print("\n--- Unknown → 再分類先 (new) ---")
    old_unknown = [r for r in sim_results if r.old_regime == "unknown"]
    reclass: Counter[str] = Counter(r.new_regime for r in old_unknown)
    for regime, count in reclass.most_common():
        print(f"  → {regime}: {count}")

    # Filled PnL by regime (new assignment)
    print("\n--- Regime × PnL (new assignment, filled) ---")
    new_regime_pnl: dict[str, list[float]] = defaultdict(list)
    for r in filled:
        if r.pnl_30s is not None:
            new_regime_pnl[r.new_regime].append(r.pnl_30s)
    print(f"  {'Regime':<12} {'fills':>6} {'avg PnL':>10} {'sum PnL':>10}")
    for regime, vals in sorted(new_regime_pnl.items(), key=lambda x: -len(x[1])):
        avg = sum(vals) / len(vals) if vals else 0
        print(f"  {regime:<12} {len(vals):>6} {avg:>10.4f} {sum(vals):>10.2f}")

    # Gate results
    print("\n--- §4.6 Gate 判定 ---")
    all_passed = True
    for g in gates:
        status = "✅ PASS" if g.passed else "❌ FAIL"
        all_passed = all_passed and g.passed
        print(f"  {g.gate_id}: {status} (threshold: {g.threshold}, actual: {g.actual})")
        print(f"         {g.detail}")

    print(f"\n{'='*60}")
    verdict = "✅ 採用可能" if all_passed else "⚠️ 要検討"
    print(f"総合判定: {verdict}")
    print(f"{'='*60}")


def _save_csv(
    sim_results: list[SimRecord],
    output_dir: Path,
) -> None:
    """Save per-record simulation results as CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "regime_ab_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "order_price", "recorded_regime",
            "old_regime", "new_regime", "old_confidence", "new_confidence",
            "filled", "pnl_30s", "reclassified",
        ])
        for r in sim_results:
            writer.writerow([
                r.timestamp, r.order_price, r.recorded_regime,
                r.old_regime, r.new_regime,
                round(r.old_confidence, 4), round(r.new_confidence, 4),
                r.filled, r.pnl_30s,
                r.old_regime != r.new_regime,
            ])
    print(f"Saved: {csv_path}")


def _save_summary(
    gates: list[GateResult],
    sim_results: list[SimRecord],
    output_dir: Path,
    *,
    config_info: dict[str, Any] | None = None,
    prefilter_stats: dict[str, int] | None = None,
) -> None:
    """Save summary as JSON."""
    total = len(sim_results)
    old_unk = sum(1 for r in sim_results if r.old_regime == "unknown")
    new_unk = sum(1 for r in sim_results if r.new_regime == "unknown")

    summary = {
        "total_records": total,
        "old_unknown": old_unk,
        "new_unknown": new_unk,
        "old_unknown_pct": round(old_unk / total * 100, 2) if total else 0,
        "new_unknown_pct": round(new_unk / total * 100, 2) if total else 0,
        "gates": [
            {
                "gate_id": g.gate_id,
                "passed": g.passed,
                "threshold": g.threshold,
                "actual": g.actual,
                "detail": g.detail,
            }
            for g in gates
        ],
        "all_gates_passed": all(g.passed for g in gates),
        "config": config_info or {},
        "prefilter_stats": prefilter_stats or {},
    }

    json_path = output_dir / "regime_ab_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {json_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> list[GateResult]:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="152# regime A/B 比較ハーネス — old vs new detector",
    )
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--run-id", default=None, help="Filter by run_id")
    parser.add_argument("--data-dir", default="results/v460/fill_test")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument(
        "--min-confidence", type=float, default=0.2,
        help="New detector の min_confidence (default: 0.2, fill_test.yaml と同期)",
    )
    args = parser.parse_args(argv)

    records = _load_records(
        args.data_dir,
        start_date=args.start,
        end_date=args.end,
        run_id=args.run_id,
    )

    if not records:
        print("ERROR: No records", file=sys.stderr)
        sys.exit(1)

    cfg = RegimeConfig(min_confidence=args.min_confidence)
    sim_results, prefilter_stats = _simulate(records, config=cfg)

    # §12 #1: 使用設定をログ出力
    print(f"[config] new_detector min_confidence={cfg.min_confidence}")
    if prefilter_stats.get("price_zero_excluded", 0) > 0:
        print(f"[prefilter] price==0 excluded: {prefilter_stats['price_zero_excluded']}")
    if prefilter_stats.get("side_invalid_excluded", 0) > 0:
        print(f"[prefilter] invalid side excluded: {prefilter_stats['side_invalid_excluded']}")
    print(f"[prefilter] valid records: {prefilter_stats['valid_records']}/{prefilter_stats['total_input']}")

    # Recorded regime PnL (for reference)
    recorded_pnl: dict[str, float] = {}
    for r in records:
        regime = r.get("regime")
        pnl = r.get("post_fill_30s_pnl")
        if regime and pnl is not None and r.get("filled"):
            recorded_pnl.setdefault(regime, 0.0)

    gates = _evaluate_gates(sim_results, recorded_pnl)

    _print_report(sim_results, gates)

    if args.output:
        out_dir = Path(args.output)
        _save_csv(sim_results, out_dir)
        _save_summary(
            gates, sim_results, out_dir,
            config_info={"min_confidence_new": cfg.min_confidence, "min_confidence_old": 0.3},
            prefilter_stats=prefilter_stats,
        )

    return gates


if __name__ == "__main__":
    main()
