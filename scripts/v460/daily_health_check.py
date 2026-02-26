#!/usr/bin/env python3
"""146# P2-04/P3-04: 日次ヘルスチェック + KPI バッチランナー.

gate_judgment (per-run + Monte Carlo) + oracle_baseline + trades_health を
一括実行し、JSON レポートと PASS/FAIL verdict を出力する。

タスクスケジューラ / cron で毎日 06:00 UTC (15:00 JST) に実行する想定。

Usage:
  # デフォルト (全チェック)
  python scripts/v460/daily_health_check.py

  # Monte Carlo をスキップ (高速)
  python scripts/v460/daily_health_check.py --skip-monte-carlo

  # 結果を JSON 保存
  python scripts/v460/daily_health_check.py --output reports/daily/2026-02-23.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


from ztb.io.json_io import write_json


CheckReport = dict[str, object]


class DailyHealthReport(TypedDict):
    timestamp_utc: str
    date: str
    checks: list[CheckReport]
    overall_healthy: bool


def _run_trades_health(days: int = 3) -> CheckReport:
    """P2-09/P0-03: trades データ健全性チェック."""
    from ztb.data.trades_health import check_trades_health

    result = check_trades_health(lookback_days=days)
    return {
        "check": "trades_health",
        "healthy": result.healthy,
        "available_days": result.available_days,
        "missing_days": result.missing_days,
        "stale_hours": round(result.stale_hours, 1),
        "message": result.message,
    }


def _run_feature_freshness() -> CheckReport:
    """136# P1-02: feature 鮮度チェック."""
    from ztb.data.trades_health import check_feature_freshness

    result = check_feature_freshness()
    return {
        "check": "feature_freshness",
        "fresh": result.fresh,
        "trades_stale_hours": round(result.trades_stale_hours, 1),
        "ob_stale_hours": round(result.ob_stale_hours, 1),
        "details": result.details,
        "message": result.message,
    }


def _run_gate_judgment(
    results_dir: str,
    latest_run: bool = True,
    monte_carlo: bool = True,
) -> CheckReport:
    """P0-07: Gate 判定 (per-run + optional Monte Carlo)."""
    from scripts.v460.gate_judgment import run_gate_judgment_for_results_dir

    report: CheckReport = {"check": "gate_judgment"}
    try:
        result = run_gate_judgment_for_results_dir(
            results_dir=results_dir,
            latest_run=latest_run,
            monte_carlo=False,  # 日次 MC は本関数後段で実行
        )
        g12 = result.get("g1_2_full")
        g11 = result.get("g1_1_quick")
        if isinstance(g12, dict):
            report["verdict"] = g12.get("gate_result", "UNKNOWN")
            report["g1_2_full"] = g12
        elif isinstance(g11, dict):
            report["verdict"] = g11.get("gate_result", "UNKNOWN")
        else:
            report["verdict"] = "UNKNOWN"
        if isinstance(g11, dict):
            report["g1_1_quick"] = g11
        report["run_scope"] = result.get("run_scope", "ALL")
        if "comparison" in result:
            report["comparison"] = result["comparison"]
    except Exception as e:
        logger.warning("gate_judgment failed: %s", e)
        report["verdict"] = "ERROR"
        report["error"] = str(e)

    if monte_carlo:
        try:
            from ztb.metrics.fill_quality import load_fill_records
            from ztb.risk.pnl_monte_carlo import MonteCarloConfig, PnLMonteCarloSimulator

            records = load_fill_records(results_dir)
            if len(records) >= 10:
                sim = PnLMonteCarloSimulator(MonteCarloConfig(n_simulations=5_000))
                mc_result = sim.run(records)
                report["monte_carlo"] = {
                    "pnl_mean_jpy": round(mc_result.pnl_mean_jpy, 0),
                    "var_95_jpy": round(mc_result.var_95_jpy, 0),
                    "prob_loss": round(mc_result.prob_loss, 3),
                    "prob_profit": round(mc_result.prob_profit, 3),
                    "g11_pass": mc_result.g11_pass,
                }
            else:
                report["monte_carlo"] = {"skipped": True, "reason": f"n={len(records)} < 10"}
        except Exception as e:
            logger.warning("Monte Carlo failed: %s", e)
            report["monte_carlo"] = {"error": str(e)}

    return report


def _run_oracle_baseline(results_dir: str) -> CheckReport:
    """P2-04: Oracle 日次 KPI."""
    report: CheckReport = {"check": "oracle_baseline"}
    try:
        from scripts.v460.analysis.oracle_baseline import run_oracle_baseline

        result = run_oracle_baseline(results_dir=results_dir)
        if result:
            all_metrics = result.get("all", {})
            report["oracle_pnl_mean_bps"] = all_metrics.get("pnl_mean_bps")
            report["oracle_pnl_positive_ratio"] = all_metrics.get("pnl_positive_ratio")
            report["ph3_check"] = result.get("ph3_check", {})
            report["n_records"] = result.get("params", {}).get("n_records_total", 0)
        else:
            report["skipped"] = True
            report["reason"] = "no fill records"
    except Exception as e:
        logger.warning("Oracle baseline failed: %s", e)
        report["error"] = str(e)
    return report



def _run_stopgap_health(results_dir: str, window_hours: int = 168) -> CheckReport:
    """168# §8 #8: Stopgap ヘルスレポート統合."""
    report: CheckReport = {"check": "stopgap_health"}
    try:
        from scripts.v460.lib.stopgap_health import (
            generate_health_report as gen_stopgap,
            load_fill_records,
        )

        results_path = Path(results_dir)
        records = load_fill_records(results_path)
        if not records:
            report["skipped"] = True
            report["reason"] = "no fill records"
            return report
        stopgap_rpt = gen_stopgap(records, window_hours=window_hours)
        n_total = stopgap_rpt.total_records
        n_filled = stopgap_rpt.total_filled
        report["n_records"] = n_total
        report["overall_fill_rate"] = round(n_filled / n_total, 4) if n_total else 0
        report["n_alerts"] = len(stopgap_rpt.alerts) if stopgap_rpt.alerts else 0
        # Stopgap exit checks summary (stopgap_checks は dict のリスト)
        if stopgap_rpt.stopgap_checks:
            exits = []
            for ec in stopgap_rpt.stopgap_checks:
                exits.append({
                    "stopgap_id": ec["stopgap_id"],
                    "verdict": ec["verdict"],
                    "metrics": ec.get("metrics"),
                    "criteria": ec.get("criteria"),
                })
            report["exit_checks"] = exits
            n_breach = sum(
                1 for ec in stopgap_rpt.stopgap_checks
                if ec.get("verdict") == "BREACH"
            )
            report["n_exit_breaches"] = n_breach
        # Critical alerts (alerts は既に dict のリスト)
        if stopgap_rpt.alerts:
            report["alerts"] = stopgap_rpt.alerts[:5]
    except Exception as e:
        logger.warning("stopgap_health failed: %s", e)
        report["error"] = str(e)
    return report


def _run_side_regime_dashboard(results_dir: str) -> CheckReport:
    """168# §8 #8: Side×Regime 3指標ダッシュボード統合."""
    report: CheckReport = {"check": "side_regime_dashboard"}
    try:
        from scripts.v460.analysis.side_regime_dashboard import run_dashboard

        dashboard = run_dashboard(results_dir=results_dir, with_judgment=False)
        report["total_records"] = dashboard.get("total_records", 0)
        report["total_filled"] = dashboard.get("total_filled", 0)
        report["overall_fill_rate"] = dashboard.get("overall_fill_rate", 0)
        # Side summaries
        if "side_summary" in dashboard:
            report["side_summary"] = dashboard["side_summary"]
        # Regime × side detail
        if "regime_side_detail" in dashboard:
            report["n_regime_side_groups"] = len(dashboard["regime_side_detail"])
    except Exception as e:
        logger.warning("side_regime_dashboard failed: %s", e)
        report["error"] = str(e)
    return report


def run_daily_health_check(
    results_dir: str = "results/v460/fill_test",
    skip_monte_carlo: bool = False,
    skip_oracle: bool = False,
    output_path: str | None = None,
) -> DailyHealthReport:
    """日次ヘルスチェック実行.

    Returns:
        全チェック結果の統合 dict.
    """
    now = datetime.now(timezone.utc)
    report: DailyHealthReport = {
        "timestamp_utc": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "checks": [],
        "overall_healthy": True,
    }

    # 1. Trades health
    logger.info("=== [1/6] Trades Health Check ===")
    trades_result = _run_trades_health()
    report["checks"].append(trades_result)
    if not trades_result.get("healthy", False):
        report["overall_healthy"] = False
        logger.warning("trades_health: UNHEALTHY — %s", trades_result.get("message"))
    else:
        logger.info("trades_health: OK")

    # 2. Feature freshness
    logger.info("=== [2/6] Feature Freshness Check ===")
    freshness_result = _run_feature_freshness()
    report["checks"].append(freshness_result)
    if not freshness_result.get("fresh", False):
        report["overall_healthy"] = False
        logger.warning("feature_freshness: STALE — %s", freshness_result.get("message"))
    else:
        logger.info("feature_freshness: OK")

    # 3. Gate judgment + Monte Carlo
    logger.info("=== [3/6] Gate Judgment %s ===", "(+ Monte Carlo)" if not skip_monte_carlo else "")
    gate_result = _run_gate_judgment(
        results_dir=results_dir,
        latest_run=True,
        monte_carlo=not skip_monte_carlo,
    )
    report["checks"].append(gate_result)
    verdict = gate_result.get("verdict", "UNKNOWN")
    if verdict not in ("PASS", "WATCH"):
        report["overall_healthy"] = False
        logger.warning("gate_judgment: %s", verdict)
    else:
        logger.info("gate_judgment: %s", verdict)

    # 4. Oracle baseline
    if not skip_oracle:
        logger.info("=== [4/6] Oracle Baseline KPI ===")
        oracle_result = _run_oracle_baseline(results_dir=results_dir)
        report["checks"].append(oracle_result)
        logger.info("oracle_baseline: done (mean=%.3f bps)",
                     oracle_result.get("oracle_pnl_mean_bps", 0) or 0)
    else:
        logger.info("=== [4/6] Oracle Baseline: skipped ===")

    # 5. 168# Stopgap health
    logger.info("=== [5/6] Stopgap Health ===")
    stopgap_result = _run_stopgap_health(results_dir=results_dir)
    report["checks"].append(stopgap_result)
    n_breaches = stopgap_result.get("n_exit_breaches", 0)
    if n_breaches > 0:
        report["overall_healthy"] = False
        logger.warning("stopgap_health: %d EXIT BREACHES", n_breaches)
    else:
        logger.info("stopgap_health: OK (fill_rate=%.1f%%)",
                     (stopgap_result.get("overall_fill_rate") or 0) * 100)

    # 6. 168# Side × Regime dashboard
    logger.info("=== [6/6] Side×Regime Dashboard ===")
    dashboard_result = _run_side_regime_dashboard(results_dir=results_dir)
    report["checks"].append(dashboard_result)
    logger.info("side_regime_dashboard: %d records, fill_rate=%.1f%%",
                 dashboard_result.get("total_records", 0),
                 (dashboard_result.get("overall_fill_rate") or 0) * 100)

    # Summary
    status = "HEALTHY" if report["overall_healthy"] else "UNHEALTHY"
    logger.info("=" * 50)
    logger.info("Daily Health Check: %s", status)
    logger.info("=" * 50)

    # 168# Discord 通知
    try:
        from ztb.utils.notify import get_notifier
        _notifier = get_notifier()
        _fields = {"checks": str(len(report["checks"]))}
        for chk in report["checks"]:
            name = chk.get("check", "?")
            if chk.get("error"):
                _fields[name] = f"ERROR: {chk['error']}"
            elif chk.get("n_exit_breaches", 0) > 0:
                _fields[name] = f"BREACH x{chk['n_exit_breaches']}"
            else:
                _fields[name] = "OK"
        _notifier.notify_daily_health(now.strftime("%Y-%m-%d"), report["overall_healthy"], _fields)
    except Exception:
        logger.debug("Discord notification skipped", exc_info=True)

    # Save
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json(out, report, indent=2, ensure_ascii=False)
        logger.info("Report saved to: %s", out)

    return report


def main() -> None:
    """CLI エントリポイント."""
    parser = argparse.ArgumentParser(
        description="146# P2-04/P3-04: Daily health check + KPI batch runner"
    )
    parser.add_argument(
        "--results-dir",
        default="results/v460/fill_test",
        help="Fill records directory (default: results/v460/fill_test)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSON output path (e.g. reports/daily/2026-02-23.json)",
    )
    parser.add_argument(
        "--skip-monte-carlo",
        action="store_true",
        help="Skip Monte Carlo simulation (faster)",
    )
    parser.add_argument(
        "--skip-oracle",
        action="store_true",
        help="Skip Oracle baseline computation",
    )
    args = parser.parse_args()
    run_daily_health_check(
        results_dir=args.results_dir,
        skip_monte_carlo=args.skip_monte_carlo,
        skip_oracle=args.skip_oracle,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
