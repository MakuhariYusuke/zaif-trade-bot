"""
Backtest reporting module.

Generates JSON and Markdown reports from backtest results.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .metrics import BacktestMetrics


class ReportGenerator:
    """Generates backtest reports in multiple formats."""

    @staticmethod
    def generate_json_report(
        metrics: BacktestMetrics,
        equity_curve: List[Dict[str, Any]],
        orders: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        output_path: str,
    ) -> str:
        """Generate JSON report."""

        # Convert timestamps to strings for JSON serialization
        def serialize_timestamps(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {
                    k: (v.isoformat() if hasattr(v, "isoformat") else v)
                    for k, v in obj.items()
                }
            return obj

        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "strategy": metadata.get("strategy", "unknown"),
                "dataset": metadata.get("dataset", "unknown"),
                **metadata,
            },
            "metrics": {
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "calmar_ratio": metrics.calmar_ratio,
                "total_return": metrics.total_return,
                "cagr": metrics.cagr,
                "annualized_return": metrics.annualized_return,
                "max_drawdown": metrics.max_drawdown,
                "volatility": metrics.volatility,
                "total_trades": metrics.total_trades,
                "win_rate": metrics.win_rate,
                "avg_win": metrics.avg_win,
                "avg_loss": metrics.avg_loss,
                "profit_factor": metrics.profit_factor,
                "turnover": metrics.turnover,
                "estimated_slippage_bps": metrics.estimated_slippage_bps,
                "deflated_sharpe_ratio": metrics.deflated_sharpe_ratio,
                "pvalue_bootstrap": metrics.pvalue_bootstrap,
            },
            "equity_curve": [serialize_timestamps(point) for point in equity_curve],
            "orders": [serialize_timestamps(order) for order in orders],
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return output_path

    @staticmethod
    def generate_markdown_report(
        metrics: BacktestMetrics, metadata: Dict[str, Any], output_path: str
    ) -> str:
        """Generate Markdown executive summary."""

        md_content = f"""# Backtest Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Strategy:** {metadata.get("strategy", "Unknown")}
**Dataset:** {metadata.get("dataset", "Unknown")}

## Performance Summary

### Risk-Adjusted Returns
- **Sharpe Ratio:** {metrics.sharpe_ratio:.3f}
- **Sortino Ratio:** {metrics.sortino_ratio:.3f}
- **Calmar Ratio:** {metrics.calmar_ratio:.3f}

### Returns
- **Total Return:** {metrics.total_return:.2%}
- **CAGR:** {metrics.cagr:.2%}
- **Annualized Return:** {metrics.annualized_return:.2%}

### Risk Metrics
- **Maximum Drawdown:** {metrics.max_drawdown:.2%}
- **Volatility (Annualized):** {metrics.volatility:.2%}

## Trading Statistics

- **Total Trades:** {metrics.total_trades}
- **Win Rate:** {metrics.win_rate:.1%}
- **Average Win:** {metrics.avg_win:.4f}
- **Average Loss:** {metrics.avg_loss:.4f}
- **Profit Factor:** {metrics.profit_factor:.2f}
- **Turnover (Annualized):** {metrics.turnover:.2f}

## Assumptions & Notes

- **Slippage Estimate:** {metrics.estimated_slippage_bps} bps per trade
- **Risk-Free Rate:** 2.0% (annualized)
- **Benchmark:** Buy & Hold (not shown in this report)

## Interpretation

### Sharpe Ratio
- > 1.0: Good risk-adjusted returns
- > 2.0: Excellent risk-adjusted returns
- < 0.5: Poor risk-adjusted returns

### Maximum Drawdown
- < 10%: Low risk
- 10-20%: Moderate risk
- > 20%: High risk

### Win Rate & Profit Factor
- Win Rate > 50% with Profit Factor > 1.5: Good strategy
- Profit Factor > 2.0: Excellent strategy

---
*This report was generated automatically. For detailed analysis, refer to the accompanying JSON and CSV files.*
"""

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return output_path

    @staticmethod
    def generate_equity_csv(
        equity_curve: List[Dict[str, Any]], output_path: str
    ) -> str:
        """Generate equity curve CSV."""

        if not equity_curve:
            # Create empty file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            Path(output_path).touch()
            return output_path

        import csv

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "equity"])
            writer.writeheader()
            for point in equity_curve:
                writer.writerow(point)

        return output_path

    @staticmethod
    def generate_orders_csv(orders: List[Dict[str, Any]], output_path: str) -> str:
        """Generate orders CSV."""

        if not orders:
            # Create empty file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            Path(output_path).touch()
            return output_path

        import csv

        # Get all unique keys from orders
        fieldnames: set[str] = set()
        for order in orders:
            fieldnames.update(order.keys())

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
            writer.writeheader()
            writer.writerows(orders)

        return output_path
