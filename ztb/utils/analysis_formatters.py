"""
Analysis result formatters for consistent output formatting across the codebase.

This module provides standardized formatting functions for analysis results,
performance metrics, and trading statistics.
"""

import json
from typing import Any, Dict


def format_performance_summary(performance_summary: Dict[str, Any]) -> str:
    """Format performance summary for display."""
    lines = ["=== PERFORMANCE SUMMARY ==="]

    if "total_signals_generated" in performance_summary:
        lines.append(
            f"Total Signals Generated: {performance_summary['total_signals_generated']}"
        )

    if "total_patterns_recognized" in performance_summary:
        lines.append(
            f"Total Patterns Recognized: {performance_summary['total_patterns_recognized']}"
        )

    if "uptime_seconds" in performance_summary:
        lines.append(f"Uptime: {performance_summary['uptime_seconds']:.2f} seconds")

    if "cache_hit_rate" in performance_summary:
        lines.append(f"Cache Hit Rate: {performance_summary['cache_hit_rate']:.2f}%")

    return "\n".join(lines)


def format_pattern_statistics(pattern_statistics: Dict[str, Any]) -> str:
    """Format pattern statistics for display."""
    lines = ["=== PATTERN STATISTICS ==="]

    if not pattern_statistics:
        lines.append("No pattern statistics available")
        return "\n".join(lines)

    # Group patterns by type
    pattern_groups = {}
    for pattern_name, stats in pattern_statistics.items():
        pattern_type = pattern_name.split("_")[0] if "_" in pattern_name else "other"
        if pattern_type not in pattern_groups:
            pattern_groups[pattern_type] = []
        pattern_groups[pattern_type].append((pattern_name, stats))

    for pattern_type, patterns in pattern_groups.items():
        lines.append(f"\n{pattern_type.upper()} PATTERNS:")
        for pattern_name, stats in patterns:
            if isinstance(stats, dict):
                recognition_rate = stats.get("recognition_rate", "N/A")
                effectiveness = stats.get("effectiveness_score", "N/A")
                lines.append(
                    f"  {pattern_name}: recognition={recognition_rate}, effectiveness={effectiveness}"
                )
            else:
                lines.append(f"  {pattern_name}: {stats}")

    return "\n".join(lines)


def format_trading_results_summary(trading_results_summary: Dict[str, Any]) -> str:
    """Format trading results summary for display."""
    lines = ["=== TRADING RESULTS SUMMARY ==="]

    if "total_trades" in trading_results_summary:
        lines.append(f"Total Trades: {trading_results_summary['total_trades']}")

    if "profitable_trades" in trading_results_summary:
        total_trades = trading_results_summary.get("total_trades", 0)
        profitable_trades = trading_results_summary["profitable_trades"]
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        lines.append(f"Profitable Trades: {profitable_trades} ({win_rate:.1f}%)")

    if "total_profit" in trading_results_summary:
        lines.append(f"Total Profit: {trading_results_summary['total_profit']:.4f}")

    return "\n".join(lines)


def format_complete_analysis(analysis: Dict[str, Any]) -> str:
    """Format complete analysis results for display."""
    sections = []

    if "performance_summary" in analysis:
        sections.append(format_performance_summary(analysis["performance_summary"]))

    if "pattern_statistics" in analysis:
        sections.append(format_pattern_statistics(analysis["pattern_statistics"]))

    if "trading_results_summary" in analysis:
        sections.append(
            format_trading_results_summary(analysis["trading_results_summary"])
        )

    return "\n\n".join(sections)


def format_analysis_as_json(analysis: Dict[str, Any], indent: int = 2) -> str:
    """Format analysis results as JSON string."""
    return json.dumps(analysis, indent=indent, default=str)


def print_analysis_results(analysis: Dict[str, Any], use_json: bool = False) -> None:
    """Print analysis results in formatted or JSON format."""
    if use_json:
        print(format_analysis_as_json(analysis))
    else:
        print(format_complete_analysis(analysis))


def print_formatted_metrics(
    metrics: Dict[str, Any], title: str = "Analysis Results"
) -> None:
    """Print formatted metrics dictionary with consistent formatting."""
    print(f"\n{'='*50}")
    print(f" {title} ")
    print(f"{'='*50}")

    for key, value in metrics.items():
        if isinstance(value, float):
            if (
                "rate" in key.lower()
                or "pct" in key.lower()
                or "percent" in key.lower()
            ):
                print(f"{key.replace('_', ' ').title()}: {value:.2f}%")
            elif (
                "profit" in key.lower()
                or "return" in key.lower()
                or "pnl" in key.lower()
            ):
                print(f"{key.replace('_', ' ').title()}: {value:.4f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value:.4f}")
        elif isinstance(value, int):
            print(f"{key.replace('_', ' ').title()}: {value:,}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")

    print(f"{'='*50}\n")
