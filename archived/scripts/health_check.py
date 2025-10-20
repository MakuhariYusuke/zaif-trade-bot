#!/usr/bin/env python3
"""
Zaif Trade Bot Health Check CLI Tool.

Usage:
    python scripts/health_check.py [--format json|table] [--verbose]

This tool performs comprehensive health checks on the trading bot system
and reports the results in various formats.
"""

import argparse
import json
import sys
from typing import Any, Dict

import colorama
from colorama import Fore, Style

# Add project root to path
sys.path.insert(0, ".")

from ztb.ops.health import run_health_check


def format_as_table(results: Dict[str, Any]) -> str:
    """Format health check results as a colored table."""
    colorama.init(autoreset=True)

    lines = []
    lines.append("Zaif Trade Bot Health Check Results")
    lines.append("=" * 50)
    lines.append("")

    # Overall status
    status = results["status"]
    if status == "healthy":
        status_color = Fore.GREEN
    elif status == "warning":
        status_color = Fore.YELLOW
    else:
        status_color = Fore.RED

    lines.append(f"Overall Status: {status_color}{status.upper()}{Style.RESET_ALL}")
    lines.append(f"Total Checks: {results['total_checks']}")
    lines.append(f"Healthy: {Fore.GREEN}{results['healthy']}{Style.RESET_ALL}")
    lines.append(f"Warning: {Fore.YELLOW}{results['warning']}{Style.RESET_ALL}")
    lines.append(f"Critical: {Fore.RED}{results['critical']}{Style.RESET_ALL}")
    lines.append("")

    # Individual checks
    lines.append("Detailed Results:")
    lines.append("-" * 50)

    for check in results["checks"]:
        status = check["status"]
        if status == "healthy":
            status_icon = f"{Fore.GREEN}✓{Style.RESET_ALL}"
        elif status == "warning":
            status_icon = f"{Fore.YELLOW}⚠{Style.RESET_ALL}"
        else:
            status_icon = f"{Fore.RED}✗{Style.RESET_ALL}"

        lines.append(f"{status_icon} {check['name']}: {check['message']}")

        if check.get("details"):
            for key, value in check["details"].items():
                lines.append(f"    {key}: {value}")

        lines.append("")

    # Performance monitoring section
    if "performance" in results:
        perf = results["performance"]
        lines.append("")
        lines.append("Performance Monitoring:")
        lines.append("-" * 50)

        # Current snapshot
        current = perf["current_snapshot"]
        lines.append("Current System State:")
        lines.append(f"  CPU Usage: {current['cpu_percent']:.1f}%")
        lines.append(f"  Memory Usage: {current['memory_percent']:.1f}%")
        lines.append(f"  Disk Usage: {current['disk_usage_percent']:.1f}%")
        lines.append(
            f"  Network Sent: {current['network_bytes_sent'] / (1024**2):.1f} MB"
        )
        lines.append(
            f"  Network Received: {current['network_bytes_recv'] / (1024**2):.1f} MB"
        )
        lines.append("")

        # Performance trends
        if perf["trends"]:
            lines.append("Performance Trends:")
            concerning_count = 0

            for trend in perf["trends"]:
                if trend["is_concerning"]:
                    concerning_count += 1
                    trend_icon = f"{Fore.RED}⚠{Style.RESET_ALL}"
                else:
                    trend_icon = f"{Fore.GREEN}✓{Style.RESET_ALL}"

                direction_icon = {
                    "increasing": "📈",
                    "decreasing": "📉",
                    "stable": "➡️",
                }.get(trend["trend_direction"], "➡️")

                lines.append(
                    f"{trend_icon} {trend['metric']}: {direction_icon} {trend['trend_direction'].title()}"
                )
                lines.append(f"    Current: {trend['current_value']:.1f}%")
                lines.append(f"    24h Avg: {trend['average_24h']:.1f}%")
                lines.append(f"    7d Avg: {trend['average_7d']:.1f}%")
                lines.append(f"    Analysis: {trend['analysis']}")
                lines.append("")

            if concerning_count > 0:
                lines.append(
                    f"{Fore.RED}⚠ {concerning_count} concerning performance trend(s) detected{Style.RESET_ALL}"
                )
            else:
                lines.append(
                    f"{Fore.GREEN}✓ All performance metrics within normal ranges{Style.RESET_ALL}"
                )
        else:
            lines.append("Performance trends: No historical data available yet")
            lines.append("Trends will be available after collecting more data points.")

        # Summary
        summary = perf["summary"]
        lines.append("")
        lines.append(
            f"Performance Summary: {summary['data_points']} data points, {summary['concerning_trends']} concerning trends"
        )

    return "\n".join(lines)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Zaif Trade Bot Health Check Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/health_check.py                    # Table format (default)
  python scripts/health_check.py --format json     # JSON format
  python scripts/health_check.py --verbose         # Verbose table output
        """,
    )

    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output with detailed information",
    )

    args = parser.parse_args()

    try:
        # Run health check
        results = run_health_check()

        if args.format == "json":
            # JSON output
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            # Table output
            output = format_as_table(results)
            print(output)

        # Exit with appropriate code
        if results["status"] == "critical":
            sys.exit(2)  # Critical
        elif results["status"] == "warning":
            sys.exit(1)  # Warning
        else:
            sys.exit(0)  # Healthy

    except Exception as e:
        print(f"Error running health check: {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
