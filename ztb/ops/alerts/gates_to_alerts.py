#!/usr/bin/env python3
"""
Gate alerts glue for Zaif Trade Bot.

Reads gates.json and sends alerts for failures to webhook.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Add the ztb package to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "ztb"))

from ztb.utils.cli_common import CLIFormatter, CommonArgs, create_standard_parser

try:
    from ztb.ops.monitoring.alert_notifier import send_webhook

    has_alert_notifier = True
except ImportError:
    send_webhook: Optional[Callable[..., bool]] = None  # type: ignore[no-redef]
    has_alert_notifier = False


def load_gates(gates_path: Path) -> Dict[str, Any]:
    """Load gates.json."""
    if not gates_path.exists():
        return {}

    try:
        with open(gates_path, "r") as f:
            return json.load(f)  # type: ignore[no-any-return]
    except Exception:
        return {}


def check_gate_failures(
    gates_data: Dict[str, Any], correlation_id: str
) -> List[Dict[str, Any]]:
    """Check for gate failures and create alerts."""
    alerts = []
    gates = gates_data.get("gates", [])

    for gate in gates:
        name = gate.get("name", "unknown")
        status = gate.get("status", "unknown")
        error = gate.get("error", "")

        if status.lower() == "fail":
            level = "ERROR" if "critical" in error.lower() else "WARN"
            alerts.append(
                {
                    "title": f"Gate Failure: {name}",
                    "message": f"Gate {name} failed: {error}",
                    "level": level,
                    "correlation_id": correlation_id,
                    "timestamp": datetime.now().isoformat(),
                }
            )

    return alerts


def send_alerts(
    alerts: List[Dict[str, Any]],
    webhook_url: str,
    correlation_id: str,
    platform: str = "discord",
) -> int:
    """Send compact alerts to webhook."""
    if not alerts:
        return 0

    if not has_alert_notifier:
        print("alert_notifier not available", file=sys.stderr)
        return 1

    if not webhook_url:
        print("No webhook URL provided", file=sys.stderr)
        return 1

    # Create compact alert with failing gates list
    failing_gates = [a["title"].replace("Gate Failure: ", "") for a in alerts]
    gate_list = ", ".join(failing_gates[:3])  # Limit to 3 for brevity
    if len(failing_gates) > 3:
        gate_list += f" (+{len(failing_gates) - 3} more)"

    # Quick link to artifacts
    artifacts_url = f"https://example.com/artifacts/{correlation_id}"  # Placeholder, can be configured

    compact_alert = {
        "title": f"Gate Failures in {correlation_id}",
        "message": f"Failing gates: {gate_list}\nArtifacts: {artifacts_url}",
        "level": "ERROR" if any(a["level"] == "ERROR" for a in alerts) else "WARN",
        "correlation_id": correlation_id,
        "timestamp": datetime.now().isoformat(),
    }

    title = f"🚨 Gate Alert: {correlation_id}"
    assert send_webhook is not None
    success = send_webhook(
        webhook_url, title, correlation_id, [compact_alert], platform
    )

    return 0 if success else 1


def main() -> int:
    parser = create_standard_parser("Send gate failure alerts")
    CommonArgs.add_correlation_id(parser)
    parser.add_argument(
        "--webhook",
        help=CLIFormatter.format_help("Slack webhook URL", "SLACK_WEBHOOK_URL env"),
    )
    parser.add_argument(
        "--discord-webhook",
        help=CLIFormatter.format_help("Discord webhook URL", "ZTB_DISCORD_WEBHOOK env"),
    )
    CommonArgs.add_artifacts_dir(parser)

    args = parser.parse_args()

    # Determine webhook and platform
    webhook_url = args.webhook or os.getenv("SLACK_WEBHOOK_URL")
    discord_webhook_url = args.discord_webhook or os.getenv("ZTB_DISCORD_WEBHOOK")

    if discord_webhook_url:
        webhook_url = discord_webhook_url
        platform = "discord"
    else:
        platform = "slack"

    gates_path = (
        Path(args.artifacts_dir) / args.correlation_id / "reports" / "gates.json"
    )

    gates_data = load_gates(gates_path)
    if not gates_data:
        print("No gates.json found, skipping")
        return 0

    alerts = check_gate_failures(gates_data, args.correlation_id)
    if not alerts:
        print("No gate failures detected")
        return 0

    if not webhook_url:
        print("No webhook URL configured", file=sys.stderr)
        return 1

    return send_alerts(alerts, webhook_url, args.correlation_id, platform)


if __name__ == "__main__":
    sys.exit(main())
