#!/usr/bin/env python3
"""
Compatibility wrapper for external integrations.

Provides standardized JSON interface for external tools to interact with zaif-trade-bot.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

# ztb モジュールのインポートのためパスを追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from ztb.ops.reports.live_status_snapshot import create_status_embed
except ImportError as e:
    create_status_embed = None
    IMPORT_ERROR_MSG = str(e)


def run_command_safely(cmd: str, timeout: int = 300) -> Dict[str, Any]:
    """Run a command safely and return structured result."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path.cwd(),
        )

        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "command": cmd,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Command timed out after {timeout}s",
            "command": cmd,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "command": cmd}


def handle_status_request(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle status snapshot request."""
    if create_status_embed is None:
        return {
            "success": False,
            "action": "status",
            "error": f"Failed to import create_status_embed: {IMPORT_ERROR_MSG}",
        }
    try:
        embed = create_status_embed()

        # Convert embed to plain dict for JSON serialization
        status = {
            "title": embed.get("title"),
            "color": embed.get("color"),
            "fields": embed.get("fields", []),
            "timestamp": embed.get("timestamp"),
        }

        return {"success": True, "action": "status", "data": status}
    except Exception as e:
        return {"success": False, "action": "status", "error": str(e)}


def handle_run_request(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle run command request."""
    cmd = params.get("command")
    if not cmd:
        return {
            "success": False,
            "action": "run",
            "error": "Missing 'command' parameter",
        }

    timeout = params.get("timeout", 300)

    result = run_command_safely(cmd, timeout)
    result["action"] = "run"

    return result


def handle_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Handle a single request."""
    action = request.get("action")
    params = request.get("params", {})

    if action == "status":
        return handle_status_request(params)
    elif action == "run":
        return handle_run_request(params)
    else:
        return {
            "success": False,
            "action": action,
            "error": f"Unknown action: {action}",
        }


def main():
    """Main entry point - read JSON from stdin, write JSON to stdout."""
    try:
        # Read JSON request from stdin (read one line to avoid blocking)
        input_data = sys.stdin.readline()
        if not input_data.strip():
            request = {}
        else:
            request = json.loads(input_data)

        # Handle request
        response = handle_request(request)

        # Write JSON response to stdout
        print(json.dumps(response, ensure_ascii=False, indent=2))

        # Exit with appropriate code
        sys.exit(int(not response.get("success", False)))

    except json.JSONDecodeError as e:
        error_response = {"success": False, "error": f"Invalid JSON input: {e}"}
        print(json.dumps(error_response, ensure_ascii=False), file=sys.stdout)
        sys.exit(1)
    except Exception as e:
        error_response = {"success": False, "error": f"Unexpected error: {e}"}
        print(json.dumps(error_response, ensure_ascii=False), file=sys.stdout)
        sys.exit(1)


if __name__ == "__main__":
    main()
