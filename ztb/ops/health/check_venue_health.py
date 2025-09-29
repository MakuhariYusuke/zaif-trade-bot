#!/usr/bin/env python3
"""
Venue health check script for Coincheck public API connectivity.

Checks REST API connectivity, WebSocket connection, latency, and rate limit status.
Outputs JSON report or graceful degradation for offline environments.
"""

import asyncio
import json
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Add the ztb package to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "ztb"))

from ztb.utils.cli_common import CLIValidator, CommonArgs, create_standard_parser

try:
    import requests
    import websockets
    from websockets.exceptions import ConnectionClosedError, WebSocketException
except ImportError as e:
    print(f"Missing dependencies: {e}", file=sys.stderr)
    print("Install with: pip install requests websockets", file=sys.stderr)
    sys.exit(1)


class VenueHealthChecker:
    """Health checker for trading venue APIs."""

    def __init__(self, venue: str, symbol: str, timeout: int = 5):
        self.venue = venue
        self.symbol = symbol
        self.timeout = timeout

        # Coincheck API endpoints
        self.rest_base = "https://coincheck.com"
        self.ws_url = "wss://ws-api.coincheck.com/"

        # Health check results
        self.results: Dict[str, Any] = {
            "venue": venue,
            "symbol": symbol,
            "timestamp": None,
            "connectivity": {"internet": False, "rest_api": False, "websocket": False},
            "latency": {"rest_ms": None, "ws_connect_ms": None},
            "rate_limits": {"remaining": None, "reset_time": None},
            "status": "unknown",
            "errors": [],
        }

    def check_internet_connectivity(self) -> bool:
        """Check basic internet connectivity."""
        try:
            # Try to resolve a well-known host
            socket.gethostbyname("8.8.8.8")
            self.results["connectivity"]["internet"] = True
            return True
        except socket.gaierror:
            self.results["connectivity"]["internet"] = False
            return False

    def check_rest_api(self) -> bool:
        """Check REST API connectivity and measure latency."""
        try:
            # Coincheck ticker endpoint
            url = f"{self.rest_base}/api/exchange/ticker"
            start_time = time.time()

            response = requests.get(url, timeout=self.timeout)

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                self.results["connectivity"]["rest_api"] = True
                self.results["latency"]["rest_ms"] = round(latency_ms, 2)

                # Check rate limit headers if available
                remaining = response.headers.get("X-RateLimit-Remaining")
                reset_time = response.headers.get("X-RateLimit-Reset")

                if remaining:
                    self.results["rate_limits"]["remaining"] = int(remaining)
                if reset_time:
                    self.results["rate_limits"]["reset_time"] = int(reset_time)

                return True
            else:
                self.results["errors"].append(
                    f"REST API returned status {response.status_code}"
                )
                return False

        except requests.exceptions.RequestException as e:
            self.results["errors"].append(f"REST API error: {str(e)}")
            return False

    async def check_websocket(self) -> bool:
        """Check WebSocket connectivity and measure connection latency."""
        try:
            start_time = time.time()

            async with websockets.connect(
                self.ws_url, extra_headers={"User-Agent": "HealthCheck/1.0"}
            ) as websocket:
                connect_time = (time.time() - start_time) * 1000

                # Send a simple ping or subscription message
                # Coincheck WebSocket expects JSON-RPC format
                ping_msg = {
                    "jsonrpc": "2.0",
                    "method": "subscribe",
                    "params": {
                        "channel": f"btc_jpy-ticker"  # Simplified channel name
                    },
                    "id": 1,
                }

                await websocket.send(json.dumps(ping_msg))

                # Wait for response or timeout
                try:
                    response = await asyncio.wait_for(
                        websocket.recv(), timeout=self.timeout
                    )
                    # Parse response to ensure it's valid JSON
                    json.loads(response)
                    self.results["connectivity"]["websocket"] = True
                    self.results["latency"]["ws_connect_ms"] = round(connect_time, 2)
                    return True
                except (asyncio.TimeoutError, json.JSONDecodeError) as e:
                    self.results["errors"].append(f"WebSocket response error: {str(e)}")
                    return False

        except (WebSocketException, ConnectionClosedError, OSError) as e:
            self.results["errors"].append(f"WebSocket connection error: {str(e)}")
            return False

    def determine_overall_status(self) -> str:
        """Determine overall health status."""
        if not self.results["connectivity"]["internet"]:
            return "offline"

        rest_ok = self.results["connectivity"]["rest_api"]
        ws_ok = self.results["connectivity"]["websocket"]

        if rest_ok and ws_ok:
            return "healthy"
        elif rest_ok or ws_ok:
            return "degraded"
        else:
            return "unhealthy"

    async def run_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        self.results["timestamp"] = time.time()

        # Check internet first
        if not self.check_internet_connectivity():
            self.results["status"] = "offline"
            return self.results

        # Run REST and WebSocket checks concurrently
        rest_task = asyncio.create_task(asyncio.to_thread(self.check_rest_api))
        ws_task = self.check_websocket()

        await asyncio.gather(rest_task, ws_task)

        # Determine overall status
        self.results["status"] = self.determine_overall_status()

        return self.results


def main():
    parser = create_standard_parser("Check venue API health")
    parser.add_argument(
        "--venue", required=True, help="Trading venue (e.g., coincheck)"
    )
    parser.add_argument(
        "--symbol", required=True, help="Trading symbol (e.g., BTC_JPY)"
    )
    CommonArgs.add_timeout(parser)

    args = parser.parse_args()

    # Validate venue
    try:
        venue = CLIValidator.validate_venue(args.venue)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    checker = VenueHealthChecker(venue, args.symbol, args.timeout)

    try:
        results = asyncio.run(checker.run_checks())

        # Output results
        if results["status"] == "offline":
            print("SKIP: No internet connectivity detected")
            sys.exit(0)
        else:
            print(json.dumps(results, indent=2))

        # Exit with appropriate code
        if results["status"] == "healthy":
            sys.exit(0)
        elif results["status"] == "degraded":
            sys.exit(1)  # Warning
        else:
            sys.exit(2)  # Critical

    except KeyboardInterrupt:
        print("Health check interrupted", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Health check failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
