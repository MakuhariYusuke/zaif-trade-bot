"""
Unit tests for venue health check script.
"""

import asyncio
import socket

# Add scripts directory to path for imports
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

scripts_dir = Path(__file__).parent.parent.parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from check_venue_health import VenueHealthChecker


class TestVenueHealthChecker:
    """Test cases for VenueHealthChecker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.checker = VenueHealthChecker("coincheck", "BTC_JPY", timeout=5)

    def test_init(self):
        """Test initialization."""
        assert self.checker.venue == "coincheck"
        assert self.checker.symbol == "BTC_JPY"
        assert self.checker.timeout == 5
        assert self.checker.results["venue"] == "coincheck"
        assert self.checker.results["symbol"] == "BTC_JPY"

    @patch("socket.gethostbyname")
    def test_check_internet_connectivity_success(self, mock_gethostbyname):
        """Test successful internet connectivity check."""
        mock_gethostbyname.return_value = "8.8.8.8"

        result = self.checker.check_internet_connectivity()

        assert result is True
        assert self.checker.results["connectivity"]["internet"] is True

    @patch("socket.gethostbyname")
    def test_check_internet_connectivity_failure(self, mock_gethostbyname):
        """Test failed internet connectivity check."""
        mock_gethostbyname.side_effect = socket.gaierror("Name resolution failure")

        result = self.checker.check_internet_connectivity()

        assert result is False
        assert self.checker.results["connectivity"]["internet"] is False

    @patch("requests.get")
    def test_check_rest_api_success(self, mock_get):
        """Test successful REST API check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {
            "X-RateLimit-Remaining": "299",
            "X-RateLimit-Reset": "1640995200",
        }
        mock_get.return_value = mock_response

        result = self.checker.check_rest_api()

        assert result is True
        assert self.checker.results["connectivity"]["rest_api"] is True
        assert self.checker.results["latency"]["rest_ms"] is not None
        assert self.checker.results["rate_limits"]["remaining"] == 299
        assert self.checker.results["rate_limits"]["reset_time"] == 1640995200

    @patch("requests.get")
    def test_check_rest_api_failure(self, mock_get):
        """Test failed REST API check."""
        import requests

        mock_get.side_effect = requests.exceptions.RequestException(
            "Connection timeout"
        )

        result = self.checker.check_rest_api()

        assert result is False
        assert self.checker.results["connectivity"]["rest_api"] is False
        assert len(self.checker.results["errors"]) > 0

    @pytest.mark.asyncio
    async def test_check_websocket_success(self):
        """Test successful WebSocket check."""
        with patch("websockets.connect") as mock_connect:
            mock_websocket = AsyncMock()
            mock_websocket.__aenter__ = AsyncMock(return_value=mock_websocket)
            mock_websocket.__aexit__ = AsyncMock(return_value=None)
            mock_websocket.send = AsyncMock()
            mock_websocket.recv = AsyncMock(
                return_value='{"jsonrpc": "2.0", "result": "ok"}'
            )

            mock_connect.return_value = mock_websocket

            result = await self.checker.check_websocket()

            assert result is True
            assert self.checker.results["connectivity"]["websocket"] is True
            assert self.checker.results["latency"]["ws_connect_ms"] is not None

    @pytest.mark.asyncio
    async def test_check_websocket_failure(self):
        """Test failed WebSocket check."""
        from websockets.exceptions import WebSocketException

        with patch("websockets.connect") as mock_connect:
            mock_connect.side_effect = WebSocketException("Connection failed")

            result = await self.checker.check_websocket()

            assert result is False
            assert self.checker.results["connectivity"]["websocket"] is False
            assert len(self.checker.results["errors"]) > 0

    def test_determine_overall_status_offline(self):
        """Test status determination for offline."""
        self.checker.results["connectivity"]["internet"] = False

        status = self.checker.determine_overall_status()

        assert status == "offline"

    def test_determine_overall_status_healthy(self):
        """Test status determination for healthy."""
        self.checker.results["connectivity"]["internet"] = True
        self.checker.results["connectivity"]["rest_api"] = True
        self.checker.results["connectivity"]["websocket"] = True

        status = self.checker.determine_overall_status()

        assert status == "healthy"

    def test_determine_overall_status_degraded(self):
        """Test status determination for degraded."""
        self.checker.results["connectivity"]["internet"] = True
        self.checker.results["connectivity"]["rest_api"] = True
        self.checker.results["connectivity"]["websocket"] = False

        status = self.checker.determine_overall_status()

        assert status == "degraded"

    def test_determine_overall_status_unhealthy(self):
        """Test status determination for unhealthy."""
        self.checker.results["connectivity"]["internet"] = True
        self.checker.results["connectivity"]["rest_api"] = False
        self.checker.results["connectivity"]["websocket"] = False

        status = self.checker.determine_overall_status()

        assert status == "unhealthy"

    @pytest.mark.asyncio
    async def test_run_checks_offline(self):
        """Test full check run when offline."""
        with patch("socket.gethostbyname", side_effect=socket.gaierror("No internet")):
            results = await self.checker.run_checks()

            assert results["status"] == "offline"
            assert results["connectivity"]["internet"] is False

    @pytest.mark.asyncio
    async def test_run_checks_online(self):
        """Test full check run when online."""
        with (
            patch("socket.gethostbyname", return_value="8.8.8.8"),
            patch("asyncio.to_thread") as mock_to_thread,
            patch.object(
                self.checker, "check_websocket", new_callable=AsyncMock
            ) as mock_ws,
        ):
            # Mock REST API success
            async def mock_rest_check():
                self.checker.results["connectivity"]["rest_api"] = True
                self.checker.results["latency"]["rest_ms"] = 150.5
                return True

            mock_to_thread.return_value = asyncio.create_task(mock_rest_check())

            # Mock WebSocket success
            mock_ws.return_value = True
            self.checker.results["connectivity"]["websocket"] = True
            self.checker.results["latency"]["ws_connect_ms"] = 200.3

            results = await self.checker.run_checks()

            assert results["status"] == "healthy"
            assert results["connectivity"]["internet"] is True
            assert results["connectivity"]["rest_api"] is True
            assert results["connectivity"]["websocket"] is True
            assert results["latency"]["rest_ms"] == 150.5
            assert results["latency"]["ws_connect_ms"] == 200.3


def test_main_unsupported_venue(capsys):
    """Test main function with unsupported venue."""
    with patch(
        "sys.argv",
        ["check_venue_health.py", "--venue", "unsupported", "--symbol", "BTC_JPY"],
    ):
        with pytest.raises(SystemExit) as exc_info:
            from check_venue_health import main

            main()

        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "Unsupported venue" in captured.err


def test_main_offline_mode(capsys):
    """Test main function in offline mode."""
    with patch("socket.gethostbyname", side_effect=socket.gaierror("No internet")):
        with patch(
            "sys.argv",
            ["check_venue_health.py", "--venue", "coincheck", "--symbol", "BTC_JPY"],
        ):
            with pytest.raises(SystemExit) as exc_info:
                from check_venue_health import main

                asyncio.run(main())

            assert exc_info.value.code == 0

            captured = capsys.readouterr()
            assert "SKIP: No internet connectivity detected" in captured.out
