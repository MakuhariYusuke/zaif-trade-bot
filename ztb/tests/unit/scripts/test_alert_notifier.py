import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from ztb.ops.monitoring.alert_notifier import load_alerts, send_webhook


def test_load_alerts():
    with tempfile.TemporaryDirectory() as tmp:
        jsonl_path = Path(tmp) / "alerts.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(
                '{"timestamp": "2023-09-29T12:00:00", "level": "WARN", "alert_type": "test"}\n'
            )
            f.write(
                '{"timestamp": "2023-09-29T12:05:00", "level": "ERROR", "alert_type": "test2"}\n'
            )

        alerts = load_alerts(jsonl_path, 86400 * 365 * 10, "WARN")
        assert len(alerts) == 2


def test_load_alerts_filter_level():
    with tempfile.TemporaryDirectory() as tmp:
        jsonl_path = Path(tmp) / "alerts.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(
                '{"timestamp": "2023-09-29T12:00:00", "level": "WARN", "alert_type": "test"}\n'
            )
            f.write(
                '{"timestamp": "2023-09-29T12:05:00", "level": "ERROR", "alert_type": "test2"}\n'
            )

        alerts = load_alerts(jsonl_path, 86400 * 365 * 10, "ERROR")
        assert len(alerts) == 1
        assert alerts[0]["level"] == "ERROR"


def test_send_webhook_discord_embeds():
    with patch("ztb.ops.alert_notifier.requests") as mock_requests:
        mock_response = mock_requests.post.return_value
        mock_response.raise_for_status.return_value = None

        alerts = [{"level": "ERROR", "message": "Critical error occurred"}]
        success = send_webhook(
            "https://discord.com/api/webhooks/test",
            "Alert Title",
            "corr123",
            alerts,
            "discord",
        )
        assert success

        # Check payload structure
        call_args = mock_requests.post.call_args
        payload = call_args[1]["json"]
        assert "embeds" in payload
        embed = payload["embeds"][0]
        assert embed["title"] == "Alert Title"
        assert embed["color"] == 0xFF0000  # Red for errors
        assert "Correlation ID" in embed["fields"][0]["name"]


def test_send_webhook_discord_rate_limit_retry():
    with (
        patch("ztb.ops.alert_notifier.requests") as mock_requests,
        patch("ztb.ops.alert_notifier.time.sleep") as mock_sleep,
    ):
        # First call: 429 with Retry-After
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_response_429.headers.get.return_value = "2"
        mock_response_429.raise_for_status.side_effect = Exception("429")

        # Second call: success
        mock_response_ok = MagicMock()
        mock_response_ok.status_code = 200
        mock_response_ok.raise_for_status.return_value = None

        mock_requests.post.side_effect = [mock_response_429, mock_response_ok]

        alerts = [{"level": "WARN", "message": "Warning message"}]
        success = send_webhook(
            "https://discord.com/api/webhooks/test",
            "Test Alert",
            "corr123",
            alerts,
            "discord",
        )

        assert success
        assert mock_requests.post.call_count == 2
        mock_sleep.assert_called_with(2.0)  # Respects Retry-After header


def test_send_webhook_discord_rate_limit_no_retry_after():
    with (
        patch("ztb.ops.alert_notifier.requests") as mock_requests,
        patch("ztb.ops.alert_notifier.time.sleep") as mock_sleep,
    ):
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers.get.return_value = None  # No Retry-After
        mock_response.raise_for_status.side_effect = Exception("429")

        mock_requests.post.return_value = mock_response

        alerts = [{"level": "WARN", "message": "Warning message"}]
        success = send_webhook(
            "https://discord.com/api/webhooks/test",
            "Test Alert",
            "corr123",
            alerts,
            "discord",
        )

        assert not success  # Should fail after retries
        mock_sleep.assert_called_with(1.0)  # Default retry time
