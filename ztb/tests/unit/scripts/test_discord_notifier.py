#!/usr/bin/env python3
"""
Unit tests for Discord notification support in alert_notifier.py
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Add scripts directory to path for importing
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))

from alert_notifier import send_webhook


class TestDiscordNotifier(unittest.TestCase):
    @patch("alert_notifier.requests")
    def test_send_webhook_discord_success(self, mock_requests: Mock) -> None:
        """Test sending Discord webhook successfully."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_requests.post.return_value = mock_response

        alerts = [
            {
                "level": "ERROR",
                "message": "Test error",
                "timestamp": "2024-01-01T00:00:00",
            }
        ]
        result = send_webhook(
            "https://discord.com/api/webhooks/123/abc",
            "Test Title",
            "corr123",
            alerts,
            "discord",
        )

        self.assertTrue(result)
        mock_requests.post.assert_called_once()
        call_args = mock_requests.post.call_args
        payload = call_args[1]["json"]
        self.assertIn("content", payload)
        self.assertIn("username", payload)
        self.assertIn("Test Title", payload["content"])
        self.assertIn("corr123", payload["content"])
        self.assertIn("Test error", payload["content"])

    @patch("alert_notifier.requests")
    def test_send_webhook_discord_rate_limit_retry(self, mock_requests):
        """Test Discord webhook with rate limit retry."""
        # First call: 429, second: success
        mock_response_429 = Mock()
        mock_response_429.status_code = 429
        mock_response_429.headers.get.return_value = "1"
        mock_response_429.raise_for_status.side_effect = Exception("429")

        mock_response_200 = Mock()
        mock_response_200.status_code = 200
        mock_response_200.raise_for_status.return_value = None

        mock_requests.post.side_effect = [mock_response_429, mock_response_200]

        alerts = [{"level": "WARN", "message": "Rate limit test"}]
        result = send_webhook(
            "https://discord.com/api/webhooks/123/abc",
            "Title",
            "corr123",
            alerts,
            "discord",
        )

        self.assertTrue(result)
        self.assertEqual(mock_requests.post.call_count, 2)

    @patch("alert_notifier.requests")
    @patch("alert_notifier.time.sleep")
    def test_send_webhook_discord_max_retries(
        self, mock_sleep: Mock, mock_requests: Mock
    ) -> None:
        """Test Discord webhook fails after max retries."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Exception("Server error")
        mock_requests.post.return_value = mock_response

        alerts = [{"level": "ERROR", "message": "Test"}]
        result = send_webhook(
            "https://discord.com/api/webhooks/123/abc",
            "Title",
            "corr123",
            alerts,
            "discord",
        )

        self.assertFalse(result)
        self.assertEqual(mock_requests.post.call_count, 3)  # max_retries = 3

    @patch("alert_notifier.requests")
    @patch.dict(
        "os.environ",
        {
            "ZTB_DISCORD_USERNAME": "Custom Bot",
            "ZTB_DISCORD_AVATAR_URL": "http://example.com/avatar.png",
        },
    )
    def test_send_webhook_discord_custom_username(self, mock_requests: Mock) -> None:
        """Test Discord webhook with custom username and avatar."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_requests.post.return_value = mock_response

        alerts = [{"level": "INFO", "message": "Test"}]
        result = send_webhook(
            "https://discord.com/api/webhooks/123/abc",
            "Title",
            "corr123",
            alerts,
            "discord",
        )

        self.assertTrue(result)
        payload = mock_requests.post.call_args[1]["json"]
        self.assertEqual(payload["username"], "Custom Bot")
        self.assertEqual(payload["avatar_url"], "http://example.com/avatar.png")

    @patch("alert_notifier.requests")
    def test_send_webhook_slack_fallback(self, mock_requests: Mock) -> None:
        """Test Slack webhook still works as default."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_requests.post.return_value = mock_response

        alerts = [{"level": "ERROR", "message": "Test"}]
        result = send_webhook(
            "https://hooks.slack.com/services/123/abc",
            "Title",
            "corr123",
            alerts,
            "slack",
        )

        self.assertTrue(result)
        payload = mock_requests.post.call_args[1]["json"]
        self.assertIn("title", payload)
        self.assertIn("alerts", payload)
        self.assertEqual(payload["title"], "Title")

    @patch("alert_notifier.requests")
    def test_send_webhook_no_requests(self, mock_requests: Mock) -> None:
        """Test behavior when requests is not available."""
        # Simulate HAS_REQUESTS = False
        with patch("alert_notifier.HAS_REQUESTS", False):
            alerts = [{"level": "ERROR", "message": "Test"}]
            result = send_webhook(
                "https://discord.com/api/webhooks/123/abc",
                "Title",
                "corr123",
                alerts,
                "discord",
            )
            self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
