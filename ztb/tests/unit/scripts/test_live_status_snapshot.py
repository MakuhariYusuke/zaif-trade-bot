#!/usr/bin/env python3
"""
Unit tests for live_status_snapshot.py
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ztb.ops.reports.live_status_snapshot import create_status_embed, load_json_file


class TestLiveStatusSnapshot(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create mock trade-state.json
        self.trade_state = {
            "phase": 2,
            "consecutiveDays": 5,
            "totalSuccess": 15,
            "lastDate": "2025-09-29",
            "ordersPerDay": 3,
        }
        (self.temp_dir / "trade-state.json").write_text(json.dumps(self.trade_state))

        # Create mock trade-config.json
        self.trade_config = {
            "pair": "btc_jpy",
            "phase": 2,
            "maxOrdersPerDay": 5,
            "maxLossPerDay": 2,
            "slippageGuardPct": 1e-7,
        }
        (self.temp_dir / "trade-config.json").write_text(json.dumps(self.trade_config))

        # Create mock stats.json
        self.stats = {
            "date": "2025-09-29",
            "data": [
                {
                    "pair": "btc_jpy",
                    "stats": {"trades": 10, "wins": 7, "realizedPnl": 15000},
                },
                {
                    "pair": "eth_jpy",
                    "stats": {"trades": 5, "wins": 3, "realizedPnl": 5000},
                },
            ],
        }
        (self.temp_dir / "stats.json").write_text(json.dumps(self.stats))

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    @patch("ztb.scripts.live_status_snapshot.Path")
    def test_create_status_embed_full_data(self, mock_path):
        """Test creating embed with all data available."""

        # Mock Path to return paths in temp dir
        def path_side_effect(path_str):
            return self.temp_dir / path_str

        mock_path.side_effect = path_side_effect

        embed = create_status_embed()

        self.assertEqual(embed["title"], "🤖 Live Bot Status Snapshot")
        self.assertEqual(embed["color"], 0x00FF00)

        fields = embed["fields"]
        self.assertEqual(len(fields), 5)  # trade state, config, summary, btc, eth

        # Check trade state field
        trade_field = fields[0]
        self.assertEqual(trade_field["name"], "📊 Trade State")
        self.assertIn("Phase: 2", trade_field["value"])
        self.assertIn("Consecutive Days: 5", trade_field["value"])

        # Check config field
        config_field = fields[1]
        self.assertEqual(config_field["name"], "⚙️ Configuration")
        self.assertIn("Active Pair: btc_jpy", config_field["value"])

        # Check summary field
        summary_field = fields[2]
        self.assertEqual(summary_field["name"], "📈 Performance Summary")
        self.assertIn("Total Trades: 15", summary_field["value"])
        self.assertIn("Total Wins: 10", summary_field["value"])
        self.assertIn("Total P&L: ¥20,000", summary_field["value"])

    @patch("ztb.scripts.live_status_snapshot.Path")
    @patch("ztb.scripts.live_status_snapshot.send_webhook")
    def test_main_success(self, mock_send_webhook, mock_path):
        """Test main function with successful webhook send."""
        mock_path.cwd.return_value = self.temp_dir
        mock_send_webhook.return_value = True

        with patch(
            "sys.argv",
            ["live_status_snapshot.py", "https://discord.com/api/webhooks/..."],
        ):
            with patch("sys.exit") as mock_exit:
                from ztb.ops.reports.live_status_snapshot import main

                main()

                # Should not exit with error
                mock_exit.assert_not_called()

                # Should call send_webhook once
                self.assertEqual(mock_send_webhook.call_count, 1)
                args, kwargs = mock_send_webhook.call_args
                webhook_url = args[0]
                payload = args[1]

                self.assertEqual(webhook_url, "https://discord.com/api/webhooks/...")
                self.assertIn("embeds", payload)
                self.assertEqual(len(payload["embeds"]), 1)

    @patch("ztb.scripts.live_status_snapshot.send_webhook")
    def test_main_webhook_failure(self, mock_send_webhook):
        """Test main function with webhook failure."""
        mock_send_webhook.return_value = False

        with patch(
            "sys.argv",
            ["live_status_snapshot.py", "https://discord.com/api/webhooks/..."],
        ):
            with patch("sys.exit") as mock_exit:
                from ztb.ops.reports.live_status_snapshot import main

                main()

                # Should exit with error
                mock_exit.assert_called_once_with(1)

    def test_load_json_file_exists(self):
        """Test loading existing JSON file."""
        test_file = self.temp_dir / "test.json"
        test_data = {"key": "value"}
        test_file.write_text(json.dumps(test_data))

        result = load_json_file(test_file)
        self.assertEqual(result, test_data)

    def test_load_json_file_not_exists(self):
        """Test loading non-existent JSON file."""
        result = load_json_file(self.temp_dir / "nonexistent.json")
        self.assertIsNone(result)

    def test_load_json_file_invalid_json(self):
        """Test loading invalid JSON file."""
        test_file = self.temp_dir / "invalid.json"
        test_file.write_text("invalid json")

        result = load_json_file(test_file)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
