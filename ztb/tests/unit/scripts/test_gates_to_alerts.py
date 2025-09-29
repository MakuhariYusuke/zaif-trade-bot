import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ztb.ops.alerts.gates_to_alerts import check_gate_failures, load_gates, send_alerts


class TestGatesToAlerts(unittest.TestCase):
    def test_load_gates_success(self):
        """Test loading gates.json."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = {"gates": [{"name": "gate1", "status": "pass"}]}
            json.dump(data, f)
            f.flush()
            temp_path = Path(f.name)

        try:
            result = load_gates(temp_path)
        finally:
            temp_path.unlink()

        self.assertEqual(result["gates"][0]["name"], "gate1")

    def test_load_gates_missing(self):
        """Test loading non-existent gates.json."""
        result = load_gates(Path("nonexistent.json"))
        self.assertEqual(result, {})

    def test_check_gate_failures(self):
        """Test checking for gate failures."""
        gates_data = {
            "gates": [
                {"name": "gate1", "status": "pass"},
                {"name": "gate2", "status": "fail", "error": "test error"},
                {"name": "gate3", "status": "fail", "error": "critical error"},
            ]
        }

        alerts = check_gate_failures(gates_data, "test_id")

        self.assertEqual(len(alerts), 2)
        self.assertEqual(alerts[0]["level"], "WARN")
        self.assertEqual(alerts[1]["level"], "ERROR")
        self.assertIn("test error", alerts[0]["message"])
        self.assertIn("critical error", alerts[1]["message"])

    @patch("ztb.scripts.gates_to_alerts.send_webhook")
    def test_send_alerts_compact_success(self, mock_send):
        """Test sending compact alerts."""
        mock_send.return_value = True
        alerts = [
            {"title": "Gate Failure: gate1", "level": "WARN", "message": "test"},
            {"title": "Gate Failure: gate2", "level": "ERROR", "message": "critical"},
            {"title": "Gate Failure: gate3", "level": "WARN", "message": "another"},
        ]

        result = send_alerts(
            alerts, "https://discord.com/webhook", "test_id", "discord"
        )
        self.assertEqual(result, 0)
        mock_send.assert_called_once()

        # Check compact format
        args, kwargs = mock_send.call_args
        self.assertEqual(args[0], "https://discord.com/webhook")
        self.assertEqual(args[1], "🚨 Gate Alert: test_id")
        compact_alert = args[3][0]
        self.assertIn("gate1, gate2, gate3", compact_alert["message"])
        self.assertIn("https://example.com/artifacts/test_id", compact_alert["message"])
        self.assertEqual(compact_alert["level"], "ERROR")  # Highest level

    @patch("ztb.scripts.gates_to_alerts.send_webhook")
    def test_send_alerts_failure(self, mock_send):
        """Test sending alerts with failure."""
        mock_send.return_value = False

        alerts = [{"title": "Gate Failure: gate1", "level": "ERROR", "message": "test"}]
        result = send_alerts(
            alerts, "https://discord.com/webhook", "test_id", "discord"
        )
        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
