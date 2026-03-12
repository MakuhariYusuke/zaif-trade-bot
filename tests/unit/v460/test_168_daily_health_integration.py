"""168# §8 #8: daily_health_check の stopgap/dashboard 統合テスト.

_run_stopgap_health / _run_side_regime_dashboard の単体テスト。
外部依存は mock で隔離。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from scripts.v460.daily_health_check import (
    _run_side_regime_dashboard,
    _run_stopgap_health,
    run_daily_health_check,
)


# ======================================================================
# Helpers
# ======================================================================

def _make_stopgap_report(
    *,
    total_records: int = 100,
    total_filled: int = 40,
    stopgap_checks: list | None = None,
    alerts: list | None = None,
) -> MagicMock:
    """Minimal DailyHealthReport mock from stopgap_health."""
    rpt = MagicMock()
    rpt.total_records = total_records
    rpt.total_filled = total_filled
    rpt.stopgap_checks = stopgap_checks or []
    rpt.alerts = alerts or []
    rpt.daily_metrics = []
    rpt.model_used_breakdown = []
    rpt.filters_applied = {}
    return rpt


def _make_dashboard_result(
    *,
    total_records: int = 200,
    total_filled: int = 80,
    overall_fill_rate: float = 0.4,
    side_summary: dict | None = None,
    regime_side_detail: list | None = None,
) -> dict:
    return {
        "total_records": total_records,
        "total_filled": total_filled,
        "overall_fill_rate": overall_fill_rate,
        "side_summary": side_summary or {"buy": {}, "sell": {}},
        "regime_side_detail": regime_side_detail or [],
    }


# ======================================================================
# _run_stopgap_health tests
# ======================================================================

class TestRunStopgapHealth:
    """_run_stopgap_health のテスト."""

    def test_no_records(self):
        """レコードなしのとき skipped を返す."""
        with patch("scripts.v460.daily_health_check.Path"):
            with patch(
                "scripts.v460.lib.stopgap_health.load_fill_records",
                return_value=[],
            ):
                result = _run_stopgap_health("dummy_dir")

        assert result["check"] == "stopgap_health"
        assert result["skipped"] is True
        assert result["reason"] == "no fill records"

    def test_basic_report(self):
        """基本的なレポート生成."""
        mock_rpt = _make_stopgap_report(
            total_records=100,
            total_filled=40,
            stopgap_checks=[
                {
                    "stopgap_id": "2a",
                    "verdict": "PASS",
                    "metrics": {"sell_skip_rate": 0.1},
                    "criteria": "sell_skip < 0.2",
                },
            ],
            alerts=[],
        )

        with patch(
            "scripts.v460.lib.stopgap_health.load_fill_records",
            return_value=[{"filled": True}],
        ), patch(
            "scripts.v460.lib.stopgap_health.generate_health_report",
            return_value=mock_rpt,
        ):
            result = _run_stopgap_health("dummy_dir")

        assert result["check"] == "stopgap_health"
        assert result["n_records"] == 100
        assert result["overall_fill_rate"] == 0.4
        assert result["n_alerts"] == 0
        assert len(result["exit_checks"]) == 1
        assert result["exit_checks"][0]["verdict"] == "PASS"
        assert result["n_exit_breaches"] == 0

    def test_breach_detected(self):
        """EXIT BREACH がカウントされる."""
        mock_rpt = _make_stopgap_report(
            total_records=50,
            total_filled=10,
            stopgap_checks=[
                {"stopgap_id": "2a", "verdict": "BREACH",
                 "metrics": {"val": 0.9}, "criteria": "< 0.3"},
                {"stopgap_id": "2c", "verdict": "PASS",
                 "metrics": {"val": 0.1}, "criteria": "< 0.5"},
                {"stopgap_id": "6a", "verdict": "BREACH",
                 "metrics": {"val": 0.7}, "criteria": "< 0.2"},
            ],
            alerts=[
                {"severity": "CRITICAL", "stopgap_id": "2a", "message": "breach!"},
            ],
        )

        with patch(
            "scripts.v460.lib.stopgap_health.load_fill_records",
            return_value=[{"filled": True}],
        ), patch(
            "scripts.v460.lib.stopgap_health.generate_health_report",
            return_value=mock_rpt,
        ):
            result = _run_stopgap_health("dummy_dir")

        assert result["n_exit_breaches"] == 2
        assert result["n_alerts"] == 1
        assert result["alerts"][0]["severity"] == "CRITICAL"

    def test_exception_handled(self):
        """例外時にerrorフィールドを返す."""
        with patch(
            "scripts.v460.lib.stopgap_health.load_fill_records",
            side_effect=FileNotFoundError("no dir"),
        ):
            result = _run_stopgap_health("nonexistent_dir")

        assert result["check"] == "stopgap_health"
        assert "error" in result


# ======================================================================
# _run_side_regime_dashboard tests
# ======================================================================

class TestRunSideRegimeDashboard:
    """_run_side_regime_dashboard のテスト."""

    def test_basic_dashboard(self):
        """基本的なダッシュボード生成."""
        mock_result = _make_dashboard_result(
            total_records=200,
            total_filled=80,
            overall_fill_rate=0.4,
            side_summary={"buy": {"fill_rate": 0.5}, "sell": {"fill_rate": 0.3}},
            regime_side_detail=[
                {"regime": "ranging", "side": "buy", "metrics": {}},
                {"regime": "ranging", "side": "sell", "metrics": {}},
                {"regime": "trending_up", "side": "buy", "metrics": {}},
            ],
        )

        with patch(
            "scripts.v460.analysis.side_regime_dashboard.run_dashboard",
            return_value=mock_result,
        ):
            result = _run_side_regime_dashboard("dummy_dir")

        assert result["check"] == "side_regime_dashboard"
        assert result["total_records"] == 200
        assert result["total_filled"] == 80
        assert result["overall_fill_rate"] == 0.4
        assert "buy" in result["side_summary"]
        assert result["n_regime_side_groups"] == 3

    def test_empty_dashboard(self):
        """空のダッシュボード."""
        mock_result = {
            "total_records": 0,
            "total_filled": 0,
            "overall_fill_rate": 0,
        }

        with patch(
            "scripts.v460.analysis.side_regime_dashboard.run_dashboard",
            return_value=mock_result,
        ):
            result = _run_side_regime_dashboard("dummy_dir")

        assert result["total_records"] == 0
        assert "side_summary" not in result
        assert "n_regime_side_groups" not in result

    def test_exception_handled(self):
        """例外時にerrorフィールドを返す."""
        with patch(
            "scripts.v460.analysis.side_regime_dashboard.run_dashboard",
            side_effect=RuntimeError("dashboard broke"),
        ):
            result = _run_side_regime_dashboard("dummy_dir")

        assert result["check"] == "side_regime_dashboard"
        assert "error" in result
        assert "dashboard broke" in result["error"]


# ======================================================================
# Integration: run_daily_health_check with new checks
# ======================================================================

class TestDailyHealthCheckIntegration:
    """run_daily_health_check に check 5/6 が含まれることを確認."""

    @patch("scripts.v460.daily_health_check._run_side_regime_dashboard")
    @patch("scripts.v460.daily_health_check._run_stopgap_health")
    @patch("scripts.v460.daily_health_check._run_oracle_baseline")
    @patch("scripts.v460.daily_health_check._run_gate_judgment")
    @patch("scripts.v460.daily_health_check._run_feature_freshness")
    @patch("scripts.v460.daily_health_check._run_trades_health")
    def test_6_checks_included(
        self,
        mock_trades: MagicMock,
        mock_freshness: MagicMock,
        mock_gate: MagicMock,
        mock_oracle: MagicMock,
        mock_stopgap: MagicMock,
        mock_dashboard: MagicMock,
    ):
        """6つのチェックすべてが checks リストに含まれる."""
        mock_trades.return_value = {"check": "trades_health", "healthy": True}
        mock_freshness.return_value = {"check": "feature_freshness", "fresh": True}
        mock_gate.return_value = {"check": "gate_judgment", "verdict": "PASS"}
        mock_oracle.return_value = {"check": "oracle_baseline"}
        mock_stopgap.return_value = {
            "check": "stopgap_health", "n_records": 50,
            "overall_fill_rate": 0.4, "n_exit_breaches": 0,
        }
        mock_dashboard.return_value = {
            "check": "side_regime_dashboard",
            "total_records": 100, "overall_fill_rate": 0.35,
        }

        with patch("ztb.utils.notify.get_notifier") as mock_get_notifier:
            mock_get_notifier.return_value = MagicMock()
            report = run_daily_health_check(skip_monte_carlo=True)

        assert len(report["checks"]) == 6
        check_names = [c["check"] for c in report["checks"]]
        assert "stopgap_health" in check_names
        assert "side_regime_dashboard" in check_names
        assert report["overall_healthy"] is True

    @patch("scripts.v460.daily_health_check._run_side_regime_dashboard")
    @patch("scripts.v460.daily_health_check._run_stopgap_health")
    @patch("scripts.v460.daily_health_check._run_oracle_baseline")
    @patch("scripts.v460.daily_health_check._run_gate_judgment")
    @patch("scripts.v460.daily_health_check._run_feature_freshness")
    @patch("scripts.v460.daily_health_check._run_trades_health")
    def test_stopgap_breach_makes_unhealthy(
        self,
        mock_trades: MagicMock,
        mock_freshness: MagicMock,
        mock_gate: MagicMock,
        mock_oracle: MagicMock,
        mock_stopgap: MagicMock,
        mock_dashboard: MagicMock,
    ):
        """Stopgap BREACH で overall_healthy = False."""
        mock_trades.return_value = {"check": "trades_health", "healthy": True}
        mock_freshness.return_value = {"check": "feature_freshness", "fresh": True}
        mock_gate.return_value = {"check": "gate_judgment", "verdict": "PASS"}
        mock_oracle.return_value = {"check": "oracle_baseline"}
        mock_stopgap.return_value = {
            "check": "stopgap_health", "n_records": 50,
            "overall_fill_rate": 0.2, "n_exit_breaches": 2,
        }
        mock_dashboard.return_value = {
            "check": "side_regime_dashboard",
            "total_records": 50, "overall_fill_rate": 0.2,
        }

        with patch("ztb.utils.notify.get_notifier") as mock_get_notifier:
            mock_get_notifier.return_value = MagicMock()
            report = run_daily_health_check(skip_monte_carlo=True)

        assert report["overall_healthy"] is False
