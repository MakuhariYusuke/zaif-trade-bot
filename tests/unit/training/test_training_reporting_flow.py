from __future__ import annotations

from unittest.mock import Mock

from ztb.training.unified_trainer.reporting import (
    persist_ensemble_report,
    persist_training_report,
)


def test_persist_training_report_runs_generate_save_and_summary() -> None:
    reporter = Mock()
    reporter.generate_report.return_value = {"training_stats": {"reward": 1.0}}
    reporter.save_report.return_value = "reports/training_report.json"

    report, report_path = persist_training_report(
        reporter,
        {"algorithm": "sac"},
        {"reward": 1.0},
        True,
    )

    assert report == {"training_stats": {"reward": 1.0}}
    assert report_path == "reports/training_report.json"
    reporter.generate_report.assert_called_once()
    reporter.save_report.assert_called_once_with(
        {"training_stats": {"reward": 1.0}},
        output_dir="reports",
    )
    reporter.print_summary.assert_called_once_with({"training_stats": {"reward": 1.0}})


def test_persist_ensemble_report_handles_empty_report() -> None:
    reporter = Mock()
    reporter.generate_ensemble_report.return_value = {}

    report, report_path = persist_ensemble_report(
        reporter,
        {"members": 3},
        [{"member": "a"}],
    )

    assert report == {}
    assert report_path == ""
    reporter.save_ensemble_report.assert_not_called()
