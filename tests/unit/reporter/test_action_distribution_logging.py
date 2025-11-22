"""Tests to ensure action distribution snapshots are recorded in training report events."""

from ztb.training.unified_trainer import reporting


def test_reporter_logs_action_distribution():
    reporter = reporting.TrainingReporter(None)

    # Simulate training progress with action distribution
    step = 1000
    total_steps = 2000
    stats = {"action_distribution": {"HOLD": 0.02, "BUY": 0.03, "SELL": 0.95}, "step": step}

    reporter.log_training_progress(step, total_steps, stats)

    events = reporter.get_events()
    assert any(e for e in events if e["type"] == "training_progress" and e["data"].get("action_distribution")), (
        "Reporter should store action_distribution in training_progress events"
    )
