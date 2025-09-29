"""
Unit tests for CI utils
"""

from ztb.utils.ci_utils import collect_ci_metrics, notify_ci_results


def test_collect_ci_metrics():
    """Test collect_ci_metrics returns dict"""
    result = collect_ci_metrics()
    assert isinstance(result, dict)


def test_notify_ci_results():
    """Test notify_ci_results does not raise"""
    metrics = {"coverage": 80.0, "time": 10.0}
    notify_ci_results(metrics, "discord")  # Should not raise
