#!/usr/bin/env python3
"""Tests for report catalog caching and lookup behavior."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ztb.reporting.services import catalog


def _write_report(
    reports_dir: Path,
    file_name: str,
    model_name: str,
    action_distribution: dict[str, object] | None = None,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "configuration": {"training": {"model_name": model_name}},
        "training_stats": {"action_distribution": action_distribution or {}},
    }
    report_path = reports_dir / file_name
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    return report_path


@pytest.fixture(autouse=True)
def clear_catalog_cache() -> None:
    catalog.clear_report_cache()
    yield
    catalog.clear_report_cache()


def test_find_reports_uses_cache_for_unchanged_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reports_dir = tmp_path / "reports"
    _write_report(reports_dir, "training_report_1.json", "model_a")
    _write_report(reports_dir, "training_report_2.json", "model_b")

    calls: list[Path] = []

    def fake_read_json(path: Path) -> object:
        path_obj = Path(path)
        calls.append(path_obj)
        return json.loads(path_obj.read_text(encoding="utf-8"))

    monkeypatch.setattr(catalog, "read_json", fake_read_json)

    matches = catalog.find_reports_for_model("model_a", reports_dir=reports_dir)
    assert len(matches) == 1
    assert len(calls) == 2

    calls.clear()
    matches_again = catalog.find_reports_for_model("model_a", reports_dir=reports_dir)
    assert len(matches_again) == 1
    assert len(calls) == 0


def test_find_reports_refreshes_cache_when_report_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reports_dir = tmp_path / "reports"
    report = _write_report(reports_dir, "training_report_1.json", "model_a")

    calls: list[Path] = []

    def fake_read_json(path: Path) -> object:
        path_obj = Path(path)
        calls.append(path_obj)
        return json.loads(path_obj.read_text(encoding="utf-8"))

    monkeypatch.setattr(catalog, "read_json", fake_read_json)

    assert len(catalog.find_reports_for_model("model_a", reports_dir=reports_dir)) == 1
    assert len(calls) == 1

    # Update file contents and mtime; cache key should change.
    report.write_text(
        json.dumps(
            {
                "configuration": {"training": {"model_name": "model_b"}},
                "training_stats": {"action_distribution": {}},
            }
        ),
        encoding="utf-8",
    )

    calls.clear()
    assert len(catalog.find_reports_for_model("model_b", reports_dir=reports_dir)) == 1
    assert len(calls) == 1


def test_get_latest_report_uses_mtime_not_lexicographic(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    older = _write_report(reports_dir, "training_report_z.json", "model_a")
    newer = _write_report(reports_dir, "training_report_a.json", "model_a")

    base_ns = 1_700_000_000_000_000_000
    os.utime(older, ns=(base_ns, base_ns))
    os.utime(newer, ns=(base_ns + 1_000_000, base_ns + 1_000_000))

    latest = catalog.get_latest_report_for_model("model_a", reports_dir=reports_dir)
    assert latest is not None
    assert latest.name == "training_report_a.json"


def test_extract_action_distribution_normalizes_values(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    report = _write_report(
        reports_dir,
        "training_report_1.json",
        "model_a",
        action_distribution={"BUY": "0.7", "SELL": 0.2, "HOLD": "invalid"},
    )

    distribution = catalog.extract_action_distribution(report)
    assert distribution["BUY"] == pytest.approx(0.7)
    assert distribution["SELL"] == pytest.approx(0.2)
    assert distribution["HOLD"] == pytest.approx(0.0)


def test_get_recent_training_reports_returns_latest_limit(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    oldest = _write_report(reports_dir, "training_report_old.json", "model_a")
    middle = _write_report(reports_dir, "training_report_mid.json", "model_a")
    newest = _write_report(reports_dir, "training_report_new.json", "model_a")

    base_ns = 1_700_000_000_000_000_000
    os.utime(oldest, ns=(base_ns, base_ns))
    os.utime(middle, ns=(base_ns + 1_000_000, base_ns + 1_000_000))
    os.utime(newest, ns=(base_ns + 2_000_000, base_ns + 2_000_000))

    recent = catalog.get_recent_training_reports(limit=2, reports_dir=reports_dir)
    assert [path.name for path in recent] == [
        "training_report_new.json",
        "training_report_mid.json",
    ]


def test_load_training_report_returns_none_for_invalid_json(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    broken = reports_dir / "training_report_broken.json"
    broken.write_text("{invalid json", encoding="utf-8")

    assert catalog.load_training_report(broken) is None


def test_extract_reward_components_supports_nested_and_normalizes_values(
    tmp_path: Path,
) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report = reports_dir / "training_report_nested.json"
    report.write_text(
        json.dumps(
            {
                "configuration": {"training": {"model_name": "model_a"}},
                "training_stats": {
                    "reward_components": {"balance_penalty": "0.25", "invalid": "x"}
                },
            }
        ),
        encoding="utf-8",
    )

    components = catalog.extract_reward_components(report)
    assert components["balance_penalty"] == pytest.approx(0.25)
    assert components["invalid"] == pytest.approx(0.0)


def test_report_model_name_cache_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reports_dir = tmp_path / "reports"
    _write_report(reports_dir, "training_report_1.json", "model_1")
    _write_report(reports_dir, "training_report_2.json", "model_2")
    _write_report(reports_dir, "training_report_3.json", "model_3")

    monkeypatch.setattr(catalog, "REPORT_MODEL_NAME_CACHE_MAX_SIZE", 2)

    catalog.find_reports_for_model("model_1", reports_dir=reports_dir)
    assert len(catalog._REPORT_MODEL_NAME_CACHE) <= 2


def test_cache_does_not_accumulate_stale_entries_for_same_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reports_dir = tmp_path / "reports"
    report = _write_report(reports_dir, "training_report_1.json", "model_1")
    monkeypatch.setattr(catalog, "REPORT_MODEL_NAME_CACHE_MAX_SIZE", 32)

    assert len(catalog.find_reports_for_model("model_1", reports_dir=reports_dir)) == 1
    initial_cache_size = len(catalog._REPORT_MODEL_NAME_CACHE)
    assert initial_cache_size == 1

    report.write_text(
        json.dumps(
            {
                "configuration": {"training": {"model_name": "model_2"}},
                "training_stats": {"action_distribution": {}},
            }
        ),
        encoding="utf-8",
    )
    assert len(catalog.find_reports_for_model("model_2", reports_dir=reports_dir)) == 1
    assert len(catalog._REPORT_MODEL_NAME_CACHE) == 1
