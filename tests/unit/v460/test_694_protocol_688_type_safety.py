from __future__ import annotations

from unittest.mock import patch

import pytest

from scripts.v460.analysis.protocols.protocol_688 import (
    Protocol688,
    Protocol688Config,
    _adverse_selection_payload,
    _spread_payload,
)
from scripts.v460.analysis.run_protocol import main


class TestProtocol688Config:
    def test_protocol_688_config_defaults(self) -> None:
        config = Protocol688Config()
        assert config.as_rate_warn_threshold == pytest.approx(0.25)
        assert config.as_rate_alert_threshold == pytest.approx(0.35)
        assert config.pnl_warn_threshold_bps == pytest.approx(-0.5)
        assert config.spread_bucket_edges == (1500.0, 2500.0, 3500.0)
        assert config.min_section_samples == 5

    def test_section_with_empty_records(self) -> None:
        payload = Protocol688().execute([]).json_payload
        assert isinstance(payload["basic"], dict)
        assert isinstance(payload["spread"], dict)

    def test_section_with_missing_fields(self) -> None:
        records = [{"cycle_id": "x", "timestamp": 1.0}]
        spread_payload = _spread_payload(records, config=Protocol688Config())
        adverse_payload = _adverse_selection_payload(records, config=Protocol688Config())
        assert isinstance(spread_payload, dict)
        assert adverse_payload["count"] == 0

    def test_protocol_execute_embeds_config_payload(self) -> None:
        payload = Protocol688().execute([]).json_payload
        config_payload = payload["config"]
        assert isinstance(config_payload, dict)
        assert config_payload["min_section_samples"] == 5


class TestRunProtocolValidation:
    def test_run_protocol_invalid_protocol_id(self) -> None:
        with pytest.raises(SystemExit):
            main(["--protocol", "does_not_exist"])

    def test_run_protocol_invalid_days(self) -> None:
        with pytest.raises(SystemExit):
            main(["--protocol", "688", "--days", "0"])

    def test_run_protocol_protocol_failure_is_reported(self) -> None:
        with (
            patch("scripts.v460.analysis.run_protocol.load_records_with_filters", return_value=[]),
            patch("scripts.v460.analysis.run_protocol.PROTOCOL_REGISTRY", {"688": _BrokenProtocol}),
        ):
            with pytest.raises(SystemExit):
                main(["--protocol", "688"])


class _BrokenProtocol:
    description = "broken"

    def __call__(self) -> _BrokenProtocol:
        return self

    def execute(self, records: list[dict[str, object]], *, output_dir=None):
        del records, output_dir
        raise ValueError("boom")
