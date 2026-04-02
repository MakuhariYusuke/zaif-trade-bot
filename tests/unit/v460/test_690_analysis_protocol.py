from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import patch

from scripts.v460.analysis.analysis_common import (
    Record,
    filter_by_date_range,
    filter_by_days,
)
from scripts.v460.analysis.protocols import PROTOCOL_REGISTRY
from scripts.v460.analysis.protocols.protocol_688 import Protocol688
from scripts.v460.analysis.protocols.protocol_695_regime_as import Protocol695RegimeAS
from scripts.v460.analysis.protocols.protocol_695_trend5s import Protocol695Trend5s
from scripts.v460.analysis.run_protocol import build_parser, main


def _record(
    *,
    cycle_id: str,
    ts: float,
    side: str = "buy",
    filled: bool = True,
    pnl: float | None = 1.0,
    regime: str = "ranging",
    cancel_reason: str | None = None,
) -> Record:
    return cast(
        Record,
        {
            "cycle_id": cycle_id,
            "timestamp": ts,
            "side": side,
            "filled": filled,
            "post_fill_30s_pnl": pnl,
            "regime": regime,
            "cancel_reason": cancel_reason,
            "git_sha": "sha123",
            "spread_at_order": 1200.0,
            "adverse_selected": bool(pnl is not None and pnl < 0.0),
            "skip_gate_hour_offset": 0.2 if side == "sell" else 0.0,
        },
    )


class TestProtocol688:
    def test_protocol688_returns_result_on_empty_records(self) -> None:
        result = Protocol688().execute([])

        assert result.text_report != ""
        assert result.json_payload["protocol"] == "688"
        assert result.warnings

    def test_protocol688_json_contains_required_sections(self) -> None:
        records = [
            _record(cycle_id="a", ts=1_710_000_000.0, side="buy", pnl=1.2),
            _record(cycle_id="b", ts=1_710_000_600.0, side="sell", pnl=-0.8, regime="strong_up"),
            _record(cycle_id="c", ts=1_710_001_200.0, side="sell", filled=False, pnl=None, cancel_reason="timeout"),
        ]

        result = Protocol688().execute(records)

        assert {
            "basic",
            "side",
            "nfq",
            "adverse_selection",
            "spread",
            "hour",
            "sha",
            "regime",
            "side_regime_cross",
            "sell_hour_offset_boost",
        } <= set(result.json_payload)

    def test_protocol_registry_contains_688(self) -> None:
        assert "688" in PROTOCOL_REGISTRY

    def test_protocol_registry_contains_695_protocols(self) -> None:
        assert "695_trend5s" in PROTOCOL_REGISTRY
        assert "695_regime_as" in PROTOCOL_REGISTRY

    def test_protocol695s_execute(self) -> None:
        records = [
            _record(cycle_id="a", ts=1_710_000_000.0, side="sell", pnl=1.2),
            _record(
                cycle_id="b",
                ts=1_710_000_060.0,
                side="sell",
                filled=False,
                pnl=None,
                cancel_reason="trend_5s_sell_guard_veto",
            ),
        ]

        assert Protocol695Trend5s().execute(records).json_payload["protocol"] == "695_trend5s"
        assert Protocol695RegimeAS().execute(records).json_payload["protocol"] == "695_regime_as"


class TestAnalysisCommonFilters:
    def test_filter_by_days(self) -> None:
        records = [
            _record(cycle_id="old", ts=1_710_000_000.0),
            _record(cycle_id="new", ts=1_710_000_000.0 + 4 * 86400),
        ]

        filtered = filter_by_days(records, 2)

        assert [record["cycle_id"] for record in filtered] == ["new"]

    def test_filter_by_date_range(self) -> None:
        records = [
            _record(cycle_id="a", ts=1_711_929_600.0),  # 2024-04-01 UTC
            _record(cycle_id="b", ts=1_712_016_000.0),  # 2024-04-02 UTC
            _record(cycle_id="c", ts=1_712_102_400.0),  # 2024-04-03 UTC
        ]

        filtered = filter_by_date_range(records, "2024-04-02", "2024-04-03")

        assert [record["cycle_id"] for record in filtered] == ["b", "c"]


class TestRunProtocolCli:
    def test_list_option_prints_available_protocols(self, capsys) -> None:
        assert main(["--list"]) == 0
        out = capsys.readouterr().out
        assert "688" in out
        assert "695_trend5s" in out

    def test_cli_parser_accepts_protocol_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--protocol", "688", "--days", "1"])
        assert args.protocol == "688"
        assert args.days == 1

    def test_run_protocol_executes_with_days_filter(self, tmp_path: Path) -> None:
        records = [
            _record(cycle_id="recent", ts=1_710_000_000.0),
            _record(cycle_id="recent_sell", ts=1_710_000_600.0, side="sell", pnl=-0.5),
        ]
        with (
            patch("scripts.v460.analysis.run_protocol.load_records_with_filters", return_value=records),
            patch("scripts.v460.analysis.run_protocol.write_output") as write_output,
            patch("scripts.v460.analysis.run_protocol.write_json_output") as write_json_output,
        ):
            assert main(["--protocol", "688", "--days", "1", "--output-dir", str(tmp_path)]) == 0

        assert write_output.called
        assert write_json_output.called

    def test_run_protocol_json_mode_uses_json_writer(self, tmp_path: Path) -> None:
        with (
            patch("scripts.v460.analysis.run_protocol.load_records_with_filters", return_value=[]),
            patch("scripts.v460.analysis.run_protocol.write_output") as write_output,
            patch("scripts.v460.analysis.run_protocol.write_json_output") as write_json_output,
        ):
            assert main(["--protocol", "688", "--json", "--output-dir", str(tmp_path)]) == 0

        assert write_json_output.call_count >= 2
        assert write_output.call_count == 0
