"""
Canary Harness Tests with Failure Injection

Tests system resilience using table-driven failure injection scenarios.
Runs in paper/replay mode to ensure safe testing.
"""

import asyncio
import json
import os
import pytest
import time
from pathlib import Path
from typing import Dict, Any

from ztb.utils.fault_injection import FaultInjectionConfig, get_fault_injector, inject_fault


class TestCanaryHarness:
    """Test canary harness with failure injection."""

    @pytest.fixture
    def canary_cases(self):
        """Load canary test cases from JSON."""
        cases_path = Path(__file__).parent / "canary_cases.json"
        with open(cases_path, 'r') as f:
            return [FaultInjectionConfig(**case) for case in json.load(f)]

    @pytest.fixture
    def fault_injector(self):
        """Get fault injector instance."""
        return get_fault_injector()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("case", [
        {"name": "ws_disconnect_transient", "type": "ws_disconnect", "duration_s": 2.0, "severity": 0.3, "expected_action": "continue"},
        {"name": "network_delay_spike", "type": "network_delay", "duration_s": 1.0, "severity": 0.8, "expected_action": "pause"},
        {"name": "data_gap_short", "type": "data_gap", "duration_s": 3.0, "severity": 0.5, "expected_action": "continue"},
        {"name": "duplicate_ticks_burst", "type": "duplicate_ticks", "duration_s": 1.0, "severity": 0.6, "expected_action": "continue"},
        {"name": "slow_disk_checkpoint", "type": "slow_disk", "duration_s": 5.0, "severity": 0.7, "expected_action": "pause"},
        {"name": "cpu_pause_brief", "type": "cpu_pause", "duration_s": 0.5, "severity": 0.4, "expected_action": "continue"},
        {"name": "corrupted_checkpoint_retry", "type": "corrupted_checkpoint", "duration_s": 1.0, "severity": 0.9, "expected_action": "resume"},
        {"name": "stream_throttle_high", "type": "stream_throttle", "duration_s": 4.0, "severity": 0.8, "expected_action": "pause"}
    ], ids=lambda x: x["name"])
    async def test_fault_injection_scenario(self, case, fault_injector, caplog):
        """Test individual fault injection scenario."""
        config = FaultInjectionConfig(**case)

        start_time = time.time()

        # Inject fault
        async with await fault_injector.inject_fault(config):
            # Simulate some work during fault
            await asyncio.sleep(0.1)

        end_time = time.time()
        duration = end_time - start_time

        # Verify fault was active
        assert fault_injector.is_fault_active(config.name) is False  # Should be deactivated

        # Check duration is reasonable (within 10% of expected)
        expected_min = config.duration_s * 0.9
        expected_max = config.duration_s * 1.1
        assert expected_min <= duration <= expected_max, f"Duration {duration} not in range [{expected_min}, {expected_max}]"

        # Check logs contain correlation ID
        log_messages = [record.message for record in caplog.records]
        fault_start_logs = [msg for msg in log_messages if "FAULT_INJECTION_START" in msg and config.name in msg]
        fault_end_logs = [msg for msg in log_messages if "FAULT_INJECTION_END" in msg and config.name in msg]

        assert len(fault_start_logs) == 1, "Should have exactly one fault start log"
        assert len(fault_end_logs) == 1, "Should have exactly one fault end log"

        # Check correlation ID is present
        start_log = fault_start_logs[0]
        end_log = fault_end_logs[0]
        assert "correlation_id=" in start_log
        assert "correlation_id=" in end_log

        # Extract correlation IDs and verify they match
        start_cid = start_log.split("correlation_id=")[1].split()[0]
        end_cid = end_log.split("correlation_id=")[1].split()[0]
        assert start_cid == end_cid, "Correlation IDs should match"

    @pytest.mark.asyncio
    async def test_canary_suite_execution_time(self, canary_cases, fault_injector):
        """Test that the full canary suite executes within time limit."""
        start_time = time.time()

        # Run all cases sequentially
        for case in canary_cases:
            async with await fault_injector.inject_fault(case):
                await asyncio.sleep(0.1)  # Minimal work

        end_time = time.time()
        total_duration = end_time - start_time

        # Suite should complete in under 90 seconds
        assert total_duration < 90.0, f"Suite took {total_duration}s, exceeds 90s limit"

    @pytest.mark.asyncio
    async def test_fault_isolation(self, fault_injector):
        """Test that faults don't interfere with each other."""
        config1 = FaultInjectionConfig(
            name="fault1", fault_type="network_delay", duration_s=1.0, severity=0.5, expected_action="continue"
        )
        config2 = FaultInjectionConfig(
            name="fault2", fault_type="data_gap", duration_s=1.0, severity=0.5, expected_action="continue"
        )

        # Start first fault
        async with await fault_injector.inject_fault(config1):
            assert fault_injector.is_fault_active("fault1")
            assert not fault_injector.is_fault_active("fault2")

            # Start second fault
            async with await fault_injector.inject_fault(config2):
                assert fault_injector.is_fault_active("fault1")
                assert fault_injector.is_fault_active("fault2")

            # Second fault ended
            assert fault_injector.is_fault_active("fault1")
            assert not fault_injector.is_fault_active("fault2")

        # Both faults ended
        assert not fault_injector.is_fault_active("fault1")
        assert not fault_injector.is_fault_active("fault2")

    def test_json_cases_load(self, canary_cases):
        """Test that canary cases can be loaded from JSON."""
        assert len(canary_cases) >= 8, "Should have at least 8 test cases"

        for case in canary_cases:
            assert isinstance(case.name, str)
            assert isinstance(case.fault_type, str)
            assert isinstance(case.duration_s, (int, float))
            assert 0.0 <= case.severity <= 1.0
            assert case.expected_action in ["pause", "resume", "continue"]

    @pytest.mark.asyncio
    async def test_fault_handler_registration(self, fault_injector):
        """Test that fault handlers can be registered and called."""
        call_count = 0

        async def test_handler(config: FaultInjectionConfig):
            nonlocal call_count
            call_count += 1
            assert config.name == "test_fault"

        fault_injector.register_handler("test_type", test_handler)

        config = FaultInjectionConfig(
            name="test_fault", fault_type="test_type", duration_s=0.1, severity=0.5, expected_action="continue"
        )

        async with await fault_injector.inject_fault(config):
            pass

        assert call_count == 1, "Handler should have been called once"