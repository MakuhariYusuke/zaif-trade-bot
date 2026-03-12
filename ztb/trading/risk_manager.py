#!/usr/bin/env python3
"""Minimal stub for RiskManager so tests that patch ztb.trading.risk_manager succeed.

This module intentionally provides a minimal RiskManager surface used by tests
to patch out the implementation. It is not intended to be a production-level
implementation but provides a safe default to keep imports working.
"""
from typing import Any

class RiskManager:
    def __init__(self, *args, **kwargs):
        self.logger = None

    def evaluate_risk(self, *args, **kwargs) -> dict[str, Any]:
        return {"risk_level": "low"}

    def get_status(self) -> dict[str, Any]:
        return {"status": "active"}
