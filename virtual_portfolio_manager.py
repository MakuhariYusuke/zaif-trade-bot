"""Tiny stub for VirtualPortfolioManager required by legacy integration tests.

Provides minimal API surface used during import-time for collection.
"""
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class VirtualPortfolioManager:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def create_portfolio(self, name: str):
        return {"name": name, "id": f"pf_{name}"}
