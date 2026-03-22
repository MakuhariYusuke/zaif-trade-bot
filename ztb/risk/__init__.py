"""Risk management package with lazy export loading."""

from __future__ import annotations

from importlib import import_module

_LAZY_MODULE_ATTRS: dict[str, tuple[str, str]] = {
    "get_risk_profile": ("ztb.risk.profiles", "get_risk_profile"),
    "create_custom_risk_profile": ("ztb.risk.profiles", "create_custom_risk_profile"),
    "RiskRuleEngine": ("ztb.risk.rules", "RiskRuleEngine"),
    "RiskChecker": ("ztb.risk.checks", "RiskChecker"),
    "RiskManager": ("ztb.risk.checks", "RiskManager"),
    "PnLMonteCarloSimulator": ("ztb.risk.pnl_monte_carlo", "PnLMonteCarloSimulator"),
    "MonteCarloConfig": ("ztb.risk.pnl_monte_carlo", "MonteCarloConfig"),
    "MonteCarloResult": ("ztb.risk.pnl_monte_carlo", "MonteCarloResult"),
    "ToxicityAssessment": ("ztb.risk.toxicity_types", "ToxicityAssessment"),
    "ToxicityLevel": ("ztb.risk.toxicity_types", "ToxicityLevel"),
}

__all__ = [
    "get_risk_profile",
    "create_custom_risk_profile",
    "RiskRuleEngine",
    "RiskChecker",
    "RiskManager",
    "PnLMonteCarloSimulator",
    "MonteCarloConfig",
    "MonteCarloResult",
    "ToxicityAssessment",
    "ToxicityLevel",
]

def __getattr__(name: str) -> object:
    target = _LAZY_MODULE_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value

def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
