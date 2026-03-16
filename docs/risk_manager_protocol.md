# Risk Manager Protocol Migration Guide

This document explains the `RiskManagerProtocol` and the migration pattern used across the codebase.

Background
---
The repository previously had several risk manager implementations with differing method names and attributes, causing runtime AttributeErrors and confusing debugging when used by adapters, backtest runners, and live trading components.

Objectives
---
- Standardize the minimum RiskManager interface the rest of the codebase expects.
- Keep tests green and avoid runtime failures during migration.
- Provide a compatibility wrapper to allow incremental migration of legacy classes.

Key Components
---
- `ztb/trading/risk/interfaces.py`: defines `RiskManagerProtocol` (runtime-checkable).
- `ztb/trading/risk/backtest_risk_manager.py`: a `BacktestRiskManager` implementation for backtests and diagnostics.
- `ztb/trading/risk/compat.py`: contains `GenericRiskManagerAdapter` and `ensure_risk_manager_protocol` for wrapping legacy implementations.

How to Migrate
---
1. If you are creating a new RiskManager, implement `RiskManagerProtocol` directly by adding the required methods and attributes (see `interfaces.py`).
2. For legacy classes where direct changes are impractical, wrap instances at the instantiation site: `ensure_risk_manager_protocol(RiskManager(...))`.
3. Prefer the wrapper only during an incremental migration. Aim to remove usage of `GenericRiskManagerAdapter` after the entire codebase is updated.

Testing
---
- Unit tests have been added to `tests/unit/` to verify that major components instantiate protocol-compatible risk managers.
- `ensure_risk_manager_protocol` is also covered by tests in `tests/unit/test_risk_compat.py`.

Notes
---
- The compatibility adapter provides conservative fallbacks for missing functionality so tests and backtests continue to run even if the legacy implementation lacks modern APIs.
- Remove the adapter usage after confirming all modules and training scripts implement the protocol directly.

Migration Checklist
---
- [x] Add `RiskManagerProtocol` and use it for the major RiskManager implementations.
- [x] Add `ensure_risk_manager_protocol` and use in key instantiation sites.
- [x] Add tests to verify protocol compliance for adapters and runners.
- [ ] Replace any remaining legacy wrapper usage with direct implementation and remove `GenericRiskManagerAdapter`.
