# Module Ownership and Migration Plan

This document tracks canonical module locations and planned renames for the Zaif Trade Bot codebase.

## Canonical Modules

### Circuit Breaker

- **Canonical**: `ztb/utils/circuit_breaker.py`
- **Shim**: `ztb/risk/circuit_breakers_compat.py`
- **Status**: New implementation (v3.1)
- **Migration**: Remove shim after all imports updated

### Order State Machine

- **Canonical**: `ztb/trading/order_state_machine.py`
- **Shim**: `ztb/live/order_state_compat.py`
- **Status**: New implementation (v3.1)
- **Migration**: Remove shim after all imports updated

### Results Validator

- **Canonical**: `schema/results_validator.py`
- **Status**: New implementation (v3.1)
- **Migration**: No shim needed (new module)

### Kill Switch

- **Canonical**: `ztb/utils/kill_switch.py`
- **Status**: New implementation (v3.1)
- **Migration**: No shim needed (new module)

### Reconciliation

- **Canonical**: `ztb/trading/reconciliation.py`
- **Status**: New implementation (v3.1)
- **Migration**: No shim needed (new module)

## Migration Timeline

1. **Phase 1 (v3.1)**: Add shims for backward compatibility
2. **Phase 2 (v3.2)**: Update all imports to canonical modules
3. **Phase 3 (v3.3)**: Remove compatibility shims

## Import Guidelines

- Always import from canonical modules for new code
- Use shims only for existing code that cannot be immediately updated
- Add TODO comments when using shims

## Contact

For questions about module ownership, contact the architecture team.