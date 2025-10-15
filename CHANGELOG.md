# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added `_archive_price_history` method to `LiveTrader` class for memory management by archiving price history to disk.
- Added PositionManager integration in LiveTrader for better position and PnL management.
- Added advanced auto-stop system initialization in LiveTrader.
- Added dry-run functionality verification with SAC model `sac_v420_hold_relaxed.zip`.

### Changed
- Refactored `data_generation.py` into a `DataGenerator` class with improved caching, error handling, and performance optimizations.
- Enhanced `talib_wrapper.py` with instance-based caching, better validation, and configurable strictness.
- Refactored `live_trader.py` initialization into smaller, more maintainable methods with better error handling.
- Improved code structure in `data_generation.py` with better error handling and performance optimizations.
- Improved code structure in `talib_wrapper.py` with enhanced wrapper functions and validation.
- Improved code structure in `live_trader.py` with additional methods and integrations.
- Improved code structure in `checkpoint.py` with better organization and error handling.
- Fixed import path issue in `main.py` for proper module loading.
- Enhanced `live_trader.py` with comprehensive error handling in initialization and async/sync price fetching methods.
- Added `_get_current_price_sync()` method for synchronous price access with fallback handling.
- Improved robustness of LiveTrader initialization with graceful handling of adapter and notifier failures.
- Added comprehensive unit tests for LiveTrader initialization and error scenarios.

### Fixed
- Fixed syntax errors in `live_trader.py` including unterminated docstrings, null bytes, and BOM issues.
- Resolved parentheses imbalance in `live_trader.py`.
- Fixed module import issues in live trading main script.

### Removed
- Removed problematic docstring sections causing syntax errors.