This directory contains unit and integration tests for trading strategies.

Conventions:

- Each strategy has its own subdirectory (e.g., action_signal_guide) when there are multiple test files.
- Unit tests should be named test_<strategy>_unit.py and contain fast-running, isolated tests.
- Integration tests should be named test_<strategy>_integration.py and exercise data pipelines or pattern recognizers.
- Legacy or backup tests are archived under test_<strategy>_*_legacy.py to avoid duplication but keep for reference.

Migration done for action_signal_guide:
- Consolidated unit tests into strategies/action_signal_guide/test_action_signal_guide_unit.py
- Consolidated heavy/integration tests into strategies/action_signal_guide/test_action_signal_guide_integration.py
- Replaced old files with shims or legacy placeholders to avoid duplication while keeping history.

If more duplicates are found, apply the same pattern: move canonical tests into the strategy subfolder and replace older files with shims that import the canonical tests, or archive them into legacy files.
