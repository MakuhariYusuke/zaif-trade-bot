Test structure and conventions

This repository uses layered testing to keep quick feedback fast and still cover integration scenarios.

Layers and conventions
- tests/unit: Fast unit tests. Should be deterministic and not require heavy deps, external services, or long runtimes. These tests are auto-marked as "unit" by `tests/conftest.py`.
- tests/integration: Integration tests that may require heavier libraries, temporary files, or external services. Marked as "integration".
- tests/e2e: Very slow end-to-end tests. Marked as "slow".
- tests/tools: Tests for analysis and tools scripts; prefer running these as unit tests with minimal environment-side effects.

Running tests
- Run all unit tests: pytest -m unit
- Run all integration tests: pytest -m integration
- Run fast tests only: pytest -m "not slow and not integration"

Best practices
- Prefer using `RewardUtils` and other central utilities for shared logic; avoid copying logic into tests.
- Tests should keep side effects minimal and avoid importing heavy modules at top-level; use content-based checks when appropriate.
- Add a test for every refactor to prevent regressions.

CI
- CI should run unit tests by default and integration/e2e on scheduled pipelines or PRs that opt-in.
