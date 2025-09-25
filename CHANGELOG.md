# Changelog

## 2.4.0 - 2025-09-26

### Changed

- **Directory Structure Reorganization**: Unified Python codebase under `ztb/` directory
  - Moved `scripts/` contents to appropriate `ztb/` subdirectories
  - Created `ztb/experiments/` for experimental scripts and scaling tests
  - Moved feature tests to `ztb/features/`
  - Consolidated notification scripts in `ztb/utils/notify/`
  - Organized tools in `ztb/tools/`
  - Updated README.md with comprehensive Python/ML layer documentation

### Added

- **ztb/experiments/**: New directory for experimental code including:
  - `ml_reinforcement_1k.py`: 1k-step reinforcement learning evaluation
  - Future scaling tests (100k, 1M steps) will be placed here
- **ztb/utils/notify/notify_1k_test_results.py**: Dedicated test result notification script
- **ztb/tools/archive_coverage.py**: Coverage data archival utility

### Improved

- **Code Organization**: Clear separation between TypeScript (src/) and Python (ztb/) codebases
- **Discoverability**: Consistent directory structure improves code navigation
- **CI/CD Preparation**: Better organization for automated testing and deployment pipelines
- **Documentation**: Enhanced README with Python/ML architecture overview

### Technical Details

- **File Movements**:
  - `scripts/test_all_features.py` → `ztb/features/test_all_features.py`
  - `scripts/ml_reinforcement_1k.py` → `ztb/experiments/ml_reinforcement_1k.py`
  - `scripts/notify_1k_test_results.py` → `ztb/utils/notify/notify_1k_test_results.py`
  - `scripts/archive_coverage.py` → `ztb/tools/archive_coverage.py`
- **Directory Structure**: Maintained logical grouping (features, evaluation, trading, experiments, utils, tests, tools)
- **Import Paths**: Updated relative imports to maintain functionality after reorganization


## 2.3.0 - 2025-09-24

### Added

- feature_sets.yaml に Minimal / Balanced / Medium / Large / Extended のセットを定義
- harmful.md を追加し harmful 特徴量の基準と再評価条件を明文化
- experimental_evaluator.py / ablation_runner.py を追加し experimental 特徴量の定期評価をCIに統合

### Changed

- wave4.py を experimental.py にリネーム、役割を「次期候補モジュール」として整理
- features.yaml を再編成、harmful 特徴量は registry から除外
- trend/ volatility/ momentum/ volume/ ディレクトリへ特徴量を分割移動

### Removed

- TypeScript 由来のテスト資産を削除（Python側に完全統一）


## 2.2.3 - 2025-09-23

### Added

- **Operational Improvements for Feature Evaluation System**
  - Externalized evaluation thresholds in `config/evaluation.yaml` for maintainability
  - Implemented Slack/Discord notification system (`scripts/notifier.py`) for automated alerts
  - Added re-evaluation list management (`re_evaluate_list.yaml`) to track feature re-evaluation cycles
  - Updated CI/CD workflow (`.github/workflows/ablation.yml`) with automated notification integration
  - Enhanced `scripts/generate_weekly_report.py` with configurable notifications and re-evaluation tracking

- **Configuration Management**: Centralized evaluation parameters (thresholds, min_samples) in external config file
- **Notification System**: Added support for Slack and Discord webhook notifications with structured summaries
- **CI/CD Integration**: Automated notification delivery in GitHub Actions workflows

### Performance

- **Operational Efficiency**: Streamlined feature evaluation workflow with automated notifications and tracking
- **Maintainability**: Externalized configuration reduces hard-coded values and improves deployment flexibility

## 2.2.2 - 2025-09-20

Observability & Metrics Enhancements.

- EventBus: `slowHandlerCount`, `slowRatio`, configurable `EVENTBUS_SLOW_HANDLER_MS` (WARN: `[EVENT] slow-handler`).
- Metrics Dash: 表示列に slow / slow% 追加、phaseEscalations / phaseDowngrades カウント表示対応。
- Trade Phase Tracking: escalation / downgrade イベントを集計し Slack summary へ反映。
- Slack Summary: coverage, commit SHA, RSS / eventLoop delay p95 (SYS) を追加。`--json` オプション導入。
- System Metrics: `SYS` カテゴリで RSS / heap / handles / event loop p95 を interval 収集 (`SYSTEM_METRICS_INTERVAL_MS`).
- Validation: trade-live 起動時 validate + system metrics 自動開始。
- Tests: slow handler WARN / slowRatio / phase counts / system metrics / slack summary JSON 追加。

## 2.2.1 - 2025-09-19

Patch: 安定化と不要コード整理のみ（後方互換）。

- Cleanup: remove deprecated adapters `adapters/risk-service` and `adapters/position-store` (migrated tests to core/fs implementations).
- Stabilization (price-cache): corrupted JSON recovery emits guaranteed single synchronous `CACHE_ERROR` (Windows race hardening).
- Stabilization (indicator-service): missing-volume WARN test now captures structured console args (logger suppression safe).
- Stabilization (ml-simulate): Windows file visibility race mitigated via direct candidate retry + test sleep (timeout消滅)。
- CI/Observability: EVENT/METRICS 出力と metrics-dash の追加強化（平均/latency p95, handler counts）。
- Dev: EventBus subscribe/publish の型推論改善。

## 2.2.0 - 2025-09-19

- README: 全面見直し（Quick Start 最上段、EventBus/publishAndWait、TEST_MODE/Vitest の安定化ポイント、Rate Limiter 章の統合・最新化、live:minimal の使い方整理、旧 services 記述の削除）。
- EventBus: テスト時に `publishAndWait()` で `TRADE_PLAN`/`TRADE_PHASE` を同期発行（レース回避）。
- Trade Live: TEST_MODE の昇格閾値デフォルトを緩和（1→2 を 1 日で許容）。
- Toolkit: `sleep` を Vitest/TEST_MODE 下で自動的に短縮。
- Rate Limiter: テストスイートがカスタム limiter を注入した場合に強制有効化（メトリクス系テストの安定化）。
- Docs: スクリプト一覧/主要環境変数/注意事項/CI/Coverage の説明を現状に整合。

## 2.1.0 - 2025-09-18

- Errors: エラーコードを統一し、`EVENT/ERROR` を全レイヤで発火
	- BaseService: `CIRCUIT_OPEN`/`RATE_LIMITED`/最終失敗で `EVENT/ERROR` を publish
	- price-cache: 讀み書き失敗で `CACHE_ERROR` を publish
	- zaif-private: `NONCE`/`SIGNATURE`/`API_ERROR`/`NETWORK` を publish（必須メタ付き）
- CI/Artifacts: マージ後の別名 `coverage-merged/coverage-merged.json` を出力
- ts-prune: 結果を日付付き `ci/reports/ts-prune-YYYYMMDD.json` として永続化
- Cleanups (Batch 4): モジュール内専用の型 export を非公開化
	- adapters/indicator-service: `IndicatorSnapshot`, `IndicatorServiceOptions` を非公開化
	- adapters/execution-service: `OrderBookLevel`, `OrderSnapshot`, `SubmitParams`, `SubmitRetryParams` を非公開化
	- adapters/market-service: `MarketOverview` を非公開化
- Tests: `unit`/`integration-fast`/`cb-rate`/`event-metrics` 全てグリーン
- Deps: axios / vite / vitest / coverage-v8 を最新安定版へ更新（アドバイザリ 0）

## 2.0.0 - 2025-09-17

- Breaking: 旧 `src/services/*` / `src/strategies/*` を削除（実体を完全除去）。`@adapters/*` / `@application/*` に移行。
- Core purity: I/O と env 依存は adapters へ移設。
- BaseService: `src/adapters/base-service.ts` へ移動。
- Logging: 必須メタ（requestId, pair, side, amount, price, retries, cause.code）を API/EXEC/ORDER カテゴリに付与。
- Tools: `report-summary` を同期化しテスト安定化。
- Tests: 参照更新とスモールテスト追加。カバレッジ閾値（Statements ≥ 70%）維持。
	- テストユーティリティを `__tests__/helpers/*` に集約。`src/tools/tests/*` は削除。
	- 旧 `src/strategies/*` のエイリアスシムを削除（アプリ層の `@application/strategies/*` を利用）。

### Migration

- 型は `@contracts`、実装は `@adapters/*` / `@application/*` を利用してください。
- 例: `import { createServicePositionStore } from '@adapters/position-store'`
- 例: `import { runBuyStrategy } from '@application/strategies/buy-strategy-app'`
