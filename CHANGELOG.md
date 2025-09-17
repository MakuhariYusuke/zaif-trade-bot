# Changelog

## Unreleased

- CI: テストを5分割（unit/integration/cb-rate/event-metrics/long）し、並列実行対応（`.github/workflows/test-matrix.yml`）。
- EventBus: 型推論強化（subscribe/publishの型絞り込み）。
- EVENT/METRICS: 平均/ p95 / ハンドラ別の件数・失敗を定期出力（`EVENT_METRICS_INTERVAL_MS`）。
- metrics-dash: EVENT/METRICS表示を追加（スパークライン、タイプ別集計）。

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
