# Changelog

## Unreleased

- CI: テストを5分割（unit/integration/cb-rate/event-metrics/long）し、並列実行対応（`.github/workflows/test-matrix.yml`）。
- EventBus: 型推論強化（subscribe/publishの型絞り込み）。
- EVENT/METRICS: 平均/ p95 / ハンドラ別の件数・失敗を定期出力（`EVENT_METRICS_INTERVAL_MS`）。
- metrics-dash: EVENT/METRICS表示を追加（スパークライン、タイプ別集計）。

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
