# Changelog

## 2.0.0 - 2025-09-17

- Breaking: 旧 `src/services/*` / `src/strategies/*` を削除。`@adapters/*` / `@application/*` に移行。
- Core purity: I/O と env 依存は adapters へ移設。
- BaseService: `src/adapters/base-service.ts` へ移動。
- Logging: 必須メタ（requestId, pair, side, amount, price, retries, cause.code）を API/EXEC/ORDER カテゴリに付与。
- Tools: `report-summary` を同期化しテスト安定化。
- Tests: 参照更新とスモールテスト追加。カバレッジ閾値（Statements ≥ 70%）維持。

### Migration
- 型は `@contracts`、実装は `@adapters/*` / `@application/*` を利用してください。
- 例: `import { createServicePositionStore } from '@adapters/position-store'`
- 例: `import { runBuyStrategy } from '@application/strategies/buy-strategy-app'`
