# Zaif Trade Bot

軽量・検証重視の Zaif/Coincheck 向け自動売買ボット。SELL ファースト戦略、モック駆動テスト、堅牢なリトライ/CB/Rate 制御を備え、拡張しやすい土台を提供します。

---

## すぐ使う（Quick Start）

1) 依存インストール
```
npm install
```

2) 最速検証（モック + ドライラン）
```
$env:USE_PRIVATE_MOCK="1"; $env:DRY_RUN="1"; npm start
```

3) 最小ライブ検証（自己責任・即キャンセル）
```
npm run live:minimal
```
ヒント: PowerShell では環境変数の一時設定に `$env:NAME="value"` を使用します。

安全上の注意: 本リポジトリは学習・検証目的の参考実装です。実運用はご自身の責任で、極小サイズから十分に検証してください。

---

## アーキテクチャ概要

- `core/`: 純粋ロジック（I/O・fs・env に非依存）
- `adapters/`: I/O・外部 API 委譲（FS/HTTP/署名/レート制御等）
- `application/`: 戦略やユースケースのオーケストレーション（イベント購読/集計）
- `app/`: メインの定期実行ループ（`npm start`）
- `tools/`: live/paper/ml/stats の各 CLI

移行状況メモ
- 旧 `src/services/*` は削除済みです。新規/既存コードとも `@adapters/*` を使用してください。
- 型は `src/contracts` に集約し `@contracts` で import できます。

---

## アプリケーションイベント（EventBus）

軽量なインメモリ EventBus で実行と副作用（ポジション/統計/ロギング）を分離します。外部公開はしません。

- バス: `src/application/events/bus.ts`（既定は非同期 dispatch）
- 型: `src/application/events/types.ts`
- サブスクライバ: `src/application/events/subscribers/*`
- 登録: `registerAllSubscribers()` を `src/app/index.ts` で呼び出し

主なイベント（抜粋）
- `ORDER_SUBMITTED` / `ORDER_FILLED` / `ORDER_CANCELED` / `ORDER_EXPIRED`
- `SLIPPAGE_REPRICED`（スリッページによりリプライス）
- `TRADE_PLAN` / `TRADE_PHASE`（trade-live ツールの計画・段階進行）

共通メタデータ: `requestId`, `eventId`, `pair`, `side`, `amount`, `price`, `orderId?`, `retries?`, `cause?`

購読例（型安全）
```ts
import { getEventBus } from '@application/events';
getEventBus().subscribe('ORDER_FILLED', (e) => {
  console.log('filled', e.orderId, e.filled, e.avgPrice);
});
```

Idempotency: `eventId`/`requestId` をキーに重複に耐性を持つ実装にしてください。

テスト時の同期発行: 競合を避けるため、Vitest/TEST_MODE では `publishAndWait()` を用いて `TRADE_PLAN`/`TRADE_PHASE` を同期発行しています。

---

## Circuit Breaker（内蔵）

- 状態: `CLOSED`/`OPEN`/`HALF_OPEN`。判定は失敗率・連続失敗・レイテンシ中央値。
- 適用: `API-PUBLIC`（観測のみ）, `API-PRIVATE`/`EXEC`（ゲート対象）。
- ログ: 遷移は `CB/INFO`、ブロックは `CB/ERROR blocked`（`CIRCUIT_OPEN`）。
- 既定（環境で上書き可）: `CB_WINDOW_SIZE=50`, `CB_FAILURE_THRESHOLD=0.5`, `CB_MAX_CONSEC_FAIL=5`, `CB_LATENCY_THRESHOLD_MS=30000`, `CB_HALF_OPEN_TRIAL=5`, `CB_COOLDOWN_MS=60000`。

withRetry 使用例
```ts
import { withRetry } from '@adapters/base-service';
await withRetry(() => publicApi.fetchDepth('btc_jpy'), 'fetchDepth', 2, 100, { category: 'API-PUBLIC' });
await withRetry(() => privateApi.getBalance(), 'getBalance', 3, 150, { category: 'API-PRIVATE' });
await withRetry(() => exec.cancelOrder(orderId), 'cancelOrder', 3, 150, { category: 'EXEC' });
```

---

## Rate Limiter（カテゴリ別・予約枠対応）

- 概要: トークンバケツでスループットを平準化（プロセス内シングルトン）。適用順は CB → Rate → Execute。
- 既定: `capacity=100`, `refill=10 tokens/sec`, 予約枠 10%（`opType:'ORDER'` のみ使用）。
- 動作: 最大 1 秒取得待機。未取得で `RATE_LIMITED` を投げます。
- ログ: `RATE/INFO acquired`, `RATE/WARN waited`, `RATE/ERROR rejected`, `RATE/METRICS metrics`（直近 N=50 の平均待機/拒否率/カテゴリ詳細）。

環境変数（最新）
- `RATE_LIMITER_ENABLED` (1|0, 既定 1)
- `RATE_CAPACITY` / `RATE_REFILL`（共通）
- `RATE_RESERVE_RATIO`（0..1, 既定 0.1）
- `RATE_METRICS_INTERVAL_MS`（既定 60000, 0 で無効）
- カテゴリ別: `RATE_CAPACITY_PUBLIC|PRIVATE|EXEC`, `RATE_REFILL_PUBLIC|PRIVATE|EXEC`
  - 後方互換: `RATE_REFILL_PER_SEC`, `RATE_PRIORITY_RESERVE` も解釈されます。

withRetry の指定例
```ts
await withRetry(() => publicApi.getTicker('btc_jpy'), 'getTicker', 2, 100, { category: 'API-PUBLIC', priority: 'low', opType: 'QUERY' });
await withRetry(() => privateApi.getBalance(), 'getBalance', 3, 150, { category: 'API-PRIVATE', priority: 'normal' });
await withRetry(() => exec.placeOrder(req), 'placeOrder', 3, 150, { category: 'EXEC', priority: 'high', opType: 'ORDER' });
await withRetry(() => exec.cancelOrder(id), 'cancelOrder', 3, 150, { category: 'EXEC', priority: 'high', opType: 'CANCEL' });
```

テスト時の扱い（重要）
- 既定でテストはレート制御を無効化しますが、スイート側でカスタム limiter を注入した場合はグローバルフラグで強制有効になります（メトリクス系テストの安定性向上）。

---

## キャッシュ・メトリクス（任意）

- 市場系キャッシュのヒット/ミス/ステールをカウントし、`CACHE/METRICS` を定期出力。
- 主に `market:ticker`/`market:orderbook`/`market:trades` を可視化。
- 環境: `CACHE_METRICS_INTERVAL_MS`, `MARKET_CACHE_TTL_MS`。

---

## ダッシュボード（軽量 CLI）

直近の RATE/CACHE/EVENT メトリクスを要約表示します（`LOG_JSON=1` 前提）。
```
npm run dash
npm run dash -- --watch         # 2 秒ごと
npm run dash -- --watch 5000    # 5 秒ごと
```
PowerShell 例
```
npm run dash -- --file .\logs\trades-2025-09-17.log --lines 8000
```

使い方（補足）
- フラグ: `--file <path>`（既定は最新ログ自動検出/`METRICS_LOG`）、`--lines N`（末尾N行）、`--watch [ms]`（継続監視）、`--json`（1回分のJSON出力）
- インタラクティブ操作: 上下矢印で RATE/CACHE/EVENT を切替、左右でスパークライン幅、`q` で終了
- 表示: RATE は p95/拒否率とカテゴリ別、CACHE はヒット率/ステール率、EVENT はタイプ別 pub/calls/errors/avg/p95 と最頻ハンドラ
- TRADE_PHASE: `TRADE_STATE_FILE`（既定 `trade-state.json`）を読み取り、現在フェーズと累計成功を併記
- 精度: `LOG_JSON=1` を推奨（プレーンテキストは簡易パーサのフォールバック）

JSON 出力例（`--json`）
```json
{
	"rate": { "avgWaitMs": 120, "rejectRate": 0.03, "details": { "PUBLIC": {"acquired": 50} } },
	"cache": { "ticker": { "hits": 90, "misses": 10, "stale": 2, "hitRate": 0.9 } },
	"events": { "windowMs": 60000, "types": { "ORDER_FILLED": { "publishes": 3 } } },
	"tradePhase": { "phase": 2, "totalSuccess": 21 }
}
```

---

## 観測性（Observability）まとめ

- ログ形式: 既定はプレーン。`LOG_JSON=1` で JSONL（`ts`,`level`,`category`,`message`,`data[]`）。カテゴリーは `RATE`/`CACHE`/`EVENT` などを使用
- メトリクス:
	- Rate Limiter: `RATE/METRICS` を定期出力（平均待機/拒否率/カテゴリ別詳細）。間隔は `RATE_METRICS_INTERVAL_MS`
	- Cache: `CACHE/METRICS`（ヒット/ミス/ステール）。`CACHE_METRICS_INTERVAL_MS`
	- Event: `EVENT/METRICS`（タイプ別 pub/calls/errors/latency）。ダッシュが拾います
- レッドアクション: 機密キー/シークレット/Authorization/生ペイロード等は自動マスク（`src/utils/logger.ts` の `addLoggerRedactFields` で拡張可）
- 推奨レベル: 運用は `INFO`、調査/CI は `DEBUG`。テスト時は低レベルログを抑制しつつ、重要イベント（ERROR/WARN/メトリクス）は出力
- 付随メタデータ: `requestId`/`eventId`/`pair`/`side` などをイベントに付与（冪等/相関用）

---

## Trade Live（段階制御 / Phase-driven）

- 設定: `trade-config.json`（`TRADE_CONFIG_FILE` で切替）
- 状態: `trade-state.json`（`TRADE_STATE_FILE` で切替）
- コマンド: `npm run trade:live` / `npm run trade:live:dry`
- 昇格ルールは環境で調整: `PROMO_TO2_DAYS`, `PROMO_TO3_SUCCESS`, `PROMO_TO4_SUCCESS`
- テストでは `publishAndWait()` により `TRADE_PLAN`/`TRADE_PHASE` を同期発行し、昇格閾値も TEST_MODE デフォルトで緩和（1 日で 1→2 を許可）

### Trading Phases 詳細

- 設定ファイル: `trade-config.json`（`TRADE_CONFIG_FILE` で差し替え）
	- 例:
		```json
		{ "pair":"btc_jpy", "phase":1, "phaseSteps":[
			{"phase":1,"ordersPerDay":1}, {"phase":2,"ordersPerDay":3}, {"phase":3,"ordersPerDay":5}, {"phase":4,"ordersPerDay":10}
		]}
		```
- 状態ファイル: `trade-state.json`（`TRADE_STATE_FILE`）。`{ phase, consecutiveDays, totalSuccess, lastDate }`
- 昇格ルール（環境で上書き）: `PROMO_TO2_DAYS`（既定 5・TEST_MODE は 1）/`PROMO_TO3_SUCCESS`（既定 20）/`PROMO_TO4_SUCCESS`（既定 50）
- CLI:
	- ドライラン: `npm run -s trade:live:dry > trade-plan.json`（標準出力に計画 JSON: `pair/phase/plannedOrders/today`）
	- ライブ: `npm run trade:live`（当日の成功数に応じて `trade-state.json` を更新し、必要時 `EVENT/TRADE_PHASE` を発行）
- テスト安定化: Vitest/TEST_MODE では `publishAndWait()` で `TRADE_PLAN/TRADE_PHASE` を同期発行し、日次成功の既定を 1 に短縮

---

## 日常の使い方

戦略モード切替（`TRADE_MODE=SELL|BUY`）
- SELL/BUY で戦略を切替。ログに `[SIGNAL][mode=SELL|BUY]` が出ます。

オーダーフロー（`TRADE_FLOW`）
- `BUY_ONLY|SELL_ONLY|BUY_SELL|SELL_BUY`（既定 BUY_SELL）。
- `TEST_FLOW_QTY>0` で 1 サイクル内にフロー実行。DRY_RUN=1 なら即時 fill をシミュレート。

Coincheck 切替
```
$env:EXCHANGE="coincheck"; npm start
```
注意: 最小単位や丸めが異なります。小サイズで開始してください。

---

## スクリプト一覧（主要）

package.json に定義された主なコマンド:

- 実行
  - `npm start`（メインループ）
  - `npm run live:minimal`（最小ライブ検証・即キャンセル）
  - `npm run health`（署名/nonce/permission ヘルス）
- テスト
  - `npm run test:unit` / `:int-fast` / `:cb-rate` / `:event-metrics` / `:long`
  - `npm run test`（カバレッジ込みフル）
- ツール
  - `npm run dash`（メトリクスダッシュ）
  - `npm run stats:today` / `stats:graph`
  - ML: `ml:export` / `ml:search` / `feature:importance`

---

## TEST_MODE/Vitest の振る舞い（安定化の工夫）

- EventBus: `TRADE_PLAN`/`TRADE_PHASE` を `publishAndWait()` で同期発行。
- Trade Live: テストでは daySuccess 既定を 1 に短縮し、1→2 昇格を観測しやすくしています。
- Sleep: `utils/toolkit.sleep` は Vitest/TEST_MODE で自動的に短縮（デフォルト ~1ms）。
- Live minimal: テスト時は重いサブプロセス/サマリー書き込みをスキップし、タイムアウト回避。
- Rate Limiter: 既定無効。ただしテストでカスタム limiter をセットした場合は強制有効化しメトリクスを出力。
- 一時ディレクトリ: テストは一時領域を用い、実データと混ざらないよう配慮（`STATS_DIR`, `POSITION_STORE_DIR` など）。

---

## Coverage / CI

- GitHub Actions で `unit / integration / cb-rate / event-metrics / long` に分割実行。
- Coverage レポートは Pages に公開可能。ワークフロー: `.github/workflows/coverage-pages.yml`。
- 既定しきい値: Statements >= 70%。

### テストマトリクス（分割）
- `unit`: 純粋ロジック/ユーティリティ
- `integration-fast`: 主要フローの高速結合（EventBus 同期発行でレース回避）
- `cb-rate`: Circuit Breaker/Rate Limiter の待機・拒否・メトリクス
- `event-metrics`: EVENT/METRICS の集計とハンドラ健全性
- `long`（任意）: 長時間スモーク

### Nightly / Paper
- `paper-nightly`: 1 日分をドライランし、`reports/day-YYYY-MM-DD/` に成果をコミット
- 主要成果: `stats-timeline.svg`, `stats-diff.json`, `report-summary-paper.json`, `ml-search-top.json`, `trade-plan.json`
- `trade-plan.json`: `npm run -s trade:live:dry` の出力 JSON。レポート index にリンク
- `ts-prune.yml`: 未使用エクスポート件数の推移を `ci/reports/ts-prune-*.json` に保存（count/top20 サマリ付き）

### CI チェックリスト
- テスト green: unit / integration-fast / cb-rate / event-metrics
- Coverage >= 70%（Pages 有効なら `/coverage/` で HTML 確認）
- `trade:live:dry` が成功し `trade-plan.json` が生成（レポート index に含まれる）
- `npm run dash -- --json` が rate/cache/event/tradePhase を返す
- ts-prune 件数が直近から悪化していない（WoW）

---

## ストレージ破損時の挙動（Recovery Semantics）

対象: price-cache (`price_cache.json`), position-store (`positions.json` など fs 保存物)。

- 原則: 読み込み時に JSON パース失敗すると「破損」と見なし、イベント/ログを 1 回だけ出します。
	- price-cache: `CACHE_ERROR`（2.2.1 で同期単発保証）。
	- position-store: 破損は即座に空状態として再初期化（次の write が健全ファイルを再生成）。
- リカバリ: 過去履歴の再構築は行いません（シンプルさと一貫性優先）。
- 推奨運用: 日次/週次で `logs/` or `artifacts/` を外部へローテーションし、必要なら別途履歴保管層を実装。
- チューニング: 高頻度書き込み競合は atomic write（`fs-atomic`）と一時ファイル rename で抑制しています。

FAQ:
- Q: 破損イベント後はトレード止めるべき？ → 重要度によります。price-cache は再生成されますが、履歴指標計算が乖離するリスクがあるため再計測/監視を推奨。

---

## Live 起動前セーフティチェックリスト

次の最低限を満たしてから `npm run trade:live` もしくは `live:minimal` を実行してください。

1. DRY RUN から開始: `$env:DRY_RUN="1"` で期待ログ/イベントを確認済み
2. API キー: 取引所（Zaif/Coincheck）側で権限/残高/手数料を確認（最初は極小数量）
3. `TRADE_CONFIG_FILE`: phaseSteps が意図通り（例: `1,3,10,25`）か再チェック
4. SAFETY_MODE 有効: `$env:SAFETY_MODE="1"` と clamp ログ動作確認
5. Rate/Circuit 設定: 既定の `CB_*` / `RATE_*` を本番レイテンシ・API 制限に合わせ調整済み
6. ログ出力形式: `LOG_JSON=1` でメトリクス/イベントを可視化できる状態
7. タイムゾーン/時計: OS 時刻同期が取れている（約定タイムスタンプ依存指標を守る）
8. バックアップ: `trade-state.json` / ポジションファイルの初期バックアップを取得
9. テスト済み: `npm run test:unit` / `:int-fast` / `:cb-rate` / `:event-metrics` が local green
10. 失敗ハンドリング: `EVENT/ERROR` ログを監視する簡易アラート（tail or dashboard）用意

追加推奨:
- 小さな `TEST_FLOW_QTY` で 1 サイクル流し fill→EXIT ライフサイクルを live:minimal で先に確認
- 監視: 5〜10 分間は手動監視し、意図しない連続注文が無いか確認

---

---

## Live Minimal（DRY_RUN 統合）

最小の発注→即キャンセル検証を `live:minimal` に統合。`DRY_RUN=1` ならモック/ドライラン、`DRY_RUN=0` なら実発注（要 API キー）。

環境変数
- `EXCHANGE` / `PAIR` / `TRADE_FLOW`
- `TEST_FLOW_QTY`（DRY_RUN=1 既定 0.002）
- `TEST_FLOW_RATE` / `ORDER_TYPE=market|limit`
- `DRY_RUN=0|1`, `SAFETY_MODE=1`, `SAFETY_CLAMP_PCT`, `EXPOSURE_WARN_PCT`

PowerShell 例（実発注・即キャンセル）
```powershell
$env:EXCHANGE="coincheck"; $env:TRADE_FLOW="BUY_ONLY"; $env:TEST_FLOW_QTY="1"; $env:TEST_FLOW_RATE="490"; $env:DRY_RUN="0"; $env:SAFETY_MODE="1"; npm run live:minimal
```

---

## テクニカル指標と特徴量ログ

`features-logger` が RSI/SMA/MACD/ATR/BB 幅など多様な指標を自動計算し JSONL で記録します。観測開始時に 1 回だけサンプル WARN を出力します。

---

## 主要な環境変数（抜粋）

| 変数 | 用途 |
|------|------|
| `EXCHANGE` | 取引所切替（`zaif`/`coincheck`） |
| `DRY_RUN` | 実注文せずシミュレーション |
| `PAIR` | 取引ペア（例: `btc_jpy`） |
| `TRADE_MODE` | `SELL`/`BUY` |
| `TRADE_FLOW` / `TEST_FLOW_QTY` | フロー/数量（ワンショット検証） |
| `ZAIF_API_KEY`/`ZAIF_API_SECRET` | Zaif 認証 |
| `COINCHECK_API_KEY`/`COINCHECK_API_SECRET` | Coincheck 認証 |
| `SAFETY_MODE`/`SAFETY_CLAMP_PCT`/`EXPOSURE_WARN_PCT` | 安全クランプ/露出警告 |
| `RATE_*` / `CB_*` / `CACHE_*` | レート/CB/キャッシュ計測 |

より詳細な一覧はコードの各モジュール（`utils/config` ほか）を参照してください。

---

## 注意事項（重要）

本リポジトリは学習・検証目的です。実運用前に以下を検討してください。
- API レート制限/指数バックオフ、WebSocket 約定照合、秘密情報管理、冗長化/復旧/アラート、時刻同期（NTP）/nonce 管理。

---

## 変更履歴 / ライセンス

- 変更履歴は `CHANGELOG.md` を参照してください。
- ライセンスは検討中です。現時点では個人学習向けの利用を想定しています（将来 OSS ライセンス付与予定）。


| 種別 | 規則 | 例 |
|------|------|----|
| 型 / Interface | PascalCase | `PrivateApi`, `OrderLifecycleSummary` |
| 関数 / 変数 | camelCase | `fetchMarketOverview`, `appendPriceSamples` |
| 定数 / 環境変数 | UPPER_SNAKE_CASE | `RISK_MIN_TRADE_SIZE` |
| ファイル | kebab-case.ts | `position-store.ts` |

### 命名規則（詳細）

注: 本規約は厳密適用ではなく「原則＋例外を許容する運用」です。既存コードや外部API仕様との整合、スピード/安全性の観点で、合理的な範囲で弾力運用してください。PR 時に逸脱理由がわかるようコメントや説明を添えるとスムーズです。

- ディレクトリ/ファイル
	- ディレクトリは kebab-case、ソースは kebab-case.ts を基本とする。
	- ツール類は `src/tools/`、公開APIは `src/api/*-public.ts` か `public-router.ts` に集約。
	- 取引所ごとの差分は `src/api/EXCHANGE-*.ts` に分離し、app/core からは抽象化された関数を呼ぶ。

- 型・命名の接尾辞
	- リクエスト/レスポンス/結果型は `FooRequest` / `FooResponse` / `FooResult` / `FooParams` / `FooOptions` を使用。
	- ドメイン横断の集計は `*Summary`（例: `OrderLifecycleSummary`）。
	- 列挙は TypeScript の文字列リテラル Union を優先（`"bid"|"ask"` / `"BUY"|"SELL"`）。

- 関数の語彙（prefix）
	- 取得系: `get*`（キャッシュ/同期）, `fetch*`（外部IO/HTTP）, 読み込み: `load*`, 保存: `save*`, 
		変更: `update*`, 追加: `append*`, 削除: `remove*`。
	- 計算系: `calculate*`（SMA/RSI 等）, 記述/整形: `describe*`, 管理: `manage*`（例: `manageTrailingStop`）。
	- 非同期関数名に Async 接尾辞は付けない（Promise/await で判別）。

- 変数/定数/真偽値
	- ランタイム不変の構成値は `UPPER_SNAKE_CASE`（環境変数名も同様）。
	- 真偽値は `is*/has*/should*/can*` を推奨（`isArmed`, `hasCreds`, `shouldExit`, `canPlaceOrder`）。

- 時刻・単位・ペア表記
	- 内部 `ts` はミリ秒（ms）。外部APIで秒を受け取る場合は即 ms に正規化して保存。
	- 金額は JPY、数量は BTC を基本。`amount` はベース資産数量、`price` は見積通貨（JPY）。
	- ペアは `btc_jpy` の snake_case・小文字に統一。

- サイド/方向の表現
	- 取引API層（Zaif/Coincheck私設）では `"bid"|"ask"`、上位の発注ヘルパでは `"BUY"|"SELL"` を使用。
	- 境界の変換は `core/market.ts`（`placeLimitOrder`）で行う。

- ID と数値の扱い
	- 外部API由来の `order_id` は一貫して文字列として扱う（比較/マップ用途のため）。
	- 金額/数量は Number、必要に応じ 8 桁程度で丸め/切り捨て（BTC）。

- ログ/イベント
	- トレードログは `logSignal/logOrder/logExecution/logTradeError/logTradeInfo` を利用。
	- 日次集計に反映するカウンタは `utils/daily-stats.ts` のインクリメント関数を用いる。

- エラー・結果
	- 失敗結果は `err(code, message, cause?)`、成功は `ok(value)` を推奨（`utils/result.ts`）。
	- 例外メッセージは外部レスポンスの要点を含めつつ機密はログに出さない。

- テスト/ツール命名
	- 最小実行ツールは `src/tools/*`、テストユーティリティは `__tests__/helpers/*`。
	- フロー検証は `test-<exchange>-flow.ts` の形式を推奨。

### 既存リネームの指針（対比）

| 旧 | 新 | 備考 |
|----|----|------|
| `trailManager` | `manageTrailingStop` | リスクサービス/コアで統一 |
| `calcSMA` / `calcRSI` | `calculateSma` / `calculateRsi` | 関数は動詞始まり |
| `submitWithRetry` | `submitOrderWithRetry` | 役割が明確になるように |
| `appendPrices` | `appendPriceSamples` | 意味の明確化 |
| `getRecentPrices` | `getPriceSeries` | シリーズ取得の意味合い |
| `writeTradeLog` | `logTrade` | 既存ログAPIに寄せる |
| `ok`(GuardResult) | `isAllowed` | 真偽を明示 |
| `fillCount` | `filledCount` | 語形統一 |

チェックリスト（PR時の確認）
- [ ] 取引所差異は `api/public-router.ts` / PrivateApi 実装で吸収し、app/core は抽象 API のみ参照
- [ ] `btc_jpy`・JPY/BTC の単位整合、ms/秒の正規化の一貫性
- [ ] Request/Response/Params/Result の接尾辞の付与と型レベルでの I/O 明示
- [ ] ログ種別・日次カウンタの適切な利用（重複ロギングの抑制）

---

## ⚙️ 主な環境変数

| 変数 | 用途 | 例 |
|------|------|----|
| `ZAIF_API_KEY` / `ZAIF_API_SECRET` | 認証キー | `xxx` |
| `USE_PRIVATE_MOCK` | プライベートAPIモック利用 | `1` |
| `EXCHANGE` | 取引所選択 (`zaif`/`coincheck`) | `zaif` |
| `COINCHECK_API_KEY` / `COINCHECK_API_SECRET` | Coincheck用キー | `xxx` |
| `DRY_RUN` | 約定シミュレーションのみ | `1` |
| `PAIR` | 取引ペア | `btc_jpy` |
| `LOOP_INTERVAL_MS` | メインループ周期 | `15000` |
| `KILL_SWITCH` | 新規停止+全キャンセル | `1` |
| `RISK_MIN_TRADE_SIZE` | 最小数量 | `0.0001` |
| `RISK_MAX_SLIPPAGE_PCT` | 許容スリッページ | `0.005` |
| `SMA_SHORT` / `SMA_LONG` | 短期・長期 SMA 期間 | `9` / `26` |
| `SELL_SMA_SHORT` / `SELL_SMA_LONG` | SELL モード専用の SMA 期間（未設定時は SMA_SHORT/LONG） | `9` / `26` |
| `BUY_SMA_SHORT` / `BUY_SMA_LONG` | BUY モード専用の SMA 期間（未設定時は SMA_SHORT/LONG） | `9` / `26` |
| `RSI_PERIOD` | RSI 期間 | `14` |
| `SELL_RSI_OVERBOUGHT` | SELL 判定用 RSI 上限（以上で売り強化） | `70` |
| `BUY_RSI_OVERSOLD` | BUY 判定用 RSI 下限（以下で買い強化） | `30` |
| `RISK_TRAIL_TRIGGER_PCT` | 利幅確保でトレール武装 | `0.05` |
| `RISK_TRAIL_STOP_PCT` | トレール利確幅 | `0.03` |
| `TRADE_MODE` | 戦略モード `SELL`/`BUY`（デフォルト `SELL`） | `SELL` |
| `MOCK_FORCE_EXIT` | モック強制部分 fill 継続 | `1` |
| `MOCK_FORCE_IMMEDIATE_FILL` | 初回即 partial fill | `1` |
| `TRADE_FLOW` | BUY/SELL フロー: `BUY_ONLY`/`SELL_ONLY`/`BUY_SELL`/`SELL_BUY`（デフォルト `BUY_SELL`） | `BUY_SELL` |
| `TEST_FLOW_QTY` | TRADE_FLOW 実行時の数量（>0 で有効化） | `0.002` |
| `LOG_LEVEL` | ログレベル `DEBUG`/`INFO`/`WARN`/`ERROR`（既定: `INFO`） | `DEBUG` |
| `LOG_JSON` | ログを1行JSONで出力（`1`で有効） | `1` |
| `MAX_NONCE_RETRIES` | Nonce リトライ回数 | `5` |
| `RETRY_BACKOFF_MS` | リトライ初期待機（指数バックオフの基準） | `300` |
| `RETRY_ATTEMPTS` | 公開API取得など withRetry の試行回数（services 層）。`1` で 1 回試行のみ | `2` |
| `RETRY_BACKOFF_FACTOR` | バックオフ係数（指数） | `1.5` |
| `RETRY_MAX_BACKOFF_MS` | バックオフ上限 | `3000` |
| `RETRY_JITTER_MS` | バックオフに加えるジッター | `100` |
| `NONCE_PERSIST` | Nonce の永続化（`0`で無効、既定: 有効） | `1` |
| `NONCE_RESTORE_ON_ERROR` | Nonce エラー時に復元（`0`で無効、既定: 有効） | `1` |
| `SAFETY_MODE` | 安全クランプ有効化（10%デフォルト） | `1` |
| `SAFETY_CLAMP_PCT` | クランプ割合（0-1）。例: 0.1=10% | `0.1` |
| `EXPOSURE_WARN_PCT` | 露出警告閾値（0-1）。既定 0.05=5% | `0.05` |
| `ML_SEARCH_MODE` | ML 探索モード `grid`/`random`/`earlystop` | `grid` |
| `ML_RANDOM_STEPS` | `random` モードでの試行回数 | `200` |
| `ML_EARLY_PATIENCE` | `earlystop` の打ち切り猶予ステップ | `10` |
| `ML_EARLY_MAX_STEPS` | `earlystop` の最大試行ステップ | `300` |
| `ML_MAX_WORKERS` | ML 探索の並列ワーカー数（CI では 1 推奨） | `1` |

追加: `RETRY_TIMEOUT_MS`, `RETRY_PRICE_OFFSET_PCT`, `CLOCK_SKEW_TOLERANCE_MS` などはコード参照。

---

## 🚀 使い方

### 1. 依存インストール
```
npm install
```

### 2. モック + ドライラン (最速検証)
```
USE_PRIVATE_MOCK=1 DRY_RUN=1 npm start
```

### 3. シナリオテスト
```
npm run mock:scenario
```

### 4. スモークテスト
```
npm run mock:smoke
```

### 5. 本番 (自己責任)
### 6. Coincheck 確認 (残高/発注/取消/履歴 最小テスト)
```
EXCHANGE=coincheck DRY_RUN=0 npm run test:coincheck
```
注意: 数量・rate・手数料・最小単位に留意し、極小サイズでテストしてください。

---

## 🎛 戦略モード切替 (TRADE_MODE)

- 環境変数 `TRADE_MODE=SELL|BUY` で戦略を切替（デフォルトは SELL）。
- モードごとに独立した実装:
	- `core/strategies/sell-strategy.ts` — SELL ファーストの売り起点戦略
	- `core/strategies/buy-strategy.ts` — BUY モード（買い戻し/反発起点）
- `app/index.ts` が `utils/config.loadTradeMode()` でモードを読み、該当戦略を実行。
- ログには `[SIGNAL][mode=SELL]` / `[SIGNAL][mode=BUY]` が出力されます。

---

## 🔄 発注フロー (TRADE_FLOW)

- `TRADE_FLOW=BUY_ONLY|SELL_ONLY|BUY_SELL|SELL_BUY`（既定: BUY_SELL）
- `TEST_FLOW_QTY>0` を設定すると、アプリの1サイクル内でフローを実行します。
	- BUY_SELL: BUY 約定確認後に SELL を必ず実施
	- SELL_BUY: SELL 約定確認後に BUY を必ず実施
- DRY_RUN=1 の場合は即時 fill をシミュレートします。
- 日次統計（stats）はフローに応じて buyEntries/sellEntries を加算します。

ワンショット実行（単発スモーク）
```
$env:USE_PRIVATE_MOCK="1"
$env:DRY_RUN="1"
$env:TRADE_FLOW="BUY_SELL"
$env:TEST_FLOW_QTY="0.002"
npm run smoke:once
```

---

## Coincheck でも npm start が動作します

環境変数で `EXCHANGE=coincheck` を指定すれば、共通の戦略ループ (`src/app/index.ts`) がそのまま Coincheck を利用します。

注意事項:
- 取引最小単位や丸めは Zaif と異なります。テストは小サイズ (例: 0.005 BTC) で開始し、エラー内容に応じて調整してください。
- 公開APIの板/トレードは取引所差異を吸収済みですが、銘柄は `btc_jpy` 固定を前提にしています。
- レート制限に注意し、短時間に連続発注/取消を避けてください。
```
ZAIF_API_KEY=... ZAIF_API_SECRET=... DRY_RUN=0 npm start
```

---

## 🧪 テスト / ユーティリティ

| コマンド | 内容 |
|----------|------|
| `npm run mock:scenario` | モックでシナリオ実行 (パラメータ組合せ) |
| `npm run mock:smoke` | 最小スモーク (起動～一連 cycle) |
| `npm run health` | 署名・nonce / permission ヘルスチェック |

補助: ツール実行の統合エントリ

`npm run tool -- <name>` でツールを統一的に呼び出せます（内部的に `src/tools/run.ts` が委譲）。

例:

```powershell
npm run tool -- live:health
npm run tool -- paper:mock-scenario
npm run tool -- ml:export
npm run tool -- stats:today -- --diff
```

### Coverage レポート（GitHub Pages）

main ブランチへの push または手動実行で `coverage-pages` ワークフローが走り、`coverage/` を GitHub Pages に公開します。しきい値 (statements >= 70%) を満たさない場合は CI が失敗します。

- Pages 公開先: リポジトリの Pages 設定で確認（`/coverage/` 以下に HTML レポート）
- ワークフロー: `.github/workflows/coverage-pages.yml`
 - 公開 URL（既定）: https://MakuhariYusuke.github.io/zaif-trade-bot/
	 - 本ワークフローは `coverage/` 配下を Pages ルートに配置するため、トップで coverage レポート（index.html）が表示されます。
	 - 404 の場合は GitHub Pages 有効化を確認し、`/index.html` で直接参照してください。

---

## 📑 日次レポートのリポジトリ内公開（デフォルト）

各ワークフロー（paper-nightly / paper-ml / paper-matrix / live-ml / live-trade）の末尾で、生成物をブランチに直接コミットします。

- パス: `reports/day-YYYY-MM-DD/`
- 目次: `reports/day-YYYY-MM-DD/index.md`（当日の主要アーティファクトへのリンクを列挙）
- 保持期間: 14 日分（古い日付は自動で削除）
- Slack / GitHub コメントにも「Added report: reports/day-YYYY-MM-DD/index.md」を追記します

ローカルや CI の成果を Pages に依存せず参照でき、履歴もブランチに残ります。

### GitHub Pages（任意）

Pages に ML/Stats のダッシュボードを公開するワークフロー（`.github/workflows/pages-ml-stats.yml`）は、既定で無効化されています。

- 有効化するには、リポジトリの Actions Variables に `ENABLE_PAGES=1` を設定してください（Settings → Secrets and variables → Actions → Variables）。
- その上でリポジトリの Pages を GitHub Actions 発行に設定します（Settings → Pages → Build and deployment → Source: GitHub Actions）。

Pages を無効のままでも、上記の `reports/day-YYYY-MM-DD/` で日次レポートは参照可能です。

### JSONL ベースライン（CSV 廃止）

特徴量・統計の出力は JSONL に統一しました（1 行 1 レコード追記）。移行期間の CSV 読み取り互換は削除済みです。

- 特徴量: `logs/features/<source>/<pair>/features-YYYY-MM-DD.jsonl`
- 日次統計スナップショット: `logs/pairs/<pair>/stats-YYYY-MM-DD.jsonl` および合算 `logs/stats-YYYY-MM-DD.jsonl`
- ML データセット: ルート `ml-dataset.jsonl`

ユーティリティ `readFeatureCsvRows(dir)` は JSONL のみを読みます（名称は互換維持）。

### テクニカル指標の自動付与

`features-logger` にインジケータ計算を内蔵し、足取り（近似 OHLC）から以下を出力します。

- トレンド系: SMA/EMA/WMA/HMA/KAMA（短期/長期）
- オシレーター: RSI、ROC、Momentum、CCI、Williams %R、ストキャスティクス
- ボラティリティ: 標準偏差（StdDev）、ATR、ボリンジャーバンド幅（BB width）、ドンチャン幅（Donchian width）
- 複合/その他: MACD（line/signal/hist）、DMI/ADX、パラボリックSAR、一目均衡表（転換/基準/先行/遅行）
- 価格乖離/帯: 移動平均からの乖離率（deviation%）、エンベロープ、簡易 Fibonacci 位置

エイリアス列（使い勝手向上のため）
- `rsi14`（既定 14 の RSI）、`atr14`（既定 14 の ATR）、`macd_hist`（MACD ヒストグラム）

観測開始時に 1 度だけ、WARN ログでサンプルを出します。
- 例: `[WARN] [INDICATOR] sample pair=btc_jpy rsi14=57.3 atr14=123.4 macd_hist=-0.002 ...`
- `IND_LOG_EVERY_N` > 0 の場合、N レコードごとに `[IND]` ダイジェストを DEBUG 出力します。

制御用の主な環境変数:

- `IND_LOG_EVERY_N` = N 件ごとに `[IND]` ダイジェストを DEBUG ログに出力（0 で無効）

### 週次トレンド（7 日集計）

`npm run trend:weekly` で直近 7 日の集計を生成し、以下へ保存します。

- `reports/day-YYYY-MM-DD/trend-7d.json`（当日ディレクトリ）
- `reports/week-YYYY-WW/weekly-summary.json`（ISO 週）
- `reports/latest/trend-weekly.json`（最新へのリンク）

ワークフロー（paper-nightly / paper-ml など）から自動実行・コミットされ、Slack/PR コメントにも `Trend7d` が含まれます。

#### 通知（Slack / GitHub コメント）: Trend7dWin%

---

## Rate Limiter

- 集中トークンバケツで API 呼び出しを制御します。
- 既定: `capacity=100`, `refill=10 tokens/sec`, 予約枠 `RATE_PRIORITY_RESERVE=0.1`（高優先専用）。
- 優先度: `high`（発注/取消）、`normal`（約定ポーリング等）、`low`（市場データ/統計）。
- `BaseService.withRetry` のフロー順序: CircuitBreaker → RateLimiter → 実行。

ログ仕様
- `RATE/INFO` … 取得成功（`waitedMs` 同報）
- `RATE/WARN` … 200ms 超の待機
- `RATE/ERROR` … 1 秒以内にトークン取得できず拒否（`code: RATE_LIMITED`）

環境変数
- `RATE_CAPACITY`（既定 100）
- `RATE_REFILL_PER_SEC`（既定 10）
- `RATE_PRIORITY_RESERVE`（既定 0.1）

使い方（カテゴリと優先度の例）
```ts
await withRetry(() => getTicker('btc_jpy'), 'getTicker', 2, 100, { category: 'API-PUBLIC', priority: 'low' });
await withRetry(() => private.placeLimitOrder(...), 'placeLimitOrder', 3, 150, { category: 'API-PRIVATE', priority: 'normal' });
await withRetry(() => exec.cancelOrder(id), 'cancelOrder', 3, 150, { category: 'EXEC', priority: 'high' });
```

`report-summary-*.json` を各ワークフロー（live-ml / live-trade / paper-ml / paper-nightly）で生成し、Totals に PnL/Win%/MaxDD に加えて 7 日移動の勝率 `Trend7dWin%` を含めて通知します。

例（Totals 1 行表示）:

```
Totals: pnl=+12.34, winRate=61.1%, maxDD=3.2, Trend7dWin%=64.3
```

サマリー JSON（抜粋）:

```json
{
	"source": "live",
	"totals": {

		"winRate": 0.611,
		"maxDrawdown": 3.2,
		"trend7dWinRate": 0.643
	}
}
```
## 🧼 Core 純化と I/O の外出し（今回の変更）

- 目標: core は純粋ロジックのみ。I/O（fs/HTTP）、logger、環境変数依存は adapters に集約。
- 対象: `core/risk.ts` と `core/position-store.ts` の純粋化を実施。
	- `core/risk.ts`: I/O と logger を撤去し、環境変数のみを読む `getRiskConfig()` と計算系関数に限定。
	- `core/position-store.ts`: FS/永続化と logger を撤去し、実体は `@adapters/position-store-fs` に委譲（API は不変）。
- アダプタ:
	- `@adapters/risk-config`: リスク設定読み込み/ポジション永続化の I/O を担当。
	- `@adapters/position-store-fs`: ポジションストアの FS 実装（環境変数 `POSITION_STORE_DIR`/`POSITION_STORE_FILE` を解決）。
	- `@adapters/risk-service`: 旧 API 互換のため、`getRiskConfig`/`getPositions`/`savePositionsToFile` の別名エクスポートを提供。

互換性:
- 既存の import は shim/再エクスポートで動作を維持しつつ、初回のみ CONFIG/WARN を 1 回出力（次メジャーで削除予定）。
- テスト/カバレッジは既存基準（Statements >= 70%）を維持。

推奨移行先（新規コード）:
- PositionStore: `import { loadPosition, savePosition } from '@adapters/position-store-fs'`
- Risk 設定/永続: `import { getRiskConfig, getPositions, savePositionsToFile } from '@adapters/risk-service'`

---

## 🧰 Paper Matrix（シナリオ実行）

GitHub Actions の `paper-matrix` では、モック駆動の多様なシナリオを並列実行して統計/レポートを生成・アーカイブします（tar.gz）。主なプリセットと用途:

- normal: 既定の軽負荷動作
- error: エラー頻度を上げる（`SCENARIO_PAPER_ERROR_RATE=0.2`）
- latency: レイテンシ付加（`SCENARIO_PAPER_LATENCY_MS=200`）
- timeout: API タイムアウト付加（`SCENARIO_PAPER_TIMEOUT_MS=1000`）
- composite: error+latency の複合
- hf-light: 高頻度ライト（`LOOP=2000`）
- hf-mid: 高頻度ミドル（`LOOP=5000`）
- hf-stress: 高頻度ストレス（`LOOP=10000`）
- sweep-rsi: RSI しきい値スイープ
- sweep-sma: SMA パラメータスイープ
- high-error: 高エラー率（`SCENARIO_PAPER_ERROR_RATE=0.5`）
- stress: レイテンシと軽負荷ループ（`SCENARIO_PAPER_LATENCY_MS=500`, `LOOP=1000`）

ローカル実行の例（PowerShell）:

```powershell
$env:USE_PRIVATE_MOCK="1"; $env:DRY_RUN="1"; $env:SCENARIO_PAPER_LATENCY_MS="200"; npm run mock:scenario
```

CI ではシナリオごとに `stats-<scenario>.json/.svg` と `report-summary-<scenario>.json` をまとめて `stats-<scenario>.tar.gz` としてアップロードしています。

---
## 🧠 ML 検索/レポート（grid/random/earlystop）

ツール群（`src/tools/ml/*`）で特徴量の集約/探索/レポート生成を行います。データセットには以下の列が含まれます（抜粋）。

- 時系列特徴: 価格、SMA、RSI 等
- 付加列: `source`（paper/live）、`tradeFlow`、`durationSec`（エントリー〜エグジットの概算秒）

探索モード（`ML_SEARCH_MODE`）
- `grid`: しきい値の格子探索（従来）
- `random`: ランダム探索。`ML_RANDOM_STEPS` で試行回数を制御
- `earlystop`: 早期打ち切り探索。`ML_EARLY_PATIENCE` と `ML_EARLY_MAX_STEPS` で制御
- 並列度は `ML_MAX_WORKERS`（CI では 1 推奨）

---

## 📦 Path alias と移行ガイド

- 型は `src/contracts` に集約しました。以後は `@contracts` で import してください。
	- 例: `import { PositionStore, RiskManager } from '@contracts'`
- 実装は `@adapters/*` と `@application/*` を利用してください（旧 `src/services/*` と `src/strategies/*` は削除済み）。
	- 例: `import { createServicePositionStore } from '@adapters/position-store'`
	- 例: `import { runBuyStrategy } from '@application/strategies/buy-strategy-app'`

生成物（ルート直下）
- `ml-dataset.jsonl`（特徴量データ・JSON Lines）
- `ml-search-results.csv`（全試行） / `ml-search-top.json`（上位）
- `report-ml-<mode>.json` / `report-ml-<mode>.csv`（mode は `grid|random|earlystop`）
- Feature Importance: `report-ml-feature-importance.json`（Top N 特徴量） / `importance.csv`

ローカル実行例（PowerShell）

```powershell
# 特徴量のエクスポート
npm run tool -- ml:export

# grid 探索
$env:ML_SEARCH_MODE="grid"; npm run tool -- ml:search

# random 探索（200 ステップ）
$env:ML_SEARCH_MODE="random"; $env:ML_RANDOM_STEPS="200"; npm run tool -- ml:search

# early stopping 探索（猶予 10、最大 300）
$env:ML_SEARCH_MODE="earlystop"; $env:ML_EARLY_PATIENCE="10"; $env:ML_EARLY_MAX_STEPS="300"; npm run tool -- ml:search

# Feature Importance（相関ベースの簡易重要度・Top20）
npm run feature:importance
```

補足: 簡易シミュレーション（ml-simulate）の勝率算出
- トレード判定: pnl が数値、または win フラグが存在する行をトレードとしてカウント
- 勝ち判定: win が 1 | true | '1' の行を勝ちとしてカウント
- PnL 集計: pnl がある場合のみ合算（win フラグのみの行は PnL=0 とみなす）

CI 連携
- `paper-ml` / `live-ml` ワークフローで `grid` の後に `random` を実行し、`report-ml-random.json/.csv` をアーティファクト化
- 併せて Feature Importance を実行し、`report-ml-feature-importance.json` と `importance.csv` を成果物に含め、レポート index にリンクします
- 通知（Slack/GitHub コメント）に「ML(random) Top: Win%/PnL/params」に加えて「Top Features: name1,name2,name3…」を 1 行で追記します

備考: `stats-graph` は paper / live の PnL・勝率のテキストオーバーレイを含む SVG を出力します。

---
## インジケータ（補足）

- 実装: 純粋関数群は `src/utils/indicators.ts` に集約（副作用なし）。EMA/SMA/RSI/MACD/ATR/ボリンジャー/一目/CCI/ROC/Momentum/HMA/KAMA/Donchian/Choppiness/Aroon/Vortex/SuperTrend 等
- エイリアス推奨: `rsi14`/`atr14`/`macd_hist` など既定パラメータのショート名を採用
- スナップショット: 指標は features-logger から JSONL に追記。`IND_LOG_EVERY_N>0` で N レコードおきに `[IND]` ダイジェストを DEBUG 出力
- 欠落サンプルの WARN: 監視開始時に 1 回だけ代表値を WARN ログで提示（冗長抑制のため once）
- 体感的な注意点: 体積（出来高）依存の指標は現行フローで取得が無い場合があり、その際は null を返します（WARN 付与）。アルゴ側で null セーフに取り扱ってください

---
### テスト実行時の環境変数の注意

テストはファイル出力（統計やポジションストア）を行うため、実データと混ざらないよう一時ディレクトリを使うことを推奨します。テストコード側でも設定していますが、手動実行時は以下を任意のパスに設定してください。

- `STATS_DIR`: 日次統計の出力先ディレクトリ（例: `./.tmp-stats/logs`）
- `POSITION_STORE_DIR`: ペア別のポジションファイルを保存するディレクトリ（例: `./.positions-test`）
- `POSITION_STORE_FILE`: 旧式の単一ストアファイルのパス（互換目的のみ）

PowerShell 例:

```powershell
$env:STATS_DIR=".tmp-stats\\logs"; $env:POSITION_STORE_DIR=".positions-test"; npm run test
```

---

## � SAFETY_MODE=1（数量クランプと WARN ログ）

`SAFETY_MODE=1` を指定すると、発注数量は残高の一定割合（既定 10%: `SAFETY_CLAMP_PCT=0.1`）にクランプされます。クランプが発生した場合は WARN ログが出力されます。

例:

```
[WARN] [SAFETY] amount clamped side=bid requested=20000 clamped=10000 pct=10.0%
```

この挙動はユニットテストで検証済みで、CI でも維持されます。

---

## �📊 ログ & 日次統計

ログ種別: SIGNAL / ORDER / EXECUTION / ERROR / INFO。`logs/trades-YYYY-MM-DD.log` に JSON 追記。

日次統計 `stats-YYYY-MM-DD.json`
- filledCount / realizedPnl / 各 *RetryCount / trailArmedTotal / trailExitTotal / trailStops / sellEntries 他

Trailing Stop:
1. 価格がエントリー比 `RISK_TRAIL_TRIGGER_PCT` 上昇で武装
2. 最高値を追従し `highestPrice * (1 - RISK_TRAIL_STOP_PCT)` を割れたら EXIT

---

## 🔍 約定ヒューリスティック (fallback)

`order_id` で特定できない場合:
1. `submittedAt ± SLIPPAGE_TIME_WINDOW_MS`
2. side 一致
3. 価格偏差 <= `TOL_PRICE_PCT`
4. 部分累積量が目標量に到達で完了

---

## ♻️ リネーム / コード規約差分 (最近適用)

| 旧 | 新 |
|----|----|
| `IPrivateApi` | `PrivateApi` |
| `TradeHistoryItem` | `TradeHistoryRecord` |
| `trailManager` | `manageTrailingStop` |
| `calcSMA` / `calcRSI` | `calculateSma` / `calculateRsi` |
| `submitWithRetry` | `submitOrderWithRetry` |
| `appendPrices` | `appendPriceSamples` |
| `getRecentPrices` | `getPriceSeries` |
| `writeTradeLog` | `logTrade` |
| `ok` (GuardResult) | `isAllowed` |
| `fillCount` | `filledCount` |
| `*Retries` | `*RetryCount` |

---

## ⚠️ 注意事項

本リポジトリは学習・検証目的の参考実装です。実運用前に以下を必ず検討してください。
- API レート制限と指数バックオフ
- 完全な約定照合 (WebSocket / order executions)
- 本番用秘密情報管理 (Vault / KMS 等)
- 冗長化 / 障害復旧 / アラート設計
- 高精度時刻同期 (NTP) と nonce 管理

---

## 🛠 今後の拡張候補

- WebSocket 板/トレード購読
- マルチペア同時戦略 / ポートフォリオ管理
- PnL / リスク指標 (Sharpe, 最大ドローダウン) 生成
- Backtest エンジン統合
- Prometheus / OpenTelemetry 連携

---

## 📄 ライセンス・利用条件

ライセンスは未確定ですが、現時点では学習・検証目的での個人利用を想定しています。
商用利用や再配布はご遠慮ください。今後 OSS ライセンス（MIT など）を付与する方針です。
正式なライセンス決定後は LICENSE ファイルおよび README にて告知します。

***

## 最小ライブ検証（自己責任）/ Live Minimal（DRY_RUN統合）

発注→即キャンセルの最小検証を、単一スクリプト `live:minimal` に統合しました。DRY_RUN=1 ならモック/ドライラン、DRY_RUN=0 なら実発注（要API鍵）で動きます。少額で慎重に実施してください。

実行コマンド: `npm run live:minimal`

使用する環境変数:
- EXCHANGE: 取引所（例: coincheck, zaif）
- TRADE_FLOW: BUY_ONLY | SELL_ONLY | BUY_SELL | SELL_BUY
- TEST_FLOW_QTY: 発注数量（DRY_RUN=1 の場合は未設定時 0.002 を既定値として使用）
- TEST_FLOW_RATE: 価格（未指定なら板の最優先を利用）。`ORDER_TYPE=market` の場合は未指定でOK
- ORDER_TYPE: `market` または `limit`（省略可）
- DRY_RUN: `0`（実発注）/ `1`（ドライラン）
- SAFETY_MODE: 1 で残高の10%以内に数量をクランプ
 - SAFETY_CLAMP_PCT: クランプ割合（例: 0.1=10%）
 - EXPOSURE_WARN_PCT: 露出警告閾値（例: 0.05=5%）

PowerShell 実行例（DRY_RUN=0 実発注・即キャンセル）:

```powershell
$env:EXCHANGE="coincheck"; $env:TRADE_FLOW="BUY_ONLY"; $env:TEST_FLOW_QTY="1"; $env:TEST_FLOW_RATE="490"; $env:DRY_RUN="0"; $env:SAFETY_MODE="1"; npm run live:minimal
```

DRY_RUN=1（ドライラン/モック）:

```powershell
$env:USE_PRIVATE_MOCK="1"; $env:EXCHANGE="coincheck"; $env:TRADE_FLOW="BUY_ONLY"; $env:TEST_FLOW_QTY="0.002"; $env:DRY_RUN="1"; npm run live:minimal
```

通知に Top 3（ML random）の表を追加しました（Slack/GitHub コメント）。例:

| # | pair | Win% | PnL | Params |
| --- | --- | --- | --- | --- |
| 1 | btc_jpy | 62 | 12.3 | S=9,L=26,RSI=70,30 |
| 2 | eth_jpy | 59 | 10.8 | S=11,L=29,RSI=65,28 |
| 3 | xrp_jpy | 57 | 9.4 | S=7,L=21,RSI=60,25 |

注: Live最小トレード検証は features ログ（JSONL のみ）も保存します。収集したデータは `npm run ml:export`（src/tools/ml/ml-export.ts）でデータセット化し（`ml-dataset.jsonl` を出力）、`npm run ml:search`（src/tools/ml/ml-search.ts）で簡易探索が可能です。

### ヘルスチェック

`npm run health` で以下を確認します。
- Private API: `healthCheck()`（未実装の場合は get_info2）
- Public API: `ticker` と `orderbook` のベスト気配、および `EXPOSURE_WARN_PCT` の現在値（warnPct）

## Live Minimal Trade (GitHub Actions)

手動で最小ライブトレードを走らせ、要約をSlack/GitHubに通知するワークフローを用意しています。

- ワークフロー: `.github/workflows/live-trade.yml`
- 実行条件: 手動起動（Actionsタブ → Live Minimal Trade → Run workflow）
- 使用Secrets:
	- `COINCHECK_KEY` / `COINCHECK_SECRET`（CoincheckのAPIキー）
	- `SLACK_WEBHOOK_URL`（任意。未設定なら通知スキップ）
- 固定パラメータ（必要に応じ変更可）:
	- `EXCHANGE=coincheck`, `PAIR=xrp_jpy`, `TRADE_FLOW=BUY_ONLY`
	- `TEST_FLOW_QTY=500`, `TEST_FLOW_RATE=490`, `DRY_RUN=0`, `FEATURES_SOURCE=live`

成果物/通知:
- Artifacts: `logs/live/summary-*.json`, `logs/features/live/**`
- Slack: 成功時 ✅/失敗時 ❌ とサマリ本文（Pair/Buy/Sell/PnL/WinRate/Warn）
- GitHub: PR/commit へ要約コメント投稿


改善 PR / Issue 歓迎。

***

## 機械学習用データの利用例（Python）

`npm run ml:export` で `ml-dataset.jsonl`（JSON Lines）を生成できます。以下は最小の分類タスク例です。

scikit-learn:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_json('ml-dataset.jsonl', lines=True)
X = df[["rsi","sma_short","sma_long","price","qty"]].fillna(0)
y = df["win"].fillna(0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, pred))
```

TensorFlow (Keras):

```python
import pandas as pd
import tensorflow as tf

df = pd.read_json('ml-dataset.jsonl', lines=True)
X = df[["rsi","sma_short","sma_long","price","qty"]].fillna(0).values
y = df["win"].fillna(0).astype(int).values

model = tf.keras.Sequential([
	tf.keras.layers.Input(shape=(X.shape[1],)),
	tf.keras.layers.Dense(32, activation='relu'),
	tf.keras.layers.Dense(16, activation='relu'),
	tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=5, batch_size=64)
```

PyTorch:

```python
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

df = pd.read_json('ml-dataset.jsonl', lines=True)
X = torch.tensor(df[["rsi","sma_short","sma_long","price","qty"]].fillna(0).values, dtype=torch.float32)
y = torch.tensor(df["win"].fillna(0).astype(int).values, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = nn.Sequential(
	nn.Linear(X.shape[1], 32), nn.ReLU(),
	nn.Linear(32, 16), nn.ReLU(),
	nn.Linear(16, 1), nn.Sigmoid()
)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

for epoch in range(5):
	for xb, yb in loader:
		pred = model(xb)
		loss = loss_fn(pred, yb)
		opt.zero_grad(); loss.backward(); opt.step()
	print('epoch', epoch, 'loss', float(loss))
```
