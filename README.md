# Zaif Trade Bot

軽量・検証重視の **Zaif 自動売買ボット**。SELL ファースト戦略 / モック駆動テスト / リスク管理を抑えた拡張しやすい土台です。

---

## 📌 概要

- SELL エントリーを起点に SMA / RSI / トレーリングストップでエグジット判断
- DRY_RUN で実注文なしの検証 / USE_PRIVATE_MOCK=1 でモック fills シミュレーション
- EXCHANGE=zaif|coincheck で取引所を切替 (初期値 zaif)
- 冪等 nonce・署名 / 約定ヒューリスティック補完 / 日次統計とトレードログ

---

## 📂 ディレクトリ構成

```
src/
 ├─ app/                 # メイン戦略ループ / 起動制御
 ├─ core/                # execution / market / risk / position-store などコアドメイン
 │   └─ strategies/      # モード別戦略: sell-strategy.ts / buy-strategy.ts
 ├─ api/                 # Zaif REST (public/private) + mock 実装
 ├─ services/            # 将来拡張向けサービス層 (一部レガシー呼称残り)
 ├─ types/               # 型定義 (PrivateApi, OrderLifecycleSummary 等)
 ├─ utils/               # logger, signer, daily-stats, price-cache
 └─ tools/
	 ├─ live/            # ライブ環境向けツール（health, 最小トレード, coincheck/zaif テスト等）
	 ├─ paper/           # モック/ペーパー用ツール（シナリオ, スモーク, リセット）
	 ├─ ml/              # 機械学習データ生成・探索
	 ├─ stats/           # 日次統計の取得・グラフ化
	 └─ tests/           # 統合テスト系
```

---

## 🧩 命名規則 (抜粋)

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
	- 最小実行ツールは `src/tools/*`、テストスクリプトは `src/tools/tests/*`。
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

#### 通知（Slack / GitHub コメント）: Trend7dWin%

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
		"pnl": 12.34,
		"winRate": 0.611,
		"maxDrawdown": 3.2,
		"trend7dWinRate": 0.643
	}
}
```

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

生成物（ルート直下）
- `ml-dataset.csv`（特徴量データ）
- `ml-search-results.csv`（全試行） / `ml-search-top.json`（上位）
- `report-ml-<mode>.json` / `report-ml-<mode>.csv`（mode は `grid|random|earlystop`）

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
```

CI 連携
- `paper-ml` / `live-ml` ワークフローで `grid` の後に `random` を実行し、`report-ml-random.json/.csv` をアーティファクト化
- 通知（Slack/GitHub コメント）に「ML(random) Top: Win%/PnL/params」の 1 行サマリを含めます

備考: `stats-graph` は paper / live の PnL・勝率のテキストオーバーレイを含む SVG を出力します。

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

注: Live最小トレード検証は features ログ（CSV/JSON）も保存します。収集したデータは `npm run ml:export`（src/tools/ml/ml-export.ts）でデータセット化し、`npm run ml:search`（src/tools/ml/ml-search.ts）で簡易探索が可能です。

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

`npm run ml:export` で `ml-dataset.csv` を生成できます。以下は最小の分類タスク例です。

scikit-learn:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('ml-dataset.csv')
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

df = pd.read_csv('ml-dataset.csv')
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

df = pd.read_csv('ml-dataset.csv')
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
