# 013# Exchange API 実装調査レポート

- **Track**: Ph2-audit
- **Date**: 2026-02-13
- **Author**: Copilot
- **Scope**: 既存取引所 API 基盤の棚卸し、v460 実装との重複分析、統合提言

---

## §1 調査目的

1. プロジェクトに散在する取引所 API 実装の全容を把握する
2. v460 で新設した部品との機能重複を特定する
3. 実取引開始に向けて削減・統合可能な箇所を明らかにする

---

## §2 既存実装の全体地図

### §2.1 アーキテクチャ層

```
┌──────────────────────────────────────────────────────────────────┐
│  系統③ ztb/trading/live_trader/ (本番用, 1,857行+14ファイル)      │
│  LiveTrader: SAC/PPO + PositionManager + Discord + Health       │
│  main.py → trading_loop.py → action_prediction → order_manager  │
│                    │ uses                                        │
│  ┌─────────────────▼───────────────────────────────────────────┐ │
│  │ 系統② ztb/trading/live/ (取引所抽象化層)                     │ │
│  │ IBroker ← CoincheckAdapter (756行) + BitFlyerAdapter (528行)│ │
│  │        ← SimBroker (338行) + broker_registry                │ │
│  │ BaseExchangeAdapter (414行)                                  │ │
│  └──────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  系統① ztb/live_trading/ (初期プロトタイプ, 770行)                │
│  TradingAPI (ccxt モック, 305行) + LiveTrader (464行)             │
│  独自型: OrderInfo / Position                                    │
│  現状: モック状態だが、プロジェクト目的の核心部品として要維持       │
└──────────────────────────────────────────────────────────────────┘

ztb/trading/live_trade.py (51行): deprecated ラッパー → 系統③へ転送
```

> **注 (Appendix D 反映)**: 013# 初版では 2 系統のみ記載していたが、系統③ `ztb/trading/live_trader/` が完全に欠落していた。上図は D.4 に基づき 3 系統として再構成。

### §2.2 各実装の詳細

| # | コンポーネント | 行数 | 取引所 | 継承 | 実 API | v460 市場データ |
|---|---------------|------|--------|------|--------|----------------|
| A | `CoincheckAdapter` | 756 | Coincheck | `IBroker` 直接 | ✅ place/cancel/status/balance + public ticker/orderbook/trades | ✅ `get_orderbook` / `get_recent_trades` |
| B | `BitFlyerAdapter` | 528 | bitFlyer | `BaseExchangeAdapter` | ✅ 全メソッド実装 | ✅ `get_orderbook` / `get_recent_trades` |
| C | `BaseExchangeAdapter` | 414 | 汎用 | `IBroker` (ABC) | dry-run共通ロジック | ✗ (default raises) |
| D | `ZaifAdapter` | 62 | Zaif | `IBroker` 直接 | ✗ 全 `NotImplementedError` | ✗ |
| E | `SimBroker` (simulation) | 338 | なし | `IBroker` | dry-run のみ | ✗ |
| F | `SimBroker` (registry) | 40 | なし | なし (独自) | dry-run のみ | ✗ |
| G | `TradingAPI` (live_trading) | 305 | Zaif | なし | ccxt モック | ✗ |
| H | `LiveTrader` | 464 | 汎用 | なし | `TradingAPI` 依存 | ✗ |

### §2.3 データ収集系

| # | コンポーネント | 行数 | 対象 | v460 関連度 |
|---|---------------|------|------|-------------|
| I | `MarketDataCollector` | 357 | 汎用 (IBroker) | **v460 中核** — tick raw + 1min集約 |
| J | `CoincheckDataFetcher` (v456) | 298 | Coincheck trades | **高** — /api/trades → OHLC |
| K | `CoincheckDataFetcher` (tools) | 147 | Coincheck candle | 中 — candle_rates → CSV |
| L | `CoinGeckoStream` | 335 | CoinGecko | 低 (設計パターン参考) |
| M | `BinanceData` | 318 | Binance | 低 (Parquet保存参考) |
| N | `StreamBuffer` | 431 | 汎用 | **補完的** — 高機能メモリバッファ |
| O | `stream_convert` | 44 | 汎用 | 低 — CSV→Parquet変換 |

### §2.4 設定・インフラ

| # | コンポーネント | 行数 | 内容 |
|---|---------------|------|------|
| P | `coincheck.yaml` | 105 | venue定義 (symbol/fee/risk/market_data) |
| Q | `bitflyer.yaml` | 106 | venue定義 |
| R | `BaseExchangeConfig` | 96 | .env→資格情報ロード基底 |
| S | `CoincheckConfig` | 40 | `COINCHECK_API_KEY/SECRET` |
| T | `BitflyerConfig` | 56 | `BITFLYER_API_KEY/SECRET` |
| U | `ExchangeProfile` | 84 | fee_model/slippage/latency のデータクラス |
| V | `RateLimiter` | — | 共通レート制限 (要求/秒) |
| W | WebSocket clients | ~120 | **ダミー stub** (yfinance 互換のみ) |

---

## §3 重複分析

### §3.1 CRITICAL: CoincheckAdapter の設計不整合

**問題**: `CoincheckAdapter` が `BaseExchangeAdapter` を継承せず、`IBroker` を直接実装している。

```python
# BitFlyerAdapter — 正  (BaseExchangeAdapter 経由)
class BitFlyerAdapter(BaseExchangeAdapter): ...

# CoincheckAdapter — 不整合 (IBroker 直接)
class CoincheckAdapter(IBroker): ...
```

**結果として発生する重複** (CoincheckAdapter 内にコピペされたコード):

| メソッド | Base版 (行) | Coincheck版 (行) | 同一性 |
|----------|-----------|-----------------|--------|
| `_simulate_delay()` | L110-114 | L149-155 | ~100% |
| `_check_rate_limit()` | L116-118 | L230-232 | 100% |
| `_generate_order_id()` | L120-122 | L234-236 | ~90% (uuid vs counter) |
| `_get_balance_dry_run()` | L125-130 | L606-611 | ~90% |
| dry-run order logic | L132-210 | L291-364 | ~85% |
| dry-run cancel | L257-264 | L400-408 | ~90% |
| dry-run position | L276-278 | L423-425 | ~95% |

**推定重複行数: ~150行** — 全756行の約20%。

### §3.2 HIGH: ztb/live_trading/ と ztb/trading/live/ の二重系統

| 機能 | `ztb/live_trading/` | `ztb/trading/live/` |
|------|--------------------|--------------------|
| 注文発行 | `TradingAPI.create_order()` (ccxt モック) | `IBroker.place_order()` (実API) |
| 残高取得 | `TradingAPI.get_balance()` (ccxt モック) | `IBroker.get_balance()` (実API) |
| ティッカー | `TradingAPI.get_ticker()` (モック) | `IBroker.get_current_price()` (実API) |
| ループ | `LiveTrader._trading_loop()` | `PaperTrader` / `run_fill_test.py` |
| Position管理 | `LiveTrader.positions` (独自) | `IBroker.get_positions()` |
| 型定義 | `OrderInfo` / `Position` (独自dataclass) | `Order` / `Position` (broker_interfaces) |
| 実用性 | **使われていない** (全メソッドがモック) | **v460で使用中** |

**結論 (D.6 訂正済)**: ~~`ztb/live_trading/` は完全に死んだコード~~ → `ztb/live_trading/` はモック状態の初期プロトタイプ。`TradingAPI` は ccxt を TODO コメントで参照するのみで、実際には定数モック値を返す。v460 の `trading/live/` 系とは型定義レベルで互換性がない。ただし、ユーザーから「このシステムの要」と明確な訂正があり、プロジェクト目的の核心部品として維持し、系統②③ との統合計画が必要。

### §3.3 MEDIUM: データ収集の trades API 重複

```
CoincheckAdapter.get_recent_trades()   ← v460 (リアルタイムtick)
         ↕ 同一エンドポイント
CoinCheckDataFetcher._get_trades_page() ← v456 (バッチ更新)
```

| 側面 | v460 (`CoincheckAdapter`) | v456 (`update_data_coincheck.py`) |
|------|--------------------------|----------------------------------|
| エンドポイント | `GET /api/trades?pair=..&limit=100` | `GET /api/trades?pair=..&limit=500&order=desc` |
| pagination | なし (最新100件のみ) | あり (`before` パラメータ) |
| リトライ | `NetworkError` 例外のみ | 指数バックオフ (最大5回) |
| レート制限 | `RateLimiter` (5 req/s) | `time.sleep(0.5)` 固定 |
| 出力 | `list[TradeRecord]` → `MarketDataCollector` | OHLCV DataFrame → CSV |
| 用途 | 5秒間隔ポーリング | 日次バッチ更新 |

エンドポイントは同一だが、用途が根本的に異なる (リアルタイム vs バッチ)。統合するよりも共存が適切。

### §3.4 LOW: SimBroker の二重定義

- `ztb/trading/live/simulation/sim_broker.py` → `SimBroker(IBroker)` — 338行、本格ペーパートレーダー
- `ztb/trading/live/registry/broker_registry.py` → `SimBroker` — 40行、簡易テスト用

別用途だが命名が衝突。レジストリ版は `SimpleSimBroker` 等にリネームが妥当。

---

## §4 実取引への導線の評価

### §4.1 Coincheck (v460 メイン取引所)

**即時使用可能な機能**:

| 機能 | 実装状態 | 備考 |
|------|---------|------|
| 認証 (HMAC-SHA256) | ✅ `_create_signature()` + `_make_api_request()` | nonce ベース |
| 注文発行 | ✅ `place_order()` — real パス実装済 | `POST /api/exchange/orders` |
| 注文キャンセル | ✅ `cancel_order()` — real パス実装済 | `DELETE /api/exchange/orders/{id}` |
| 注文状態確認 | ✅ `get_order_status()` — opens + transactions 二段階 | 009# P2-0 |
| 未決済注文一覧 | ✅ `get_open_orders()` — real パス実装済 | 009# P2-0 |
| ポジション (推定) | ✅ `get_positions()` — balance から推定 | spot なので直接 API なし |
| 残高取得 | ✅ `get_balance()` — real パス実装済 | `GET /api/accounts/balance` |
| 価格取得 | ✅ `get_current_price()` — public ticker | `GET /api/ticker` |
| 板情報 | ✅ `get_orderbook()` — public | `GET /api/order_books` |
| 約定履歴 | ✅ `get_recent_trades()` — public | `GET /api/trades` |
| 資格情報管理 | ✅ `CoincheckConfig` + `.env` | `COINCHECK_API_KEY/SECRET` |
| レート制限 | ✅ `RateLimiter` (5 req/s) | venue YAML では 300/min |
| Venue YAML | ✅ `coincheck.yaml` | min_lot, fee, risk_limits |

**結論 (D.6 訂正済)**: Coincheck は実取引に必要な全 API の骨格が実装済み。~~`dry_run=False` に切り替えるだけで本番稼働可能~~ → **本番前に以下の解決が必須**: C-3 署名不一致修正 ✅済、C-7 order_type マッピング修正 ✅済、D-1 OrderManager→CoincheckAdapter 接続 ✅済、D-3 post_only 実装 ✅済。

### §4.2 bitFlyer

**即時使用可能な機能**:

| 機能 | 実装状態 |
|------|---------|
| 認証 (HMAC-SHA256) | ✅ `_generate_signature()` + `_get_headers()` |
| 注文発行 | ✅ `_place_order_real()` — `/v1/me/sendchildorder` |
| 注文キャンセル | ✅ `_cancel_order_real()` — `/v1/me/cancelchildorder` |
| 注文状態確認 | ✅ `_get_order_status_real()` — `/v1/me/getchildorders` |
| 未決済注文一覧 | ✅ `_get_open_orders_real()` |
| ポジション | ✅ `_get_positions_real()` — `/v1/me/getpositions` |
| 残高取得 | ✅ `_get_balance_real()` — `/v1/me/getbalance` |
| 価格取得 | ✅ `_get_current_price_real()` — `/v1/ticker` |
| 板情報 | ✅ `get_orderbook()` — `/v1/board` |
| 約定履歴 | ✅ `get_recent_trades()` — `/v1/executions` |
| 資格情報管理 | ✅ `BitflyerConfig` + `.env` |
| Venue YAML | ✅ `bitflyer.yaml` |

**結論**: bitFlyer も全 API 実装済み。`BaseExchangeAdapter` 継承で dry/real 切り替えがクリーン。
v460 の初期戦略は Coincheck (maker fee 0%) だが、bitFlyer (maker rebate -0.02%) は代替/併用候補。

### §4.3 Zaif

**状態**: `ZaifAdapter` は全メソッド `NotImplementedError`。`TradingAPI` (live_trading) は ccxt 前提のモックのみ。**使用不可**。

---

## §5 v460 実装の重複排除可否

### §5.1 CoincheckAdapter → BaseExchangeAdapter 継承化

| 項目 | 判定 | 理由 |
|------|------|------|
| **技術的可否** | ✅ 可能 | dry-run ロジック ~150行を Base に委譲可。BitFlyerAdapter と同じパターンに統一 |
| **リスク** | ⚠ 中 | 既存テスト 78件の signature 変更なし確認が必要。real モードの `place_order()` 内の dry-run fallback 分岐が混在しており分離が必要 |
| **効果** | 約150行削減 + 保守性向上 | DRY原則遵守、BaseExchangeAdapter の一元管理 |
| **推奨時期** | G1.1 fill test 完了後 | 現在 fill test で real API を使用中のため変更リスク回避 |

具体的な変更内容:
1. `class CoincheckAdapter(IBroker)` → `class CoincheckAdapter(BaseExchangeAdapter)`
2. `_simulate_delay`, `_check_rate_limit`, dry-run 状態管理を `super().__init__()` に委譲
3. `_place_order_real`, `_cancel_order_real` 等の abstract メソッドを実装
4. `place_order()` 内の inline dry-run/real 分岐を Base のディスパッチに統一

### §5.2 ztb/live_trading/ の位置づけ (D.6 訂正済)

| 項目 | 判定 |
|------|------|
| **技術的可否** | ~~✅ 安全に廃止可能~~ → ❌ 即時廃止不可 |
| **理由** | 全メソッドがモック状態だが、ユーザーが「このシステムの要」と明言。5 件の参照が存在。段階的統合が必要 |
| **効果** | ~~770行削減~~ → 依存除去 → 代替導線整備 → 統合検討の段階移行 |
| **推奨**: | ~~`archived/` へ移動~~ → 系統②③ との統合計画を策定 |

### §5.3 データ収集の統合

| 対象 | 判定 | 理由 |
|------|------|------|
| `tools/fetch_coincheck_data.py` | 維持 | candle_rates API (集約済み OHLCV) は用途が異なる |
| `scripts/v456/update_data_coincheck.py` | 維持 | バッチ更新用。v460 の tick 収集とは目的が異なる |
| `StreamBuffer` の v460 組込み | ⬜ 将来検討 | 現在の list バッファで 24h は問題ないが、72h+ では StreamBuffer の圧縮・容量制限が有効 |

### §5.4 SimBroker 命名衝突

| 項目 | 判定 |
|------|------|
| `broker_registry.py` の `SimBroker` | `SimpleSimBroker` にリネーム推奨 |
| **効果** | 命名空間の明確化。import 時の混乱防止 |

---

## §6 結論と提言

### §6.1 即座に対応すべき事項

| # | 項目 | 重要度 | 工数 |
|---|------|--------|------|
| ~~1~~ | ~~`ztb/live_trading/` を `archived/` へ移動~~ | ~~HIGH~~ | D.6 訂正: 削除。即時 archive は不可。段階移行が必要 |
| 2 | `broker_registry.SimBroker` リネーム | LOW | 5min |

### §6.2 G1.1 fill test 完了後に対応すべき事項

| # | 項目 | 重要度 | 工数 | 削減行数 |
|---|------|--------|------|----------|
| 3 | CoincheckAdapter → BaseExchangeAdapter 継承化 | HIGH | 2-3h | ~150行 |
| 4 | WebSocket stub を削除 or 実装 | LOW | — | ~120行 |

### §6.3 既に適切に構築されている部分

- **IBroker インターフェース**: 抽象度が適切。v460 で `get_orderbook` / `get_recent_trades` を非 abstract で追加した設計は後方互換性を維持しておりクリーン。
- **BaseExchangeAdapter**: dry-run/real 切り替えの Template Method パターンが正しい。BitFlyerAdapter はこれを正しく活用。
- **MarketDataCollector**: IBroker に依存する設計で取引所非依存。Coincheck ↔ bitFlyer の切り替えが adapter 差し替えのみで可能。
- **Config 系**: BaseExchangeConfig → CoincheckConfig/BitflyerConfig の継承、.env 対応が整備済み。
- **Venue YAML**: symbol, fee, risk, market_data の定義が網羅的。PaperTrader から参照される導線も構築済み。

### §6.4 実取引開始までの残チェックリスト

実装は整っているが、以下の確認が本番前に必要:

| # | チェック項目 | 現状 |
|---|------------|------|
| 1 | `place_order()` real パスの order_type マッピング | ✅ **修正済**: `"buy"` / `"sell"` (指値), `"market_buy"` / `"market_sell"` (成行) — Coincheck 公式 4 値にマッピング。~~`limit_buy` / `limit_sell` は存在しない値~~ (D.6 訂正) |
| 2 | Market buy の `amount` vs `market_buy_amount` | ✅ **修正済**: market_buy は `market_buy_amount` (JPY 金額) を送信。`quantity * current_price` で BTC→JPY 変換。market_sell は `amount` (BTC 数量) を送信 |
| 3 | Content-Type ヘッダ | ✅ **確認済**: `"application/x-www-form-urlencoded"` + `urlencode` body で統一。C-3 署名修正と整合 |
| 4 | エラーハンドリングの体系化 | ⬜ 未対応: `success: false` の体系的チェックは追加済み (`place_order` 内) だが、全 API メソッドへの横展開は将来タスク |
| 5 | 指値注文の `rate` 型 | ⬜ 低優先: BTC_JPY は整数で問題なし。他ペア対応時に要修正 |

---

## Appendix A: ファイル一覧と行数

### 取引所 Adapter 系 (合計 2,775行)
| ファイル | 行数 |
|----------|------|
| `ztb/trading/live/exchanges/base/broker_interfaces.py` | 227 |
| `ztb/trading/live/exchanges/base/adapter.py` | 414 |
| `ztb/trading/live/exchanges/base/config.py` | 96 |
| `ztb/trading/live/exchanges/coincheck/adapter.py` | 756 |
| `ztb/trading/live/exchanges/coincheck/config.py` | 40 |
| `ztb/trading/live/exchanges/bitflyer/adapter.py` | 528 |
| `ztb/trading/live/exchanges/bitflyer/config.py` | 56 |
| `ztb/trading/live/simulation/sim_broker.py` | 338 |
| `ztb/trading/live/registry/broker_registry.py` | 168 |
| `ztb/trading/live/simulation/paper_trader.py` | 564 (※全行未読) |

### 初期プロトタイプ (要維持, 合計 ~770行) ※D.6 訂正: 「死コード」→「初期プロトタイプ」
| ファイル | 行数 | 状態 |
|----------|------|------|
| `ztb/live_trading/trading_api.py` | 305 | モック状態 (要統合検討) |
| `ztb/live_trading/live_trader.py` | 464 | TradingAPI 依存 (要統合検討) |

### 系統③ 本番トレーディングシステム (合計 ~1,857行) ※013# 初版で欠落
| ファイル | 行数 |
|----------|------|
| `ztb/trading/live_trader/live_trader.py` | ~600 |
| `ztb/trading/live_trader/components/` | ~1,200 |
| `ztb/trading/live_trade.py` | 51 |

### データ収集系 (合計 ~1,930行)
| ファイル | 行数 |
|----------|------|
| `ztb/data/market_data_collector.py` | 357 |
| `scripts/v456/update_data_coincheck.py` | 298 |
| `tools/fetch_coincheck_data.py` | 147 |
| `ztb/data/coin_gecko_stream.py` | 335 |
| `ztb/data/binance_data.py` | 318 |
| `ztb/data/stream_buffer.py` | 431 |
| `ztb/data/stream_convert.py` | 44 |

### v460 固有 (合計 ~1,830行)
| ファイル | 行数 |
|----------|------|
| `scripts/v460/run_fill_test.py` | 479 |
| `scripts/v460/run_observation.py` | 96 |
| `ztb/data/market_data_collector.py` | 357 |
| `ztb/metrics/fill_quality.py` | 300 |
| `ztb/features/microstructure.py` | 133 |
| `scripts/v460/lib/*.py` | ~465 |

## Appendix B: 改訂履歴

| 日付 | 変更 | 理由 |
|------|------|------|
| 2026-02-13 | 初版作成 | ユーザー要請: 既存 API 基盤調査 + v460 重複排除分析 |

## Appendix C: 追補レビュー (第2巡, 2026-02-13)

013# 本文は維持し、末尾に追補のみ追加する。
第2巡の再点検で、以下の見落とし/過小評価を確認した。

| # | 重大度 | 追補指摘 | 根拠 | 推奨対応 |
|---|--------|----------|------|----------|
| C-1 | HIGH | `ztb/live_trading/` を「安全に廃止可能」と断定するのは時期尚早。参照が残存。 | `tests/unit/experiments/test_phase_3_3_live_trading_integration.py:12`, `tests/unit/integration/paper_trading_experiment.py:22`, `scripts/v456/get_account_info.py:23`, `ztb/live_trading/__init__.py:10` | 即時 archive ではなく、依存除去 -> 代替導線整備 -> 最終 archive の段階移行に修正。 |
| C-2 | HIGH | 「`dry_run=False` だけで本番稼働可能」という結論と、未解決チェックリストが同居しており論理矛盾。 | `docs/v460/013_ph2_rpt_exchange_api_audit.md:176`, `docs/v460/013_ph2_rpt_exchange_api_audit.md:279` | §4.1 の結論を「実装は概ね完了、ただし事前検証必須」にトーンダウン。 |
| C-3 | HIGH | Coincheck 認証で、署名対象と実送信ボディの不一致リスクが未監査。 | `ztb/trading/live/exchanges/coincheck/adapter.py:178`, `ztb/trading/live/exchanges/coincheck/adapter.py:200`, `ztb/trading/orders/test_live_order.py:45`, `ztb/trading/orders/test_live_order.py:49` | POST 時は「署名文字列 = 実際に送る urlencode 済み body」の一致検証を必須チェックに追加。 |
| C-4 | HIGH | async/sync 整合性の見落とし。private API 経路で同期 `requests` が event loop をブロック。 | `ztb/trading/live/exchanges/coincheck/adapter.py:266`, `ztb/trading/live/exchanges/coincheck/adapter.py:395`, `ztb/trading/live/exchanges/coincheck/adapter.py:570`, `ztb/trading/live/exchanges/bitflyer/adapter.py:159` | private API も `asyncio.to_thread` か `httpx.AsyncClient` に統一。HFT 志向では優先度高。 |
| C-5 | MEDIUM | bitFlyer private API の `product_code` 取り扱い検証が不足。symbol 正規化も一貫していない。 | `ztb/trading/live/exchanges/bitflyer/adapter.py:211`, `ztb/trading/live/exchanges/bitflyer/adapter.py:267`, `ztb/trading/live/exchanges/bitflyer/adapter.py:295`, `ztb/trading/live/exchanges/bitflyer/adapter.py:353` | private API は `BTC_JPY` 正規化を強制し、必須パラメータ要件を実 API で再検証。 |
| C-6 | MEDIUM | 二重系統分析で `ztb/trading/live_trader/` 系の棚卸しが抜け、実際は「2系統」ではなく「3系統」。 | `ztb/trading/live_trader/live_trader.py:49`, `ztb/trading/live_trader/live_trader.py:68` | アーキテクチャ地図を 3 系統で再整理し、残す系統/捨てる系統の判定軸を明文化。 |
| C-7 | MEDIUM | Coincheck `order_type` 値の断定は根拠提示不足。現リポジトリ実装と記述が衝突。 | `docs/v460/013_ph2_rpt_exchange_api_audit.md:279`, `ztb/trading/live/exchanges/coincheck/adapter.py:257`, `ztb/trading/orders/test_live_order.py:41` | 断定文を「公式 API 仕様の再確認要」に変更。 |
| C-8 | LOW | 棚卸し行数・記述の鮮度ズレ。 | `docs/v460/013_ph2_rpt_exchange_api_audit.md:25`, `docs/v460/013_ph2_rpt_exchange_api_audit.md:33`, `docs/v460/013_ph2_rpt_exchange_api_audit.md:38`, `docs/v460/013_ph2_rpt_exchange_api_audit.md:61` | 行数は自動集計結果に固定し、手動更新を避ける。 |
| C-9 | LOW | CoincheckAdapter のクラス説明が実装実態と不整合（「real trading stubbed」表記）。 | `ztb/trading/live/exchanges/coincheck/adapter.py:71`, `ztb/trading/live/exchanges/coincheck/adapter.py:250` | クラス docstring を現状実装に合わせて更新。 |

### 追補結論

- 013# の方向性（重複排除・統合）は妥当だが、**本番投入可否と廃止判断は再校正が必要**。
- 特に C-3 / C-4 / C-5 は、API correctness と実運用レイテンシに直結するため、次フェーズで最優先に検証すべき。

---

## Appendix D: C-finding 検証結果 + 追加発見 (2026-02-14)

Appendix C の全 9 指摘をソースコード・公式 API ドキュメントで照合した結果を記載する。
000# §0 「本プロジェクトの大義は『短期間での高収益性システム』の実現」を判断基準として据える。

### D.1 C-finding 検証一覧

| C# | 重大度 | 判定 | 検証結果 |
|----|--------|------|----------|
| C-1 | HIGH | **妥当 + 昇格** | `from ztb.live_trading` で 5 件の参照が存在。さらに重要: ユーザーから「このシステムの要。収益を上げることが一番の目的」と明確な訂正あり。013# §3.2 の「完全に死んだコード」および §5.2 の「安全に廃止可能」は **根本的に誤り** である。000# §0 に照らし、live_trading/ はプロジェクト目的の中核部品。 |
| C-2 | HIGH | **妥当** | §4.1 「`dry_run=False` だけで本番稼働可能」と §6.4 の未解決 5 項目（order_type, market_buy_amount, 署名, エラーハンドリング, rate 精度）が論理矛盾。正: 「実装の骨格は完了、**本番前に §6.4 の解決が必須**」。 |
| C-3 | HIGH | **妥当 · 致命的** | `_make_api_request()` L178: 署名 = `nonce + url + json.dumps(data)`。L200: 実送信 = `urllib.parse.urlencode(data)`。Coincheck 公式ドキュメント: 「SIGNATUREは、ACCESS-NONCE, リクエスト先URL, **リクエストのボディ** を全て文字列にし連結」。**署名対象と送信ボディが不一致 → 本番で認証失敗が確定**。修正: 署名文字列を `nonce + url + urlencode(data)` に統一するか、Content-Type を `application/json` に変えて `json.dumps` で統一。 |
| C-4 | HIGH | **妥当** | async メソッド 8 個中 3 個 (`place_order`, `cancel_order`, `get_balance`) が同期 `requests` を `asyncio.to_thread` なしで直接呼出し。残り 5 個 (`get_order_status`, `get_open_orders`, `get_current_price`, `get_orderbook`, `get_recent_trades`) は 009# P2-0 で `asyncio.to_thread` 対応済み。**3/8 が未対応でイベントループをブロック**。 |
| C-5 | MEDIUM | **妥当** | bitFlyer: Public API (`get_orderbook`, `get_recent_trades`) は `symbol.upper()` 正規化あり。Private API (`place_order` L211 等) は `symbol` をそのまま `product_code` に渡す。"btc_jpy" → "BTC_JPY" 変換が欠落し、API エラーの原因となりうる。 |
| C-6 | MEDIUM → **HIGH昇格** | **妥当 · 重大見落とし** | `ztb/trading/live_trader/` (16 ファイル, LiveTrader 1,857 行) が 013# から完全に欠落。これは SAC/PPO モデル + CoincheckAdapter + PositionManager + Discord 通知 + ヘルスモニタリングを統合した **本番用トレーディングシステム** である。013# §2.1 のアーキテクチャ地図は「2 系統」だが実際は **3 系統** が存在する。`ztb/trading/live_trade.py` (51 行) は deprecated ラッパーとしてこのモジュールを指す。 |
| C-7 | MEDIUM | **妥当 · 013# に誤記あり** | Coincheck 公式 API: order_type は `"buy"` (指値買), `"sell"` (指値売), `"market_buy"` (成行買), `"market_sell"` (成行売) の **4 値**。013# §6.4 が記載した `"limit_buy"` / `"limit_sell"` は **存在しない値**。現コード `order_data["order_type"] = side` は指値注文には有効だが、成行注文で `"market_buy"` / `"market_sell"` への変換が未実装。さらに market_buy は `amount` ではなく `market_buy_amount` (JPY 金額) が必須。 |
| C-8 | LOW | **妥当** | 行数の微小な乖離は確認。実害は低い。 |
| C-9 | LOW | **妥当** | L71: `"Real trading is not implemented; all operations are simulated for testing."` — 実際には real API パスが place/cancel/status/balance に存在。docstring 更新要。 |

### D.2 013# が見落としていた追加発見

| D# | 重大度 | 発見 | 根拠 | 影響 |
|----|--------|------|------|------|
| D-1 | **HIGH** | `OrderManager.execute_trade()` の live モードが未接続 | `ztb/trading/live_trader/components/order_manager.py` L55: `# TODO: Implement actual exchange API trading calls` → `return False`。production LiveTrader → CoincheckAdapter.`place_order()` のブリッジが存在しない。 | **実取引が実行できない**。システム目的に直結するブロッカー。 |
| D-2 | **HIGH** | `ztb/live_trading/` の位置づけ根本誤認 | 013# §3.2: 「完全に死んだコード」。ユーザー: 「このシステムの要」。000# §0: 「短期間での高収益性システム」。 | 013# の中核的結論が覆る。§5.2 の廃止提言は撤回すべき。 |
| D-3 | **MEDIUM** | Coincheck `time_in_force: "post_only"` 未活用 | Coincheck 公式 API: `time_in_force` パラメータで `"post_only"` (Maker 注文のみ発注) をサポート。v460 は maker-only (手数料 0%) 戦略だがこのフラグを使っていない。 | taker 約定を防げない。maker-only 戦略の保証が API レベルで欠如。 |
| D-4 | **MEDIUM** | Coincheck WebSocket API の未活用 | Public: `wss://ws-api.coincheck.com` (trades, orderbook チャンネル, 0.1 秒間隔配信)。Private: `wss://stream.coincheck.com` (order-events, execution-events)。現在は 5 秒間隔の REST ポーリング。 | マイクロストラクチャ戦略としてレイテンシ面で不利。約定通知も REST ポーリングに依存。 |
| D-5 | **LOW** | Coincheck 新規注文レート制限の超過設定 | Coincheck 公式: 新規注文は **秒間 4 リクエスト** まで (5 以上で 429)。注文詳細は秒間 1 リクエスト。現在の `RateLimiter`: 5 req/s。 | 本番で 429 エラーが発生するリスク。 |

### D.3 000# 整合性評価

013# が見失っていた最も重要な視点:

- **000# §0**: 「本プロジェクトの大義は『短期間での高収益性システム』の実現」
- **000# §2**: ph5 = G4-live (Paper trading 運用検証) → 実取引への導線
- **000# §4**: 「Coincheck（主）/ Bitflyer / Zaif を含む maker 手数料 0% の国内取引所とし、API 品質・流動性に応じて切替可能な設計」

013# は取引所 API を「技術部品」としてのみ棚卸ししたが、**これらの部品が存在する理由** — すなわち実取引で収益を上げること — を軸に評価すべきだった。特に:

1. `ztb/live_trading/` と `ztb/trading/live_trader/` は ph5 → 実運用の車輪であり、「死コード」ではなく **未完の中核部品**
2. C-3 (署名不一致) と D-1 (OrderManager TODO) は、**実取引が物理的に不可能** であることを意味する
3. D-3 (`post_only` 未使用) は、000# §4 の「全取引は maker 注文（手数料 0%）で執行」の保証手段が欠如していることを意味する

### D.4 3 系統アーキテクチャの正確な全体像

013# §2.1 を以下に訂正する:

```
┌──────────────────────────────────────────────────────────────────┐
│  系統③ ztb/trading/live_trader/ (本番用, 1,857行+14ファイル)      │
│  LiveTrader: SAC/PPO + PositionManager + Discord + Health       │
│  main.py → trading_loop.py → action_prediction → order_manager  │
│                    │ uses                                        │
│  ┌─────────────────▼───────────────────────────────────────────┐ │
│  │ 系統② ztb/trading/live/ (取引所抽象化層)                     │ │
│  │ IBroker ← CoincheckAdapter (756行) + BitFlyerAdapter (528行)│ │
│  │        ← SimBroker (338行) + broker_registry                │ │
│  │ BaseExchangeAdapter (414行)                                  │ │
│  └──────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  系統① ztb/live_trading/ (初期プロトタイプ, 770行)                │
│  TradingAPI (ccxt モック, 305行) + LiveTrader (464行)             │
│  独自型: OrderInfo / Position                                    │
│  現状: モック状態だが、プロジェクト目的の核心部品として要維持       │
└──────────────────────────────────────────────────────────────────┘

ztb/trading/live_trade.py (51行): deprecated ラッパー → 系統③へ転送
```

**系統間関係**:
- 系統③ が系統② の CoincheckAdapter を利用する設計 (broker_registry 経由)
- 系統① は系統②③ と型互換なし (独自 OrderInfo vs IBroker.Order)
- 系統③ の OrderManager.execute_trade() が系統② へのブリッジだが **TODO 状態**

### D.5 優先修正事項 (013# §6 への追補)

| 優先度 | 項目 | 理由 | 工数見積 |
|--------|------|------|----------|
| **P0** | C-3: 署名文字列の修正 | 修正なしでは private API が全て認証失敗 | 30min |
| **P0** | D-1: OrderManager → CoincheckAdapter 接続 | 修正なしでは実取引が不可能 | 2-3h |
| **P1** | C-7/§6.4#1: order_type マッピング修正 | 成行注文で API エラー。limit 注文は動作するが market_buy は market_buy_amount 必須 | 1h |
| **P1** | D-3: `time_in_force: "post_only"` 実装 | maker-only 戦略の API レベル保証 | 30min |
| **P1** | C-4: place_order/cancel_order/get_balance の asyncio.to_thread 対応 | イベントループブロック解消 | 30min |
| **P2** | C-5: bitFlyer product_code 正規化統一 | 現時点では Coincheck が主のため緊急度は低い | 30min |
| **P2** | D-4: WebSocket API 検討 | REST ポーリング → WS 移行でレイテンシ改善 | 1-2日 |
| **P3** | C-9: docstring 更新 | 技術的影響なしだが誤解を招く | 10min |
| **P3** | D-5: RateLimiter を 4 req/s に修正 | 429 エラー予防 | 10min |

### D.6 013# 本文で撤回すべき記述

| 箇所 | 現記述 | 訂正 |
|------|--------|------|
| §3.2 結論 | 「`ztb/live_trading/` は完全に死んだコード」 | 「モック状態の初期プロトタイプ。プロジェクト目的の中核部品として維持し、系統②③ との統合計画が必要」 |
| §4.1 結論 | 「`dry_run=False` に切り替えるだけで本番稼働可能」 | 「実装骨格は完了。ただし C-3 署名不一致・C-7 order_type マッピング・D-1 OrderManager 未接続の解決が本番前に必須」 |
| §5.2 判定 | 「✅ 安全に廃止可能」「~770 行削減」 | 撤回。依存除去 → 代替導線整備 → 統合検討の段階移行が必要 |
| §6.1 #1 | 「`ztb/live_trading/` を `archived/` へ移動 — 10min」 | 削除。即時 archive は不可 |
| §6.4 #1 | 「`market_buy` / `market_sell` / `limit_buy` / `limit_sell`」 | 「`buy` / `sell` (指値), `market_buy` / `market_sell` (成行) — Coincheck 公式 4 値」 |
| §2.1 | 2 系統のみ記載 | 3 系統 (D.4 参照) に拡張。特に系統③ `ztb/trading/live_trader/` を追加 |
| Appendix A | 「死コード」セクションに live_trading 掲載 | 「初期プロトタイプ (要維持)」に変更。系統③ のファイル一覧を追加 |

## Appendix E: 改訂履歴 (更新)

| 日付 | 変更 | 理由 |
|------|------|------|
| 2026-02-13 | 初版作成 | ユーザー要請: 既存 API 基盤調査 + v460 重複排除分析 |
| 2026-02-13 | Appendix C 追加 (第2巡レビュー) | 外部レビューによる 9 指摘 |
| 2026-02-14 | Appendix D 追加 (C-finding 検証 + 追加発見) | C-1～C-9 全件をソースコード・公式 API docs で照合。追加発見 D-1～D-5 を記録。000# 整合性評価。§3.2/§4.1/§5.2/§6.1/§6.4 の訂正箇所を特定 |
| 2026-02-14 | **D.6 全修正を実装 + 本文訂正** | C-3 署名修正、C-4 async統一、C-5 bitFlyer正規化、C-7 order_type 4値、C-9 docstring、D-1 OrderManager接続、D-3 post_only、D-5 RateLimiter 4req/s を全実装。§2.1/§3.2/§4.1/§5.2/§6.1/§6.4/AppA を訂正。97 tests all pass |
