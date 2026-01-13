# BTC/JPY データ更新スクリプト群

複数のデータソースから BTC/JPY 1分足データを自動更新するスクリプト集です。

## 概要

| スクリプト | ソース | 特徴 | 適用範囲 |
|-----------|--------|------|---------|
| `update_data_comprehensive.py` | 複数（優先度順） | **推奨**。複数ソースを自動試行 | 本番用 |
| `update_data_coincheck.py` | CoinCheck | 日本の取引所、JPY ペア対応 | 推奨第1選択肢 |
| `update_data_bitflyer.py` | BitFlyer | 日本の大手取引所 | 予備選択肢（制限あり） |
| `update_data_yahoo.py`（既存） | YahooFinance | グローバル、最後の手段 | 直近7日のみ |

---

## 使用方法

### 1. 推奨: マルチソース自動更新

```bash
# 自動的に CoinCheck > BitFlyer > YahooFinance の順で試行
python scripts/v456/update_data_comprehensive.py

# 特定のソースのみ試行
python scripts/v456/update_data_comprehensive.py --source coincheck

# 30日分取得（デフォルト）
python scripts/v456/update_data_comprehensive.py --days 30
```

### 2. CoinCheck（推奨）

```bash
python scripts/v456/update_data_coincheck.py [--days 30] [--output-file data/btc_jpy_1m_v456.csv]
```

**メリット:**
- 日本の取引所で JPY ペアに特化
- 約定履歴から正確な OHLC を再構成可能
- API 制限が比較的緩い

**デメリット:**
- 完全に無料ではない場合がある
- メンテナンス時間がある

### 3. BitFlyer（予備）

```bash
python scripts/v456/update_data_bitflyer.py [--days 30]
```

**メリット:**
- 日本の大手取引所
- 安定性が高い

**デメリット:**
- REST API に直接 OHLC エンドポイントがない
- WebSocket API の使用が推奨される

### 4. YahooFinance（最終手段）

```bash
python scripts/v456/update_data_yahoo.py
```

**メリット:**
- セットアップが簡単
- グローバルデータ

**デメリット:**
- 直近 7日分のみ
- 精度がやや劣る

---

## 出力ファイル

スクリプトは自動的に以下のファイルを検出・更新します（優先順）:

1. `data/btc_jpy_1m_v456.csv` (最新)
2. `data/btc_jpy_1m_v455.csv`
3. `data/btc_jpy_1m_v454.csv`

### ファイル形式

```csv
timestamp,open,high,low,close,volume
2024-01-13 12:00:00+00:00,8500000,8510000,8490000,8505000,1.25
2024-01-13 12:01:00+00:00,8505000,8515000,8500000,8512000,1.30
...
```

---

## トラブルシューティング

### CoinCheck がエラーを返す

```
[CoinCheck] ✗ Error: API connection failed
```

**原因:**
- API ダウンタイム
- ネットワーク接続問題
- Rate limit 到達

**対策:**
- 数分待ってから再試行
- BitFlyer に自動フォールバック（comprehensive モード）
- ネットワーク接続を確認

### BitFlyer から OHLC が取得できない

```
[BitFlyer] Note: BitFlyer REST API has limitations for historical OHLC data
```

**理由:**
- BitFlyer REST API に直接 OHLC エンドポイントがない
- ティッカーのみ利用可能（最新価格）

**推奨:**
- CoinCheck を使用（より完全な約定データ）
- WebSocket API への移行を検討

### YahooFinance が直近 7日のみを返す

```
[YahooFinance] No new data after last timestamp
```

**理由:**
- YahooFinance の 1分足データは過去 7日のみ保持
- Crypto の提供範囲制限

**推奨:**
- 定期的に実行（毎日など）
- CoinCheck を主要ソースとして使用

---

## 仕組み

### マージロジック

```
新規データ取得
    ↓
既存ファイルの最後のタイムスタンプを確認
    ↓
新規データをフィルタ（最後より後）
    ↓
OHLC 妥当性検証 (high >= low など)
    ↓
既存データと新規データをマージ
    ↓
重複排除（新規データが優先）
    ↓
タイムスタンプでソート
    ↓
CSV に保存
```

### Rate Limiting

- CoinCheck: 0.5秒/呼び出し（デフォルト）
- BitFlyer: 0.5秒/呼び出し（デフォルト）
- YahooFinance: Backoff 付き（接続エラー時）

---

## 本番運用での推奨設定

### 定期実行（Cron / Windows Task Scheduler）

**毎日 00:10 に実行:**

```bash
# Linux / macOS
10 0 * * * cd /path/to/zaif-trade-bot && python scripts/v456/update_data_comprehensive.py

# Windows (PowerShell)
# Schedule task to run: powershell -c "cd C:\path\to\zaif-trade-bot; python scripts\v456\update_data_comprehensive.py"
```

### ログ記録

```bash
python scripts/v456/update_data_comprehensive.py >> logs/data_update.log 2>&1
```

### エラーハンドリング

スクリプトは自動的に以下を行います:
- 複数ソースの自動フォールバック
- 接続エラー時の Exponential Backoff
- OHLC 妥当性検証（警告）

---

## API 仕様

### CoinCheck API

**エンドポイント:**
- `GET /api/trades?pair=btc_jpy&limit=100&page=N`

**レート制限:**
- 推定: 300 req/hour

**制限事項:**
- ページネーション方式（古いデータへアクセスする場合、多数ページ必要）

### BitFlyer API

**エンドポイント:**
- `GET /v1/ticker?product_code=BTC_JPY` (ティッカーのみ)
- `/v1/me/getexecutions` (約定履歴、認証必須)

**レート制限:**
- パブリック: 不明
- プライベート: 200 req/minute

**制限事項:**
- REST API に OHLC がない
- WebSocket API 推奨 (`lightning_executions`)

### YahooFinance

**エンドポイント:**
- `GET /v10/finance/chart/BTC-JPY?interval=1m&range=7d`

**レート制限:**
- 緩い（非公式API）

**制限事項:**
- 1分足は直近 7日のみ
- 信頼性は BitFlyer/CoinCheck より低い

---

## 今後の拡張

### 1. WebSocket 対応

```python
# update_data_bitflyer_websocket.py
# BitFlyer WebSocket API で継続的にストリーミング
# リアルタイム約定データから 1分足を合成
```

### 2. 複数ペア対応

```bash
python scripts/v456/update_data_comprehensive.py --pair btc_jpy --pair eth_jpy
```

### 3. データ品質チェック

```bash
python scripts/v456/validate_data_quality.py --check-gaps --check-ohlc
```

### 4. バックフィル機能

```bash
python scripts/v456/backfill_data.py --start-date 2023-01-01 --source coincheck
```

---

## 参考リンク

- [CoinCheck API ドキュメント](https://coincheck.com/ja/documents/exchange/api)
- [BitFlyer API ドキュメント](https://lightning.bitflyer.jp/docs)
- [BitFlyer WebSocket API](https://bf-lightning-api.docs.apiary.io/)
- [YahooFinance yfinance](https://pypi.org/project/yfinance/)

---

## ライセンスと免責

このスクリプトは教育目的で提供されています。
API 利用規約を遵守した上で使用してください。

データ精度について:
- CoinCheck: 高精度（約定履歴から再構成）
- BitFlyer: 中精度（制限あり）
- YahooFinance: 低精度（直近 7日のみ）

実際の取引には、充分なテスト後に使用してください。
