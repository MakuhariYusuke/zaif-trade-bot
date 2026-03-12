# Week 3 タスク 3-2: データ更新システム構築

**Status**: ✅ COMPLETED  
**Date**: 2026-01-13  
**Task**: BitFlyer/CoinCheck API による1分足データ更新システム

---

## 概要

検証フェーズに先立ち、BTC/JPY 1分足データを最新に保つための自動更新システムを構築しました。

### 実装内容

1. **update_data_comprehensive.py** （推奨）
   - 複数ソースを優先順位で自動試行
   - YahooFinance > BitFlyer > CoinCheck の順で実行
   - エラー時の自動フォールバック

2. **update_data_coincheck.py** (最終手段)
   - CoinCheck REST API から約定データを取得
   - 約定履歴から正確な1分足OHLC を再構成
   - ページネーション対応（複数日取得可能）

3. **update_data_bitflyer.py** (補完)
   - BitFlyer REST API でティッカー取得
   - 注記: OHLC エンドポイントなし（制限あり）
   - WebSocket API への移行を推奨

4. **check_data_sources.py** (環境チェック)
   - API アクセス可能性を検証
   - 既存データファイル確認
   - スクリプト可用性確認

### 動作確認

```
✓ BitFlyer API: 利用可能
✓ YahooFinance: データ取得可能  
✓ 既存データ: btc_jpy_1m_v454.csv (27,011レコード)
✓ すべての更新スクリプト利用可能

実行例: python scripts/v456/update_data_comprehensive.py
Result: ✓ Added 1 new records
```

---

## 使用方法

### クイックスタート

```bash
# 推奨: 複数ソース自動試行
python scripts/v456/update_data_comprehensive.py

# 特定ソースのみ指定
python scripts/v456/update_data_comprehensive.py --source coincheck

# 環境チェック
python scripts/v456/check_data_sources.py
```

### 定期実行設定

**Linux/macOS (Cron):**
```bash
# 毎日 00:10 に実行
10 0 * * * cd /path/to/zaif-trade-bot && python scripts/v456/update_data_comprehensive.py >> logs/data_update.log 2>&1
```

**Windows (Task Scheduler):**
```powershell
# PowerShell で以下を実行
$trigger = New-ScheduledTaskTrigger -Daily -At 0:10am
$action = New-ScheduledTaskAction -Execute "python" -Argument "scripts\v456\update_data_comprehensive.py" -WorkingDirectory "C:\path\to\zaif-trade-bot"
Register-ScheduledTask -TaskName "UpdateBTCData" -Trigger $trigger -Action $action
```

---

## API 制限と対応

### CoinCheck
- **状態**: 最終手段
- **API**: `/api/trades?pair=btc_jpy`
- **精度**: 高（約定データから再構成）
- **制限**: ページネーション必須（古いデータは複数ページ）

### BitFlyer  
- **状態**: ✓ 利用可能（制限あり）
- **API**: `/v1/ticker?product_code=BTC_JPY`
- **精度**: 中（ティッカーのみ）
- **制限**: OHLC エンドポイントなし
- **推奨**: WebSocket API への移行

### YahooFinance
- **状態**: 優先
- **API**: yfinance ライブラリ経由
- **精度**: 低（直近7日のみ）
- **制限**: 非公式API、信頼性低い

---

## 技術実装

### マージロジック

```python
# 1. 既存データを読み込み
df_existing = pd.read_csv("data/btc_jpy_1m_v454.csv")
last_timestamp = df_existing.index.max()

# 2. 新規データを取得
df_new = fetcher.fetch_recent_ohlc(days=30)

# 3. 重複排除（最後のタイムスタンプより後のみ）
df_new_filtered = df_new[df_new.index > last_timestamp]

# 4. マージ
merged = pd.concat([df_existing, df_new_filtered])
merged = merged[~merged.index.duplicated(keep='last')]
merged = merged.sort_index()

# 5. 保存
merged.to_csv("data/btc_jpy_1m_v454.csv")
```

### OHLC 検証

```python
def validate_ohlc_data(df):
    # high >= max(open, close)
    # low <= min(open, close)
    # high >= low
    # 有限値チェック
```

### Rate Limiting

- CoinCheck: 0.5秒/呼び出し
- BitFlyer: 0.5秒/呼び出し  
- Exponential Backoff: 接続エラー時

---

## Week 4への影響

### 検証フェーズ

FastIntradayEnvV456 の学習に使用するデータを最新に維持:

```python
# 検証スタート時
python scripts/v456/update_data_comprehensive.py

# 学習中
# -> 別プロセスで定期更新可能
```

### データフレッシュネス

- **推奨更新頻度**: 日1回（早朝）
- **最大遅延許容度**: 24時間
- **リアルタイム要件**: なし（バックテスト検証用）

---

## トラブルシューティング

### "No new data after last timestamp"

**原因**: 既存データが既に最新  
**対策**: 正常です。24時間後に再実行してください。

### "Max retries exceeded"

**原因**: API サーバー接続エラー  
**対策**:  
1. ネットワーク接続確認
2. API サーバーステータス確認
3. しばらく待って再試行

### "OHLC validation failed"

**原因**: データ品質問題  
**対策**: 警告レベル。データは保存されますが、検証を確認してください。

---

## ファイル構成

```
scripts/v456/
├── update_data_comprehensive.py     (メイン：複数ソース対応)
├── update_data_coincheck.py         (最終手段）
├── update_data_bitflyer.py          (補完)
├── check_data_sources.py            (環境チェック)
└── DATA_UPDATE_README.md            (詳細ドキュメント)

data/
└── btc_jpy_1m_v454.csv            (取得済み：27,011レコード)
```

---

## 次フェーズへの接続

### Week 4: MLP SAC 学習フェーズ

```python
# 学習開始時
python scripts/v456/update_data_comprehensive.py

# 学習中は古いデータで実行
# （新規データは別プロセスで蓄積）

# 学習完了後、最新データでバックテスト
python scripts/v456/update_data_comprehensive.py
python scripts/v456/backtest_mlp.py
```

---

## 推奨実装

本番運用では以下を推奨:

1. **定期実行**: cron/TaskScheduler で毎日実行
2. **ログ記録**: 標準出力をファイルに記録
3. **エラー通知**: メール/Slack 通知設定
4. **バックアップ**: 日次バックアップ（事故対応用）
5. **モニタリング**: データ品質チェック（週1回）

---

## 参考リンク

- [CoinCheck API](https://coincheck.com/ja/documents/exchange/api)
- [BitFlyer API](https://lightning.bitflyer.jp/docs)
- [BitFlyer WebSocket](https://bf-lightning-api.docs.apiary.io/)
- [yfinance Documentation](https://pypi.org/project/yfinance/)

