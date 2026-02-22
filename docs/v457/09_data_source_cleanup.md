# v457 データ整理案（BTC/JPY 実データの混乱解消）

## 実施済み
- `data/btc_jpy_real_dataset.csv` を1分足OHLCVとして再作成（`btc_jpy_extended_dataset.csv` + `btc_jpy_1m_v451.csv` を結合）
- 旧 `data/btc_jpy_real_dataset.csv`（1時間足）を `data/archives/btc_jpy/btc_jpy_real_dataset_1h_legacy.csv` に移動
- 取引ログの誤命名データを `data/trade_logs/` に分離  
  - `data/trade_logs/btc_jpy_trade_log_dataset.csv`  
  - `data/trade_logs/btc_jpy_yahoo_trade_log_dataset.csv`

## 1. 1分足・生OHLCVの有力候補（主要データ候補）
※ いずれも `timestamp, open, high, low, close, volume` を満たす
- `data/btc_jpy_1m_v451.csv`（約141k行 / 1m）
- `data/btc_jpy_training_data.csv`（約141k行 / 1m）
- `data/btc_jpy_backtest_data.csv`（約90k行 / 1m）
- `data/btc_jpy_1m_dataset.csv`（約24k行 / 1m）
- `data/yahoo_finance/btc_jpy_1m.csv`（約7k行 / 1m）
- `data/btc_jpy_1m_latest.csv`（約6k行 / 1m）

## 2. “主要にはなり得ない”ため Archive 候補
### 2.1 非1分足（時間足違い）
- `data/btc_jpy_15m_dataset.csv`
- `data/btc_jpy_5m_dataset.csv`
- `data/btc_jpy_5m_yfinance.csv`
- `data/yahoo_finance/btc_jpy_5m.csv`
- `data/yahoo_finance/btc_jpy_5m_converted.csv`
- `data/yahoo_finance/btc_jpy_15m.csv`
- `data/yahoo_finance/btc_jpy_15m_converted.csv`
- `data/yahoo_finance/btc_jpy_30m.csv`
- `data/yahoo_finance/btc_jpy_30m_converted.csv`
- `data/yahoo_finance/btc_jpy_1h_converted.csv`
- `data/yahoo_finance/btc_jpy_1h_recent.csv`
- `data/btc_jpy_v434_extended.csv`（日足）
- `data/btc_jpy_yahoo_real_20251021*.csv`（日足一式）

### 2.2 超短尺 / 破損 / 周期不整合
- `data/btc_jpy_15m_from_test_minute.csv`（1行）
- `data/btc_jpy_5m_from_test_minute.csv`（2行）
- `data/trade_logs/btc_jpy_trade_log_dataset.csv`（100行）
- `data/btc_jpy_1m_latest_7d_20251213_155436.csv`（推定2分間隔）

### 2.3 特徴量加工済みで生OHLCVではない
- `data/btc_jpy_balanced_v426_dataset.csv`
- `data/btc_jpy_correlation_aware_v426_dataset.csv`
- `data/btc_jpy_diverse_dataset.csv`
- `data/btc_jpy_yahoo_real_20251021_featured_scaled.csv`
- `data/btc_jpy_yahoo_real_20251021_featured_selected.csv`
- `data/btc_jpy_yahoo_real_20251021_featured_corrected.csv`（timestamp無し）

## 3. “紛らわしい名称”の解消案
- `data/btc_jpy_real_dataset.csv` は **実際は1時間足**。  
  旧設定・旧スクリプトで広く参照されているため、  
  **「1分足の実データをこの名称に統一する」**のが混乱解消には最短です。

### 対応済み
- `data/btc_jpy_real_dataset.csv` は **1分足の実データ** に統一済み  
- 1時間足は `data/archives/btc_jpy/btc_jpy_real_dataset_1h_legacy.csv` に退避済み

## 4. データ取得スクリプト整理（Yahoo / CoinCheck / BitFlyer）

### 4.1 “正規ルート”（推奨）
- `scripts/v456/update_data_comprehensive.py`  
  優先順: YahooFinance → BitFlyer → CoinCheck
- `scripts/v456/update_data_simple.py`（Yahoo優先）
- `scripts/v456/update_data_bitflyer.py`（補完）
- `scripts/v456/update_data_coincheck.py`（最終手段）

### 4.2 “補助・簡易”
- `scripts/v456/update_data_simple.py`（yfinance簡易取得）
- `scripts/download_yfinance_data.py`（単機能）
- `scripts/convert_yahoo_data.py`（フォーマット統一用）

### 4.3 “テスト・検証用途”
- `scripts/tests/test_yahoo_finance_data.py`
- `scripts/analysis/download_extended_yahoo_data.py`
- `scripts/utils/fetch_yahoo_finance_btc.py`

## 5. 実行の整理（案）
- **Yahoo取得は1本化**  
  `scripts/v456/update_data_simple.py` を残し、他は “legacy” に寄せる。
- **BitFlyer/CoinCheckは comprehensive を起点**  
  実運用は `update_data_comprehensive.py` から実施。

## 6. 次のアクション（意思決定が必要）
1. Archive対象ファイルの **移動実行**  
2. Yahoo/BitFlyer取得スクリプトの **“正規ルート”明記**（README更新）
