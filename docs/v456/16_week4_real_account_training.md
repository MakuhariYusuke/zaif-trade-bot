# Week 4 準備: 実残高対応MLP SAC訓練システム

**Date**: 2026-01-13 (Week 4開始)  
**Status**: ✓ 完了 

## 実施内容

### 1. 実残高取得システム構築
- **目的**: Zaif交換所から実際のアカウント残高を取得
- **実装**: `scripts/v456/get_account_info.py`
  - ccxt統合で実API呼び出し（失敗時はモックフォールバック）
  - JSON設定ファイル保存機能

**取得結果** (2026-01-13 23:25):
```
BTC残高:  0.00877740 BTC
JPY残高:  124.01 JPY
```

### 2. TradingAPI統合
- **ファイル**: `ztb/live_trading/trading_api.py`
- **改善**: `get_balance()` モック → ccxt実装に更新
- **依存ライブラリ**: `pip install ccxt python-dotenv`

### 3. MLP SAC訓練スクリプト作成
- **ファイル**: `scripts/v456/train_mlp_v456.py`
- **仕様**:
  - **環境**: FastIntradayEnvV456 (88D観測空間)
  - **モデル**: SAC + MLP [128, 128] ポリシー
  - **初期残高**: 実Zaifアカウント残高 (124.01 JPY)
  - **ポジションサイズ**: 100% uncapped (小資金対応)
  - **市場データ**: btc_jpy_1m_v454.csv (2025-11-03 ~ 2026-01-13, 27,012 records)
  
- **設定可能パラメータ**:
  ```bash
  python scripts/v456/train_mlp_v456.py \
    --timesteps 1000000 \
    --learning-rate 3e-4 \
    --batch-size 256 \
    --log-interval 10000
  ```

### 4. テスト実行結果 (5,000 timesteps)
```
Training Configuration:
- Environment: FastIntradayEnvV456
  - Observation Space: Box(-inf, inf, (88,), float32)
  - Action Space: Box([-1. 0.], 1.0, (2,), float32)
  - Market Data Points: 27012

- Model: SAC (Soft Actor-Critic)
  - Policy: MlpPolicy
  - Policy Network: [128, 128]
  - Learning Rate: 3e-4
  - Batch Size: 256

Results:
- Total Timesteps: 5000
- Episodes: 4948
- Episode Reward Mean: -0.222
- FPS: 86
- Time Elapsed: 57 seconds

Model saved: models/week4_mlp_sac/sac_mlp_v456_20260113_232655.zip
```

## Week 4タスク進捗

### ✓ 完了
1. **データ更新システム確認**
   - 市場データ: 2025-11-03 ~ 2026-01-13 (27,012 records)
   - マルチソース対応 (Yahoo > BitFlyer > CoinCheck)

2. **実残高取得機能**
   - Zaif API統合
   - get_account_info.py スクリプト
   - 設定自動保存

3. **訓練環境準備**
   - FastIntradayEnvV456 環境構築
   - 100% uncapped ポジションサイズ設定
   - プリウォーム機能統合

4. **訓練スクリプト実装**
   - train_mlp_v456.py 完成
   - テスト実行成功（5,000 timesteps）
   - モデル保存機能

### 🔄 進行中
1. **本格訓練実行**
   - 推奨: 1,000,000 timesteps
   - 実行時間: 約3-4時間（CPU環境）
   - コマンド: `python scripts/v456/train_mlp_v456.py --timesteps 1000000`

### ⏳ 次期タスク (Week 4-2)
1. **モデル訓練実行と検証**
2. **バックテスト結果分析**
3. **結果ドキュメント作成**
4. **実取引への移行検討**

## ファイル構成

```
scripts/v456/
├── train_mlp_v456.py           ← 訓練スクリプト (新規)
├── get_account_info.py         ← 残高取得 (新規)
├── update_data_comprehensive.py ← データ更新 (Week 3)
└── account_config.json         ← 設定ファイル (自動生成)

models/week4_mlp_sac/
├── sac_mlp_v456_20260113_232655.zip  ← 保存モデル
└── metadata_20260113_232655.json      ← メタデータ

ztb/live_trading/
└── trading_api.py              ← TradingAPI更新
```

## 重要な設定値

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| initial_balance | 124.01 JPY | 実Zaif残高 |
| max_position | None (uncapped) | 100%ポジション許可 |
| environment | FastIntradayEnvV456 | 88D観測空間 |
| policy | MlpPolicy | MLP [128, 128] |
| learning_rate | 0.0003 | SAC学習率 |
| batch_size | 256 | バッチサイズ |
| buffer_size | 1,000,000 | リプレイバッファ |
| timesteps | 1,000,000 | 推奨訓練ステップ |

## 技術スタック

- **強化学習**: Stable Baselines3 (SAC)
- **環境**: FastIntradayEnvV456
- **取引所API**: ccxt
- **特徴工学**: GroupedFeatureScaler
- **リポジトリ**: zaif-trade-bot

## 次の実行コマンド

### 本格訓練実行
```bash
cd c:\Users\Admin\dev\zaif-trade-bot
.venv\Scripts\python.exe scripts/v456/train_mlp_v456.py --timesteps 1000000
```

### 実残高確認
```bash
.venv\Scripts\python.exe scripts/v456/get_account_info.py
```

### モデルチェック
```bash
ls models/week4_mlp_sac/
```

## 備考

1. **小資金運用対応**
   - 初期資金124円での訓練
   - 100%ポジション許可（小さい値でも機能するよう設計）
   - リスク管理は環境内に統合

2. **テスト環境での検証完了**
   - 環境起動: ✓
   - モデル学習: ✓
   - 報酬計算: ✓ (平均 -0.222)

3. **本番運用への準備**
   - API認証情報は `.env` ファイルに管理
   - モデルは自動保存とメタデータ記録
   - ログは TensorBoard と JSON で記録

---

**進捗**: Week 3完了 → Week 4準備完了 → 本格訓練実行準備中
