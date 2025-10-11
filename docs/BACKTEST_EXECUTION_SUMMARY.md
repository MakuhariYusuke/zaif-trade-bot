# バックテスト実行結果サマリー

## 実行日時
2025年10月10日

## 目的
v381（110特徴量）とv384（68厳選特徴量）を過去のBTC/JPY市場データで検証

## 実行状況

### v384モデル（68特徴量）
- **モデルパス**: `models/ppo_reward_v384_curated_60.zip`
- **特徴量数**: 68（厳選版）
- **訓練ステップ数**: 50,000
- **環境との互換性**: ✅ 完全互換（両方とも68特徴量）

### v381モデル（110特徴量）
- **モデルパス**: `models/ppo_reward_v381_revised_profit_focused.zip`
- **特徴量数**: 110（全特徴量）
- **訓練ステップ数**: 不明（設定ファイルに記載なし）
- **環境との互換性**: ❌ 次元不一致

## 発見された問題

### 次元不一致エラー
```
ValueError: Error: Unexpected observation shape (68,) for Box environment, 
please use (110,) or (n_env, 110) for the observation shape.
```

**原因**:
1. 環境（HeavyTradingEnv）は現在の`models/features_schema.json`を使用
2. このschemaはv384訓練時に保存された68特徴量版
3. v381モデルは110特徴量を期待
4. → 環境が生成する観測（68次元）とモデルの期待（110次元）が不一致

### v384バックテスト結果（初期実行）

**10エピソード実行**:
- 全エピソード完了（エラーなし）
- 総アクション数: 990
- アクション分布:
  - HOLD: 990 (100%)
  - BUY: 0 (0%)
  - SELL: 0 (0%)

**分析**:
- モデルは環境と完全互換✅
- ただし、全アクションがHOLDのみ
- これは以下のいずれかを示唆:
  1. 訓練期間が短すぎた（50kステップ）
  2. 報酬関数が保守的すぎる
  3. 初期状態が取引に不適
  4. エピソード長が短すぎる（99ステップ/エピソード）

## 解決策の方向性

### オプション1: v381用の環境を再構築
特徴量フィルタリングを無効化してv381をテスト:
```python
config = {
    "enable_feature_filtering": False,  # 全110特徴量を使用
    # ... other settings
}
```

**課題**: 
- v381訓練時のfeature schemaが保存されていない
- 110特徴量の順序が不明

### オプション2: v384の長期訓練
68特徴量でより長い訓練を実行:
```bash
python run_training.py --config configs/training/ppo_reward_v384_curated_60_extended.json --timesteps 200000
```

**メリット**:
- 現在の環境と完全互換
- より実戦的な動作を学習可能

### オプション3: 紙上取引（Paper Trading）
実時間でのシミュレーション:
```bash
python live_trade.py --model models/ppo_reward_v384_curated_60.zip --paper-trade
```

**メリット**:
- リアルタイム市場データを使用
- バックテストの次元問題を回避

## 推奨アクション

### 即時実行可能
1. ✅ **v384の延長訓練**: 200k-500kステップで再訓練
2. ✅ **紙上取引テスト**: 実時間シミュレーションで検証
3. ⏳ **TensorBoard分析**: v381とv384の訓練曲線比較（既に起動済み）

### 中期的対応
1. ⏳ **特徴量スキーマ管理の改善**:
   - モデルごとにfeature schemaを保存
   - 推論時に自動で適切なschemaを選択

2. ⏳ **バックテスト環境の改善**:
   - モデルの期待次元を自動検出
   - 動的に環境の観測空間を調整

## 技術的詳細

### 現在の環境設定
```python
config = {
    "reward_scaling": 0.01,
    "transaction_cost": 0.00505,
    "max_position_size": 1.05,
    "risk_free_rate": 0.05,
}
```

### データセット
- **優先**: `ml-dataset-enhanced.csv`
- **代替1**: `btc_jpy_real_dataset.csv`
- **代替2**: `btc_jpy_yahoo_real_dataset.csv`

### 実行ログ
```
Loaded X,XXX rows from ml-dataset-enhanced.csv
Data columns: 73
Environment observation space: 68 features
```

## 結論

**v384モデル（68特徴量）**:
- ✅ 訓練成功
- ✅ 環境互換性確認
- ⚠️ 実戦性不明（HOLDのみ）
- 📊 TensorBoard分析待ち

**v381モデル（110特徴量）**:
- ✅ 訓練成功（履歴あり）
- ❌ バックテスト不可（次元不一致）
- 💡 紙上取引で検証可能

**次のステップ**:
1. TensorBoardでv381/v384の訓練メトリクスを比較
2. v384を200k-500kステップで再訓練
3. 紙上取引で両モデルを実時間テスト

---

**作成日**: 2025-10-10  
**ステータス**: バックテスト実行中（v384）
