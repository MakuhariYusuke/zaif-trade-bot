# v381 vs v384 検証状況レポート

## 実行日時
2025年10月10日

## 完了した作業

### 1. TensorBoardログの修正 ✅
- **問題**: `run_training.py`で「TensorBoard logs: None」エラー
- **修正**: `checkpoint_dir`を正しく使用するように修正
- **結果**: TensorBoardが正常に起動 (http://localhost:6006)

### 2. v384訓練の完了 ✅
- **モデル**: `ppo_reward_v384_curated_60.zip`
- **特徴量**: 68特徴量（厳選版）
- **訓練時間**: 約2分（50,000ステップ）
- **最終報酬**: -674
- **アクション分布**: HOLD 48%, BUY 29%, SELL 23%
- **Early stopping**: target_kl=0.07で安定動作

### 3. 特徴量フィルタリングの確認 ✅
- **削減率**: 110特徴量 → 68特徴量（38%削減）
- **削除された特徴**:
  - HeikinAshi OHLC (4)
  - Time constants (5)
  - Ichimoku個別スパン (5)
  - 高相関ペア (20)
  - 訓練ラベル (2)
  - その他冗長指標 (6)

### 4. ドキュメント作成 ✅
- `docs/V381_VS_V384_FEATURE_COMPARISON.md`
- `docs/BACKTEST_EXECUTION_SUMMARY.md`
- 包括的な比較分析

## 直面した課題

### バックテストの次元不一致問題 ⚠️

**問題**:
```
ValueError: Unexpected observation shape (68,) for Box environment, 
please use (110,) or (n_env, 110) for the observation shape.
```

**原因**:
1. v381モデルは110特徴量で訓練
2. 現在の環境は`models/features_schema.json`を使用（v384が保存した68特徴量版）
3. v381の期待次元（110）と環境の出力次元（68）が不一致

**影響**:
- v381のバックテストが実行不可
- v384のバックテストは成功（環境と互換）

### v384バックテスト結果の解釈 🤔

**観測結果**:
- 10エピソード全てで990 HOLD、0 BUY、0 SELL
- つまり、全アクションがHOLD（取引なし）

**可能な原因**:
1. 訓練期間が短すぎた（50kステップ）→ 取引戦略が未成熟
2. エピソード長が短い（99ステップ/エピソード）→ 取引機会が少ない
3. 初期状態が取引に不適（市場条件）
4. 報酬関数が過度に保守的

## 現在実行中のタスク

### 1. TensorBoard分析 🔄
- **URL**: http://localhost:6006
- **比較対象**: v381 vs v384
- **指標**:
  - rollout/ep_rew_mean: 平均エピソード報酬
  - train/approx_kl: KL divergence
  - train/loss: 総損失
  - pan_action_counts: アクション分布

### 2. 紙上取引テスト 🔄
- **コマンド**: `python live_trade.py --model-path models/ppo_reward_v384_curated_60.zip --duration-hours 0.05 --dry-run`
- **目的**: 実時間シミュレーションでv384の動作を検証
- **実行時間**: 3分間（0.05時間）
- **ステータス**: 起動中

### 3. バックテストスクリプト 🔄
- `backtest_v384_simple.py`: v384専用の簡易バックテスト
- **ステータス**: データ読み込み中（時間がかかっている）

## 推奨される次のステップ

### 即座に実行可能

#### オプションA: v384の延長訓練 ⭐ 推奨
```bash
# 200kステップで再訓練（50kの4倍）
python run_training.py --config configs/training/ppo_reward_v384_curated_60.json --timesteps 200000
```

**メリット**:
- より実戦的な戦略を学習
- 68特徴量版で安定した結果
- 環境との完全互換

#### オプションB: TensorBoard分析
```bash
# すでに起動済み: http://localhost:6006
# ブラウザで以下を確認:
# 1. v381とv384の学習曲線比較
# 2. アクション分布の推移
# 3. 報酬の収束状況
```

#### オプションC: 紙上取引の結果確認
```bash
# 現在実行中のタスクの結果を待つ
# 3分後にログを確認
```

### 中期的対応

#### 1. 特徴量スキーマ管理の改善
```python
# モデルごとにfeature schemaを保存
# 推論時に自動で適切なschemaを選択
model_metadata = {
    "model_name": "v384",
    "num_features": 68,
    "feature_schema": "curated_features.py::CURATED_FEATURES",
    "schema_hash": "f7be18533fa61876...",
}
```

#### 2. バックテスト環境の柔軟化
```python
# モデルの期待次元を自動検出
# 動的に環境の観測空間を調整
env = AdaptiveBacktestEnv(
    model_path=model_path,
    auto_detect_features=True
)
```

## 技術的サマリー

### v384モデル（68特徴量）
| 項目 | 値 |
|------|------|
| 訓練ステップ | 50,000 |
| 訓練時間 | ~2分 |
| 最終報酬 | -674 |
| 特徴量数 | 68（厳選版） |
| 環境互換性 | ✅ 完全互換 |
| バックテスト | ✅ 実行可能（要改善） |
| TensorBoard | ✅ ログあり |

### v381モデル（110特徴量）
| 項目 | 値 |
|------|------|
| 訓練ステップ | 不明（設定に記載なし） |
| 訓練時間 | 不明 |
| 最終報酬 | 不明 |
| 特徴量数 | 110（全特徴量） |
| 環境互換性 | ❌ 次元不一致 |
| バックテスト | ❌ 実行不可 |
| TensorBoard | ✅ ログあり |

## 結論

### 成果
1. ✅ v384訓練成功（68特徴量）
2. ✅ TensorBoard起動成功（比較可能）
3. ✅ 特徴量削減達成（38%削減）
4. ✅ 包括的なドキュメント作成

### 課題
1. ⚠️ v381バックテスト不可（次元不一致）
2. ⚠️ v384の取引活動が不足（HOLDのみ）
3. 🔄 紙上取引テスト実行中

### 推奨アクション
**即座**:
1. TensorBoardでv381/v384の訓練メトリクスを比較
2. 紙上取引の結果を確認（実行中）

**次の訓練**:
3. v384を200k-500kステップで再訓練
4. より積極的な取引を促す報酬関数の調整を検討

---

**作成日**: 2025-10-10  
**ステータス**: 検証作業継続中  
**TensorBoard**: http://localhost:6006 （起動中）  
**紙上取引**: 実行中（3分間）
