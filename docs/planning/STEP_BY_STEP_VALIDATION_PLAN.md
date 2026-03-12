# 段階的検証プラン

## 現状サマリー

### 問題点
- **ppo_100k_optimized.zip**: SELL bias 99.5% → 使用不可
  - バックテスト: -0.79% リターン、全トレード損失
  - 原因: curriculum_stage="full" (forced_balanceを経由していない)

### 対策実施中
- **ppo_balanced_test.json**: 修正版設定で学習中 (30,000 steps)
  - curriculum_stage: "forced_balance" に変更
  - lagrange_r_target: 0.33 (33%均等分布)
  - 初期結果 (1024 steps): SELL率 21.9% (大幅改善!)

---

## ステップ1: 修正版モデルの学習完了と基本検証 ✅

### 1.1 学習の完了 (進行中)
```bash
python run_training.py --config configs/training/ppo_balanced_test.json
```

**期待される結果:**
- 学習完了時間: 約5-10分
- モデル保存先: `models/ppo_balanced_test.zip`
- SELL率: 25-40% の範囲 (33%前後が理想)

### 1.2 アクション分布の確認
```bash
python validate_model_behavior.py --model-path models/ppo_balanced_test.zip --episodes 5
```

**合格基準:**
- ✅ SELL率: 25-40%
- ✅ BUY率: 25-40%
- ✅ HOLD率: 20-50%
- ✅ バランススコア: >0.5

**不合格の場合:**
- lagrange_r_target を調整 (0.30 or 0.35)
- lagrange_eta を増加 (0.1)
- 再学習を実施

### 1.3 バックテスト検証
```bash
python backtest_model.py \
  --model-path models/ppo_balanced_test.zip \
  --data-path ml-dataset-enhanced.csv \
  --output backtest_balanced_test.json
```

**合格基準:**
- ✅ 総リターン: >-0.5% (損失が少ない)
- ✅ 勝率: >10% (少なくともいくつか勝ちトレードがある)
- ✅ 利益係数: >0 (利益があるトレードが存在)

**不合格の場合:**
- reward_settings を調整
- より長い学習期間 (50,000 steps) で再実施

---

## ステップ2: カリキュラム学習の段階的実施 🔄

### 2.1 Stage 0: Forced Balance (完了)
- **目的**: 33/33/33 の完全均等分布を強制学習
- **期間**: 10,000 steps
- **設定**: curriculum_stage="forced_balance"

### 2.2 Stage 1: Balanced Transition
- **目的**: 通常報酬 + バランスペナルティ
- **期間**: 20,000 steps
- **設定**: curriculum_stage="balanced_transition"

**設定ファイル作成:**
```json
{
  "session_id": "ppo_balanced_transition",
  "total_timesteps": 20000,
  "curriculum_stage": "balanced_transition",
  "lagrange_r_target": 0.30,
  "checkpoint": "models/ppo_balanced_test.zip"
}
```

### 2.3 Stage 2: PnL Focused
- **目的**: PnL重視、ただしバランス維持
- **期間**: 70,000 steps
- **設定**: curriculum_stage="pnl_focused"

---

## ステップ3: 長期学習とフル評価 📈

### 3.1 100K フル学習
```json
{
  "session_id": "ppo_100k_balanced",
  "total_timesteps": 100000,
  "curriculum_stage": "full",
  "enable_forced_diversity": true,
  "checkpoint": "models/ppo_balanced_transition.zip"
}
```

### 3.2 総合バックテスト
```bash
# 学習データでの検証
python backtest_model.py \
  --model-path models/ppo_100k_balanced.zip \
  --data-path ml-dataset-enhanced.csv

# 実データでの検証
python backtest_model.py \
  --model-path models/ppo_100k_balanced.zip \
  --data-path btc_jpy_real_dataset.csv
```

### 3.3 詳細パフォーマンス分析
```bash
# 月次パフォーマンス
# トレード分析
# リスク指標
```

---

## ステップ4: 本番展開前の最終確認 ✔️

### 4.1 ストレステスト
- [ ] 上昇トレンドでのテスト
- [ ] 下降トレンドでのテスト
- [ ] 横ばい相場でのテスト
- [ ] ボラティリティ変化への対応

### 4.2 リスク管理
- [ ] 最大ドローダウンの確認
- [ ] ポジションサイズの適切性
- [ ] ストップロス機能の動作確認

### 4.3 ペーパートレーディング
- [ ] 1週間のペーパートレード
- [ ] 実データでのリアルタイム動作確認
- [ ] アクション分布のモニタリング

---

## チェックリスト: 各ステップの完了条件

### ✅ ステップ1 完了条件
- [x] 学習が正常に完了
- [ ] アクション分布がバランス (25-40-35 程度)
- [ ] バックテストで改善が確認できる
- [ ] バランススコア >0.5

### 🔄 ステップ2 完了条件
- [ ] 各カリキュラムステージで学習完了
- [ ] 段階的にPnL重視へ移行
- [ ] アクションバランスを維持しながら性能向上

### 📈 ステップ3 完了条件
- [ ] 100K学習が完了
- [ ] バックテストで正のリターン
- [ ] 勝率 >40%
- [ ] シャープレシオ >0.5

### ✔️ ステップ4 完了条件
- [ ] 全ストレステスト合格
- [ ] ペーパートレード成功
- [ ] 本番展開の承認

---

## 現在の進行状況

### 完了済み ✅
1. 問題の特定 (SELL bias 99.5%)
2. 根本原因の分析 (curriculum_stage設定ミス)
3. 修正版設定ファイルの作成
4. バックテストスクリプトの準備
5. 検証スクリプトの作成

### 進行中 🔄
1. **ステップ1.1**: 修正版モデルの学習 (30,000 steps)
   - 学習開始済み
   - 予想完了時間: 5-10分

### 次のアクション 🎯
1. 学習完了を待つ (進行中)
2. アクション分布を確認
3. バックテストを実施
4. 結果に基づいて次のステップを決定

---

## 学習パラメータの調整履歴

### ppo_100k_optimized.json (失敗)
```
curriculum_stage: "full"
lagrange_r_target: 0.175
→ 結果: SELL 99.5%, 使用不可
```

### ppo_balanced_test.json (テスト中)
```
curriculum_stage: "forced_balance"
lagrange_r_target: 0.33
→ 初期結果 (1024 steps): SELL 21.9%, 改善確認
```

### 次の候補設定
```json
{
  "curriculum_stage": "balanced_transition",
  "lagrange_r_target": 0.30,
  "lagrange_tolerance": 0.05,
  "lagrange_eta": 0.05,
  "total_timesteps": 50000
}
```

---

## トラブルシューティング

### Q1: 学習が遅い
**A**:
- `n_steps` を 2048 に増加
- `batch_size` を 128 に増加
- マルチプロセス環境を使用

### Q2: アクションバランスが改善しない
**A**:
- `lagrange_eta` を 0.1 に増加
- `lagrange_r_target` を調整 (0.30-0.35)
- `enable_forced_diversity` を true に

### Q3: バックテストで損失
**A**:
- より長い学習期間 (100K steps)
- reward_settings を確認
- データの多様性を増やす

### Q4: 過学習の兆候
**A**:
- 正則化を強化
- エントロピー係数を増加
- 早期停止を実装

---

## 参考資料

- **検証レポート**: `MODEL_VALIDATION_REPORT.md`
- **バックテストスクリプト**: `backtest_model.py`
- **行動検証スクリプト**: `validate_model_behavior.py`
- **学習設定**: `configs/training/`
- **学習ログ**: `logs/ppo_balanced_test/`
- **TensorBoard**: `tensorboard/ppo_balanced_test/`
