# 1M学習アンサンブル運用マニュアル

**目的**: CustomPPO横展開完了後、1M学習×アンサンブルで儲かるモデルを探す
**中間目標**: 反復学習に最適な設定を見つけ、学習バッチを回す
**作成日**: 2025年10月7日

---

## クイックスタート（コピペ可）

### 前提条件

✅ CustomPPO横展開完了
✅ PAN/Target Entropy/Reverse-as-Close実装済み
✅ データ準備完了: `ml-dataset-enhanced.csv`
✅ 既存のアンサンブル実装: `ztb/training/ensemble.py`

### Step 1: プレフライト・ゲート

```bash
# スキーマ/スケーラ/フィンガープリント検証（既存スクリプト使用）
python scripts/preflight_schema_scaler_check.py \
  --model-dir models/ppo_session_candidate \
  --strict
```

**期待結果**:
```
✅ Feature schema valid
✅ Scaler statistics valid
✅ Config fingerprint valid
```

**失敗時**: エラーメッセージを確認し、スキーマ/スケーラを修正してから進む。

---

## Step 2: アンサンブル構成（3モデル）

### 多様化軸

CustomPPOの設定を活かしつつ、以下の軸で多様化:

| モデル | ent_coef | reward_multipliers | seed | allow_reverse |
|--------|----------|-------------------|------|---------------|
| **A** | 0.6 | [1.0, 1.0, 0.8] | 101 | False |
| **B** | 0.7 | [1.0, 1.0, 0.9] | 202 | False |
| **C** | 0.8 | [1.0, 1.0, 1.0] | 303 | **True** |

**設計方針**:
- **共通**: CustomPPO (PAN=True, Target Entropy=True)
- **多様化**: ent_coef（探索度）、SELL reward倍率、reverse許可
- **固定**: データ分割、学習率、バッチサイズ、その他PPOパラメータ

---

## Step 3: 1Mロングラン設計（ステージング）

### 段階的学習スケジュール

| ステージ | ステップ範囲 | 設定 | 目的 |
|---------|-------------|------|------|
| **0: Warmup** | 0–50k | weights=1.0, λ=0, 基準LR | 基礎学習 |
| **1: 移行** | 50k–200k | weights導入、λ解放 | バイアス緩和開始 |
| **2: 本編** | 200k–800k | 標準設定 | メイン学習 |
| **3: 収束** | 800k–1M | LRアニーリング | 安定化 |

### チェックポイント

- **頻度**: 25k毎 (計40チェックポイント)
- **保存**: モデル + 統計 + diagnostics
- **評価**: Rolling OOS (300-500 steps)

### 監視指標

**必須**:
- `legal_sell_rate` (移動平均)
- `grad_norm(SELL)`
- `adv_mean(SELL)`
- `entropy` (target達成率)
- `KL violations`

**パフォーマンス**:
- `trades_per_1k`
- `max_drawdown`
- `Sharpe_proxy`

---

## Step 4: 実行コマンド

### アンサンブルA (ent_coef=0.6)

```bash
python -m ztb.training.unified_trainer \
  --config configs/train/ensemble_A_1M.yaml \
  --algorithm ppo \
  --total_timesteps 1000000
```

### アンサンブルB (ent_coef=0.7)

```bash
python -m ztb.training.unified_trainer \
  --config configs/train/ensemble_B_1M.yaml \
  --algorithm ppo \
  --total_timesteps 1000000
```

### アンサンブルC (ent_coef=0.8, reverse=True)

```bash
python -m ztb.training.unified_trainer \
  --config configs/train/ensemble_C_1M.yaml \
  --algorithm ppo \
  --total_timesteps 1000000
```

---

## Step 5: ローリング評価（自動化）

### 25k毎の自動評価

```bash
# チェックポイント発見時に自動実行
python scripts/rolling_eval.py \
  --checkpoint_dir checkpoints/ensemble_A \
  --eval_steps 500 \
  --output artifacts/eval/ensemble_A_rolling.csv
```

### ゲート判定基準

**合格ライン**:
- `Sharpe_proxy > 0`
- `legal_sell_rate >= 0.15`
- `cost_gate_trigger_rate` < 0.5

**停止条件**:
- 2回連続でゲート失敗 → その枝は停止

---

## Step 6: アンサンブル集計

### 最終評価（1M完了後）

```bash
python scripts/ensemble_aggregation.py \
  --models \
    models/ensemble_A/final \
    models/ensemble_B/final \
    models/ensemble_C/final \
  --weights confidence \
  --eval_steps 500 \
  --seeds 3 \
  --output artifacts/ensemble_final_eval.json
```

### 重み付け方式

**confidence-weighted**:
```python
weight_i = recent_sharpe_i × calibration_factor_i
# calibration: 予測確率と実績の一致度
```

**フォールバック**:
- all-masked連発 → weight=0に一時変更
- 最小1モデルは保持

---

## 設定ファイル例

### ensemble_A_1M.yaml

```yaml
algorithm: "ppo"
data_path: "ml-dataset-enhanced.csv"
session_id: "ensemble_A_1M_custom_ppo"
total_timesteps: 1000000

# PPO基本設定
learning_rate: 3.0e-4
n_steps: 2048
batch_size: 64
n_epochs: 10
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2

# 探索設定（モデルAは控えめ）
ent_coef: 0.6
ent_coef_schedule: "cosine_decay"
ent_coef_final: 0.2
vf_coef: 0.5
max_grad_norm: 0.5

# CustomPPO バイアス緩和
custom_ppo:
  enable_pan: true
  enable_target_entropy: true
  enable_stratified_sampling: false
  target_entropy_ratio: 0.7
  lr_temperature: 3.0e-4
  initial_temperature: 0.01

# 環境設定
environment:
  transaction_cost: 0.001
  max_position_size: 1.0
  curriculum_stage: "full"
  allow_reverse: false  # モデルAはreverse禁止

# Reward設定（SELL倍率調整）
reward_profit_bonus_multipliers: [1.0, 1.0, 0.8]  # BUY, HOLD, SELLの順
reward_settings:
  enable_forced_diversity: false

# チェックポイント
checkpoint_interval: 25000
checkpoint_dir: "./checkpoints/ensemble_A_1M"
tensorboard_log: "./logs/ensemble_A_1M"
model_dir: "./models/ensemble_A_1M"

# シード固定
seed: 101

# ステージング（将来拡張用）
staging:
  warmup_steps: 50000
  transition_steps: 150000
  main_steps: 600000
  annealing_steps: 200000

verbose: 1
offline_mode: true
```

---

## 合否判定

### 短距離A/B（未実装時は10kテストで代替）

```bash
python run_smoke_test.py --config smoke_test_10k_ensemble_A.json
```

**合格基準**:
- `legal_sell_rate >= 0.15`
- `prob_std_mean > 0` (予測分散あり)
- `cost_gate_trigger > 0` (発火している)

### ローリングOOS（25k毎）

**合格基準**:
- `Sharpe_proxy > 0` を2連続達成

### 長尺Paper（500+ steps × 3seed）

**合格基準**:
- Sharpe > 0
- MDD/回転率が閾値内
- Regime別（上昇/下降/横ばい）で破綻なし

---

## トラブルシューティング

### SELL率が0%のまま

**原因候補**:
1. Curriculum stage設定ミス → `curriculum_stage: "full"`確認
2. PAN未有効化 → `enable_pan: true`確認
3. Target Entropy未動作 → ログで`entropy_num_updates`確認

**対策**:
- `reward_profit_bonus_multipliers`のSELL倍率を上げる（0.8 → 0.9）
- `ent_coef`を上げて探索を促進（0.6 → 0.7）

### grad_normゼロ張り付き

**原因**: SELL勾配が消失

**対策**:
- PAN統計確認: `train/pan_action_counts`
- Advantage確認: `train/adv_mean(SELL)`
- 自動停止＆ダンプ（将来実装）

### メモリ不足

**対策**:
- `n_steps`を下げる（2048 → 1024）
- `batch_size`を下げる（64 → 32）
- チェックポイント削除自動化

---

## 次のステップ提案

### 今すぐできること

1. ✅ **ensemble_A_1M.yaml作成**（上記テンプレート使用）
2. ✅ **ensemble_B_1M.yaml作成**（ent_coef=0.7に変更）
3. ✅ **ensemble_C_1M.yaml作成**（ent_coef=0.8, allow_reverse=trueに変更）
4. ✅ **1Mトレーニング開始**（並列実行可）

### 追加実装で効果大

1. **τ×Tスイープ**（temperature/tau最適化） - 2-3日
2. **ローリング評価自動化** - 1-2日
3. **confidence-weighted集計** - 2-3日
4. **Gradプローブ自動停止** - 1日

### 長期的価値

1. **RecurrentPPO/LSTM化** - 2-3週間（最優先）
2. **自己教師タスク併用** - 3-4週間（高優先度）

---

## 参考資料

- `CUSTOM_PPO_SUCCESS_REPORT.md`: CustomPPO統合成功レポート
- `CUSTOM_PPO_ROLLOUT_REPORT.md`: CustomPPO横展開詳細
- `FINAL_ROLLOUT_AND_ROADMAP.md`: 最終サマリー & ロードマップ
- `ADVANCED_IMPROVEMENTS_PROPOSAL.md`: 次世代改善提案

---

**作成者**: GitHub Copilot
**最終更新**: 2025年10月7日
**ステータス**: 運用準備完了
