# 23. v457.2 Strategy Plan: Profit-First Retraining

**作成日**: 2026-01-18
**参照**:
- `docs/v457/17_v457_enhancement_roadmap.md` (Roadmap)
- `docs/v457/22_v457_1_phase2_frequency_control_review.md` (Feedback)
- `docs/v457/24_v457_2_strategy_plan_review.md` (Plan Review)

## 1. 現状分析と課題 (Based on Verification)

v457.1 (Frequency Control) の厳密な検証により、以下の事実が確定しました。

1.  **Tiny Edge**: モデルは価格の方向をわずかに予測できており、**Gross PnL (手数料抜き) はプラス** (Profit Factor 1.14) である。
2.  **Fatal Cost**: しかし、1トレードあたりの利益 (+47 JPY) に対し、コスト (-3,600 JPY) が圧倒的に大きく、**Net PnL は壊滅的**である。
3.  **Bang-Bang Control**: モデルの出力は常に最大値 (±1.0) に張り付いており、リスク管理や確信度の概念が欠落している（常に全力）。

## 2. v457.2 のコアコンセプト: "Profit First"

### A. 報酬関数の刷新と厳密化
これまでの「理論上のPnL」ではなく、「財布に残る金 (Net PnL)」を報酬とします。Doc 24 の指摘に基づき、**報酬スケール**を調整してコストが埋もれないようにします。

- **実装方針**:
  - `FastIntradayEnvV456` に `reward_settings` を通じ `reward_scale: 10.0` (仮) 等を設定し、1円の重みを大きくする。
  - `fee_penalty_weight` などのパラメータを有効化するため、`compute_hft_reward` とのインターフェース整合を確認・修正する(必要に応じて)。
  - **既存資産活用**: `ztb/trading/rewards/fast_intraday.py` のロジックを最大限活用。

### B. エントロピー制御 & Callbacks (Existing Impl Utilization)
探索と収束制御のため、レビュー未及の既存資産を活用します。

- **Callback活用**: `ztb/training/callbacks/advanced_callbacks.py`
  - `EarlyStoppingCallback`: 無駄な学習を早期に切り上げ、Overfittingを防ぐ。
  - `BestModelSaveCallback`: 最終ステップではなく、評価スコアが最良の時点のモデルを保存する。
- **Entropy**: `ent_coef` を固定値（0.01-0.05程度）で開始し、Actionの張り付き (`Bang-Bang`) を抑制する。

### C. データセット多様性 (Regime Awareness)
「高ボラティリティ相場」だけでなく、「動かない相場（レンジ）」でのHold訓練を強化します。
- **既存資産活用**: `ztb/trading/environment/factory_v456.py` の Regime Feature 生成ロジックを利用し、データの偏りを意識した学習を行う（または低ボラ場面でのHold報酬を高める）。

## 3. 実装計画

### Step 1: Config & Tooling
v457.2 用の学習コンフィグ `config/v457_2/train_config.json` を作成。コメントアウト(`//`)を使用しない正規JSON形式とする。

```json
{
    "reward_settings": {
        "reward_scale": 10.0,
        "reward_clip": 100.0,
        "alpha": 1.0,  // PnL weight
        "edge_penalty_rate": 0.0,
        "vol_floor": 0.0005
    },
    "model_params": {
        "ent_coef": 0.05,
        "learning_rate": 0.0001
    },
    "callbacks": {
        "early_stopping_patience": 50000
    }
}
```

### Step 2: Training Script Enhancement
`train_v456_simple.py` をベースに、以下の既存モジュールをインポートして強化する。

1.  **Config Loading**: `ztb/training/utils/v457_config_utils.py` を使用して外部JSONをロード。
2.  **Environment Setup**: `ztb/trading/environment/utils/fast_intraday_env_v456_utils.py` を使用。
3.  **Callbacks**: `ztb/training/callbacks/advanced_callbacks.py` から `EarlyStoppingCallback`, `BestModelSaveCallback` を組み込む。

### Step 3: Execution
- 期間: 
  - Phase 1 (Exploration): 高エントロピーで「手数料の痛み」を徹底的に学習。
  - Phase 2 (Exploitation): エントロピーを下げて利益最大化。

## 4. 期待される成果
- 取引回数が自然に（Wrapperなしで）減少する。
- Profit Factor (Net) が 1.0 を超える。
- アクション分布が ±1.0 以外の中間値（0.0付近のHold）を含むようになる。

---
この計画に基づき、`config/v457_2` の作成と学習スクリプトの改修を行います。

## 5. Execution Results (2026-01-18)

### Summary
- **Execution**: 10,000 steps training completed successfully.
- **Model**: Saved to `models/v457_2/final/sac_v457_2_final_*.zip`.
- **Modifications**:
  - `ztb/trading/rewards/fast_intraday.py`: Updated `compute_hft_reward` to accept `fee_penalty_weight` and `**kwargs`. Added logic: `extra_fee_penalty = (fee + slippage) * fee_penalty_weight`.
  - `train_v457_2.py`: Created and verified based on v456 template + advanced callbacks.
- **Observations**:
  - Training ran without errors.
  - `BestModelSaveCallback` did not generate checkpoints because the episode length (141k) > training steps (10k), preventing `ep_rew_mean` calculation.
  - Final model is available for backtesting.
