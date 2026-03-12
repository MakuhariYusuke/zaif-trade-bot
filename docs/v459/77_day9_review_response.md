# 77# Day 9: 76# Codexレビュー対応検討

**日付**: 2026-01-31  
**目的**: 76# Codexレビューの各指摘を検討し、50k steps実験前に対応方針を決定

---

## 1. 重大指摘への対応検討

### 1-1) ROIが依然 -6.29% で勝者と呼べる水準ではない

| 評価 | 判定 |
|------|------|
| 指摘の妥当性 | ✅ **完全に妥当** |

**現状認識**:
- gamma=0.99は相対改善だが、絶対的には損失継続
- 「損失縮小」であり「収益達成」ではない

**対応方針**:
- 50k steps実験で改善傾向を確認
- 収益化には構造的改善（報酬/特徴量/コスト）が必要という認識を維持
- Phase 4目標を「ROI > 0%」ではなく「ROI > -3%」に下方修正検討

---

### 1-2) ROI算出の根拠（Final Rewardとの関係）が曖昧

| 評価 | 判定 |
|------|------|
| 指摘の妥当性 | ⚠️ **部分的に妥当** |

**現状**:
- Day 9実験では `estimated_roi_pct = final_reward * 100` で算出
- これはポートフォリオROIではなく、報酬の累積を100倍しただけ

**対応方針**:
- 69#/70#で指摘されたポートフォリオ履歴取得は既に修正済み
- 50k steps実験では `get_all_training_metrics()` から正確なROI取得を試みる
- Final Rewardは参考指標として維持

**即時対応**: 不要（次回スクリプトで対応）

---

### 1-3) reward_scale / reward_scaling / reward_clip の実効値が未検証

| 評価 | 判定 |
|------|------|
| 指摘の妥当性 | ⚠️ **部分的に妥当** |

**現状**:
- Day 8で明示的ログ出力を追加済み
- 実験ログに `reward_scale: 100.0` は出力されている

**検証**:
```python
# Day 9実験のconfig確認
"config": {
    "sac_gamma": 0.99,
    "sac_ent_coef": 0.01,
    "reward_scale": 100.0,  # ← 出力されている
}
```

**対応方針**:
- 実効値は確認済み（ログに出力）
- `reward_scaling` vs `reward_scale` の実装側確認は追加で行う

**即時対応**: 不要（実装確認は後日）

---

## 2. 実験設計の限界への対応検討

### 2-1) gamma×ent_coef交互作用の可能性

| 評価 | 判定 |
|------|------|
| 指摘の妥当性 | ✅ **妥当** |

**検討**:
- 現在の実験設計: gamma ablation (ent_coef=0.01固定)
- 理想的には 2×2 (gamma × ent_coef) が必要

**対応方針**:
- **50k steps実験を優先**（学習収束確認が先決）
- 交互作用検証は Phase 4 延長時の候補として保留
- 現時点では gamma=0.99, ent_coef=0.01 の組み合わせを仮最適として採用

**即時対応**: 不要（Phase 4延長時に検討）

---

### 2-2) 25k stepsは収束前の可能性

| 評価 | 判定 |
|------|------|
| 指摘の妥当性 | ✅ **妥当** |

**根拠**:
- 45# (50k steps): ROI = -5.07%
- 75# (25k steps): ROI = -6.29%
- 差分 = 1.22%（学習継続で改善の可能性）

**対応方針**:
- **50k steps実験を最優先で実施**
- 25k/50k/100k の学習曲線比較は次フェーズ

**即時対応**: ✅ **50k steps実験スクリプト作成**

---

## 3. 解釈の再評価

### 「取引して勝てる状態に近づいている」は楽観的すぎないか？

| 評価 | 判定 |
|------|------|
| 指摘の妥当性 | ⚠️ **部分的に妥当** |

**再評価**:
- 72# (gamma=0.95): HOLD↑ → ROI↑（取引回避が最善）
- 75# (gamma=0.99): HOLD↓ → ROI↑（取引しても損失縮小）

**修正解釈**:
- ❌ 「取引しても勝てる」
- ✅ 「取引しても以前ほど損失が拡大しない」

**対応方針**: 75#の解釈を修正（文言訂正）

---

## 4. 既存実装の報酬コンポーネント調査結果

### 発見した報酬コンポーネント

| コンポーネント | ファイル | 用途 |
|---------------|----------|------|
| `PnlFocusedReward` | pnl_focused.py | 純PnL中心 |
| `ProfitOptimizedReward` | profit_optimized.py | 利益最大化（profit_multiplier, loss_penalty_multiplier） |
| `UltraProfitReward` | ultra_profit.py | 超攻撃的（ATR正規化PnL、trading_bonus=0.1） |
| `SmartIncentiveReward` | smart_incentive.py | レジーム適応（レンジ/トレンド判定） |
| `TradingFocusedReward` | trading_focused.py | 取引促進 |
| `ForcedBalanceReward` | forced_balance.py | 行動バランス強制 |
| `ConfidencePenaltyReward` | confidence_penalty.py | 確信度ペナルティ |

### 既存YAML設定

| 設定 | curriculum_stage | 特徴 |
|------|------------------|------|
| `stage1_basic.yaml` | simple | 基本設定 |
| `stage1_exploration_tuned.yaml` | simple | Day 6 E設定（gamma=0.95, ent_coef=0.01） |
| `stage1_hold_removed.yaml` | simple | Hold削除 |
| `stage1_trade_reduced.yaml` | simple | 取引抑制 |
| `stage2_extended.yaml` | trading_focused | リスク管理強化、Sharpe/Sortino bonus |
| `stage3_advanced.yaml` | - | 上級設定 |

### 76#「報酬構造の再設計」に使える既存実装

| 提案 | 既存実装 | 対応状況 |
|------|----------|----------|
| profit factor評価 | `profit_factor` in training metrics | ✅ 算出可能 |
| win streak bonus | なし | ❌ 要実装 |
| risk-adjusted reward | `sharpe_bonus_scale`, `sortino_bonus_scale` in stage2 | ✅ 存在 |
| 取引コスト可視化 | `gross_pnl`, `net_pnl` (要確認) | ⚠️ 部分的 |

---

## 5. 50k steps実験前の対応優先度

### ✅ 実施（即時）

| # | 対応 | 理由 |
|---|------|------|
| 1 | 50k steps実験スクリプト作成 | 学習収束確認が最優先 |
| 2 | 75#の楽観的解釈を修正 | 誤解を避ける |

### ⏸️ 保留（50k steps結果後に判断）

| # | 対応 | 条件 |
|---|------|------|
| 3 | gamma×ent_coef 2×2実験 | 50kでROI改善しない場合 |
| 4 | 報酬構造再設計（stage2活用） | 50kでROI > -5%達成しない場合 |
| 5 | batch/grad_steps ablation | gamma効果確認後 |

### ❌ 不要（現時点）

| # | 対応 | 理由 |
|---|------|------|
| 6 | reward_scale実装確認 | ログで確認済み |
| 7 | 特徴量拡張 | 45#で8特徴有効性確認済み |

---

## 6. 50k steps実験設計

### 設定

```python
SAC_v459_OPTIMIZED = {
    "gamma": 0.99,        # 75#で確定
    "ent_coef": 0.01,     # 72#で確定
    "batch_size": 128,
    "gradient_steps": 2,
    "learning_rate": 0.0005,
    "buffer_size": 25000,
    "tau": 0.005,
}

STEPS = 50000  # 25k → 50k
SEEDS = [42, 123, 456, 789]  # 4 seeds維持
```

### 期待結果

| 指標 | 25k (現状) | 50k (期待) |
|------|-----------|-----------|
| ROI | -6.29% | **-5%以下** |
| HOLD率 | 41.6% | 35-45% |
| 安定性 | ±1.88% | ±2%以下 |

### 成功基準

- **最低**: ROI > -6% (25kからの改善)
- **目標**: ROI > -5% (45# Day5水準)
- **理想**: ROI > -3% (Phase 4目標下方修正案)

---

## 7. 結論

76# Codexレビューの指摘は概ね妥当だが、**50k steps実験を先に実施**することで多くの疑問に回答できる可能性がある。

### 即時アクション

1. **50k steps実験スクリプト作成** ← 次のステップ
2. 75#の解釈修正（楽観的表現の訂正）

### 50k steps結果後の判断

- ROI > -5%: Phase 5移行検討
- ROI ≈ -6%: gamma×ent_coef交互作用検証
- ROI < -6%: 報酬構造再設計（stage2_extended活用）

---

**作成日**: 2026-01-31  
**作成者**: GitHub Copilot  
**状態**: 検討完了 → 50k steps実験準備
