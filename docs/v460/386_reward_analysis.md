# 386# 報酬関数分析レポート

## 概要

385# baseline (gamma=0.80) で Seeds 123/456 が OOS 負になる根本原因を分析。
報酬のペナルティコンポーネントが PnL 報酬に対して過大であることを発見。

## 報酬コンポーネント構成

```
reward = pnl_reward          # PnL × 1.0 × 1.0 = O(1-50) JPY
       + position_penalty    # -0.01 × (pos/max_pos)² = O(-0.01)
       + hold_penalty        # HOLD時 -0.01
       + consistency_penalty # BUY→SELL反転時 -0.05
       + confidence_penalty  # 損失×高確信時 O(-0.04)
       + balance_penalty     # BUY/SELL比率偏り時 O(-0.05〜-0.5)  ← 問題
       + balance_shaping     # バランス改善時 +0.05
       + entropy_shaping     # 行動多様性 +0.01
       → asymmetric_scaler(×1.0) → clip[-80, 80]
```

## 問題分析

### 🔴 P0-3: `balance_penalty_value=1.0` が PnL に対して過大

| 条件 | PnL reward | balance_penalty | 比率 |
|------|-----------|-----------------|------|
| 微小 PnL step (5 JPY) | +5.0 | -0.5 (10%偏り) | 10% |
| 中程度 PnL step (20 JPY) | +20.0 | -0.5 | 2.5% |
| PnL ≈ 0 step | +0.0 | -0.5 | **支配** |

**影響**: PnL が小さい step でエージェントが「PnL 最大化」より「バランス維持」を学習。
市場状況に関わらず BUY/SELL を均等に打とうとし、トレンド追従が阻害される。

### 🔴 P0-4: `hold_penalty=-0.01` が過剰取引を誘発

- Seed 456: 59K trades on OOS (50K training steps 以上) = ティック毎に複数取引
- hold_penalty は HOLD するたびに -0.01 のコスト
- PnL が 0-5 JPY の低ボラ期間では hold_penalty の回避が優先される
- 結果: marginal な取引が増え、cumulative で負の PnL

### 🟡 P1-4: `consistency_penalty=-0.05` がトレンド転換対応を阻害

- BUY→SELL 反転を罰するため、トレンド転換時の方向転換が遅れる
- Seed 123 の保守性 (22K trades) の一因

### Seed 別影響

| Seed | OOS ROI | 主要影響因子 |
|------|---------|------------|
| 42 | +4.37% | PnL が十分大きく、ペナルティの影響が相対的に小さい |
| 123 | -0.39% | consistency_penalty → 方向転換遅延 + 取引機会損失 |
| 456 | -0.44% | hold_penalty → 過剰取引 + balance_penalty → 不適切なバランス取引 |
| 789 | +2.00% | 42 と類似、強い PnL シグナルがペナルティを上回る |

## 推奨調整案 (387# 予定)

gamma=0.95 実験結果を待ち、G2 PASS 不達の場合に適用。

| 対象 | 現行値 | 推奨値 | 根拠 |
|------|--------|--------|------|
| `balance_penalty_value` | 1.0 | **0.1** | PnL SNR 改善、バランス強制の緩和 |
| `hold_penalty_weight` | 0.01 | **0.001** | 過剰取引の抑制 |
| `consistency_penalty` | -0.05 | **-0.01** | トレンド転換対応の改善 |
| `confidence_penalty_threshold` | 0.05 | **0.2** | 過剰なconfidence 検出の緩和 |

### 科学的アプローチ

1. **386# (現在)**: gamma=0.95 のみ変更 → 報酬ペナルティの影響を切り分け
2. **387# (予定)**: gamma=0.95 + reward tuning → 複合効果の検証
3. **388# (予定)**: 最適 gamma + 最適 reward → G2 PASS 達成

## PnL スケーリングについて

`reward_scaling=5.0-10.0` にすると PnL 報酬が 5-10 倍になり、
ペナルティとの SNR が改善される。ただし SAC の entropy term α との
バランスに注意が必要。

current: PnL(5 JPY) vs balance_penalty(-0.5) → SNR = 10
scaled:  PnL(25 JPY) vs balance_penalty(-0.5) → SNR = 50

→ penalty 縮小 + PnL スケーリングの組み合わせが最も効果的。
