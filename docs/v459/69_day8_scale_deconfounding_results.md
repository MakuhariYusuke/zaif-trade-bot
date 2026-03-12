# 69. Day 8 スケール交絡解消実験結果

**日付**: 2026-01-30  
**目的**: 68# AI Reviewで指摘されたreward_scale交絡を解消し、SAC_TUNEDの真の因果効果を測定する。

---

## 1. 実験概要

### 1.1 背景

Day 7の因果分離実験で以下の問題が指摘された（68# Review）：
- **reward_scale交絡**: S1(scale=1.0) vs S2(scale=100.0)の差がSAC効果と分離できていない
- **統計的検出力不足**: 2 seedsでは不十分
- **ROI/Sharpe/MaxDD未提示**: Final Rewardからの推定に依存

### 1.2 対応策

```
Phase A: スケール交絡解消
- 全設定でreward_scale=100.0統一
- SAC_DEFAULT vs SAC_TUNEDの純粋比較
- Quick mode: 2 seeds × 25,000 steps
```

---

## 2. 実験設定

### 2.1 SAC ハイパーパラメータ

| Parameter | SAC_DEFAULT | SAC_TUNED | 差異 |
|-----------|-------------|-----------|------|
| learning_rate | 0.0003 | 0.0005 | +67% |
| buffer_size | 25,000 | 25,000 | 同一 |
| batch_size | 256 | 128 | -50% |
| gamma | 0.99 | 0.95 | -4% |
| gradient_steps | 1 | 2 | +100% |
| **ent_coef** | **"auto"** | **0.01 (固定)** | **重要差異** |

### 2.2 報酬設定（統一）

```python
REWARD_S1_SCALED = {
    "reward_scale": 100.0,  # Day 7のS1(1.0)→統一
    "reward_clip": [-100.0, 100.0],
    "trade_frequency_penalty": 0.0,
    "action_smoothing": 0.0
}
```

---

## 3. 実験結果

### 3.1 Phase A 結果サマリー

| 実験設定 | Seeds | ROI Mean | ROI Std | HOLD率 | Final Reward |
|----------|-------|----------|---------|--------|--------------|
| **S1_scaled_default** | 2 | **-4.70%** | 0.12% | 32.6% | -0.047 |
| **S1_scaled_tuned** | 2 | **-29.45%** | 6.23% | 62.6% | -0.294 |

### 3.2 各実験の詳細

#### S1_scaled_default (SAC_DEFAULT + scale=100)

| Seed | ROI (%) | Training Time | Steps/sec | Action Distribution |
|------|---------|---------------|-----------|---------------------|
| 42 | -4.58% | 1316s | 19.0 | HOLD:32.3%, BUY:34.0%, SELL:33.7% |
| 123 | -4.81% | 1303s | 19.2 | HOLD:32.9%, BUY:33.4%, SELL:33.7% |

#### S1_scaled_tuned (SAC_TUNED + scale=100)

| Seed | ROI (%) | Training Time | Steps/sec | Action Distribution |
|------|---------|---------------|-----------|---------------------|
| 42 | -23.21% | 1918s | 13.0 | **HOLD:58.5%**, BUY:17.1%, SELL:24.4% |
| 123 | -35.68% | 1930s | 13.0 | **HOLD:66.8%**, BUY:10.2%, SELL:23.0% |

---

## 4. 因果効果の測定

### 4.1 SAC Effect（スケール統一後）

```
SAC Effect = ROI(S1_scaled_tuned) - ROI(S1_scaled_default)
           = -29.45% - (-4.70%)
           = -24.75%
```

**結論**: スケール統一後も、**SAC_TUNEDはSAC_DEFAULTより24.75%悪い**

### 4.2 Day 7 → Day 8 比較

| 設定 | Day 7 ROI | Day 8 ROI | 変化 | 解釈 |
|------|-----------|-----------|------|------|
| S1_default | -2.52% (scale=1) | -4.70% (scale=100) | -2.18% | スケール100での悪化 |
| S1_tuned | **-134.88%** (scale=1) | **-29.45%** (scale=100) | **+105.43%** | 暴走が大幅改善 |

### 4.3 スケール効果の分離

```
Day 7 SAC Effect (scale=1):   -134.88% - (-2.52%) = -132.36%
Day 8 SAC Effect (scale=100): -29.45% - (-4.70%)  = -24.75%

スケール交絡の寄与 = -132.36% - (-24.75%) = -107.61%
純SAC Effect      = -24.75%
```

**重要発見**: Day 7で観測された-132%の悪化のうち、**約82%はスケール交絡**であり、**純SAC効果は-25%**

---

## 5. 行動パターン分析

### 5.1 HOLD率の比較

| 設定 | HOLD率 | 解釈 |
|------|--------|------|
| S1_scaled_default | 32.6% | バランス良好（約1/3ずつ） |
| S1_scaled_tuned | **62.6%** | **過剰なHOLD偏向** |

### 5.2 SAC_TUNEDの問題行動

SAC_TUNEDでは：
- **ent_coef=0.01（固定）** → 探索不足
- HOLD率60%超 → 取引機会損失
- seed間分散6.23% → 不安定

---

## 6. 68# Review 対応状況

### 6.1 指摘事項への対応

| 68# 指摘 | 対応 | 結果 |
|----------|------|------|
| reward_scale交絡 | 100.0に統一 | ✅ 交絡解消、効果分離成功 |
| 2 seedsは不十分 | 2 seeds実行（Quickモード） | ⚠️ 4 seedsで要追検証 |
| ROI/Sharpe/MaxDD未提示 | ztb.metrics統合 | ⚠️ metrics=0問題あり（後述） |

### 6.2 未解決の問題

**ztb.metrics出力がゼロ問題**:
- `sharpe_ratio: 0.0`, `max_drawdown_pct: 0.0`, `win_rate_pct: 0.0`
- 原因: UnifiedTrainerからのポートフォリオ履歴取得失敗
- 対策: バックテスト実施による独立検証が必要

---

## 7. 結論と解釈

### 7.1 主要発見

1. **68# の指摘は部分的に正しかった**
   - スケール交絡（-107.6%）は確かに存在し、Day 7結果を歪めていた
   - スケール統一で暴走（-134.9%→-29.5%）は大幅改善

2. **しかしSAC_TUNEDは依然として有害**
   - スケール交絡解消後も-25%の悪化効果
   - 原因はent_coef=0.01（固定）による探索不足が濃厚
   - HOLD率60%超という過剰な取引抑制

3. **ent_coef="auto" の優位性**
   - SAC_DEFAULTのent_coef="auto"は環境に適応
   - 固定値は報酬スケールとの不整合を引き起こす

### 7.2 因果関係の修正モデル

```
Day 7の誤った因果モデル:
  SAC_TUNED → -132% ROI悪化

Day 8で修正した因果モデル:
  SAC_TUNED      → -25% ROI悪化（純効果）
  Scale Mismatch → -108% ROI悪化（交絡）
  Total Effect   → -132% ROI悪化
```

---

## 8. 次のステップ

### 8.1 即座の推奨

1. **ent_coef ablation**: ent_coef=0.01, 0.05, 0.1, "auto" の4点比較
2. **4 seeds検証**: 統計的有意性の確保
3. **バックテスト**: ztb.metrics問題回避のための独立評価

### 8.2 実験提案

```python
# Ablation: ent_coef vs scale
experiments = [
    ("scale=100, ent_coef=auto", expected="baseline"),
    ("scale=100, ent_coef=0.01", expected="bad"),
    ("scale=100, ent_coef=0.05", expected="moderate"),
    ("scale=10, ent_coef=0.01", expected="test interaction"),
]
```

---

## 9. 技術的メタデータ

```json
{
  "experiment_date": "2026-01-30",
  "total_experiments": 4,
  "total_time_minutes": 107.9,
  "seeds": [42, 123],
  "timesteps_per_experiment": 25000,
  "data_file": "btc_jpy_1m_v451_optimized_features.parquet",
  "framework": "UnifiedTrainer + ztb.metrics",
  "metrics_issue": "sharpe/maxdd/winrate all zeros due to portfolio history extraction failure"
}
```

---

## 10. Appendix: 生データ

### A. Day 7 結果（比較用）

| 設定 | ROI Mean | ROI Std | reward_scale |
|------|----------|---------|--------------|
| S1_default | -2.52% | 15.5% | 1.0 |
| S1_tuned | -134.88% | 16.6% | 1.0 |
| S2_default | +0.04% | 0.002% | 100.0 |
| S2_tuned | +0.14% | 0.006% | 100.0 |

### B. Day 8 Phase A 結果

| 設定 | ROI Mean | ROI Std | reward_scale |
|------|----------|---------|--------------|
| S1_scaled_default | -4.70% | 0.12% | 100.0 |
| S1_scaled_tuned | -29.45% | 6.23% | 100.0 |

---

**Document ID**: 69  
**Status**: Phase A Complete, Pending AI Review  
**Author**: Copilot  
**Requires**: External AI Review for validation
