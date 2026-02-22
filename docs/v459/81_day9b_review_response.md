# 81# Day 9b 50k検証: Codexレビュー対応

## 79# Codexレビューへの評価

### A. ROI計算の妥当性問題

**Codex指摘:**
> estimated_roi = final_reward × 100 は資産ベースROIとは異なる指標

**評価: ✅ 正当な指摘**

- 累積報酬とROIは直接等価ではない
- 報酬は瞬時リターンの和、ROIは初期資産対比の最終資産変化
- 報酬スケールによって解釈が変わる

**対応方針:**
```python
# 環境からfinal_balanceを取得して正確なROI計算
roi_accurate = (final_balance - initial_balance) / initial_balance * 100
```

45# run_ab_feature_test.pyには既にfinal_balance取得ロジックあり → 再利用

---

### B. update intensity問題

**Codex指摘:**
| 項目 | 45# Day5 | 78# Day9b |
|------|----------|-----------|
| lr | 0.0003 | 0.0005 |
| grad_steps | 1 | 2 |
| buffer_size | 100k | 25k |
| steps/update | 低強度 | **高強度** |

**評価: ✅ 重要な洞察**

- update_intensity ≈ lr × grad_steps / buffer_size
- Day5: 0.0003 × 1 / 100000 = 3e-9
- Day9b: 0.0005 × 2 / 25000 = 4e-8 ← **13倍の強度**

高update intensityは:
- 早期に良い結果（25kで-6.29%）
- 長期的に過適合・崩壊（50kで-36.83%）

**対応:** batch×grad_steps ablationで検証

---

### C. 45# Day5設定再現の必要性

**Codex指摘:**
> 唯一50kで-5%を達成したSAC_DEFAULTをそのまま再現し、50kで再試験せよ

**評価: ✅ 必須アクション**

45# SAC_DEFAULTを忠実に再現した50k実験がベースライン。
差分原因を特定するための比較基準。

---

### D. gamma×ent_coef 2×2実験

**Codex指摘:**
> ent_coef=0.01固定でgammaのみ試験は交互作用を見落とす

**評価: ✅ 正当**

- gamma=0.99 × ent_coef="auto" (45# Day5相当)
- gamma=0.99 × ent_coef=0.01 (75#設定)
- gamma=0.95 × ent_coef="auto"
- gamma=0.95 × ent_coef=0.01

---

## 実施予定実験（Day 10）

### 優先順位付き実験リスト

| 優先 | 実験カテゴリ | 実験数 | 時間見積 |
|------|------------|--------|---------|
| 1 | 45# Day5再現 (SAC_DEFAULT, 50k) | 2 seeds | 2h |
| 2 | gamma×ent_coef 2×2 (50k) | 4 configs × 2 seeds = 8 | 8h |
| 3 | batch×grad_steps 2×2 (25k) | 4 configs × 2 seeds = 8 | 4h |
| 4 | stage2報酬構造 (25k) | 2 configs × 2 seeds = 4 | 2h |

**合計:** 22実験、約16時間

### SAC設定比較

| 設定名 | gamma | ent_coef | batch | grad_steps | buffer | lr |
|--------|-------|----------|-------|------------|--------|------|
| SAC_DEFAULT (45#) | 0.99 | auto | 256 | 1 | 100k | 0.0003 |
| SAC_TUNED (78#) | 0.99 | 0.01 | 128 | 2 | 25k | 0.0005 |

---

## 成功基準

1. **45# Day5再現**: 50kで ROI > -10% を達成できるか
2. **最良設定特定**: gamma×ent_coef 2×2の最良組み合わせ
3. **update intensity最適化**: batch×grad_stepsの安定設定発見
4. **報酬設計**: stage2でさらなる改善があるか

---

作成日: 2025-01-XX
関連: 78#, 79#, 80#
