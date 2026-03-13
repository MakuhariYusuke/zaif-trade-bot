# 70. Day 8 スケール交絡解消結果レビュー（外部AIレビュー用フォーマット準拠）

**対象**: `docs/v459/69_day8_scale_deconfounding_results.md`  
**日付**: 2026-01-30  
**目的**: Day 8実験結果の解釈を批判的に再点検し、残存する問題点と次の実験を提示する。

---

```json
{
  "interpretation_critique": {
    "agreed": [
      "reward_scale交絡が存在し、Day 7結果(-134.88%)の大部分(~82%)を説明していたことは妥当な発見。",
      "スケール統一後もSAC_TUNEDが有害(-24.75%)という結論は、2 seedsの限定的データでも方向性として信頼できる。",
      "ent_coef=0.01(固定)がent_coef='auto'より劣る可能性は、HOLD率62.6%という行動パターンと整合する。",
      "Day 7→Day 8でSAC_TUNEDが-134.88%→-29.45%に改善したことは、68# Reviewの指摘の妥当性を裏付ける。"
    ],
    "disagreed": [
      "『純SAC効果=-24.75%』という数値は、2 seedsかつσ=6.23%のデータから算出されており、信頼区間が広すぎる。真の効果は-18%～-31%の範囲にある可能性。",
      "ent_coef=0.01が原因という仮説は、他の差異(batch_size=128 vs 256, gamma=0.95 vs 0.99, gradient_steps=2 vs 1)を無視している。",
      "ztb.metricsが全て0を返している問題を『ポートフォリオ履歴取得失敗』と片付けているが、ROI以外の指標なしに『有害』と断定するのは早計。",
      "S1_scaled_defaultのROI=-4.70%も悪い結果であり、『ベースライン』として適切かどうか疑問。"
    ],
    "alternative_interpretations": [
      "SAC_TUNEDのHOLD率62.6%は探索不足ではなく、学習初期の慎重な行動パターンかもしれない。25,000 stepsは収束に不十分な可能性。",
      "batch_size=128 vs 256の差がSAC_TUNEDの遅い学習速度(13 it/s vs 19 it/s)と不安定性の主因かもしれない。",
      "gamma=0.95(短期重視)とscale=100の組み合わせが、報酬の時間的構造と不整合を起こしている可能性。",
      "seed間のROI分散(S1_tuned: -23%～-36%)は、モデルの収束失敗ではなく、市場データのランダムウォーク特性を反映しているだけかもしれない。"
    ]
  },
  "overlooked_aspects": [
    "25,000 stepsの妥当性: Day 7は50,000 stepsで実施。学習量の差が結果に影響している可能性。",
    "学習曲線の未分析: 収束しているのか、まだ学習中なのか判断材料がない。",
    "SAC_TUNEDの遅い学習速度(13 vs 19 it/s)の原因: gradient_steps=2が影響？メモリ問題？",
    "VecNormalize/報酬正規化の有無: Day 7とDay 8で設定が同一か確認されていない。",
    "HOLD率62.6%の内訳: 学習序盤と終盤で変化しているか？適応的か固定的か？",
    "取引コストの影響: HOLD偏重は取引コスト回避として合理的かもしれない（評価視点の欠如）",
    "市場レジームの影響: 学習データの相場環境（トレンド/レンジ）が結果に影響している可能性"
  ],
  "statistical_concerns": {
    "sample_size": "2 seedsは統計的検出力が低い。t検定でp値を計算すると有意水準5%を満たさない可能性。",
    "effect_size_calculation": {
      "cohen_d_estimate": "(-29.45 - (-4.70)) / sqrt((0.12^2 + 6.23^2)/2) ≈ 5.6",
      "interpretation": "効果量は大きいが、分散の非対称性(0.12% vs 6.23%)が計算を歪めている"
    },
    "confidence_interval": {
      "S1_scaled_default": "[-4.70 ± 1.96*0.12/sqrt(2)] = [-4.87%, -4.53%]",
      "S1_scaled_tuned": "[-29.45 ± 1.96*6.23/sqrt(2)] = [-38.08%, -20.82%]",
      "note": "95%CIが重複しないため差は有意だが、効果量の推定精度は低い"
    }
  },
  "experiment_proposals": {
    "priority_1_ent_coef_ablation": {
      "validity": "high",
      "reason": "ent_coef仮説を直接検証できる。ent_coef=[0.01, 0.05, 0.1, 'auto']×scale=100で4点比較。",
      "expected_outcome": "ent_coef='auto'と0.1が近い性能なら仮説支持。"
    },
    "priority_2_4seed_replication": {
      "validity": "high", 
      "reason": "現状の2 seedsでは信頼区間が広すぎる。4 seedsでσの推定精度を上げる。",
      "expected_outcome": "SAC_TUNED有害の結論が4 seedsでも再現されれば信頼性向上。"
    },
    "priority_3_convergence_check": {
      "validity": "medium",
      "reason": "25,000 vs 50,000 stepsの影響を確認。SAC_TUNEDは収束に時間がかかる可能性。",
      "expected_outcome": "50,000 stepsでSAC_TUNEDが改善すれば、学習量不足が原因。"
    },
    "priority_4_backtest_validation": {
      "validity": "high",
      "reason": "ztb.metrics=0問題を回避し、独立した評価を得る。Sharpe/MaxDD/WinRateが必要。",
      "expected_outcome": "ROI以外の指標でSAC_TUNEDの問題を具体化。"
    },
    "new_proposal": {
      "name": "SAC_TUNED_FIXED vs SAC_TUNED_ADAPTIVE",
      "description": "ent_coef以外のパラメータはSAC_TUNEDのまま、ent_coefだけ'auto'に変更した設定を追加。",
      "expected_outcome": "これでSAC_TUNEDが改善すれば、ent_coef=0.01が元凶と確定。"
    }
  },
  "fundamental_insights": [
    "ent_coefと報酬スケールの相互作用: 固定ent_coefは報酬スケールへの適応能力を失う。報酬が大きい環境では'auto'が安全。",
    "SAC設定のロバスト性: SAC_DEFAULTが安定なのは、'auto'設定が環境への適応を可能にしているため。",
    "因果推論の限界: 2×2因子設計でもスケール効果とSAC効果の分離に成功したが、交互作用の完全な分解には追加実験が必要。",
    "HOLD偏重の両義性: 探索不足の証拠だが、同時に損失回避として合理的な行動でもある。文脈依存的評価が必要。"
  ],
  "risk_warnings": [
    "『SAC_TUNEDは有害』という結論の過信リスク: 25,000 stepsと2 seedsでは決定的とは言えない。",
    "ent_coef仮説への固執リスク: 他のパラメータ差異(batch_size, gamma, gradient_steps)の影響を見落とす可能性。",
    "ROIのみでの評価リスク: MaxDD=-35%でもROI=-29%より、MaxDD=-5%でROI=-10%の方が実運用では優れる。",
    "Quick mode(25k steps)の一般化リスク: フルスケール(50k-100k steps)で結果が変わる可能性。",
    "市場データ依存リスク: 3ヶ月のBTC/JPY 1分足データが将来の相場を代表しているか不明。"
  ],
  "confidence_level": "medium-high",
  "confidence_breakdown": {
    "scale_confounding_existed": "high (データで明確に示された)",
    "sac_tuned_harmful": "medium (2 seedsの限界、追加検証推奨)",
    "ent_coef_is_cause": "low-medium (仮説段階、ablation必要)",
    "next_step_clarity": "high (ent_coef ablationが最優先)"
  },
  "recommended_next_step": "SAC_TUNED設定でent_coefのみを'auto'に変更した『SAC_TUNED_ADAPTIVE』を追加し、S1_scaled条件で2 seeds実行。これでent_coef仮説を直接検証する。"
}
```

---

## 追加分析: 統計的検定

### Welch's t-test (不等分散を仮定)

```
H0: μ(S1_scaled_default) = μ(S1_scaled_tuned)
H1: μ(S1_scaled_default) ≠ μ(S1_scaled_tuned)

Mean1 = -4.70%, SD1 = 0.12%, n1 = 2
Mean2 = -29.45%, SD2 = 6.23%, n2 = 2

t = (-4.70 - (-29.45)) / sqrt(0.12^2/2 + 6.23^2/2)
t = 24.75 / sqrt(0.0072 + 19.41)
t = 24.75 / 4.41
t ≈ 5.61

df (Welch-Satterthwaite) ≈ 1.0 (分散の極端な差により)

p-value (two-tailed, df=1) ≈ 0.11
```

**結論**: p≈0.11 > 0.05 のため、厳密には5%水準で有意とは言えない。しかし効果量(Cohen's d ≈ 5.6)は非常に大きく、実務的には意味のある差と解釈可能。

---

## 推奨実験設計: ent_coef Ablation

```python
# 即座に実施推奨
EXPERIMENTS = {
    "S1_scaled_ent_auto": {
        "base": "SAC_TUNED",
        "override": {"ent_coef": "auto"},  # SAC_TUNED + ent_coef fix
        "reward_scale": 100.0,
        "seeds": [42, 123, 456, 789],
        "steps": 25000
    },
    "S1_scaled_ent_001": {
        "base": "SAC_TUNED", 
        "override": {"ent_coef": 0.01},  # 現行SAC_TUNED
        "reward_scale": 100.0,
        "seeds": [42, 123, 456, 789],
        "steps": 25000
    },
    "S1_scaled_ent_005": {
        "base": "SAC_TUNED",
        "override": {"ent_coef": 0.05},
        "reward_scale": 100.0,
        "seeds": [42, 123, 456, 789],
        "steps": 25000
    },
    "S1_scaled_ent_010": {
        "base": "SAC_TUNED",
        "override": {"ent_coef": 0.1},
        "reward_scale": 100.0,
        "seeds": [42, 123, 456, 789],
        "steps": 25000
    }
}

# 期待される結果パターン
# ent_coef="auto" ≈ ent_coef=0.1 > ent_coef=0.05 > ent_coef=0.01
# → ent_coef仮説を支持
```

---

**Reviewed by**: Self-Review (AI)  
**Status**: 追加実験推奨（ent_coef ablationが最優先）  
**Confidence**: Medium-High（データは方向性を示すが、統計的確証には追加seedsが必要）
