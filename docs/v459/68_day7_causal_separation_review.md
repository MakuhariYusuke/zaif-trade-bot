# 68. Day 7 因果分離結果レビュー（外部AIレビュー用フォーマット準拠）

**対象**: `docs/v459/67_day7_causal_separation_results.md`  
**日付**: 2026-01-30  
**目的**: 因果分離の解釈を批判的に再点検し、改善点と次の実験を提示する。

---

```json
{
  "interpretation_critique": {
    "agreed": [
      "E報酬設計がSAC_TUNEDの暴走を抑える方向に働いている可能性は高い（報酬クリップ＋スケール＋取引抑制の合成効果）。",
      "SAC_TUNED単独の結果が悪化している事実は、報酬信号の弱さ・スケール差の影響を疑う根拠として妥当。",
      "Day6とDay7のE系結果が近い点は、再現性の初期シグナルとして有用。"
    ],
    "disagreed": [
      "『SAC_TUNED単独は有害』という結論は早計。reward_scale=1 と 100 の差が巨大な交絡要因で、SAC設定そのものの因果効果は未確定。",
      "交互作用(+1.325)の算出は、Final Reward→ROI換算という不確かな変換に依存しており、効果量として過信できない。",
      "Final Rewardのみで『成功』と断定するのは危険。実ROI/Sharpe/MaxDDが未提示のため、利益性の保証にはならない。"
    ],
    "alternative_interpretations": [
      "SAC_TUNEDは“報酬スケールが十分大きい環境”でのみ安定する設定であり、S1_tunedの崩壊はスケール不一致の副作用かもしれない。",
      "E報酬のreward_clipが極端値を抑制し、報酬分散を小さくした結果として『安定に見える』だけの可能性がある（実ROIが改善しているかは別問題）。",
      "action_smoothingやtrade_frequency_penaltyの影響で“取引が減っただけ”の可能性があり、最適化というより“行動抑制”による損失最小化かもしれない。"
    ]
  },
  "overlooked_aspects": [
    "reward_scaleとreward_scalingの実装差異（実コードがどちらを参照しているか）による実効スケールの不一致",
    "Final Rewardがtrain報酬なのかeval報酬なのか不明確（評価指標の定義曖昧）",
    "取引コスト0.1%の実効影響（gross/net PnLの差）が未提示",
    "VecNormalizeや報酬正規化の有無が結果に与える影響",
    "時間帯・レジーム偏り（3ヶ月の一部レンジ相場に偏っている可能性）",
    "過学習兆候（train vs testの乖離、seed間の分散）が未評価"
  ],
  "experiment_proposals": {
    "proposal_A": {
      "validity": "high",
      "reason": "reward_scale交絡を除去する最短の因子分離。SAC_DEFAULT下でスケールだけ変更することで純粋効果を測れる。"
    },
    "proposal_B": {
      "validity": "medium",
      "reason": "時間短縮には有効だが、学習量が減るため安定性低下のリスクあり。Eの優位性が再現されるか要検証。"
    },
    "proposal_C": {
      "validity": "high",
      "reason": "2 seedsは不十分。分散が大きいタスクのため最低4 seedsで再検証すべき。"
    },
    "new_proposal": "E_reward_only / E_sac_only / E_full の3点比較を必須化。さらに reward_clip=[-2,2] / [-1,1] / [-0.5,0.5] を掛け合わせた小規模格子実験で感度を把握する。"
  },
  "fundamental_insights": [
    "SACの“最適設定”は報酬スケールと分散に強く依存し、報酬設計なしのハイパーパラメータ調整は破綻しやすい。",
    "reward_clipは学習の安定化には寄与するが、利益の尾部情報を削るため、ROI改善と必ずしも一致しない。",
    "HOLD比率の増加は“安定化”に見える一方、取引機会損失を招くため、ROIとの同時評価が不可欠。"
  ],
  "risk_warnings": [
    "Final RewardベースのROI推定は誤差が大きく、誤った意思決定につながる可能性がある。
    ",
    "reward_scaleとreward_clipの組み合わせ次第で学習が飽和・収束停止するリスク。",
    "S2_tunedの低分散は過学習または報酬クリップによる“見かけの安定”の可能性がある。",
    "取引抑制による成績改善は“利益改善”ではなく“損失抑制”の可能性がある。"
  ],
  "confidence_level": "medium",
  "recommended_next_step": "まず『純PnL + reward_scale=100 + SAC_DEFAULT』を追加し、スケール交絡を除去した上でROI/Sharpe/MaxDDを測定。次にE_reward_only/E_sac_onlyで因果分解を確定する。"
}
```

---

**Reviewed by**: Codex  
**Status**: 改善提案（因果分離の妥当性を高めるための追加実験が必要）
