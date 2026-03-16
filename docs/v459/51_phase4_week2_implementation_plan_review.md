# 51. Phase 4 Week 2 実装計画レビュー

**対象**: `docs/v459/50_phase4_week2_implementation_plan.md`  
**日付**: 2026-01-28  
**結論**: 方針は0番の「段階的報酬設計」に回帰できており筋が良い。  
ただし、**実装経路の不一致・評価基準の不整合・MTF生成のリークリスク**が残っており、
このまま進めると「設計は正しいが効果が出ない」危険がある。  

---

## 1. 重大な指摘（修正必須）

1) **取引コストの計算が誤りで意思決定を歪める**  
`docs/v459/50_phase4_week2_implementation_plan.md:71-74`  
- **0.2%×275回=55%** は「全資金フル回転」を仮定した過大評価。  
- 実コストは **回転率（turnover）×取引コスト** で、ポジション比率に依存する。  
- 「275/50,000=0.55%」も意味が薄く、**平均保有ステップ**や**1,000ステップあたり取引数**で示すべき。  
**提案**: 実験ログから `transaction_costs` を集計し、`LongTermMetrics.transaction_cost_efficiency` に寄せて可視化。

2) **報酬設定の実装経路がコードと不整合（設定が無効化される可能性）**  
`docs/v459/50_phase4_week2_implementation_plan.md:118-151`  
- `ztb/training/reward_config.py` は存在せず、`hold_penalty / transaction_cost_penalty / drawdown_penalty_scale / action_change_penalty` は
  現行の `RewardSettings`/`RewardConfigSchema` で **未定義**。  
- このままでは**設定が反映されない**か、**期待と違う箇所に効く**可能性が高い。  
**提案**: `ztb/training/reward_config_schema.py` と `ztb/trading/environment/utils/config.py` の既存キーに合わせ、
YAML経由で制御（`trade_frequency_penalty`, `trade_cooldown_steps`, `hold_penalty_multiplier`, `drawdown_penalty_factor` など）。

3) **MTF特徴の生成にリークリスク（将来情報混入）**  
`docs/v459/50_phase4_week2_implementation_plan.md:220-233`  
- `resample(...).ffill()` は、**同一5分バー内の未来終値**を使う危険がある。  
- この設計だと学習が過度に楽になり、実運用で崩れる。  
**提案**: 5分/15分バーは「確定済みバーのみ」参照（`shift(1)` + right-closed resampleなど）。

4) **評価基準の母数が不整合（合否判定がぶれる）**  
`docs/v459/50_phase4_week2_implementation_plan.md:165-176` / `325-345` / `452-454`  
- 計画では **10実験**なのに、Gate判定は**5実験中2**と記載。  
- 完了条件も「2/10」になっており、**判定ロジックが矛盾**している。  
**提案**: 「config×seed」の母数を統一し、**最低ラインは“全seed平均”**で判断。

5) **Stage 1 の「純PnL」定義が曖昧（コスト反映・スケーリング未整理）**  
`docs/v459/50_phase4_week2_implementation_plan.md:118-131`  
- 取引コストが環境側で控除されている場合は良いが、未反映なら**ROI評価が過度に楽観**になる。  
- `reward_scale`/`reward_clip`を明示しないと、**報酬が小さすぎて学習が鈍化**する可能性が高い。  
**提案**: Stage 1は「PnL net of cost」を明確化し、`reward_scale` を固定（例: 10〜100）。

---

## 2. 中程度の懸念（調整推奨）

- **16特徴を追加しても相関削減で8に戻る可能性**  
  `docs/v459/50_phase4_week2_implementation_plan.md:193-205` / `393-398`  
  `feature_names` / `target_feature_count` / `correlation_reduction` の整合を明記すべき。

- **新規スクリプト作成が重複になりがち**  
  `docs/v459/50_phase4_week2_implementation_plan.md:165-170` / `431-444`  
  既存の `scripts/v459/run_ab_reward_experiments.py` で実験マトリクスを回せるなら、
  **新規スクリプトの追加はDRY違反**になる。

- **MTF実行時間の見積もりが仮定のみ**  
  `docs/v459/50_phase4_week2_implementation_plan.md:241-247`  
  小さなベンチマークを先に実施し、**実測値を計画に差し込む**のが安全。

- **成功基準が緩すぎる可能性**  
  `docs/v459/50_phase4_week2_implementation_plan.md:174-177`  
  「1実験でROI>0%」は偶然のリスクが高い。  
  **同一configで2seed平均が0%超**などに修正した方が妥当。

---

## 3. 既存実装の活用（DRY・実装安定性）

- **報酬設定**: `RewardConfigSchema` と `configs/rewards/*.yaml` を必ず通す。  
  既存キーで制御し、コードに新フィールドを増やさない。  
  例: `trade_frequency_penalty`, `trade_cooldown_steps`, `hold_penalty_multiplier`, `action_smoothing`。

- **Stage 1実装**: `PnLFocusedRewardCalculator` や `use_simple_reward` を優先。  
  独自 `calculate_reward()` を新設するより安全。

- **MTF生成**: `ztb/features/generators/multi_timeframe` の既存パイプラインを縮退させる方向で設計。  
  新規スクリプトは“最後の手段”にする。

---

## 4. 確認したい点（意思決定に必要）

1. 取引コストは環境側で**必ず控除済み**か？（報酬側に二重で入れていないか）  
2. `target_feature_count` / `correlation_reduction` が**16特徴を許容**する設定になっているか？  
3. 現行の報酬設定ファイルはどれが「実験B〜E」に対応するか？  

---

## 5. 最終判断

**計画の方向性は正しいが、実装パスと評価基準の整合が不足**している。  
上記の「不整合3点（コスト計算／報酬設定経路／評価母数）」を修正すれば、  
Week 2計画として十分実行可能。  

---

**Reviewed by**: Codex  
**Status**: 要修正（実装経路の整合・評価基準の統一・MTFリーク防止）  
