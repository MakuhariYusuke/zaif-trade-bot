# 95# Gate0/PhaseB 再レビュー（Phase5前チェック）

**対象**: `docs/v459/94_gate0_phaseb_verification_results.md`  
**参照**: `docs/v459/00_project_proposal_v459.md`, `docs/v459/66_doc00_consistency_check.md`  
**日付**: 2026-02-06

---

## Findings（重大度順）

### [Critical] 「確定した事実」の主張が統計的に過大
- 根拠データが `1 seed / 10,000 steps` のみで、断定には不十分。
- `scripts/v459/run_phase45_p1.py:89` で `SEEDS = [42]`、`scripts/v459/run_phase45_p1.py:90` で `TOTAL_TIMESTEPS = 10000`。
- にもかかわらず、`docs/v459/94_gate0_phaseb_verification_results.md:131` 以降で「確定した事実」と断定している。
- 0番/66番の要求（4seed以上、統計検定、複数指標）と不整合。

### [Critical] P1-1は実装上「PnLのみ」ではない
- 実験文書は「ペナルティ全無効」を謳うが、実装では複合報酬経路を通る可能性が高い。
- `RewardSettings.use_simple_reward` は既定 `False`（`ztb/trading/environment/utils/config.py:44`）。
- `scripts/v459/run_phase45_p1.py` では `use_simple_reward=True` も `custom_reward_params.type="pnl_centered"` も設定していない。
- `HeavyTradingEnv` は通常 `RewardCalculator` を使用（`ztb/trading/environment/heavy_env/mixins/initialization.py:655` 付近）。
- `RewardCalculator.calculate_reward()` では `confidence_penalty`、`signal_integrator`、`balance_shaping` 等が加算/減算される（`ztb/trading/environment/components/calculators/reward_calculator.py:820` 以降）。
- よって「P1-1で純PnLが確認できた」という解釈は現時点では成立しない。

### [High] Gate0は「伝播確認」であり「有効化確認」ではない
- 現在のGate0は EXPECTED/ACTUAL のキー一致確認中心。
- しかし未知キーは `custom_reward_params` に保持されるため、**一致しても報酬計算で効いていない**ケースを排除できない。
- `RewardSettings.from_dict()` が未知キーを `custom_reward_params` へ格納（`ztb/trading/environment/utils/config.py:156` 以降）。
- `SACTrainer._collect_actual_reward_params()` は `custom_reward_params` をマージしてログ比較（`ztb/training/unified_trainer/algorithms/sac_trainer.py:181` 以降）。
- 次段階として「パラメータ変更で reward component が実際に変化するか」の検証が必要。

### [High] Cost Ratio 指標が不安定で判断を誤りやすい
- `cost_ratio = costs / |gross_pnl|` は gross が0近傍だと発散し、比較指標として不安定。
- `docs/v459/94_gate0_phaseb_verification_results.md:92` の 4,256% / 15,771% はその典型。
- `ztb/training/unified_trainer/algorithms/sac_trainer.py:283` の計算式自体がこの性質を持つ。
- 実運用判断には `turnover`（約定金額総和）基準のコスト率を併記すべき。

### [High] データリーク警告が残っている
- 実行ログに `train_end_index not provided ... may cause data leakage` が出ている。
- これは損益推定の信頼性を直接毀損するため、Phase5以前に必ず解消すべき。
- `docs/v459/00_project_proposal_v459.md` と `docs/v459/66_doc00_consistency_check.md` のリーク防止方針とも不整合。

### [Medium] メモリ圧迫が再現性/運用性リスク
- ログ上で高メモリ警告が継続（100%超が長時間）。
- 実験の安定再現性に影響し、長時間検証のボトルネックになる。

### [Medium] Phase5移行判定に必要な指標が未充足
- 0番/66番で要求される `Net ROI / Profit Factor / Sharpe / MaxDD / 統計検定` が未完了。
- 現在の94は `gross/net/fee` の初期分解としては有用だが、Phase5判定材料としては不足。

---

## 評価（94の良い点）

1. Gate0導入で「設定名不一致」を検出・修正できた点は有効。  
2. gross/net/fee を同時記録した点は、原因分解の土台として正しい。  
3. 「コスト負け」仮説に具体データを付与できた点は前進。

---

## 改善提案（次の実行順）

### 1. Gate0.5（有効化確認）を追加
- 目的: 「伝播した」ではなく「計算に効いた」を確認。
- 方法:
  - 単一シナリオで `balance_penalty=0` と `balance_penalty>0` を比較し、`reward_components` の該当項が差分を出すことを自動テスト化。
  - unknown key が入った時は `warning` で即失敗扱いにする。

### 2. 「真のPnLのみ」実験を再定義
- 現状のP1-1は複合報酬が混在し得るため、再定義が必要。
- 最低条件:
  - `custom_reward_params.type = "pnl_centered"` を明示。
  - `profit_multiplier=1.0`, `loss_multiplier=1.0`。
  - signal/dynamic shaping 経路が無効であることをログで確認。

### 3. 実験最小母数を引き上げ
- 最低でも `4 seeds`、できれば `4 seeds x 2 splits`。
- 0番/66番に合わせるなら最終的に `n>=16` を満たす構成へ。

### 4. コスト評価軸を変更
- `cost_ratio(costs/|gross|)` に加えて、以下を必須化:
  - `turnover = sum(abs(delta_position) * execution_price)`
  - `fee_rate_effective = total_fees / turnover`
  - `cost_roi = total_fees / initial_balance`
- これで「過剰取引」か「エッジ不足」かを識別しやすくなる。

### 5. リーク警告を先に解消
- `train_end_index` を必須入力にし、未指定時は実験を fail-fast。
- これを直さない限り、Phase5判断に使える結果にならない。

### 6. Phase5判定の暫定Gateを明文化
- 現時点での移行前提:
  - `Balance ROI > 0`（4seed平均）
  - `Profit Factor > 1.1`（最低）
  - `MaxDD < 15%`
  - `Model > Random`（有意差つき）
- これを満たさない場合は、Phase5ではなくPhase4.5継続が妥当。

---

## 0番/66番との整合性サマリー

- **一致している点**: 設定伝播チェック、コスト分解の導入。  
- **未達の点**: seed数、統計検定、主要KPI群、リーク防止、ベースライン有意差。  
- **判断**: 94は「原因探索の中間成果」としては有効だが、Phase5移行判定としては時期尚早。

---

## 総合結論

94の方向性は正しいが、結論の強さがデータ量を上回っている。  
次は「PnL-onlyの実装純度」と「統計的有効性」を先に確定させるべき。  
これを通せば、Phase5に向けた意思決定の質が大きく上がる。
