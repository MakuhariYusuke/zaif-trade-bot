# 55. 実行前レビュー（Day6 報酬調整スクリプト一式）

**対象**: `docs/v459/54_ai_review_prompt.md`  
**レビュー範囲**: `scripts/v459/run_day6_reward_tuning.py`, `scripts/v459/run_ab_reward_experiments.py`, `configs/rewards/*.yaml`  
**日付**: 2026-01-28  
**結論**: **現状は no-go**。理由は「報酬設定の適用経路が誤り」「Reward YAMLがスキーマ不整合で読み込み失敗」「意図したパラメータが実際の報酬計算に反映されない」ため。長時間実行の前に修正が必須。  

---

## 重大な指摘（Critical）

1) **報酬設定が適用されず、さらに `update()` で例外化**  
   - `scripts/v459/run_day6_reward_tuning.py:118-131`  
   - `load_reward_config()` は **RewardSettingsオブジェクト**を返すため、`dict.update()` で **TypeError**。  
   - さらに、`training.reward` は **SACTrainer/EnvironmentConfig が参照しない**ため、仮に例外が出なくても **設定が反映されない**。  
   **影響**: B〜E 実験が実質ベースライン化 or 実行失敗。  
   **修正案**: `training.environment.reward_settings` に **dict** を投入する。  
   例: `RewardConfigSchema.load_and_validate()` → dict を `environment.reward_settings` へ設定。  

2) **Reward YAMLがスキーマ不整合で読み込み失敗**  
   - `configs/rewards/stage1_hold_removed.yaml` / `stage1_trade_reduced.yaml` / `stage1_exploration_tuned.yaml`  
   - 必須項目 `description` が欠落。  
   - `max_drawdown_penalty_weight` 等の **未知キー** が多数 → `RewardConfigSchema.validate()` で **即エラー**。  
   **影響**: C/D/E が起動時点で失敗。  
   **修正案**: `reward_config_schema.py` の `OPTIONAL_FIELDS` に準拠するキーに統一。  

3) **「Hold削除」「取引抑制」の意図が実コードに反映されない**  
   - 例: `action_smoothing` は `behavior_optimization.action_smoothing` を参照（`reward_calculator.py`）、YAMLのトップレベル `action_smoothing` は無効。  
   - `Hold削除` は `trading_bonus` を0にしているが、Holdペナルティは `custom_reward_params.hold_penalty_*` 系で決まる。  
   **影響**: 設計意図と実効果が乖離し、実験が無意味になる。  
   **修正案**: `behavior_optimization` 配下のキーに統一、または明示的に `hold_penalty_multiplier: 0` などを custom_reward_params に設定。  

---

## 高優先の懸念（High）

4) **JSON保存が失敗する可能性**  
   - `scripts/v459/run_day6_reward_tuning.py:186-235`  
   - `convert_to_json_serializable` が **Path/datetime/dataclass** を処理しない。  
   - 報酬設定を正しく入れると `RewardSettings` が混入し、`json.dump` 失敗リスク。  
   **修正案**: `run_ab_reward_experiments.py` の `convert_to_native()` を流用するか、`report/config` を保存前に `dataclasses.asdict()` で変換。  

5) **過去のWindows KeyboardInterrupt対策が未適用**  
   - `scripts/v459/run_day6_reward_tuning.py` で `ZTB_SAFE_DATETIME` / `ZTB_SKIP_SCIPY` / `ZTB_SKIP_SKLEARN` 未設定。  
   - 既知のSciPy/Sklearn import問題が再発する可能性。  
   **修正案**: `run_ab_reward_experiments.py` と同じ環境変数セットを適用。  

---

## 中程度の懸念（Medium）

6) **`use_precomputed_features` が未使用フラグ**  
   - `scripts/v459/run_day6_reward_tuning.py:110-116`  
   - コア実装側にこのフラグの参照が存在しない。  
   **提案**: `feature_names` を明示するか、`correlation_reduction/target_feature_count` を固定し、特徴数が想定からズレないよう制御。  

7) **ベースライン比較の再現性が弱い**  
   - A(Baseline) が「デフォルト設定」に依存。  
   - デフォルトが変化すると比較不能。  
   **提案**: baseline 用に明示的な YAML を固定し、差分を明確化。  

---

## 追加で見直したい点（Low）

- `reward_clip_min/max = [-1, 1]` と `reward_scale = 100` の組み合わせで **ほぼ全報酬がクリップ**される可能性。  
- `eval_freq=10000, n_eval_episodes=5` は 50k ステップに対して重め。時間短縮が目的なら `n_eval_episodes=1-2` へ。  

---

## 必須の修正方針（最短パス）

1. **Reward設定の適用経路を修正**  
   - `training.environment.reward_settings` に **dict** を入れる。  
   - `load_reward_config()` の代わりに `RewardConfigSchema.load_and_validate()` を使用。  

2. **YAMLのキーをスキーマ準拠に修正**  
   - 必須キー: `name/description/curriculum_stage/reward_scale` を全ファイルに追加。  
   - 未知キーを排除 or schema拡張。  

3. **Hold削除/取引抑制のキーを実装に合わせる**  
   - `behavior_optimization.action_smoothing` など、実コードが読むキーへ変更。  

---

## 追加テスト提案

- **config_load_stage1_hold_removed**: 3つのstage1_*.yamlが `RewardConfigSchema` を通過することを確認。  
- **reward_settings_injection**: `training.environment.reward_settings` が `EnvironmentConfig` に取り込まれることを検証。  
- **report_json_serialization**: `save_results()` が RewardSettings を含んでも落ちないか確認。  

---

## Go/No-Go 判定

**no-go**  
上記 Critical 3点の修正完了後に再レビュー推奨。  

---

## 実行時に監視すべきログキーワード（参考）

- `Failed to normalize environment config`  
- `Config validation failed`  
- `Training failed (success=False)`  

---

**Reviewed by**: Codex  
**Status**: no-go（修正必須）  
