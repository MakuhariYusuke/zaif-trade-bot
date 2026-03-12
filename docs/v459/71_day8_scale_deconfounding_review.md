# 71. Day 8 スケール交絡解消実験レビュー（追加指摘）

**対象**: `docs/v459/69_day8_scale_deconfounding_results.md` / `docs/v459/70_day8_scale_deconfounding_review.md`  
**日付**: 2026-01-30  
**結論**: 69/70は良いが、**“実効reward_scaleの不一致”**と**“評価指標の定義ずれ”**が残っており、結論の強度を下げている。  
追加の改善点は「実装参照キーの確認」「ROI推定の根拠明示」「25k vs 50kの収束差分の隔離」「ent_coef以外の要因分解」。

---

## 1. 追加の重大指摘（70番未記載）

### 1) **reward_scale vs reward_scaling の実装参照が不整合**
- `RewardCalculator` 側は `reward_scaling` を参照するパスが多く、`reward_scale` が実効的に反映されているか不明。  
- 69で「scale=100に統一」と書いていても、**実コードで適用されていない可能性**がある。  
**改善案**: 実験ログに `reward_scale` と `reward_scaling` の両方を出力し、環境側が参照しているキーを確定させる。

### 2) **reward_clip の定義が結果解釈と一致していない**
- 結果では `reward_clip: [-100,100]` と記載されているが、
  設定スキーマ上は `reward_clip_min` / `reward_clip_max` のみ。  
- “clipしているつもり”が反映されていない可能性がある。  
**改善案**: 設定辞書に `reward_clip_min/max` を必ず明示し、レポートにも両方を出力。

### 3) **ROI推定の根拠が曖昧（Final Reward → ROI換算の式が不明）**
- 69のROIは計算式が明示されていない。  
- Final RewardとROIの混同は結論を歪める。  
**改善案**: ROIはポートフォリオ履歴から算出し、Final Rewardは補助指標に格下げ。

---

## 2. 中程度の見落とし

### 4) **学習時間差の因果混入**
- SAC_TUNEDは `gradient_steps=2` で学習速度が低下（13 it/s）。  
- 同じ25k stepsでも**学習更新数が倍**であり、計算負荷と発散リスクが違う。  
**改善案**: `gradient_steps=1`でSAC_TUNEDを再試行し、更新回数の影響を切り分ける。

### 5) **HOLD増加の意味づけが単純化されている**
- HOLD増加は「探索不足」だけでなく、
  **コスト回避・高分散相場での合理行動**かもしれない。  
**改善案**: HOLD比率ではなく **取引回数/turnover/平均ポジション変化**で評価。

### 6) **25k stepsは過渡期の可能性が高い**
- 25kは quick mode であり、SAC設定差が“収束前の一時状態”に過度に現れる。  
**改善案**: 25k/50k/100k の学習曲線比較を最小限追加。

---

## 3. 追加の改善提案（70番との差分）

### ✅ A. 実装反映確認ログの追加（即時）
- **学習開始時に出力**: reward_scale / reward_scaling / reward_clip_min/max / ent_coef / gamma / gradient_steps  
- これで「意図した設定が実効で反映されたか」を明文化できる。

### ✅ B. ent_coef以外の“ボトルネック仮説”検証
- **Ablation候補**:
  - gamma=0.95 → 0.99（短期化の影響）
  - batch_size=128 → 256（不安定性の原因）
  - gradient_steps=2 → 1（更新過多）

### ✅ C. ROI/Sharpe/MaxDD取得失敗の切り分け
- UnifiedTrainerのreportにポートフォリオ履歴が入っていない場合、
  **バックテスト専用パスでの算出を強制**する。  
- 目標: ROI/Sharpe/MaxDD がゼロのままの実験は結論対象から除外。

---

## 4. 結論（追加視点）

1. **スケール交絡の除去は成功しているが、実効スケールの適用が未検証**であり、
   「scale=100統一」の前提はまだ脆い。  
2. **SAC_TUNEDが有害である可能性は高いが、ent_coefだけを原因とするのは不十分**。  
3. **評価指標の定義（Final Reward vs ROI）を揃えない限り、因果結論は不安定**。  

---

## 5. 追加の最優先アクション

1. **実効設定のログ出力**（reward_scale / reward_scaling / clip / ent_coef）
2. **SAC_TUNEDの1要因ずつの切り分け（gamma/batch/grad_steps）**
3. **ROI/Sharpe/MaxDDが取得できる評価パスに固定**

---

**Reviewed by**: Codex  
**Status**: 追加改善提案（“実装反映の確定”と“評価定義の一致”が優先）  
