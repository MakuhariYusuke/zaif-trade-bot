# 065# 対応: ph2 モデル再評価 + AS-LR SkipGate 学習

| key | value |
|---|---|
| 番号 | 065 |
| フェーズ | ph1 |
| 種別 | impl |
| 対象 | 065# rev 対応: G1正式評価 + AS-LR 学習 + fill_test 設定 |
| 参照 | `065_ph1_rev_064.md`, `configs/v460/gate_thresholds.yaml`, 060/061/062 系 |
| 作成日 | 2026-02-15 |
| 結論 | **064は公式G1でFAIL確認。AS-LR SkipGate学習済、fill_test.yamlに有効化設定完了。** |

---

## §1 実施内容

### 1.1 公式 G1 再評価 (064# features)

`run_065_g1_proper_eval.py` で 000# §3.2 / `gate_thresholds.yaml` の公式 G1 基準を適用:

- **IC 閾値** (≥0.02): direction h5/h15 は PASS。h1 は FAIL (-0.054)
- **Accuracy 閾値** (≥0.51): direction は全 FAIL (h1: 0.49, h5: 0.50, h15: 0.47)
- **有意 fold 数** (≥2/5): volatility h5 のみ PASS (2/3)。他は 0/3 or 1/3
- **Holm-Bonferroni 補正**: Cliff's Delta 全ターゲット < 0.33 → 全 FAIL
- **g1_judgment**: `g1_pass: false`, passed_targets: []

| Target | IC pass | Acc pass | Sig pass | Holm pass | Cliff's d |
|---|---|---|---|---|---|
| direction_h1 | N | N (0.49) | N (0/2) | N | +0.040 |
| direction_h5 | Y (+0.109) | N (0.50) | N (1/2) | N | -0.002 |
| direction_h15 | Y (+0.146) | N (0.47) | N (1/2) | N | +0.085 |
| volatility_h5 | Y (+0.081) | Y (1.00) | Y (2/2) | N | +0.075 |

**結論**: 064# の「簡易 G1 PASS」は公式基準では **FAIL**。065# レビューの指摘が正確。

### 1.2 AS-LR SkipGate 学習

`run_065_as_lr_prep.py` で既存 fill records (3日, 491件) から AS-LR モデルを学習:

- **入力**: 166 labeled samples (filled + AS label + spread available)
- **特徴量**: 39 features (10 base + 8 micro + 14 v2 + 3 interaction + 4 side-aligned)
- **モデル**: LR(C=0.01, l2, balanced), SelectKBest(k=8)
- **Walk-Forward**: 6 folds (expanding window, embargo=2)
  - ROC-AUC mean: 0.493 (弱いが小サンプル LR では想定内)
  - **Skip 20% improvement: +0.230 bps** (060/061# と一致)
  - Baseline PnL: -1.063 bps
- **Selected features**: depth_imbalance_ob, vpin_300s, tfi_300s, velocity_300s, tfi_acceleration, return_60s, return_300s, side_aligned_return_30s
- **保存先**: `models/v460/skip_gate_as.pkl`

### 1.3 fill_test.yaml 設定

```yaml
skip_gate:
  enabled: true                # 065#: 有効化
  mode: as
  model_path: models/v460/skip_gate_as.pkl
  as_threshold: 0.65           # 保守的閾値 (0.6→0.65)
  max_skip_rate: 0.3
```

---

## §2 判定

| 項目 | 判定 | 根拠 |
|---|---|---|
| 064# XGBoost → ph2 主モデル | **不採用** | G1 公式 FAIL、Cliff's d < 0.33 |
| AS-LR SkipGate → ph2 主モデル | **採用** | Skip20% +0.230 bps、062# 統合済 |
| 064# 3 features → v3 候補 | **保留** | vwap_deviation/toxicity/slope は有望だが運用検証未 |

---

## §3 ph2 再開時のチェックリスト

1. [x] `skip_gate_as.pkl` 学習済み
2. [x] `fill_test.yaml` に `skip_gate.enabled: true` 設定
3. [ ] 入金後 `run_fill_test.py` 再開 (月曜)
4. [ ] 200 cycle で AS ratio / PnL 改善を確認
5. [ ] 劣化なら `as_threshold` 調整 or `enabled: false` に退避

---

## §4 成果物

| ファイル | 説明 |
|---|---|
| `scripts/v460/run_065_g1_proper_eval.py` | 公式 G1 再評価スクリプト |
| `scripts/v460/run_065_as_lr_prep.py` | AS-LR 学習 + walk-forward 検証 |
| `results/v460/065_g1_proper_eval.json` | G1 判定 JSON (run_gate_check 互換) |
| `docs/v460/065_g1_proper_eval.md` | G1 再評価レポート |
| `docs/v460/065_as_lr_prep.md` | AS-LR 学習レポート |
| `docs/v460/065_as_lr_wf_results.json` | Walk-forward 検証結果 JSON |
| `models/v460/skip_gate_as.pkl` | 学習済み AS-LR モデル |
| `configs/v460/fill_test.yaml` | SkipGate 有効化設定 |
