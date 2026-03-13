# 189# Alt Horizon 訓練 + MacroRegime 統合 + retrain multi-horizon 拡張

## 概要

186# Phase Plan の Phase C (ev_weighted SkipGate) と Phase D (MacroRegime) の
本格実装を完了。188# で整備したインフラ上に:

1. **Alt horizon モデル訓練** — buy/pnl120, sell/pnl30 の補完モデルを実訓練
2. **ev_weighted 判定有効化** — YAML で `ev_weighted_enabled: true` 設定
3. **MacroRegime fill_cycle 統合** — FillRecord 拡張 + conflict detection
4. **retrain_scheduler multi-horizon 拡張** — alt horizon 自動再学習

## 変更ファイル一覧

### 新規作成

| ファイル | 説明 |
|---------|------|
| `scripts/v460/ml/train_alt_horizon.py` | Alt horizon スタンドアロン訓練スクリプト (~300行) |
| `models/v460/skip_gate_lgbm_pnl120_buy.pkl` | buy alt モデル (pnl120 回帰) |
| `models/v460/skip_gate_lgbm_pnl30_sell.pkl` | sell alt モデル (pnl30 回帰) |
| `reports/v460/ml_189/alt_pnl120_buy_report.json` | buy alt 訓練レポート |
| `reports/v460/ml_189/alt_pnl30_sell_report.json` | sell alt 訓練レポート |
| `tests/unit/v460/test_189_alt_horizon_macro_integration.py` | 189# テスト (42件) |
| `docs/v460/189_alt_horizon_macro_integration.md` | 本ドキュメント |

### 変更

| ファイル | 変更内容 |
|---------|---------|
| `scripts/v460/ml/retrain_scheduler.py` | `_DEFAULT_CONFIG` に alt_horizon 3キー追加、`load_retrain_config` に alt モデルパス継承、`_retrain_side_specific` に train_specs 拡張 |
| `scripts/v460/lib/fill_cycle_executor.py` | `_build_fill_record` に macro 4引数追加、`run_single_cycle` に MacroRegime 更新ブロック追加 |
| `scripts/v460/lib/fill_config.py` | `FillTestConfig` に macro 5フィールド追加、YAML `regime.macro` サブセクションパース |
| `ztb/metrics/fill_quality.py` | `FillRecord` に macro 4フィールド追加 |
| `scripts/v460/lib/config_hot_reload.py` | `enable_macro_regime`, `macro_regime_conflict_action` を hot-reload 対象追加 |
| `scripts/v460/run_fill_test.py` | `MacroRegimeDetector` 初期化ブロック追加 |
| `configs/v460/fill_test.yaml` | ev_weighted 有効化 + alt モデルパス + macro セクション + retrain alt_horizon |

## 訓練結果

### buy alt (pnl120)

| 指標 | 値 |
|------|-----|
| サンプル数 | 431 |
| target_mean | -0.218 bps |
| positive_rate | 51.3% |
| **skip20 pnl30 改善** | **+0.806 bps** |
| **skip20 pnl120 改善** | **+2.075 bps** |

### sell alt (pnl30)

| 指標 | 値 |
|------|-----|
| サンプル数 | 908 |
| target_mean | -0.375 bps |
| positive_rate | 44.3% |
| **skip20 pnl30 改善** | **+1.272 bps** |
| **skip20 pnl120 改善** | **+1.453 bps** |

## アーキテクチャ

### ev_weighted 判定フロー

```
buy サイクル:
  primary (pnl30) → pred_pnl_short
  alt     (pnl120) → pred_pnl_long
  ev = w30 × pred_pnl_short + w120 × pred_pnl_long
  skip if ev < threshold

sell サイクル:
  primary (pnl120) → pred_pnl_long
  alt     (pnl30)  → pred_pnl_short
  ev = w30 × pred_pnl_short + w120 × pred_pnl_long
  skip if ev < threshold
```

### MacroRegime 統合フロー

```
fill_cycle_executor.run_single_cycle():
  1. micro regime 更新 (既存)
  2. macro regime 更新 (189# NEW)
     → MacroRegimeDetector.update(t, price)
     → compose_regimes(micro, conf, macro_result)
     → aligned=False の場合:
        - conflict_action="downgrade" → regime_str = "ranging"
        - conflict_action="log" → ログ記録のみ
  3. FillRecord に macro_{trend,slope_5m,slope_15m,aligned} 記録
```

### retrain multi-horizon

```
_retrain_side_specific():
  train_specs = []
  for side in (buy, sell):
    train_specs.append((side, primary_path, primary_target, "primary"))
    if alt_horizon_enabled:
      train_specs.append((side, alt_path, alt_target, "alt"))
  for spec in train_specs:
    retrain_model(spec)
```

## 設定 (YAML)

```yaml
skip_gate:
  model_path_buy_long: models/v460/skip_gate_lgbm_pnl120_buy.pkl
  model_path_sell_short: models/v460/skip_gate_lgbm_pnl30_sell.pkl
  ev_weighted_enabled: true
  ev_w30: 0.4
  ev_w120: 0.6

regime:
  macro:
    enabled: false          # 初期は false (観測フェーズ)
    bucket_sec: 30.0
    slope_threshold: 1.0
    strong_threshold: 3.0
    conflict_action: log    # "log" or "downgrade"

retrain:
  alt_horizon_enabled: true
  target_buy_alt: pnl120
  target_sell_alt: pnl30
```

## テスト

42 件のテスト、9 テストクラス:

1. `TestRetrainMultiHorizon` (5件) — train_specs 構築、YAML 継承
2. `TestFillRecordMacroFields` (5件) — to_dict/from_dict roundtrip
3. `TestFillTestConfigMacroYAML` (4件) — regime.macro パース
4. `TestComposeRegimesConflict` (6件) — conflict 検出ロジック
5. `TestFillCycleExecutorMacroIntegration` (6件) — mock ベース統合
6. `TestHotReloadMacroKeys` (2件) — hot-reload キー確認
7. `TestYAMLIntegrity` (7件) — YAML 整合性検証
8. `TestMacroRegimeEdgeCases` (4件) — エッジケース
9. `TestTrainAltHorizonScript` (3件) — スクリプト構造

全テスト: **2439/2439 passed** (v460 ユニットテスト)

## 今後の展望

1. **MacroRegime Phase E**: `conflict_action=downgrade` への移行判断
   - 一定期間 `log` で conflict 率とその後の PnL を観測
   - conflict 時の PnL が有意に悪化していれば `downgrade` に切替
2. **ev_weighted 重み最適化**: w30/w120 の grid search (0.3:0.7, 0.5:0.5 等)
3. **alt モデル品質監視**: online_monitor で alt モデルの劣化を検知
4. **MacroRegime 特徴量**: macro_slope を SkipGate 入力特徴量に追加
