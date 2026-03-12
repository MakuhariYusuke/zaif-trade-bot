# 125# LGBM PnL120 回帰モデル構築・S1 適用

> Session 125.1# — 2026-02-28
> 前提: 123# Sell 構造問題分析, 124# 外部レビュー

---

## 概要

124# レビューに基づき以下を実施:
1. **S1 即時適用**: sell 時間フィルタに UTC04/15 追加（最小侵襲介入）
2. **次世代モデル構築**: LGBM PnL120 回帰モデル（OB 特徴量付き）
3. **mode=pnl adaptive threshold**: PnL 回帰用の動的閾値較正を実装

## 124# レビュー検証結果

| 指摘 | 主張 | 検証結果 |
|------|------|----------|
| E9: quarantine 歪み | 176件隔離で指標歪曲 | ✅ blank_run_id 149 + invalid_price 27 = 176 |
| E10: rb30 データ不足 | post-rb30 でモデル評価不可 | ✅ post-rb30: 14件/7filled — 統計的に不十分 |
| E11: adaptation 干渉 | online learning が汚染 | ✅ adaptation.enabled=false (122# R2 で既に無効化) |

## S1: 売り時間フィルタ拡張

```yaml
# Before
skip_utc_hours_sell: [8, 14, 16]

# After (125#)
skip_utc_hours_sell: [4, 8, 14, 15, 16]
```

追加した時間帯:
- **UTC04 (JST13)**: -2.842 bps/trade, n=11 — 昼休み明け最悪帯
- **UTC15 (JST00)**: -1.403 bps/trade, n=24 — 日付変更帯

124# レビュー推奨: S1 単独で 48h 観察 → 効果確認後にモデル切替。

## モデル探索結果 (train_sg_v3.py)

Non-reverse BOTH Top 5 (WF OOS):

| Rank | Model | S20%_30 | S20%_120 | Score |
|------|-------|---------|----------|-------|
| 1 | **LGBM_reg_pnl120_full** | +0.075 | +0.324 | **+0.249** |
| 2 | LGBM_reg_pnl120_base | +0.008 | +0.092 | +0.226 |
| 3 | GBM_sklearn_rb30_base (現行) | +0.117 | +0.221 | +0.190 |
| 4 | LGBM_cons_profitable30_base | +0.004 | +0.233 | +0.165 |
| 5 | GBM_sklearn_profitable30_base | +0.034 | +0.216 | +0.162 |

**発見**: engineered 特徴量は base より劣化。改善に寄与するのは OB 特徴量のみ。

## 新モデル: LGBM PnL120 回帰

- **パイプライン**: SimpleImputer(median) → StandardScaler → LGBMRegressor
- **特徴量**: 19 (16 base + 3 OB: spread_bps_ob, depth_imbalance_ob, side_aligned_imbalance)
- **訓練データ**: 286 samples (pnl120 available subset, 39.7% coverage)
- **OB 特徴量カバレッジ**: 57.2% (NaN は Imputer 処理)

### Full-data 評価 (参考値、OOS ではない)

| 指標 | Skip 20% PnL30 改善 | Skip 20% PnL120 改善 |
|------|---------------------|----------------------|
| 全体 | +0.622 bps | +1.977 bps |
| Sell 限定 | +0.622 bps | **+2.277 bps** |
| Buy 限定 | +0.643 bps | +1.571 bps |

### デプロイ設定 (未適用、S1 効果確認後)

```yaml
skip_gate:
  mode: pnl                                          # as → pnl
  model_path: models/v460/skip_gate_lgbm_pnl120.pkl  # rb30 → lgbm_pnl120
  use_ob_features: true                              # false → true
  adaptive_threshold: true                           # 動的較正
  threshold_bps: 0.0                                 # 初期値 (adaptive で上書き)
```

## mode=pnl adaptive threshold 実装 (skip_gate.py)

### 課題
mode=as には `_calibrate_threshold()` があったが、mode=pnl は固定 `threshold_bps` のみ。
PnL 回帰で `threshold_bps=0.0` だと予測分布の ~50% がスキップされ、過剰。

### 解決
`_calibrate_pnl_threshold()` メソッドを追加:
- per-side PnL 予測履歴を追跡 (`_pred_pnl_history_buy/sell`)
- 目標 skip 率に対応する下位分位点で閾値を動的設定
- `adaptive_step` で段階的収束
- pickle 後方互換 (遅延初期化)

### テスト
8 テスト追加 (`TestPnlAdaptiveThreshold`):
- ウォームアップ中の静的閾値使用
- min_samples 到達後の較正開始
- per-side 独立性
- adaptive 無効時の固定閾値
- side=None フォールバック
- 段階的ステップ
- pickle 後方互換

**テスト結果**: 53 passed, 0 failed (既存 46 + 新規 7)

## ファイル変更一覧

| ファイル | 変更内容 |
|----------|----------|
| `configs/v460/fill_test.yaml` | S1: skip_utc_hours_sell に UTC04/15 追加 |
| `scripts/v460/ml/skip_gate.py` | `_calibrate_pnl_threshold()` 追加 (~70行) |
| `scripts/v460/ml/deploy_sg_v4.py` | NEW: LGBM PnL120 モデルデプロイスクリプト |
| `models/v460/skip_gate_lgbm_pnl120.pkl` | NEW: 訓練済みモデル |
| `reports/v460/ml_125/deploy_lgbm_pnl120_report.json` | NEW: デプロイレポート |
| `tests/unit/v460/test_skip_gate_d8.py` | TestPnlAdaptiveThreshold 8テスト追加 |
| `docs/v460/index.md` | 124#, 125# エントリ追加 |

## 次ステップ

1. **S1 効果観察** (48h): fill_test は既に `skip_utc_hours_sell: [4, 8, 14, 15, 16]` で稼働中
2. **S1 効果確認後**: YAML を pnl120 に切替 → fill_test 再起動
3. **OB 特徴量カバレッジ監視**: 57.2% → skip_gate_evaluator の OB 取得安定性を確認
4. **pnl120 データ蓄積**: 現行 39.7% → 120s 後の価格取得率改善を検討
