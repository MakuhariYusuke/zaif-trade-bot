# 288# rpt: SkipGate 再訓練評価 + retrain_scheduler 整合性確認

**日付**: 2026-03-06
**種別**: rpt (調査・分析レポート)
**前提**: 287# (`e7d2f50d9`), retrain_scheduler v126#
**命名**: ph2+ph3 混合 — fill_test 計測 (ph2) と ML モデル訓練 (ph3先行) の交差領域

---

## 1. 背景

280# (R-3) で「SkipGate 再訓練は Gate 計測と並行実施可能」と判定。
287# の AttributeError 修正後、新 SHA `e7d2f50d9bda` での計測が開始された時点で、
蓄積データ 6,548 clean records (21暦日, 70 SHA) を活用した全データ再訓練を試行。

同時に、1時間毎に自動実行される `retrain_scheduler.py` (126#) との整合性を確認。

---

## 2. 全データ再訓練の結果

### 2.1 実行コマンド

```bash
.venv\Scripts\python.exe scripts/v460/ml/retrain_scheduler.py \
  --once --all-runs --config configs/v460/fill_test.yaml
```

### 2.2 結果サマリー

| モデル | ターゲット | samples | WF score | 棄却理由 | 判定 |
|---|---|---|---|---|---|
| unified | pnl30 | 1,706 | -0.72 | negative_pnl_improvement | ❌ REJECT |
| buy/primary | pnl30 | 940 | -0.22 | statistical_gate (p=0.51) | ❌ REJECT |
| buy/alt | pnl120 | 433 | -0.06 | statistical_gate (p=0.41) | ❌ REJECT |
| sell/primary | pnl120 | 411 | -2.20 | negative_pnl_improvement | ❌ REJECT |
| sell/alt | pnl30 | 885 | -0.06 | statistical_gate (p=0.35) | ❌ REJECT |

**全 5 モデルが品質ゲートにより正当に棄却**。既存モデルファイルは変更なし。

### 2.3 解釈

棄却は**ポジティブな結果**:
- 既存モデルは `retrain_scheduler` が SHA 単位で漸進的に最適化済み
- 70 SHA 混合データは特徴量分布の不連続性（コード変更による動作差）を含み、
  最新 SHA 特化モデルより劣化する
- Walk-Forward 品質ゲート + 統計ゲートが正しく機能している

### 2.4 オンラインモニター (直近 100 fill)

| 指標 | 値 | 評価 |
|---|---|---|
| pass_mean_pnl | **+0.332 bps** | 🟢 正のエッジ |
| skip_precision | **96.8%** | 🟢 非常に高い |
| buy pass_pnl | -0.947 bps | 🟡 buy 側は負 (AS 高い) |
| buy win_rate | 35.3% | 🟡 改善余地あり |
| sell pass_pnl | **+1.573 bps** | 🟢 sell 側で収益貢献 |
| sell win_rate | 60.0% | 🟢 良好 |

---

## 3. retrain_scheduler 整合性

### 3.1 機構の概要 (126#)

```
fill_test (親プロセス)
  └── retrain_scheduler (子プロセス, PID 56880/56328)
        interval: 3600s (1時間)
        latest_run_only: true → 最新 SHA のデータのみ学習
        quality_gate: WF OOS + statistical_gate
        deploy: アトミック書込 → hot_reload で自動反映
```

### 3.2 稼働状態

| 項目 | 値 |
|---|---|
| プロセス | PID 56880/56328 (2026-03-06 02:26 開始) |
| 累計履歴 | 686 エントリ |
| 成功デプロイ | 22 回 |
| 直近デプロイ | 2026-03-05 14:08 (unified + sell/alt) |
| 現在 | bootstrap starvation (新 SHA で 7 records < 25 閾値) |

### 3.3 直近のデプロイ履歴

```
2026-02-28 00:04  skip_gate_lgbm_pnl120.pkl
2026-02-28 03:59  skip_gate_lgbm_pnl120.pkl
2026-02-28 11:04  skip_gate_lgbm_pnl120.pkl
2026-03-03 06:25  skip_gate_lgbm_pnl120.pkl
2026-03-03 10:25  skip_gate_lgbm_pnl120.pkl
2026-03-05 06:07  skip_gate_lgbm_pnl120.pkl
2026-03-05 14:08  skip_gate_lgbm_pnl120.pkl     (unified)
2026-03-05 14:08  skip_gate_lgbm_pnl30_sell.pkl  (sell/alt)
```

### 3.4 bootstrap 回復見通し

- cycle_interval: 120秒 → ~30 records/hour
- bootstrap 閾値: 25 records → **~50分で回復** (02:26 開始 → ~03:16 頃)
- side 別閾値: 50 records → ~100分 (buy/sell 各半数の場合)
- 品質ゲート通過 → 自動デプロイ → hot_reload = **手動介入不要**

### 3.5 整合性判定

| 観点 | 評価 |
|---|---|
| `--all-runs` と 1h 再学習の競合 | ⚠️ なし — `--all-runs` は全棄却、ファイル未変更 |
| 仮にデプロイされた場合の競合 | 🟢 次回 1h 再学習が上書き (WF 品質ゲートで劣化防止) |
| SHA 変更後の自動回復 | 🟢 bootstrap 機構で ~50分後に自動再開 |
| hot_reload との連携 | 🟢 既存 hot_reload_check_interval=120s で自動検出 |

---

## 4. モデルバックアップ

`models/v460/backup_pre288/` に再訓練前のモデルを保存済み:
- skip_gate_lgbm_pnl120.pkl
- skip_gate_lgbm_pnl30_buy.pkl
- skip_gate_lgbm_pnl120_sell.pkl
- skip_gate_lgbm_pnl120_buy.pkl
- skip_gate_lgbm_pnl30_sell.pkl

全棄却により実際のモデルファイルは変更されていないが、監査証跡として保持。

---

## 5. 結論と推奨

### 5.1 現行 SkipGate の評価

**現行モデルは十分に最適化されている**。
- retrain_scheduler が漸進的に 22 回のデプロイを実施
- オンラインモニターは正の PnL エッジ (+0.332 bps) を確認
- 全データ混合が改善しない = 最新データへの特化が正しい戦略

### 5.2 280# R-3 への回答

280# で「SkipGate 再訓練」を R-3 (高優先) として挙げていたが:

> **R-3 は retrain_scheduler により既に自動達成されている。**
> 手動介入は不要。品質ゲートが劣化を防止しつつ、漸進的最適化が継続中。

### 5.3 今後のアクション

1. **コードフリーズ継続**: 168h 計測 (R-1) に集中。SHA `e7d2f50d9bda` を維持
2. **retrain_scheduler の自然回復を待つ**: ~50分でbootstrap回復、以降1h毎に自動再学習
3. **buy 側の改善余地**: pass_pnl=-0.947bps は課題だが、R-4 (spread_adaptive) 等で対応
4. **次回の手動介入**: G1.2-full PASS 後の Ph3 移行時に包括的再訓練を検討

---

## 関連

- 280# [280_ph2_rpt_position_and_remaining_tasks.md](280_ph2_rpt_position_and_remaining_tasks.md) — R-3 SkipGate 再訓練
- 126# [126_ph2_impl_retrain_hot_reload.md](126_ph2_impl_retrain_hot_reload.md) — retrain_scheduler 実装
- 287# [287_ph2_fix_balance_forced_switch_attribute.md](287_ph2_fix_balance_forced_switch_attribute.md) — 直前の修正
- 097# [097_ph2_skipgate_retrain_preorder.md](097_ph2_skipgate_retrain_preorder.md) — preorder features 再訓練
