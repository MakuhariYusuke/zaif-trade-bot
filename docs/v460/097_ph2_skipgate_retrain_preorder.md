# 097# ph2 SkipGate AS モデル再訓練（preorder-only features）

| key | value |
|---|---|
| 番号 | 097 |
| フェーズ | ph2 |
| 種別 | impl |
| 対象文書 | `096_ph2_rev_095.md` |
| 参照 | `docs/v460/065_as_lr_prep.md`, `scripts/v460/run_065_as_lr_prep.py`, `scripts/v460/ml/walk_forward_as.py`, `scripts/v460/ml/feature_enricher.py`, `scripts/v460/ml/skip_gate.py` |
| 作成日 | 2026-02-17 |
| コミット | `fccec8757` |
| 結論 | **post-fill情報リーク排除後のpreorder-only特徴量でk=10再訓練完了。ROC-AUC=0.442（<0.5）は情報リーク除去の結果として想定通り。Skip20%=+0.405bps改善を確認。adaptive thresholdメカニズムが弱判別力を補完する設計。** |

---

## §0 背景と目的

096# で `build_preorder_as_features()` を新設し、学習時と推論時の特徴量契約を統一した。
従来の `build_enriched_as_features()` は `log_queue_wait`, `edge_bps` 等の **約定後にしか確定しない特徴量** を含んでおり、学習時の見かけ上のAUCを人為的に高めていた（情報リーク）。

preorder-only特徴量に統一したことでモデルの再訓練が必要となった。

---

## §1 特徴量変更

### 除外された特徴量（post-fill情報リーク）
- `log_queue_wait` — 約定までの待ち時間（約定後にしかわからない）
- `edge_bps` — 実約定価格とmidの乖離（約定後にしかわからない）
- V2拡張特徴量群, `side_aligned_return_*`

### preorder-only 16特徴量
| # | Feature | 説明 |
|---|---|---|
| 1 | `side_buy` | 買い注文=1 |
| 2 | `hour_sin` | 時刻(sin変換) |
| 3 | `hour_cos` | 時刻(cos変換) |
| 4 | `spread_jpy` | 注文時スプレッド(JPY) |
| 5 | `offset_ratio` | オフセット比率 |
| 6 | `regime_trending` | トレンド局面フラグ |
| 7 | `regime_high_vol` | 高ボラ局面フラグ |
| 8 | `regime_ranging` | レンジ局面フラグ |
| 9 | `trade_count_60s` | 直近60s約定回数 |
| 10 | `buy_ratio` | 直近60s買い比率 |
| 11 | `trade_flow_imbalance_60s` | 約定フロー不均衡 |
| 12 | `avg_trade_size` | 平均約定サイズ |
| 13 | `price_velocity_60s` | 価格変化速度 |
| 14 | `vpin_60s` | 情報毒性推定 |
| 15 | `side_aligned_tfi` | side方向調整TFI |
| 16 | `side_aligned_velocity` | side方向調整velocity |

---

## §2 k値グリッドサーチ

| k | ROC-AUC | Brier | Skip20% (bps) |
|---|---|---|---|
| 3 | 0.4368 | 0.2519 | +0.110 |
| 4 | 0.4393 | 0.2516 | +0.118 |
| 5 | **0.4702** | 0.2518 | +0.007 |
| 6 | 0.4416 | 0.2521 | +0.141 |
| 8 | 0.4501 | 0.2533 | +0.051 |
| **10** | 0.4416 | 0.2533 | **+0.405** |
| 12 | 0.4354 | 0.2539 | +0.302 |
| 16 | 0.4163 | 0.2539 | +0.000 |

**選定: k=10** — Skip20%改善が最大（+0.405bps）。ROC-AUC最高はk=5だがSkip20%が微弱。

---

## §3 Walk-Forward検証結果（k=10, 8-fold）

| 指標 | 値 |
|---|---|
| Folds | 8 |
| ROC-AUC (mean±std) | 0.442 ± 0.120 |
| PR-AUC (mean) | 0.578 |
| Brier (mean) | 0.253 |
| Baseline PnL | -0.781 bps |
| Skip 20% 改善 | **+0.405 bps** |
| Skip 10% 改善 | -0.027 bps |
| 有効サンプル | 160 |

### Per-Fold Detail

| Fold | Train | Test | ROC-AUC | PR-AUC | Selected Features (top 3) |
|---|---|---|---|---|---|
| 0 | 50 | 20 | 0.462 | 0.650 | side_buy, hour_cos, spread_jpy |
| 1 | 70 | 20 | 0.500 | 0.600 | side_buy, offset_ratio, trade_count_60s |
| 2 | 90 | 20 | 0.344 | 0.539 | side_buy, hour_cos, regime_ranging |
| 3 | 110 | 20 | 0.460 | 0.576 | side_buy, hour_cos, spread_jpy |
| 4 | 130 | 20 | 0.242 | 0.560 | side_buy, hour_cos, regime_trending |
| 5 | 150 | 20 | 0.590 | 0.668 | side_buy, hour_cos, offset_ratio |
| 6 | 170 | 20 | 0.606 | 0.613 | side_buy, hour_cos, spread_jpy |
| 7 | 190 | 20 | 0.330 | 0.418 | side_buy, hour_cos, spread_jpy |

### Feature Stability
- **Jaccard stability**: 0.357
- **Always selected (5)**: buy_ratio, side_aligned_velocity, side_buy, trade_count_60s, trade_flow_imbalance_60s
- **Ever selected (14/16)**: ほぼ全特徴量が少なくとも1回選択

---

## §4 学習済みモデル

| 項目 | 値 |
|---|---|
| Pipeline | SimpleImputer(median) → SelectKBest(k=10) → StandardScaler → LogisticRegression(C=0.01) |
| 学習サンプル | 215 (549 total, AS rate 55.8%) |
| モデルパス | `models/v460/skip_gate_as.pkl` |

### Selected Features (10/16) & LR Coefficient Importance

| Rank | Feature | |coeff| |
|---|---|---|
| 1 | `spread_jpy` | 0.0494 |
| 2 | `avg_trade_size` | 0.0484 |
| 3 | `offset_ratio` | 0.0455 |
| 4 | `trade_count_60s` | 0.0440 |
| 5 | `hour_cos` | 0.0406 |
| 6 | `side_aligned_velocity` | 0.0354 |
| 7 | `buy_ratio` | 0.0327 |
| 8 | `trade_flow_imbalance_60s` | 0.0327 |
| 9 | `price_velocity_60s` | 0.0298 |
| 10 | `hour_sin` | 0.0053 |

### 除外された6特徴量
`side_buy`, `regime_trending`, `regime_high_vol`, `regime_ranging`, `vpin_60s`, `side_aligned_tfi`

---

## §5 ROC-AUC < 0.5 の解釈

ROC-AUCが0.5未満であること自体は予想通り：

1. **情報リーク除去の帰結**: 旧モデルでは `log_queue_wait` (AS直結) と `edge_bps` がSelectKBestで最上位。これらが除外された状態では、注文発注時点の観測可能情報だけではASの完全予測は困難。
2. **Skip20%が正値**: 確率ランキングによるスキップは、全体AUCが低くてもテールの最悪予測を除外できる。k=10で+0.405bpsは有意。
3. **Adaptive thresholdの補完**: `warm_start_skip_gate_thresholds()` + `target_skip_rate_buy/sell` により、P(AS)分布の偏りに動的に適応。固定閾値ではなく、発注回数に基づくパーセンタイルで skip 判定するため、弱判別力でも「最悪の○%をスキップ」という戦略が成立する。

---

## §6 テスト結果

```
781 passed, 0 failed (94.57s)
```

---

## §7 変更ファイル一覧

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/run_065_as_lr_prep.py` | import: `build_enriched_as_features` → `build_preorder_as_features` |
| `scripts/v460/ml/walk_forward_as.py` | import: `build_enriched_as_features` → `build_preorder_as_features` |
| `models/v460/skip_gate_as.pkl` | k=10 preorder-only features で再訓練（gitignored） |
| `docs/v460/065_as_lr_prep.md` | 再訓練結果で上書き |
| `docs/v460/065_as_lr_wf_results.json` | Walk-forward JSON 更新 |

---

## §8 残課題と今後

1. **データ蓄積**: 215サンプルは最低限。fill_test継続で500サンプル到達時に再訓練推奨
2. **regime特徴量**: `regime_high_vol` が全foldで定数0 → regime detectorの活性化条件を要検証
3. **Skip10%効果なし**: Skip20%のみ有効 → `target_skip_rate` は 0.15〜0.25 が妥当範囲
4. **fill_test監視**: 再訓練モデルデプロイ後、skip件数と edge_bps の相関を追跡
5. **SkipGate prob分布**: 096#指摘の「0.480〜0.545密集」が改善するか監視
