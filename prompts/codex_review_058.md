# Codex Review 依頼: 058# ML Enrichment + PnL Skip Gate

## プロジェクト概要

BTC/JPY 自動取引ボット (v460 "Microstructure Edge")。Coincheck で maker 注文 (手数料 0%) を出し、板の最良気配にオフセットを加えた指値注文で利益を狙うシステム。

**現在の問題**: fill test (実弾テスト) を 3 日間実施したが、**平均 PnL が -0.51 bps** で赤字。  
→ 058# で PnL 予測ベースの skip gate を構築し、赤字注文を事前にスキップすることで **+0.03 bps** まで改善。

## レビュー対象ファイル

### 1. `scripts/v460/ml/feature_enricher.py` (448行)
- raw orderbook snapshots (15,258件) と trades (866,830件) から fill record にマイクロストラクチャ特徴量を付与
- 板の binary search マッチング (±5秒)、約定フローの rolling window (60秒) 統計
- 21 特徴量: Base(10) + Micro(8) + Interaction(3)

### 2. `scripts/v460/ml/skip_gate.py` (375行)
- Ridge (α=10.0) による PnL 予測 → skip/pass 判定
- `SkipGate` クラス: evaluate, save, load
- `build_features_from_market_state()`: リアルタイム特徴量構築
- レート制限: 直近20件中 70% 以上のスキップを防止

### 3. `scripts/v460/ml/run_ml_pipeline.py` (392行)
- AS 分類器 + PnL 回帰パイプラインの CLI
- `--enriched` フラグで enriched 特徴量使用

### 4. `tests/unit/v460/test_enricher_skip_gate.py` (511行)
- 23 テスト (608 total passed)

### 5. 依存ファイル (057#)
- `scripts/v460/ml/data_loader.py` — fill records → ML 特徴量
- `scripts/v460/ml/as_classifier.py` — AS 予測
- `scripts/v460/ml/fill_classifier.py` — Fill 予測

## レビュー観点

### A. 正確性
1. `_compute_trade_features()` の VPIN 計算: `|buy_vol - sell_vol| / total_vol` — これは簡易 VPIN であり、学術的な Volume-Synchronized PIN とは異なる。この簡易計算で十分か?
2. `_find_nearest_ob()` の binary search: `np.searchsorted` + ±1 候補チェック。エッジケース (先頭/末尾) はカバーされているか?
3. NaN 処理: median 補完は小サンプル (373) で偏りを生まないか?

### B. 統計的妥当性
4. Ridge α=10.0: デフォルト寄りだがグリッドサーチ未実施。α の選択は結果に大きく影響するか?
5. TimeSeriesSplit CV: 5-fold で 373 サンプル → 各 fold ~74 サンプル。統計的有意性はあるか?
6. 30 秒 PnL ウィンドウ: maker 注文のスリッページ評価として 30 秒は適切か? 10秒/60秒/300秒と比較すべきか?

### C. 設計判断
7. AS 二値分類 → PnL 連続回帰への方針転換は妥当か? 他にどのようなアプローチが考えられるか? (例: Quantile regression, Survival analysis)
8. `max_skip_rate=0.7` の根拠は? 流動性の低い市場では過剰スキップが致命的。
9. Interaction 特徴量 (side_aligned_*): side × market state の交互作用は他にもありえるか?

### D. 実運用上の懸念
10. モデルの劣化 (concept drift): 3 日分のデータで学習したモデルが 1 週間後も有効か?
11. `build_features_from_market_state()` のレイテンシ: リアルタイムで板・約定データを処理するオーバーヘッドは?
12. pickle のセキュリティ: 信頼済みモデルのみだが、改ざんリスクはないか?

### E. テスト品質
13. `TestIntegration` で real data を使用: CI 環境で raw data がない場合の動作は?
14. 過学習検出テスト: CV スコアが train スコアと大きく乖離していないかの自動チェックはあるか?

## 主要結果 (再掲)

| 指標 | Before | After | Δ |
|---|---|---|---|
| Mean PnL | -0.51 bps | +0.03 bps | **+0.53 bps** |
| Skip rate | 0% | 62% | |
| Keep orders | 310 | 117 | |
| Top feature | - | `offset_ratio` (|coef|=0.97) | |

## ドキュメント

詳細は `docs/v460/058_ph2_ml_enrichment_skip_gate.md` を参照。
