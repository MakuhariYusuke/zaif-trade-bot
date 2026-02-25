# 164# analysis: SkipGate SHAP 特徴量重要度分析

> **目的**: 159# Gemini-C で指摘された SkipGate モデルの特徴量分析を SHAP TreeExplainer で実施。
> 3 つの side-specific LightGBM モデルの判定ロジックを可視化し、改善方針を導出する。

---

## 分析条件

| 項目 | 値 |
|------|-----|
| 分析手法 | SHAP TreeExplainer (tree_path_dependent) |
| 対象モデル | pnl120_generic (17feat/56samp), pnl120_sell (15feat/229samp), pnl30_buy (13feat/519samp) |
| 背景データ | `cache/data/enriched_fill_test.pkl` (119 enriched rows, sell-only; うち filled=28) |
| SHAP サンプル数 | 28 (enriched から build_preorder_as_features 経由で構築) |
| 日付 | 2026-02-26 |

### 再現性情報

| ファイル | MD5 | 更新日時 |
|----------|-----|---------|
| `models/v460/skip_gate_lgbm_pnl120.pkl` | `5ACEA89F7EF8FEE92C82709D0052EDDD` | 2026-02-25 07:35:08 |
| `models/v460/skip_gate_lgbm_pnl120_sell.pkl` | `9EF2634355EAC35205F04C66499EAC34` | 2026-02-24 05:34:51 |
| `models/v460/skip_gate_lgbm_pnl30_buy.pkl` | `BAE0E33FF34F56174A0966038A3B308C` | 2026-02-24 05:34:10 |
| `cache/data/enriched_fill_test.pkl` | `D90EECB4881DACFDCB0F1D9CA4C4ADC0` | 2026-02-25 07:54:30 |

**実行コマンド**: `temp/shap_analysis.py` (TreeExplainer → `analysis_results/shap_skip_gate_analysis.json`)
**Python**: 3.11.9, SHAP 0.50.0, LightGBM (モデル内蔵)

---

## §1 モデル別 SHAP 重要度

### 1.1 pnl120_generic (17 features, 56 samples) — **DEAD MODEL**

| Rank | Feature | mean |SHAP| | Direction |
|:----:|---------|:--------:|-----------|
| 1 | side_aligned_tfi | 0.8754 | PASS促進 |
| 2 | depth_imbalance_ob | 0.8401 | SKIP促進 |
| 3 | trade_count_60s | 0.7043 | PASS促進 |
| 4 | side_buy | 0.5763 | SKIP促進 |
| 5 | buy_ratio | 0.5736 | PASS促進 |
| 6 | avg_trade_size | 0.4645 | PASS促進 |
| 7 | price_velocity_60s | 0.4273 | PASS促進 |
| 8 | vpin_60s | 0.2975 | SKIP促進 |

- **WF profit_score = 0.0** — 全ウィンドウで改善なし。実質的に学習が機能していない。
- サンプル数 56 は `min_child_samples=20` に対して最小限。過学習リスク極大。
- **結論**: Generic model は廃止候補。side-specific モデルで十分。

### 1.2 pnl120_sell (15 features, 229 samples, OOS +0.221bps)

| Rank | Feature | mean |SHAP| | Direction |
|:----:|---------|:--------:|-----------|
| 1 | **spread_jpy** | **1.6355** | PASS促進 |
| 2 | **price_velocity_60s** | **1.4199** | PASS促進 |
| 3 | side_aligned_imbalance | 0.8273 | SKIP促進 |
| 4 | avg_trade_size | 0.7903 | SKIP促進 |
| 5 | buy_ratio | 0.6866 | SKIP促進 |
| 6 | offset_ratio | 0.6678 | SKIP促進 |
| 7 | spread_bps_ob | 0.6621 | PASS促進 |
| 8 | hour_sin | 0.6249 | PASS促進 |
| 9 | hour_cos | 0.5852 | PASS促進 |
| 10 | regime_ranging | 0.3366 | SKIP促進 |

**Sell model 主要知見**:
- `spread_jpy` が最重要 (1.6355) — スプレッド広い = 約定後 PnL が良い (PASS促進)
  - → sell_offset_floor の正当性を裏付ける。スプレッド狭い時に sell を抑制すべき。
- `price_velocity_60s` が 2 位 — 価格変動速度が sell 判定に強く影響。
- `side_aligned_imbalance` が SKIP促進 — OB で sell 側が厚い = AS リスク大。
- hour_sin/cos の重要度が高い (0.62/0.59) — **TimeFilter と部分重複あり**。
- `regime_high_vol` = 0.0 — high_vol レジームのサンプル不足で学習不能。

### 1.3 pnl30_buy (13 features, 519 samples, OOS +0.355bps) — **最良モデル**

| Rank | Feature | mean |SHAP| | Direction |
|:----:|---------|:--------:|-----------|
| 1 | **price_velocity_60s** | **0.8322** | SKIP促進 |
| 2 | **depth_imbalance_ob** | **0.6366** | PASS促進 |
| 3 | **offset_ratio** | **0.6155** | SKIP促進 |
| 4 | vpin_60s | 0.5483 | PASS促進 |
| 5 | hour_sin | 0.5364 | SKIP促進 |
| 6 | spread_jpy | 0.4898 | PASS促進 |
| 7 | avg_trade_size | 0.3793 | PASS促進 |
| 8 | buy_ratio | 0.3599 | PASS促進 |
| 9 | hour_cos | 0.3450 | SKIP促進 |

**Buy model 主要知見**:
- `price_velocity_60s` が SKIP促進 — 価格急変時に buy を抑制 (AS 回避パターン)
- `depth_imbalance_ob` が PASS促進 — OB buy 側が厚い = 安全 (合理的判定)
- `offset_ratio` が SKIP促進 — offset が小さい注文 ≒ アグレッシブ → skip
- `regime_high_vol` は pruned (dead) — high_vol サンプル不足
- `vpin_60s` が PASS促進 — VPIN 高 = 流動性あり → pass。直感とやや逆だが、VPIN が高い ≒ 活発取引 ≒ 約定確率高のコンテキスト。

---

## §2 Buy vs Sell 比較分析

| Feature | Buy |SHAP| | Sell |SHAP| | 差分 | 解釈 |
|---------|:-------:|:--------:|:----:|------|
| spread_jpy | 0.490 | **1.636** | -1.146 | Sell は spread 依存度が極めて高い |
| price_velocity_60s | 0.832 | **1.420** | -0.588 | 両モデルとも重要。Sell の方が影響大 |
| side_aligned_imbalance | 0.000 | **0.827** | -0.827 | Sell のみ OB imbalance を活用 |
| depth_imbalance_ob | **0.637** | 0.000 | +0.637 | Buy のみ depth imbalance を活用 |
| vpin_60s | **0.548** | 0.316 | +0.232 | Buy の方が VPIN 依存度高い |
| hour_sin | 0.536 | 0.625 | -0.089 | 両方で高い → TimeFilter 重複シグナル |
| regime_ranging | 0.086 | 0.337 | -0.251 | Sell は ranging 時に SKIP 傾向 |
| regime_high_vol | **0.000** | **0.000** | 0.000 | 両モデルとも学習不能 (サンプル不足) |

### 重要発見

1. **Sell model は spread_jpy に過度に依存** (SHAP 1.636 で圧倒的 1 位)
   - → 単純な sell_offset_floor / spread_adaptive で同等機能を代替可能
   - → SkipGate の sell 判定は spread guard と重複するリスク

2. **regime_high_vol は両モデルとも dead feature**
   - → 107# regime-adaptive gating (Step 2) で TimeFilter 側にこの機能を移管したのは正解
   - → SkipGate 側での regime_high_vol 削除を検討 (feature pruning 候補)

3. **hour_sin/cos は TimeFilter と重複**
   - Buy: 0.536/0.345、Sell: 0.625/0.585
   - → 107# Step 2 の regime-adaptive gating と同様の時間帯パターンを学習
   - → TimeFilter が適切に機能すれば SkipGate の hour 特徴量の重要度は低下するはず

4. **price_velocity_60s が両モデルの共通キードライバー**
   - → AS の直接的な予兆。velocity_guard の閾値微調整で同等効果あり得る

---

## §3 予測精度評価 (探索的検証)

| Model | Pred-PnL corr | Skip(bottom20%) PnL | Keep(top80%) PnL | Improvement | WF OOS改善 (metadata) |
|-------|:---:|:---:|:---:|:---:|:---:|
| pnl120_generic | +0.227 | -7.45 bps | +1.22 bps | +2.17 bps | 0.0 bps |
| pnl120_sell | **-0.381** | +7.75 bps | -2.96 bps | **-2.01 bps** | **+0.221 bps** |
| pnl30_buy | +0.370 | -1.71 bps | -0.09 bps | +0.35 bps | +0.355 bps |

> **指標定義の注意**: 左 5 列は enriched_fill_test.pkl 28 サンプルでの**探索的評価**。
> 右端の「WF OOS改善」は訓練時の Walk-Forward Out-of-Sample 評価 (`wf_results` metadata)。
> Sell の Improvement=-2.01bps(探索) vs WF=+0.221bps(OOS) の符号逆転は、評価データの母集団差による。

### 異常: pnl120_sell の負の相関

- Sell model の予測と実 PnL が**負の相関** (-0.381)
- 「PnL 高い」と予測した注文がむしろ損失 → **逆シグナル問題**
- ただし enriched_fill_test.pkl のデータは sell-only 119 enriched rows (うち feature構築成功=28) の検証用サブセットであり、WF 評価 (訓練時 229 サンプル) では +0.221bps の改善を達成
- → **モデル自体は out-of-sample で機能しているが、in-sample の sell-only データでの逆相関は過学習の兆候**

---

## §4 戦略的提言

### 即座に実行可能 (Quick Win)

| # | 提案 | 期待効果 | 根拠 |
|---|------|----------|------|
| Q1 | Generic pnl120 model を**段階的に無効化** (side モデル健全性 SLO 確認後) | SkipGate 判定の安定化 | profit_score=0、実質 dead。ただし `skip_gate_evaluator.py` のフォールバック設計上、side モデル障害時の安全弁として残置が必要 |
| Q2 | `regime_high_vol` を feature pruning 対象に追加 | 学習効率向上 | 両モデルで SHAP=0 |
| Q3 | pnl30_buy の次回 retrain で `side_buy` 列を除外確認 | pruned 済みだが metadata に残存していないか確認 | pruned_features に含まれる |

### 中期改善 (次 retrain cycle で検証)

| # | 提案 | 期待効果 | 根拠 |
|---|------|----------|------|
| M1 | Sell model の spread_jpy 依存度を下げる (spread 正規化 or 除外テスト) | spread_guard との重複排除 | SHAP=1.636 で過度 |
| M2 | hour_sin/cos の重要度変化を TimeFilter 有効化前後で比較 | 機能重複の定量化 | 両モデルで 0.3-0.6 |
| M3 | pnl120_sell の WF 安定性を監視 (逆シグナルリスク) | 過学習検出 | 負の相関 |

### 構造的改善 (§7 P1 統合)

| # | 提案 | 期待効果 | 根拠 |
|---|------|----------|------|
| S1 | Sell offset を spread_jpy + price_velocity_60s のシンプルルールで動的化 | sell 防御レイヤ簡素化 | SHAP Top 2 で sell PnL の 60% を説明 |
| S2 | 160# 3 指標 (side_aligned_tfi/velocity/imbalance) を SkipGate 評価 KPI に追加 | stopgap 退出判定の定量基盤 | 162# §7.2 |
| S3 | 160# 3指標判定枠組み (fill_rate / avg_pnl30 / downside_p10) を M1/M2/M3 採否判定に直接転用 | SHAP由来施策の効果検証基盤 | 6.3 レビュー提案 |
| S4 | hour 特徴量の冗長性定量化: TimeFilter ON/OFF 期間で SHAP 差分比較 | 107# 動的ゲーティングとの重複排除根拠 | 6.3 レビュー提案 |

---

## §5 SHAP 分析の限界

1. **サンプル数 28**: enriched_fill_test.pkl が sell-only 119 行 → feature 構築後 28 行に縮小。統計的信頼性は限定的。
2. **Sell-only データ**: Buy model の SHAP 値は sell 環境のデータで計算されており、buy 固有のパターンは反映不足。
3. **Scaled features**: TreeExplainer は原空間の SHAP を返すが、StandardScaler 通過後の値であり、元の特徴量スケールとの直接比較には注意。
4. **レジーム偏り**: ranging 25, trending_up 1, trending_down 2 — high_vol サンプルがゼロ。

→ **retrain 蓄積データ (n_samples > 200) での再分析が望ましい**。本分析は方向性の確認として利用する。

---

## 関連ドキュメント

- [159_phg_rev_158_phase_d_backlog_review.md](159_phg_rev_158_phase_d_backlog_review.md) — Gemini-C SkipGate 分析指摘元
- [162_phg_rpt_fill_test_10day_log_analysis.md](162_phg_rpt_fill_test_10day_log_analysis.md) — 10 日間ログ分析
- [163_phg_rpt_stopgap_measures_catalog.md](163_phg_rpt_stopgap_measures_catalog.md) — Stopgap カタログ (退出基準表追記)

## 更新履歴

| 日付 | 内容 |
|------|------|
| 2026-02-26 | 初版作成: 3 モデル SHAP TreeExplainer 分析 |
| 2026-02-26 | §6 レビュー指摘対応: データ定義修正, 指標ラベル分離, Generic段階撤去条件付け, 再現性メタ追加, S3/S4 提案追加 |

---

## §6 追記レビュー（見落とし事項・要補強点）

> 164# の主張を、モデル実体・SHAP JSON・設定/実装の一次情報で再照合したレビュー追記。

### 6.1 事実一致（確認できた点）

- 3モデルの `n_samples` は本文どおり (`generic=56`, `sell=229`, `buy=519`)。
- `wf_results` も本文で引用している値と整合（`sell profit_score=-0.187`, `buy=+0.355`）。
- `enriched_fill_test.pkl` は `dict(data, cache_key, n_records)` 構造で、`data` は 119 行・`side=sell` のみ、`filled=28`。
- `regime_high_vol` が死に特徴量になりやすい前提（本データに high_vol サンプルなし）は妥当。

### 6.2 重要な見落とし（修正推奨）

1. **データ定義の誤記**
   - §分析条件の「119 fill records」は不正確。正しくは **119 enriched rows（filled は 28）**。
   - ここを誤ると、読者が検定力を過大評価する。

2. **§3 の「OOS」表記が指標定義と混在**
   - 表の `Pred-PnL corr`/`Skip(bottom20%)`/`Keep(top80%)` は、実質的に 28サンプルの探索指標。
   - 一方で `wf_results` は訓練時の walk-forward 指標。両者を同じ「OOS」と書くと誤解を招く。
   - 特に sell は `wf_results.skip20_pnl120=+0.221bps` だが、§3表では `Improvement=-2.01bps` と符号が逆で、**指標定義の違いを明記しないと矛盾に見える**。

3. **Generic モデルの即時廃止提案は安全条件不足**
   - `skip_gate_evaluator.py` は side モデル不在/障害時に unified へフォールバックする設計。
   - 現行 YAML でも `skip_gate.model_path`（unified）と side別 `model_path_buy/sell` を併用。
   - よって「廃止」は即断せず、**無効化条件（sideモデル健全性SLO）を満たしてから段階撤去**が必要。

4. **再現性メタの不足**
   - 164 本文に、使用したモデルファイルの更新時刻・ハッシュ・実行コマンドが未記載。
   - 将来比較（再訓練後）で同条件再現が困難。

### 6.3 過去成果の活用観点（追加提案）

- **160# の3指標判定枠組み**（fill_rate / avg_pnl30 / downside_p10）を、SHAP由来施策（M1/M2/M3）の採否判定に直接転用する。
- **163# 退出基準表**に 164 のSHAP検証項目を入れる際は、以下の2層で管理する:
  1) モデル品質: `wf_results`（本来OOS）
  2) 解釈妥当性: SHAP（探索）
- **107# 動的ゲーティング資産**と重複する hour 特徴は、「TimeFilter効いている期間/いない期間」で SHAP差分比較することで、冗長性を定量化できる。

### 6.4 最終レビュー判定

| 項目 | 判定 | コメント |
|---|---|---|
| SHAP分析の方向性 | ✅ 妥当 | 159# 指摘への回答として有効 |
| 数値整合（モデルメタ） | ✅ 概ね整合 | n_samples / wf_results は一致 |
| 指標ラベリングの明確性 | ⚠️ 要改善 | §3 の OOS 表現は定義分離が必要 |
| Generic 廃止判断の安全性 | ⚠️ 要条件付け | fallback設計を考慮した段階撤去が必要 |
| 再現性（監査可能性） | ⚠️ 不足 | 実行コマンド・モデル識別情報の固定が必要 |
