# 377# ph3 統合方針: 375#/376# レビュー反映 軌道修正

| 項目 | 値 |
|---|---|
| 文書番号 | 377# |
| フェーズ | ph3 (SAC Sidecar → Live Injection) |
| 前提 | 374# (4Phase設計), 375# (Codex), 376# (Gemini 3.1 Pro) |
| 作成 | Copilot |
| ステータス | **ACTIVE — Phase 3.1 コード実装完了、SAC live 起動待ち** |

---

## §1 問題提起

374# は SAC 連続値活用の 4 Phase 設計を提案したが、外部レビュー 2 件 (375# Codex, 376# Gemini 3.1 Pro) により以下の構造的問題が判明:

1. **数値感の致命的誤り**: `max_boost_bps=3.0` は base_offset の 28.7 倍、median spread (2,282 JPY) を超える 3,274 JPY に相当 → 自殺的
2. **データパリティ未達**: `build_features.py` に M2-M5 市場理論特徴量がゼロ行。Phase 3.2 の obs 拡張は物理的に不可能
3. **埋もれた v456 資産**: `FastIntradayEnvV456` (1062行, 88-dim, GroupedFeatureScaler, Ichimoku, regime 13D one-hot) が放置状態
4. **SAC live 不在**: retained logs に sidecar trace ゼロ件。`sidecar_signal.json` の更新実績、retrain 履歴が未確認
5. **α 検証指標の不適切さ**: return correlation ではなく maker uplift (fill_rate, post_fill_30s_pnl, adverse_selected) で評価すべき

本文書は、これら全指摘を統合した **修正後の実行方針** を定義する。

---

## §2 レビュー合意事項

### §2.1 両レビューの一致点

| 論点 | 375# (Codex) | 376# (Gemini 3.1 Pro) | 合意 |
|---|---|---|---|
| max_boost_bps | 0.1-0.3 bps ladder | 0.15 bps 絶対上限 | **0.15 bps (ladder: 0.1/0.15/0.2)** |
| Phase 3.2 | HOLD (parquet M2-M5 欠落) | HOLD (build_features.py ゼロ行) | **HOLD — データパリティ復元が先決** |
| Phase 3.3/3.4 | NO-GO | NO-GO | **NO-GO — α 未証明段階では危険** |
| SAC live presence | 0件の sidecar trace | sidecar_signal.json 不在 | **live 存在確認が Phase 3.1 の前提** |
| 環境選択 | LiteTradingEnv 妥当 | FastIntradayEnvV456 復活 | **3.1 は LiteTradingEnv、3.2+ で FastIntradayEnvV456 検討** |

### §2.2 レビュー間の相違点

| 論点 | 375# (Codex) | 376# (Gemini 3.1 Pro) | 本文書の判断 |
|---|---|---|---|
| max_boost_bps 上限 | 0.3 bps まで許容 | 0.15 bps 厳格 | **0.15 を default、0.2 を hard ceiling** (保守的に) |
| FastIntradayEnvV456 の位置付け | 言及なし | 即時復活を推奨 | **段階的: 3.1 は LiteTradingEnv、データパリティ復元後に FastIntradayEnvV456 を評価** |
| v454 MarketRegimeClassifier | 言及なし | offline regime annotation に即活用 | **build_features.py M2-M5 追加時に統合検討** |

---

## §3 修正された Phase 判定

| Phase | 名称 | 旧判定 (374#) | 新判定 | 条件 |
|---|---|---|---|---|
| **3.1** | Proportional Boost | GO | **条件付き GO** | max_boost_bps=0.15、SAC live presence 確認後 |
| **3.2** | Regime-Aware Obs | GO | **HOLD** | build_features.py M2-M5 ETL 完了後 |
| **3.3** | Parameter Modulation | GO | **NO-GO** | Phase 3.1 で α 証明後のみ再検討 |
| **3.4** | Closed-Loop Reward | GO | **NO-GO** | causal confusion リスク → offline attribution のみ |

---

## §4 Phase 3.1 実行計画 (修正版)

### §4.1 Phase 3.1 前提条件チェックリスト

Phase 3.1 **コード実装は完了** (374# impl `82675725d`)。live 稼働には以下が必要:

- [x] `compute_sidecar_offset_bps_v2()` + `_shaping_fn()` 実装 (sidecar_types.py)
- [x] FillConfig 5フィールド + YAML parser + hot-reload 配線
- [x] CycleGateAggregator v1/v2 切替 + `sidecar_enabled` ガード
- [x] fill_config_validation.py sidecar バリデーション (max_boost≤0.20 ceiling)
- [x] 66 tests (55 core + 11 validation), 4637 v460 total pass
- [ ] `cache/sidecar_signal.json` が scheduler により定期更新されている → **❌ 未生成**
- [ ] `logs/sac_retrain_history.jsonl` に retrain 履歴が存在する → **❌ 未生成**
- [ ] `fill_records` に `sidecar_offset_bps` non-null エントリが存在する → **❌ (signal なし)**
- [ ] log に sidecar 関連エントリが出力されている → **❌ 0 件**

**SAC Live 状況 (2026-03-11 調査):**
- SAC checkpoints: 1310件存在 (最新 2/11, 実験用 `HeavyTradingEnv`)
- training parquet: 136MB 存在 (`data/btc_jpy_1m_full_registry_features.parquet`)
- `sac_retrain_scheduler.py`: standalone CLI、fill_test への自動統合なし
- `orchestrator_mid_cycle.py`: signal 読み込み済み (None → offset=0 で安全フォールバック)
- **結論**: コードは完全に配線済み。scheduler の初回実行が必要。

### §4.2 修正されたパラメータ

```yaml
# configs/v460/fill_test.yaml (Phase 3.1 追加セクション)
sidecar:
  max_boost_bps: 0.15        # 375#/376# 合意: 絶対上限 0.2
  dead_zone: 0.10             # SAC noise floor
  shaping: linear             # 初期は linear で開始
  enabled: true               # hot-reload で無効化可能
```

### §4.3 検証指標 (maker uplift — 375# §6 準拠)

| 指標 | 定義 | 判定基準 |
|---|---|---|
| `fill_rate` | sidecar 有/無の fill 率差分 | `Δfill_rate > 0` |
| `post_fill_30s_pnl` | fill 後 30 秒の PnL (bps) | `ΔPFL > 0` |
| `adverse_selected` | 逆選択率差分 | `ΔAS ≤ 0` (悪化しない) |
| `postonly_crossing_skip` | post-only 違反回避率 | 悪化しない |

### §4.4 Ladder テスト計画

```
Step 1: max_boost_bps = 0.10  (24h 以上)
Step 2: max_boost_bps = 0.15  (24h 以上)
Step 3: max_boost_bps = 0.20  (24h 以上、0.15 が良好な場合のみ)

各ステップで §4.3 の 4 指標を same-SHA 比較。
悪化傾向 → 即時 max_boost_bps = 0.0 (hot-reload)。
```

---

## §5 データパリティ復元ロードマップ (Phase 3.2 前提)

### §5.1 現状の欠落

376# §2.1 が指摘した `build_features.py` の欠落:

| 市場理論 | 必要な列 | build_features.py の状態 | parquet の状態 |
|---|---|---|---|
| M2 Bayesian Regime | posterior_trending_up/down, posterior_ranging, posterior_volatile (4 dim) | **ゼロ行** | **欠落** |
| M3 σ-Clustering | vol_cluster (1 dim) | **ゼロ行** | **欠落** |
| M4 GLFT Fill Prob | fill_prob (1 dim) | **ゼロ行** | **欠落** |
| M5 VPIN | vpin_vol_sync (1 dim) | **ゼロ行** | **欠落** |

### §5.2 復元計画

```
Phase A: build_features.py に M2-M5 計算ロジック追加
  ├── M2: BayesianRegimeFilter.update(return) → 4-dim posterior
  ├── M3: VolatilityRegimeClassifier.classify(vol_ratio) → 1-dim
  ├── M4: fill_prob = A · exp(-k · δ) → 1-dim
  └── M5: compute_vpin_volume_sync() → 1-dim

Phase B: parquet 再生成 (7 列追加)
  └── 既存 OHLCV parquet + M2-M5 列

Phase C: LiteTradingEnv / FastIntradayEnvV456 の obs 対応
  ├── LiteTradingEnv: feature_columns に M2-M5 追加 (19-dim)
  └── FastIntradayEnvV456: 既存 88-dim の MTF/regime 列と整合確認
```

### §5.3 v454 MarketRegimeClassifier の活用

376# §2.3 で発見された `ztb.analysis.regime.market_regime_classifier.MarketRegimeClassifier` を Phase A の M2 計算に活用可能:

```python
# 既存コード (scripts/v454/verify_regime_distribution.py)
from ztb.analysis.regime.market_regime_classifier import MarketRegimeClassifier
classifier = MarketRegimeClassifier(config)
# → offline 訓練データの regime annotation に直接使用可能
```

---

## §6 FastIntradayEnvV456 評価 (376# 提案)

### §6.1 資産概要

| 属性 | 値 |
|---|---|
| ファイル | `ztb/trading/environment/fast_intraday_env_v456.py` |
| 行数 | 1062 |
| Observation dim | 88 (base:30 + MTF:27 + cyclical:6 + global:6 + regime:13 + account:6) |
| Action space | 2D (target_position, ttl_fraction) or 1D (position only) |
| Scaler | GroupedFeatureScaler (feature group 別正規化) |
| 市場理論 | Ichimoku "Lost Alpha Restoration" signal, CyclicalTimeFeatureExtractor |
| Regime | 13-dim one-hot: 4 regime × 3 confidence + vol_regime |
| Entry gate | Optional (external gate integration) |

### §6.2 LiteTradingEnv との比較

| 観点 | LiteTradingEnv (302行) | FastIntradayEnvV456 (1062行) |
|---|---|---|
| Obs dim | 12 | 88 |
| Action | 1D continuous [-1,+1] | 2D (position + TTL) or 1D |
| Scaler | なし (raw) | GroupedFeatureScaler |
| 市場理論 | なし | Ichimoku, regime, cyclical time |
| 要求データ | OHLCV (build_features.py で十分) | 30 base + 27 MTF + 13 regime 列 (**データパリティ必須**) |
| Sidecar 適性 | ◎ (軽量、即時利用可能) | △ (データパリティ復元後) |

### §6.3 判断

- **Phase 3.1**: LiteTradingEnv で実施 (データ要件なし、即時開始可能)
- **Phase 3.2+**: FastIntradayEnvV456 を復活候補として評価 (データパリティ復元後)
- **v459 教訓 (374# §13.1)**: 88-dim 訓練 → 5-dim 推論の次元不一致は致命的。**訓練と推論の env 一致が絶対条件**

---

## §7 SAC Live Observability 達成基準

### §7.1 最低要件

| 項目 | 確認方法 | 現状 (2026-03-11) |
|---|---|---|
| `sidecar_signal.json` 更新 | `stat cache/sidecar_signal.json` の mtime | **❌ NOT FOUND** |
| retrain 履歴 | `wc -l logs/sac_retrain_history.jsonl` | **❌ NOT FOUND** |
| fill_records sidecar fields | `jq '.sidecar_offset_bps' fill_records_*.jsonl \| grep -v null \| wc -l` | **❌ (signal なし)** |
| log sidecar entries | `grep -c 'sidecar' fill_test.log` | **❌ 0 件** |

**根本原因**: `sac_retrain_scheduler.py` は standalone CLI であり、fill_test から自動起動されない。初回手動実行が必要:
```bash
python scripts/v460/ml/sac_retrain_scheduler.py --config configs/v460/experiments/g2_sac_train.yaml --once
```

### §7.2 達成後のみ Phase 3.1 着手可能

上記 4 項目のうち 3 項目以上で positive confirmation が得られた場合に Phase 3.1 を開始する。
1 項目でも negative なら、まず sidecar の live 動作確認と修正を行う。

---

## §8 コード修正サマリ

### §8.1 初版 (377# 作成時)

| # | 修正内容 | ファイル | 根拠 |
|---|---|---|---|
| C1 | `training_metrics.total_timesteps` → `trade_count` にリネーム | `sac_retrain_scheduler.py` L794 | 375# §2.9: 誤ラベル |
| C2 | index.md 361/363 重複エントリ除去 + 369-372 欠番追加 | `docs/v460/index.md` | 375# §2.8 |
| C3 | 374# v3.0 改版: max_boost_bps 3.0→0.15、Phase 判定修正、§16 追加 | `374_ph3_design_*.md` | 375#/376# 全般 |
| C4 | 本文書 377# 作成 | `377_ph3_unified_direction_*.md` | 統合方針の明文化 |
| C5 | `DEFAULT_SIDECAR_BOOST_BPS` 0.3→0.15 に修正 | `sidecar_types.py` | 376# §3: 0.15 絶対上限 |

### §8.2 Phase 3.1 実装 (`82675725d`)

| # | 修正内容 | ファイル | 根拠 |
|---|---|---|---|
| C6 | `compute_sidecar_offset_bps_v2()` + `_shaping_fn()` 実装 | `sidecar_types.py` | 374# §3.1 |
| C7 | FillConfig 5 sidecar フィールド追加 | `fill_config.py` | 374# §10.1 |
| C8 | YAML `sidecar:` section parsing | `fill_config_parser.py` | 374# §10.1 |
| C9 | 5 sidecar keys hot-reload 対象追加 | `config_hot_reload.py` | 374# §10.1 |
| C10 | `_apply_sidecar_offset()` v1/v2 切替 | `cycle_gate_aggregator.py` | 374# §3.1 |
| C11 | sidecar validation (max_boost≤0.20 ceiling) | `fill_config_validation.py` | セルフレビュー横展開 |
| C12 | `import math` module-level 化 | `sidecar_types.py` | セルフレビュー |
| C13 | sidecar log 精度 `.2f`→`.4f` | `fill_cycle_executor.py` | セルフレビュー |
| C14 | `sidecar:` section YAML 追加 | `fill_test.yaml` | Phase 3.1 |
| C15 | 66 tests (55 core + 11 validation) | `test_374_proportional_boost.py` | 374# |

---

## §9 未解決事項と今後の作業

### §9.1 ブロッカー (Phase 3.2 前)

| 項目 | 工数見積 | 依存 |
|---|---|---|
| build_features.py M2-M5 計算ロジック追加 | 8-12h | M2: BayesianRegimeFilter, M3: VolatilityRegimeClassifier, M4: GLFT, M5: VPIN |
| parquet 再生成 (M2-M5 列追加) | 2-4h | build_features.py 完了後 |
| FastIntradayEnvV456 復活評価 | 4-8h | parquet M2-M5 列 |
| Feature contract 文書 (375# §6 P3) | 4h | build_features.py 調査完了後 |

### §9.2 Phase 3.1 残作業

| 項目 | 工数見積 | 状態 |
|---|---|---|
| ~~`compute_sidecar_offset_bps_v2()` 実装~~ | ~~2h~~ | ✅ 完了 (`82675725d`) |
| ~~fill_config / YAML / hot-reload 配線~~ | ~~2h~~ | ✅ 完了 (`82675725d`) |
| ~~テスト (66 件)~~ | ~~2h~~ | ✅ 完了 (55 core + 11 validation) |
| ~~バリデーション横展開~~ | ~~1h~~ | ✅ 完了 (fill_config_validation.py) |
| SAC scheduler 初回実行 | 0.5h | ⏳ 手動実行待ち |
| SAC live presence 確認 (§7.1) | — | ❌ 0/4 positive |

### §9.3 観察項目

- SAC α の存在確認 (maker uplift 指標) は Phase 3.1 実施中に並行で観察
- FastIntradayEnvV456 の 88-dim 一 parquet 列の整合性は §5.2 Phase C で精査
- v454 MarketRegimeClassifier の現行コードとの互換性は build_features.py 改修時に確認

---

## §10 批判的考察

### 10.1 本文書自体の限界

1. **「修正済み」は「正しい」を意味しない**: max_boost_bps を 0.15 に下げただけでは、SAC α の存在自体は証明されていない。0.15 でも無意味な可能性がある。
2. **FastIntradayEnvV456 の「復活」は容易ではない**: 1062 行の env が v456 当時のデータ形式に依存している可能性がある。現行 parquet との互換性は未検証。
3. **build_features.py の M2-M5 追加は工数過小見積もり**: BayesianRegimeFilter は path-dependent であり、単純な列追加では済まない (375# §5.2 指摘)。online と offline の計算結果一致を保証する必要がある。

### 10.2 Profit-first の再確認

375# §8 の核心メッセージ:

> **SAC を大きくする前に、SAC を live に存在させ、0.1-0.3bps の極小 modifier として 1bps でも増分価値を same-SHA で証明すること。**

本文書はこの方針に完全に準拠する。Phase 3.1 は「SAC が利益寄与できるか」の最小検証であり、その結果次第で Phase 3.2+ のロードマップ全体が変わりうる。

---

## 改版履歴

| 日付 | 版 | 内容 |
|---|---|---|
| 2026-03-18 | 1.0 | 初版: 375#/376# レビュー統合方針 + 軌道修正 + データパリティロードマップ + FastIntradayEnvV456 評価 |
| 2026-03-11 | 2.0 | Phase 3.1 コード実装完了反映 (`82675725d`): §4.1 チェックリスト更新, §7.1 SAC live 調査結果 (0/4), §8.2 実装サマリ追加, §9.2 残作業ステータス更新, M2-M5 build_features.py proxy 追加着手 |
| 2026-03-11 | 2.1 | 378# SAC scheduler 実行検証: SB3 stub 回避修正→v2.7.1ロード成功, Env/Model作成成功, `--once`履歴記録修正。P0 全7ファイル 377# 仕様完全準拠確認。§10 市場理論導入全タイムライン追記 |

---

## §10 市場理論タイムライン (035# 以降の段階的導入)

366# M1-M5 以前に fill_test へ段階的に導入された市場理論の全体像を以下に整理する。

### §10.1 初期フェーズ: 観測レイヤー構築 (035#–107#, 2/14–2/18)

| 文書 | 導入された市場理論 | 理論的根拠 | 実装先 |
|---|---|---|---|
| **035#** | **4状態レジーム分類** (trending/ranging/high_vol/unknown) + ヒステリシス + 信頼度ゲート | Hamilton (1989) Markov-Switching Model, Lo (2004) Adaptive Market Hypothesis | `regime_detector.py` |
| **054#** | **S1: Orderbook Imbalance** — `(bid_vol − ask_vol) / total ∈ [−1, +1]` による情報非対称性検知 | Glosten-Milgrom (1985): 情報トレーダーの存在が bid-ask spread を決定 | `maker_price.py` |
| **107#** | **VPIN 閾値トリガー** + **Volatility Guard** (σ + velocity) — 静的 time_filter → 動的ゲーティング | Easley-López de Prado-O'Hara (2012): VPIN = 情報非対称性連続指標 | `maker_price.py` |

### §10.2 在庫管理フェーズ (162#–228#, 2/25–3/1)

| 文書 | 導入された市場理論 | 理論的根拠 | 実装先 |
|---|---|---|---|
| **162#** | **Inventory Skewing (線形)** — `inv_net_imbalance × factor` で在庫偏重に応じた非対称 offset | Stoll (1978), Ho & Stoll (1981): MM在庫管理の基本原理 | `maker_price.py` |
| **200#** | **Velocity Modulation** — 短期 velocity が方向一致するかで boost/suppress | 動量ベースのサイド整合性 | `regime_detector.py`, `maker_price.py` |
| **226#** | **loss_boost 指数減衰** — `mult(t) = 1 + (M−1)·exp(−t/τ)` | Guéant-Lehalle-Fernandez-Tapia (2013): リスク調整の指数的減衰 | `maker_price.py` |
| **227#** | **EMA smoothed velocity** — bid-ask bounce ノイズフィルタ | EMA = IIR フィルタ理論 | `regime_detector.py` |
| **228#** | **Time-decay imbalance** — 古い fill 履歴の影響を指数減衰 | AS理論: 在庫ペナルティの時間的減衰 | `maker_price.py` |

### §10.3 理論統合フェーズ (257#–330#, 3/3–3/8)

| 文書 | 導入された市場理論 | 理論的根拠 | 実装先 |
|---|---|---|---|
| **257#/258#** | **AS Reservation Price** — `r = s − q·γ·σ²·τ` (在庫×ボラ連動 offset) + **VPIN Continuous** (二次関数ランプ) | Avellaneda-Stoikov (2008), Roll (1984): spread-based σ proxy | `maker_price.py`, `maker_microstructure.py` |
| **266#** | **GLFT τ動的化** `τ_eff = τ_base / vol_ratio` + **AS δ\*** + **Kyle λ** `λ = spread/(2·depth)` + **Amihud ILLIQ** `(spread/mid)/depth` | GLFT (2013), Kyle (1985), Amihud (2002) | `maker_microstructure.py` |
| **283#/284#** | **Buy-side AS 防御** — microprice 急落時の offset 拡大 | AS損失の非対称性に対する防御 | `maker_price.py` |
| **305#** | **Parkinson σ推定器** — `σ_P = ln(H/L)/(2√ln2)` + **PnL Execution Quality 分解** | Parkinson (1980), Kissell & Glantz (2003) | `maker_microstructure.py` |
| **306#** | **Microprice Side Selection** + **Queue Position** `P_fill = exp(−depth/lot)` + **Dynamic Cycle Interval** `interval = base × σ_ref/σ` | Gatheral microprice, Block Bootstrap | `maker_price.py`, `side_selector.py`, `fill_loop_orchestrator.py` |
| **324#** | **RSI Modulation** — ztb 既存実装の活用 (opt-in) | Wilder (1978): Relative Strength Index | `maker_price.py` |
| **330#** | **σ floor** — `σ=0` は AS δ\*/Kyle λ/Amihud を完全無効化するため最小フロア設定 | 数値安定性ガード | `maker_microstructure.py` |

### §10.4 市場理論提案・実装フェーズ (366#–378#, 3/10–3/11)

| 文書 | 導入された市場理論 | 理論的根拠 | 実装先 |
|---|---|---|---|
| **366# M1** | **Microprice L1→L5** Gatheral (2018) 指数減衰重み `w_k = exp(−0.5k)` | Gatheral (2018): multi-level microprice | `maker_price.py` |
| **366# M2** | **Bayesian Regime Filter** — Phase A: 4次元事後確率ベクトル + 遷移行列 | Hamilton (1989): Markov-Switching Bayesian 更新 | `bayesian_regime_filter.py`, `regime_detector.py` |
| **366# M3** | **σ-Clustering** — vol regime 自動検出 → offset キャリブレーション | K-Means/GMM ボラティリティクラスタリング | `regime_detector.py` |
| **366# M4** | **GLFT Fill Probability** — `A(δ) = A·exp(−kδ)` 到着率モデル + 動的 k | GLFT (2013): fill probability model | `fill_probability_model.py`, `maker_microstructure.py` |
| **366# M5** | **Volume-Sync VPIN** — Volume-bucket 化 + `compute_vpin_volume_sync` | Easley-López de Prado (2012): Volume-synchronized PIN | `feature_enricher.py` |
| **366# T5** | **Welford Online Variance** — O(1) per update (O(n)→O(1)) | Welford (1962): online variance | `regime_detector.py` |
| **378#** | **SB3 stub 回避** — `sys.modules`/`sys.path` 操作で本物の SB3 v2.7.1 をロード | — | `sac_retrain_scheduler.py` |

### §10.5 fill_test offset パイプライン全景 (378# 時点, 13+ ステージ)

```
compute(side, spread, mid_price, ...)
  │
  ├── [096#] base_offset_ratio (side別)
  ├── [162#] Inventory Skewing              ← q × factor (Ho & Stoll 1981)
  ├── [088#] Sell Offset Floor
  ├── [258#] AS Reservation Shift           ← q·γ·σ²·τ (Avellaneda-Stoikov 2008)
  │    ├── [305#] Parkinson σ               ← ln(H/L)/(2√ln2) (Parkinson 1980)
  │    ├── [266#] GLFT τ動的化              ← τ_eff = τ_base/vol_ratio (GLFT 2013)
  │    └── [266#] AS δ* floor               ← γσ²τ + (2/γ)ln(1+γ/k)
  ├── [163#] Regime Boosts (5段)
  ├── [163#] Spread Adaptive
  ├── [266#] Kyle λ                         ← spread/(2·depth) (Kyle 1985)
  ├── [266#] Amihud ILLIQ                   ← (spread/mid)/depth (Amihud 2002)
  ├── [366# M4] GLFT Fill Probability       ← A·exp(−kδ) (GLFT 2013)
  ├── [107#] Volatility Guard               ← velocity + VPIN continuous (258#)
  ├── [054#] Imbalance Risk                 ← OB imbalance (Glosten-Milgrom 1985)
  ├── [226#] Loss Boost                     ← 指数減衰 (GLFT 2013)
  ├── [100#] FastFillDefense
  ├── [283#] Buy AS Guard                   ← microprice 急落防御
  ├── [306#] Offset Ceiling                 ← 無制限膨張防止
  ├── [365#] Sidecar Offset (v2)            ← SAC signal × max_boost_bps × shaping
  └── Finalize (spread guard + 価格組立)
```

### §10.6 SAC Scheduler 動作検証結果 (378#, 2026-03-11)

| 項目 | 結果 |
|---|---|
| SB3 ロード | ✅ v2.7.1 (site-packages) — stub 回避修正適用 |
| Data ロード | ✅ `btc_jpy_1m_full_registry_features.parquet` (1,216,930行 × 77列) |
| Rolling window | ✅ 7d = 10,080行 (Train: 8,064 / Val: 2,016) |
| 環境作成 | ✅ 12特徴量, continuous_1d action space, coincheck profile |
| モデル作成 | ✅ Cold-start SAC model |
| `--once` 履歴記録 | ✅ `_append_history()` 追加済み |
| 50,000 step 学習 | ⏳ 実行中 (計算量大 — 結果待ち) |
| `sidecar_signal.json` | ⏳ OOS gate 通過時のみ生成 |

#### SAC Scheduler 修正内容

1. **SB3 stub 回避**: `sys.modules` から `stable_baselines3.*` 全モジュール除去 + `sys.path` からプロジェクトルート一時除外 → 本物の SB3 v2.7.1 をロード。`__version__` 属性検証で stub 混入を検出
2. **`--once` mode 履歴記録**: `main()` 内で `_append_history()` を呼ぶよう修正 (従来は `run_scheduler` ループ内のみ)

### §10.7 P0 仕様準拠確認 (全7ファイル ✅)

| ファイル | sidecar 機能 | 準拠 |
|---|---|---|
| `sidecar_types.py` | `compute_sidecar_offset_bps_v2()`, `_shaping_fn()`, `max_boost_bps=0.15` | ✅ |
| `fill_config.py` | 5 フィールド (enabled, max_boost_bps, dead_zone, shaping, use_v2) | ✅ |
| `fill_config_parser.py` | sidecar YAML parse (5 keys) | ✅ |
| `config_hot_reload.py` | 5 sidecar keys hot-reload 対応 | ✅ |
| `cycle_gate_aggregator.py` | v1/v2 切替 + `sidecar_enabled` ガード | ✅ |
| `fill_config_validation.py` | `max_boost_bps ≤ 0.20` ceiling | ✅ |
| `fill_test.yaml` | `sidecar:` セクション (0.15 / 0.10 / linear / v2) | ✅ |

### §10.8 次のステップ

1. SAC scheduler 50,000 step 学習完了後の OOS 結果確認
2. `sidecar_signal.json` 生成確認 → live presence 4 項目の充足
3. OOS gate 不通過の場合: `min_gross_roi` / `min_trade_count` 閾値の検討
4. fill_test 実運用での sidecar_offset_bps の観測
