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
