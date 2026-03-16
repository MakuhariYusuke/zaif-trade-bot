# 355# API 呼出し重複排除 + ph3 障害分析

> **種別**: impl (ph2) + rpt (ph3 準備)  
> **フェーズ**: ph2 G1.1-exec (レイテンシ最適化) / ph3 先行調査  
> **前提**: 354# 品質改善完了, 広域探索 6 エリアスキャン  
> **日付**: 2026-03-09  
> **コミット**: `7ce3aa1bb`

---

## §1 背景

Bot fill test 稼働中 (168h 蓄積中) の並行作業として、6 エリア横断スキャンを実施。
最もインパクトの大きい **L-1/L-2: API 呼出し重複排除** を選定・実装。

---

## §2 API 呼出しフロー分析

### サイクル内 API 呼出しマップ (変更前)

| # | タイミング | API | depth/limit | 削減対象 |
|---|---|---|---|---|
| A | imbalance pre-fetch | `get_orderbook` | depth=5 | 基準 |
| B | trades recorder | `get_recent_trades` | limit=100 | 基準 |
| C | compute() maker_price | `get_orderbook` | depth=1 | ✅ A のキャッシュで既に不要 |
| D | SkipGate trades | `get_recent_trades` | limit=50 | ✅ **B のスーパーセット** |
| E | SkipGate OB | `get_orderbook` | depth=5 | ✅ **A と同一 depth** |
| F | postonly guard | `get_orderbook` | depth=1 | ⚠ 鮮度要件あり (50-200ms 後) |

### スーパーセット関係

- OB: A (depth=5) ⊇ C (depth=1) ⊇ F (depth=1)
- Trades: B (limit=100) ⊇ D (limit=50)

---

## §3 実装: L-1/L-2 prefetch 共有

### 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/lib/skip_gate_evaluator.py` | `evaluate()` に `prefetched_ob: OrderBookSnapshot \| None`, `prefetched_trades: object \| None` 追加 (keyword-only, default=None で後方互換) |
| `scripts/v460/lib/fill_cycle_executor.py` | `_prefetched_trades` 変数追加、`_evaluate_skip_gate()` 経由で `_last_ob_snapshot` と `_prefetched_trades` を注入 |

### 効果

| 指標 | Before | After |
|---|---|---|
| OB fetch / cycle (発注前) | 3 回 (A+C+E) | 1 回 (A のみ) |
| Trades fetch / cycle | 2 回 (B+D) | 1 回 (B のみ) |
| 推定レイテンシ削減 | — | **200-500ms / cycle** |

### 設計判断

1. **F (postonly guard) は共有対象外**: 発注直前のスプレッド確認であり、A から 50-200ms 経過後の鮮度要件がある。stale OB で taker 約定するリスクを回避
2. **後方互換**: prefetched パラメータは全て `None` デフォルト。None 時は従来通り adapter から fetch → テスト影響ゼロ
3. **trades limit 差異**: recorder (limit=100) ⊃ SkipGate (limit=50)。SkipGate 側 `_normalize_recent_trades()` で切り詰められるため問題なし

---

## §4 セルフレビュー

### ✅ 良い点

1. **後方互換性**: keyword-only + default=None で既存呼出し全てそのまま動作
2. **障害分離**: trades fetch 失敗時 `_prefetched_trades = None` のまま → SkipGate は従来パスで自力 fetch
3. **OB prefetch 失敗時**: `_last_ob_snapshot` が前回値のまま → SkipGate にも前回値 or None が渡る → SkipGate は None なら自力 fetch
4. **型安全**: `OrderBookSnapshot | None` で明示型付け (skip_gate_evaluator 側)

### ⚠ 注意点・潜在リスク

| # | 項目 | リスク | 対策提案 |
|---|---|---|---|
| R-1 | OB staleness | imbalance pre-fetch → SkipGate 評価まで 10-30ms。この間の板変動は無視される | 現行サイクル間隔 (5-30s) と比較して無視可能。ログに prefetch 再利用を示す指標追加で監視可 |
| R-2 | Trades limit 超過分のデータ | recorder limit=100 の結果を渡すが、SkipGate は limit=50 を期待。`_normalize_recent_trades` は件数で切り詰めない | `_normalize_recent_trades` のコードを確認 → 全件を使用する設計。VPIN 等の集約値は件数増で精度向上方向なので問題なし |
| R-3 | `_ob_fetch_total_count` カウンタ | prefetch 使用時も `_ob_fetch_total_count += 1` される。実際の API 呼出しと乖離 | 監視用カウンタ。ログ解析時に「OB fetch 回数」の意味が「OB 使用回数」に変わることを認識 |
| R-4 | `_maker_price._last_ob_snapshot` 直接参照 | Mixin パターンで内部属性をクロス参照。カプセル化違反 | 既存パターンの延長 (L393-398 で既に直接参照)。getter 化は将来課題 |

### 判定: 即座に要修正の問題なし

R-1〜R-4 は全て低リスクで、現行設計の延長線上。

---

## §5 ph3 推進時の障害分析

### ブロッカー一覧

| # | ブロッカー | 重大度 | 概要 | 推定工数 |
|---|---|---|---|---|
| **B1** | `g2_sac_train.yaml` 不在 | **High** | `configs/v460/experiments/g2_sac_train.yaml` が存在しない。`task_sac_train.py` L12 が参照 | 0.5 日 |
| **B2** | 特徴量次元体系の断絶 | **High** | v460 の 10 microstructure 特徴量 (fill 品質評価用) と FeatureRegistry の ~191 特徴量 (RL 訓練用) の間に adapter/injection 機構がない。SAC に適した特徴量セットの定義と統合が必要 | 2-3 日 |
| **B3** | feature_columns の env 未注入 | **Medium** | `task_sac_train.py` L162-168 で `feature_columns` を構築するが `EnvironmentConfig.feature_names` にセットしない。env は DataFrame 全カラムを使用 | 0.5 日 |
| **B4** | multi-seed ラッパー不在 | **Medium** | 4-seed × 50K steps の結果を G2 judgment 形式 (`seed_results`, `convergence`) に集約する orchestrator が未実装 | 1-2 日 |
| **B5** | SACTrainer 3 重実装 | **Low** | `unified_trainer/`, `ztb/training/sac_trainer.py`, `trainers/sac_trainer.py` の 3 つが並存。`task_sac_train.py` は SB3 `SAC` を直接使用しており unified_trainer の checkpoint/callback が利用不可 | 設計判断 |

### 355# 変更の ph3 影響: なし

- `prefetched_ob` / `prefetched_trades` は keyword-only + default=None
- SAC 訓練パスでは `SkipGateEvaluator.evaluate()` を呼ばない (fill_test 専用)
- `HeavyTradingEnv` は `SkipGateEvaluator` に依存しない

### ph3 推進ロードマップ (リスク順)

```
B2 特徴量体系統合 ─┐
                    ├→ B1 g2_sac_train.yaml → B3 env 注入修正 → B4 multi-seed → G2 gate
B5 trainer 選定   ─┘
```

**クリティカルパス**: B2 (特徴量体系) が最大のボトルネック。v460 の microstructure 特徴量を FeatureRegistry に統合するか、SAC 専用のデータパイプラインを別途構築するかの設計判断が必要。

---

## §6 広域探索で特定した他の施策候補

### Bot 稼働中に実行可能 (bot 非干渉)

| # | 項目 | 工数 | インパクト |
|---|---|---|---|
| M-2 | Discord/Slack 通知 (DD halt, kill 発動) | 小 | 運用安全性 |
| E-2 | tail_loss → daily_health_check 統合 | 小 | 大損失早期検知 |
| D-1 | trades ギャップレポート自動生成 | 小 | データ品質 |
| T-1 | configs/ レガシー整理 (v1-v450) | 小 | 見通し改善 |

### 中期 (restart 時デプロイ)

| # | 項目 | 工数 | インパクト |
|---|---|---|---|
| M-4 | cycle_time メトリクス (API latency) | 中 | ボトルネック可視化 |
| M-1 | Prometheus 実体化 (shim→real) | 中 | Grafana 接続 |

---

## §7 検証

- v460 全 2,512 テスト pass
- 既存 5 fail (`test_260_compute_extract_regime_split`) は pre-existing (test_260 が参照する `_regime_boost_ranging` メソッドが 260# 後に削除済み)
