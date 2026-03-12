# 296# P2/P3 クリーンアップ + v459 資産調査

> **日付**: 2026-03-06  
> **種別**: `ph2_impl` (実装) + 調査  
> **コミット**: `0a80cea9d`  
> **前提**: 295# hot-reload comprehensive coverage  
> **テスト**: 3930 passed, 32 skipped

---

## §1 実施内容

### §1.1 B-14: except 変数なしログ追加 (280# §2.4 P3)

**課題**: `except Exception:` で例外変数を捕捉していない箇所が16箇所。デバッグ時にどの例外が発生したか不明。

**対応**: 全16箇所に `as e` を追加し、ログメッセージに `%s` で例外情報を出力。

| カテゴリ | 対応 | ファイル数 | 箇所数 |
|---------|------|-----------|--------|
| サイレント（ログなし） | debug ログ新設 | 2 (manifest.py, ab_judgment.py) | 4 |
| 既存 exc_info=True | `as e` + `%s` 追加 | 8 | 12 |

**変更ファイル**: fill_loop_orchestrator.py, lock_manager.py, manifest.py, fill_test_cli.py, fill_cycle_executor.py, event_logger.py, ob_utils.py, pnl_measurer.py, resilience.py, ab_judgment.py

### §1.2 F-2: cancel_reason Literal 型化 (170# §10.5 P2)

**課題**: cancel_reason に CR 定数を使わず文字列リテラルを直接使用している箇所が複数存在。新規追加された cancel_reason が CancelReason Literal / AUDIT_CANCEL_REASONS に未登録。

**対応**:

#### 新規 CR 定数の追加 (cancel_reasons.py)

| 定数名 | 値 | 由来 |
|--------|-----|------|
| `FORCED_BUY_DELAY` | `"forced_buy_delay"` | 294# |
| `DEGRADED_LIQUIDATION_DUTY_SKIP` | `"degraded_liquidation_duty_skip"` | 234# |
| `ONE_SIDED_COOLDOWN_SKIP` | `"one_sided_cooldown_skip"` | 234# |
| `ONE_SIDED_FREEZE_SKIP` | `"one_sided_freeze_skip"` | 234# |
| `SKIP_GATE_RULE_UNKNOWN_SELL` | `"skip_gate_rule_unknown_sell"` | 296# |

#### 文字列リテラル → CR 定数への置換

| ファイル | 箇所数 | 内容 |
|---------|--------|------|
| fill_loop_orchestrator.py | 5 | forced_buy_delay, one_sided_*, degraded_*, toxicity_* |
| fill_cycle_executor.py | 8 | orderbook_*, post_only_*, insufficient_*, minimum_*, api_* |
| order_monitor.py | 4 | stale_adverse_drift, stale_skip_gate_blocked, stale_reprice_failed (set/比較) |
| skip_gate_evaluator.py | 1 | skip_gate_rule_unknown_sell |

**追加対応**: order_monitor.py, skip_gate_evaluator.py に `from scripts.v460.lib import cancel_reasons as CR` import 追加。

### §1.3 B-17: MCB/SAD 型安全化 (280# §2.6 LOW)

**課題**: `_mcb: object | None` / `_sad: object | None` — Protocol 未使用、属性アクセスが型未検証。

**対応**:
- `TYPE_CHECKING` ブロックで `MicroCircuitBreaker`, `SpreadAnomalyDetector` をインポート
- `_mcb: MicroCircuitBreaker | None`, `_sad: SpreadAnomalyDetector | None` に型注釈更新
- circular import 回避: TYPE_CHECKING ガード利用

### §1.4 スキップ項目

| タスク | 理由 |
|--------|------|
| C-9: `_opposite_side` 共通ユーティリティ昇格 | 調査の結果、fill_loop_orchestrator.py 内のみで完結。重複なし→不要 |
| D-1: index.md ファイル名37件リネーム | 工数大 (15分+) / インパクト低 → 後回し |

---

## §2 v459 資産調査

### §2.1 結論

**直接移植は不要。v460 は v459 を一切 import していない。**

v459 のコア成果物 (`CausalOnlineScaler`, `CausalGroupedFeatureScaler` 等) は ztb/ 共通層に既に存在し、v460 から利用済み。v459 の God Object (`run_phase_c.py` 1,277行) やコピペ増殖構造は、v460 の `run_experiment.py` + `lib/` + YAML 宣言的管理で根本解消済み。

### §2.2 参考価値のある資産

| 資産 | パス | 活用方法 |
|------|------|---------|
| **gate_c3_comparison.py** | scripts/v459/ | scipy不要の Mann-Whitney U + Cliff's Delta + Holm-Bonferroni → **F-4 で直接参考** |
| Phase E 診断ロジック | scripts/v459/ | IC多面評価 → SkipGate品質ゲート改善 |
| TTL 概念 | scripts/v459/ | 最小保有期間 → ポジション管理改善のヒント |
| subprocess メモリリーク回避 | scripts/v459/ | 長時間稼働堅牢化のパターン |

### §2.3 保守継続すべきテスト

| テスト | パス | 理由 |
|--------|------|------|
| test_causal_scaler_v459.py | tests/unit/v459/ | ztb コアの回帰テスト |
| test_reporter_v459.py | tests/unit/v459/ | ztb コアの回帰テスト |
| test_p03_cost_double_count.py | tests/unit/v459/ | コスト二重計上バグ防止 |

---

## §3 F-4 / G-2 実施前分析

### §3.1 F-4: StatisticalValidator A/B テスト統合

**結論: 丸ごと統合は不適切。cherry-pick 方式を推奨。**

| 現行 (ab_judgment.py) | 欠落 | 推奨ソース |
|----------------------|------|-----------|
| Welch's t 検定 | ✅ あり | — |
| Cohen's d | ✅ あり | — |
| **Holm-Bonferroni** | ❌ なし | **gate_c3_comparison.py** (pure Python) |
| **Mann-Whitney U** | ❌ なし | **gate_c3_comparison.py** (pure Python) |
| **Cliff's Delta** | ❌ なし | **gate_c3_comparison.py** (pure Python) |
| Bootstrap CI | ❌ なし | ab_test/analyzer.py |
| 多重検定補正 | ❌ なし | statistical_validator.py or gate_c3 |

**statistical_validator.py 自体の問題**:
- SAC/リターン系列前提 → fill_test の FillRecord (bps) とインタフェース不一致
- `statsmodels` 依存 → gate_c3 の pure Python 版のほうが依存軽量
- `Any` 型多用 → v460 型安全方針に反する

**推奨実装**: gate_c3 から MannWhitneyU + Cliff's Delta + Holm-Bonferroni を ab_judgment.py に移植。工数 0.8日、ライブ影響なし。

### §3.2 G-2: 168# P3 残タスク

| 項目 | 現状 | 判定 |
|------|------|------|
| **P3-4** UnifiedTrainer | 2,835L → 2,227L (22%削減済)。まだ god object だが機能的に安定 | **v461 繰越維持** |
| **P3-5** pnl120→pnl30 統一 | 「保持期間延長」施策と同義。A/B テストデータが必要 | **別線管理** |
| **P3-6** asyncio.to_thread 残5 | coincheck 9箇所 + bitflyer 4箇所 対応済み。残0 | **✅ 完了** |
| **CircuitBreaker** 統合 | live_tr では統合済、**fill_loop は未統合** | P1 即時実装可 |
| **DrawdownController** 統合 | RiskManager 内部のみ、**fill_loop は未統合** | P1 即時実装可 |

**即時実行推奨**: F-4 (Holm + Mann-Whitney cherry-pick) が最も費用対効果が高い。CircuitBreaker / DrawdownController は取引ロジックへの影響が大きいため fill test の安定期に実施推奨。

---

## §4 次アクション

| 優先度 | アクション | 工数 | 前提 |
|--------|-----------|------|------|
| **P1** | F-4: gate_c3 から Holm + Mann-Whitney + Cliff's Delta を ab_judgment.py に移植 | 0.8日 | なし |
| P2 | CircuitBreaker の fill_loop 統合 | 0.3日 | fill test 安定期 |
| P2 | DrawdownController の fill_loop 統合 | 0.5日 | fill test 安定期 |
| P3 | D-1: index.md ファイル名リネーム 37件 | 0.2日 | なし |
