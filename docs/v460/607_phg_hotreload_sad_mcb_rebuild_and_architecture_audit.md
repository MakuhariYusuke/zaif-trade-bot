# 607# SAD/MCB hot-reload 再構築 + アーキテクチャ監査

- **日付**: 2026-03-25
- **前提**: 606# で SAD/MCB を `enabled: true` に変更。hot-reload 時に enabled 変更が反映されない P0 バグを発見・修正。加えて多角的アーキテクチャ監査を実施。

---

## §1 P0: hot-reload 時の SAD/MCB 再構築

### 問題

`config_hot_reload.py` は FillConfig の各フィールドを差分更新するが、`_COMPONENT_REBUILD_PREFIXES` に MCB/SAD が未登録だった。YAML で `mcb_enabled: false` に変更しても、ランタイムの `MicroCircuitBreaker` / `SpreadAnomalyDetector` オブジェクトは古い設定のまま動作し続ける。

**影響**: 606# で有効化した SAD/MCB を hot-reload で一時無効化する運用が不可能。プロセス再起動が必要。

### 修正

| ファイル | 変更内容 |
|---|---|
| `config_hot_reload.py` | `_COMPONENT_REBUILD_PREFIXES` に `"mcb_": "_rebuild_mcb"`, `"sad_": "_rebuild_sad"` を追加 |
| `config_hot_reload.py` | `_HotReloadableRunner` Protocol に `_rebuild_mcb()`, `_rebuild_sad()` メソッド宣言を追加 |
| `run_fill_test.py` | `_rebuild_mcb()` メソッド実装 — `export_state()` で状態退避 → 新 MCBConfig で再構築 → `import_state()` で復元 |
| `run_fill_test.py` | `_rebuild_sad()` メソッド実装 — 同様のパターン |

### 設計方針

- `export_state()` / `import_state()` で状態継承し、warm-up データ喪失を防止
- lazy import で循環参照回避
- 既存の `_rebuild_daily_drawdown_guard` パターンに準拠

---

## §2 P1: SAD warmup ウィンドウのログ可視化

### 問題

SAD は `_spread_buffer` に 3 サンプル蓄積するまで常に `NORMAL` を返す（約 15 秒間）。この間はスプレッド異常検知が無効で、プロセス起動直後の脆弱性ウィンドウとなる。

### 修正

`spread_anomaly_detector.py` の早期リターン箇所に `logger.debug("[607#] SAD warmup: buffer=%d/3, spread protection inactive")` を追加。

### 判断

15 秒の warmup はスプレッド中央値算出に必要な最小サンプル数（3）であり、短縮はノイズ増加のリスクがある。ログによる可視化で十分な対応とした。

---

## §3 アーキテクチャ広域監査: 誤警報の分析

606# レビューの延長として、fill_test 全体のリスク広域スキャンを実施。以下の懸念事項は詳細調査の結果 **非問題 (Non-Issue)** と判定。

### §3.1 DD halt / MCB HALT / SAD FROZEN 時の未決済注文リーク

**初期評価**: CRITICAL — halt 発動時に `cancel_all_orders()` のような呼び出しがなく、オープンオーダーが市場に残る可能性。

**調査結果**: **NON-ISSUE (同期ループアーキテクチャ)**

fill_test は **同期ループ (A パターン)** で動作する:
```
cycle start → check circuit breakers → place order → wait fill/cancel → record → cycle end
```
各サイクルは注文の完了（約定またはキャンセル）を待ってから終了する。`_check_circuit_breakers` はサイクル開始時（= live order が存在しない時点）に実行されるため、halt 中に孤立するオープンオーダーは構造的に発生しない。

### §3.2 価格データの鮮度問題

**初期評価**: HIGH — WebSocket キャッシュからの stale データで注文が入る可能性。

**調査結果**: **NON-ISSUE (REST API アーキテクチャ)**

`self.adapter.get_orderbook()` は毎サイクル REST API を直接呼び出す。WebSocket 常時接続のキャッシュではないため、取得データは常に最新。

### §3.3 セルフトレード防止

**初期評価**: MEDIUM

**調査結果**: **解決済み (postonly_guard)**

`_submit_order_phase` 内の `postonly_guard` で、buy ≥ best_ask または sell ≤ best_bid の場合はスキップする仕組みが稼働中。

### §3.4 並行アクセス競合

**初期評価**: MEDIUM

**調査結果**: **NON-ISSUE (シングルスレッド)**

fill_test は単一スレッドで逐次実行。race condition は構造的に発生しない。

---

## §4 テストドリフト検知: stale allowlist エントリ除去

`test_336_yaml_code_drift_prevention.py` の `KNOWN_YAML_OVERRIDES` に、606# で YAML 値をコード値に合わせた結果 allowlist が不要になった 2 項目が残存していた。

- `skip_gate_model_path_buy` — YAML とコード値が一致済み
- `skip_gate_model_path_buy_long` — YAML とコード値が一致済み

→ 除去。

---

## §5 既存テスト修正: observation_space 次元不整合

`tests/unit/trading/test_environment.py` の `test_correlation_reduction_respects_target_count` が env_tracker による +3 内部次元を考慮していなかった。

- 修正: `observation_space.shape[0]` の期待値を `5 + internal_dim` に変更

---

## §6 アーキテクチャ洞察

### fill_test 同期ループの安全性

fill_test の **A パターン（同期ループ）** は、各サイクルが「注文→完了→記録→次」と直列実行されるため、以下が構造的に保証される:

1. **circuit breaker チェック時に live order なし** — halt/freeze で注文キャンセル不要
2. **価格データは常に最新** — REST API 毎回呼び出し
3. **競合状態なし** — シングルスレッド + 同期 I/O

これは非同期 / WebSocket 駆動のマーケットメイカーとは根本的に異なる安全特性であり、多くの「汎用的」リスク懸念が fill_test には該当しない理由となっている。

---

## §7 変更一覧

| ファイル | 変更行数 | 概要 |
|---|---|---|
| `scripts/v460/lib/config_hot_reload.py` | +4 | MCB/SAD rebuild prefix + Protocol 宣言 |
| `scripts/v460/run_fill_test.py` | +33 | `_rebuild_mcb()` / `_rebuild_sad()` 実装 |
| `scripts/v460/lib/spread_anomaly_detector.py` | +4 | warmup debug ログ |
| `tests/unit/v460/test_336_yaml_code_drift_prevention.py` | -2 | stale allowlist 除去 |
| `tests/unit/trading/test_environment.py` | +2 | observation_space 次元修正 |
