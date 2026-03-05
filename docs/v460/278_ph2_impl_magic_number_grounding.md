# 278# マジックナンバー根拠化 + 271#-277# セルフレビュー

## 概要

v460 プロダクションコード内のマジックナンバー (已むを得ず固定値で運用していたもの) を
理論的根拠ある計算・config・名前付き定数に置き換える改修。
併せて 271#-276# の全変更をセルフレビューし、発見した HIGH バグ (B1) を修正。

## 調査結果

### マジックナンバー調査 (35 件検出)

| 優先度 | 件数 | 対応 |
|--------|------|------|
| 高優先 | 6 | config 化 + 導出 |
| 中優先 | 7 | 導出 + 名前付き定数 |
| 低優先 | 5 | 名前付き定数/現状維持 |

### セルフレビュー結果 (271#-276#)

| カテゴリ | HIGH | MEDIUM | LOW |
|----------|------|--------|-----|
| バグ | 1 | 0 | 1 |
| 一貫性 | 0 | 1 | 1 |
| テストカバレッジ | 0 | 1 | 2 |
| 型安全 | 0 | 3 | 1 |
| DRY | 0 | 0 | 1 |
| **計** | **1** | **5** | **6** |

## 改修内容

### A. FillTestConfig 新規フィールド (5 件)

| フィールド | デフォルト | 理論的根拠 |
|-----------|-----------|-----------|
| `phantom_detection_sleep_multiplier` | 3.0 | Avellaneda-Stoikov §3.2 在庫リスク待機 |
| `halt_persist_interval` | 10 | halt_sleep × N 回で state save (再起動巻き戻し防止) |
| `stop_condition_check_interval` | 30 | 30 × cycle_interval = 1h の監視周期 |
| `fallback_duration_sec` | 3600.0 | Kyle (1985) price discovery 収束に 1h |
| `unknown_regime_max_consecutive` | 10 | Hamilton (1989) regime 再評価猶予 = 20 分 |

### B. __post_init__ 構造的整合性バリデーション (3 件)

1. `max_cycle_sleep_sec >= cycle_interval_sec × halt_sleep_multiplier` (halt キャップ防止)
2. `order_timeout_sec <= cycle_interval_sec` (次サイクル遅延防止)
3. `lock_stale_heartbeat_sec >= lock_heartbeat_period_sec × 3` (誤 stale 判定防止)

### C. orchestrator マジックナンバー → config/導出 (6 箇所)

| 旧値 | 新参照 | 説明 |
|------|--------|------|
| `3600.0` (L703, 2 箇所) | `config.fallback_duration_sec` | 停止条件 fallback 持続時間 |
| `% 30` (L1252) | `config.stop_condition_check_interval` | 停止条件モニター間隔 |
| `_HALT_PERSIST_INTERVAL = 10` (L1393) | `config.halt_persist_interval` | halt state save 間隔 |
| `multiplier=3.0` (L1559) | `config.phantom_detection_sleep_multiplier` | phantom sleep 倍率 |
| `[-100:]` + `>= 10` (L710-712) | `sell_dynamic_kill_window × 2` + `min_adapt_samples // 5` | PnL 平均計算の窓と最小サンプル |
| `>= 10 and % 10` (L2189) | `quiescence_gate_blocks_threshold // 2` | gate block ログ間隔 |

### D. cycle_gate_aggregator (1 箇所)

`UNKNOWN_REGIME_MAX_CONSECUTIVE = 10` → `config.unknown_regime_max_consecutive` から __init__ で設定。
クラス属性として後方互換を維持 (既存テストが `.UNKNOWN_REGIME_MAX_CONSECUTIVE` を参照)。

### E. MCB 改善 (3 箇所)

1. `maxlen=720` → `86400 / config.check_call_interval_sec` で導出 (cycle_interval 変更に追従)
2. `min_samples = 10` → `_MIN_SIGMA_SAMPLES = 10` (名前付きクラス定数)
3. `default_pct * 0.1` → `_SIGMA_FLOOR_RATIO = 0.1` (名前付きクラス定数)

### F. B1 warmup TZ 不一致修正 (HIGH severity)

**問題**: `_warmup_daily_drawdown_from_records` が `datetime.now(timezone.utc)` で日付判定していたが、
`DailyDrawdownGuard._today()` は `_day_reset_tz` (デフォルト JST UTC+9) を使用。
JST 0:00–9:00 (= UTC 15:00–24:00) に再起動すると日付判定が不一致になり、
当日分 PnL が DD guard に投入されず halt が遅延するリスクがあった。

**修正**: warmup で `guard._today()` および `guard._day_reset_tz` を使用し、
DD guard と完全に同一の TZ で日付境界を判定する。

### G. 定数命名・コメント改善

- `_STATE_SAVE_INTERVAL_SEC = 300.0` に理論的導出コメント追加
- `_TRADES_HEALTH_*` パラメータを名前付きローカル定数化
- `_effective_sleep` docstring 更新 (config 化済みパラメータの反映)

## テスト

- 新規: `test_277_magic_number_grounding.py` (34 tests)
- 修正: `test_169` (cycle_interval=60 + order_timeout=45 整合)、`test_181` (config mock 追加)、
  `test_276` (max_cycle_sleep_sec 整合)、`test_fill_test_config` (cycle_interval=60 整合)
- 結果: **3827 passed, 32 skipped** (前回 3793 → +34)

## 変更ファイル

| ファイル | 変更種別 |
|---------|---------|
| `scripts/v460/lib/fill_config.py` | 新規フィールド 5 件 + __post_init__ 検証 3 件 + flat_keys 5 件 |
| `scripts/v460/lib/fill_loop_orchestrator.py` | マジックナンバー → config 参照 6 箇所 + B1 TZ fix |
| `scripts/v460/lib/cycle_gate_aggregator.py` | UNKNOWN_REGIME config 化 |
| `scripts/v460/lib/micro_circuit_breaker.py` | maxlen 導出 + 名前付き定数 |
| `scripts/v460/run_fill_test.py` | MCBConfig に check_call_interval_sec 追加 |
| `configs/v460/fill_test.yaml` | 新規 6 フィールド追加 |
| `tests/unit/v460/test_277_magic_number_grounding.py` | 新規 34 tests |
| `tests/unit/v460/test_169_config_hot_reload.py` | 整合性修正 |
| `tests/unit/v460/test_181_ev_weighted_stop_conditions.py` | config mock 追加 |
| `tests/unit/v460/test_276_blocking_policy_dry.py` | max_cycle_sleep_sec 整合 |
| `tests/unit/v460/test_fill_test_config.py` | cycle_interval 整合 |
