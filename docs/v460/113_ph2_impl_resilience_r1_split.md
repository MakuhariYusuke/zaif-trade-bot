# 113# Resilience 統合 + R1 God Method 分割

| key | value |
|-----|-------|
| type | impl (実装完了報告) |
| scope | ph2 fill_test 耐障害性 + コード品質 |
| date | 2026-02-19 |
| commit | `cfb0d3a93` |
| parent | `112_phg_rev_111_legacy_asset.md`, `106_ph2_fix_refactoring_r1_r10.md` |
| purpose | 112# レビュー指摘の即時実施 + 106# R1 God method 分割 |

---

## §0 Executive Summary

112# レビュー §3.1 Tier-1/Tier-2 の 5 項目すべてと、111# §5 改訂推奨 5 件を一括実施。

| 区分 | 内容 | 結果 |
|------|------|------|
| **Tier-1** | CircuitBreaker API ガード | ✅ 実装済 |
| **Tier-1** | HealthMonitor 定期監視 + GC | ✅ 実装済 |
| **Tier-1** | R1 run_single_cycle 分割 | ✅ 755→307行 (59%削減) |
| **Tier-2** | StatePersistence 状態保存/復元 | ✅ 実装済 |
| **111# §5** | ドキュメント修正 (dead判定, SLO, Tier) | ✅ 反映済 |

テスト: **857 全 PASS** (835 既存 + 22 新規)

---

## §1 新規モジュール: `scripts/v460/lib/resilience.py` (231行)

### §1.1 設計方針

- ztb 既存資産を **ファサードパターン** で薄くラップ
- fill_test 固有のデフォルト値をモジュール側で保持
- `run_fill_test.py` の肥大化を防ぎつつ、必要な機能を注入

### §1.2 提供コンポーネント

#### `create_api_circuit_breaker()`

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `failure_threshold` | 5 | OPEN 遷移までの連続失敗数 |
| `recovery_timeout` | 120s | OPEN→HALF_OPEN の待機時間 |
| `success_threshold` | 2 | HALF_OPEN→CLOSED の連続成功数 |
| `timeout` | 30s | API 呼出タイムアウト |

状態遷移: `CLOSED → (5 fail) → OPEN → (120s) → HALF_OPEN → (2 success) → CLOSED`

#### `FillTestHealthMonitor`

```python
class FillTestHealthMonitor:
    def maybe_check(self, cycle_count: int) -> dict[str, Any] | None
    def maybe_gc(self, cycle_count: int) -> None
```

| 閾値 | 値 | アクション |
|------|-----|----------|
| `rss_warn` | 1,500 MB | ログ WARNING |
| `rss_critical` | 2,500 MB | ログ CRITICAL |
| `disk_warn` | 2 GB | ログ WARNING |
| `gc_interval` | 100 サイクル | `gc.collect()` |
| `check_interval` | 300s | システムチェック |

#### `FillTestStatePersistence`

```python
class FillTestStatePersistence:
    def save(self, state: FillTestState) -> Path
    def load(self, run_id: str) -> FillTestState | None
```

アトミック書込み（tmp → replace）。`FillTestState` dataclass で状態を構造化:

```python
@dataclass
class FillTestState:
    run_id: str
    cycle_count: int
    total_attempts: int
    filled_count: int
    cumulative_pnl_jpy: float
    current_lot: float
    soft_loss_cap_triggered: bool
    buy_offset_ratio: float
    sell_offset_ratio: float
    saved_at: str        # ISO 8601
    started_at: str      # ISO 8601
```

---

## §2 run_fill_test.py 統合箇所

### §2.1 `__init__` (コンポーネント初期化)

```python
# --- Resilience components (113#) --------------------------------
self._circuit_breaker = create_api_circuit_breaker()
self._health_monitor = FillTestHealthMonitor()
self._state_persistence = FillTestStatePersistence()
```

`atexit.register` 直前に配置。3 コンポーネントを self 属性として保持。

### §2.2 `run_continuous` (ループ内統合)

| 挿入箇所 | 処理 |
|----------|------|
| progress_log 直後 | `_health_monitor.maybe_check(cycle)` + `maybe_gc(cycle)` |
| progress_log_interval ごと | `_state_persistence.save(current_state)` |
| ループ正常終了時 | `_state_persistence.save(final_state)` |

### §2.3 `run_single_cycle` (CircuitBreaker ガード)

```
┌─ CircuitBreaker OPEN? ─→ skip (cancel_reason="circuit_breaker_open")
│
├─ 通常処理 (注文→約定監視→PnL計測)
│
├─ 全注文失敗 → _on_failure() 記録
│
└─ 正常完了 → _on_success() 記録
```

---

## §3 R1: `run_single_cycle` 分割 (106# R1 完了)

### §3.1 Before / After

| 指標 | Before | After | 変化 |
|------|--------|-------|------|
| `run_single_cycle` 行数 | ~755行 | ~307行 | **-59%** |
| 抽出メソッド数 | 0 | 3 | +3 |
| 結果伝達 dataclass | 0 | 3 | +3 |

### §3.2 抽出メソッド

#### `_evaluate_skip_gate()` → `_SkipGateResult`

**責務**: SkipGate ML モデルによるエントリー判定
**行数**: ~115行
**戻り値**:

```python
@dataclass
class _SkipGateResult:
    should_skip: bool
    gate_info: dict[str, Any]
    early_return_record: dict[str, Any] | None  # skip時のレコード
```

#### `_monitor_fill_polling()` → `_FillMonitorResult`

**責務**: 注文後の約定監視 + stale order 検出 & cancel-replace
**行数**: ~230行
**戻り値**:

```python
@dataclass
class _FillMonitorResult:
    filled: bool
    fill_price: float
    fill_qty: float
    cancel_reason: str
    order_id: str
    order_info: dict[str, Any]
    elapsed_sec: float
```

#### `_measure_post_fill_pnl()` → `_PnlMeasurement`

**責務**: 約定後 30s/60s/120s の PnL 計測 + early exit 判定
**行数**: ~110行
**戻り値**:

```python
@dataclass
class _PnlMeasurement:
    pnl_30s_bps: float
    pnl_60s_bps: float
    pnl_120s_bps: float
    exit_price: float
    exit_reason: str
```

### §3.3 `run_single_cycle` 残留責務

分割後も `run_single_cycle` に残る責務:

1. レジーム判定 + mid_price 取得
2. 価格・オフセット計算
3. `_evaluate_skip_gate()` 呼出 + skip 判定
4. 注文発行 (BUY/SELL)
5. `_monitor_fill_polling()` 呼出 + 約定判定
6. `_measure_post_fill_pnl()` 呼出 + PnL 記録
7. CircuitBreaker success/failure 記録
8. サイクルレコード構築 + JSONL 出力

これはオーケストレーション層として適正なサイズ。

---

## §4 テスト

### §4.1 新規テスト: `tests/unit/v460/test_113_resilience.py` (22件)

| クラス | テスト数 | 対象 |
|--------|---------|------|
| `TestCircuitBreakerFactory` | 3 | CB 生成、デフォルト値、カスタムパラメータ |
| `TestHealthMonitor` | 3 | デフォルト、間隔スキップ、GC 間隔 |
| `TestStatePersistence` | 4 | save/load、nonexistent→None、JSON生成、atomic write |
| `TestR1MethodExtraction` | 6 | delegation assertion、<400行検証、メソッド存在、dataclass存在 |
| `TestR1CircuitBreakerInRunSingleCycle` | 3 | OPENガード、success記録、failure記録 |
| `TestR1ResilienceInRunContinuous` | 3 | health check、state persistence、init確認 |

### §4.2 既存テスト修正 (8件)

| ファイル | 修正内容 |
|---------|---------|
| `test_094_stale_order.py` (6件) | source inspection: `run_single_cycle` → `_monitor_fill_polling` |
| `test_fill_quality.py` (2件) | source inspection: e3→`_measure_post_fill_pnl`, vpin→`_evaluate_skip_gate` |

### §4.3 全体結果

```
857 passed, 0 failed, 83 warnings (132.05s)
```

---

## §5 変更ファイル一覧

| ファイル | 操作 | 差分 |
|---------|------|------|
| `scripts/v460/lib/resilience.py` | **NEW** | +231行 |
| `scripts/v460/run_fill_test.py` | MODIFIED | +1,074/-427行 (net +161行) |
| `tests/unit/v460/test_113_resilience.py` | **NEW** | +22テスト |
| `tests/unit/v460/test_094_stale_order.py` | MODIFIED | 6テスト修正 |
| `tests/unit/v460/test_fill_quality.py` | MODIFIED | 2テスト修正 |
| `docs/v460/111_phg_rpt_legacy_asset_research.md` | MODIFIED | §4.1 + §10 修正 |

---

## §6 112# レビュー §5 改訂推奨への対応状況

| # | 112# 推奨 | 対応 | 備考 |
|---|-----------|------|------|
| 1 | §9 パス不一致修正 | **不要** | 実コード検証で全パス存在確認済。112# §2.1 の指摘が不正確 |
| 2 | §4.1 dead判定修正 | ✅ | 「完全Dead」→「fill_test 経路では未活用」に修正 |
| 3 | §10 SLO/Gate 閾値表 | ✅ | 10指標 + 閾値 + 判定基準を追記 |
| 4 | §10 Tier-1/2/3 統合順序 | ✅ | Tier-1/Tier-2 分類を追記 |
| 5 | R1 責務分割方針 | ✅ | 本 113# で R1 完了 (3メソッド+3 dataclass抽出) |

---

## §7 残作業 (本 113# スコープ外)

### 即時〜短期

| # | 項目 | 根拠 | 優先度 |
|---|------|------|--------|
| A1 | PnL Monte Carlo 定期実行 | 111# §10 #4, Tier-2 | MEDIUM |
| A2 | 112# §3.3 運用失敗モードテスト | API 429/5xx burst, OOM, 再起動復元 | MEDIUM |
| A3 | 112# §3.4 ph3 Stop 条件明文化 | v456-v459 再発防止 | LOW (ph3 前) |
| A4 | 106# R3 SkipGate warm_start 単体テスト | テスト不足 | LOW |

### ph2 完了前

| # | 項目 | 根拠 |
|---|------|------|
| B1 | v458 Walk-Forward バグ 6件修正 | 111# §10 #6 |
| B2 | BacktestReporter 統一 (3重定義→単一) | 111# §10 #7 |
| B3 | CheckVenueHealth プリフライト追加 | 111# §10 #8 |

### 106# R1-R10 進捗

| # | 項目 | 状態 | 実施 # |
|---|------|------|--------|
| R1 | `run_single_cycle` 分割 | ✅ **完了** | **113#** |
| R2 | `BPS_FACTOR` 定数化 | ✅ 完了 | 106# |
| R3 | SkipGate テスト不足 | ✅ 後続で大幅補強 | session037 |
| R4 | ドキュメント命名違反 28件 | ❌ 後日 | — |
| R5 | lib → ztb 移動 | ✅ 主要部分前倒し | session037 |
| R6 | utils 70+ ファイル分割 | ❌ 後日 | — |
| R7 | config/configs 重複整理 | ❌ 後日 | — |
| R8 | `# type: ignore` 解消 | ✅ 部分実施 (1/3) | 106# |
| R9 | インライン import 整理 | ✅ 完了 | 106# |
| R10 | 100番重複解消 | ✅ 完了 | 105# |

### 111# §10 Tier 別進捗

| Tier | 項目 | 状態 |
|------|------|------|
| **Tier-1** | CircuitBreaker | ✅ 113# |
| **Tier-1** | HealthMonitor | ✅ 113# |
| **Tier-1** | R1 run_single_cycle 分割 | ✅ 113# |
| **Tier-2** | StatePersistence | ✅ 113# |
| **Tier-2** | PnL Monte Carlo | ❌ 未着手 |
| **Tier-3** | RiskRuleEngine + Profiles + AutoStop | ❌ 未着手 |
| **Tier-3** | Reconciliation | ❌ 未着手 |
| 通知 | GatesToAlerts / DiscordNotifier | ❌ 未着手 |
| 監視 | watch_1m | ❌ 未着手 (HealthMonitor で部分カバー) |
| データ | DataValidation / MemoryCache | ❌ 未着手 |

## 2026-03-21 補遺

113# 時点では `R3` / `R5` を deferred としていたが、その後の session037 で次が前倒しされた。

- `R3`:
  - `SkipGate` の runtime helper
  - result metadata
  - FillRecord extra payload
  - final FillRecord context/builder
  まで migration test / focused test を補強
- `R5`:
  - `cancel_reasons`
  - `param_adapter`
  - `lot_sizer`
  - `fast_fill_defense`
  - `sac_common`
  - `regime_detector`
  - `bayesian_regime_filter`
  を canonical 化

このため、113# の deferred 表現は「当時の状況」としては妥当だが、
現時点の進捗としては更新が必要な段階に入っている。
