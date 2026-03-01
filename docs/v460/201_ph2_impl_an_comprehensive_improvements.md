# 201# Phase 2: A–N 包括的改善実装

> **前提**: `200_ph2_resp_199_codex_gemini_review_eval.md` の P0 実装 (commit `9b9ed6780`) 完了後、  
> 残り提案 A–N + 追加発見 5 件 (10-A〜10-E) を一括実装  
> **テスト**: 新規 25/25 通過、既存 2686/2686 通過 (pre-existing 4 件除外)

---

## 1. 実装サマリ

### 1.1 P0 バグ修正 (致命的)

| ID | 内容 | ファイル | 影響 |
|---|---|---|---|
| **10-A** | `_soft_drawdown_interval_multiplier` 日次リセット漏れ | fill_loop_orchestrator.py | multiplier=3.0 が永続化し全サイクルが 3 倍間隔 |

### 1.2 提案 A–N 実装

| ID | 内容 | ファイル | 優先度 |
|---|---|---|---|
| **B/I** | postonly_guard crossing → skip-cycle | fill_cycle_executor.py, cancel_reasons.py | P1 |
| **C** | low_vol_boost 比例スケーリング | maker_price.py, fill_config.py | P2 |
| **E** | balance_forced 時間ベース cooldown | fill_loop_orchestrator.py, fill_config.py | P3 |
| **G** | sell PnL wait 動的 (vol-scaled) | cycle_strategy.py, fill_cycle_executor.py | P2 |
| **H** | regime velocity 軸 (opt-in) | regime_detector.py | P3 |
| **K** | halt レコード削減 | fill_loop_orchestrator.py | P2 |
| **L** | velocity SSOT モジュール | velocity_math.py (NEW), skip_gate_evaluator.py | P1 |
| **M** | ev_as_offset warning zone + DRY | fill_config.py, fill_cycle_executor.py, skip_gate_evaluator.py | P2 |

### 1.3 追加改善 (10-B〜10-E)

| ID | 内容 | ファイル | 種別 |
|---|---|---|---|
| **10-B** | 冗長三項演算子の簡素化 | order_monitor.py | コード品質 |
| **10-C** | ループ内 import → モジュールレベル | fill_cycle_executor.py | パフォーマンス |
| **10-D** | ハードコード文字列 `"(avg -1.98bps loss)"` 除去 | fill_loop_orchestrator.py | 保守性 |
| **10-E** | `3.0` → YAML 設定化 (`soft_drawdown_interval_multiplier`) | fill_loop_orchestrator.py, fill_config.py | 拡張性 |

---

## 2. 実装詳細

### 2.1 [10-A] soft_drawdown_interval_multiplier 日次リセット (P0)

**問題**: P0-2 で導入された `_soft_drawdown_interval_multiplier = 3.0` は、日次境界 (`maybe_reset_day()`) でリセットされておらず、一度設定されると永続的に 3 倍間隔で動作し続ける。

**修正箇所**: `fill_loop_orchestrator.py` — while ループ先頭

```python
# BEFORE: is_halted() が先、maybe_reset_day() が後
if self.is_halted():
    ...

# AFTER: maybe_reset_day() を先に実行し、日次境界でリセット
if self.maybe_reset_day():
    self._soft_drawdown_interval_multiplier = 1.0

if self.is_halted():
    ...
```

**テスト**: `TestSoftDrawdownIntervalMultiplierReset` (1 test)

---

### 2.2 [B/I] postonly_guard crossing → skip-cycle

**問題**: postonly_guard が crossing 検出時に best bid/ask にスナップし、offset パイプライン全体を無効化。

**修正**: crossing 検出 → `POSTONLY_CROSSING_SKIP` cancel reason で break。circuit_breaker の `async_on_failure()` はこの理由では bypass。

**新定数**: `cancel_reasons.py` に `POSTONLY_CROSSING_SKIP = "postonly_crossing_skip"` 追加

**テスト**: `TestPostonlyCrossingSkip` (3 tests)

---

### 2.3 [C] low_vol_boost 比例スケーリング

**問題**: `vol_ratio` が閾値より常時低いため、boost が定数化 (binary on/off)。

**修正**: `fill_config.low_vol_boost_proportional = True` 時、閾値からの距離に比例したブースト適用。

```python
_ratio = 1.0 - vol_ratio / threshold  # 0~1
_low_vol_boost = min_val + (max_val - min_val) * _ratio
```

**設定**: `low_vol_boost_proportional: true`, `low_vol_boost_min: 1.0`

**テスト**: `TestLowVolBoostProportional` (3 tests)

---

### 2.4 [E] balance_forced 時間ベース cooldown

**問題**: balance_forced イベントが短時間で連続発生する場合の検出機構がない。

**修正**: `_last_balance_forced_time` と `_balance_forced_freq_count` で頻度監視。`balance_forced_cooldown_sec` 以内の連続発生時に WARNING ログ出力。

**設定**: `balance_forced_cooldown_sec: 0.0` (デフォルト無効)

**テスト**: `TestBalanceForcedCooldown` (1 test)

---

### 2.5 [G] sell PnL wait 動的 vol-scaled

**問題**: 売り PnL 待機時間がレジームに固定で、ボラティリティ変動に非対応。

**修正**: `cycle_strategy.effective_post_fill_wait()` に `vol_ratio` パラメータ追加。vol_ratio → 逆数べき乗 (0.3) → 0.7x〜1.5x のスケール係数。低 vol で待機延長、高 vol で短縮。

```python
_vol_scale = max(0.7, min(1.5, 1.0 / vol_ratio ** 0.3))
```

**後方互換性**: `vol_ratio=None` で元の動作を維持。

**テスト**: `TestSellPnlWaitDynamic` (4 tests)

---

### 2.6 [H] regime velocity 軸 (opt-in)

**問題**: regime 判定が価格変動幅のみで速度を未考慮。

**修正**: `RegimeConfig` に `velocity_modulation: bool = False` 追加。有効時、短窓 velocity が trend 方向と一致 → confidence +0.15、逆行 → -0.20。

**設計意図**: opt-in (デフォルト無効) により既存動作に影響なし。十分なデータ蓄積後に有効化。

**テスト**: `TestRegimeVelocityModulation` (2 tests)

---

### 2.7 [K] halt レコード削減

**問題**: HALT 中のログ記録が毎サイクル発生し、JSONL の 54% が non-trade レコード。

**修正**: halt 開始時 + `progress_log_interval` ごとのみ記録。halt 終了時に持続サイクル数をログ出力。

---

### 2.8 [L] velocity SSOT モジュール

**問題**: velocity_offset 計算が `skip_gate_evaluator.py` と `maker_price.py` で独立・符号逆転。

**修正**: `scripts/v460/lib/velocity_math.py` を新規作成。`compute_velocity_offset_multiplier()` を SSOT として抽出。`skip_gate_evaluator._compute_velocity_offset_multiplier()` は wrapper として維持。

**テスト**: `TestVelocityMath` (4 tests)

---

### 2.9 [M] ev_as_offset warning zone + DRY

**問題**: (1) ev_offset 計算が executor と evaluator で重複、(2) 負 EV を通しすぎる。

**修正**:
- `fill_config.compute_ev_offset_multiplier()` にロジック統合 (DRY)
- warning zone: `ev < warning_threshold` かつ `ev < 0` → offset_factor で乗算引締め
- 設定: `ev_warning_threshold: -4.0`, `ev_warning_offset_factor: 0.7`

**テスト**: `TestEvOffsetWarningZone` (5 tests)

---

## 3. 新規ファイル

| ファイル | 目的 |
|---|---|
| `scripts/v460/lib/velocity_math.py` | velocity offset 計算 SSOT |
| `tests/unit/v460/test_200_an_improvements.py` | 200# 全改善テスト (25 tests) |

---

## 4. 設定変更 (fill_test.yaml)

```yaml
daily_drawdown:
  soft_drawdown_interval_multiplier: 3.0   # 10-E: YAML外部化

regime:
  low_vol_boost_proportional: true          # C: 比例スケーリング
  low_vol_boost_min: 1.0                    # C: 最小ブースト値

skip_gate:
  ev_warning_threshold: -4.0               # M: warning zone 閾値
  ev_warning_offset_factor: 0.7            # M: warning zone 引締め係数
```

---

## 5. テスト結果

```
tests/unit/v460/test_200_an_improvements.py — 25 passed
tests/unit/v460/ (全体)                     — 2686 passed, 3 deselected
```

Pre-existing failures (4 件、git stash で確認済):
- `test_088::test_early_return_paths_have_run_id`
- `test_088::test_make_skip_record_used_for_all_skips`
- `test_168::test_total_halt_days`
- `test_168::test_total_halt_days_increments`

---

## 6. 変更ファイル一覧

| ファイル | 変更種別 | 関連 ID |
|---|---|---|
| `scripts/v460/lib/cancel_reasons.py` | 定数追加 | B/I |
| `scripts/v460/lib/fill_config.py` | 設定追加 + DRY関数 | C, E, M, 10-E |
| `scripts/v460/lib/fill_loop_orchestrator.py` | リセット + halt削減 + cooldown | 10-A, K, E, 10-D, 10-E |
| `scripts/v460/lib/fill_cycle_executor.py` | skip化 + vol_ratio + DRY | B/I, G, M, 10-C |
| `scripts/v460/lib/maker_price.py` | 比例boost | C |
| `scripts/v460/lib/order_monitor.py` | 三項演算子簡素化 | 10-B |
| `scripts/v460/lib/cycle_strategy.py` | vol-scaled wait | G |
| `scripts/v460/lib/regime_detector.py` | velocity軸 | H |
| `scripts/v460/lib/skip_gate_evaluator.py` | SSOT委譲 + DRY | L, M |
| `scripts/v460/lib/velocity_math.py` | **NEW** — SSOT | L |
| `configs/v460/fill_test.yaml` | 新設定追加 | C, M, 10-E |
| `tests/unit/v460/test_145_structural_fixes.py` | テスト修正 | B/I |
| `tests/unit/v460/test_200_an_improvements.py` | **NEW** — 25 tests | ALL |

---

## 7. Post-commit レビュー修正 (201#)

Self-review で発見した HIGH 1 / MED 4 / LOW 1 を追加修正。

| 重要度 | 内容 | ファイル |
|---|---|---|
| **HIGH** | 動的属性 (`_soft_drawdown_interval_multiplier` 等) のクラスレベル宣言追加 (mypy/IDE 対応) | orchestrator, executor |
| **MED** | `balance_forced_cooldown_sec` YAML 配線漏れ修正 | fill_config.py |
| **MED** | `__post_init__` バリデーション追加 (`soft_drawdown_interval_multiplier > 0`, `low_vol_boost_min` 範囲, `balance_forced_cooldown_sec >= 0`) | fill_config.py |
| **MED** | postonly crossing 連続発生カウンタ (`_postonly_crossing_streak`) — 3 連続で WARNING | fill_cycle_executor.py |
| **LOW** | cycle_strategy コメント数値例修正 (`×1.3` → `×1.23`) | cycle_strategy.py |

テスト追加: +10 tests (計 35 tests 通過)
